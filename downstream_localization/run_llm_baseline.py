"""LLM CSI-adapter baselines for downstream RX localization.

This is a supervised continuous-input baseline. CSI is not serialized as text:
the local CSI encoder first produces a compact CSI representation, which is
projected into a short soft-prefix sequence for a causal LLM. The model then
regresses the relative RX coordinate (dx, dy).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.encoder import CSIEncoder  # noqa: E402

from downstream_localization.data_io import (  # noqa: E402
    LocalizationCorpus,
    LocalizationDataset,
    collate_localization,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_localization.pretrained import _state as checkpoint_state  # noqa: E402


MODEL_IDS = {
    "deepseek_r1_qwen3_8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "internlm3_8b": "internlm/internlm3-8b-instruct",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
}
DEFAULT_BACKBONE_TRAINING = {
    "deepseek_r1_qwen3_8b": "frozen",
    "internlm3_8b": "frozen",
    "qwen3_5_2b": "full",
}


class CoordinateNormalizer:
    def __init__(self, targets: torch.Tensor):
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0, correction=0).clamp(min=1e-6)

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean.to(target.device)) / self.std.to(target.device)

    def denormalize(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.std.to(target.device) + self.mean.to(target.device)


def localization_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(prediction.float() - target.float(), dim=-1).cpu().numpy()
    return {
        "mean_m": float(np.mean(error)),
        "median_m": float(np.median(error)),
        "p90_m": float(np.percentile(error, 90)),
        "within_5m_pct": float(100.0 * np.mean(error <= 5.0)),
        "within_10m_pct": float(100.0 * np.mean(error <= 10.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model directory or Hugging Face model ID; defaults to the official ID.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quantization", choices=("auto", "4bit", "none"), default="auto")
    parser.add_argument("--backbone-training", choices=("auto", "frozen", "full"), default="auto")
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-width", type=int, default=512)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--no-save-backbone",
        action="store_true",
        help=(
            "For full fine-tuning smoke runs only: skip saving the full LLM backbone. "
            "The final test then uses the last backbone state."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_regime(args: argparse.Namespace) -> None:
    if args.backbone_training == "auto":
        args.backbone_training = DEFAULT_BACKBONE_TRAINING[args.model]
    if args.quantization == "auto":
        args.quantization = "4bit" if args.backbone_training == "frozen" else "none"
    if args.backbone_training == "full" and args.quantization != "none":
        raise ValueError("Full-parameter LLM training requires --quantization none.")


def model_info_from_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(checkpoint.get("args", {}))
    state = checkpoint_state(checkpoint)
    encoder_key = "csi.input_proj.spatial_linear.weight"
    if encoder_key not in state:
        fallback_key = "csi_encoder.input_projection.weight"
        if fallback_key not in state:
            raise ValueError(f"Checkpoint does not contain {encoder_key}")
        encoder_key = fallback_key
    return {
        "d_token": int(state[encoder_key].shape[1]),
        "token_norm_mode": args.get("token_norm_mode", "std"),
        "use_continuous_config_encoding": bool(
            args.get(
                "use_continuous_config_encoding",
                any(key.startswith("csi.config_enc.") for key in state),
            )
        ),
    }


def select_subset(
    indices: torch.Tensor,
    size: int,
    seed: int,
    name: str,
) -> torch.Tensor:
    if size <= 0:
        raise ValueError(f"{name} size must be positive")
    if size > len(indices):
        raise ValueError(f"Requested {size} {name} samples, but split has {len(indices)}")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)
    return indices[order[:size]].clone()


def make_loader(
    corpus: LocalizationCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    return DataLoader(
        LocalizationDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_localization,
        generator=torch.Generator().manual_seed(args.seed + seed_offset),
        **({"prefetch_factor": 1} if args.num_workers else {}),
    )


def load_backbone(args: argparse.Namespace, *, model_path_override: str | None = None) -> nn.Module:
    try:
        from transformers import AutoModel, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Install transformers, accelerate and bitsandbytes in the selected LLM environment."
        ) from error

    model_path = model_path_override or args.model_path or MODEL_IDS[args.model]
    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.model == "internlm3_8b",
        "local_files_only": bool(model_path_override) or not args.allow_download,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "device_map": {"": args.device},
    }
    if args.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    text_config = getattr(backbone.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    trainable = args.backbone_training == "full"
    backbone.requires_grad_(trainable)
    if args.model == "qwen3_5_2b" and trainable:
        freeze_qwen35_auxiliary_modules(backbone)
    backbone.train(trainable)
    if args.gradient_checkpointing:
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError(f"{args.model} does not expose gradient checkpointing")
        backbone.gradient_checkpointing_enable()
    return backbone


def freeze_qwen35_auxiliary_modules(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if ".visual." in name or ".mtp." in name or name.startswith("mtp."):
            parameter.requires_grad = False


def backbone_hidden_size(backbone: nn.Module) -> int:
    config = backbone.config
    for candidate in (getattr(config, "text_config", None), config):
        size = getattr(candidate, "hidden_size", None)
        if size is not None:
            return int(size)
    raise ValueError("Could not determine LLM hidden size from its config")


class LLMLocalizationRegressor(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        model_info: dict[str, Any],
        adapter_width: int,
        prefix_tokens: int,
        compute_dtype: torch.dtype,
        train_backbone: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.compute_dtype = compute_dtype
        self.train_backbone = train_backbone
        self.prefix_tokens = int(prefix_tokens)
        hidden_size = backbone_hidden_size(backbone)
        self.csi_encoder = CSIEncoder(
            d_token=model_info["d_token"],
            d_model=384,
            d_clip=256,
            token_norm_mode=model_info["token_norm_mode"],
            use_continuous_config_encoding=model_info["use_continuous_config_encoding"],
        )
        self.projector = nn.Sequential(
            nn.Linear(256, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, self.prefix_tokens * hidden_size),
        )
        self.position = nn.Parameter(torch.zeros(1, self.prefix_tokens, hidden_size))
        self.readout = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.readout, std=0.02)

    def train(self, mode: bool = True) -> "LLMLocalizationRegressor":
        super().train(mode)
        self.backbone.train(mode and self.train_backbone)
        return self

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        csi = self.csi_encoder(**encoder_kwargs(batch)).float()
        prefix = self.projector(csi).view(csi.shape[0], self.prefix_tokens, -1)
        prefix = prefix + self.position
        readout = self.readout.expand(csi.shape[0], -1, -1)
        embeds = torch.cat((prefix, readout), dim=1).to(self.compute_dtype)
        attention_mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        outputs = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        if hidden.ndim != 3 or hidden.shape[1] != embeds.shape[1]:
            raise ValueError(f"Unexpected LLM hidden-state shape: {tuple(hidden.shape)}")
        return self.head(hidden[:, -1, :].float())

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state = {
            "position": self.position.detach().cpu().clone(),
            "readout": self.readout.detach().cpu().clone(),
        }
        for module_name in ("csi_encoder", "projector", "head"):
            module = getattr(self, module_name)
            state.update(
                {
                    f"{module_name}.{key}": value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            )
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.position.data.copy_(state["position"].to(self.position.device))
        self.readout.data.copy_(state["readout"].to(self.readout.device))
        for module_name in ("csi_encoder", "projector", "head"):
            module = getattr(self, module_name)
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(module_state, strict=True)


def adapter_parameters(model: LLMLocalizationRegressor) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]


@torch.inference_mode()
def evaluate(
    model: LLMLocalizationRegressor,
    loader: DataLoader,
    normalizer: CoordinateNormalizer,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets, indices = [], [], []
    for batch_number, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        prediction = model(batch)
        predictions.append(normalizer.denormalize(prediction.float()).cpu())
        targets.append(batch["target_xy"].cpu())
        indices.append(batch["indices"].cpu())
        if batch_number % 50 == 0 or batch_number == len(loader):
            print(f"inference_batches={batch_number}/{len(loader)}", flush=True)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return localization_metrics(prediction, target), prediction, target, torch.cat(indices)


def train(
    model: LLMLocalizationRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: CoordinateNormalizer,
    args: argparse.Namespace,
    device: torch.device,
    model_path: str,
) -> tuple[int, float, list[dict[str, float]]]:
    groups: list[dict[str, Any]] = [
        {"params": adapter_parameters(model), "lr": args.learning_rate}
    ]
    if args.backbone_training == "full":
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.compute_dtype == "fp16")
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "best_adapter.pt"
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            batch = move_tensor_batch(batch, device)
            target = normalizer.normalize(batch["target_xy"])
            prediction = model(batch)
            loss = F.smooth_l1_loss(prediction, target, reduction="none").sum(dim=-1).mean()
            window_start = (step // args.grad_accum) * args.grad_accum
            window_size = min(args.grad_accum, num_batches - window_start)
            scaler.scale(loss / window_size).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in groups for parameter in group["params"]],
                    max_norm=5.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(target)
            count += len(target)
        val_metrics, _, _, _ = evaluate(model, val_loader, normalizer, device)
        record = {
            "epoch": float(epoch),
            "train_loss": total / max(count, 1),
            "val_median_m": val_metrics["median_m"],
            "val_mean_m": val_metrics["mean_m"],
            "val_p90_m": val_metrics["p90_m"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"val_median={record['val_median_m']:.4f}m "
            f"epoch_seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )
        if record["val_median_m"] < best_val:
            best_val = record["val_median_m"]
            best_epoch = epoch
            stale = 0
            if args.backbone_training == "full" and not args.no_save_backbone:
                model.backbone.save_pretrained(
                    args.output_dir / "best_backbone",
                    safe_serialization=True,
                    max_shard_size="4GB",
                )
            torch.save(
                {
                    "format": "llm_csi_localization_adapter_v1",
                    "model": args.model,
                    "model_path": model_path,
                    "backbone_training": args.backbone_training,
                    "adapter_width": args.adapter_width,
                    "prefix_tokens": args.prefix_tokens,
                    "adapter_state_dict": model.adapter_state_dict(),
                    "target_mean": normalizer.mean,
                    "target_std": normalizer.std,
                    "epoch": epoch,
                    "best_val_median_m": best_val,
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was saved")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_adapter_state_dict(saved["adapter_state_dict"])
    return best_epoch, best_val, history


def save_predictions(
    path: Path,
    result: tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor],
    corpus: LocalizationCorpus,
) -> None:
    _, prediction, target, indices = result
    tx_xy = corpus.tx_xyz_m[indices, :2]
    predicted_rx = tx_xy + prediction
    true_rx = corpus.rx_xyz_m[indices, :2]
    error = torch.linalg.vector_norm(predicted_rx - true_rx, dim=-1)
    np.savez_compressed(
        path,
        indices=indices.numpy(),
        predicted_relative_xy_m=prediction.numpy(),
        true_relative_xy_m=target.numpy(),
        predicted_rx_xy_m=predicted_rx.numpy(),
        true_rx_xy_m=true_rx.numpy(),
        error_m=error.numpy(),
    )


def write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    resolve_training_regime(args)
    if any(
        value <= 0
        for value in (
            args.train_size,
            args.val_size,
            args.test_size,
            args.batch_size,
            args.grad_accum,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.backbone_learning_rate,
            args.adapter_width,
            args.prefix_tokens,
            args.cpu_threads,
        )
    ) or args.num_workers < 0:
        raise ValueError("Sizes/rates must be positive and num-workers nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LLM localization baseline requires CUDA")
    if args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU lacks BF16 support; use --compute-dtype fp16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)

    corpus = LocalizationCorpus.load(args.data)
    train_indices = select_subset(corpus.split_indices["train"], args.train_size, args.seed + 1, "train")
    val_indices = select_subset(corpus.split_indices["val"], args.val_size, args.seed + 2, "val")
    test_indices = select_subset(corpus.split_indices["test"], args.test_size, args.seed + 3, "test")
    normalizer = CoordinateNormalizer(corpus.relative_xy_m[train_indices])
    model_info = model_info_from_checkpoint(args.checkpoint)
    first_shape = corpus.samples[0].tokens.shape
    if int(first_shape[1]) != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={first_shape[1]} but checkpoint expects {model_info['d_token']}"
        )
    loaders = {
        "train": make_loader(corpus, train_indices, args, shuffle=True, seed_offset=11),
        "val": make_loader(corpus, val_indices, args, shuffle=False, seed_offset=12),
        "test": make_loader(corpus, test_indices, args, shuffle=False, seed_offset=13),
    }
    model_path = args.model_path or MODEL_IDS[args.model]
    print(
        f"model={args.model} path={model_path} device={device} "
        f"quantization={args.quantization} backbone_training={args.backbone_training} "
        f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}",
        flush=True,
    )
    print(
        f"target_mean_m={normalizer.mean.tolist()} target_std_m={normalizer.std.tolist()}",
        flush=True,
    )
    started = time.time()
    backbone = load_backbone(args)
    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    model = LLMLocalizationRegressor(
        backbone=backbone,
        model_info=model_info,
        adapter_width=args.adapter_width,
        prefix_tokens=args.prefix_tokens,
        compute_dtype=dtype,
        train_backbone=args.backbone_training == "full",
    )
    del backbone
    model.csi_encoder.to(device)
    model.projector.to(device)
    model.head.to(device)
    model.position.data = model.position.data.to(device)
    model.readout.data = model.readout.data.to(device)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable_parameters={trainable_count}", flush=True)

    best_epoch, best_val, history = train(
        model, loaders["train"], loaders["val"], normalizer, args, device, model_path
    )
    if args.backbone_training == "full" and not args.no_save_backbone:
        del model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        model.backbone = load_backbone(args, model_path_override=str(args.output_dir / "best_backbone"))
        model.backbone.requires_grad_(False)
        model.train_backbone = False
    test_result = evaluate(model, loaders["test"], normalizer, device)
    save_predictions(args.output_dir / "llm_localization_test_predictions.npz", test_result, corpus)
    metrics = test_result[0]
    row = {
        "model": args.model,
        "backbone_training": args.backbone_training,
        "quantization": args.quantization,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "best_epoch": best_epoch,
        "best_val_median_m": best_val,
        **metrics,
    }
    write_metrics_csv(args.output_dir / "metrics.csv", row)
    summary = {
        **row,
        "model_path": model_path,
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "target_mean_m": normalizer.mean.tolist(),
        "target_std_m": normalizer.std.tolist(),
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    print(f"outputs={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
