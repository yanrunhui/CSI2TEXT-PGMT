"""LLM backbones with CSI adapters for full-CSI beam selection.

This is a supervised continuous-input baseline, not zero-shot text prompting.
Beam gains and physical labels are never passed to the model at inference time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_beam_selection.data_io import BeamCorpus, FullCSIBeamDataset  # noqa: E402
from downstream_beam_selection.full_csi_experiment import (  # noqa: E402
    class_distribution,
    report_test,
    set_seed,
)


MODEL_IDS = {
    "deepseek_r1_qwen3_8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "internlm3_8b": "internlm/internlm3-8b-instruct",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
}
CSI_SHAPE = (16, 8, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument(
        "--model-path", default=None,
        help="Local model directory or Hugging Face model ID; defaults to the official ID.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true", help="Permit missing weights to be fetched.")
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument(
        "--backbone-training", choices=("frozen", "full"), default="frozen",
        help="Full updates all pretrained LLM weights; frozen updates only the CSI adapter/head.",
    )
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-width", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Trade compute for activation memory in the LLM backbone.",
    )
    return parser.parse_args()


def collate_csi(items: list[tuple[int, Any, torch.Tensor]]) -> dict[str, torch.Tensor]:
    indices, samples, targets = zip(*items)
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "tokens": torch.stack([sample.tokens.float() for sample in samples]),
        "target_class": torch.stack(targets).long(),
    }


def make_loader(
    corpus: BeamCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        FullCSIBeamDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_csi,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        **({"prefetch_factor": 1} if args.num_workers else {}),
    )


def load_backbone(
    args: argparse.Namespace, *, model_path_override: str | None = None,
    for_test: bool = False,
) -> nn.Module:
    try:
        from transformers import AutoModel, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers, accelerate and bitsandbytes in the model's environment; "
            "see downstream_beam_selection/LLM_BASELINES.md"
        ) from exc

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
    trainable = args.backbone_training == "full" and not for_test
    backbone.requires_grad_(trainable)
    backbone.train(trainable)
    if args.gradient_checkpointing and not for_test:
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError(f"{args.model} does not expose gradient checkpointing")
        backbone.gradient_checkpointing_enable()
    return backbone


def backbone_hidden_size(backbone: nn.Module) -> int:
    config = backbone.config
    for candidate in (getattr(config, "text_config", None), config):
        size = getattr(candidate, "hidden_size", None)
        if size is not None:
            return int(size)
    raise ValueError("Could not determine LLM hidden size from its config")


class LLMBeamClassifier(nn.Module):
    """Sixteen complete CSI tokens, a causal backbone, one beam readout."""

    def __init__(
        self, backbone: nn.Module, adapter_width: int, compute_dtype: torch.dtype,
        train_backbone: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.compute_dtype = compute_dtype
        self.train_backbone = train_backbone
        hidden_size = backbone_hidden_size(backbone)
        self.projector = nn.Sequential(
            nn.Linear(8 * 128, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, hidden_size),
        )
        self.position = nn.Parameter(torch.zeros(1, CSI_SHAPE[0], hidden_size))
        self.readout = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 64))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.readout, std=0.02)

    def train(self, mode: bool = True) -> "LLMBeamClassifier":
        super().train(mode)
        self.backbone.train(mode and self.train_backbone)
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tuple(tokens.shape[1:]) != CSI_SHAPE:
            raise ValueError(f"Expected CSI [batch,{CSI_SHAPE}], got {tuple(tokens.shape)}")
        # Normalize per link, preserving all relative antenna/beam/subcarrier structure.
        # Absolute link power cannot change argmax_b G_b and is not a target here.
        x = tokens.float()
        rms = x.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        x = (x / rms).reshape(x.shape[0], CSI_SHAPE[0], -1)
        csi_embeds = self.projector(x) + self.position
        readout = self.readout.expand(x.shape[0], -1, -1)
        embeds = torch.cat((csi_embeds, readout), dim=1).to(self.compute_dtype)
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
        for module_name in ("projector", "head"):
            module = getattr(self, module_name)
            state.update({
                f"{module_name}.{key}": value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            })
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.position.data.copy_(state["position"].to(self.position.device))
        self.readout.data.copy_(state["readout"].to(self.readout.device))
        for module_name in ("projector", "head"):
            module = getattr(self, module_name)
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(module_state, strict=True)


def send_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["tokens"].to(device, non_blocking=True),
        batch["target_class"].to(device, non_blocking=True),
    )


@torch.inference_mode()
def validation_loss(
    model: LLMBeamClassifier, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        tokens, target = send_batch(batch, device)
        logits = model(tokens)
        total += float(F.cross_entropy(logits, target)) * len(target)
        count += len(target)
    return total / max(count, 1)


@torch.inference_mode()
def predict_test(
    model: LLMBeamClassifier, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    parts = []
    for batch in loader:
        tokens, _ = send_batch(batch, device)
        parts.append(model(tokens).float().cpu())
    return torch.cat(parts)


def train(
    model: LLMBeamClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    model_path: str,
) -> tuple[int, float, list[dict[str, float]]]:
    adapter_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": adapter_parameters, "lr": args.learning_rate}
    ]
    if args.backbone_training == "full":
        parameter_groups.append({
            "params": [parameter for parameter in model.backbone.parameters()
                       if parameter.requires_grad],
            "lr": args.backbone_learning_rate,
        })
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.compute_dtype == "fp16")
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "best_adapter.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            tokens, target = send_batch(batch, device)
            logits = model(tokens)
            loss = F.cross_entropy(logits, target)
            # Account for a short final accumulation window.
            window_start = (step // args.grad_accum) * args.grad_accum
            window_size = min(args.grad_accum, num_batches - window_start)
            scaler.scale(loss / window_size).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(target)
            count += len(target)
        val_loss = validation_loss(model, val_loader, device)
        record = {
            "epoch": float(epoch),
            "train_loss": total / count,
            "val_loss": val_loss,
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"val_loss={val_loss:.6f} epoch_seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            if args.backbone_training == "full":
                model.backbone.save_pretrained(
                    args.output_dir / "best_backbone",
                    safe_serialization=True,
                    max_shard_size="4GB",
                )
            torch.save({
                "format": "llm_csi_adapter_v2",
                "model": args.model,
                "model_path": model_path,
                "backbone_training": args.backbone_training,
                "adapter_width": args.adapter_width,
                "adapter_state_dict": model.adapter_state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "preprocessing": "per-link RMS normalization of all 16x8x128 CSI tokens",
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was saved")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_adapter_state_dict(saved["adapter_state_dict"])
    return best_epoch, best_val, history


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in (
        args.batch_size, args.grad_accum, args.epochs, args.patience,
        args.learning_rate, args.backbone_learning_rate,
        args.adapter_width, args.cpu_threads,
    )) or args.num_workers < 0:
        raise ValueError("Training sizes/rates must be positive and num-workers nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LLM baseline training requires a CUDA device")
    if args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU lacks BF16 support; use --compute-dtype fp16")
    if args.backbone_training == "full" and args.quantization != "none":
        raise ValueError(
            "Full-parameter LLM training cannot update 4-bit base weights; "
            "use --quantization none"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an experiment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    corpus = BeamCorpus.load(args.data)
    for index, sample in enumerate(corpus.samples):
        if tuple(sample.tokens.shape) != CSI_SHAPE:
            raise ValueError(f"Sample {index} has tokens {tuple(sample.tokens.shape)}")
    distribution = class_distribution(corpus)
    (args.output_dir / "class_distribution.json").write_text(
        json.dumps(distribution, indent=2) + "\n", encoding="utf-8"
    )
    loaders = {
        name: make_loader(corpus, corpus.split_indices[name], args, shuffle=name == "train")
        for name in ("train", "val", "test")
    }
    model_path = args.model_path or MODEL_IDS[args.model]
    print(
        f"model={args.model} path={model_path} device={device} "
        f"quantization={args.quantization} backbone_training={args.backbone_training} "
        f"train={len(corpus.split_indices['train'])} "
        f"val={len(corpus.split_indices['val'])} test={len(corpus.split_indices['test'])}",
        flush=True,
    )
    backbone = load_backbone(args)
    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    model = LLMBeamClassifier(
        backbone, args.adapter_width, dtype,
        train_backbone=args.backbone_training == "full",
    )
    del backbone
    model.projector.to(device)
    model.head.to(device)
    model.position.data = model.position.data.to(device)
    model.readout.data = model.readout.data.to(device)
    expected_trainable = args.backbone_training == "full"
    if not all(
        parameter.requires_grad == expected_trainable
        for parameter in model.backbone.parameters()
    ):
        raise RuntimeError("LLM backbone trainable state does not match requested mode")
    print(
        f"backbone_training={args.backbone_training} "
        f"trainable_parameters={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}",
        flush=True,
    )
    best_epoch, best_val, history = train(
        model, loaders["train"], loaders["val"], args, device, model_path
    )
    if args.backbone_training == "full":
        # Reload the validation-selected backbone, not the last training epoch.
        del model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        model.backbone = load_backbone(
            args,
            model_path_override=str(args.output_dir / "best_backbone"),
            for_test=True,
        )
        model.train_backbone = False
    test_indices = corpus.split_indices["test"]
    logits = predict_test(model, loaders["test"], device)
    if len(logits) != len(test_indices):
        raise RuntimeError("Test prediction count does not match test split")
    metrics = report_test(corpus, test_indices, logits, args.output_dir)
    result = {
        "model": args.model,
        "model_path": model_path,
        "baseline_type": (
            "full_finetune_llm_and_csi_adapter"
            if args.backbone_training == "full" else "frozen_llm_trainable_csi_adapter"
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "metrics": metrics,
        "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        row = {
            "model": args.model,
            "backbone_training": args.backbone_training,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            **metrics,
        }
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({"model": args.model, "best_epoch": best_epoch, **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
