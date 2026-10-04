"""LLM backbones with CSI adapters for outage prediction.

This is a supervised continuous-input baseline. The LLM predicts total received
power from CSI, then the deterministic outage rule is applied exactly as in
downstream_outage_prediction.run_experiment.
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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
)
from downstream_outage_prediction.run_experiment import (  # noqa: E402
    TOTAL_POWER_NAME,
    breakdown_metrics,
    outage_metrics,
    power_metrics,
    save_plots,
    save_predictions_csv,
    validate_sample_bandwidths,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-width", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--noise-figure-db", type=float, default=7.0)
    parser.add_argument("--thermal-noise-density-dbm-hz", type=float, default=-174.0)
    parser.add_argument("--required-snr-db", type=float, default=5.0)
    parser.add_argument("--safety-margin-db", type=float, default=0.0)
    parser.add_argument("--allow-bandwidth-mismatch", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for activation memory in the LLM backbone.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_training_regime(args: argparse.Namespace) -> None:
    if args.backbone_training == "auto":
        args.backbone_training = DEFAULT_BACKBONE_TRAINING[args.model]
    if args.quantization == "auto":
        args.quantization = "4bit" if args.backbone_training == "frozen" else "none"


class SampleListDataset(Dataset):
    def __init__(self, samples: list[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.samples[index]


def sample_los_status(sample: Any) -> str:
    return str(getattr(getattr(sample, "semantic_key", None), "los_status", "")).lower()


def load_samples(path: Path) -> list[Any]:
    return PreprocessedCSIDataset.from_pt(str(path)).samples


def target_power_dbw(sample: Any) -> float:
    try:
        value = float(getattr(sample, TOTAL_POWER_NAME))
    except (TypeError, ValueError):
        value = math.nan
    return value


def collate_samples(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    d_tokens = {int(sample.tokens.shape[1]) for sample in samples}
    n_freqs = {int(sample.tokens.shape[2]) for sample in samples}
    if len(d_tokens) != 1 or len(n_freqs) != 1:
        raise ValueError(
            f"Mixed token shapes in one batch: d_token={sorted(d_tokens)} n_freq={sorted(n_freqs)}"
        )
    batch_size = len(samples)
    max_tokens = max(int(sample.n_tokens) for sample in samples)
    d_token = next(iter(d_tokens))
    n_freq = next(iter(n_freqs))
    tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=torch.float32)
    beam_positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    target = torch.empty(batch_size, dtype=torch.float32)
    for row, sample in enumerate(samples):
        count = int(sample.n_tokens)
        tokens[row, :count] = sample.tokens[:count].float()
        beam_positions[row, :count] = sample.beam_positions[:count].float()
        token_mask[row, :count] = True
        target[row] = float(target_power_dbw(sample))
    return {
        "samples": samples,
        "tokens": tokens,
        "beam_positions": beam_positions,
        "token_mask": token_mask,
        "target_dbw": target,
        "group_ids": [str(sample.group_id) for sample in samples],
        "config_keys": [str(sample.config_key) for sample in samples],
        "los_status": [sample_los_status(sample) for sample in samples],
    }


def make_loader(
    samples: list[Any],
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SampleListDataset(samples),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_samples,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        **({"prefetch_factor": 1} if args.num_workers else {}),
    )


def load_backbone(
    args: argparse.Namespace,
    *,
    model_path_override: str | None = None,
    for_test: bool = False,
) -> nn.Module:
    try:
        from transformers import AutoModel, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers, accelerate and bitsandbytes in the LLM environment."
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
            raise ValueError(f"{args.model} does not expose gradient checkpointing.")
        backbone.gradient_checkpointing_enable()
    return backbone


def backbone_hidden_size(backbone: nn.Module) -> int:
    config = backbone.config
    for candidate in (getattr(config, "text_config", None), config):
        size = getattr(candidate, "hidden_size", None)
        if size is not None:
            return int(size)
    raise ValueError("Could not determine LLM hidden size from its config.")


class LLMOutageRegressor(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        max_tokens: int,
        d_token: int,
        n_freq: int,
        adapter_width: int,
        compute_dtype: torch.dtype,
        train_backbone: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.max_tokens = int(max_tokens)
        self.d_token = int(d_token)
        self.n_freq = int(n_freq)
        self.compute_dtype = compute_dtype
        self.train_backbone = train_backbone
        hidden_size = backbone_hidden_size(backbone)
        per_token_dim = d_token * n_freq + 3
        self.projector = nn.Sequential(
            nn.Linear(per_token_dim, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, hidden_size),
        )
        self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        self.readout = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.readout, std=0.02)

    def train(self, mode: bool = True) -> "LLMOutageRegressor":
        super().train(mode)
        self.backbone.train(mode and self.train_backbone)
        return self

    def forward(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens [batch,tokens,d_token,n_freq], got {tuple(tokens.shape)}")
        batch_size, n_tokens, d_token, n_freq = tokens.shape
        if n_tokens > self.max_tokens or d_token != self.d_token or n_freq != self.n_freq:
            raise ValueError(
                "Unexpected CSI shape: "
                f"got {tuple(tokens.shape)}, expected max_tokens<={self.max_tokens}, "
                f"d_token={self.d_token}, n_freq={self.n_freq}."
            )
        x = tokens.float()
        weights = token_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = (weights.sum(dim=(1, 2, 3)) * d_token * n_freq).clamp_min(1.0)
        rms = (x.square().mul(weights).sum(dim=(1, 2, 3)) / denom).sqrt().clamp_min(1e-12)
        x = x / rms.view(batch_size, 1, 1, 1)
        flat = x.reshape(batch_size, n_tokens, d_token * n_freq)
        log_rms = torch.log10(rms).view(batch_size, 1, 1).expand(-1, n_tokens, 1)
        features = torch.cat((flat, log_rms, beam_positions.float()), dim=-1)
        embeds = self.projector(features) + self.position[:, :n_tokens]
        readout = self.readout.expand(batch_size, -1, -1)
        embeds = torch.cat((embeds, readout), dim=1).to(self.compute_dtype)
        attention_mask = torch.cat(
            (
                token_mask,
                torch.ones(batch_size, 1, device=token_mask.device, dtype=torch.bool),
            ),
            dim=1,
        ).long()
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
        return self.head(hidden[:, -1, :].float()).squeeze(-1)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state = {
            "position": self.position.detach().cpu().clone(),
            "readout": self.readout.detach().cpu().clone(),
        }
        for module_name in ("projector", "head"):
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
        for module_name in ("projector", "head"):
            module = getattr(self, module_name)
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(module_state, strict=True)


def send_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["tokens"].to(device, non_blocking=True),
        batch["beam_positions"].to(device, non_blocking=True),
        batch["token_mask"].to(device, non_blocking=True),
        batch["target_dbw"].to(device, non_blocking=True),
    )


def normalize_power(target_dbw: torch.Tensor, device: torch.device) -> torch.Tensor:
    target_index = PHYSICS_TARGET_NAMES.index(TOTAL_POWER_NAME)
    scale = PHYSICS_TARGET_SCALES[target_index].to(device=device, dtype=target_dbw.dtype)
    offset = PHYSICS_TARGET_OFFSETS[target_index].to(device=device, dtype=target_dbw.dtype)
    return (target_dbw - offset) / scale


def denormalize_power(prediction: torch.Tensor) -> torch.Tensor:
    target_index = PHYSICS_TARGET_NAMES.index(TOTAL_POWER_NAME)
    scale = float(PHYSICS_TARGET_SCALES[target_index])
    offset = float(PHYSICS_TARGET_OFFSETS[target_index])
    return prediction.float().cpu() * scale + offset


@torch.inference_mode()
def validation_loss(
    model: LLMOutageRegressor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        tokens, beam_positions, token_mask, target_dbw = send_batch(batch, device)
        valid = torch.isfinite(target_dbw)
        if not bool(valid.any()):
            continue
        prediction = model(tokens, beam_positions, token_mask)
        target = normalize_power(target_dbw, device)
        loss = F.smooth_l1_loss(prediction[valid], target[valid], reduction="sum")
        total += float(loss)
        count += int(valid.sum().item())
    if count == 0:
        raise ValueError("Validation split contains no finite total-power targets.")
    return total / count


@torch.inference_mode()
def predict(
    model: LLMOutageRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    group_ids: list[str] = []
    config_keys: list[str] = []
    los_statuses: list[str] = []
    for batch_number, batch in enumerate(loader, start=1):
        tokens, beam_positions, token_mask, target_dbw = send_batch(batch, device)
        prediction = model(tokens, beam_positions, token_mask)
        predictions.append(denormalize_power(prediction))
        targets.append(target_dbw.float().cpu())
        group_ids.extend(batch["group_ids"])
        config_keys.extend(batch["config_keys"])
        los_statuses.extend(batch["los_status"])
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(f"inference_batches={batch_number}/{len(loader)}", flush=True)
    return {
        "prediction_dbw": torch.cat(predictions).numpy(),
        "target_dbw": torch.cat(targets).numpy(),
        "group_ids": np.asarray(group_ids, dtype=str),
        "config_keys": np.asarray(config_keys, dtype=str),
        "los_status": np.asarray(los_statuses, dtype=str),
    }


def train(
    model: LLMOutageRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    model_path: str,
    input_shape: dict[str, int],
) -> tuple[int, float, list[dict[str, float]]]:
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": adapter_parameters, "lr": args.learning_rate}
    ]
    if args.backbone_training == "full":
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.backbone_learning_rate,
            }
        )
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
            tokens, beam_positions, token_mask, target_dbw = send_batch(batch, device)
            valid = torch.isfinite(target_dbw)
            if not bool(valid.any()):
                continue
            prediction = model(tokens, beam_positions, token_mask)
            target = normalize_power(target_dbw, device)
            loss = F.smooth_l1_loss(prediction[valid], target[valid])
            window_start = (step // args.grad_accum) * args.grad_accum
            window_size = min(args.grad_accum, num_batches - window_start)
            scaler.scale(loss / window_size).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            valid_count = int(valid.sum().item())
            total += float(loss.detach()) * valid_count
            count += valid_count
        if count == 0:
            raise ValueError("Training split contains no finite total-power targets.")
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
            torch.save(
                {
                    "format": "llm_csi_outage_adapter_v1",
                    "model": args.model,
                    "model_path": model_path,
                    "backbone_training": args.backbone_training,
                    "adapter_width": args.adapter_width,
                    "input_shape": input_shape,
                    "target": TOTAL_POWER_NAME,
                    "target_scale": float(PHYSICS_TARGET_SCALES[PHYSICS_TARGET_NAMES.index(TOTAL_POWER_NAME)]),
                    "target_offset": float(PHYSICS_TARGET_OFFSETS[PHYSICS_TARGET_NAMES.index(TOTAL_POWER_NAME)]),
                    "adapter_state_dict": model.adapter_state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "preprocessing": "per-sample RMS shape normalization plus log10(RMS) amplitude feature",
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was saved.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_adapter_state_dict(saved["adapter_state_dict"])
    return best_epoch, best_val, history


def input_shape_from_samples(splits: dict[str, list[Any]]) -> dict[str, int]:
    all_samples = [sample for samples in splits.values() for sample in samples]
    if not all_samples:
        raise ValueError("No samples found.")
    d_tokens = {int(sample.tokens.shape[1]) for sample in all_samples}
    n_freqs = {int(sample.tokens.shape[2]) for sample in all_samples}
    if len(d_tokens) != 1 or len(n_freqs) != 1:
        raise ValueError(
            f"All splits must share d_token/n_freq, got d_token={sorted(d_tokens)} "
            f"n_freq={sorted(n_freqs)}."
        )
    max_tokens = max(int(sample.n_tokens) for sample in all_samples)
    return {
        "max_tokens": max_tokens,
        "d_token": next(iter(d_tokens)),
        "n_freq": next(iter(n_freqs)),
    }


def finite_target_count(samples: list[Any]) -> int:
    return sum(math.isfinite(target_power_dbw(sample)) for sample in samples)


def evaluate_and_save(
    *,
    model: LLMOutageRegressor,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    inference = predict(model, loader, device)
    valid_mask = np.isfinite(inference["target_dbw"]) & np.isfinite(inference["prediction_dbw"])
    if not bool(np.any(valid_mask)):
        raise ValueError(f"{split_name} split contains no valid total-power pairs.")
    for key in inference:
        if isinstance(inference[key], np.ndarray):
            inference[key] = inference[key][valid_mask]

    prediction_dbw = inference["prediction_dbw"].astype(np.float64)
    target_dbw = inference["target_dbw"].astype(np.float64)
    prediction_dbm = prediction_dbw + 30.0
    target_dbm = target_dbw + 30.0
    noise_power_dbm = (
        args.thermal_noise_density_dbm_hz
        + 10.0 * math.log10(args.bandwidth_hz)
        + args.noise_figure_db
    )
    noise_power_dbw = noise_power_dbm - 30.0
    true_snr_db = target_dbw - noise_power_dbw
    predicted_snr_db = prediction_dbw - noise_power_dbw
    true_outage = true_snr_db < args.required_snr_db
    predicted_threshold = args.required_snr_db + args.safety_margin_db
    predicted_outage = predicted_snr_db < predicted_threshold

    metrics: dict[str, Any] = {
        "format": "llm_csi_power_rule_outage_v1",
        "split": split_name,
        "model": args.model,
        "backbone_training": args.backbone_training,
        "quantization": args.quantization,
        "data_count": int(valid_mask.size),
        "valid_sample_count": int(np.sum(valid_mask)),
        "invalid_sample_count": int(valid_mask.size - np.sum(valid_mask)),
        "bandwidth_hz": args.bandwidth_hz,
        "noise_figure_db": args.noise_figure_db,
        "thermal_noise_density_dbm_hz": args.thermal_noise_density_dbm_hz,
        "noise_power_dbm": noise_power_dbm,
        "noise_power_dbw": noise_power_dbw,
        "required_snr_db": args.required_snr_db,
        "safety_margin_db": args.safety_margin_db,
        "predicted_decision_threshold_db": predicted_threshold,
        "power": power_metrics(prediction_dbw, target_dbw),
        "outage": outage_metrics(predicted_outage, true_outage),
        "los_nlos_breakdown": breakdown_metrics(
            prediction_dbw,
            target_dbw,
            predicted_outage,
            true_outage,
            inference["los_status"],
        ),
    }

    rows: list[dict[str, Any]] = []
    error_db = prediction_dbw - target_dbw
    for index in range(len(prediction_dbw)):
        rows.append(
            {
                "sample_index": index,
                "group_id": inference["group_ids"][index],
                "config_key": inference["config_keys"][index],
                "los_status": inference["los_status"][index],
                "true_total_rx_power_dbw": target_dbw[index],
                "predicted_total_rx_power_dbw": prediction_dbw[index],
                "true_total_rx_power_dbm": target_dbm[index],
                "predicted_total_rx_power_dbm": prediction_dbm[index],
                "power_error_db": error_db[index],
                "absolute_power_error_db": abs(error_db[index]),
                "noise_power_dbm": noise_power_dbm,
                "true_snr_db": true_snr_db[index],
                "predicted_snr_db": predicted_snr_db[index],
                "required_snr_db": args.required_snr_db,
                "predicted_decision_threshold_db": predicted_threshold,
                "true_link_status": "outage" if true_outage[index] else "normal",
                "predicted_link_status": "outage" if predicted_outage[index] else "normal",
                "correct": bool(predicted_outage[index] == true_outage[index]),
                "missed_outage": bool(true_outage[index] and not predicted_outage[index]),
                "false_alarm": bool(not true_outage[index] and predicted_outage[index]),
            }
        )

    split_dir = args.output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    save_predictions_csv(split_dir / "predictions.csv", rows)
    np.savez_compressed(
        split_dir / "predictions.npz",
        group_id=inference["group_ids"],
        config_key=inference["config_keys"],
        los_status=inference["los_status"],
        true_total_rx_power_dbw=target_dbw,
        predicted_total_rx_power_dbw=prediction_dbw,
        power_error_db=error_db,
        noise_power_dbm=np.asarray(noise_power_dbm),
        true_snr_db=true_snr_db,
        predicted_snr_db=predicted_snr_db,
        true_outage=true_outage,
        predicted_outage=predicted_outage,
    )
    if not args.no_plots:
        save_plots(
            split_dir,
            prediction_dbw,
            target_dbw,
            predicted_snr_db,
            true_snr_db,
            predicted_outage,
            true_outage,
            args.required_snr_db,
            metrics,
        )
    print(
        f"{split_name}_power_MAE_db={metrics['power']['mae_db']:.4f} "
        f"{split_name}_accuracy_pct={metrics['outage']['accuracy_pct']:.4f} "
        f"{split_name}_outage_f1_pct={metrics['outage']['outage_f1_pct']:.4f}",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    resolve_training_regime(args)
    if any(
        value <= 0
        for value in (
            args.batch_size,
            args.grad_accum,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.backbone_learning_rate,
            args.adapter_width,
            args.cpu_threads,
            args.bandwidth_hz,
        )
    ) or args.num_workers < 0:
        raise ValueError("Training sizes/rates must be positive and num-workers nonnegative.")
    if args.safety_margin_db < 0.0:
        raise ValueError("--safety-margin-db must be nonnegative.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LLM baseline training requires a CUDA device.")
    if args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU lacks BF16 support; use --compute-dtype fp16.")
    if args.backbone_training == "full" and args.quantization != "none":
        raise ValueError("Full-parameter LLM training cannot update 4-bit base weights.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an experiment.")

    for path in (args.train_data, args.val_data, args.test_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)

    splits = {
        "train": load_samples(args.train_data),
        "val": load_samples(args.val_data),
        "test": load_samples(args.test_data),
    }
    for split_name, samples in splits.items():
        if finite_target_count(samples) != len(samples):
            raise ValueError(f"{split_name} split has missing {TOTAL_POWER_NAME} targets.")
        validate_sample_bandwidths(
            samples,
            expected_hz=args.bandwidth_hz,
            allow_mismatch=args.allow_bandwidth_mismatch,
        )
    input_shape = input_shape_from_samples(splits)
    loaders = {
        name: make_loader(samples, args, shuffle=name == "train")
        for name, samples in splits.items()
    }
    model_path = args.model_path or MODEL_IDS[args.model]
    summary = {
        "model": args.model,
        "model_path": model_path,
        "device": str(device),
        "quantization": args.quantization,
        "backbone_training": args.backbone_training,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "input_shape": input_shape,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **summary,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)

    backbone = load_backbone(args)
    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    model = LLMOutageRegressor(
        backbone=backbone,
        adapter_width=args.adapter_width,
        compute_dtype=dtype,
        train_backbone=args.backbone_training == "full",
        **input_shape,
    )
    del backbone
    model.projector.to(device)
    model.head.to(device)
    model.position.data = model.position.data.to(device)
    model.readout.data = model.readout.data.to(device)

    adapter_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    )
    backbone_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    )
    print(
        f"trainable_adapter_parameters={adapter_trainable} "
        f"trainable_backbone_parameters={backbone_trainable}",
        flush=True,
    )
    best_epoch, best_val, history = train(
        model,
        loaders["train"],
        loaders["val"],
        args,
        device,
        model_path,
        input_shape,
    )
    if args.backbone_training == "full":
        del model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        model.backbone = load_backbone(
            args,
            model_path_override=str(args.output_dir / "best_backbone"),
            for_test=True,
        )
        model.train_backbone = False

    val_metrics = evaluate_and_save(
        model=model,
        loader=loaders["val"],
        args=args,
        device=device,
        split_name="validation",
    )
    test_metrics = evaluate_and_save(
        model=model,
        loader=loaders["test"],
        args=args,
        device=device,
        split_name="test",
    )
    result = {
        "model": args.model,
        "model_path": model_path,
        "baseline_type": (
            "full_finetune_llm_and_csi_adapter"
            if args.backbone_training == "full"
            else "frozen_llm_trainable_csi_adapter"
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        row = {
            "model": args.model,
            "backbone_training": args.backbone_training,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "validation_power_mae_db": val_metrics["power"]["mae_db"],
            "validation_accuracy_pct": val_metrics["outage"]["accuracy_pct"],
            "validation_outage_f1_pct": val_metrics["outage"]["outage_f1_pct"],
            "test_power_mae_db": test_metrics["power"]["mae_db"],
            "test_accuracy_pct": test_metrics["outage"]["accuracy_pct"],
            "test_outage_f1_pct": test_metrics["outage"]["outage_f1_pct"],
        }
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(
        json.dumps(
            {
                "model": args.model,
                "best_epoch": best_epoch,
                "test_power_mae_db": test_metrics["power"]["mae_db"],
                "test_accuracy_pct": test_metrics["outage"]["accuracy_pct"],
                "test_outage_f1_pct": test_metrics["outage"]["outage_f1_pct"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
