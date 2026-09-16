from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.data_io import (  # noqa: E402
    MCSCorpus,
    MCSDataset,
    collate_mcs,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_mcs_adaptation.pretrained import (  # noqa: E402
    PHYSICS_FEATURE_ORDER,
    build_pretrained_model,
    extract_full_mcs_features,
    physics_training_warnings,
    require_mcs_physics_checkpoint,
)
from data.dataset import PHYSICS_TARGET_NAMES, PHYSICS_TARGET_SCALES  # noqa: E402
from models.encoder import CSIEncoder  # noqa: E402


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DATA = (
    DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/d2los_300k_mcs_power_profiles.pt"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt"
)
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_mcs_adaptation/outputs/seed23421"
DISPLAY_NAMES = {
    "scratch": "Scratch + Tx power",
    "scratch_physics": "Scratch encoder + Tx power + pretrained physics",
    "full_embedding": "Full embedding + Tx power",
    "full_physics": "Full embedding + Tx power + adjusted predicted physics",
}
TX_POWER_FEATURE_SCALE_DBM = 30.0
TX_POWER_FEATURE_NAME = "transmit_power_dbm_div_30"
TOTAL_POWER_FEATURE_INDEX = PHYSICS_FEATURE_ORDER.index(
    "predicted_total_received_power_normalized"
)
TOTAL_POWER_TARGET_INDEX = PHYSICS_TARGET_NAMES.index("total_received_power_dbw")
TOTAL_POWER_TARGET_SCALE_DB = float(PHYSICS_TARGET_SCALES[TOTAL_POWER_TARGET_INDEX])
FULL_PHYSICS_AUXILIARY_ORDER = (
    TX_POWER_FEATURE_NAME,
    "predicted_total_received_power_at_current_tx_normalized",
    *PHYSICS_FEATURE_ORDER[1:],
)
ADJUSTED_PHYSICS_FEATURE_ORDER = FULL_PHYSICS_AUXILIARY_ORDER[1:]


def normalized_transmit_power(transmit_power_dbm: torch.Tensor) -> torch.Tensor:
    """Use a fixed physical scale, independent of train/validation/test statistics."""
    return transmit_power_dbm.float().unsqueeze(-1) / TX_POWER_FEATURE_SCALE_DBM


def operating_point_features(
    corpus: MCSCorpus,
    base_physics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build explicit Tx input and adjust only predicted total received power."""
    if base_physics.shape != (len(corpus.samples), len(PHYSICS_FEATURE_ORDER)):
        raise ValueError("Unexpected frozen physics feature shape")
    transmit_power = normalized_transmit_power(corpus.transmit_power_dbm)
    adjusted_physics = base_physics.clone()
    adjusted_physics[:, TOTAL_POWER_FEATURE_INDEX] += (
        corpus.power_offset_db / TOTAL_POWER_TARGET_SCALE_DB
    )
    return transmit_power, adjusted_physics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate downstream MCS link adaptation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DISPLAY_NAMES),
        default=list(DISPLAY_NAMES),
    )
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument(
        "--safety-penalty-weight",
        type=float,
        default=0.0,
        help="Optional expected aggressive-class distance penalty; zero is pure cross-entropy.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCSHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class RawAmplitudeStatistics(nn.Module):
    """Scale-sensitive features from the unchanged base-power propagation CSI."""

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        weights = token_mask.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        raw_power = (tokens * weights).square()
        valid_beams = token_mask.sum(dim=1).to(tokens.dtype).clamp(min=1.0)
        elements = (valid_beams * tokens.shape[2] * tokens.shape[3]).clamp(min=1.0)
        mean_power = raw_power.sum(dim=(1, 2, 3)) / elements
        peak_power = raw_power.amax(dim=(1, 2, 3))
        beam_power = raw_power.mean(dim=(2, 3))
        beam_mean = beam_power.sum(dim=1) / valid_beams
        beam_peak = beam_power.amax(dim=1)
        frequency_power = raw_power.sum(dim=(1, 2)) / (
            valid_beams[:, None] * tokens.shape[2]
        )
        frequency_db = 10.0 * torch.log10(frequency_power.clamp(min=1e-30))
        q10 = torch.quantile(frequency_db, 0.1, dim=1)
        q90 = torch.quantile(frequency_db, 0.9, dim=1)
        return torch.stack(
            [
                10.0 * torch.log10(mean_power.clamp(min=1e-30)) / 100.0,
                10.0 * torch.log10(peak_power.clamp(min=1e-30)) / 100.0,
                10.0
                * torch.log10(
                    (beam_peak / beam_mean.clamp(min=1e-30)).clamp(min=1e-30)
                )
                / 20.0,
                frequency_db.std(dim=1, correction=0) / 20.0,
                (q90 - q10) / 20.0,
            ],
            dim=-1,
        )


class ScratchMCSModel(nn.Module):
    """Random CSI encoder plus base-CSI amplitude statistics and explicit Tx power."""

    def __init__(self, model_info: dict[str, Any], num_classes: int):
        super().__init__()
        self.encoder = CSIEncoder(
            d_token=int(model_info["d_token"]),
            d_model=384,
            d_clip=256,
            token_norm_mode=str(model_info["token_norm_mode"]),
            use_continuous_config_encoding=bool(model_info["use_continuous_config_encoding"]),
        )
        # The main encoder is per-sample normalized. These statistics preserve the
        # base-link scale, while Tx power identifies the requested operating point.
        self.raw_amplitude = RawAmplitudeStatistics()
        self.head = MCSHead(256 + 5 + 1, num_classes)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        embedding = F.normalize(self.encoder(**encoder_kwargs(batch)), dim=-1)
        raw_amplitude = self.raw_amplitude(batch["tokens"], batch["token_mask"])
        transmit_power = normalized_transmit_power(batch["transmit_power_dbm"])
        return self.head(torch.cat([embedding, raw_amplitude, transmit_power], dim=-1))


class ScratchPhysicsMCSModel(nn.Module):
    """Trainable random encoder with frozen pretrained physics predictions."""

    def __init__(self, model_info: dict[str, Any], num_classes: int):
        super().__init__()
        self.encoder = CSIEncoder(
            d_token=int(model_info["d_token"]),
            d_model=384,
            d_clip=256,
            token_norm_mode=str(model_info["token_norm_mode"]),
            use_continuous_config_encoding=bool(model_info["use_continuous_config_encoding"]),
        )
        self.head = MCSHead(256 + 1 + len(PHYSICS_FEATURE_ORDER), num_classes)

    def forward(
        self,
        batch: dict[str, Any],
        predicted_physics: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_physics.shape != (
            batch["tokens"].shape[0],
            len(PHYSICS_FEATURE_ORDER),
        ):
            raise ValueError("Scratch+Physics received invalid physics features")
        embedding = F.normalize(self.encoder(**encoder_kwargs(batch)), dim=-1)
        transmit_power = normalized_transmit_power(batch["transmit_power_dbm"])
        predicted_physics = predicted_physics.to(dtype=embedding.dtype)
        transmit_power = transmit_power.to(dtype=embedding.dtype)
        return self.head(
            torch.cat([embedding, transmit_power, predicted_physics], dim=-1)
        )


def make_loader(
    corpus: MCSCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        MCSDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_mcs,
        generator=generator,
    )


def class_weights(corpus: MCSCorpus, mode: str) -> torch.Tensor | None:
    labels = corpus.labels[corpus.split_indices["train"]]
    counts = torch.bincount(labels, minlength=len(corpus.mcs_names) + 1).float()
    print(f"training_class_counts={counts.long().tolist()}")
    if bool((counts == 0).any()):
        missing = torch.flatnonzero(counts == 0).tolist()
        raise ValueError(
            f"Training split has no samples for classes {missing}; "
            "recalibrate or redesign the system profile"
        )
    if mode == "none":
        return None
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = labels.numel() / (present.sum() * counts[present])
    print(f"cross_entropy_class_weights={weights.tolist()}")
    return weights


def training_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    safety_penalty_weight: float,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, weight=weights)
    if safety_penalty_weight > 0.0:
        classes = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        aggressive_distance = (classes[None, :] - target[:, None]).clamp(min=0.0)
        expected_distance = (torch.softmax(logits, dim=-1) * aggressive_distance).sum(dim=-1)
        loss = loss + safety_penalty_weight * expected_distance.mean()
    return loss


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def selection_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    corpus: MCSCorpus,
) -> dict[str, float]:
    prediction = prediction.long().cpu()
    target = target.long().cpu()
    indices = indices.long().cpu()
    count = len(prediction)
    predicted_mcs = prediction - 1
    oracle_mcs = target - 1
    predicted_active = predicted_mcs >= 0
    oracle_active = oracle_mcs >= 0
    selected_bler = torch.zeros(count, dtype=torch.float32)
    selected_efficiency = torch.zeros(count, dtype=torch.float32)
    if bool(predicted_active.any()):
        selected_bler[predicted_active] = corpus.estimated_bler_per_mcs[
            indices[predicted_active], predicted_mcs[predicted_active]
        ]
        selected_efficiency[predicted_active] = corpus.spectral_efficiency[
            predicted_mcs[predicted_active]
        ]
    oracle_bler = torch.zeros(count, dtype=torch.float32)
    oracle_efficiency = torch.zeros(count, dtype=torch.float32)
    if bool(oracle_active.any()):
        oracle_bler[oracle_active] = corpus.estimated_bler_per_mcs[
            indices[oracle_active], oracle_mcs[oracle_active]
        ]
        oracle_efficiency[oracle_active] = corpus.spectral_efficiency[oracle_mcs[oracle_active]]
    target_bler = float(corpus.system["target_bler"])
    violation = predicted_active & (selected_bler > target_bler + 1e-7)
    goodput = selected_efficiency * (1.0 - selected_bler)
    oracle_goodput = oracle_efficiency * (1.0 - oracle_bler)
    model_goodput = float(goodput.mean())
    oracle_mean_goodput = float(oracle_goodput.mean())
    outage_target = target == 0
    outage_prediction = prediction == 0
    outage_tp = int((outage_target & outage_prediction).sum())
    outage_fn = int((outage_target & ~outage_prediction).sum())
    outage_fp = int((~outage_target & outage_prediction).sum())
    active_selected_bler = (
        float(selected_bler[predicted_active].mean())
        if bool(predicted_active.any())
        else math.nan
    )
    return {
        "accuracy_pct": float(100.0 * (prediction == target).float().mean()),
        "aggressive_selection_rate_pct": float(100.0 * (prediction > target).float().mean()),
        "conservative_selection_rate_pct": float(100.0 * (prediction < target).float().mean()),
        "mean_absolute_mcs_error": float((prediction - target).abs().float().mean()),
        "bler_violation_rate_pct": float(100.0 * violation.float().mean()),
        "mean_selected_bler_pct": float(100.0 * selected_bler.mean()),
        "mean_active_transmission_bler_pct": 100.0 * active_selected_bler,
        "mean_spectral_efficiency_bit_s_hz": float(selected_efficiency.mean()),
        "mean_goodput_bit_s_hz": model_goodput,
        "oracle_goodput_bit_s_hz": oracle_mean_goodput,
        "goodput_ratio_pct": 100.0 * safe_ratio(model_goodput, oracle_mean_goodput),
        "predicted_outage_rate_pct": float(100.0 * outage_prediction.float().mean()),
        "true_outage_rate_pct": float(100.0 * outage_target.float().mean()),
        "outage_recall_pct": 100.0 * safe_ratio(outage_tp, outage_tp + outage_fn),
        "outage_false_alarm_rate_pct": 100.0 * safe_ratio(outage_fp, int((~outage_target).sum())),
    }


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    flat = target.long().cpu() * classes + prediction.long().cpu()
    return torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.inference_mode()
def evaluate_trainable_encoder(
    model: nn.Module,
    loader: DataLoader,
    corpus: MCSCorpus,
    device: torch.device,
    weights: torch.Tensor | None,
    args: argparse.Namespace,
    auxiliary_features: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_all, targets, indices = [], [], []
    loss_sum = 0.0
    samples = 0
    device_weights = weights.to(device) if weights is not None else None
    amp = device.type == "cuda" and not args.no_amp
    for cpu_batch in loader:
        auxiliary_batch = None
        if auxiliary_features is not None:
            auxiliary_batch = auxiliary_features[cpu_batch["indices"]].to(
                device, non_blocking=True
            )
        batch = move_tensor_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = (
                model(batch)
                if auxiliary_batch is None
                else model(batch, auxiliary_batch)
            )
            loss = training_loss(
                logits, batch["target_class"], device_weights, args.safety_penalty_weight
            )
        batch_count = len(batch["target_class"])
        loss_sum += float(loss) * batch_count
        samples += batch_count
        logits_all.append(logits.float().cpu())
        targets.append(batch["target_class"].cpu())
        indices.append(batch["indices"].cpu())
    logits = torch.cat(logits_all)
    target = torch.cat(targets)
    index = torch.cat(indices)
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    return {
        "loss": loss_sum / max(samples, 1),
        "metrics": selection_metrics(prediction, target, index, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target,
        "indices": index,
    }


def train_trainable_encoder(
    name: str,
    model: nn.Module,
    corpus: MCSCorpus,
    args: argparse.Namespace,
    device: torch.device,
    weights: torch.Tensor | None,
    auxiliary_features: torch.Tensor | None,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(args.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.scratch_lr, weight_decay=args.weight_decay)
    train_loader = make_loader(corpus, corpus.split_indices["train"], args, True, 11)
    val_loader = make_loader(corpus, corpus.split_indices["val"], args, False, 12)
    test_loader = make_loader(corpus, corpus.split_indices["test"], args, False, 13)
    device_weights = weights.to(device) if weights is not None else None
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_state = None
    best_val_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for cpu_batch in train_loader:
            auxiliary_batch = None
            if auxiliary_features is not None:
                auxiliary_batch = auxiliary_features[cpu_batch["indices"]].to(
                    device, non_blocking=True
                )
            batch = move_tensor_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = (
                    model(batch)
                    if auxiliary_batch is None
                    else model(batch, auxiliary_batch)
                )
                loss = training_loss(
                    logits,
                    batch["target_class"],
                    device_weights,
                    args.safety_penalty_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val = evaluate_trainable_encoder(
            model,
            val_loader,
            corpus,
            device,
            weights,
            args,
            auxiliary_features=auxiliary_features,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val["loss"]),
            **val["metrics"],
        }
        history.append(record)
        print(
            f"{name} epoch={epoch:03d} train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} acc={record['accuracy_pct']:.2f}% "
            f"aggressive={record['aggressive_selection_rate_pct']:.2f}%",
            flush=True,
        )
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best_state = cpu_state_dict(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "best_val_loss": best_val_loss,
            "history": history,
            **checkpoint_metadata,
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_trainable_encoder(
        model,
        test_loader,
        corpus,
        device,
        weights,
        args,
        auxiliary_features=auxiliary_features,
    )


def train_scratch(
    corpus: MCSCorpus,
    model_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    weights: torch.Tensor | None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    model = ScratchMCSModel(model_info, len(corpus.mcs_names) + 1)
    return train_trainable_encoder(
        "scratch",
        model,
        corpus,
        args,
        device,
        weights,
        auxiliary_features=None,
        checkpoint_metadata={
            "contains_pretrained_weights": False,
            "uses_pretrained_physics_features": False,
            "uses_oracle_physics_as_input": False,
            "uses_raw_csi_amplitude_statistics": True,
            "uses_explicit_transmit_power": True,
            "transmit_power_feature": TX_POWER_FEATURE_NAME,
            "csi_amplitude_scaled_by_power_profile": False,
        },
    )


def train_scratch_physics(
    corpus: MCSCorpus,
    model_info: dict[str, Any],
    adjusted_physics: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    weights: torch.Tensor | None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    model = ScratchPhysicsMCSModel(model_info, len(corpus.mcs_names) + 1)
    return train_trainable_encoder(
        "scratch_physics",
        model,
        corpus,
        args,
        device,
        weights,
        auxiliary_features=adjusted_physics,
        checkpoint_metadata={
            "contains_pretrained_weights": False,
            "uses_pretrained_physics_features": True,
            "pretrained_physics_checkpoint": str(args.checkpoint.resolve()),
            "auxiliary_feature_order": list(FULL_PHYSICS_AUXILIARY_ORDER),
            "uses_oracle_physics_as_input": False,
            "uses_raw_csi_amplitude_statistics": False,
            "uses_explicit_transmit_power": True,
            "transmit_power_feature": TX_POWER_FEATURE_NAME,
            "csi_amplitude_scaled_by_power_profile": False,
        },
    )


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@torch.inference_mode()
def load_or_extract_features(
    corpus: MCSCorpus,
    data_path: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    pretrained_model: Any,
    prototype_keys: list,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = args.feature_cache or args.output_dir / "frozen_features.pt"
    signature = {
        "format": "csi_mcs_frozen_features_v2_explicit_tx_power",
        "data": file_signature(data_path),
        "checkpoint": file_signature(checkpoint_path),
        "sample_count": len(corpus.samples),
        "physics_order": list(PHYSICS_FEATURE_ORDER),
        "csi_amplitude_scaled_by_power_profile": False,
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(cached, dict)
            and cached.get("signature") == signature
            and tuple(cached.get("embedding", torch.empty(0)).shape) == (len(corpus.samples), 256)
            and tuple(cached.get("physics", torch.empty(0)).shape)
            == (len(corpus.samples), len(PHYSICS_FEATURE_ORDER))
            and cached.get("uses_oracle_physics_as_input") is False
        ):
            print(f"loaded_feature_cache={cache_path}")
            return cached["embedding"].float(), cached["physics"].float()
        print(f"ignoring_stale_feature_cache={cache_path}")
    loader = make_loader(corpus, torch.arange(len(corpus.samples)), args, False, 21)
    embedding = torch.empty(len(corpus.samples), 256, dtype=torch.float32)
    physics = torch.empty(len(corpus.samples), len(PHYSICS_FEATURE_ORDER), dtype=torch.float32)
    amp = device.type == "cuda" and not args.no_amp
    checkpoint_args = dict(checkpoint.get("args", {}))
    for step, cpu_batch in enumerate(loader, start=1):
        batch = move_tensor_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            batch_embedding, batch_physics = extract_full_mcs_features(
                pretrained_model, batch, prototype_keys, checkpoint_args
            )
        index = batch["indices"].cpu()
        embedding[index] = batch_embedding.float().cpu()
        physics[index] = batch_physics.float().cpu()
        if step % 100 == 0 or step == len(loader):
            print(f"feature_batches={step}/{len(loader)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": embedding,
            "physics": physics,
            "signature": signature,
            "physics_order": list(PHYSICS_FEATURE_ORDER),
            "uses_oracle_physics_as_input": False,
        },
        cache_path,
    )
    return embedding, physics


@torch.inference_mode()
def evaluate_head(
    head: MCSHead,
    features: torch.Tensor,
    indices: torch.Tensor,
    corpus: MCSCorpus,
    device: torch.device,
    args: argparse.Namespace,
    weights: torch.Tensor | None,
) -> dict[str, Any]:
    head.eval()
    logits_all = []
    loss_sum = 0.0
    sample_count = 0
    device_weights = weights.to(device) if weights is not None else None
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        logits = head(features[batch_indices].to(device)).float()
        target = corpus.labels[batch_indices].to(device)
        loss = training_loss(logits, target, device_weights, args.safety_penalty_weight)
        loss_sum += float(loss) * len(batch_indices)
        sample_count += len(batch_indices)
        logits_all.append(logits.cpu())
    logits = torch.cat(logits_all)
    target = corpus.labels[indices]
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    return {
        "loss": loss_sum / max(sample_count, 1),
        "metrics": selection_metrics(prediction, target, indices, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target.clone(),
        "indices": indices.clone(),
    }


def train_frozen_head(
    name: str,
    features: torch.Tensor,
    corpus: MCSCorpus,
    args: argparse.Namespace,
    device: torch.device,
    weights: torch.Tensor | None,
    auxiliary_feature_order: list[str] | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    head = MCSHead(features.shape[1], len(corpus.mcs_names) + 1).to(device)
    train_indices = corpus.split_indices["train"]
    val_indices = corpus.split_indices["val"]
    test_indices = corpus.split_indices["test"]
    loader = DataLoader(
        TensorDataset(features[train_indices], corpus.labels[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 31),
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    device_weights = weights.to(device) if weights is not None else None
    best_state = None
    best_val_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(
                head(feature_batch), target_batch, device_weights, args.safety_penalty_weight
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate_head(head, features, val_indices, corpus, device, args, weights)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val["loss"]),
            **val["metrics"],
        }
        history.append(record)
        print(
            f"{name} epoch={epoch:03d} train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} acc={record['accuracy_pct']:.2f}% "
            f"aggressive={record['aggressive_selection_rate_pct']:.2f}%",
            flush=True,
        )
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best_state = cpu_state_dict(head)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    head.load_state_dict(best_state)
    torch.save(
        {
            "head_state": best_state,
            "input_dim": int(features.shape[1]),
            "best_val_loss": best_val_loss,
            "history": history,
            "auxiliary_feature_order": list(auxiliary_feature_order or []),
            "uses_oracle_physics_as_input": False,
            "uses_explicit_transmit_power": True,
            "transmit_power_feature": TX_POWER_FEATURE_NAME,
            "csi_amplitude_scaled_by_power_profile": False,
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_head(head, features, test_indices, corpus, device, args, weights)


def save_predictions(name: str, result: dict[str, Any], corpus: MCSCorpus, output_dir: Path) -> None:
    indices = result["indices"].long()
    prediction = result["prediction"].long()
    target = result["target"].long()
    predicted_mcs = prediction - 1
    selected_bler = torch.zeros(len(indices), dtype=torch.float32)
    selected_efficiency = torch.zeros(len(indices), dtype=torch.float32)
    active = predicted_mcs >= 0
    if bool(active.any()):
        selected_bler[active] = corpus.estimated_bler_per_mcs[
            indices[active], predicted_mcs[active]
        ]
        selected_efficiency[active] = corpus.spectral_efficiency[predicted_mcs[active]]
    selected_goodput = selected_efficiency * (1.0 - selected_bler)
    target_bler = float(corpus.system["target_bler"])
    violation = active & (selected_bler > target_bler + 1e-7)
    rows = []
    for row, corpus_index in enumerate(indices.tolist()):
        predicted_class = int(prediction[row])
        target_class = int(target[row])
        rows.append(
            {
                "corpus_index": corpus_index,
                "group_id": corpus.group_ids[corpus_index],
                "physical_group_id": corpus.physical_group_ids[corpus_index],
                "transmit_power_dbm": float(corpus.transmit_power_dbm[corpus_index]),
                "power_offset_db": float(corpus.power_offset_db[corpus_index]),
                "average_snr_db": float(corpus.average_snr_db[corpus_index]),
                "oracle_class": target_class,
                "oracle_mcs": target_class - 1,
                "predicted_class": predicted_class,
                "predicted_mcs": predicted_class - 1,
                "selected_bler": float(selected_bler[row]),
                "selected_spectral_efficiency_bit_s_hz": float(selected_efficiency[row]),
                "estimated_goodput_bit_s_hz": float(selected_goodput[row]),
                "bler_violation": bool(violation[row]),
                "aggressive": predicted_class > target_class,
                "conservative": predicted_class < target_class,
                "correct": predicted_class == target_class,
            }
        )
    with (output_dir / f"{name}_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / f"{name}_test_predictions.npz",
        indices=indices.numpy(),
        group_ids=np.asarray([corpus.group_ids[index] for index in indices]),
        physical_group_ids=np.asarray([corpus.physical_group_ids[index] for index in indices]),
        transmit_power_dbm=corpus.transmit_power_dbm[indices].numpy(),
        power_offset_db=corpus.power_offset_db[indices].numpy(),
        predicted_class=prediction.numpy(),
        oracle_class=target.numpy(),
        predicted_probabilities=result["probability"].numpy(),
        average_snr_db=corpus.average_snr_db[indices].numpy(),
        effective_snr_per_mcs=corpus.effective_snr_per_mcs[indices].numpy(),
        estimated_bler_per_mcs=corpus.estimated_bler_per_mcs[indices].numpy(),
        selected_bler=selected_bler.numpy(),
        selected_spectral_efficiency=selected_efficiency.numpy(),
        estimated_goodput=selected_goodput.numpy(),
        bler_violation=violation.numpy(),
    )


def save_confusion(
    name: str, result: dict[str, Any], corpus: MCSCorpus, output_dir: Path
) -> np.ndarray:
    labels = ["Outage", *corpus.mcs_names]
    matrix = confusion_matrix(result["prediction"], result["target"], len(labels)).numpy()
    with (output_dir / f"{name}_confusion.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])
    return matrix


def plot_confusion(name: str, matrix: np.ndarray, labels: list[str], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 6.4))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Oracle class")
    axis.set_title(DISPLAY_NAMES[name])
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion.png", dpi=220)
    plt.close(fig)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.patience <= 0
        or args.batch_size <= 0
        or args.num_workers < 0
        or args.safety_penalty_weight < 0.0
    ):
        raise ValueError("Invalid training arguments")
    if not args.data.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing data or checkpoint: {args.data}, {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    corpus = MCSCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Keep the frozen model on CPU while Scratch is trained so the two encoders
    # do not occupy GPU memory at the same time.
    pretrained, prototype_keys, model_info = build_pretrained_model(
        checkpoint, torch.device("cpu")
    )
    if corpus.samples[0].tokens.shape[1] != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={corpus.samples[0].tokens.shape[1]} but checkpoint expects {model_info['d_token']}"
        )
    selected = list(dict.fromkeys(args.models))
    if any(name in selected for name in ("scratch_physics", "full_physics")):
        require_mcs_physics_checkpoint(checkpoint, pretrained)
        for message in physics_training_warnings(dict(checkpoint.get("args", {}))):
            warnings.warn(
                message
                + "; models using pretrained physics must be interpreted cautiously"
            )
    weights = class_weights(corpus, args.class_weight)
    print(
        f"device={device} samples={len(corpus.samples)} "
        f"splits={{{', '.join(f'{key}:{len(value)}' for key, value in corpus.split_indices.items())}}} "
        f"models={','.join(selected)}"
    )
    start = time.time()
    results: dict[str, dict[str, Any]] = {}
    if "scratch" in selected:
        results["scratch"] = train_scratch(corpus, model_info, args, device, weights)

    if any(
        name in selected
        for name in ("scratch_physics", "full_embedding", "full_physics")
    ):
        pretrained.to(device)
        embedding, physics = load_or_extract_features(
            corpus,
            args.data,
            args.checkpoint,
            checkpoint,
            pretrained,
            prototype_keys,
            args,
            device,
        )
        pretrained.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        transmit_power, adjusted_physics = operating_point_features(corpus, physics)
        if "scratch_physics" in selected:
            results["scratch_physics"] = train_scratch_physics(
                corpus,
                model_info,
                adjusted_physics,
                args,
                device,
                weights,
            )
        if "full_embedding" in selected:
            results["full_embedding"] = train_frozen_head(
                "full_embedding",
                torch.cat([embedding, transmit_power], dim=-1),
                corpus,
                args,
                device,
                weights,
                auxiliary_feature_order=[TX_POWER_FEATURE_NAME],
            )
        if "full_physics" in selected:
            results["full_physics"] = train_frozen_head(
                "full_physics",
                torch.cat([embedding, transmit_power, adjusted_physics], dim=-1),
                corpus,
                args,
                device,
                weights,
                auxiliary_feature_order=list(FULL_PHYSICS_AUXILIARY_ORDER),
            )

    metric_rows = []
    labels = ["Outage", *corpus.mcs_names]
    for name, result in results.items():
        save_predictions(name, result, corpus, args.output_dir)
        matrix = save_confusion(name, result, corpus, args.output_dir)
        if not args.no_plots:
            plot_confusion(name, matrix, labels, args.output_dir)
        metric_rows.append({"model": name, "display_name": DISPLAY_NAMES[name], **result["metrics"]})
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {
        "metrics": {
            row["model"]: {key: value for key, value in row.items() if key not in {"model", "display_name"}}
            for row in metric_rows
        },
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split_sizes": {name: len(indices) for name, indices in corpus.split_indices.items()},
        "class_histogram": dict(Counter(corpus.labels.tolist())),
        "class_mapping": {0: "Outage", **{index + 1: name for index, name in enumerate(corpus.mcs_names)}},
        "target_bler": float(corpus.system["target_bler"]),
        "base_physics_feature_order": list(PHYSICS_FEATURE_ORDER),
        "adjusted_physics_feature_order": list(ADJUSTED_PHYSICS_FEATURE_ORDER),
        "full_physics_auxiliary_feature_order": list(FULL_PHYSICS_AUXILIARY_ORDER),
        "physics_features_are_predictions_not_labels": True,
        "transmit_power_is_explicit_classifier_input": True,
        "transmit_power_feature": TX_POWER_FEATURE_NAME,
        "csi_amplitude_scaled_by_power_profile": False,
        "predicted_total_power_adjustment_scale_db": TOTAL_POWER_TARGET_SCALE_DB,
        "scratch_uses_labels_as_input": False,
        "elapsed_seconds": time.time() - start,
        "args": jsonable_args(args),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False, allow_nan=True))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
