from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.encoder import CSIEncoder  # noqa: E402

from downstream_cp_selection.data_io import (  # noqa: E402
    CPCorpus,
    CPDataset,
    collate_cp,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_cp_selection.pretrained import (  # noqa: E402
    PHYSICS_FEATURE_ORDER,
    build_pretrained_model,
    extract_full_cp_features,
    physics_training_warnings,
)


MODEL_NAMES = (
    "scratch",
    "scratch_physics",
    "full_embedding",
    "full_physics",
    "full_physics_finetune",
)
DISPLAY_NAMES = {
    "scratch": "From scratch",
    "scratch_physics": "Scratch + frozen physics",
    "full_embedding": "Full embedding",
    "full_physics": "Full embedding + physics",
    "full_physics_finetune": "Full embedding + physics + partial fine-tuning",
}
DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DATA = DEFAULT_REPO / "downstream_cp_selection/artifacts/d2los_50k_cp_labels.pt"
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_cp_selection/outputs/first_run"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "artifacts/pretrain_d2los_100k_upa8x8_nf128_los50k_nlos50k_reflection_path/checkpoint_last.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate adaptive-CP selection models."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Full CSI2TEXT-PGMT checkpoint; also defines the fair Scratch architecture.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"]
    )
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument(
        "--safety-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of expected CP-outside-energy excess above the label threshold. "
            "Applied to full_physics_finetune; zero reproduces ordinary CE training."
        ),
    )
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=2,
        help="How many final pretrained Transformer layers to fine-tune.",
    )
    parser.add_argument(
        "--unfreeze-all-encoder",
        action="store_true",
        help=(
            "Fine-tune every CSI encoder parameter, including input projection, "
            "metadata/position encodings, all Transformer layers, and output projection."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Use inverse-frequency weights computed only from the training split.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional shared location for frozen Full features.",
    )
    parser.add_argument(
        "--exclude-reflection-path-count",
        action="store_true",
        help=(
            "Remove predicted reflection-path count from Full+Physics classifier "
            "inputs while keeping the same frozen feature cache."
        ),
    )
    parser.add_argument(
        "--exclude-delay-spread",
        action="store_true",
        help=(
            "Remove predicted delay spread from Full+Physics classifier inputs "
            "while keeping the same frozen feature cache."
        ),
    )
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ScratchCPModel(nn.Module):
    def __init__(
        self,
        d_token: int,
        token_norm_mode: str,
        use_continuous: bool,
        num_classes: int,
        physics_dim: int = 0,
    ):
        super().__init__()
        if physics_dim < 0:
            raise ValueError("physics_dim must be non-negative")
        self.physics_dim = int(physics_dim)
        self.encoder = CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous,
        )
        self.head = CPHead(256 + self.physics_dim, num_classes)

    def forward(
        self,
        batch: dict[str, Any],
        frozen_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder(**encoder_kwargs(batch))
        if self.physics_dim == 0:
            if frozen_physics is not None:
                raise ValueError("Plain Scratch model must not receive frozen physics")
            features = embedding
        else:
            expected_shape = (embedding.shape[0], self.physics_dim)
            if frozen_physics is None or tuple(frozen_physics.shape) != expected_shape:
                raise ValueError(
                    f"Expected frozen physics with shape {expected_shape}, got "
                    f"{None if frozen_physics is None else tuple(frozen_physics.shape)}"
                )
            features = torch.cat([embedding, frozen_physics.detach()], dim=-1)
        return self.head(features)


class FullPhysicsFineTuneCPModel(nn.Module):
    """Pretrained CSI encoder with a frozen-teacher physics side input."""

    def __init__(
        self,
        pretrained_encoder: CSIEncoder,
        num_classes: int,
        physics_dim: int,
        unfreeze_last_n_layers: int,
        unfreeze_all_encoder: bool,
    ):
        super().__init__()
        self.encoder = copy.deepcopy(pretrained_encoder)
        if physics_dim <= 0:
            raise ValueError("physics_dim must be positive")
        self.physics_dim = int(physics_dim)
        self.head = CPHead(256 + self.physics_dim, num_classes)

        layers = self.encoder.transformer.layers
        if not 0 <= unfreeze_last_n_layers <= len(layers):
            raise ValueError(
                f"unfreeze_last_n_layers must be in [0, {len(layers)}], "
                f"got {unfreeze_last_n_layers}"
            )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if unfreeze_all_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        elif unfreeze_last_n_layers:
            for layer in layers[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        # Always adapt the pretrained output normalization/projection to the CP task.
        for module in (self.encoder.final_norm, self.encoder.proj):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        batch: dict[str, Any],
        frozen_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frozen_physics is None:
            raise ValueError("Fine-tuned Full+Physics requires frozen teacher physics")
        if frozen_physics.shape[-1] != self.physics_dim:
            raise ValueError(
                f"Expected {self.physics_dim} frozen physics features, "
                f"got {frozen_physics.shape[-1]}"
            )
        embedding = F.normalize(self.encoder(**encoder_kwargs(batch)), dim=-1)
        return self.head(torch.cat([embedding, frozen_physics.detach()], dim=-1))


def make_loader(
    corpus: CPCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        CPDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_cp,
        generator=generator,
    )


def class_weights(corpus: CPCorpus, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    labels = corpus.labels[corpus.split_indices["train"]]
    counts = torch.bincount(labels, minlength=len(corpus.cp_lengths_ns)).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = labels.numel() / (present.sum() * counts[present])
    print(f"training_class_counts={counts.long().tolist()}")
    print(f"cross_entropy_class_weights={weights.tolist()}")
    return weights


def classification_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    corpus: CPCorpus,
) -> dict[str, float]:
    prediction = prediction.long().cpu()
    target = target.long().cpu()
    indices = indices.long().cpu()
    cp = corpus.cp_lengths_ns.float()
    predicted_cp = cp[prediction]
    target_cp = cp[target]
    selected_outside = corpus.outside_energy_ratios[indices, prediction]
    threshold = corpus.max_outside_energy_ratio
    return {
        "accuracy_pct": float(100.0 * (prediction == target).float().mean()),
        "underprediction_rate_pct": float(100.0 * (prediction < target).float().mean()),
        "overprediction_rate_pct": float(100.0 * (prediction > target).float().mean()),
        "mean_predicted_cp_ns": float(predicted_cp.mean()),
        "mean_oracle_cp_ns": float(target_cp.mean()),
        "mean_cp_outside_energy_pct": float(100.0 * selected_outside.mean()),
        "p95_cp_outside_energy_pct": float(
            100.0 * torch.quantile(selected_outside.float(), 0.95)
        ),
        "unsafe_outside_energy_rate_pct": float(
            100.0 * (selected_outside > threshold + 1e-7).float().mean()
        ),
    }


def confusion_matrix(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    flat = target.long().cpu() * num_classes + prediction.long().cpu()
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def summed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    losses = F.cross_entropy(logits, target, weight=weight, reduction="none")
    denominator = (
        float(weight[target].sum()) if weight is not None else float(target.numel())
    )
    return losses.sum(), denominator


def expected_safety_penalty(
    logits: torch.Tensor,
    outside_energy_ratios: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Expected amount by which a probabilistically selected CP is unsafe."""

    probabilities = torch.softmax(logits.float(), dim=-1)
    unsafe_excess = torch.relu(outside_energy_ratios.float() - float(threshold))
    if probabilities.shape != unsafe_excess.shape:
        raise ValueError(
            "Logits and outside_energy_ratios must have the same [batch, class] shape"
        )
    return (probabilities * unsafe_excess).sum(dim=-1).mean()


@torch.inference_mode()
def evaluate_scratch(
    model: nn.Module,
    loader: DataLoader,
    corpus: CPCorpus,
    device: torch.device,
    amp: bool,
    weight: torch.Tensor | None,
    physics_features: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_all, targets, indices = [], [], []
    total_loss = 0.0
    denominator = 0.0
    device_weight = weight.to(device) if weight is not None else None
    for batch in loader:
        frozen_physics = None
        if physics_features is not None:
            frozen_physics = physics_features[batch["indices"]].to(
                device, non_blocking=True
            )
        batch = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(batch, frozen_physics)
            loss, batch_denominator = summed_cross_entropy(
                logits, batch["target_class"], device_weight
            )
        total_loss += float(loss)
        denominator += batch_denominator
        logits_all.append(logits.float().cpu())
        targets.append(batch["target_class"].cpu())
        indices.append(batch["indices"].cpu())
    logits = torch.cat(logits_all)
    target = torch.cat(targets)
    index = torch.cat(indices)
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    safety_loss = expected_safety_penalty(
        logits,
        corpus.outside_energy_ratios[index],
        corpus.max_outside_energy_ratio,
    )
    return {
        "loss": total_loss / max(denominator, 1e-12),
        "safety_loss": float(safety_loss),
        "metrics": classification_metrics(prediction, target, index, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target,
        "indices": index,
    }


def train_scratch(
    corpus: CPCorpus,
    model_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    weight: torch.Tensor | None,
    *,
    name: str = "scratch",
    physics_features: torch.Tensor | None = None,
    physics_feature_order: list[str] | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    physics_feature_order = list(physics_feature_order or [])
    if physics_features is None:
        if physics_feature_order:
            raise ValueError("physics_feature_order was provided without physics features")
        physics_dim = 0
    else:
        expected_shape = (len(corpus.samples), len(physics_feature_order))
        if len(physics_feature_order) == 0:
            raise ValueError("Scratch+Physics requires at least one physics feature")
        if tuple(physics_features.shape) != expected_shape:
            raise ValueError(
                f"Frozen physics must have shape {expected_shape}, "
                f"got {tuple(physics_features.shape)}"
            )
        physics_dim = len(physics_feature_order)
    model = ScratchCPModel(
        d_token=model_info["d_token"],
        token_norm_mode=model_info["token_norm_mode"],
        use_continuous=model_info["use_continuous_config_encoding"],
        num_classes=len(corpus.cp_lengths_ns),
        physics_dim=physics_dim,
    ).to(device)
    train_loader = make_loader(corpus, corpus.split_indices["train"], args, True, 11)
    val_loader = make_loader(corpus, corpus.split_indices["val"], args, False, 12)
    test_loader = make_loader(corpus, corpus.split_indices["test"], args, False, 13)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scratch_lr, weight_decay=args.weight_decay
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    device_weight = weight.to(device) if weight is not None else None
    best_state = None
    best_val_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            frozen_physics = None
            if physics_features is not None:
                frozen_physics = physics_features[batch["indices"]].to(
                    device, non_blocking=True
                )
            batch = move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = F.cross_entropy(
                    model(batch, frozen_physics),
                    batch["target_class"],
                    weight=device_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_result = evaluate_scratch(
            model,
            val_loader,
            corpus,
            device,
            amp,
            weight,
            physics_features=physics_features,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_result["loss"]),
            **val_result["metrics"],
        }
        history.append(row)
        print(
            f"{name} epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['accuracy_pct']:.2f}% "
            f"val_under={row['underprediction_rate_pct']:.2f}%"
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "best_val_loss": best_val_loss,
            "history": history,
            "cp_lengths_ns": corpus.cp_lengths_ns,
            "class_weights": weight,
            "uses_frozen_physics": physics_features is not None,
            "physics_order": physics_feature_order,
            "physics_source": (
                "frozen_pretrained_teacher_predictions"
                if physics_features is not None
                else None
            ),
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_scratch(
        model,
        test_loader,
        corpus,
        device,
        amp,
        weight,
        physics_features=physics_features,
    )


def train_full_physics_finetune(
    corpus: CPCorpus,
    pretrained_encoder: CSIEncoder,
    physics_features: torch.Tensor,
    physics_feature_order: list[str],
    args: argparse.Namespace,
    device: torch.device,
    weight: torch.Tensor | None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    if tuple(physics_features.shape) != (
        len(corpus.samples),
        len(physics_feature_order),
    ):
        raise ValueError("Frozen physics feature cache has an unexpected shape")
    model = FullPhysicsFineTuneCPModel(
        pretrained_encoder=pretrained_encoder,
        num_classes=len(corpus.cp_lengths_ns),
        physics_dim=len(physics_feature_order),
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        unfreeze_all_encoder=args.unfreeze_all_encoder,
    ).to(device)
    trainable_encoder = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if not trainable_encoder:
        raise RuntimeError("No pretrained encoder parameters were selected for fine-tuning")
    train_loader = make_loader(corpus, corpus.split_indices["train"], args, True, 41)
    val_loader = make_loader(corpus, corpus.split_indices["val"], args, False, 42)
    test_loader = make_loader(corpus, corpus.split_indices["test"], args, False, 43)
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_encoder, "lr": args.finetune_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    device_weight = weight.to(device) if weight is not None else None
    best_state = None
    best_val_objective = math.inf
    best_val_components: dict[str, float] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(
        "full_physics_finetune trainable_parameters="
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)} "
        f"unfreeze_last_n_layers={args.unfreeze_last_n_layers} "
        f"unfreeze_all_encoder={args.unfreeze_all_encoder} "
        f"physics_features={','.join(physics_feature_order)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, ce_losses, safety_losses = [], [], []
        for batch in train_loader:
            frozen_physics = physics_features[batch["indices"]].to(
                device, non_blocking=True
            )
            batch = move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(batch, frozen_physics)
                ce_loss = F.cross_entropy(
                    logits, batch["target_class"], weight=device_weight
                )
                safety_loss = expected_safety_penalty(
                    logits,
                    batch["outside_energy_ratios"],
                    corpus.max_outside_energy_ratio,
                )
                loss = ce_loss + args.safety_loss_weight * safety_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            ce_losses.append(float(ce_loss.detach()))
            safety_losses.append(float(safety_loss.detach()))
        val_result = evaluate_scratch(
            model,
            val_loader,
            corpus,
            device,
            amp,
            weight,
            physics_features=physics_features,
        )
        val_objective = float(
            val_result["loss"]
            + args.safety_loss_weight * val_result["safety_loss"]
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "train_ce_loss": float(np.mean(ce_losses)),
            "train_safety_loss": float(np.mean(safety_losses)),
            "val_ce_loss": float(val_result["loss"]),
            "val_safety_loss": float(val_result["safety_loss"]),
            "val_objective": val_objective,
            **val_result["metrics"],
        }
        history.append(row)
        print(
            f"full_physics_finetune epoch={epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"val_objective={row['val_objective']:.6f} "
            f"val_safety={row['val_safety_loss']:.6f} "
            f"val_acc={row['accuracy_pct']:.2f}% "
            f"val_under={row['underprediction_rate_pct']:.2f}%"
        )
        if row["val_objective"] < best_val_objective:
            best_val_objective = row["val_objective"]
            best_val_components = {
                "ce_loss": row["val_ce_loss"],
                "safety_loss": row["val_safety_loss"],
            }
            best_state = cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Full+Physics fine-tuning did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "best_val_objective": best_val_objective,
            "best_val_components": best_val_components,
            "history": history,
            "cp_lengths_ns": corpus.cp_lengths_ns,
            "class_weights": weight,
            "physics_order": list(physics_feature_order),
            "physics_source": "frozen_pretrained_teacher_predictions",
            "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
            "unfreeze_all_encoder": args.unfreeze_all_encoder,
            "safety_loss_weight": args.safety_loss_weight,
            "trainable_parameter_names": trainable_names,
            "args": vars(args),
        },
        args.output_dir / "full_physics_finetune_best.pt",
    )
    return evaluate_scratch(
        model,
        test_loader,
        corpus,
        device,
        amp,
        weight,
        physics_features=physics_features,
    )


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@torch.inference_mode()
def load_or_extract_features(
    corpus: CPCorpus,
    checkpoint: dict[str, Any],
    pretrained_model: nn.Module,
    prototype_keys: list,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = args.feature_cache or args.output_dir / "frozen_features.pt"
    expected = {
        "checkpoint": file_signature(args.checkpoint),
        "data": file_signature(args.data),
        "physics_order": list(PHYSICS_FEATURE_ORDER),
    }
    if cache_path.exists() and not args.rebuild_feature_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == expected:
            embedding = cached.get("embedding")
            physics = cached.get("physics")
            if tuple(embedding.shape) == (len(corpus.samples), 256) and tuple(
                physics.shape
            ) == (len(corpus.samples), len(PHYSICS_FEATURE_ORDER)):
                print(f"using feature cache: {cache_path}")
                return embedding.float(), physics.float()

    loader = make_loader(corpus, torch.arange(len(corpus.samples)), args, False, 21)
    embeddings = torch.empty(len(corpus.samples), 256, dtype=torch.float32)
    physics = torch.empty(
        len(corpus.samples), len(PHYSICS_FEATURE_ORDER), dtype=torch.float32
    )
    amp = device.type == "cuda" and not args.no_amp
    checkpoint_args = dict(checkpoint.get("args", {}))
    for step, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            embedding, physical = extract_full_cp_features(
                pretrained_model, batch, prototype_keys, checkpoint_args
            )
        index = batch["indices"].cpu()
        embeddings[index] = embedding.float().cpu()
        physics[index] = physical.float().cpu()
        if step % 100 == 0 or step == len(loader):
            print(f"feature batches={step}/{len(loader)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": embeddings,
            "physics": physics,
            "signature": expected,
            "physics_order": list(PHYSICS_FEATURE_ORDER),
            "contains_oracle_physics": False,
        },
        cache_path,
    )
    return embeddings, physics


@torch.inference_mode()
def evaluate_head(
    head: CPHead,
    features: torch.Tensor,
    indices: torch.Tensor,
    corpus: CPCorpus,
    device: torch.device,
    batch_size: int,
    weight: torch.Tensor | None,
) -> dict[str, Any]:
    head.eval()
    logits_all = []
    total_loss = 0.0
    denominator = 0.0
    device_weight = weight.to(device) if weight is not None else None
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        logits = head(features[batch_indices].to(device)).float()
        target = corpus.labels[batch_indices].to(device)
        loss, batch_denominator = summed_cross_entropy(
            logits, target, device_weight
        )
        total_loss += float(loss)
        denominator += batch_denominator
        logits_all.append(logits.cpu())
    logits = torch.cat(logits_all)
    target = corpus.labels[indices]
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    return {
        "loss": total_loss / max(denominator, 1e-12),
        "metrics": classification_metrics(prediction, target, indices, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target.clone(),
        "indices": indices.clone(),
    }


def train_frozen_head(
    name: str,
    features: torch.Tensor,
    corpus: CPCorpus,
    args: argparse.Namespace,
    device: torch.device,
    weight: torch.Tensor | None,
    physics_feature_order: list[str] | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    head = CPHead(features.shape[1], len(corpus.cp_lengths_ns)).to(device)
    train_idx = corpus.split_indices["train"]
    val_idx = corpus.split_indices["val"]
    test_idx = corpus.split_indices["test"]
    train_dataset = TensorDataset(features[train_idx], corpus.labels[train_idx])
    generator = torch.Generator().manual_seed(args.seed + 31)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    device_weight = weight.to(device) if weight is not None else None
    best_state = None
    best_val_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                head(feature_batch), target_batch, weight=device_weight
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_result = evaluate_head(
            head, features, val_idx, corpus, device, args.batch_size, weight
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_result["loss"]),
            **val_result["metrics"],
        }
        history.append(row)
        print(
            f"{name} epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['accuracy_pct']:.2f}% "
            f"val_under={row['underprediction_rate_pct']:.2f}%"
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = cpu_state_dict(head)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} training did not produce a checkpoint")
    head.load_state_dict(best_state)
    torch.save(
        {
            "head_state": best_state,
            "input_dim": features.shape[1],
            "best_val_loss": best_val_loss,
            "history": history,
            "cp_lengths_ns": corpus.cp_lengths_ns,
            "class_weights": weight,
            "physics_order": list(physics_feature_order or []),
            "contains_oracle_physics": False,
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_head(
        head, features, test_idx, corpus, device, args.batch_size, weight
    )


def save_predictions(
    name: str, result: dict[str, Any], corpus: CPCorpus, output_dir: Path
) -> None:
    indices = result["indices"]
    prediction = result["prediction"]
    target = result["target"]
    selected_outside = corpus.outside_energy_ratios[indices, prediction]
    np.savez_compressed(
        output_dir / f"{name}_test_predictions.npz",
        indices=indices.numpy(),
        group_ids=np.asarray([corpus.group_ids[int(i)] for i in indices]),
        predicted_class=prediction.numpy(),
        target_class=target.numpy(),
        predicted_cp_ns=corpus.cp_lengths_ns[prediction].numpy(),
        target_cp_ns=corpus.cp_lengths_ns[target].numpy(),
        predicted_probabilities=result["probability"].numpy(),
        selected_outside_energy_ratio=selected_outside.numpy(),
        all_cp_outside_energy_ratios=corpus.outside_energy_ratios[indices].numpy(),
    )


def save_confusion(
    name: str, result: dict[str, Any], corpus: CPCorpus, output_dir: Path
) -> None:
    matrix = confusion_matrix(
        result["prediction"], result["target"], len(corpus.cp_lengths_ns)
    )
    labels = [f"{value:g} ns" for value in corpus.cp_lengths_ns.tolist()]
    with (output_dir / f"{name}_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])

    figure, axis = plt.subplots(figsize=(6.5, 5.6))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted CP")
    axis.set_ylabel("True optimal CP")
    axis.set_title(DISPLAY_NAMES[name])
    limit = max(int(matrix.max()), 1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            axis.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > limit / 2 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=220)
    plt.close(figure)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch-size, epochs, and patience must be positive")
    if args.safety_loss_weight < 0.0:
        raise ValueError("safety-loss-weight must be non-negative")
    selected = (
        list(MODEL_NAMES)
        if "all" in args.models
        else list(dict.fromkeys(args.models))
    )
    active_physics_order = list(PHYSICS_FEATURE_ORDER)
    if args.exclude_delay_spread:
        delay_name = "predicted_delay_spread_ns/200"
        if delay_name not in active_physics_order:
            raise ValueError(f"Cannot exclude missing physics feature {delay_name}")
        active_physics_order.remove(delay_name)
    if args.exclude_reflection_path_count:
        reflection_name = "predicted_reflection_path_count/10"
        if reflection_name not in active_physics_order:
            raise ValueError(f"Cannot exclude missing physics feature {reflection_name}")
        active_physics_order.remove(reflection_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)
    corpus = CPCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained_model, prototype_keys, model_info = build_pretrained_model(
        checkpoint, device
    )
    first_shape = corpus.samples[0].tokens.shape
    if first_shape[1] != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={first_shape[1]} but checkpoint expects "
            f"{model_info['d_token']}"
        )
    weight = class_weights(corpus, args.class_weight)
    checkpoint_warnings = (
        physics_training_warnings(dict(checkpoint.get("args", {})))
        if any(
            name in selected
            for name in (
                "scratch_physics",
                "full_physics",
                "full_physics_finetune",
            )
        )
        else []
    )
    if args.exclude_reflection_path_count:
        checkpoint_warnings = [
            message for message in checkpoint_warnings if "reflection-path-count" not in message
        ]
    if args.exclude_delay_spread:
        checkpoint_warnings = [
            message for message in checkpoint_warnings if "delay-spread" not in message
        ]
    for message in checkpoint_warnings:
        warnings.warn(
            message
            + "; models using frozen predicted physics must be interpreted cautiously"
        )

    print(
        f"device={device} samples={len(corpus.samples)} "
        f"splits={{{', '.join(f'{k}:{len(v)}' for k, v in corpus.split_indices.items())}}} "
        f"models={','.join(selected)}"
    )
    print(f"cp_lengths_ns={corpus.cp_lengths_ns.tolist()}")
    start_time = time.time()
    results: dict[str, dict[str, Any]] = {}

    if "scratch" in selected:
        results["scratch"] = train_scratch(
            corpus, model_info, args, device, weight, name="scratch"
        )

    if any(
        name in selected
        for name in (
            "scratch_physics",
            "full_embedding",
            "full_physics",
            "full_physics_finetune",
        )
    ):
        embedding, physics = load_or_extract_features(
            corpus,
            checkpoint,
            pretrained_model,
            prototype_keys,
            args,
            device,
        )
        active_indices = [
            list(PHYSICS_FEATURE_ORDER).index(name) for name in active_physics_order
        ]
        active_physics = physics[:, active_indices]
        print(f"active_physics_features={','.join(active_physics_order)}")
        # Frozen features are now cached in CPU tensors.  Release the large
        # teacher model from GPU before training heads/the partially unfrozen student.
        pretrained_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if "full_embedding" in selected:
            results["full_embedding"] = train_frozen_head(
                "full_embedding", embedding, corpus, args, device, weight
            )
        if "scratch_physics" in selected:
            results["scratch_physics"] = train_scratch(
                corpus,
                model_info,
                args,
                device,
                weight,
                name="scratch_physics",
                physics_features=active_physics,
                physics_feature_order=active_physics_order,
            )
        if "full_physics" in selected:
            results["full_physics"] = train_frozen_head(
                "full_physics",
                torch.cat([embedding, active_physics], dim=-1),
                corpus,
                args,
                device,
                weight,
                physics_feature_order=active_physics_order,
            )
        if "full_physics_finetune" in selected:
            results["full_physics_finetune"] = train_full_physics_finetune(
                corpus,
                pretrained_model.csi,
                active_physics,
                active_physics_order,
                args,
                device,
                weight,
            )

    metric_rows = []
    for name, result in results.items():
        save_predictions(name, result, corpus, args.output_dir)
        save_confusion(name, result, corpus, args.output_dir)
        metric_rows.append(
            {"model": name, "display_name": DISPLAY_NAMES[name], **result["metrics"]}
        )
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "metrics": {
            row["model"]: {
                key: value
                for key, value in row.items()
                if key not in ("model", "display_name")
            }
            for row in metric_rows
        },
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split_sizes": {
            name: len(indices) for name, indices in corpus.split_indices.items()
        },
        "cp_lengths_ns": corpus.cp_lengths_ns.tolist(),
        "max_outside_energy_ratio": corpus.max_outside_energy_ratio,
        "physics_feature_order": active_physics_order,
        "physics_features_are_predictions_not_labels": True,
        "checkpoint_warnings": checkpoint_warnings,
        "elapsed_seconds": time.time() - start_time,
        "args": jsonable_args(args),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
