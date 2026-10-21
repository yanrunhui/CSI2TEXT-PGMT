from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
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

from downstream_localization.data_io import (  # noqa: E402
    LocalizationCorpus,
    LocalizationDataset,
    collate_localization,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_localization.pretrained import (  # noqa: E402
    build_pretrained_model,
    extract_full_features,
)


MODEL_NAMES = (
    "scratch",
    "scratch_physics",
    "pretrained_finetune",
    "full_embedding",
    "full_physics",
)
DISPLAY_NAMES = {
    "scratch": "From scratch",
    "scratch_physics": "Scratch + frozen physics",
    "pretrained_finetune": "Pretrained encoder fine-tuned",
    "full_embedding": "Full embedding",
    "full_physics": "Full embedding + physics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-model CSI localization experiment.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*MODEL_NAMES, "all"),
        default=["all"],
    )
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Ignore an existing frozen_features.pt and extract it again.",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional explicit frozen feature cache path shared across experiment runs.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CoordinateNormalizer:
    def __init__(self, targets: torch.Tensor):
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0, correction=0).clamp(min=1e-6)

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean.to(target.device)) / self.std.to(target.device)

    def denormalize(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.std.to(target.device) + self.mean.to(target.device)


class LocalizationHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ScratchLocalizationModel(nn.Module):
    def __init__(
        self,
        d_token: int,
        token_norm_mode: str,
        use_continuous: bool,
        physics_dim: int = 0,
    ):
        super().__init__()
        self.physics_dim = int(physics_dim)
        self.encoder = CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous,
        )
        self.head = LocalizationHead(256 + self.physics_dim)

    def forward(
        self,
        batch: dict[str, Any],
        frozen_physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder(**encoder_kwargs(batch))
        if self.physics_dim == 0:
            if frozen_physics is not None:
                raise ValueError("Plain Scratch model must not receive physics features")
            features = embedding
        else:
            if frozen_physics is None or frozen_physics.shape != (
                embedding.shape[0],
                self.physics_dim,
            ):
                raise ValueError(
                    f"Expected frozen physics [{embedding.shape[0]}, {self.physics_dim}]"
                )
            features = torch.cat([embedding, frozen_physics.detach()], dim=-1)
        return self.head(features)


def make_loader(
    corpus: LocalizationCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        LocalizationDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_localization,
        generator=generator,
    )


def localization_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(prediction.float() - target.float(), dim=-1).cpu().numpy()
    return {
        "mean_m": float(np.mean(error)),
        "median_m": float(np.median(error)),
        "p90_m": float(np.percentile(error, 90)),
        "within_0.1m_pct": float(100.0 * np.mean(error <= 0.1)),
        "within_0.25m_pct": float(100.0 * np.mean(error <= 0.25)),
        "within_0.5m_pct": float(100.0 * np.mean(error <= 0.5)),
        "within_1m_pct": float(100.0 * np.mean(error <= 1.0)),
        "within_5m_pct": float(100.0 * np.mean(error <= 5.0)),
        "within_10m_pct": float(100.0 * np.mean(error <= 10.0)),
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def csi_encoder_state_dict(pretrained_model: nn.Module) -> dict[str, torch.Tensor]:
    encoder = getattr(pretrained_model, "csi", None)
    if encoder is None:
        raise ValueError("Pretrained model does not expose a CSI encoder as .csi")
    return {
        key: value.detach().cpu().clone()
        for key, value in encoder.state_dict().items()
    }


def load_initial_encoder_state(
    model: ScratchLocalizationModel,
    initial_encoder_state: dict[str, torch.Tensor] | None,
) -> int:
    if initial_encoder_state is None:
        return 0
    current = model.encoder.state_dict()
    compatible = {
        key: value
        for key, value in initial_encoder_state.items()
        if key in current and current[key].shape == value.shape
    }
    if len(compatible) != len(current):
        missing = sorted(set(current) - set(compatible))
        skipped = sorted(set(initial_encoder_state) - set(compatible))
        raise ValueError(
            "Pretrained CSI encoder is incompatible with localization encoder; "
            f"loaded={len(compatible)} expected={len(current)} "
            f"missing={missing[:5]} skipped={skipped[:5]}"
        )
    model.encoder.load_state_dict(compatible, strict=True)
    return len(compatible)


@torch.inference_mode()
def evaluate_scratch(
    model: ScratchLocalizationModel,
    loader: DataLoader,
    normalizer: CoordinateNormalizer,
    device: torch.device,
    amp: bool,
    physics_features: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets, indices = [], [], []
    for batch in loader:
        frozen_physics = None
        if physics_features is not None:
            frozen_physics = physics_features[batch["indices"]].to(
                device, non_blocking=True
            )
        batch = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            normalized_prediction = model(batch, frozen_physics)
        predictions.append(normalizer.denormalize(normalized_prediction.float()).cpu())
        targets.append(batch["target_xy"].cpu())
        indices.append(batch["indices"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return localization_metrics(prediction, target), prediction, target, torch.cat(indices)


def train_scratch(
    name: str,
    corpus: LocalizationCorpus,
    normalizer: CoordinateNormalizer,
    model_info: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    physics_features: torch.Tensor | None = None,
    train_indices: torch.Tensor | None = None,
    initial_encoder_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_everything(args.seed)
    if physics_features is not None and physics_features.shape != (len(corpus.samples), 4):
        raise ValueError(
            f"Frozen physics must be [{len(corpus.samples)}, 4], "
            f"got {tuple(physics_features.shape)}"
        )
    model = ScratchLocalizationModel(
        model_info["d_token"],
        model_info["token_norm_mode"],
        model_info["use_continuous_config_encoding"],
        physics_dim=0 if physics_features is None else 4,
    ).to(device)
    initialized_encoder_tensors = load_initial_encoder_state(
        model, initial_encoder_state
    )
    if initialized_encoder_tensors:
        print(
            f"{name} initialized CSI encoder from checkpoint "
            f"({initialized_encoder_tensors} tensors)"
        )
    effective_train_indices = (
        corpus.split_indices["train"]
        if train_indices is None
        else torch.as_tensor(train_indices, dtype=torch.long)
    )
    if len(effective_train_indices) == 0:
        raise ValueError("The localization training subset is empty")
    allowed_train_indices = set(corpus.split_indices["train"].tolist())
    if not set(effective_train_indices.tolist()).issubset(allowed_train_indices):
        raise ValueError("train_indices must be a subset of the original training split")
    train_loader = make_loader(corpus, effective_train_indices, args, True, 11)
    val_loader = make_loader(corpus, corpus.split_indices["val"], args, False, 12)
    test_loader = make_loader(corpus, corpus.split_indices["test"], args, False, 13)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scratch_lr, weight_decay=args.weight_decay
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_state = None
    best_median = math.inf
    stale_epochs = 0
    history = []
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
            target = normalizer.normalize(batch["target_xy"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(batch, frozen_physics)
                loss = F.smooth_l1_loss(
                    prediction, target, reduction="none"
                ).sum(dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_metrics, _, _, _ = evaluate_scratch(
            model,
            val_loader,
            normalizer,
            device,
            amp,
            physics_features=physics_features,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(
            f"{name} epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_median={val_metrics['median_m']:.4f}m"
        )
        if val_metrics["median_m"] < best_median:
            best_median = val_metrics["median_m"]
            best_state = cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "target_mean": normalizer.mean,
            "target_std": normalizer.std,
            "best_val_median_m": best_median,
            "history": history,
            "uses_frozen_physics": physics_features is not None,
            "initial_encoder": (
                "pretrained" if initial_encoder_state is not None else "random"
            ),
            "fine_tunes_encoder": True,
            "physics_order": (
                ["p_los", "first_path_delay_ns/3000", "sin_aoa", "cos_aoa"]
                if physics_features is not None
                else []
            ),
            "train_indices": effective_train_indices.clone(),
            "train_sample_count": len(effective_train_indices),
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_scratch(
        model,
        test_loader,
        normalizer,
        device,
        amp,
        physics_features=physics_features,
    )


def checkpoint_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@torch.inference_mode()
def load_or_extract_features(
    corpus: LocalizationCorpus,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    pretrained_model: nn.Module,
    prototype_keys: list,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    explicit_cache = getattr(args, "feature_cache", None)
    cache_path = (
        Path(explicit_cache)
        if explicit_cache is not None
        else args.output_dir / "frozen_features.pt"
    )
    signature = checkpoint_signature(checkpoint_path)
    if cache_path.exists() and not args.rebuild_feature_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("checkpoint_signature") == signature and len(cached["embedding"]) == len(corpus.samples):
            print(f"using feature cache: {cache_path}")
            return cached["embedding"].float(), cached["physics"].float()

    loader = make_loader(corpus, torch.arange(len(corpus.samples)), args, False, 21)
    embeddings = torch.empty(len(corpus.samples), 256, dtype=torch.float32)
    physics = torch.empty(len(corpus.samples), 4, dtype=torch.float32)
    amp = device.type == "cuda" and not args.no_amp
    for step, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            embedding, physical = extract_full_features(
                pretrained_model, batch, prototype_keys, dict(checkpoint.get("args", {}))
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
            "checkpoint_signature": signature,
            "physics_order": ["p_los", "first_path_delay_ns/3000", "sin_aoa", "cos_aoa"],
        },
        cache_path,
    )
    return embeddings, physics


@torch.inference_mode()
def evaluate_head(
    head: LocalizationHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    normalizer: CoordinateNormalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    head.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        normalized = head(features[batch_indices].to(device)).float().cpu()
        predictions.append(normalizer.denormalize(normalized))
    prediction = torch.cat(predictions)
    target = targets[indices]
    return localization_metrics(prediction, target), prediction, target, indices.clone()


def train_frozen_head(
    name: str,
    features: torch.Tensor,
    corpus: LocalizationCorpus,
    normalizer: CoordinateNormalizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_everything(args.seed)
    head = LocalizationHead(features.shape[1]).to(device)
    train_idx = corpus.split_indices["train"]
    val_idx = corpus.split_indices["val"]
    test_idx = corpus.split_indices["test"]
    normalized_targets = normalizer.normalize(corpus.relative_xy_m)
    train_dataset = TensorDataset(features[train_idx], normalized_targets[train_idx])
    generator = torch.Generator().manual_seed(args.seed + 31)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    best_state = None
    best_median = math.inf
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(
                head(feature_batch), target_batch, reduction="none"
            ).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_metrics, _, _, _ = evaluate_head(
            head,
            features,
            corpus.relative_xy_m,
            val_idx,
            normalizer,
            device,
            args.batch_size,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(
            f"{name} epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_median={val_metrics['median_m']:.4f}m"
        )
        if val_metrics["median_m"] < best_median:
            best_median = val_metrics["median_m"]
            best_state = cpu_state_dict(head)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    head.load_state_dict(best_state)
    torch.save(
        {
            "head_state": best_state,
            "input_dim": features.shape[1],
            "target_mean": normalizer.mean,
            "target_std": normalizer.std,
            "best_val_median_m": best_median,
            "history": history,
            "args": vars(args),
        },
        args.output_dir / f"{name}_best.pt",
    )
    return evaluate_head(
        head,
        features,
        corpus.relative_xy_m,
        test_idx,
        normalizer,
        device,
        args.batch_size,
    )


def save_predictions(
    name: str,
    result: tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor],
    corpus: LocalizationCorpus,
    output_dir: Path,
) -> None:
    _, prediction, target, indices = result
    tx_xy = corpus.tx_xyz_m[indices, :2]
    predicted_rx = tx_xy + prediction
    true_rx = corpus.rx_xyz_m[indices, :2]
    error = torch.linalg.vector_norm(predicted_rx - true_rx, dim=-1)
    np.savez_compressed(
        output_dir / f"{name}_test_predictions.npz",
        indices=indices.numpy(),
        predicted_relative_xy_m=prediction.numpy(),
        true_relative_xy_m=target.numpy(),
        predicted_rx_xy_m=predicted_rx.numpy(),
        true_rx_xy_m=true_rx.numpy(),
        error_m=error.numpy(),
    )


def plot_cdf(results: dict[str, tuple], output: Path) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for name, (_, prediction, target, _) in results.items():
        error = torch.linalg.vector_norm(prediction - target, dim=-1).numpy()
        sorted_error = np.sort(error)
        cdf = np.arange(1, len(error) + 1) / len(error)
        plt.plot(sorted_error, cdf, linewidth=2.0, label=DISPLAY_NAMES[name])
    plt.xlabel("2D localization error (m)")
    plt.ylabel("Empirical CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, epochs, and patience must be positive")
    selected = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)
    corpus = LocalizationCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained_model, prototype_keys, model_info = build_pretrained_model(checkpoint, device)

    first_shape = corpus.samples[0].tokens.shape
    if first_shape[1] != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={first_shape[1]} but checkpoint expects {model_info['d_token']}"
        )
    train_targets = corpus.relative_xy_m[corpus.split_indices["train"]]
    normalizer = CoordinateNormalizer(train_targets)
    print(f"device={device} samples={len(corpus.samples)} models={','.join(selected)}")
    print(f"target_mean_m={normalizer.mean.tolist()} target_std_m={normalizer.std.tolist()}")
    start_time = time.time()
    results: dict[str, tuple] = {}

    if "scratch" in selected:
        results["scratch"] = train_scratch(
            "scratch", corpus, normalizer, model_info, args, device
        )

    if "pretrained_finetune" in selected:
        results["pretrained_finetune"] = train_scratch(
            "pretrained_finetune",
            corpus,
            normalizer,
            model_info,
            args,
            device,
            initial_encoder_state=csi_encoder_state_dict(pretrained_model),
        )

    if any(
        name in selected
        for name in ("scratch_physics", "full_embedding", "full_physics")
    ):
        embedding, physics = load_or_extract_features(
            corpus,
            checkpoint,
            args.checkpoint,
            pretrained_model,
            prototype_keys,
            args,
            device,
        )
        if "scratch_physics" in selected:
            results["scratch_physics"] = train_scratch(
                "scratch_physics",
                corpus,
                normalizer,
                model_info,
                args,
                device,
                physics_features=physics,
            )
        if "full_embedding" in selected:
            results["full_embedding"] = train_frozen_head(
                "full_embedding", embedding, corpus, normalizer, args, device
            )
        if "full_physics" in selected:
            results["full_physics"] = train_frozen_head(
                "full_physics",
                torch.cat([embedding, physics], dim=-1),
                corpus,
                normalizer,
                args,
                device,
            )

    metric_rows = []
    for name, result in results.items():
        save_predictions(name, result, corpus, args.output_dir)
        metric_rows.append({"model": name, "display_name": DISPLAY_NAMES[name], **result[0]})
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {
        "metrics": {row["model"]: {k: v for k, v in row.items() if k not in ("model", "display_name")} for row in metric_rows},
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split_sizes": {name: len(indices) for name, indices in corpus.split_indices.items()},
        "target_mean_m": normalizer.mean.tolist(),
        "target_std_m": normalizer.std.tolist(),
        "elapsed_seconds": time.time() - start_time,
        "seed": args.seed,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_cdf(results, args.output_dir / "localization_error_cdf.png")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
