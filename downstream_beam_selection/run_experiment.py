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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_beam_selection.beam_utils import NUM_BEAMS, normalized_beam_power  # noqa: E402
from downstream_beam_selection.data_io import (  # noqa: E402
    BeamCorpus,
    BeamDataset,
    collate_beam,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_beam_selection.pretrained import (  # noqa: E402
    LOS_K_FEATURE_ORDER,
    TX_PHYSICS_FEATURE_ORDER,
    build_pretrained_model,
    extract_beam_features,
    tx_physics_training_warnings,
)
from models.encoder import CSIEncoder  # noqa: E402


MODEL_NAMES = ("scratch", "full_embedding", "full_los_k", "full_tx_physics")
DISPLAY_NAMES = {
    "scratch": "Scratch",
    "full_embedding": "Full embedding",
    "full_los_k": "Full embedding + LoS/K",
    "full_tx_physics": "Full embedding + Tx physics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive 64-beam selection experiments")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"])
    parser.add_argument("--observed-beams", nargs="+", type=int, default=[16])
    parser.add_argument("--observation-strategy", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 8])
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--noise-power-dbw", type=float, default=-117.0)
    parser.add_argument("--skip-rate-metric", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument(
        "--allow-untrained-tx-physics",
        action="store_true",
        help="Allow Full+Tx Physics even when checkpoint metadata cannot verify AoD supervision.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BeamHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, NUM_BEAMS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ScratchBeamModel(nn.Module):
    def __init__(self, d_token: int, token_norm_mode: str, use_continuous: bool):
        super().__init__()
        self.encoder = CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous,
        )
        self.head = BeamHead(256)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.head(self.encoder(**encoder_kwargs(batch)))


def make_loader(
    corpus: BeamCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    observed_beams: int,
    *,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    dataset = BeamDataset(
        corpus,
        indices,
        observed_beams=observed_beams,
        observation_strategy=args.observation_strategy,
        observation_seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_beam,
        generator=generator,
    )


def balanced_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=NUM_BEAMS).float()
    weights = torch.ones_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def train_scratch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scratch_lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    best_state = None
    best_val = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda" and not args.no_amp,
            ):
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["target_class"], weight=class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * len(batch["target_class"])
            train_count += len(batch["target_class"])
        val_loss = evaluate_loader_loss(model, val_loader, device, class_weights)
        history.append(
            {"epoch": epoch, "train_loss": train_loss_sum / train_count, "val_loss": val_loss}
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Scratch training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "epochs_trained": len(history), "history": history}


@torch.inference_mode()
def evaluate_loader_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["target_class"], weight=class_weights)
        total += float(loss) * len(batch["target_class"])
        count += len(batch["target_class"])
    return total / max(count, 1)


@torch.inference_mode()
def scratch_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    values = []
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        values.append(model(batch).float().cpu())
    return torch.cat(values)


def feature_cache_path(args: argparse.Namespace, observed_beams: int) -> Path:
    return args.output_dir / (
        f"frozen_features_obs{observed_beams}_{args.observation_strategy}_seed{args.seed}.pt"
    )


@torch.inference_mode()
def build_feature_cache(
    corpus: BeamCorpus,
    pretrained_model,
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
    args: argparse.Namespace,
    observed_beams: int,
) -> dict[str, torch.Tensor]:
    path = feature_cache_path(args, observed_beams)
    if path.exists() and not args.rebuild_feature_cache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("sample_count") == len(corpus.samples)
            and payload.get("observed_beams") == observed_beams
            and payload.get("observation_strategy") == args.observation_strategy
            and payload.get("checkpoint") == str(args.checkpoint.resolve())
            and payload.get("data") == str(args.data.resolve())
        ):
            return payload
    loader = make_loader(
        corpus,
        torch.arange(len(corpus.samples)),
        args,
        observed_beams,
        shuffle=False,
        seed_offset=90,
    )
    device = torch.device(args.device)
    embeddings = []
    los_k = []
    tx_physics = []
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        embedding, batch_los_k, batch_tx = extract_beam_features(
            pretrained_model, batch, prototype_keys, checkpoint_args
        )
        embeddings.append(embedding.float().cpu())
        los_k.append(batch_los_k.float().cpu())
        tx_physics.append(batch_tx.float().cpu())
    payload = {
        "sample_count": len(corpus.samples),
        "observed_beams": observed_beams,
        "observation_strategy": args.observation_strategy,
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "embedding": torch.cat(embeddings),
        "los_k": torch.cat(los_k),
        "tx_physics": torch.cat(tx_physics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def frozen_input(cache: dict[str, torch.Tensor], model_name: str) -> torch.Tensor:
    embedding = cache["embedding"]
    if model_name == "full_embedding":
        return embedding
    if model_name == "full_los_k":
        return torch.cat([embedding, cache["los_k"]], dim=-1)
    if model_name == "full_tx_physics":
        return torch.cat([embedding, cache["tx_physics"]], dim=-1)
    raise ValueError(model_name)


def train_frozen_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
) -> tuple[BeamHead, dict[str, Any]]:
    device = torch.device(args.device)
    head = BeamHead(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    train_data = TensorDataset(features[train_indices], labels[train_indices])
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_state = None
    best_val = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_labels)
            count += len(batch_labels)
        head.eval()
        with torch.inference_mode():
            val_logits = head(features[val_indices].to(device))
            val_loss = float(
                F.cross_entropy(
                    val_logits,
                    labels[val_indices].to(device),
                    weight=class_weights,
                )
            )
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Frozen-head training did not produce a checkpoint")
    head.load_state_dict(best_state)
    return head, {"best_val_loss": best_val, "epochs_trained": len(history), "history": history}


@torch.inference_mode()
def frozen_logits(
    head: BeamHead, features: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    head.eval()
    outputs = []
    for start in range(0, len(indices), 2048):
        chunk = indices[start : start + 2048]
        outputs.append(head(features[chunk].to(device)).float().cpu())
    return torch.cat(outputs)


def test_beam_snr_db(
    corpus: BeamCorpus,
    test_indices: torch.Tensor,
    noise_power_dbw: float,
) -> torch.Tensor:
    values = [
        (
            10.0
            * torch.log10(
                normalized_beam_power(corpus.samples[int(index)].tokens).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
            )
            - noise_power_dbw
        ).to(torch.float16)
        for index in test_indices
    ]
    return torch.stack(values)


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q))


def evaluate_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gains_db: torch.Tensor,
    acceptable: torch.Tensor,
    top_k_values: list[int],
    observed_beams: int,
    beam_snr_db: torch.Tensor | None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    max_k = max(top_k_values)
    ranking = torch.topk(logits, k=max_k, dim=-1).indices
    oracle_gain = gains_db.gather(1, targets[:, None]).squeeze(1)
    sample_index = torch.arange(len(targets))
    oracle_rate = None
    if beam_snr_db is not None:
        oracle_snr = torch.pow(
            10.0, beam_snr_db[sample_index, targets].float() / 10.0
        )
        oracle_rate = torch.log2(1.0 + oracle_snr).mean(dim=-1)

    for k in top_k_values:
        candidates = ranking[:, :k]
        hits = (candidates == targets[:, None]).any(dim=1)
        result[f"top{k}_hit_rate"] = float(hits.float().mean())
        acceptable_hit = acceptable.gather(1, candidates).any(dim=1)
        result[f"top{k}_acceptable_1db_rate"] = float(acceptable_hit.float().mean())
        candidate_gains = gains_db.gather(1, candidates)
        best_position = candidate_gains.argmax(dim=1, keepdim=True)
        final_beam = candidates.gather(1, best_position).squeeze(1)
        final_gain = gains_db.gather(1, final_beam[:, None]).squeeze(1)
        loss = (oracle_gain - final_gain).clamp_min(0.0)
        result[f"top{k}_gain_loss_mean_db"] = float(loss.mean())
        result[f"top{k}_gain_loss_median_db"] = percentile(loss, 0.5)
        result[f"top{k}_gain_loss_p90_db"] = percentile(loss, 0.9)
        result[f"top{k}_gain_loss_p95_db"] = percentile(loss, 0.95)
        result[f"top{k}_gain_loss_le_1db"] = float((loss <= 1.0).float().mean())
        result[f"top{k}_gain_loss_le_3db"] = float((loss <= 3.0).float().mean())
        nominal_measurements = min(observed_beams + k, NUM_BEAMS)
        result[f"top{k}_nominal_measurements"] = float(nominal_measurements)
        result[f"top{k}_overhead_reduction_including_observation"] = (
            1.0 - nominal_measurements / NUM_BEAMS
        )
        result[f"top{k}_overhead_reduction_search_only"] = 1.0 - k / NUM_BEAMS
        if beam_snr_db is not None and oracle_rate is not None:
            final_snr = torch.pow(
                10.0, beam_snr_db[sample_index, final_beam].float() / 10.0
            )
            final_rate = torch.log2(1.0 + final_snr).mean(dim=-1)
            retention = final_rate / oracle_rate.clamp_min(1e-12)
            result[f"top{k}_rate_retention_mean"] = float(retention.mean())
            result[f"top{k}_rate_retention_median"] = percentile(retention, 0.5)
    return result


def save_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if any(value not in {8, 16, 32, 64} for value in args.observed_beams):
        raise ValueError("--observed-beams choices are 8, 16, 32, 64")
    if any(value <= 0 or value > NUM_BEAMS for value in args.top_k):
        raise ValueError("Every --top-k value must be in [1, 64]")
    model_names = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    corpus = BeamCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained_model, prototype_keys, model_info = build_pretrained_model(checkpoint, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    tx_warnings = tx_physics_training_warnings(checkpoint_args)
    for message in tx_warnings:
        warnings.warn(message)
    if (
        "full_tx_physics" in model_names
        and tx_warnings
        and not args.allow_untrained_tx_physics
    ):
        raise ValueError(
            "Full+Tx Physics requires a checkpoint with supervised AoD. "
            "Retrain the checkpoint or pass --allow-untrained-tx-physics only for debugging."
        )

    labels = corpus.oracle_beam
    train_indices = corpus.split_indices["train"]
    val_indices = corpus.split_indices["val"]
    test_indices = corpus.split_indices["test"]
    class_weights = None
    if args.class_weight == "balanced":
        class_weights = balanced_class_weights(labels[train_indices]).to(device)
    beam_snr_db = (
        None
        if args.skip_rate_metric
        else test_beam_snr_db(corpus, test_indices, args.noise_power_dbw)
    )
    result_rows: list[dict[str, Any]] = []
    started = time.time()

    for observed_beams in args.observed_beams:
        train_loader = make_loader(
            corpus, train_indices, args, observed_beams, shuffle=True, seed_offset=1
        )
        val_loader = make_loader(
            corpus, val_indices, args, observed_beams, shuffle=False, seed_offset=2
        )
        test_loader = make_loader(
            corpus, test_indices, args, observed_beams, shuffle=False, seed_offset=3
        )
        frozen_cache = None
        if any(name != "scratch" for name in model_names):
            frozen_cache = build_feature_cache(
                corpus,
                pretrained_model,
                prototype_keys,
                checkpoint_args,
                args,
                observed_beams,
            )

        for model_name in model_names:
            run_dir = args.output_dir / f"obs{observed_beams}" / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            if model_name == "scratch":
                model = ScratchBeamModel(
                    d_token=int(model_info["d_token"]),
                    token_norm_mode=str(model_info["token_norm_mode"]),
                    use_continuous=bool(model_info["use_continuous_config_encoding"]),
                )
                model, training = train_scratch(
                    model, train_loader, val_loader, args, class_weights
                )
                logits = scratch_logits(model, test_loader, device)
                torch.save(
                    {"model_state": model.state_dict(), "args": vars(args), "training": training},
                    run_dir / "best.pt",
                )
            else:
                assert frozen_cache is not None
                features = frozen_input(frozen_cache, model_name)
                head, training = train_frozen_head(
                    features,
                    labels,
                    train_indices,
                    val_indices,
                    args,
                    class_weights,
                )
                logits = frozen_logits(head, features, test_indices, device)
                torch.save(
                    {
                        "head_state": head.state_dict(),
                        "input_dim": int(features.shape[-1]),
                        "args": vars(args),
                        "training": training,
                    },
                    run_dir / "best.pt",
                )

            metrics = evaluate_predictions(
                logits,
                labels[test_indices],
                corpus.beam_gains_db[test_indices],
                corpus.acceptable_beams_1db[test_indices],
                sorted(set(args.top_k)),
                observed_beams,
                beam_snr_db,
            )
            row = {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "observed_beams": observed_beams,
                "observation_strategy": args.observation_strategy,
                "test_samples": len(test_indices),
                "best_val_loss": training["best_val_loss"],
                "epochs_trained": training["epochs_trained"],
                **metrics,
            }
            result_rows.append(row)
            (run_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            save_summary(result_rows, args.output_dir)
            print(
                f"model={model_name} observed_beams={observed_beams} "
                f"top1={metrics.get('top1_hit_rate', float('nan')):.4f} "
                f"top3={metrics.get('top3_hit_rate', float('nan')):.4f}"
            )

    run_manifest = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "models": model_names,
        "observed_beams": args.observed_beams,
        "top_k": args.top_k,
        "observation_strategy": args.observation_strategy,
        "los_k_feature_order": LOS_K_FEATURE_ORDER,
        "tx_physics_feature_order": TX_PHYSICS_FEATURE_ORDER,
        "noise_power_dbw": args.noise_power_dbw,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
