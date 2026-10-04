"""Full-CSI 64-class best-transmit-beam prediction (no sparse probing)."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
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

from downstream_beam_selection.beam_utils import NUM_BEAMS  # noqa: E402
from downstream_beam_selection.data_io import (  # noqa: E402
    BeamCorpus,
    FullCSIBeamDataset,
    collate_full_csi,
    collate_full_csi_with_physics,
    encoder_kwargs,
    move_tensor_batch,
)
from downstream_cp_selection.pretrained import build_pretrained_model  # noqa: E402
from data.dataset import PHYSICS_TARGET_NAMES, physics_raw_values  # noqa: E402
from models.encoder import CSIEncoder  # noqa: E402


MODEL_NAMES = ("scratch", "frozen", "fine_tune", "scratch_physics", "fine_tune_physics")
PHYSICS_MODEL_NAMES = frozenset(("scratch_physics", "fine_tune_physics"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"])
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--physics-weight", type=float, default=0.1,
        help="Shared auxiliary-loss weight for both Physics models.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--require-clean-pretrain",
        action="store_true",
        help="Stop unless checkpoint and all recorded ancestors are verified disjoint from val/test.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def audit_pretraining_sources(label_payload: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Inspect every recorded pretraining source, including transfer ancestors."""
    groups = [str(value) for value in label_payload["group_ids"]]
    splits = label_payload["split_indices"]
    val_groups = {groups[int(index)] for index in splits["val"]}
    test_groups = {groups[int(index)] for index in splits["test"]}
    train_source_raw = label_payload.get("metadata", {}).get("pretraining_train_source")
    train_source = resolve_recorded_path(train_source_raw) if train_source_raw else None
    checked: list[dict[str, Any]] = []
    unknown: list[str] = []
    seen_checkpoints: set[Path] = set()
    overlap_val: set[str] = set()
    overlap_test: set[str] = set()
    current = checkpoint_path.resolve()
    while True:
        if current in seen_checkpoints:
            unknown.append(f"checkpoint ancestry cycle: {current}")
            break
        seen_checkpoints.add(current)
        if not current.is_file():
            unknown.append(f"checkpoint not accessible: {current}")
            break
        checkpoint = torch.load(current, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_args = checkpoint.get("args", {})
        sources = [checkpoint_args.get("data_path")]
        sources.extend(checkpoint_args.get("additional_data_paths") or [])
        if not sources[0]:
            unknown.append(f"checkpoint lacks training data_path: {current}")
        for raw_source in sources:
            if not raw_source:
                continue
            source = resolve_recorded_path(raw_source)
            record: dict[str, Any] = {"checkpoint": str(current), "source": str(source)}
            if train_source is not None and source == train_source:
                record.update({"status": "verified_train_split", "val_overlap": 0, "test_overlap": 0})
            elif not source.is_file():
                record["status"] = "source_unavailable"
                unknown.append(f"pretraining source not accessible: {source}")
            else:
                try:
                    source_samples = torch.load(
                        source, map_location="cpu", weights_only=False, mmap=True
                    )
                    if not isinstance(source_samples, list):
                        raise ValueError("source is not a sample list")
                    source_groups = {
                        str(getattr(sample, "group_id", "")) for sample in source_samples
                    }
                    if "" in source_groups:
                        raise ValueError("source contains a sample without group_id")
                    matched_val = source_groups & val_groups
                    matched_test = source_groups & test_groups
                    overlap_val.update(matched_val)
                    overlap_test.update(matched_test)
                    record.update({
                        "status": "scanned",
                        "val_overlap": len(matched_val),
                        "test_overlap": len(matched_test),
                    })
                    del source_samples
                except (OSError, RuntimeError, ValueError) as exc:
                    record["status"] = "scan_failed"
                    unknown.append(f"could not inspect {source}: {exc}")
            checked.append(record)
        parent = checkpoint_args.get("checkpoint")
        del checkpoint
        gc.collect()
        if not parent:
            break
        current = resolve_recorded_path(parent)

    status = "overlap" if overlap_val or overlap_test else "unknown" if unknown else "clean"
    return {
        "status": status,
        "val_overlap_count": len(overlap_val),
        "test_overlap_count": len(overlap_test),
        "checked_sources": checked,
        "unknown_reasons": unknown,
        "meaning": (
            "clean means recorded pretraining sources and ancestors have no val/test group_id overlap; "
            "unknown means this cannot be certified"
        ),
    }


class BeamHead(nn.Module):
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, NUM_BEAMS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class FullCSIBeamClassifier(nn.Module):
    def __init__(self, encoder: CSIEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = BeamHead()

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.head(self.encode(batch))

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        embedding = self.encoder(**encoder_kwargs(batch))
        return F.normalize(embedding, dim=-1)


class FullCSIBeamPhysicsClassifier(FullCSIBeamClassifier):
    """Same beam model plus training-only heads for physical supervision."""

    def __init__(self, encoder: CSIEncoder):
        super().__init__(encoder)
        self.physics_features = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.physics_regression_head = nn.Linear(128, len(PHYSICS_TARGET_NAMES))
        self.los_head = nn.Linear(128, 1)

    def forward_with_physics(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(**encoder_kwargs(batch))
        physical_features = self.physics_features(embedding)
        return (
            self.head(F.normalize(embedding, dim=-1)),
            self.physics_regression_head(physical_features),
            self.los_head(physical_features).squeeze(-1),
        )


def physics_auxiliary_loss(
    predicted: torch.Tensor, los_logits: torch.Tensor, batch: dict[str, Any]
) -> torch.Tensor:
    mask = batch["physics_mask"]
    regression = F.smooth_l1_loss(
        predicted.float(), batch["physics_targets"].float(), reduction="none", beta=1.0
    )
    valid_count = mask.sum().clamp_min(1)
    regression_loss = (regression * mask).sum() / valid_count
    los_loss = F.binary_cross_entropy_with_logits(
        los_logits.float(), batch["los_target"].float()
    )
    return regression_loss + los_loss


def training_physics_coverage(corpus: BeamCorpus, indices: torch.Tensor) -> dict[str, int]:
    """Count finite train-only labels without inspecting validation or test targets."""

    counts = torch.zeros(len(PHYSICS_TARGET_NAMES), dtype=torch.long)
    for index in indices.tolist():
        counts += torch.isfinite(physics_raw_values(corpus.samples[index])).long()
    return {name: int(counts[index]) for index, name in enumerate(PHYSICS_TARGET_NAMES)}


def make_loader(
    corpus: BeamCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    with_physics: bool = False,
) -> DataLoader:
    # All 16 patch tokens are used; sparse-observation bookkeeping is unnecessary.
    dataset = FullCSIBeamDataset(corpus, indices)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_full_csi_with_physics if with_physics else collate_full_csi,
        generator=torch.Generator().manual_seed(args.seed),
        **({"prefetch_factor": 1} if args.num_workers else {}),
    )


@torch.inference_mode()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        logits = model(batch)
        total += float(F.cross_entropy(logits, batch["target_class"])) * len(batch["target_class"])
        count += len(batch["target_class"])
    return total / max(count, 1)


def train_encoder_and_head(
    model: FullCSIBeamClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    lr: float,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []
    use_physics = isinstance(model, FullCSIBeamPhysicsClassifier)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_total = 0.0
        train_beam_total = 0.0
        train_physics_total = 0.0
        train_count = 0
        for batch in train_loader:
            batch = move_tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if use_physics:
                    logits, physical_predictions, los_logits = model.forward_with_physics(batch)
                    auxiliary_loss = physics_auxiliary_loss(
                        physical_predictions, los_logits, batch
                    )
                else:
                    logits = model(batch)
                    auxiliary_loss = None
                beam_loss = F.cross_entropy(logits, batch["target_class"])
                loss = (
                    beam_loss + args.physics_weight * auxiliary_loss
                    if auxiliary_loss is not None else beam_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(batch["target_class"])
            train_total += float(loss.detach()) * batch_size
            train_beam_total += float(beam_loss.detach()) * batch_size
            if auxiliary_loss is not None:
                train_physics_total += float(auxiliary_loss.detach()) * batch_size
            train_count += batch_size
        val_loss = evaluate_loss(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": train_total / train_count,
            "train_beam_loss": train_beam_total / train_count,
            "train_physics_loss": train_physics_total / train_count if use_physics else None,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
        })
        print(
            f"epoch={epoch} train_loss={train_total / train_count:.6f} "
            f"train_beam_loss={train_beam_total / train_count:.6f} "
            + (f"train_physics_loss={train_physics_total / train_count:.6f} " if use_physics else "")
            + f"val_loss={val_loss:.6f} epoch_seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("End-to-end training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return {"best_val_loss": best_val, "best_epoch": best_epoch, "history": history}


@torch.inference_mode()
def model_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    logits = []
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        logits.append(model(batch).float().cpu())
    return torch.cat(logits)


@torch.inference_mode()
def extract_frozen_embeddings(
    pretrained_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    count: int,
) -> torch.Tensor:
    pretrained_model.eval()
    result = torch.empty(count, 256, dtype=torch.float32)
    for batch in loader:
        indices = batch["indices"].long()
        batch = move_tensor_batch(batch, device)
        result[indices] = pretrained_model.encode_csi(
            **encoder_kwargs(batch), normalize=True
        ).float().cpu()
    return result


def train_frozen_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[BeamHead, dict[str, Any]]:
    device = torch.device(args.device)
    head = BeamHead(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(features[train_indices], labels[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_state = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        head.eval()
        with torch.inference_mode():
            val_loss = float(F.cross_entropy(
                head(features[val_indices].to(device)), labels[val_indices].to(device)
            ))
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={total / count:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Frozen-head training produced no validation checkpoint")
    head.load_state_dict(best_state)
    return head, {"best_val_loss": best_val, "best_epoch": best_epoch, "history": history}


@torch.inference_mode()
def frozen_logits(
    head: BeamHead, features: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    head.eval()
    parts = []
    for chunk in indices.split(2048):
        parts.append(head(features[chunk].to(device)).float().cpu())
    return torch.cat(parts)


def split_metrics(
    logits: torch.Tensor, oracle: torch.Tensor, gains_db: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    ranking = logits.topk(3, dim=-1).indices
    predicted = ranking[:, 0]
    rows = torch.arange(len(oracle))
    gain_loss = (
        gains_db[rows, oracle] - gains_db[rows, predicted]
    ).clamp_min(0.0)
    support = torch.bincount(oracle, minlength=NUM_BEAMS)
    present = support > 0
    correct = torch.bincount(oracle[predicted == oracle], minlength=NUM_BEAMS)
    macro_accuracy = (correct[present].float() / support[present].float()).mean()
    return {
        "samples": float(len(oracle)),
        "top1_accuracy": float((predicted == oracle).float().mean()),
        "top3_hit_rate": float((ranking == oracle[:, None]).any(dim=1).float().mean()),
        "macro_accuracy_present_classes": float(macro_accuracy),
        "gain_loss_mean_db": float(gain_loss.mean()),
        "gain_loss_median_db": float(torch.quantile(gain_loss, 0.5)),
        "gain_loss_p90_db": float(torch.quantile(gain_loss, 0.9)),
        "gain_loss_le_1db": float((gain_loss <= 1.0).float().mean()),
    }, ranking, gain_loss


def report_test(
    corpus: BeamCorpus, test_indices: torch.Tensor, logits: torch.Tensor, run_dir: Path
) -> dict[str, Any]:
    oracle = corpus.oracle_beam[test_indices]
    gains_db = corpus.beam_gains_db[test_indices]
    metrics, ranking, gain_loss = split_metrics(logits, oracle, gains_db)
    statuses = [corpus.los_status[int(index)] for index in test_indices]
    for name in ("los", "nlos"):
        mask = torch.tensor([status == name for status in statuses], dtype=torch.bool)
        if bool(mask.any()):
            group_metrics, _, _ = split_metrics(logits[mask], oracle[mask], gains_db[mask])
            metrics.update({f"{name}_{key}": value for key, value in group_metrics.items()})
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "group_id", "los_status", "oracle_beam", "predicted_beam",
            "top3_beams", "oracle_gain_db", "predicted_gain_db", "gain_loss_db",
        ])
        for row, index in enumerate(test_indices.tolist()):
            oracle_id = int(oracle[row])
            predicted_id = int(ranking[row, 0])
            writer.writerow([
                corpus.group_ids[index], statuses[row], oracle_id, predicted_id,
                " ".join(str(int(value)) for value in ranking[row]),
                float(gains_db[row, oracle_id]), float(gains_db[row, predicted_id]),
                float(gain_loss[row]),
            ])
    return metrics


def class_distribution(corpus: BeamCorpus) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        labels = corpus.oracle_beam[corpus.split_indices[name]]
        counts = torch.bincount(labels, minlength=NUM_BEAMS)
        result[name] = {
            "samples": len(labels),
            "nonempty_classes": int((counts > 0).sum()),
            "class_counts": counts.tolist(),
            "largest_class_fraction": float(counts.max() / len(labels)),
        }
    majority = int(torch.tensor(result["train"]["class_counts"]).argmax())
    result["majority_class_from_train"] = majority
    result["majority_baseline_test_accuracy"] = float(
        (corpus.oracle_beam[corpus.split_indices["test"]] == majority).float().mean()
    )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, epochs, and patience must be positive")
    if args.cpu_threads <= 0 or args.num_workers < 0:
        raise ValueError("cpu-threads must be positive and num-workers must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(
        f"device={device} cpu_threads={torch.get_num_threads()} "
        f"loader_workers={args.num_workers}",
        flush=True,
    )
    models = list(MODEL_NAMES) if "all" in args.models else list(dict.fromkeys(args.models))
    if any(name in PHYSICS_MODEL_NAMES for name in models) and args.physics_weight <= 0.0:
        raise ValueError("--physics-weight must be positive for Physics models")

    # Run the checkpoint ancestry/group-id audit before materializing 50k CSI samples.
    label_payload = torch.load(args.data, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(label_payload, dict):
        raise ValueError("Expected a beam-label dictionary")
    provenance = audit_pretraining_sources(label_payload, args.checkpoint)
    (args.output_dir / "pretraining_overlap_audit.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pretraining_overlap={provenance['status']} "
        f"val={provenance['val_overlap_count']} test={provenance['test_overlap_count']}",
        flush=True,
    )
    if args.require_clean_pretrain and provenance["status"] != "clean":
        raise ValueError("Pretraining overlap audit is not clean; see pretraining_overlap_audit.json")
    if provenance["status"] != "clean":
        warnings.warn(
            "Pretraining val/test separation is not certified clean. Do not claim unseen-link transfer."
        )
    del label_payload
    gc.collect()

    corpus = BeamCorpus.load(args.data)
    for index, sample in enumerate(corpus.samples):
        if tuple(sample.tokens.shape) != (16, 8, 128):
            raise ValueError(
                f"Full-CSI experiment requires [16,8,128] tokens; sample {index} "
                f"has {tuple(sample.tokens.shape)}"
            )
    distribution = class_distribution(corpus)
    (args.output_dir / "class_distribution.json").write_text(
        json.dumps(distribution, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "split_sizes=" + ",".join(
            f"{name}:{distribution[name]['samples']}" for name in ("train", "val", "test")
        )
    )
    print(
        f"train_nonempty_classes={distribution['train']['nonempty_classes']} "
        f"majority_baseline_test_accuracy={distribution['majority_baseline_test_accuracy']:.4f}"
    )
    train_indices = corpus.split_indices["train"]
    val_indices = corpus.split_indices["val"]
    test_indices = corpus.split_indices["test"]
    if any(name in PHYSICS_MODEL_NAMES for name in models):
        coverage = training_physics_coverage(corpus, train_indices)
        (args.output_dir / "training_physics_coverage.json").write_text(
            json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
        )
        print(f"training_physics_coverage={coverage}", flush=True)
        if coverage["first_path_aod_az_sin"] == 0:
            raise ValueError("Training split has no first-path AoD labels")
    val_loader = make_loader(corpus, val_indices, args, shuffle=False)
    test_loader = make_loader(corpus, test_indices, args, shuffle=False)
    all_loader = make_loader(corpus, torch.arange(len(corpus.samples)), args, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    pretrained_model, _, model_info = build_pretrained_model(checkpoint, torch.device("cpu"))
    del checkpoint
    gc.collect()
    result_rows: list[dict[str, Any]] = []

    for model_name in models:
        print(f"model={model_name} starting", flush=True)
        set_seed(args.seed)
        if model_name != "frozen":
            # Reset the shuffle generator for each model so matched ablations
            # see the same training-sample order.
            train_loader = make_loader(
                corpus, train_indices, args, shuffle=True,
                with_physics=model_name in PHYSICS_MODEL_NAMES,
            )
        run_dir = args.output_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if model_name in {"scratch", "scratch_physics"}:
            encoder = CSIEncoder(
                d_token=int(model_info["d_token"]), d_model=384, d_clip=256,
                token_norm_mode=str(model_info["token_norm_mode"]),
                use_continuous_config_encoding=bool(model_info["use_continuous_config_encoding"]),
            )
            use_physics = model_name in PHYSICS_MODEL_NAMES
            model = (
                FullCSIBeamPhysicsClassifier(encoder) if use_physics
                else FullCSIBeamClassifier(encoder)
            )
            training = train_encoder_and_head(
                model, train_loader, val_loader, args, args.scratch_lr,
            )
            logits = model_logits(model, test_loader, device)
            torch.save({
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "training": training, "model": model_name,
                "physics_weight": args.physics_weight if use_physics else 0.0,
            }, run_dir / "best.pt")
            del model
        elif model_name == "frozen":
            pretrained_model.to(device)
            features = extract_frozen_embeddings(
                pretrained_model, all_loader, device, len(corpus.samples)
            )
            pretrained_model.to("cpu")
            head, training = train_frozen_head(
                features, corpus.oracle_beam, train_indices, val_indices, args
            )
            logits = frozen_logits(head, features, test_indices, device)
            torch.save({
                "head_state": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "pretrained_checkpoint": str(args.checkpoint.resolve()),
                "training": training, "model": model_name,
            }, run_dir / "best.pt")
            del head, features
        elif model_name in {"fine_tune", "fine_tune_physics"}:
            encoder = copy.deepcopy(pretrained_model.csi)
            for parameter in encoder.parameters():
                parameter.requires_grad = True
            use_physics = model_name in PHYSICS_MODEL_NAMES
            model = (
                FullCSIBeamPhysicsClassifier(encoder) if use_physics
                else FullCSIBeamClassifier(encoder)
            )
            training = train_encoder_and_head(
                model, train_loader, val_loader, args, args.fine_tune_lr,
            )
            logits = model_logits(model, test_loader, device)
            torch.save({
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "pretrained_checkpoint": str(args.checkpoint.resolve()),
                "training": training, "model": model_name,
                "physics_weight": args.physics_weight if use_physics else 0.0,
            }, run_dir / "best.pt")
            del model
        else:
            raise AssertionError(model_name)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        metrics = report_test(corpus, test_indices, logits, run_dir)
        result = {
            "model": model_name,
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "physics_weight": args.physics_weight if model_name in PHYSICS_MODEL_NAMES else 0.0,
            "pretraining_overlap_status": provenance["status"],
            **metrics,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "history.json").write_text(
            json.dumps(training["history"], indent=2) + "\n", encoding="utf-8"
        )
        result_rows.append(result)
        print(
            f"model={model_name} top1={metrics['top1_accuracy']:.4f} "
            f"top3={metrics['top3_hit_rate']:.4f} "
            f"gain_loss_median_db={metrics['gain_loss_median_db']:.4f}",
            flush=True,
        )
        del logits

    (args.output_dir / "results.json").write_text(
        json.dumps(result_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    keys = sorted({key for row in result_rows for key in row})
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result_rows)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({
            "data": str(args.data.resolve()),
            "pretrained_checkpoint": str(args.checkpoint.resolve()),
            "models": models,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "learning_rates": {
                "scratch": args.scratch_lr,
                "frozen_head": args.head_lr,
                "fine_tune": args.fine_tune_lr,
            },
            "physics_weight": args.physics_weight,
            "physics_targets": list(PHYSICS_TARGET_NAMES) + ["los_status"],
            "physics_loss": "masked normalized smooth-L1 regression plus LoS BCE; train split only",
            "input": "complete 16x8x128 CSI patches; all 64 DFT beams observed",
            "loss": "beam cross-entropy, plus physics_weight times physics_loss for Physics models",
            "validation_selection": "minimum beam validation cross-entropy for every model",
            "gain_loss": "oracle gain dB minus Top-1 predicted beam gain dB",
            "pretraining_overlap_status": provenance["status"],
        }, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
