from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_localization.data_io import LocalizationCorpus  # noqa: E402
from downstream_localization.pretrained import build_pretrained_model  # noqa: E402
from downstream_localization.run_experiment import (  # noqa: E402
    CoordinateNormalizer,
    load_or_extract_features,
    save_predictions,
    seed_everything,
    train_scratch,
)


MODEL_LABELS = {
    "scratch": "Scratch",
    "scratch_physics": "Scratch + frozen physics",
}
METRIC_LABELS = {
    "mean_m": "Mean error (m)",
    "median_m": "Median error (m)",
    "p90_m": "P90 error (m)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure localization label efficiency with nested training subsets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[23421],
        help="Downstream initialization/training seeds.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=23421,
        help="Fixed seed defining nested label subsets; independent of training seeds.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional existing frozen_features.pt from the four-model experiment.",
    )
    return parser.parse_args()


def validate_fractions(values: list[float]) -> list[float]:
    fractions = sorted(set(float(value) for value in values))
    if not fractions or any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("Every label fraction must be in (0, 1]")
    return fractions


def nested_subsets(
    train_indices: torch.Tensor,
    fractions: list[float],
    seed: int,
) -> dict[float, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
    subsets: dict[float, torch.Tensor] = {}
    previous: set[int] = set()
    for fraction in fractions:
        count = max(1, int(round(len(train_indices) * fraction)))
        selected = torch.sort(permutation[:count]).values
        current = set(selected.tolist())
        if not previous.issubset(current):
            raise RuntimeError("Label subsets are not nested")
        subsets[fraction] = selected
        previous = current
    return subsets


def los_counts(corpus: LocalizationCorpus, indices: torch.Tensor) -> tuple[int, int]:
    los = sum(
        str(corpus.samples[int(index)].semantic_key.los_status).lower() == "los"
        for index in indices
    )
    return int(los), int(len(indices) - los)


def fraction_tag(fraction: float) -> str:
    percentage = fraction * 100.0
    text = f"{percentage:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}pct"


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["label_fraction"], row["train_samples"])].append(row)
    metric_names = ("mean_m", "median_m", "p90_m", "within_5m_pct", "within_10m_pct")
    summary = []
    for (model, fraction, count), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        record: dict[str, Any] = {
            "model": model,
            "display_name": MODEL_LABELS[model],
            "label_fraction": fraction,
            "label_percent": 100.0 * fraction,
            "train_samples": count,
            "runs": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_label_efficiency(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for axis, (metric, label) in zip(axes, METRIC_LABELS.items()):
        for model in ("scratch", "scratch_physics"):
            rows = sorted(
                [row for row in summary if row["model"] == model],
                key=lambda row: row["train_samples"],
            )
            x = np.asarray([row["train_samples"] for row in rows])
            y = np.asarray([row[f"{metric}_mean"] for row in rows])
            error = np.asarray([row[f"{metric}_std"] for row in rows])
            axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=MODEL_LABELS[model],
            )
        axis.set_xscale("log")
        axis.set_xlabel("Number of labeled localization samples")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fractions = validate_fractions(args.label_fractions)
    if not args.seeds:
        raise ValueError("At least one training seed is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    corpus = LocalizationCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained_model, prototype_keys, model_info = build_pretrained_model(checkpoint, device)
    if corpus.samples[0].tokens.shape[1] != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={corpus.samples[0].tokens.shape[1]} but checkpoint "
            f"expects {model_info['d_token']}"
        )

    subsets = nested_subsets(
        corpus.split_indices["train"], fractions, args.subset_seed
    )
    args.seed = int(args.seeds[0])
    seed_everything(args.seed)
    _, physics = load_or_extract_features(
        corpus,
        checkpoint,
        args.checkpoint,
        pretrained_model,
        prototype_keys,
        args,
        device,
    )
    del pretrained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    detailed_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        indices = subsets[fraction]
        train_los, train_nlos = los_counts(corpus, indices)
        normalizer = CoordinateNormalizer(corpus.relative_xy_m[indices])
        print(
            f"label_fraction={fraction:.4f} train_samples={len(indices)} "
            f"los={train_los} nlos={train_nlos}"
        )
        for seed in args.seeds:
            args.seed = int(seed)
            for model_name, frozen_physics in (
                ("scratch", None),
                ("scratch_physics", physics),
            ):
                run_name = f"{model_name}_{fraction_tag(fraction)}_seed{seed}"
                result = train_scratch(
                    run_name,
                    corpus,
                    normalizer,
                    model_info,
                    args,
                    device,
                    physics_features=frozen_physics,
                    train_indices=indices,
                )
                save_predictions(run_name, result, corpus, args.output_dir)
                detailed_rows.append(
                    {
                        "model": model_name,
                        "display_name": MODEL_LABELS[model_name],
                        "seed": int(seed),
                        "subset_seed": args.subset_seed,
                        "label_fraction": fraction,
                        "label_percent": 100.0 * fraction,
                        "train_samples": len(indices),
                        "train_los": train_los,
                        "train_nlos": train_nlos,
                        **result[0],
                    }
                )
                write_csv(args.output_dir / "detailed_metrics.csv", detailed_rows)

    summary = aggregate_rows(detailed_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary)
    plot_label_efficiency(summary, args.output_dir / "label_efficiency.png")
    payload = {
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "fractions": fractions,
        "seeds": [int(seed) for seed in args.seeds],
        "subset_seed": args.subset_seed,
        "split_sizes": {name: len(value) for name, value in corpus.split_indices.items()},
        "summary": summary,
    }
    (args.output_dir / "label_efficiency.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
