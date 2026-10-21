from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PHYSICS_TARGET_OFFSETS, PHYSICS_TARGET_SCALES  # noqa: E402
from downstream_localization.data_io import (  # noqa: E402
    LocalizationCorpus,
    LocalizationDataset,
    collate_localization,
    move_tensor_batch,
)
from downstream_localization.pretrained import (  # noqa: E402
    build_pretrained_model,
    los_probability,
)


LIGHT_SPEED_MPS = 299_792_458.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate real-measured CSI physical consistency without path-level GT: "
            "LoS/NLoS distributions, delay-distance slope, power-distance trend, "
            "and spatial consistency."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "val", "test", "all"),
        help="Corpus splits to analyze. Default: test.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap for quick checks; samples are selected after split filtering.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=200_000,
        help="Random sample pairs for the spatial consistency curve.",
    )
    parser.add_argument(
        "--scatter-samples",
        type=int,
        default=4000,
        help="Maximum points drawn per scatter panel.",
    )
    return parser.parse_args()


def selected_indices(corpus: LocalizationCorpus, args: argparse.Namespace) -> torch.Tensor:
    if "all" in args.splits:
        indices = torch.arange(len(corpus.samples), dtype=torch.long)
    else:
        parts = [corpus.split_indices[name] for name in args.splits]
        indices = torch.cat(parts).unique(sorted=True)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        generator = torch.Generator().manual_seed(args.seed)
        if len(indices) > args.max_samples:
            order = torch.randperm(len(indices), generator=generator)[: args.max_samples]
            indices = torch.sort(indices[order]).values
    return indices


def make_loader(
    corpus: LocalizationCorpus,
    indices: torch.Tensor,
    args: argparse.Namespace,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        LocalizationDataset(corpus, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_localization,
    )


def csi_power_db_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    tokens = batch["tokens"].float()
    mask = batch["token_mask"].to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
    element_count = (
        batch["token_mask"].sum(dim=1).to(dtype=tokens.dtype)
        * tokens.shape[2]
        * tokens.shape[3]
    ).clamp(min=1.0)
    power = (tokens.square() * mask).sum(dim=(1, 2, 3)) / element_count
    return 10.0 * torch.log10(power.clamp(min=1.0e-30))


@torch.inference_mode()
def extract_predictions(
    corpus: LocalizationCorpus,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    del checkpoint_path
    device = torch.device(args.device)
    model, prototype_keys, model_info = build_pretrained_model(checkpoint, device)
    if corpus.samples[0].tokens.shape[1] != model_info["d_token"]:
        raise ValueError(
            f"Dataset d_token={corpus.samples[0].tokens.shape[1]} but checkpoint "
            f"expects {model_info['d_token']}"
        )
    indices = selected_indices(corpus, args)
    loader = make_loader(corpus, indices, args)

    rows: dict[str, list[np.ndarray]] = {
        "index": [],
        "xy_m": [],
        "xyz_m": [],
        "distance_m": [],
        "los_label": [],
        "embedding": [],
        "p_los": [],
        "first_delay_ns": [],
        "delay_spread_ns": [],
        "k_factor_db": [],
        "first_power_dbw": [],
        "total_power_dbw": [],
        "first_angle_deg": [],
        "csi_power_db": [],
    }
    scales = PHYSICS_TARGET_SCALES.to(device=device)
    offsets = PHYSICS_TARGET_OFFSETS.to(device=device)
    amp = device.type == "cuda" and not args.no_amp
    checkpoint_args = dict(checkpoint.get("args", {}))

    for step, batch in enumerate(loader, start=1):
        batch = move_tensor_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            raw_embedding = model.encode_csi(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
                normalize=False,
                config_features=batch.get("config_features"),
                antenna_coordinates=batch.get("antenna_coordinates"),
                antenna_mask=batch.get("antenna_mask"),
            )
            delay_context = model.encode_csi_delay_context(
                batch["tokens"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
                config_features=batch.get("config_features"),
            )
            first_delay_context = model.encode_first_path_delay_context(
                batch["tokens"],
                batch["token_mask"],
                beam_positions=batch.get("beam_positions"),
                freq_bin=batch.get("freq_bin"),
                bw_bin=batch.get("bw_bin"),
                subcarrier_spacing=batch.get("subcarrier_spacing"),
                config_features=batch.get("config_features"),
            )
            los_angle_context = model.encode_los_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
            )
            first_angle_context = model.encode_first_path_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
            power_context = None
            if model.use_power_branch:
                power_context = model.encode_power_context(
                    batch["tokens"],
                    batch["token_mask"],
                    delay_power_map=batch.get("delay_power_map"),
                    delay_power_profile=batch.get("delay_power_profile"),
                )
            physics = model.predict_physics_components(
                raw_embedding,
                power_context=power_context,
                delay_context=delay_context,
                first_path_delay_context=first_delay_context,
                los_angle_context=los_angle_context,
                first_path_angle_context=first_angle_context,
            )
            p_los = los_probability(model, raw_embedding, prototype_keys, checkpoint_args)

        final = physics["final"].float()
        raw = final * scales[: final.shape[1]] + offsets[: final.shape[1]]
        first_delay_ns = physics["first_path_delay_bin_soft_fused_raw"].float()
        first_angle = F.normalize(physics["first_path_angle_sincos"].float(), dim=-1, eps=1e-6)
        first_angle_deg = torch.rad2deg(torch.atan2(first_angle[:, 0], first_angle[:, 1]))
        xyz = batch["rx_xyz"].float()
        reference_xyz = batch["tx_xyz"].float()
        distance = torch.linalg.vector_norm(xyz - reference_xyz, dim=-1)
        csi_power_db = csi_power_db_from_batch(batch)
        labels = np.asarray(
            [
                str(corpus.samples[int(index)].semantic_key.los_status).lower()
                for index in batch["indices"].detach().cpu()
            ]
        )

        rows["index"].append(batch["indices"].detach().cpu().numpy())
        rows["xy_m"].append(xyz[:, :2].detach().cpu().numpy())
        rows["xyz_m"].append(xyz.detach().cpu().numpy())
        rows["distance_m"].append(distance.detach().cpu().numpy())
        rows["los_label"].append(labels)
        rows["embedding"].append(F.normalize(raw_embedding.float(), dim=-1).detach().cpu().numpy())
        rows["p_los"].append(p_los.squeeze(-1).float().detach().cpu().numpy())
        rows["first_delay_ns"].append(first_delay_ns.detach().cpu().numpy())
        rows["delay_spread_ns"].append(raw[:, 1].detach().cpu().numpy())
        rows["k_factor_db"].append(raw[:, 3].detach().cpu().numpy())
        rows["first_power_dbw"].append(raw[:, 5].detach().cpu().numpy())
        if raw.shape[1] > 10:
            rows["total_power_dbw"].append(raw[:, 10].detach().cpu().numpy())
        else:
            rows["total_power_dbw"].append(
                np.full(len(labels), np.nan, dtype=np.float32)
            )
        rows["first_angle_deg"].append(first_angle_deg.detach().cpu().numpy())
        rows["csi_power_db"].append(csi_power_db.detach().cpu().numpy())
        if step % 50 == 0 or step == len(loader):
            print(f"prediction batches={step}/{len(loader)}")

    output: dict[str, np.ndarray] = {}
    for key, parts in rows.items():
        output[key] = np.concatenate(parts, axis=0)
    return output


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = finite_mask(x, y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0.0 else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = finite_mask(x, y)
    if int(mask.sum()) < 2:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = finite_mask(x, y)
    if int(mask.sum()) < 2:
        return {
            "count": int(mask.sum()),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return {
        "count": int(mask.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
    }


def summarize_by_label(predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = predictions["los_label"]
    for label in ("los", "nlos"):
        mask = labels == label
        record: dict[str, Any] = {"label": label, "count": int(mask.sum())}
        for key in (
            "p_los",
            "first_delay_ns",
            "delay_spread_ns",
            "k_factor_db",
            "first_power_dbw",
            "total_power_dbw",
            "csi_power_db",
        ):
            values = predictions[key][mask]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                for suffix in ("mean", "median", "p25", "p75"):
                    record[f"{key}_{suffix}"] = float("nan")
                continue
            record[f"{key}_mean"] = float(np.mean(finite))
            record[f"{key}_median"] = float(np.median(finite))
            record[f"{key}_p25"] = float(np.percentile(finite, 25))
            record[f"{key}_p75"] = float(np.percentile(finite, 75))
        rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def random_spatial_pairs(
    predictions: dict[str, np.ndarray],
    pair_count: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    count = len(predictions["index"])
    if count < 2 or pair_count <= 0:
        return {}, []
    i = rng.integers(0, count, size=pair_count)
    j = rng.integers(0, count, size=pair_count)
    keep = i != j
    i = i[keep]
    j = j[keep]
    xy = predictions["xy_m"]
    distance = np.linalg.norm(xy[i] - xy[j], axis=1)
    embedding_diff = np.linalg.norm(
        predictions["embedding"][i] - predictions["embedding"][j], axis=1
    )
    delay_diff = np.abs(predictions["first_delay_ns"][i] - predictions["first_delay_ns"][j])
    spread_diff = np.abs(predictions["delay_spread_ns"][i] - predictions["delay_spread_ns"][j])
    k_diff = np.abs(predictions["k_factor_db"][i] - predictions["k_factor_db"][j])
    power_key = (
        "total_power_dbw"
        if np.isfinite(predictions["total_power_dbw"]).any()
        else "csi_power_db"
    )
    power_diff = np.abs(predictions[power_key][i] - predictions[power_key][j])

    summary = {
        "pair_count": int(len(distance)),
        "embedding_diff_spearman": spearman(distance, embedding_diff),
        "first_delay_diff_spearman": spearman(distance, delay_diff),
        "delay_spread_diff_spearman": spearman(distance, spread_diff),
        "k_factor_diff_spearman": spearman(distance, k_diff),
        "power_diff_spearman": spearman(distance, power_diff),
        "power_key": power_key,
    }
    bins = np.asarray([0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, np.inf])
    rows: list[dict[str, Any]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (distance >= left) & (distance < right)
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "distance_bin_m": f"[{left:.2f},{right:.2f})"
                if np.isfinite(right)
                else f">={left:.2f}",
                "count": int(mask.sum()),
                "distance_median": float(np.median(distance[mask])),
                "embedding_diff_median": float(np.median(embedding_diff[mask])),
                "first_delay_diff_ns_median": float(np.nanmedian(delay_diff[mask])),
                "delay_spread_diff_ns_median": float(np.nanmedian(spread_diff[mask])),
                "k_factor_diff_db_median": float(np.nanmedian(k_diff[mask])),
                "power_diff_db_median": float(np.nanmedian(power_diff[mask])),
            }
        )
    return summary, rows


def sample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(x), size=max_points, replace=False)
    return x[choice], y[choice]


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values[np.isfinite(values)]


def plot_summary(
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
    output: Path,
    scatter_samples: int,
    seed: int,
) -> None:
    labels = predictions["los_label"]
    los_mask = labels == "los"
    distance = predictions["distance_m"]
    total_power = predictions["total_power_dbw"]
    power_key = "total_power_dbw" if np.isfinite(total_power).any() else "csi_power_db"
    power = predictions[power_key]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ax = axes[0, 0]
    x, y = sample_for_plot(
        distance[los_mask],
        predictions["first_delay_ns"][los_mask],
        scatter_samples,
        seed,
    )
    ax.scatter(x, y, s=8, alpha=0.35)
    fit = metrics["delay_vs_distance_los"]
    if len(x) > 1 and math.isfinite(fit["slope"]):
        grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(grid, fit["slope"] * grid + fit["intercept"], color="tab:red", linewidth=2)
    ax.set_xlabel("Distance to array reference (m)")
    ax.set_ylabel("Predicted first-path delay (ns)")
    ax.set_title("LoS delay-distance trend")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    log_distance = np.log10(np.clip(distance[los_mask], 1.0e-6, None))
    x, y = sample_for_plot(log_distance, power[los_mask], scatter_samples, seed + 1)
    ax.scatter(x, y, s=8, alpha=0.35)
    fit = metrics["power_vs_log_distance_los"][power_key]
    if len(x) > 1 and math.isfinite(fit["slope"]):
        grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(grid, fit["slope"] * grid + fit["intercept"], color="tab:red", linewidth=2)
    ax.set_xlabel("log10(distance / m)")
    ax.set_ylabel(f"{power_key} (dB)")
    ax.set_title("LoS power-distance trend")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    data = [
        finite_values(predictions["k_factor_db"][labels == "los"]),
        finite_values(predictions["k_factor_db"][labels == "nlos"]),
    ]
    data = [values if len(values) else np.asarray([np.nan]) for values in data]
    ax.boxplot(data, labels=["LoS", "NLoS"], showfliers=False)
    ax.set_ylabel("Predicted K-factor (dB)")
    ax.set_title("LoS/NLoS K-factor distribution")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    data = [
        finite_values(predictions["delay_spread_ns"][labels == "los"]),
        finite_values(predictions["delay_spread_ns"][labels == "nlos"]),
    ]
    data = [values if len(values) else np.asarray([np.nan]) for values in data]
    ax.boxplot(data, labels=["LoS", "NLoS"], showfliers=False)
    ax.set_ylabel("Predicted delay spread (ns)")
    ax.set_title("LoS/NLoS delay-spread distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_spatial_consistency(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    x = np.arange(len(rows))
    labels = [row["distance_bin_m"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    panels = [
        ("embedding_diff_median", "Embedding L2 diff"),
        ("first_delay_diff_ns_median", "First-delay diff (ns)"),
        ("k_factor_diff_db_median", "K-factor diff (dB)"),
        ("power_diff_db_median", "Power diff (dB)"),
    ]
    for axis, (key, ylabel) in zip(axes.ravel(), panels):
        axis.plot(x, [float(row[key]) for row in rows], marker="o", linewidth=2)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_metrics(
    predictions: dict[str, np.ndarray],
    pair_count: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = predictions["los_label"]
    los_mask = labels == "los"
    distance = predictions["distance_m"]
    delay_fit = linear_fit(
        distance[los_mask],
        predictions["first_delay_ns"][los_mask],
    )
    delay_fit["theory_slope_ns_per_m"] = float(1.0e9 / LIGHT_SPEED_MPS)
    delay_fit["slope_error_ns_per_m"] = (
        float(delay_fit["slope"] - delay_fit["theory_slope_ns_per_m"])
        if math.isfinite(delay_fit["slope"])
        else float("nan")
    )

    power_fits: dict[str, Any] = {}
    log_distance = np.log10(np.clip(distance[los_mask], 1.0e-6, None))
    for key in ("total_power_dbw", "first_power_dbw", "csi_power_db"):
        fit = linear_fit(log_distance, predictions[key][los_mask])
        fit["path_loss_exponent"] = (
            float(-fit["slope"] / 10.0) if math.isfinite(fit["slope"]) else float("nan")
        )
        power_fits[key] = fit

    distribution_rows = summarize_by_label(predictions)
    by_label = {row["label"]: row for row in distribution_rows}
    los = by_label.get("los", {})
    nlos = by_label.get("nlos", {})
    label_deltas = {
        "p_los_median_los_minus_nlos": float(
            los.get("p_los_median", float("nan")) - nlos.get("p_los_median", float("nan"))
        ),
        "k_factor_median_los_minus_nlos_db": float(
            los.get("k_factor_db_median", float("nan"))
            - nlos.get("k_factor_db_median", float("nan"))
        ),
        "delay_spread_median_nlos_minus_los_ns": float(
            nlos.get("delay_spread_ns_median", float("nan"))
            - los.get("delay_spread_ns_median", float("nan"))
        ),
    }
    spatial_summary, spatial_rows = random_spatial_pairs(predictions, pair_count, seed)
    metrics = {
        "sample_count": int(len(predictions["index"])),
        "label_counts": {
            "los": int((labels == "los").sum()),
            "nlos": int((labels == "nlos").sum()),
        },
        "delay_vs_distance_los": delay_fit,
        "power_vs_log_distance_los": power_fits,
        "label_distribution_deltas": label_deltas,
        "spatial_consistency": spatial_summary,
    }
    return metrics, distribution_rows, spatial_rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = LocalizationCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictions = extract_predictions(corpus, checkpoint, args.checkpoint, args)
    metrics, distribution_rows, spatial_rows = build_metrics(
        predictions, args.pair_count, args.seed
    )

    np.savez_compressed(
        args.output_dir / "physical_consistency_predictions.npz",
        **predictions,
    )
    write_csv(args.output_dir / "los_nlos_distribution_summary.csv", distribution_rows)
    write_csv(args.output_dir / "spatial_consistency_bins.csv", spatial_rows)
    (args.output_dir / "physical_consistency_metrics.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "splits": args.splits,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_summary(
        predictions,
        metrics,
        args.output_dir / "physical_consistency_summary.png",
        args.scatter_samples,
        args.seed,
    )
    plot_spatial_consistency(
        spatial_rows,
        args.output_dir / "spatial_consistency.png",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
