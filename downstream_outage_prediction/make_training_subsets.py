from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_INPUT = (
    DEFAULT_REPO
    / "artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_REPO / "downstream_outage_prediction/artifacts/llm_subsets_seed23421"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nested LoS/NLoS-balanced train subsets for outage baselines."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=(1000, 10000))
    parser.add_argument("--seed", type=int, default=23421)
    return parser.parse_args()


def load_samples(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        samples = payload["samples"]
    else:
        raise ValueError(f"Expected a sample list in {path}.")
    if not samples:
        raise ValueError(f"Empty sample set: {path}")
    return samples


def los_status(sample) -> str:
    status = str(getattr(getattr(sample, "semantic_key", None), "los_status", "")).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(
            f"Unsupported los_status={status!r} for group_id={getattr(sample, 'group_id', '')!r}."
        )
    return status


def stable_rank(seed: int, group_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()


def group_id_digest(group_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for group_id in group_ids:
        digest.update(group_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def status_counts(samples: list) -> dict[str, int]:
    counts = Counter(los_status(sample) for sample in samples)
    return {"los": int(counts["los"]), "nlos": int(counts["nlos"])}


def main() -> None:
    args = parse_args()
    sizes = sorted(set(int(size) for size in args.train_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--train-sizes must contain positive integers.")
    if any(size % 2 for size in sizes):
        raise ValueError("Each train size must be even for exact LoS/NLoS balance.")

    samples = load_samples(args.input)
    by_status: dict[str, list] = {"los": [], "nlos": []}
    seen_group_ids: set[str] = set()
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError("Every sample must have a non-empty group_id.")
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate group_id in input: {group_id}")
        seen_group_ids.add(group_id)
        by_status[los_status(sample)].append(sample)

    largest_per_status = max(sizes) // 2
    for status, status_samples in by_status.items():
        if len(status_samples) < largest_per_status:
            raise ValueError(
                f"Need {largest_per_status} {status} samples, found "
                f"{len(status_samples)} in {args.input}."
            )
        status_samples.sort(
            key=lambda sample: stable_rank(args.seed, str(sample.group_id))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "csi_outage_llm_train_subsets_v1",
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "source_count": len(samples),
        "source_status_counts": status_counts(samples),
        "subsets": {},
    }
    previous_ids: set[str] | None = None
    for size in sizes:
        per_status = size // 2
        subset = by_status["los"][:per_status] + by_status["nlos"][:per_status]
        subset.sort(key=lambda sample: stable_rank(args.seed + size, str(sample.group_id)))
        group_ids = [str(sample.group_id) for sample in subset]
        current_ids = set(group_ids)
        if previous_ids is not None and not previous_ids.issubset(current_ids):
            raise RuntimeError("Train subsets are not nested as expected.")
        previous_ids = current_ids

        path = args.output_dir / f"train_{size}.pt"
        torch.save(subset, path)
        manifest["subsets"][str(size)] = {
            "path": str(path.resolve()),
            "count": len(subset),
            "status_counts": status_counts(subset),
            "group_ids_sha256": group_id_digest(group_ids),
        }
        print(f"train_{size}={path} count={len(subset)} counts={status_counts(subset)}")

    manifest_path = args.output_dir / "train_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
