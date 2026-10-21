from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_VALIDATION = DEFAULT_REPO / "downstream_outage_prediction/artifacts/seed23421/validation.pt"
DEFAULT_TEST = DEFAULT_REPO / "downstream_outage_prediction/artifacts/seed23421/test.pt"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO / "downstream_outage_prediction/artifacts/smoke_eval_seed23421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create small LoS/NLoS-balanced validation/test subsets for smoke runs."
    )
    parser.add_argument("--validation-input", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
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


def balanced_subset(samples: list, size: int, seed: int) -> list:
    if size <= 0 or size % 2:
        raise ValueError("Subset size must be a positive even integer.")
    by_status: dict[str, list] = {"los": [], "nlos": []}
    seen_group_ids: set[str] = set()
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError("Every sample must have a non-empty group_id.")
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate group_id in input split: {group_id}")
        seen_group_ids.add(group_id)
        by_status[los_status(sample)].append(sample)

    per_status = size // 2
    for status, status_samples in by_status.items():
        if len(status_samples) < per_status:
            raise ValueError(
                f"Need {per_status} {status} samples, found {len(status_samples)}."
            )
        status_samples.sort(key=lambda sample: stable_rank(seed, str(sample.group_id)))

    subset = by_status["los"][:per_status] + by_status["nlos"][:per_status]
    subset.sort(key=lambda sample: stable_rank(seed + size, str(sample.group_id)))
    return subset


def save_subset(path: Path, samples: list, source: Path) -> dict[str, object]:
    torch.save(samples, path)
    group_ids = [str(sample.group_id) for sample in samples]
    return {
        "path": str(path.resolve()),
        "source": str(source.resolve()),
        "count": len(samples),
        "status_counts": status_counts(samples),
        "group_ids_sha256": group_id_digest(group_ids),
    }


def main() -> None:
    args = parse_args()
    validation = balanced_subset(
        load_samples(args.validation_input),
        args.validation_size,
        args.seed + 101,
    )
    test = balanced_subset(
        load_samples(args.test_input),
        args.test_size,
        args.seed + 202,
    )
    validation_ids = {str(sample.group_id) for sample in validation}
    test_ids = {str(sample.group_id) for sample in test}
    if validation_ids & test_ids:
        raise RuntimeError("Validation/test group IDs overlap.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / f"validation_{args.validation_size}.pt"
    test_path = args.output_dir / f"test_{args.test_size}.pt"
    manifest = {
        "format": "csi_outage_smoke_eval_subsets_v1",
        "seed": args.seed,
        "validation": save_subset(validation_path, validation, args.validation_input),
        "test": save_subset(test_path, test, args.test_input),
        "cross_split_group_id_overlap": 0,
    }
    manifest_path = args.output_dir / "smoke_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"validation={validation_path} count={len(validation)} counts={status_counts(validation)}")
    print(f"test={test_path} count={len(test)} counts={status_counts(test)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
