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
    / "artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO / "downstream_outage_prediction/artifacts/seed23421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split an existing LoS/NLoS-balanced held-out set into deterministic, "
            "disjoint validation and test sets."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-per-status", type=int, default=2500)
    parser.add_argument("--test-per-status", type=int, default=2500)
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
    status = str(
        getattr(getattr(sample, "semantic_key", None), "los_status", "")
    ).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(
            f"Unsupported los_status={status!r} for group_id={sample.group_id!r}."
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
    if args.validation_per_status <= 0 or args.test_per_status <= 0:
        raise ValueError("Per-status validation/test counts must be positive.")

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

    requested_per_status = args.validation_per_status + args.test_per_status
    for status, status_samples in by_status.items():
        if len(status_samples) < requested_per_status:
            raise ValueError(
                f"Need {requested_per_status} {status} samples, found "
                f"{len(status_samples)} in {args.input}."
            )
        status_samples.sort(
            key=lambda sample: stable_rank(args.seed, str(sample.group_id))
        )

    validation = []
    test = []
    for status in ("los", "nlos"):
        status_samples = by_status[status]
        validation.extend(status_samples[: args.validation_per_status])
        test.extend(
            status_samples[
                args.validation_per_status : args.validation_per_status
                + args.test_per_status
            ]
        )

    validation.sort(key=lambda sample: stable_rank(args.seed + 1, str(sample.group_id)))
    test.sort(key=lambda sample: stable_rank(args.seed + 2, str(sample.group_id)))
    validation_ids = [str(sample.group_id) for sample in validation]
    test_ids = [str(sample.group_id) for sample in test]
    if set(validation_ids) & set(test_ids):
        raise RuntimeError("Validation/test group IDs overlap.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "validation.pt"
    test_path = args.output_dir / "test.pt"
    torch.save(validation, validation_path)
    torch.save(test, test_path)

    manifest = {
        "format": "csi_outage_split_v1",
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "source_count": len(samples),
        "source_status_counts": status_counts(samples),
        "validation_path": str(validation_path.resolve()),
        "validation_count": len(validation),
        "validation_status_counts": status_counts(validation),
        "validation_group_ids_sha256": group_id_digest(validation_ids),
        "test_path": str(test_path.resolve()),
        "test_count": len(test),
        "test_status_counts": status_counts(test),
        "test_group_ids_sha256": group_id_digest(test_ids),
        "unused_count": len(samples) - len(validation) - len(test),
        "cross_split_group_id_overlap": 0,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"validation={validation_path} count={len(validation)} counts={status_counts(validation)}")
    print(f"test={test_path} count={len(test)} counts={status_counts(test)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()

