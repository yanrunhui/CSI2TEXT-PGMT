from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_INPUT = (
    DEFAULT_REPO
    / "downstream_mcs_adaptation/artifacts/d2los_300k_mcs_power_profiles.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/llm_mcs_subsets"
PER_SAMPLE_FIELDS = (
    "sample_refs",
    "group_ids",
    "physical_group_ids",
    "transmit_power_dbm",
    "power_offset_db",
    "labels",
    "oracle_mcs",
    "total_received_power_dbw",
    "average_snr_db",
    "frequency_selectivity_db",
    "effective_snr_per_mcs",
    "estimated_bler_per_mcs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nested, class-stratified MCS manifests for LLM smoke and formal runs."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formal-train", type=int, default=10_000)
    parser.add_argument("--formal-val", type=int, default=5_000)
    parser.add_argument("--formal-test", type=int, default=5_000)
    parser.add_argument("--smoke-train", type=int, default=1_000)
    parser.add_argument("--smoke-val", type=int, default=500)
    parser.add_argument("--smoke-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23_421)
    return parser.parse_args()


def stable_rank(index: int, seed: int, salt: str) -> bytes:
    return hashlib.sha256(f"{seed}:{salt}:{index}".encode("utf-8")).digest()


def allocate_quotas(
    labels: torch.Tensor,
    candidates: list[int],
    count: int,
) -> dict[int, int]:
    if count < 0 or count > len(candidates):
        raise ValueError(f"Cannot select {count} samples from {len(candidates)} candidates")
    class_counts = Counter(int(labels[index]) for index in candidates)
    if not class_counts and count:
        raise ValueError("No candidates remain")
    total = len(candidates)
    raw = {label: count * size / total for label, size in class_counts.items()}
    quotas = {label: min(int(value), class_counts[label]) for label, value in raw.items()}
    remaining = count - sum(quotas.values())
    order = sorted(
        class_counts,
        key=lambda label: (raw[label] - int(raw[label]), class_counts[label], -label),
        reverse=True,
    )
    while remaining:
        progressed = False
        for label in order:
            if quotas[label] >= class_counts[label]:
                continue
            quotas[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Could not complete stratified quota allocation")
    return quotas


def stratified_take(
    labels: torch.Tensor,
    candidates: list[int],
    count: int,
    seed: int,
    salt: str,
) -> list[int]:
    quotas = allocate_quotas(labels, candidates, count)
    by_class: dict[int, list[int]] = {}
    for index in candidates:
        by_class.setdefault(int(labels[index]), []).append(index)
    selected: list[int] = []
    for label, values in by_class.items():
        ranked = sorted(values, key=lambda index: stable_rank(index, seed, f"{salt}:{label}"))
        selected.extend(ranked[: quotas[label]])
    return sorted(selected, key=lambda index: stable_rank(index, seed, f"{salt}:all"))


def subset_payload(
    source: dict[str, Any],
    splits: dict[str, list[int]],
    *,
    name: str,
    parent: Path,
    seed: int,
) -> dict[str, Any]:
    ordered = [index for split in ("train", "val", "test") for index in splits[split]]
    gather = torch.as_tensor(ordered, dtype=torch.long)
    payload = {
        key: value
        for key, value in source.items()
        if key not in {*PER_SAMPLE_FIELDS, "split_indices"}
    }
    for field in PER_SAMPLE_FIELDS:
        if field not in source:
            if field == "physical_group_ids":
                values = source["group_ids"]
            else:
                raise ValueError(f"Parent MCS manifest is missing {field!r}")
        else:
            values = source[field]
        if isinstance(values, torch.Tensor):
            payload[field] = values[gather].clone()
        else:
            payload[field] = [values[index] for index in ordered]

    cursor = 0
    payload["split_indices"] = {}
    for split in ("train", "val", "test"):
        size = len(splits[split])
        payload["split_indices"][split] = torch.arange(
            cursor, cursor + size, dtype=torch.long
        )
        cursor += size

    metadata = dict(source.get("metadata", {}))
    metadata.update(
        {
            "llm_subset_name": name,
            "llm_subset_parent": str(parent.resolve()),
            "llm_subset_seed": seed,
            "llm_subset_stratified_by": "mcs_class",
            "llm_subset_sampling_policy": (
                "class-stratified sampling within each existing parent split"
            ),
        }
    )
    payload["metadata"] = metadata
    return payload


def histogram(
    labels: torch.Tensor,
    indices: list[int],
    class_names: list[str],
) -> dict[str, int]:
    counts = Counter(int(labels[index]) for index in indices)
    return {name: counts.get(label, 0) for label, name in enumerate(class_names)}


def main() -> None:
    args = parse_args()
    requested = (
        args.formal_train,
        args.formal_val,
        args.formal_test,
        args.smoke_train,
        args.smoke_val,
        args.smoke_test,
    )
    if any(value <= 0 for value in requested):
        raise ValueError("All split sizes must be positive")
    if (
        args.smoke_train > args.formal_train
        or args.smoke_val > args.formal_val
        or args.smoke_test > args.formal_test
    ):
        raise ValueError("Smoke splits must not exceed their formal counterparts")

    source = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(source, dict) or source.get("format") != "csi_mcs_adaptation_v1":
        raise ValueError(f"Unsupported MCS manifest: {args.input}")
    labels = torch.as_tensor(source["labels"], dtype=torch.long)
    class_names = ["Outage", *[str(value) for value in source["mcs_names"]]]
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Parent labels must be a non-empty vector")

    formal: dict[str, list[int]] = {}
    for offset, (split, count) in enumerate(
        (("train", args.formal_train), ("val", args.formal_val), ("test", args.formal_test))
    ):
        parent_indices = torch.as_tensor(
            source["split_indices"][split], dtype=torch.long
        ).tolist()
        if count > len(parent_indices):
            raise ValueError(
                f"Formal {split} requests {count} samples, but the parent split "
                f"contains only {len(parent_indices)}"
            )
        formal[split] = stratified_take(
            labels, parent_indices, count, args.seed + offset, f"formal:{split}"
        )

    smoke: dict[str, list[int]] = {}
    for offset, (split, count) in enumerate(
        (("train", args.smoke_train), ("val", args.smoke_val), ("test", args.smoke_test))
    ):
        smoke[split] = stratified_take(
            labels, formal[split], count, args.seed + 100 + offset, f"smoke:{split}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_path = args.output_dir / "mcs_llm_formal_train10k_val5k_test5k.pt"
    smoke_path = args.output_dir / "mcs_llm_smoke_train1k_val500_test500.pt"
    torch.save(
        subset_payload(
            source,
            formal,
            name="formal_train10k_val5k_test5k",
            parent=args.input,
            seed=args.seed,
        ),
        formal_path,
    )
    torch.save(
        subset_payload(
            source,
            smoke,
            name="smoke_train1k_val500_test500",
            parent=args.input,
            seed=args.seed,
        ),
        smoke_path,
    )

    summary = {
        "parent": str(args.input.resolve()),
        "seed": args.seed,
        "sampling_unit": "power-profile example",
        "parent_split_boundaries_preserved": True,
        "formal": {
            "path": str(formal_path.resolve()),
            "sizes": {name: len(values) for name, values in formal.items()},
            "histograms": {
                name: histogram(labels, values, class_names)
                for name, values in formal.items()
            },
        },
        "smoke": {
            "path": str(smoke_path.resolve()),
            "sizes": {name: len(values) for name, values in smoke.items()},
            "histograms": {
                name: histogram(labels, values, class_names)
                for name, values in smoke.items()
            },
            "nested_in_formal": all(
                set(smoke[name]).issubset(formal[name]) for name in smoke
            ),
        },
    }
    summary_path = args.output_dir / "subset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"summary={summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
