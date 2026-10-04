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
    / "downstream_cp_selection/artifacts/d2los_50k_upa8x8_nf128_bw100mhz_cp_labels.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO / "downstream_cp_selection/artifacts/llm_cp_subsets"
ARRAY_FIELDS = (
    "labels",
    "outside_energy_ratios",
    "max_excess_delay_ns",
    "energy_delay_ns",
    "path_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nested, stratified CP manifests for LLM smoke and formal runs."
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


def allocate_quotas(labels: torch.Tensor, candidates: list[int], count: int) -> dict[int, int]:
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
        made_progress = False
        for label in order:
            if quotas[label] >= class_counts[label]:
                continue
            quotas[label] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            raise RuntimeError("Could not complete stratified quota allocation")
    return quotas


def stratified_take(
    labels: torch.Tensor,
    candidates: list[int],
    count: int,
    seed: int,
    salt: str,
) -> tuple[list[int], list[int]]:
    quotas = allocate_quotas(labels, candidates, count)
    by_class: dict[int, list[int]] = {}
    for index in candidates:
        by_class.setdefault(int(labels[index]), []).append(index)
    selected: list[int] = []
    for label, values in by_class.items():
        ranked = sorted(values, key=lambda index: stable_rank(index, seed, f"{salt}:{label}"))
        selected.extend(ranked[: quotas[label]])
    selected = sorted(selected, key=lambda index: stable_rank(index, seed, f"{salt}:all"))
    selected_set = set(selected)
    remaining = [index for index in candidates if index not in selected_set]
    return selected, remaining


def subset_payload(
    source: dict[str, Any],
    splits: dict[str, list[int]],
    *,
    name: str,
    parent: Path,
    seed: int,
) -> dict[str, Any]:
    ordered = [index for split in ("train", "val", "test") for index in splits[split]]
    remapped: dict[str, torch.Tensor] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        size = len(splits[split])
        remapped[split] = torch.arange(cursor, cursor + size, dtype=torch.long)
        cursor += size

    payload = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "sample_refs",
            "group_ids",
            "split_indices",
            *ARRAY_FIELDS,
        }
    }
    payload["sample_refs"] = [source["sample_refs"][index] for index in ordered]
    payload["group_ids"] = [source["group_ids"][index] for index in ordered]
    for field in ARRAY_FIELDS:
        values = torch.as_tensor(source[field])
        payload[field] = values[torch.as_tensor(ordered, dtype=torch.long)].clone()
    payload["split_indices"] = remapped
    metadata = dict(source.get("metadata", {}))
    metadata.update(
        {
            "llm_subset_name": name,
            "llm_subset_parent": str(parent.resolve()),
            "llm_subset_seed": seed,
            "llm_subset_stratified_by": "cp_class",
            "llm_subset_resplit_policy": (
                "stratified sampling within each parent train/val/test split"
            ),
        }
    )
    payload["metadata"] = metadata
    return payload


def histogram(labels: torch.Tensor, indices: list[int], cp_ns: torch.Tensor) -> dict[str, int]:
    counts = Counter(int(labels[index]) for index in indices)
    return {
        f"{float(cp_ns[label]):g}ns": counts.get(label, 0)
        for label in range(len(cp_ns))
    }


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
    if not isinstance(source, dict) or source.get("format") != "csi_cp_selection_v1":
        raise ValueError(f"Unsupported CP manifest: {args.input}")
    labels = torch.as_tensor(source["labels"], dtype=torch.long)
    cp_ns = torch.as_tensor(source["cp_lengths_ns"], dtype=torch.float32)
    formal_total = args.formal_train + args.formal_val + args.formal_test
    if formal_total > len(labels):
        raise ValueError(
            f"Formal splits require {formal_total} samples, parent has {len(labels)}"
        )

    formal: dict[str, list[int]] = {}
    for offset, (split, count) in enumerate(
        (("train", args.formal_train), ("val", args.formal_val), ("test", args.formal_test))
    ):
        parent_indices = torch.as_tensor(
            source["split_indices"][split], dtype=torch.long
        ).tolist()
        if count > len(parent_indices):
            raise ValueError(
                f"Formal {split} split requests {count} samples, but parent "
                f"{split} contains only {len(parent_indices)}"
            )
        formal[split], _ = stratified_take(
            labels, parent_indices, count, args.seed + offset, f"formal:{split}"
        )

    smoke: dict[str, list[int]] = {}
    for offset, (split, count) in enumerate(
        (("train", args.smoke_train), ("val", args.smoke_val), ("test", args.smoke_test))
    ):
        smoke[split], _ = stratified_take(
            labels, formal[split], count, args.seed + 100 + offset, f"smoke:{split}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_path = args.output_dir / "cp_llm_formal_train10k_val5k_test5k.pt"
    smoke_path = args.output_dir / "cp_llm_smoke_train1k_val500_test500.pt"
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
        "formal": {
            "path": str(formal_path.resolve()),
            "sizes": {name: len(values) for name, values in formal.items()},
            "histograms": {
                name: histogram(labels, values, cp_ns) for name, values in formal.items()
            },
        },
        "smoke": {
            "path": str(smoke_path.resolve()),
            "sizes": {name: len(values) for name, values in smoke.items()},
            "histograms": {
                name: histogram(labels, values, cp_ns) for name, values in smoke.items()
            },
            "nested_in_formal": all(
                set(smoke[name]).issubset(formal[name]) for name in smoke
            ),
        },
    }
    summary_path = args.output_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
