from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_localization.pretrained import _state  # noqa: E402
from scripts.pretrain import deserialize_prototype_keys  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether a CSI2TEXT-PGMT checkpoint contains the trained pieces "
            "needed by downstream localization physics features."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def has_any_prefix(state: dict[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


def count_prefix(state: dict[str, torch.Tensor], prefix: str) -> int:
    return sum(1 for key in state if key.startswith(prefix))


def tensor_shape(state: dict[str, torch.Tensor], key: str) -> list[int] | None:
    value = state.get(key)
    if value is None:
        return None
    return [int(dim) for dim in value.shape]


def prototype_label_counts(checkpoint: dict[str, Any]) -> Counter[str]:
    keys = deserialize_prototype_keys(checkpoint.get("prototype_keys")) or []
    labels = [str(getattr(key, "los_status", "")).lower() for key in keys]
    return Counter(label for label in labels if label)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary in {path}")
    state = _state(checkpoint)
    args = dict(checkpoint.get("args", {}))
    labels = prototype_label_counts(checkpoint)

    csi_weight = state.get("csi.input_proj.spatial_linear.weight")
    d_token = int(csi_weight.shape[1]) if csi_weight is not None else None

    prototype_shape = tensor_shape(state, "prototypes")
    semantic_shape = tensor_shape(state, "semantic_classifier.3.weight")
    prototype_count = prototype_shape[0] if prototype_shape else 0
    semantic_count = semantic_shape[0] if semantic_shape else 0
    label_count = sum(labels.values())
    has_los_and_nlos = labels.get("los", 0) > 0 and labels.get("nlos", 0) > 0
    semantic_weight = float(args.get("semantic_classifier_weight", 0.0) or 0.0)
    semantic_compatible = semantic_count > 0 and semantic_count == label_count
    prototype_compatible = prototype_count > 0 and prototype_count == label_count
    can_compute_p_los = has_los_and_nlos and (
        (semantic_weight > 0.0 and semantic_compatible) or prototype_compatible
    )

    checks = {
        "csi_encoder": csi_weight is not None,
        "prototype_keys_los_nlos": has_los_and_nlos,
        "learned_prototypes_compatible": prototype_compatible,
        "semantic_classifier_compatible": semantic_compatible,
        "semantic_classifier_trained": semantic_weight > 0.0,
        "can_compute_p_los": can_compute_p_los,
        "base_physics_head": has_any_prefix(state, "physics_head."),
        "first_path_delay_context_encoder": has_any_prefix(
            state, "first_path_delay_context_encoder."
        ),
        "first_path_delay_context_head": has_any_prefix(
            state, "first_path_delay_context_head."
        ),
        "first_path_delay_bin_classifier": has_any_prefix(
            state, "first_path_delay_bin_classifier."
        ),
        "first_path_delay_bin_position_head": has_any_prefix(
            state, "first_path_delay_bin_position_head."
        ),
        "first_path_angle_fusion_head": has_any_prefix(
            state, "first_path_angle_fusion_head."
        ),
        "first_path_angle_selector_fusion_head": has_any_prefix(
            state, "first_path_angle_selector_fusion_head."
        ),
        "first_path_angle_context_encoder": has_any_prefix(
            state, "first_path_angle_context_encoder."
        ),
        "los_angle_context_encoder": has_any_prefix(state, "los_angle_context_encoder."),
        "power_branch_args_enabled": bool(args.get("use_power_branch", False)),
        "power_branch_weights": has_any_prefix(state, "power_encoder."),
        "total_received_power_head": has_any_prefix(state, "total_received_power_head."),
        "shared_physics_token": has_any_prefix(state, "shared_physics_token."),
    }
    delay_ready = all(
        checks[name]
        for name in (
            "first_path_delay_context_encoder",
            "first_path_delay_context_head",
            "first_path_delay_bin_classifier",
            "first_path_delay_bin_position_head",
        )
    )
    angle_ready = checks["first_path_angle_fusion_head"] or checks[
        "first_path_angle_selector_fusion_head"
    ]
    physics_features_ready = can_compute_p_los and delay_ready and angle_ready

    return {
        "checkpoint": str(path),
        "state_tensor_count": len(state),
        "d_token": d_token,
        "args_subset": {
            key: args.get(key)
            for key in (
                "token_norm_mode",
                "use_continuous_config_encoding",
                "semantic_classifier_weight",
                "use_power_branch",
                "delay_spread_weight",
                "delay_spread_raw_weight",
                "use_delay_specific_encoder",
                "use_array_invariant_delay_encoder",
                "use_los_angle_context_encoder",
                "use_first_path_angle_context_encoder",
                "use_shared_physics_token",
            )
            if key in args
        },
        "prototype_keys_count": label_count,
        "prototype_los_status_counts": dict(labels),
        "prototype_tensor_shape": prototype_shape,
        "semantic_classifier_shape": semantic_shape,
        "physics_head_output_shape": tensor_shape(state, "physics_head.3.weight"),
        "checks": checks,
        "prefix_counts": {
            prefix: count_prefix(state, prefix)
            for prefix in (
                "csi.",
                "physics_head.",
                "semantic_classifier.",
                "first_path_delay_context_encoder.",
                "first_path_delay_context_head.",
                "first_path_delay_bin_classifier.",
                "first_path_delay_bin_position_head.",
                "first_path_angle_fusion_head.",
                "first_path_angle_selector_fusion_head.",
                "first_path_angle_context_encoder.",
                "los_angle_context_encoder.",
                "power_encoder.",
                "total_received_power_head.",
            )
        },
        "verdict": {
            "embedding_ok": csi_weight is not None,
            "p_los_ok": can_compute_p_los,
            "first_path_delay_ok": delay_ready,
            "first_path_angle_ok": angle_ready,
            "frozen_physics_4d_ok": physics_features_ready,
        },
    }


def print_human(report: dict[str, Any]) -> None:
    verdict = report["verdict"]
    checks = report["checks"]
    print(f"checkpoint: {report['checkpoint']}")
    print(f"state tensors: {report['state_tensor_count']}")
    print(f"d_token: {report['d_token']}")
    print(f"prototype los/nlos counts: {report['prototype_los_status_counts']}")
    print(f"prototype tensor shape: {report['prototype_tensor_shape']}")
    print(f"semantic classifier shape: {report['semantic_classifier_shape']}")
    print(f"physics_head output shape: {report['physics_head_output_shape']}")
    if report["args_subset"]:
        print("args subset:")
        for key, value in report["args_subset"].items():
            print(f"  {key}: {value}")
    print("required checks:")
    for key in (
        "csi_encoder",
        "prototype_keys_los_nlos",
        "learned_prototypes_compatible",
        "semantic_classifier_compatible",
        "semantic_classifier_trained",
        "can_compute_p_los",
        "first_path_delay_context_encoder",
        "first_path_delay_context_head",
        "first_path_delay_bin_classifier",
        "first_path_delay_bin_position_head",
        "first_path_angle_fusion_head",
        "first_path_angle_selector_fusion_head",
        "first_path_angle_context_encoder",
        "los_angle_context_encoder",
        "total_received_power_head",
    ):
        print(f"  {key}: {checks[key]}")
    print("verdict:")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    if not verdict["frozen_physics_4d_ok"]:
        print("recommendation:")
        if verdict["embedding_ok"]:
            print("  You can still run: --models scratch full_embedding")
        if not verdict["p_los_ok"]:
            print("  p_LoS is unavailable or incompatible; physics models may fail.")
        if not verdict["first_path_delay_ok"]:
            print("  trained first-path delay components are missing.")
        if not verdict["first_path_angle_ok"]:
            print("  trained first-path angle components are missing.")
    else:
        print("recommendation:")
        print("  The checkpoint contains the pieces needed for scratch_physics/full_physics.")


def main() -> None:
    args = parse_args()
    report = inspect_checkpoint(args.checkpoint)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
