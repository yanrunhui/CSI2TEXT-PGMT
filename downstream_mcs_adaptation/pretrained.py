from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from data.dataset import PHYSICS_TARGET_NAMES
from scripts.benchmark_full_model import build_model
from scripts.pretrain import deserialize_prototype_keys


PHYSICS_FEATURE_ORDER = (
    "predicted_total_received_power_normalized",
    "predicted_k_factor_normalized",
    "p_los",
    "predicted_delay_spread_normalized",
    "predicted_n_paths_normalized",
)


def build_pretrained_model(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[Any, list, dict[str, Any]]:
    model = build_model(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys")) or []
    d_token = int(model.csi.input_proj.spatial_linear.in_features)
    return model, prototype_keys, {
        "d_token": d_token,
        "token_norm_mode": str(getattr(model.csi, "token_norm_mode", "std")),
        "use_continuous_config_encoding": bool(
            getattr(model.csi, "use_continuous_config_encoding", False)
        ),
        "use_power_branch": bool(getattr(model, "use_power_branch", False)),
    }


def los_probability(
    model: Any,
    raw_embedding: torch.Tensor,
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
) -> torch.Tensor:
    labels = [str(key.los_status).lower() for key in prototype_keys]
    los_mask = torch.tensor(
        [label == "los" for label in labels], device=raw_embedding.device, dtype=torch.bool
    )
    if not labels or not bool(los_mask.any()) or bool(los_mask.all()):
        raise ValueError("Checkpoint prototype keys must contain both LoS and NLoS")
    semantic_trained = float(checkpoint_args.get("semantic_classifier_weight", 0.0) or 0.0) > 0.0
    if (
        semantic_trained
        and model.semantic_classifier is not None
        and model.semantic_classifier[-1].out_features == len(labels)
    ):
        probabilities = torch.softmax(model.predict_semantic(raw_embedding), dim=-1)
    elif model.prototypes is not None and model.prototypes.shape[0] == len(labels):
        probabilities = torch.softmax(
            model.logit_scale.exp()
            * F.normalize(raw_embedding, dim=-1)
            @ model.encode_prototypes(normalize=True).T,
            dim=-1,
        )
    else:
        raise ValueError("Checkpoint has no trained semantic/prototype output for p_LoS")
    return probabilities[:, los_mask].sum(dim=-1, keepdim=True)


def require_mcs_physics_checkpoint(checkpoint: dict[str, Any], model: Any) -> None:
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict) or not any(
        str(name).startswith("total_received_power_head.") for name in state
    ):
        raise ValueError("Checkpoint has no total_received_power_head")
    if not bool(getattr(model, "use_power_branch", False)):
        raise ValueError(
            "Checkpoint has use_power_branch=False, so predicted total power cannot use raw CSI amplitude"
        )


@torch.inference_mode()
def extract_full_mcs_features(
    model: Any,
    batch: dict[str, Any],
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
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
    power_context = model.encode_power_context(
        batch["tokens"],
        batch["token_mask"],
        delay_power_map=batch.get("delay_power_map"),
        delay_power_profile=batch.get("delay_power_profile"),
    )
    outputs = model.predict_physics_components(
        raw_embedding,
        power_context=power_context,
        delay_context=delay_context,
    )
    final = outputs["final"]
    p_los = los_probability(model, raw_embedding, prototype_keys, checkpoint_args).squeeze(-1)
    predicted = torch.stack(
        [
            final[:, PHYSICS_TARGET_NAMES.index("total_received_power_dbw")],
            final[:, PHYSICS_TARGET_NAMES.index("k_factor_db")],
            p_los,
            final[:, PHYSICS_TARGET_NAMES.index("delay_spread_ns")],
            final[:, PHYSICS_TARGET_NAMES.index("n_paths")],
        ],
        dim=-1,
    )
    predicted = torch.nan_to_num(predicted, nan=0.0, posinf=10.0, neginf=-10.0)
    return F.normalize(raw_embedding, dim=-1), predicted


def physics_training_warnings(checkpoint_args: dict[str, Any]) -> list[str]:
    aux_weight = float(checkpoint_args.get("aux_regression_weight", 0.0) or 0.0)
    targets = checkpoint_args.get("aux_regression_targets", ("all",))
    if isinstance(targets, str):
        targets = (targets,)
    targets = set(targets or ())
    all_targets = "all" in targets
    warnings = []
    for name in ("total_received_power_dbw", "k_factor_db", "delay_spread_ns", "n_paths"):
        directly_trained = aux_weight > 0.0 and (all_targets or name in targets)
        if name == "delay_spread_ns":
            directly_trained = directly_trained or float(
                checkpoint_args.get("delay_spread_raw_weight", 0.0) or 0.0
            ) > 0.0
        if not directly_trained:
            warnings.append(f"checkpoint args do not show supervised training for {name}")
    return warnings


__all__ = [
    "PHYSICS_FEATURE_ORDER",
    "build_pretrained_model",
    "extract_full_mcs_features",
    "physics_training_warnings",
    "require_mcs_physics_checkpoint",
]
