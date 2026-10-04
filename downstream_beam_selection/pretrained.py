from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from data.dataset import PHYSICS_TARGET_NAMES
from downstream_beam_selection.data_io import encoder_kwargs
from downstream_cp_selection.pretrained import build_pretrained_model, los_probability


LOS_K_FEATURE_ORDER = (
    "p_los",
    "predicted_k_factor_db/20",
)
TX_PHYSICS_FEATURE_ORDER = (
    "predicted_first_path_aod_sin",
    "predicted_first_path_aod_cos",
    "p_los",
    "predicted_k_factor_db/20",
    "predicted_n_paths/10",
)


@torch.inference_mode()
def extract_beam_features(
    model,
    batch: dict[str, Any],
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_embedding = model.encode_csi(**encoder_kwargs(batch), normalize=False)
    aod_context = model.encode_first_path_aod_context(
        batch["tokens"],
        batch["beam_positions"],
        batch["token_mask"],
        subcarrier_spacing=batch.get("subcarrier_spacing"),
    )
    # The AoD fusion head shares AoA's delay/LoS context inputs, so use the
    # same inference route as scripts/evaluate.py rather than the legacy fallback.
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
    power_context = None
    if model.use_power_branch:
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
        first_path_delay_context=first_delay_context,
        los_angle_context=los_angle_context,
        first_path_aod_context=aod_context,
    )
    final = outputs["final"]
    p_los = los_probability(
        model, raw_embedding, prototype_keys, checkpoint_args
    ).squeeze(-1)
    k_index = PHYSICS_TARGET_NAMES.index("k_factor_db")
    path_index = PHYSICS_TARGET_NAMES.index("n_paths")
    aod = F.normalize(outputs["first_path_aod_sincos"], dim=-1, eps=1e-6)
    los_k = torch.stack([p_los, final[:, k_index]], dim=-1)
    tx_physics = torch.cat(
        [
            aod,
            p_los.unsqueeze(-1),
            final[:, k_index].unsqueeze(-1),
            final[:, path_index].unsqueeze(-1),
        ],
        dim=-1,
    )
    return (
        F.normalize(raw_embedding, dim=-1),
        torch.nan_to_num(los_k, nan=0.0, posinf=10.0, neginf=-10.0),
        torch.nan_to_num(tx_physics, nan=0.0, posinf=10.0, neginf=-10.0),
    )


def tx_physics_training_warnings(checkpoint_args: dict[str, Any]) -> list[str]:
    aux_weight = float(checkpoint_args.get("aux_regression_weight", 0.0) or 0.0)
    aod_weight = float(checkpoint_args.get("first_path_aod_weight", aux_weight) or 0.0)
    aod_nlos_weight = float(checkpoint_args.get("first_path_aod_nlos_weight", 0.0) or 0.0)
    aux_targets = checkpoint_args.get("aux_regression_targets", ())
    if isinstance(aux_targets, str):
        aux_targets = tuple(part.strip() for part in aux_targets.split(","))
    targets = set(aux_targets or ())
    aod_trained = (aod_weight > 0.0 or aod_nlos_weight > 0.0) and (
        "all" in targets
        or "first_path_aod_az_deg" in targets
        or {
            "first_path_aod_az_sin",
            "first_path_aod_az_cos",
        }.issubset(targets)
    )
    warnings = []
    if not aod_trained:
        warnings.append(
            "checkpoint metadata does not show supervised first-path AoD training; "
            "Full+Tx Physics would use an unverified AoD head"
        )
    return warnings


__all__ = [
    "LOS_K_FEATURE_ORDER",
    "TX_PHYSICS_FEATURE_ORDER",
    "build_pretrained_model",
    "extract_beam_features",
    "tx_physics_training_warnings",
]
