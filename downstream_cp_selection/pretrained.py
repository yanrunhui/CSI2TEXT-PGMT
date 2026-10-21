from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PHYSICS_TARGET_NAMES  # noqa: E402
from models.encoder import CSIEncoder  # noqa: E402
from models.model import CSIClip  # noqa: E402
from models.text_encoder import PhysicsTextEncoder  # noqa: E402
from scripts.pretrain import deserialize_prototype_keys  # noqa: E402


PHYSICS_FEATURE_ORDER = (
    "p_los",
    "predicted_delay_spread_ns/200",
    "predicted_n_paths/10",
    "predicted_reflection_path_count/10",
)


def _state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model_state dictionary")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def _attribute_class_sizes(state: dict[str, torch.Tensor]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    pattern = re.compile(r"^attribute_classifiers\.([^.]+)\.3\.weight$")
    for key, value in state.items():
        match = pattern.match(key)
        if match is not None:
            sizes[match.group(1)] = int(value.shape[0])
    return sizes


def build_pretrained_model(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[CSIClip, list, dict[str, Any]]:
    """Reconstruct a frozen CSI2TEXT-PGMT model from checkpoint metadata/state."""

    state = _state(checkpoint)
    args = dict(checkpoint.get("args", {}))
    input_weight = state.get("csi.input_proj.spatial_linear.weight")
    if input_weight is None:
        raise ValueError("Checkpoint does not contain a CSI encoder")
    d_token = int(input_weight.shape[1])
    token_norm_mode = str(args.get("token_norm_mode", "std"))
    use_continuous = bool(
        args.get(
            "use_continuous_config_encoding",
            any(key.startswith("csi.config_enc.") for key in state),
        )
    )
    use_array_invariant = bool(args.get("use_array_invariant_delay_encoder", False))
    use_delay_specific = bool(
        args.get(
            "use_delay_specific_encoder",
            any(
                key.startswith("csi_delay_context_encoder.initial_conv.")
                or key.startswith("csi_delay_context_encoder.multi_scale_convs.")
                for key in state
            ),
        )
    ) and not use_array_invariant
    use_los_angle_context = bool(
        args.get(
            "use_los_angle_context_encoder",
            any(key.startswith("los_angle_context_encoder.") for key in state),
        )
    )
    use_first_angle_context = bool(
        args.get(
            "use_first_path_angle_context_encoder",
            any(key.startswith("first_path_angle_context_encoder.") for key in state),
        )
    )
    use_first_aod_context = bool(
        args.get(
            "use_first_path_aod_context_encoder",
            any(key.startswith("first_path_aod_context_encoder.") for key in state),
        )
    )
    use_shared_physics = bool(
        args.get(
            "use_shared_physics_token",
            any(key.startswith("shared_physics_token.") for key in state),
        )
    )
    use_pdp_aux = any(key.startswith("pdp_latent_aux_encoder.") for key in state)

    prototype_tensor = state.get("prototypes")
    num_prototypes = int(prototype_tensor.shape[0]) if prototype_tensor is not None else None
    semantic_weight = state.get("semantic_classifier.3.weight")
    semantic_classes = int(semantic_weight.shape[0]) if semantic_weight is not None else None
    vocab_weight = state.get("text.tok_embed.weight")
    vocab_size = int(vocab_weight.shape[0]) if vocab_weight is not None else 300
    pos_weight = state.get("text.pos_embed.weight")
    max_text_len = int(pos_weight.shape[0]) if pos_weight is not None else 48
    pdp_out = state.get("pdp_latent_aux_encoder.head.3.weight")
    pdp_hidden = state.get("pdp_latent_aux_encoder.head.1.weight")

    model = CSIClip(
        CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous,
        ),
        PhysicsTextEncoder(vocab_size=vocab_size, max_len=max_text_len),
        num_prototypes=num_prototypes,
        semantic_num_classes=semantic_classes,
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        attribute_num_classes=_attribute_class_sizes(state),
        use_power_branch=bool(args.get("use_power_branch", False)),
        first_path_power_mode=str(args.get("first_path_power_mode", "residual")),
        first_path_power_use_internal_gate=bool(
            args.get("first_path_power_use_internal_gate", True)
        ),
        use_delay_spread_head=bool(
            args.get("delay_spread_weight", 0.0)
            or args.get("delay_spread_raw_weight", 0.0)
        ),
        detach_delay_spread_features=bool(
            args.get("detach_delay_spread_features", False)
        ),
        detach_first_path_delay_features=bool(
            args.get("detach_first_path_delay_features", True)
        ),
        use_delay_specific_encoder=use_delay_specific,
        use_array_invariant_delay_encoder=use_array_invariant,
        use_los_angle_context_encoder=use_los_angle_context,
        use_first_path_angle_context_encoder=use_first_angle_context,
        use_first_path_aod_context_encoder=use_first_aod_context,
        los_angle_context_token_norm_mode=token_norm_mode,
        use_pdp_latent_aux=use_pdp_aux,
        pdp_latent_dim=int(pdp_out.shape[0]) if pdp_out is not None else 16,
        pdp_latent_hidden_dim=int(pdp_hidden.shape[0]) if pdp_hidden is not None else 256,
        freeze_pdp_latent_aux=True,
        use_shared_physics_token=use_shared_physics,
        shared_physics_token_residual_scale=float(
            args.get("shared_physics_token_residual_scale", 1.0)
        ),
    ).to(device)

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = sorted(set(state) - set(compatible))
    missing = sorted(set(current) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    missing_csi = [
        key for key in current if key.startswith("csi.") and key not in compatible
    ]
    if missing_csi:
        raise ValueError(
            "Checkpoint is incompatible with the CSI encoder; missing keys include "
            + ", ".join(missing_csi[:8])
        )
    if skipped or missing:
        warnings.warn(
            f"Loaded {len(compatible)} checkpoint tensors; skipped={len(skipped)}, "
            f"new/unavailable={len(missing)}. CSI encoder compatibility was verified."
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys")) or []
    return model, prototype_keys, {
        "d_token": d_token,
        "token_norm_mode": token_norm_mode,
        "use_continuous_config_encoding": use_continuous,
        "use_power_branch": model.use_power_branch,
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
    }


def los_probability(
    model: CSIClip,
    raw_embedding: torch.Tensor,
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
) -> torch.Tensor:
    labels = [str(key.los_status).lower() for key in prototype_keys]
    los_mask = torch.tensor(
        [label == "los" for label in labels],
        device=raw_embedding.device,
        dtype=torch.bool,
    )
    if not labels or not bool(los_mask.any()) or bool(los_mask.all()):
        raise ValueError(
            "Checkpoint prototype_keys must contain both LoS and NLoS classes to predict p_LoS"
        )
    semantic_is_trained = (
        float(checkpoint_args.get("semantic_classifier_weight", 0.0)) > 0.0
    )
    if (
        semantic_is_trained
        and model.semantic_classifier is not None
        and model.semantic_classifier[-1].out_features == len(labels)
    ):
        class_probability = torch.softmax(model.predict_semantic(raw_embedding), dim=-1)
    elif model.prototypes is not None and model.prototypes.shape[0] == len(labels):
        normalized = F.normalize(raw_embedding, dim=-1)
        prototypes = model.encode_prototypes(normalize=True)
        class_probability = torch.softmax(
            model.logit_scale.exp() * normalized @ prototypes.T, dim=-1
        )
    else:
        raise ValueError("No trained semantic/prototype output can provide p_LoS")
    return class_probability[:, los_mask].sum(dim=-1, keepdim=True)


@torch.inference_mode()
def extract_full_cp_features(
    model,
    batch: dict[str, Any],
    prototype_keys: list,
    checkpoint_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen Full embedding and predicted (never oracle) physics."""

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
    outputs = model.predict_physics_components(
        raw_embedding,
        power_context=power_context,
        delay_context=delay_context,
        first_path_delay_context=first_delay_context,
        los_angle_context=los_angle_context,
        first_path_angle_context=first_angle_context,
    )
    p_los = los_probability(model, raw_embedding, prototype_keys, checkpoint_args)
    final = outputs["final"]
    n_paths_index = PHYSICS_TARGET_NAMES.index("n_paths")
    delay_spread_index = PHYSICS_TARGET_NAMES.index("delay_spread_ns")
    predicted_physics = torch.stack(
        [
            p_los.squeeze(-1),
            final[:, delay_spread_index],
            final[:, n_paths_index],
            outputs["reflection_path_count_prediction"],
        ],
        dim=-1,
    )
    predicted_physics = torch.nan_to_num(
        predicted_physics, nan=0.0, posinf=10.0, neginf=-10.0
    )
    return F.normalize(raw_embedding, dim=-1), predicted_physics


def physics_training_warnings(checkpoint_args: dict[str, Any]) -> list[str]:
    """Flag likely-untrained inputs without rejecting older checkpoint schemas."""

    warnings: list[str] = []
    aux_weight = float(checkpoint_args.get("aux_regression_weight", 0.0) or 0.0)
    aux_targets = checkpoint_args.get("aux_regression_targets", ("all",))
    if isinstance(aux_targets, str):
        aux_targets = (aux_targets,)
    aux_targets = set(aux_targets or ())
    all_aux = "all" in aux_targets
    delay_trained = (
        float(checkpoint_args.get("delay_spread_weight", 0.0) or 0.0) > 0.0
        or float(checkpoint_args.get("delay_spread_raw_weight", 0.0) or 0.0) > 0.0
        or (aux_weight > 0.0 and (all_aux or "delay_spread_ns" in aux_targets))
    )
    path_trained = aux_weight > 0.0 and (all_aux or "n_paths" in aux_targets)
    reflection_trained = (
        float(
            checkpoint_args.get("reflection_path_count_regression_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    if not delay_trained:
        warnings.append("checkpoint args do not show supervised delay-spread training")
    if not path_trained:
        warnings.append("checkpoint args do not show supervised exact path-count training")
    if not reflection_trained:
        warnings.append(
            "checkpoint args do not show supervised reflection-path-count training"
        )
    return warnings


__all__ = [
    "PHYSICS_FEATURE_ORDER",
    "build_pretrained_model",
    "extract_full_cp_features",
    "physics_training_warnings",
]
