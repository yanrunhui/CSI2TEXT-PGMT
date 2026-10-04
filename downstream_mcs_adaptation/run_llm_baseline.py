"""Scratch-like CSI-to-LLM baselines for MCS link adaptation.

The propagation CSI remains unchanged across transmit-power profiles.  A random,
trainable CSI encoder and scale-sensitive raw-CSI statistics are concatenated
with explicit transmit power, projected to a short continuous prefix, processed
by a causal language-model backbone, and classified by an MCS output head.

This entry point is self-contained with respect to the conventional MCS runner:
it does not import ``run_experiment.py`` and it has no matplotlib dependency.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from downstream_mcs_adaptation.data_io import (
    MCSCorpus,
    MCSDataset,
    collate_mcs,
    encoder_kwargs,
    move_tensor_batch,
)
from models.encoder import CSIEncoder
from scripts.csi_prefix_language_model import (
    assert_qwen35_auxiliary_modules_frozen,
    freeze_qwen35_auxiliary_modules,
    language_model_hidden_size as qwen_hidden_size,
    load_base_language_model,
    set_language_model_use_cache,
)
from scripts.deepseek_csi_prefix_common import validate_deepseek_model_path
from scripts.internlm3_csi_prefix_common import (
    assert_soft_prefix_interface,
    language_model_hidden_size as internlm_hidden_size,
    load_internlm3_model,
    set_internlm3_use_cache,
)


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
MODEL_DEFAULTS = {
    "qwen35": DEFAULT_REPO / "models/Qwen3.5-2B",
    "deepseek": DEFAULT_REPO / "models/DeepSeek-R1-0528-Qwen3-8B",
    "internlm3": DEFAULT_REPO / "models/InternLM3-8B-Instruct",
}
TX_POWER_FEATURE_SCALE_DBM = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=tuple(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full-lm",
        action="store_true",
        help="Fine-tune the text backbone. This experiment restricts it to Qwen3.5-2B.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--compute-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-d-model", type=int, default=384)
    parser.add_argument("--encoder-d-clip", type=int, default=256)
    parser.add_argument("--token-norm-mode", choices=("std", "rms", "none"), default="std")
    parser.add_argument("--use-continuous-config-encoding", action="store_true")
    parser.add_argument("--align-prefix-to-token-rms", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--lm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--safety-penalty-weight", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="Positive value caps total optimizer steps for a quick pipeline check.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.prefix_length,
        args.projector_hidden_dim,
        args.encoder_d_model,
        args.encoder_d_clip,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.patience,
        args.head_learning_rate,
        args.lm_learning_rate,
        args.grad_clip,
        args.cpu_threads,
        args.log_every,
    )
    if any(value <= 0 for value in positive) or args.num_workers < 0:
        raise ValueError("Sizes, rates, and intervals must be positive; num-workers is nonnegative")
    if args.full_lm and args.model_kind != "qwen35":
        raise ValueError("--full-lm is restricted to Qwen3.5-2B")
    if args.full_lm and args.load_in_4bit:
        raise ValueError("Full-parameter training is incompatible with --load-in-4bit")
    if not args.full_lm and args.model_kind == "qwen35":
        print("WARNING: Qwen3.5 is frozen because --full-lm was not supplied", flush=True)
    if args.device != "cuda:0":
        raise ValueError(
            "Use CUDA_VISIBLE_DEVICES to select the physical GPU and pass --device cuda:0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The MCS LLM baseline requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_compute_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 was requested but the active GPU does not support it")
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name != "auto":
        raise ValueError(f"Unsupported compute dtype: {name}")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_language_model(
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> tuple[nn.Module, int]:
    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.model_kind]
    args.model_path = str(args.model_path)
    if args.model_kind == "internlm3":
        model = load_internlm3_model(args, dtype)
        assert_soft_prefix_interface(model)
        hidden_size = internlm_hidden_size(model)
        set_internlm3_use_cache(model, False)
    else:
        if args.model_kind == "deepseek":
            validate_deepseek_model_path(args.model_path)
        model = load_base_language_model(args, dtype)
        hidden_size = qwen_hidden_size(model)
        set_language_model_use_cache(model, False)

    model.requires_grad_(bool(args.full_lm))
    if args.model_kind == "qwen35":
        frozen = freeze_qwen35_auxiliary_modules(model)
        assert_qwen35_auxiliary_modules_frozen(model)
        print("qwen35_frozen_auxiliary=" + json.dumps(frozen, sort_keys=True), flush=True)
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise ValueError(f"{model.__class__.__name__} has no gradient checkpointing support")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, hidden_size


def extract_last_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states):
        value = hidden_states[-1]
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            return value
    if isinstance(outputs, dict):
        for key in ("language_model_output", "model_output", "text_model_output"):
            if key in outputs:
                try:
                    return extract_last_hidden(outputs[key])
                except ValueError:
                    pass
    for key in ("language_model_output", "model_output", "text_model_output"):
        value = getattr(outputs, key, None)
        if value is not None:
            try:
                return extract_last_hidden(value)
            except ValueError:
                pass
    raise ValueError(
        f"{type(outputs).__name__} did not expose a [batch, sequence, hidden] hidden state"
    )


class RawAmplitudeStatistics(nn.Module):
    """The same five scale-sensitive base-CSI features used by Scratch MCS."""

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        weights = token_mask.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        raw_power = (tokens * weights).square()
        valid_beams = token_mask.sum(dim=1).to(tokens.dtype).clamp(min=1.0)
        elements = (valid_beams * tokens.shape[2] * tokens.shape[3]).clamp(min=1.0)
        mean_power = raw_power.sum(dim=(1, 2, 3)) / elements
        peak_power = raw_power.amax(dim=(1, 2, 3))
        beam_power = raw_power.mean(dim=(2, 3))
        beam_mean = beam_power.sum(dim=1) / valid_beams
        beam_peak = beam_power.amax(dim=1)
        frequency_power = raw_power.sum(dim=(1, 2)) / (
            valid_beams[:, None] * tokens.shape[2]
        )
        frequency_db = 10.0 * torch.log10(frequency_power.clamp(min=1e-30))
        q10 = torch.quantile(frequency_db, 0.1, dim=1)
        q90 = torch.quantile(frequency_db, 0.9, dim=1)
        return torch.stack(
            (
                10.0 * torch.log10(mean_power.clamp(min=1e-30)) / 100.0,
                10.0 * torch.log10(peak_power.clamp(min=1e-30)) / 100.0,
                10.0
                * torch.log10((beam_peak / beam_mean.clamp(min=1e-30)).clamp(min=1e-30))
                / 20.0,
                frequency_db.std(dim=1, correction=0) / 20.0,
                (q90 - q10) / 20.0,
            ),
            dim=-1,
        )


class LLMMCSClassifier(nn.Module):
    def __init__(
        self,
        *,
        language_model: nn.Module,
        d_token: int,
        hidden_size: int,
        num_classes: int,
        prefix_length: int,
        projector_hidden_dim: int,
        encoder_d_model: int,
        encoder_d_clip: int,
        token_norm_mode: str,
        use_continuous_config_encoding: bool,
        full_lm: bool,
        align_prefix_to_token_rms: bool,
    ):
        super().__init__()
        self.language_model = language_model
        self.full_lm = bool(full_lm)
        self.prefix_length = int(prefix_length)
        self.align_prefix_to_token_rms = bool(align_prefix_to_token_rms)
        self.csi_encoder = CSIEncoder(
            d_token=d_token,
            d_model=encoder_d_model,
            d_clip=encoder_d_clip,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous_config_encoding,
        )
        self.raw_amplitude = RawAmplitudeStatistics()
        scratch_feature_dim = encoder_d_clip + 5 + 1
        self.prefix_projector = nn.Sequential(
            nn.Linear(scratch_feature_dim, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, prefix_length * hidden_size),
        )
        self.prefix_position = nn.Parameter(torch.zeros(1, prefix_length, hidden_size))

        embeddings = language_model.get_input_embeddings().weight
        sample_count = min(int(embeddings.shape[0]), 4096)
        sample_indices = torch.linspace(
            0, int(embeddings.shape[0]) - 1, steps=sample_count, device=embeddings.device
        ).long()
        with torch.no_grad():
            token_embedding_rms = embeddings[sample_indices].float().square().mean().sqrt()
        self.register_buffer("token_embedding_rms", token_embedding_rms, persistent=True)

        config = getattr(language_model, "config", None)
        text_config = getattr(config, "text_config", None)
        bos_token_id = getattr(config, "bos_token_id", None)
        if bos_token_id is None and text_config is not None:
            bos_token_id = getattr(text_config, "bos_token_id", None)
        if bos_token_id is None or not 0 <= int(bos_token_id) < embeddings.shape[0]:
            bos_token_id = 0
        with torch.no_grad():
            query = embeddings[int(bos_token_id)].detach().float().clone()
        self.query_token = nn.Parameter(query.reshape(1, 1, hidden_size))
        self.query_token_id = int(bos_token_id)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        nn.init.normal_(self.prefix_position, std=0.02)

    def train(self, mode: bool = True) -> "LLMMCSClassifier":
        super().train(mode)
        self.language_model.train(mode and self.full_lm)
        return self

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        csi = F.normalize(self.csi_encoder(**encoder_kwargs(batch)), dim=-1)
        raw_amplitude = self.raw_amplitude(batch["tokens"], batch["token_mask"])
        transmit_power = (
            batch["transmit_power_dbm"].float().unsqueeze(-1) / TX_POWER_FEATURE_SCALE_DBM
        )
        scratch_features = torch.cat(
            (csi.float(), raw_amplitude.float(), transmit_power), dim=-1
        )
        prefix = self.prefix_projector(scratch_features).view(
            scratch_features.shape[0], self.prefix_length, -1
        )
        prefix = prefix + self.prefix_position
        if self.align_prefix_to_token_rms:
            prefix_rms = prefix.float().square().mean(dim=-1, keepdim=True).sqrt()
            prefix = prefix / prefix_rms.clamp_min(1e-6).to(prefix.dtype)
            prefix = prefix * self.token_embedding_rms.to(prefix.dtype)
        query = self.query_token.expand(prefix.shape[0], -1, -1)
        embeds = torch.cat((prefix, query.to(prefix.dtype)), dim=1)
        lm_dtype = next(
            parameter.dtype
            for parameter in self.language_model.parameters()
            if parameter.is_floating_point()
        )
        embeds = embeds.to(dtype=lm_dtype)
        attention_mask = torch.ones(
            embeds.shape[:2], device=embeds.device, dtype=torch.long
        )
        outputs = self.language_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = extract_last_hidden(outputs)
        return self.output_head(hidden[:, -1].float())

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state = {
            "prefix_position": self.prefix_position.detach().cpu().clone(),
            "query_token": self.query_token.detach().cpu().clone(),
            "token_embedding_rms": self.token_embedding_rms.detach().cpu().clone(),
        }
        for module_name in ("csi_encoder", "prefix_projector", "output_head"):
            module = getattr(self, module_name)
            state.update(
                {
                    f"{module_name}.{name}": value.detach().cpu().clone()
                    for name, value in module.state_dict().items()
                }
            )
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name in ("prefix_position", "query_token", "token_embedding_rms"):
                destination = getattr(self, name)
                destination.copy_(
                    state[name].to(device=destination.device, dtype=destination.dtype)
                )
        for module_name in ("csi_encoder", "prefix_projector", "output_head"):
            prefix = f"{module_name}."
            module_state = {
                name[len(prefix) :]: value
                for name, value in state.items()
                if name.startswith(prefix)
            }
            getattr(self, module_name).load_state_dict(module_state, strict=True)


def make_loader(
    corpus: MCSCorpus,
    split: str,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    return DataLoader(
        MCSDataset(corpus, corpus.split_indices[split]),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_mcs,
        generator=torch.Generator().manual_seed(args.seed + seed_offset),
        **({"prefetch_factor": 1} if args.num_workers else {}),
    )


def class_weights(corpus: MCSCorpus, mode: str) -> torch.Tensor | None:
    labels = corpus.labels[corpus.split_indices["train"]]
    counts = torch.bincount(labels, minlength=len(corpus.mcs_names) + 1).float()
    print(f"training_class_counts={counts.long().tolist()}", flush=True)
    if bool((counts == 0).any()):
        missing = torch.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training subset has no samples for classes {missing}")
    if mode == "none":
        return None
    return labels.numel() / (len(counts) * counts)


def training_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    safety_penalty_weight: float,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.float(), target, weight=weights)
    if safety_penalty_weight > 0.0:
        classes = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        aggressive_distance = (classes[None, :] - target[:, None]).clamp(min=0.0)
        expected_distance = (torch.softmax(logits, dim=-1) * aggressive_distance).sum(dim=-1)
        loss = loss + safety_penalty_weight * expected_distance.mean()
    return loss


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def selection_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    corpus: MCSCorpus,
) -> dict[str, float]:
    prediction = prediction.long().cpu()
    target = target.long().cpu()
    indices = indices.long().cpu()
    count = len(prediction)
    predicted_mcs = prediction - 1
    oracle_mcs = target - 1
    predicted_active = predicted_mcs >= 0
    oracle_active = oracle_mcs >= 0
    selected_bler = torch.zeros(count, dtype=torch.float32)
    selected_efficiency = torch.zeros(count, dtype=torch.float32)
    if bool(predicted_active.any()):
        selected_bler[predicted_active] = corpus.estimated_bler_per_mcs[
            indices[predicted_active], predicted_mcs[predicted_active]
        ]
        selected_efficiency[predicted_active] = corpus.spectral_efficiency[
            predicted_mcs[predicted_active]
        ]
    oracle_bler = torch.zeros(count, dtype=torch.float32)
    oracle_efficiency = torch.zeros(count, dtype=torch.float32)
    if bool(oracle_active.any()):
        oracle_bler[oracle_active] = corpus.estimated_bler_per_mcs[
            indices[oracle_active], oracle_mcs[oracle_active]
        ]
        oracle_efficiency[oracle_active] = corpus.spectral_efficiency[oracle_mcs[oracle_active]]
    target_bler = float(corpus.system["target_bler"])
    violation = predicted_active & (selected_bler > target_bler + 1e-7)
    goodput = selected_efficiency * (1.0 - selected_bler)
    oracle_goodput = oracle_efficiency * (1.0 - oracle_bler)
    model_goodput = float(goodput.mean())
    oracle_mean_goodput = float(oracle_goodput.mean())
    outage_target = target == 0
    outage_prediction = prediction == 0
    outage_tp = int((outage_target & outage_prediction).sum())
    outage_fn = int((outage_target & ~outage_prediction).sum())
    outage_fp = int((~outage_target & outage_prediction).sum())
    active_selected_bler = (
        float(selected_bler[predicted_active].mean())
        if bool(predicted_active.any())
        else math.nan
    )
    return {
        "accuracy_pct": float(100.0 * (prediction == target).float().mean()),
        "aggressive_selection_rate_pct": float(100.0 * (prediction > target).float().mean()),
        "conservative_selection_rate_pct": float(100.0 * (prediction < target).float().mean()),
        "mean_absolute_mcs_error": float((prediction - target).abs().float().mean()),
        "bler_violation_rate_pct": float(100.0 * violation.float().mean()),
        "mean_selected_bler_pct": float(100.0 * selected_bler.mean()),
        "mean_active_transmission_bler_pct": 100.0 * active_selected_bler,
        "mean_spectral_efficiency_bit_s_hz": float(selected_efficiency.mean()),
        "mean_goodput_bit_s_hz": model_goodput,
        "oracle_goodput_bit_s_hz": oracle_mean_goodput,
        "goodput_ratio_pct": 100.0 * safe_ratio(model_goodput, oracle_mean_goodput),
        "predicted_outage_rate_pct": float(100.0 * outage_prediction.float().mean()),
        "true_outage_rate_pct": float(100.0 * outage_target.float().mean()),
        "outage_recall_pct": 100.0 * safe_ratio(outage_tp, outage_tp + outage_fn),
        "outage_false_alarm_rate_pct": 100.0
        * safe_ratio(outage_fp, int((~outage_target).sum())),
    }


def autocast_factory(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return lambda: nullcontext()


@torch.inference_mode()
def evaluate(
    model: LLMMCSClassifier,
    loader: DataLoader,
    corpus: MCSCorpus,
    device: torch.device,
    weights: torch.Tensor | None,
    args: argparse.Namespace,
    autocast_context,
) -> dict[str, Any]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    loss_sum = 0.0
    count_sum = 0
    device_weights = weights.to(device) if weights is not None else None
    for batch_number, cpu_batch in enumerate(loader, start=1):
        batch = move_tensor_batch(cpu_batch, device)
        with autocast_context():
            logits = model(batch)
            loss = training_loss(
                logits,
                batch["target_class"],
                device_weights,
                args.safety_penalty_weight,
            )
        count = len(batch["target_class"])
        loss_sum += float(loss) * count
        count_sum += count
        logits_all.append(logits.float().cpu())
        targets.append(batch["target_class"].cpu())
        indices.append(batch["indices"].cpu())
        if batch_number % 100 == 0 or batch_number == len(loader):
            print(f"inference_batches={batch_number}/{len(loader)}", flush=True)
    logits = torch.cat(logits_all)
    target = torch.cat(targets)
    index = torch.cat(indices)
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    return {
        "loss": loss_sum / max(count_sum, 1),
        "metrics": selection_metrics(prediction, target, index, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target,
        "indices": index,
    }


def make_optimizer(args: argparse.Namespace, model: LLMMCSClassifier):
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("language_model.")
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": adapter_parameters, "lr": args.head_learning_rate}
    ]
    lm_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("language_model.")
    ]
    if lm_parameters:
        parameter_groups.append({"params": lm_parameters, "lr": args.lm_learning_rate})
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError("--optimizer adamw8bit requires bitsandbytes") from error
        return bnb.optim.AdamW8bit(parameter_groups, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def save_best_checkpoint(
    model: LLMMCSClassifier,
    args: argparse.Namespace,
    *,
    epoch: int,
    val_loss: float,
) -> None:
    if args.full_lm:
        model.language_model.save_pretrained(
            args.output_dir / "best_backbone",
            safe_serialization=True,
            max_shard_size="4GB",
        )
    torch.save(
        {
            "format": "csi_mcs_llm_adapter_v1",
            "model_kind": args.model_kind,
            "model_path": str(args.model_path),
            "full_lm": bool(args.full_lm),
            "epoch": epoch,
            "best_val_loss": val_loss,
            "adapter_state_dict": model.adapter_state_dict(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        args.output_dir / "best_adapter.pt",
    )


def restore_best_model(
    model: LLMMCSClassifier,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> LLMMCSClassifier:
    saved = torch.load(
        args.output_dir / "best_adapter.pt", map_location="cpu", weights_only=False
    )
    if args.full_lm:
        del model.language_model
        gc.collect()
        torch.cuda.empty_cache()
        reload_args = copy.copy(args)
        reload_args.model_path = args.output_dir / "best_backbone"
        reload_args.full_lm = False
        reload_args.gradient_checkpointing = False
        language_model, _ = load_language_model(reload_args, dtype)
        language_model.requires_grad_(False)
        model.language_model = language_model
        model.full_lm = False
    model.load_adapter_state_dict(saved["adapter_state_dict"])
    return model


def train(
    model: LLMMCSClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    corpus: MCSCorpus,
    device: torch.device,
    weights: torch.Tensor | None,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> tuple[int, float, list[dict[str, float]], int]:
    optimizer = make_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    autocast_context = autocast_factory(dtype)
    device_weights = weights.to(device) if weights is not None else None
    best_val_loss = math.inf
    best_epoch = 0
    stale = 0
    global_step = 0
    history: list[dict[str, float]] = []
    stop = False

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        num_batches = len(train_loader)
        for micro_step, cpu_batch in enumerate(train_loader):
            batch = move_tensor_batch(cpu_batch, device)
            with autocast_context():
                logits = model(batch)
                loss = training_loss(
                    logits,
                    batch["target_class"],
                    device_weights,
                    args.safety_penalty_weight,
                )
                window_start = (
                    micro_step // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                window_size = min(
                    args.gradient_accumulation_steps, num_batches - window_start
                )
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            batch_count = len(batch["target_class"])
            running_loss += float(loss.detach()) * batch_count
            sample_count += batch_count

            update = (
                (micro_step + 1) % args.gradient_accumulation_steps == 0
                or micro_step + 1 == num_batches
            )
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    ],
                    max_norm=args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch} optimizer_step={global_step} "
                        f"micro_batch={micro_step + 1}/{num_batches} "
                        f"running_loss={running_loss / max(sample_count, 1):.6f}",
                        flush=True,
                    )
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    stop = True
                    break

        val_result = evaluate(
            model, val_loader, corpus, device, weights, args, autocast_context
        )
        record = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(sample_count, 1),
            "val_loss": float(val_result["loss"]),
            "val_accuracy_pct": val_result["metrics"]["accuracy_pct"],
            "val_bler_violation_rate_pct": val_result["metrics"][
                "bler_violation_rate_pct"
            ],
            "val_goodput_ratio_pct": val_result["metrics"]["goodput_ratio_pct"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print("epoch_result=" + json.dumps(record, sort_keys=True), flush=True)
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best_epoch = epoch
            stale = 0
            save_best_checkpoint(
                model, args, epoch=epoch, val_loss=best_val_loss
            )
        else:
            stale += 1
        if stop or stale >= args.patience:
            break

    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was saved")
    restore_best_model(model, args, dtype)
    return best_epoch, best_val_loss, history, global_step


def save_confusion(
    path: Path,
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_names: list[str],
) -> None:
    classes = len(class_names)
    flat = target.long() * classes + prediction.long()
    matrix = torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name, *row])


def save_predictions(
    path: Path,
    result: dict[str, Any],
    corpus: MCSCorpus,
) -> None:
    class_names = ["Outage", *corpus.mcs_names]
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "index",
            "group_id",
            "physical_group_id",
            "transmit_power_dbm",
            "target_class",
            "target_name",
            "predicted_class",
            "predicted_name",
            *[f"probability_{name}" for name in class_names],
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row, index_tensor in enumerate(result["indices"]):
            index = int(index_tensor)
            target = int(result["target"][row])
            prediction = int(result["prediction"][row])
            record: dict[str, Any] = {
                "index": index,
                "group_id": corpus.group_ids[index],
                "physical_group_id": corpus.physical_group_ids[index],
                "transmit_power_dbm": float(corpus.transmit_power_dbm[index]),
                "target_class": target,
                "target_name": class_names[target],
                "predicted_class": prediction,
                "predicted_name": class_names[prediction],
            }
            for name, value in zip(class_names, result["probability"][row].tolist()):
                record[f"probability_{name}"] = value
            writer.writerow(record)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    dtype = resolve_compute_dtype(args.compute_dtype)

    corpus = MCSCorpus.load(args.data)
    loaders = {
        "train": make_loader(corpus, "train", args, shuffle=True, seed_offset=11),
        "val": make_loader(corpus, "val", args, shuffle=False, seed_offset=12),
        "test": make_loader(corpus, "test", args, shuffle=False, seed_offset=13),
    }
    weights = class_weights(corpus, args.class_weight)
    language_model, hidden_size = load_language_model(args, dtype)
    device = next(language_model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Language model was not placed on CUDA: {device}")
    first_sample = corpus.samples[0]
    model = LLMMCSClassifier(
        language_model=language_model,
        d_token=int(first_sample.tokens.shape[1]),
        hidden_size=hidden_size,
        num_classes=len(corpus.mcs_names) + 1,
        prefix_length=args.prefix_length,
        projector_hidden_dim=args.projector_hidden_dim,
        encoder_d_model=args.encoder_d_model,
        encoder_d_clip=args.encoder_d_clip,
        token_norm_mode=args.token_norm_mode,
        use_continuous_config_encoding=args.use_continuous_config_encoding,
        full_lm=args.full_lm,
        align_prefix_to_token_rms=args.align_prefix_to_token_rms,
    )
    model.csi_encoder.to(device)
    model.prefix_projector.to(device)
    model.prefix_position.data = model.prefix_position.data.to(device)
    model.query_token.data = model.query_token.data.to(device)
    model.output_head.to(device)
    model.token_embedding_rms.data = model.token_embedding_rms.data.to(device)

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    run_info = {
        "model_kind": args.model_kind,
        "model_path": str(args.model_path),
        "backbone_training": "full" if args.full_lm else "frozen",
        "load_in_4bit": args.load_in_4bit,
        "device": str(device),
        "data": str(args.data.resolve()),
        "split_sizes": {
            name: len(indices) for name, indices in corpus.split_indices.items()
        },
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "explicit_transmit_power": True,
        "raw_amplitude_statistics": True,
        "csi_encoder_initialization": "random",
        "query_token_id": model.query_token_id,
        "token_embedding_rms": float(model.token_embedding_rms),
    }
    print("llm_mcs_training=" + json.dumps(run_info, sort_keys=True), flush=True)

    started = time.time()
    best_epoch, best_val_loss, history, optimizer_steps = train(
        model,
        loaders["train"],
        loaders["val"],
        corpus,
        device,
        weights,
        args,
        dtype,
    )
    test_result = evaluate(
        model,
        loaders["test"],
        corpus,
        device,
        weights,
        args,
        autocast_factory(dtype),
    )
    class_names = ["Outage", *corpus.mcs_names]
    save_predictions(args.output_dir / "test_predictions.csv", test_result, corpus)
    save_confusion(
        args.output_dir / "confusion_matrix.csv",
        test_result["prediction"],
        test_result["target"],
        class_names,
    )
    write_history(args.output_dir / "history.csv", history)

    metrics_row: dict[str, Any] = {
        "model": args.model_kind,
        "backbone_training": "full" if args.full_lm else "frozen",
        "quantization": "4bit" if args.load_in_4bit else "none",
        "train_size": len(corpus.split_indices["train"]),
        "val_size": len(corpus.split_indices["val"]),
        "test_size": len(corpus.split_indices["test"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "optimizer_steps": optimizer_steps,
        **test_result["metrics"],
    }
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_row))
        writer.writeheader()
        writer.writerow(metrics_row)
    summary = {
        **metrics_row,
        **run_info,
        "test_loss": test_result["loss"],
        "history": history,
        "elapsed_seconds": time.time() - started,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics_row, indent=2, ensure_ascii=False), flush=True)
    print(f"outputs={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
