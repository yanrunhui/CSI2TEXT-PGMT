from __future__ import annotations

import argparse
import csv
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

from downstream_cp_selection.data_io import (
    CPCorpus,
    CPDataset,
    collate_cp,
    move_tensor_batch,
)
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
from scripts.qwen_csi_prefix_common import CSIPrefixMapper, load_csi_encoder_checkpoint


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
MODEL_DEFAULTS = {
    "qwen35": DEFAULT_REPO / "models/Qwen3.5-2B",
    "deepseek": DEFAULT_REPO / "models/DeepSeek-R1-0528-Qwen3-8B",
    "internlm3": DEFAULT_REPO / "models/InternLM3-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CSI soft-prefix language-model baseline for CP selection."
    )
    parser.add_argument("--model-kind", choices=tuple(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full-lm",
        action="store_true",
        help=(
            "Fine-tune the language backbone as well as the CSI mapper and CP head. "
            "Intended for Qwen3.5-2B; 8B baselines should leave this disabled."
        ),
    )
    parser.add_argument("--csi-checkpoint", type=Path)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-d-model", type=int, default=384)
    parser.add_argument("--encoder-d-clip", type=int, default=256)
    parser.add_argument("--token-norm-mode", choices=("std", "rms", "none"), default="std")
    parser.add_argument("--use-continuous-config-encoding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="Positive value caps optimizer steps for a fast smoke test.",
    )
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        help=(
            "Optional separate learning rate for the CSI encoder. When omitted, "
            "--head-learning-rate is used to preserve the original baseline."
        ),
    )
    parser.add_argument("--lm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--align-prefix-to-token-rms",
        action="store_true",
        help=(
            "RMS-normalize every CSI prefix vector to the approximate RMS of the "
            "language model's native token embeddings."
        ),
    )
    parser.add_argument(
        "--append-query-token",
        action="store_true",
        help=(
            "Append a trainable query initialized from the LM BOS embedding and "
            "classify its final hidden state instead of a raw soft-prefix position."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Write first-step input/prefix/hidden/logit statistics, gradient norms, "
            "and parameter-update checks to diagnostics.json."
        ),
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_compute_dtype(name: str) -> torch.dtype:
    """Resolve precision locally so this entry point stays training-script independent."""

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


def classification_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    corpus: CPCorpus,
) -> dict[str, float]:
    """Compute CP metrics without importing plotting/training dependencies."""

    prediction = prediction.long().cpu()
    target = target.long().cpu()
    indices = indices.long().cpu()
    cp = corpus.cp_lengths_ns.float()
    predicted_cp = cp[prediction]
    target_cp = cp[target]
    selected_outside = corpus.outside_energy_ratios[indices, prediction]
    threshold = corpus.max_outside_energy_ratio
    return {
        "accuracy_pct": float(100.0 * (prediction == target).float().mean()),
        "underprediction_rate_pct": float(
            100.0 * (prediction < target).float().mean()
        ),
        "overprediction_rate_pct": float(
            100.0 * (prediction > target).float().mean()
        ),
        "mean_predicted_cp_ns": float(predicted_cp.mean()),
        "mean_oracle_cp_ns": float(target_cp.mean()),
        "mean_cp_outside_energy_pct": float(100.0 * selected_outside.mean()),
        "p95_cp_outside_energy_pct": float(
            100.0 * torch.quantile(selected_outside.float(), 0.95)
        ),
        "unsafe_outside_energy_rate_pct": float(
            100.0 * (selected_outside > threshold + 1e-7).float().mean()
        ),
    }


def make_loader(
    corpus: CPCorpus,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        CPDataset(corpus, corpus.split_indices[split]),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_cp,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def load_language_model(args: argparse.Namespace, dtype: torch.dtype) -> tuple[nn.Module, int]:
    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.model_kind]
    args.model_path = str(args.model_path)
    if args.full_lm and args.model_kind != "qwen35":
        raise ValueError("--full-lm is restricted to Qwen3.5-2B in this experiment")
    if args.full_lm and args.load_in_4bit:
        raise ValueError("Full-LM training is incompatible with --load-in-4bit")

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

    for parameter in model.parameters():
        parameter.requires_grad = bool(args.full_lm)
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


class LLMCPClassifier(nn.Module):
    def __init__(
        self,
        *,
        language_model: nn.Module,
        mapper: CSIPrefixMapper,
        hidden_size: int,
        num_classes: int,
        full_lm: bool,
        align_prefix_to_token_rms: bool,
        append_query_token: bool,
    ):
        super().__init__()
        self.language_model = language_model
        self.mapper = mapper
        self.full_lm = bool(full_lm)
        self.align_prefix_to_token_rms = bool(align_prefix_to_token_rms)
        self.append_query_token = bool(append_query_token)

        embedding_layer = language_model.get_input_embeddings()
        embedding_weight = embedding_layer.weight
        sample_count = min(int(embedding_weight.shape[0]), 4096)
        sample_indices = torch.linspace(
            0,
            int(embedding_weight.shape[0]) - 1,
            steps=sample_count,
            device=embedding_weight.device,
        ).long()
        with torch.no_grad():
            sampled_embeddings = embedding_weight[sample_indices].float()
            token_embedding_rms = sampled_embeddings.square().mean().sqrt()
        self.register_buffer(
            "token_embedding_rms",
            token_embedding_rms,
            persistent=True,
        )

        if self.append_query_token:
            config = getattr(language_model, "config", None)
            text_config = getattr(config, "text_config", None)
            bos_token_id = getattr(config, "bos_token_id", None)
            if bos_token_id is None and text_config is not None:
                bos_token_id = getattr(text_config, "bos_token_id", None)
            if bos_token_id is None or not 0 <= int(bos_token_id) < embedding_weight.shape[0]:
                bos_token_id = 0
            with torch.no_grad():
                query = embedding_weight[int(bos_token_id)].detach().float().clone()
            self.query_token = nn.Parameter(query.reshape(1, 1, hidden_size))
            self.query_token_id = int(bos_token_id)
        else:
            self.register_parameter("query_token", None)
            self.query_token_id = None
        self.cp_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mapped_prefix = self.mapper(batch)
        prefix = mapped_prefix
        if self.align_prefix_to_token_rms:
            prefix_rms = prefix.float().square().mean(dim=-1, keepdim=True).sqrt()
            prefix = prefix / prefix_rms.clamp_min(1e-6).to(prefix.dtype)
            prefix = prefix * self.token_embedding_rms.to(prefix.dtype)
        if self.query_token is not None:
            query = self.query_token.expand(prefix.shape[0], -1, -1)
            prefix = torch.cat([prefix, query.to(prefix.dtype)], dim=1)
        lm_dtype = next(
            parameter.dtype
            for parameter in self.language_model.parameters()
            if parameter.is_floating_point()
        )
        prefix = prefix.to(dtype=lm_dtype)
        attention_mask = torch.ones(
            prefix.shape[:2], device=prefix.device, dtype=torch.long
        )
        outputs = self.language_model(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = extract_last_hidden(outputs)
        pooled = hidden[:, -1].to(dtype=self.cp_head[0].weight.dtype)
        logits = self.cp_head(pooled)
        if return_diagnostics:
            return logits, {
                "mapped_prefix": mapped_prefix,
                "lm_input_prefix": prefix,
                "pooled_hidden": pooled,
            }
        return logits

    def set_training_mode(self) -> None:
        self.train()
        if not self.full_lm:
            self.language_model.eval()


def batch_tensor_diagnostics(value: torch.Tensor) -> dict[str, float | int]:
    """Describe whether a representation varies across samples in one batch."""

    value = value.detach().float()
    flat = value.reshape(value.shape[0], -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    normalized = F.normalize(flat, dim=-1, eps=1e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    if flat.shape[0] > 1:
        off_diagonal = cosine[~torch.eye(
            flat.shape[0], device=flat.device, dtype=torch.bool
        )]
        mean_pairwise_cosine = float(off_diagonal.mean())
        max_pairwise_cosine = float(off_diagonal.max())
    else:
        mean_pairwise_cosine = float("nan")
        max_pairwise_cosine = float("nan")
    mean_sample_rms = float(flat.square().mean(dim=1).sqrt().mean())
    between_sample_rms = float(centered.square().mean().sqrt())
    return {
        "batch_size": int(flat.shape[0]),
        "element_mean": float(flat.mean()),
        "element_std": float(flat.std(unbiased=False)),
        "mean_sample_rms": mean_sample_rms,
        "between_sample_rms": between_sample_rms,
        "between_to_sample_rms_ratio": between_sample_rms
        / max(mean_sample_rms, 1e-12),
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "max_pairwise_cosine": max_pairwise_cosine,
        "finite_fraction": float(torch.isfinite(flat).float().mean()),
    }


def gradient_group_diagnostics(
    model: LLMCPClassifier,
    scaler_scale: float,
) -> dict[str, dict[str, float | int]]:
    groups = {
        "csi_encoder": model.mapper.csi_encoder,
        "projector": model.mapper.projector,
        "cp_head": model.cp_head,
    }
    if model.query_token is not None:
        groups["query_token"] = model
    report: dict[str, dict[str, float | int]] = {}
    for group_name, module in groups.items():
        squared_norm = 0.0
        parameter_count = 0
        gradient_parameter_count = 0
        nonzero_gradient_count = 0
        for parameter_name, parameter in module.named_parameters():
            if group_name == "query_token" and parameter_name != "query_token":
                continue
            if not parameter.requires_grad:
                continue
            parameter_count += parameter.numel()
            if parameter.grad is None:
                continue
            gradient_parameter_count += parameter.numel()
            gradient = parameter.grad.detach().float() / max(float(scaler_scale), 1.0)
            norm = float(torch.linalg.vector_norm(gradient))
            squared_norm += norm * norm
            nonzero_gradient_count += int(torch.count_nonzero(gradient))
        report[group_name] = {
            "trainable_parameter_count": int(parameter_count),
            "gradient_parameter_count": int(gradient_parameter_count),
            "nonzero_gradient_count": int(nonzero_gradient_count),
            "gradient_l2_norm": math.sqrt(squared_norm),
        }
    return report


def diagnostic_parameter_snapshots(
    model: LLMCPClassifier,
) -> dict[str, torch.Tensor]:
    candidates: dict[str, tuple[str, nn.Module]] = {
        "csi_encoder": ("mapper.csi_encoder.", model.mapper.csi_encoder),
        "projector": ("mapper.projector.", model.mapper.projector),
        "cp_head": ("cp_head.", model.cp_head),
    }
    snapshots: dict[str, torch.Tensor] = {}
    for group_name, (_, module) in candidates.items():
        eligible = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.numel() <= 100_000
        ]
        if not eligible:
            continue
        name, parameter = eligible[-1]
        snapshots[f"{group_name}:{name}"] = parameter.detach().float().cpu().clone()
    if model.query_token is not None:
        snapshots["query_token"] = model.query_token.detach().float().cpu().clone()
    return snapshots


def parameter_update_diagnostics(
    model: LLMCPClassifier,
    before: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    parameter_groups = {
        "csi_encoder": dict(model.mapper.csi_encoder.named_parameters()),
        "projector": dict(model.mapper.projector.named_parameters()),
        "cp_head": dict(model.cp_head.named_parameters()),
    }

    report: dict[str, dict[str, float]] = {}
    for name, initial in before.items():
        if name == "query_token":
            if model.query_token is None:
                raise RuntimeError("Diagnostic query token unexpectedly disappeared")
            updated = model.query_token.detach().float().cpu()
        else:
            group_name, parameter_name = name.split(":", 1)
            updated = (
                parameter_groups[group_name][parameter_name]
                .detach()
                .float()
                .cpu()
            )
        delta = updated - initial
        report[name] = {
            "initial_l2_norm": float(torch.linalg.vector_norm(initial)),
            "update_l2_norm": float(torch.linalg.vector_norm(delta)),
            "update_max_abs": float(delta.abs().max()),
        }
    return report


def write_diagnostics(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("llm_cp_diagnostics=" + json.dumps(report, sort_keys=True), flush=True)


@torch.no_grad()
def evaluate(
    model: LLMCPClassifier,
    loader: DataLoader,
    corpus: CPCorpus,
    device: torch.device,
    autocast_context,
) -> dict[str, Any]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_tensor_batch(batch, device)
        with autocast_context():
            logits = model(batch)
            loss = F.cross_entropy(logits.float(), batch["target_class"])
        count = int(batch["target_class"].numel())
        total_loss += float(loss) * count
        total_count += count
        logits_all.append(logits.float().cpu())
        targets.append(batch["target_class"].cpu())
        indices.append(batch["indices"].cpu())
    logits = torch.cat(logits_all)
    target = torch.cat(targets)
    index = torch.cat(indices)
    probability = torch.softmax(logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    return {
        "loss": total_loss / max(total_count, 1),
        "metrics": classification_metrics(prediction, target, index, corpus),
        "prediction": prediction,
        "probability": probability,
        "target": target,
        "indices": index,
    }


def cpu_trainable_state(model: LLMCPClassifier) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: LLMCPClassifier, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise ValueError(f"Saved parameters are absent from model: {missing[:10]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(
                value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )


def save_confusion(path: Path, result: dict[str, Any], cp_ns: torch.Tensor) -> None:
    class_count = len(cp_ns)
    matrix = torch.zeros(class_count, class_count, dtype=torch.long)
    for target, prediction in zip(result["target"], result["prediction"]):
        matrix[int(target), int(prediction)] += 1
    labels = [f"{float(value):g} ns" for value in cp_ns]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLM CP baseline requires CUDA")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch-size, epochs, and patience must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = CPCorpus.load(args.data)
    train_loader = make_loader(corpus, "train", args, True, 11)
    val_loader = make_loader(corpus, "val", args, False, 12)
    test_loader = make_loader(corpus, "test", args, False, 13)

    dtype = resolve_compute_dtype(args.compute_dtype)
    language_model, hidden_size = load_language_model(args, dtype)
    device = next(language_model.parameters()).device
    first_sample = corpus.samples[0]
    mapper = CSIPrefixMapper(
        d_token=int(first_sample.tokens.shape[1]),
        qwen_hidden_size=hidden_size,
        prefix_length=args.prefix_length,
        d_model=args.encoder_d_model,
        d_clip=args.encoder_d_clip,
        projector_hidden_dim=args.projector_hidden_dim,
        token_norm_mode=args.token_norm_mode,
        use_continuous_config_encoding=args.use_continuous_config_encoding,
    ).to(device)
    csi_load = None
    if args.csi_checkpoint is not None:
        csi_load = load_csi_encoder_checkpoint(mapper, str(args.csi_checkpoint))
    model = LLMCPClassifier(
        language_model=language_model,
        mapper=mapper,
        hidden_size=hidden_size,
        num_classes=len(corpus.cp_lengths_ns),
        full_lm=args.full_lm,
        align_prefix_to_token_rms=args.align_prefix_to_token_rms,
        append_query_token=args.append_query_token,
    )
    # The language model was already placed by Transformers' device_map.  Moving
    # the whole wrapper again is both wasteful and unsupported for 4-bit modules.
    model.cp_head.to(device)

    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("mapper.csi_encoder.")
    ]
    adapter_head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("language_model.")
        and not name.startswith("mapper.csi_encoder.")
    ]
    lm_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("language_model.")
    ]
    encoder_learning_rate = (
        args.head_learning_rate
        if args.encoder_learning_rate is None
        else args.encoder_learning_rate
    )
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": encoder_learning_rate}
        )
    if adapter_head_parameters:
        parameter_groups.append(
            {"params": adapter_head_parameters, "lr": args.head_learning_rate}
        )
    if lm_parameters:
        parameter_groups.append({"params": lm_parameters, "lr": args.lm_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "llm_cp_training="
        + json.dumps(
            {
                "model_kind": args.model_kind,
                "model_path": str(args.model_path),
                "mode": "full_lm" if args.full_lm else "frozen_lm",
                "data": str(args.data),
                "split_sizes": {
                    name: len(indices) for name, indices in corpus.split_indices.items()
                },
                "prefix_length": args.prefix_length,
                "align_prefix_to_token_rms": args.align_prefix_to_token_rms,
                "append_query_token": args.append_query_token,
                "query_token_id": model.query_token_id,
                "token_embedding_rms": float(model.token_embedding_rms),
                "encoder_learning_rate": encoder_learning_rate,
                "adapter_head_learning_rate": args.head_learning_rate,
                "trainable_parameters": trainable_count,
                "total_parameters": total_count,
                "csi_checkpoint_load": csi_load,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if dtype in (torch.float16, torch.bfloat16)
        else (lambda: nullcontext())
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    best_epoch = 0
    stale = 0
    global_step = 0
    history: list[dict[str, float]] = []
    start_time = time.time()
    stop = False
    optimizer.zero_grad(set_to_none=True)
    diagnostics_path = args.output_dir / "diagnostics.json"
    diagnostics_report: dict[str, Any] = {
        "model_kind": args.model_kind,
        "training_mode": "full_lm" if args.full_lm else "frozen_lm",
        "csi_checkpoint_load": csi_load,
        "alignment": {
            "align_prefix_to_token_rms": args.align_prefix_to_token_rms,
            "append_query_token": args.append_query_token,
            "query_token_id": model.query_token_id,
            "token_embedding_rms": float(model.token_embedding_rms),
        },
    }
    diagnostic_snapshots = (
        diagnostic_parameter_snapshots(model) if args.diagnostics else {}
    )
    diagnostic_forward_recorded = False
    diagnostic_update_recorded = False

    for epoch in range(1, args.epochs + 1):
        model.set_training_mode()
        running_loss = 0.0
        micro_batches = 0
        for batch in train_loader:
            batch = move_tensor_batch(batch, device)
            with autocast_context():
                if args.diagnostics and not diagnostic_forward_recorded:
                    logits, diagnostic_features = model(
                        batch, return_diagnostics=True
                    )
                else:
                    logits = model(batch)
                    diagnostic_features = None
                loss = F.cross_entropy(logits.float(), batch["target_class"])
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if args.diagnostics and not diagnostic_forward_recorded:
                target_histogram = torch.bincount(
                    batch["target_class"].detach().cpu(),
                    minlength=len(corpus.cp_lengths_ns),
                )
                prediction_histogram = torch.bincount(
                    logits.detach().argmax(dim=-1).cpu(),
                    minlength=len(corpus.cp_lengths_ns),
                )
                diagnostics_report["first_batch"] = {
                    "target_histogram": target_histogram.tolist(),
                    "prediction_histogram": prediction_histogram.tolist(),
                    "input_tokens": batch_tensor_diagnostics(batch["tokens"]),
                    "mapped_prefix": batch_tensor_diagnostics(
                        diagnostic_features["mapped_prefix"]
                    ),
                    "lm_input_prefix": batch_tensor_diagnostics(
                        diagnostic_features["lm_input_prefix"]
                    ),
                    "pooled_hidden": batch_tensor_diagnostics(
                        diagnostic_features["pooled_hidden"]
                    ),
                    "logits": batch_tensor_diagnostics(logits),
                    "loss": float(loss.detach()),
                }
                diagnostics_report["first_backward"] = gradient_group_diagnostics(
                    model,
                    scaler.get_scale(),
                )
                write_diagnostics(diagnostics_path, diagnostics_report)
                diagnostic_forward_recorded = True
            running_loss += float(loss.detach())
            micro_batches += 1
            if micro_batches % args.gradient_accumulation_steps != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.diagnostics and not diagnostic_update_recorded:
                diagnostics_report["first_optimizer_step"] = (
                    parameter_update_diagnostics(model, diagnostic_snapshots)
                )
                write_diagnostics(diagnostics_path, diagnostics_report)
                diagnostic_update_recorded = True
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={running_loss / max(micro_batches, 1):.6f}",
                    flush=True,
                )
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop = True
                break

        if micro_batches % args.gradient_accumulation_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.diagnostics and not diagnostic_update_recorded:
                diagnostics_report["first_optimizer_step"] = (
                    parameter_update_diagnostics(model, diagnostic_snapshots)
                )
                write_diagnostics(diagnostics_path, diagnostics_report)
                diagnostic_update_recorded = True

        val_result = evaluate(model, val_loader, corpus, device, autocast_context)
        row = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            "train_loss": running_loss / max(micro_batches, 1),
            "val_loss": float(val_result["loss"]),
            **val_result["metrics"],
        }
        history.append(row)
        print(
            f"validation epoch={epoch} step={global_step} "
            f"loss={row['val_loss']:.6f} acc={row['accuracy_pct']:.2f}% "
            f"under={row['underprediction_rate_pct']:.2f}%",
            flush=True,
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_state = cpu_trainable_state(model)
            stale = 0
        else:
            stale += 1
        if stop or stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    restore_trainable_state(model, best_state)
    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {
            "trainable_state": best_state,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "cp_lengths_ns": corpus.cp_lengths_ns,
        },
        checkpoint_path,
    )
    test_result = evaluate(model, test_loader, corpus, device, autocast_context)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        indices=test_result["indices"].numpy(),
        predicted_class=test_result["prediction"].numpy(),
        target_class=test_result["target"].numpy(),
        predicted_probabilities=test_result["probability"].numpy(),
        predicted_cp_ns=corpus.cp_lengths_ns[test_result["prediction"]].numpy(),
        target_cp_ns=corpus.cp_lengths_ns[test_result["target"]].numpy(),
    )
    save_confusion(
        args.output_dir / "test_confusion_matrix.csv",
        test_result,
        corpus.cp_lengths_ns,
    )
    summary = {
        "model_kind": args.model_kind,
        "model_path": str(args.model_path),
        "training_mode": "full_lm" if args.full_lm else "frozen_lm",
        "metrics": test_result["metrics"],
        "test_loss": test_result["loss"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "global_steps": global_step,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "elapsed_seconds": time.time() - start_time,
        "split_sizes": {
            name: len(indices) for name, indices in corpus.split_indices.items()
        },
        "cp_lengths_ns": corpus.cp_lengths_ns.tolist(),
        "history": history,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
