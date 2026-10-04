from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (  # noqa: E402
    CONFIG_FEATURE_DIM,
    _ensure_continuous_fields,
    _sample_configuration_features,
    normalize_physics_targets,
    physics_raw_values,
)
from downstream_beam_selection.beam_utils import (  # noqa: E402
    NUM_BEAMS,
    observed_beam_mask,
    observed_token_indices,
)


class BeamCorpus:
    def __init__(self, payload: dict[str, Any], manifest_path: Path):
        if payload.get("format") != "csi_beam_selection_v1":
            raise ValueError("Unsupported beam-selection dataset format")
        source_path = Path(payload["source_file"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(source, list):
            raise ValueError(f"Expected a sample list in {source_path}")
        source_indices = torch.as_tensor(payload["source_indices"], dtype=torch.long)
        self.samples = [
            _ensure_continuous_fields(source[int(index)]) for index in source_indices
        ]
        self.group_ids = [str(value) for value in payload["group_ids"]]
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("Beam corpus contains duplicate physical-link group_ids")
        self.los_status = [str(value) for value in payload["los_status"]]
        self.beam_gains_db = torch.as_tensor(payload["beam_gains_db"], dtype=torch.float32)
        linear_gains = payload.get("beam_gains")
        self.beam_gains = (
            torch.as_tensor(linear_gains, dtype=torch.float32)
            if linear_gains is not None
            else torch.pow(10.0, self.beam_gains_db / 10.0)
        )
        self.oracle_beam = torch.as_tensor(payload["oracle_beam"], dtype=torch.long)
        self.oracle_top5 = torch.as_tensor(payload["oracle_top5"], dtype=torch.long)
        self.oracle_gain_db = torch.as_tensor(payload["oracle_gain_db"], dtype=torch.float32)
        self.acceptable_beams_1db = torch.as_tensor(
            payload["acceptable_beams_1db"], dtype=torch.bool
        )
        self.split_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in payload["split_indices"].items()
        }
        self.metadata = dict(payload.get("metadata", {}))
        count = len(self.samples)
        if tuple(self.beam_gains_db.shape) != (count, NUM_BEAMS):
            raise ValueError(f"beam_gains_db must have shape [{count}, {NUM_BEAMS}]")
        if tuple(self.beam_gains.shape) != (count, NUM_BEAMS):
            raise ValueError(f"beam_gains must have shape [{count}, {NUM_BEAMS}]")
        if tuple(self.oracle_beam.shape) != (count,):
            raise ValueError(f"oracle_beam must have shape [{count}]")
        if not torch.equal(self.oracle_beam, self.beam_gains.argmax(dim=1)):
            raise ValueError("oracle_beam does not match beam_gains argmax")
        all_indices = torch.cat(
            [self.split_indices[name] for name in ("train", "val", "test")]
        )
        if len(all_indices) != count or len(torch.unique(all_indices)) != count:
            raise ValueError("train/val/test indices must partition the dataset")
        split_groups = {
            name: {self.group_ids[int(index)] for index in indices}
            for name, indices in self.split_indices.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if split_groups[left] & split_groups[right]:
                raise ValueError(f"Physical-link leakage between {left} and {right}")

    @classmethod
    def load(cls, path: Path) -> "BeamCorpus":
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a dictionary in {path}")
        return cls(payload, path.resolve())


class BeamDataset(Dataset):
    def __init__(
        self,
        corpus: BeamCorpus,
        indices: torch.Tensor,
        *,
        observed_beams: int,
        observation_strategy: str,
        observation_seed: int,
    ):
        self.corpus = corpus
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.observed_beams = int(observed_beams)
        self.observation_strategy = observation_strategy
        self.observation_seed = int(observation_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        sample = self.corpus.samples[index]
        kept_tokens = observed_token_indices(
            sample.beam_positions,
            self.observed_beams,
            strategy=self.observation_strategy,
            seed=self.observation_seed,
            sample_index=index,
        )
        return {
            "index": index,
            "sample": sample,
            "kept_tokens": kept_tokens,
            "observed_beam_mask": observed_beam_mask(kept_tokens),
            "target_class": self.corpus.oracle_beam[index],
            "beam_gains_db": self.corpus.beam_gains_db[index],
            "acceptable_beams_1db": self.corpus.acceptable_beams_1db[index],
        }


class FullCSIBeamDataset(Dataset):
    """Minimal sample view for the full-CSI classifier."""

    def __init__(self, corpus: BeamCorpus, indices: torch.Tensor):
        self.corpus = corpus
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, Any, torch.Tensor]:
        index = int(self.indices[item])
        return index, self.corpus.samples[index], self.corpus.oracle_beam[index]


def collate_full_csi(items: list[tuple[int, Any, torch.Tensor]]) -> dict[str, Any]:
    """Stack complete CSI directly, without sparse-beam masks or per-row copies."""

    indices, samples, targets = zip(*items)
    batch_size = len(samples)
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "tokens": torch.stack([sample.tokens.float() for sample in samples]),
        "beam_positions": torch.stack([sample.beam_positions.float() for sample in samples]),
        "token_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "freq_bin": torch.tensor([sample.freq_bin for sample in samples], dtype=torch.long),
        "bw_bin": torch.tensor([sample.bw_bin for sample in samples], dtype=torch.long),
        "subcarrier_spacing": torch.tensor(
            [sample.subcarrier_spacing_hz for sample in samples], dtype=torch.float32
        ),
        "config_features": torch.stack(
            [_sample_configuration_features(sample) for sample in samples]
        ),
        "antenna_coordinates": torch.stack(
            [sample.antenna_coordinates_wavelengths.float() for sample in samples]
        ),
        "antenna_mask": torch.ones(batch_size, 64, dtype=torch.bool),
        "target_class": torch.stack(targets).long(),
    }


def collate_full_csi_with_physics(
    items: list[tuple[int, Any, torch.Tensor]],
) -> dict[str, Any]:
    """Add training-only physical targets to the otherwise identical CSI batch."""

    batch = collate_full_csi(items)
    samples = [sample for _, sample, _ in items]
    raw_targets = torch.stack([physics_raw_values(sample) for sample in samples])
    batch["physics_targets"], batch["physics_mask"] = normalize_physics_targets(raw_targets)
    batch["los_target"] = torch.tensor(
        [str(sample.semantic_key.los_status).lower() == "los" for sample in samples],
        dtype=torch.float32,
    )
    return batch


def collate_beam(items: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [item["sample"] for item in items]
    batch_size = len(samples)
    max_tokens = max(sample.n_tokens for sample in samples)
    d_token = int(samples[0].tokens.shape[1])
    n_freq = int(samples[0].tokens.shape[2])
    tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=torch.float32)
    positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    config_features = torch.zeros(batch_size, CONFIG_FEATURE_DIM, dtype=torch.float32)
    max_antennas = max(sample.array_rows * sample.array_cols for sample in samples)
    coordinates = torch.zeros(batch_size, max_antennas, 3, dtype=torch.float32)
    antenna_mask = torch.zeros(batch_size, max_antennas, dtype=torch.bool)

    for row, (sample, item) in enumerate(zip(samples, items)):
        count = sample.n_tokens
        positions[row, :count] = sample.beam_positions.float()
        kept = item["kept_tokens"].long()
        tokens[row, kept] = sample.tokens[kept].float()
        token_mask[row, kept] = True
        config_features[row] = _sample_configuration_features(sample)
        sample_coordinates = sample.antenna_coordinates_wavelengths.float()
        coordinates[row, : len(sample_coordinates)] = sample_coordinates
        antenna_mask[row, : len(sample_coordinates)] = True

    return {
        "indices": torch.tensor([item["index"] for item in items], dtype=torch.long),
        "tokens": tokens,
        "beam_positions": positions,
        "token_mask": token_mask,
        "freq_bin": torch.tensor([sample.freq_bin for sample in samples], dtype=torch.long),
        "bw_bin": torch.tensor([sample.bw_bin for sample in samples], dtype=torch.long),
        "subcarrier_spacing": torch.tensor(
            [sample.subcarrier_spacing_hz for sample in samples], dtype=torch.float32
        ),
        "config_features": config_features,
        "antenna_coordinates": coordinates,
        "antenna_mask": antenna_mask,
        "target_class": torch.stack([item["target_class"] for item in items]).long(),
        "beam_gains_db": torch.stack([item["beam_gains_db"] for item in items]).float(),
        "acceptable_beams_1db": torch.stack(
            [item["acceptable_beams_1db"] for item in items]
        ).bool(),
        "observed_beam_mask": torch.stack(
            [item["observed_beam_mask"] for item in items]
        ).bool(),
        "group_ids": [sample.group_id for sample in samples],
    }


def encoder_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "tokens": batch["tokens"],
        "beam_positions": batch["beam_positions"],
        "token_mask": batch["token_mask"],
        "freq_bin": batch["freq_bin"],
        "bw_bin": batch["bw_bin"],
        "subcarrier_spacing": batch["subcarrier_spacing"],
        "config_features": batch["config_features"],
        "antenna_coordinates": batch["antenna_coordinates"],
        "antenna_mask": batch["antenna_mask"],
    }


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
