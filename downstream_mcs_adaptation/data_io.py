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
    DELAY_POWER_MAP_SHAPE,
    DELAY_POWER_PROFILE_BINS,
    _ensure_continuous_fields,
    _sample_configuration_features,
)


def samples_from_pt(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    samples = payload if isinstance(payload, list) else payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected samples in {path}")
    return samples


class MCSCorpus:
    def __init__(self, payload: dict[str, Any], manifest_path: Path):
        if payload.get("format") != "csi_mcs_adaptation_v1":
            raise ValueError("Unsupported MCS-adaptation dataset format")
        source_paths = {
            name: Path(value) for name, value in dict(payload["source_files"]).items()
        }
        source_paths = {
            name: path if path.is_absolute() else manifest_path.parent / path
            for name, path in source_paths.items()
        }
        sources = {name: samples_from_pt(path) for name, path in source_paths.items()}
        self.samples = []
        for source_name, local_index in payload["sample_refs"]:
            if source_name not in sources:
                raise ValueError(f"Unknown sample source {source_name!r}")
            index = int(local_index)
            if not 0 <= index < len(sources[source_name]):
                raise IndexError(f"Sample reference {source_name}[{index}] is out of range")
            self.samples.append(_ensure_continuous_fields(sources[source_name][index]))

        count = len(self.samples)
        self.labels = torch.as_tensor(payload["labels"], dtype=torch.long)
        self.oracle_mcs = torch.as_tensor(payload["oracle_mcs"], dtype=torch.long)
        self.average_snr_db = torch.as_tensor(payload["average_snr_db"], dtype=torch.float32)
        self.effective_snr_per_mcs = torch.as_tensor(
            payload["effective_snr_per_mcs"], dtype=torch.float32
        )
        self.estimated_bler_per_mcs = torch.as_tensor(
            payload["estimated_bler_per_mcs"], dtype=torch.float32
        )
        self.spectral_efficiency = torch.as_tensor(
            payload["spectral_efficiency"], dtype=torch.float32
        )
        self.mcs_names = [str(value) for value in payload["mcs_names"]]
        self.group_ids = [str(value) for value in payload["group_ids"]]
        self.physical_group_ids = [
            str(value) for value in payload.get("physical_group_ids", self.group_ids)
        ]
        self.power_offset_db = torch.as_tensor(
            payload.get("power_offset_db", torch.zeros(count)), dtype=torch.float32
        )
        self.transmit_power_dbm = torch.as_tensor(
            payload.get(
                "transmit_power_dbm",
                torch.full((count,), float(payload["system"]["transmit_power_dbm"])),
            ),
            dtype=torch.float32,
        )
        self.system = dict(payload["system"])
        self.calibrated_mcs = list(payload["calibrated_mcs"])
        self.metadata = dict(payload.get("metadata", {}))
        self.split_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in payload["split_indices"].items()
        }
        mcs_count = len(self.mcs_names)
        if self.labels.shape != (count,) or self.oracle_mcs.shape != (count,):
            raise ValueError("MCS labels have the wrong length")
        if (
            len(self.group_ids) != count
            or len(set(self.group_ids)) != count
            or len(self.physical_group_ids) != count
            or self.power_offset_db.shape != (count,)
            or self.transmit_power_dbm.shape != (count,)
        ):
            raise ValueError("Power-profile or group metadata has the wrong shape")
        if not bool(torch.isfinite(self.power_offset_db).all()) or not bool(
            torch.isfinite(self.transmit_power_dbm).all()
        ):
            raise ValueError("Transmit powers and power offsets must be finite")
        base_transmit_power_dbm = float(
            self.metadata.get(
                "base_transmit_power_dbm", self.system["transmit_power_dbm"]
            )
        )
        expected_offsets = self.transmit_power_dbm - base_transmit_power_dbm
        if not torch.allclose(self.power_offset_db, expected_offsets, atol=1e-5, rtol=0.0):
            raise ValueError(
                "power_offset_db must equal transmit_power_dbm minus the base transmit power"
            )
        if (
            len(torch.unique(self.transmit_power_dbm)) > 1
            and self.metadata.get("power_profile_input_mode")
            != "explicit_transmit_power"
        ):
            raise ValueError(
                "Multi-power data was not generated for explicit Tx-power input; "
                "regenerate it with the current prepare_data.py"
            )
        if self.effective_snr_per_mcs.shape != (count, mcs_count):
            raise ValueError("effective_snr_per_mcs has the wrong shape")
        if self.estimated_bler_per_mcs.shape != (count, mcs_count):
            raise ValueError("estimated_bler_per_mcs has the wrong shape")
        if self.spectral_efficiency.shape != (mcs_count,):
            raise ValueError("spectral_efficiency has the wrong shape")
        if count and (int(self.labels.min()) < 0 or int(self.labels.max()) > mcs_count):
            raise ValueError("Class labels must map Outage to 0 and MCS m to m+1")
        if not torch.equal(self.labels, self.oracle_mcs + 1):
            raise ValueError("labels and oracle_mcs mapping is inconsistent")
        all_indices = torch.cat([self.split_indices[name] for name in ("train", "val", "test")])
        if (
            len(all_indices) != count
            or len(torch.unique(all_indices)) != count
            or not torch.equal(torch.sort(all_indices).values, torch.arange(count))
        ):
            raise ValueError("train/val/test must partition all samples exactly once")
        groups = {
            name: {self.physical_group_ids[int(index)] for index in indices}
            for name, indices in self.split_indices.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if groups[left] & groups[right]:
                raise ValueError(f"Physical-link leakage between {left} and {right}")

    @classmethod
    def load(cls, path: Path) -> "MCSCorpus":
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a dictionary in {path}")
        return cls(payload, path.resolve())


class MCSDataset(Dataset):
    def __init__(self, corpus: MCSCorpus, indices: torch.Tensor):
        self.corpus = corpus
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "index": index,
            "sample": self.corpus.samples[index],
            "label": self.corpus.labels[index],
            "group_id": self.corpus.group_ids[index],
            "physical_group_id": self.corpus.physical_group_ids[index],
            "transmit_power_dbm": self.corpus.transmit_power_dbm[index],
            "power_offset_db": self.corpus.power_offset_db[index],
        }


def collate_mcs(items: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [item["sample"] for item in items]
    batch_size = len(samples)
    max_tokens = max(int(sample.n_tokens) for sample in samples)
    d_token = int(samples[0].tokens.shape[1])
    n_freq = int(samples[0].tokens.shape[2])
    if any(tuple(sample.tokens.shape[1:]) != (d_token, n_freq) for sample in samples):
        raise ValueError("All samples in a batch must share d_token and n_freq")

    tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=torch.float32)
    beam_positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    config_features = torch.zeros(batch_size, CONFIG_FEATURE_DIM, dtype=torch.float32)
    delay_power_map = torch.zeros(batch_size, *DELAY_POWER_MAP_SHAPE, dtype=torch.float32)
    delay_power_profile = torch.zeros(batch_size, DELAY_POWER_PROFILE_BINS, dtype=torch.float32)
    max_antennas = max(int(sample.array_rows) * int(sample.array_cols) for sample in samples)
    antenna_coordinates = torch.zeros(batch_size, max_antennas, 3, dtype=torch.float32)
    antenna_mask = torch.zeros(batch_size, max_antennas, dtype=torch.bool)
    for row, sample in enumerate(samples):
        count = int(sample.n_tokens)
        # Keep propagation CSI exactly as extracted. The operating transmit power
        # is an explicit downstream input and must not shift the pretrained
        # encoder away from its training distribution.
        tokens[row, :count] = sample.tokens.float()
        beam_positions[row, :count] = sample.beam_positions.float()
        token_mask[row, :count] = True
        config_features[row] = _sample_configuration_features(sample)
        coordinates = sample.antenna_coordinates_wavelengths.float()
        antenna_coordinates[row, : len(coordinates)] = coordinates
        antenna_mask[row, : len(coordinates)] = True
        delay_power_map[row] = sample.delay_power_map.float()
        delay_power_profile[row] = sample.delay_power_profile.float()
    return {
        "indices": torch.tensor([item["index"] for item in items], dtype=torch.long),
        "tokens": tokens,
        "beam_positions": beam_positions,
        "token_mask": token_mask,
        "freq_bin": torch.tensor([sample.freq_bin for sample in samples], dtype=torch.long),
        "bw_bin": torch.tensor([sample.bw_bin for sample in samples], dtype=torch.long),
        "subcarrier_spacing": torch.tensor(
            [sample.subcarrier_spacing_hz for sample in samples], dtype=torch.float32
        ),
        "config_features": config_features,
        "antenna_coordinates": antenna_coordinates,
        "antenna_mask": antenna_mask,
        "delay_power_map": delay_power_map,
        "delay_power_profile": delay_power_profile,
        "target_class": torch.stack([item["label"] for item in items]).long(),
        "group_ids": [str(item["group_id"]) for item in items],
        "physical_group_ids": [str(item["physical_group_id"]) for item in items],
        "transmit_power_dbm": torch.stack(
            [torch.as_tensor(item["transmit_power_dbm"]) for item in items]
        ).float(),
        "power_offset_db": torch.stack(
            [torch.as_tensor(item["power_offset_db"]) for item in items]
        ).float(),
    }


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
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
