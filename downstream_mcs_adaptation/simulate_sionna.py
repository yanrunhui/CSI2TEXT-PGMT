from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.mcs import file_sha256, load_json, validate_uncalibrated_config  # noqa: E402


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DIR = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421"
DEFAULT_TABLE = DEFAULT_REPO / "downstream_mcs_adaptation/mcs_table.json"
RESULT_FIELDS = (
    "group_id",
    "mcs_index",
    "average_snr_db",
    "block_errors",
    "total_blocks",
)
MODULATION_BITS = {"QPSK": 2, "16QAM": 4, "64QAM": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5G-LDPC/QAM link simulation over exported D2Los frequency responses."
    )
    parser.add_argument("--requests", type=Path, default=DEFAULT_DIR / "simulation_requests.csv")
    parser.add_argument("--channels", type=Path, default=DEFAULT_DIR / "effective_channels.npz")
    parser.add_argument("--mcs-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "simulation_results.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--coded-bits", type=int, default=1536)
    parser.add_argument("--decoder-iterations", type=int, default=20)
    parser.add_argument("--batch-blocks", type=int, default=64)
    parser.add_argument("--min-block-errors", type=int, default=20)
    parser.add_argument("--max-blocks", type=int, default=200)
    parser.add_argument(
        "--max-links",
        type=int,
        help="Use only the first N deterministically selected calibration links (pilot runs).",
    )
    parser.add_argument(
        "--snr-stride",
        type=int,
        default=1,
        help="Keep every Nth value from the requested SNR grid.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    if device.type == "cuda" and device.index is None:
        # Sionna 2.x validates explicit device names and rejects plain "cuda".
        device = torch.device("cuda:0")
    return device


def read_requests(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), start=2):
            try:
                rows.append(
                    {
                        "group_id": str(raw["group_id"]),
                        "mcs_index": int(raw["mcs_index"]),
                        "average_snr_db": float(raw["average_snr_db"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid simulation request at row {line}: {raw}") from exc
    if not rows:
        raise ValueError(f"No requests in {path}")
    return rows


def result_key(group_id: str, mcs_index: int, snr_db: float) -> tuple[str, int, str]:
    return group_id, int(mcs_index), f"{float(snr_db):.8f}"


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.is_file():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            keys.add(result_key(raw["group_id"], int(raw["mcs_index"]), float(raw["average_snr_db"])))
    return keys


def request_seed(base_seed: int, row: dict[str, Any]) -> int:
    payload = (
        f"{base_seed}:{row['group_id']}:{row['mcs_index']}:"
        f"{float(row['average_snr_db']):.8f}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def filter_requests(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    group_order = list(dict.fromkeys(row["group_id"] for row in rows))
    if args.max_links is not None:
        if args.max_links <= 0:
            raise ValueError("max-links must be positive")
        permitted_groups = set(group_order[: args.max_links])
        rows = [row for row in rows if row["group_id"] in permitted_groups]
    snr_grid = sorted({float(row["average_snr_db"]) for row in rows})
    permitted_snr = set(snr_grid[:: args.snr_stride])
    rows = [row for row in rows if float(row["average_snr_db"]) in permitted_snr]
    return [row for position, row in enumerate(rows) if position % args.num_shards == args.shard_index]


class MCSLink:
    def __init__(
        self,
        row: dict[str, Any],
        coded_bits: int,
        decoder_iterations: int,
        device: torch.device,
    ):
        modulation = str(row["modulation"]).upper()
        if modulation not in MODULATION_BITS:
            raise ValueError(f"Unsupported modulation {modulation!r}")
        self.bits_per_symbol = MODULATION_BITS[modulation]
        if coded_bits % self.bits_per_symbol:
            raise ValueError(
                f"coded-bits={coded_bits} must be divisible by {self.bits_per_symbol}"
            )
        self.n = coded_bits
        self.k = int(round(float(row["coding_rate"]) * coded_bits))
        actual_rate = self.k / self.n
        if not math.isclose(actual_rate, float(row["coding_rate"]), abs_tol=1e-8):
            raise ValueError(
                f"coded-bits={coded_bits} cannot represent coding rate {row['coding_rate']}"
            )
        self.device = device
        self.encoder = LDPC5GEncoder(
            k=self.k,
            n=self.n,
            num_bits_per_symbol=self.bits_per_symbol,
            device=str(device),
        )
        self.mapper = Mapper("qam", self.bits_per_symbol, device=str(device))
        self.demapper = Demapper("app", "qam", self.bits_per_symbol, device=str(device))
        self.decoder = LDPC5GDecoder(
            self.encoder,
            hard_out=True,
            num_iter=decoder_iterations,
            device=str(device),
        )

    @torch.inference_mode()
    def simulate_batch(
        self,
        response: torch.Tensor,
        average_snr_db: float,
        batch_size: int,
        generator: torch.Generator,
    ) -> int:
        bits = torch.randint(
            0,
            2,
            (batch_size, self.k),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        ).to(torch.float32)
        codeword = self.encoder(bits)
        symbols = self.mapper(codeword)
        repeats = math.ceil(symbols.shape[-1] / response.numel())
        channel = response.repeat(repeats)[: symbols.shape[-1]]
        noise_variance = float(10.0 ** (-average_snr_db / 10.0))
        noise_scale = math.sqrt(noise_variance / 2.0)
        noise = torch.complex(
            torch.randn(
                symbols.shape,
                device=self.device,
                generator=generator,
                dtype=torch.float32,
            ),
            torch.randn(
                symbols.shape,
                device=self.device,
                generator=generator,
                dtype=torch.float32,
            ),
        ) * noise_scale
        received = channel.unsqueeze(0) * symbols + noise

        # For one spatial stream, the unbiased form of the perfect-CSI MMSE
        # estimate is y/h. Its post-equalization noise variance is N0/|h|^2.
        channel_power = channel.abs().square().clamp(min=1e-12)
        equalized = received * channel.conj().unsqueeze(0) / channel_power.unsqueeze(0)
        effective_noise = noise_variance / channel_power
        llr = self.demapper(equalized, effective_noise)
        decoded = self.decoder(llr)
        block_error = (decoded != bits).any(dim=-1)
        return int(block_error.sum().item())


def main() -> None:
    args = parse_args()
    if (
        args.coded_bits <= 0
        or args.decoder_iterations <= 0
        or args.batch_blocks <= 0
        or args.min_block_errors <= 0
        or args.max_blocks <= 0
        or args.min_block_errors > args.max_blocks
        or args.snr_stride <= 0
        or args.num_shards <= 0
        or not 0 <= args.shard_index < args.num_shards
    ):
        raise ValueError("Invalid simulation controls")
    config = load_json(args.mcs_table)
    validate_uncalibrated_config(config)
    device = resolve_device(args.device)
    requests = filter_requests(read_requests(args.requests), args)
    if not requests:
        raise ValueError("Request filters produced no work")

    archive = np.load(args.channels, allow_pickle=False)
    channel_group_ids = [str(value) for value in archive["group_id"]]
    responses = np.asarray(archive["normalized_frequency_response"], dtype=np.complex64)
    if responses.ndim != 2 or responses.shape[0] != len(channel_group_ids):
        raise ValueError("Invalid effective-channel archive")
    channel_lookup = {group_id: index for index, group_id in enumerate(channel_group_ids)}
    if len(channel_lookup) != len(channel_group_ids):
        raise ValueError("Duplicate group IDs in effective-channel archive")
    if any(row["group_id"] not in channel_lookup for row in requests):
        raise ValueError("At least one request has no exported frequency response")
    mcs_rows = {int(row["index"]): row for row in config["mcs"]}
    if any(row["mcs_index"] not in mcs_rows for row in requests):
        raise ValueError("At least one request contains an unknown MCS index")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; pass --resume or choose another output")
    done = completed_keys(args.output) if args.resume else set()
    pending = [
        row
        for row in requests
        if result_key(row["group_id"], row["mcs_index"], row["average_snr_db"]) not in done
    ]
    links = {
        index: MCSLink(mcs_rows[index], args.coded_bits, args.decoder_iterations, device)
        for index in sorted({row["mcs_index"] for row in pending})
    }
    mode = "a" if args.output.exists() and args.resume else "w"
    start = time.time()
    with args.output.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RESULT_FIELDS))
        if mode == "w" or args.output.stat().st_size == 0:
            writer.writeheader()
        for position, row in enumerate(pending, start=1):
            response = torch.as_tensor(
                responses[channel_lookup[row["group_id"]]],
                dtype=torch.complex64,
                device=device,
            )
            mean_power = float(response.abs().square().mean())
            if not math.isclose(mean_power, 1.0, rel_tol=2e-4, abs_tol=2e-4):
                raise ValueError(f"Frequency response is not unit-power for {row['group_id']}")
            generator = torch.Generator(device=device).manual_seed(request_seed(args.seed, row))
            errors = 0
            blocks = 0
            while errors < args.min_block_errors and blocks < args.max_blocks:
                batch_size = min(args.batch_blocks, args.max_blocks - blocks)
                errors += links[row["mcs_index"]].simulate_batch(
                    response,
                    row["average_snr_db"],
                    batch_size,
                    generator,
                )
                blocks += batch_size
            writer.writerow(
                {
                    "group_id": row["group_id"],
                    "mcs_index": row["mcs_index"],
                    "average_snr_db": row["average_snr_db"],
                    "block_errors": errors,
                    "total_blocks": blocks,
                }
            )
            stream.flush()
            if position % 25 == 0 or position == len(pending):
                elapsed = time.time() - start
                print(
                    f"completed={position}/{len(pending)} elapsed_s={elapsed:.1f} "
                    f"last_bler={errors / blocks:.4f}",
                    flush=True,
                )

    summary = {
        "format": "csi_mcs_sionna_simulation_v1",
        "requests": str(args.requests.resolve()),
        "channels": str(args.channels.resolve()),
        "channels_sha256": file_sha256(args.channels),
        "mcs_table": str(args.mcs_table.resolve()),
        "mcs_table_sha256": file_sha256(args.mcs_table),
        "output": str(args.output.resolve()),
        "device": str(device),
        "coded_bits": args.coded_bits,
        "decoder_iterations": args.decoder_iterations,
        "batch_blocks": args.batch_blocks,
        "min_block_errors": args.min_block_errors,
        "max_blocks": args.max_blocks,
        "selected_requests": len(requests),
        "newly_completed_requests": len(pending),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "channel_model": "exported unit-power frequency response, repeated across OFDM symbols",
        "receiver": "perfect-CSI single-stream MMSE, unbiased for soft demapping",
        "fec": "Sionna 5G NR LDPC with rate matching",
        "elapsed_seconds": time.time() - start,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
