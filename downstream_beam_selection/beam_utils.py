from __future__ import annotations

import math
from typing import Any

import torch


BEAM_GRID_SHAPE = (8, 8)
PATCH_SHAPE = (2, 2)
NUM_BEAMS = BEAM_GRID_SHAPE[0] * BEAM_GRID_SHAPE[1]
BEAMS_PER_TOKEN = PATCH_SHAPE[0] * PATCH_SHAPE[1]
NUM_TOKENS = NUM_BEAMS // BEAMS_PER_TOKEN
FFT_ORDER = "unshifted"


def beam_response_from_tokens(
    tokens: torch.Tensor,
    *,
    grid_shape: tuple[int, int] = BEAM_GRID_SHAPE,
    patch_shape: tuple[int, int] = PATCH_SHAPE,
) -> torch.Tensor:
    """Undo 2-D patch tokenization and return complex [beam, subcarrier] CSI."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [token, channel, frequency], got {tokens.shape}")
    rows, cols = grid_shape
    patch_rows, patch_cols = patch_shape
    if rows % patch_rows or cols % patch_cols:
        raise ValueError("Beam grid must be divisible by the patch shape")
    token_rows = rows // patch_rows
    token_cols = cols // patch_cols
    expected_tokens = token_rows * token_cols
    patch_size = patch_rows * patch_cols
    expected_channels = 2 * patch_size
    if tuple(tokens.shape[:2]) != (expected_tokens, expected_channels):
        raise ValueError(
            "Expected UPA-8x8 tokens with shape "
            f"[{expected_tokens}, {expected_channels}, Nf], got {tuple(tokens.shape)}"
        )

    frequency_count = int(tokens.shape[-1])
    response = torch.zeros(rows, cols, frequency_count, dtype=torch.complex64)
    for token_index in range(expected_tokens):
        token_row, token_col = divmod(token_index, token_cols)
        row_start = token_row * patch_rows
        col_start = token_col * patch_cols
        real = tokens[token_index, :patch_size].float().reshape(
            patch_rows, patch_cols, frequency_count
        )
        imag = tokens[token_index, patch_size:].float().reshape(
            patch_rows, patch_cols, frequency_count
        )
        response[
            row_start : row_start + patch_rows,
            col_start : col_start + patch_cols,
        ] = torch.complex(real, imag)
    return response.reshape(rows * cols, frequency_count)


def normalized_beam_power(tokens: torch.Tensor) -> torch.Tensor:
    """Return unit-norm DFT beam power with shape [64, Nf]."""

    response = beam_response_from_tokens(tokens)
    return response.abs().square() / float(NUM_BEAMS)


def beam_label_record(tokens: torch.Tensor, acceptable_loss_db: float = 1.0) -> dict[str, Any]:
    power = normalized_beam_power(tokens)
    gains = power.mean(dim=-1)
    if not bool(torch.isfinite(gains).all()):
        raise ValueError("Beam gains contain NaN or Inf")
    if float(gains.max()) <= 0.0:
        raise ValueError("CSI has no positive beam gain; oracle beam is undefined")
    gains_db = 10.0 * torch.log10(gains.clamp_min(torch.finfo(torch.float32).tiny))
    order = torch.argsort(gains, descending=True, stable=True)
    oracle = int(gains.argmax())
    acceptable = gains_db >= gains_db[oracle] - float(acceptable_loss_db)
    return {
        "beam_gains": gains.float(),
        "beam_gains_db": gains_db.float(),
        "oracle_beam": oracle,
        "oracle_top5": order[:5].long(),
        "oracle_gain_db": float(gains_db[oracle]),
        "acceptable_beams_1db": acceptable.bool(),
    }


def validate_beam_sample(sample: Any) -> None:
    array_type = str(getattr(sample, "array_type", ""))
    rows = int(getattr(sample, "array_rows", 0) or 0)
    cols = int(getattr(sample, "array_cols", 0) or 0)
    if (array_type, rows, cols) != ("UPA", 8, 8):
        raise ValueError(
            f"Beam task requires UPA-8x8 samples, got {array_type}-{rows}x{cols}"
        )
    beam_response_from_tokens(sample.tokens)


def verify_beam_unpack_roundtrip(tokens: torch.Tensor) -> None:
    """Check the documented 2x2-patch/row-major beam-ID mapping."""
    response = beam_response_from_tokens(tokens).reshape(8, 8, -1)
    for token_row in range(4):
        for token_col in range(4):
            chunk = response[
                token_row * 2 : (token_row + 1) * 2,
                token_col * 2 : (token_col + 1) * 2,
            ]
            repacked = torch.stack((chunk.real, chunk.imag), dim=0).reshape(8, -1)
            token_index = token_row * 4 + token_col
            if not torch.equal(repacked, tokens[token_index].float()):
                raise ValueError(f"Beam unpack/repack mismatch at patch token {token_index}")


def observed_token_indices(
    beam_positions: torch.Tensor,
    observed_beams: int,
    *,
    strategy: str,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    if observed_beams not in {8, 16, 32, 64}:
        raise ValueError("observed_beams must be one of 8, 16, 32, 64")
    token_count = int(beam_positions.shape[0])
    keep_count = observed_beams // BEAMS_PER_TOKEN
    if token_count != NUM_TOKENS:
        raise ValueError(f"Expected {NUM_TOKENS} tokens, got {token_count}")
    if keep_count == token_count:
        return torch.arange(token_count, dtype=torch.long)
    if strategy == "random":
        generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(sample_index))
        return torch.randperm(token_count, generator=generator)[:keep_count].sort().values
    if strategy != "uniform":
        raise ValueError("observation strategy must be 'uniform' or 'random'")

    positions = beam_positions.float()
    selected = [0]
    while len(selected) < keep_count:
        selected_positions = positions[torch.tensor(selected)]
        distances = torch.cdist(positions, selected_positions).amin(dim=1)
        distances[torch.tensor(selected)] = -math.inf
        selected.append(int(distances.argmax()))
    return torch.tensor(sorted(selected), dtype=torch.long)


def observed_beam_mask(token_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(NUM_BEAMS, dtype=torch.bool)
    token_cols = BEAM_GRID_SHAPE[1] // PATCH_SHAPE[1]
    for token_index in token_indices.tolist():
        token_row, token_col = divmod(int(token_index), token_cols)
        row_start = token_row * PATCH_SHAPE[0]
        col_start = token_col * PATCH_SHAPE[1]
        for row_offset in range(PATCH_SHAPE[0]):
            for col_offset in range(PATCH_SHAPE[1]):
                beam_id = (
                    (row_start + row_offset) * BEAM_GRID_SHAPE[1]
                    + col_start
                    + col_offset
                )
                mask[beam_id] = True
    return mask
