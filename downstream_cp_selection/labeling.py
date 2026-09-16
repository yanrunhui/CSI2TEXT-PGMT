from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CPLabel:
    """Ground-truth quantities derived from one link's ray paths."""

    class_index: int | None
    outside_energy_ratios: np.ndarray
    max_excess_delay_ns: float
    energy_delay_ns: float
    path_count: int


def validate_cp_lengths(cp_lengths_ns: list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(cp_lengths_ns, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two CP lengths are required")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("CP lengths must be finite and positive")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("CP lengths must be strictly increasing")
    return values


def make_cp_label(
    delay_ns: np.ndarray,
    path_loss_db: np.ndarray,
    cp_lengths_ns: list[float] | tuple[float, ...],
    max_outside_energy_ratio: float = 0.01,
) -> CPLabel:
    """Choose the shortest CP containing the requested fraction of path energy.

    RayVerse stores path loss rather than received power.  A common TX-power
    factor cancels in the energy ratio, so stable relative powers are sufficient.
    """

    cp_values = validate_cp_lengths(cp_lengths_ns)
    if not 0.0 <= max_outside_energy_ratio < 1.0:
        raise ValueError("max_outside_energy_ratio must be in [0, 1)")

    delays = np.asarray(delay_ns, dtype=np.float64).reshape(-1)
    losses = np.asarray(path_loss_db, dtype=np.float64).reshape(-1)
    if delays.shape != losses.shape:
        raise ValueError("delay_ns and path_loss_db must have identical shapes")
    valid = np.isfinite(delays) & np.isfinite(losses) & (delays >= 0.0)
    delays = delays[valid]
    losses = losses[valid]
    if delays.size == 0:
        raise ValueError("Link has no finite ray path")

    excess = delays - float(delays.min())
    # The strongest path has relative power one, preventing under/overflow even
    # when absolute path loss is very large.
    relative_power = np.power(10.0, -(losses - float(losses.min())) / 10.0)
    total_power = float(relative_power.sum())
    if not np.isfinite(total_power) or total_power <= 0.0:
        raise ValueError("Link has invalid total path energy")

    outside = np.asarray(
        [float(relative_power[excess > cp_ns].sum()) / total_power for cp_ns in cp_values],
        dtype=np.float32,
    )
    feasible = np.flatnonzero(outside <= max_outside_energy_ratio)
    class_index = int(feasible[0]) if feasible.size else None

    order = np.argsort(excess, kind="stable")
    sorted_delays = excess[order]
    cumulative = np.cumsum(relative_power[order]) / total_power
    target_inside = 1.0 - max_outside_energy_ratio
    energy_index = min(int(np.searchsorted(cumulative, target_inside, side="left")), len(order) - 1)
    return CPLabel(
        class_index=class_index,
        outside_energy_ratios=outside,
        max_excess_delay_ns=float(excess.max()),
        energy_delay_ns=float(sorted_delays[energy_index]),
        path_count=int(delays.size),
    )

