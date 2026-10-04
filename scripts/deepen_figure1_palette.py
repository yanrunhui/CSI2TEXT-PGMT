#!/usr/bin/env python3
"""Create a moderately deeper academic palette variant for Figure 1."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "figures" / "academic_recolored" / "figure1_academic_recolored.png"
OUTPUT = ROOT / "output" / "figures" / "academic_recolored" / "figure1_academic_recolored_deeper.png"


MAPPING = {
    # Three main regions.
    "#F7F7F5": "#EEF0ED",
    "#F3F7F8": "#E7EFF2",
    "#F8F6F2": "#F0EBE3",
    # Major functional containers.
    "#E8EEF4": "#D6E2EC",
    "#EDF4F1": "#DCEAE4",
    "#EEF3F6": "#DDE8EE",
    "#F4F6F7": "#E4EAED",
    "#F2F4F6": "#E5E9EC",
    "#F4F5F6": "#E7EBED",
    "#F4F4F2": "#E8E5DF",
    # Semantic-role cards.
    "#DDEBE8": "#C5DCD6",
    "#EAF0F7": "#D3E1EE",
    "#F4EEE2": "#E6D9C1",
    "#DCEAF2": "#BED9E7",
    # White cards get a subtle cool tint rather than pure white.
    "#FFFFFF": "#FAFBFC",
    # Paths and pale process arrows.
    "#56616B": "#46525D",
    "#B8C3CC": "#9BAAB6",
}


def to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    source = np.asarray(image, dtype=np.int16)
    output = source.copy()
    for old, new in MAPPING.items():
        old_rgb = to_rgb(old)
        mask = np.max(np.abs(source - old_rgb), axis=2) <= 3
        output[mask] = to_rgb(new)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output.astype(np.uint8), "RGB").save(
        OUTPUT, optimize=True, dpi=(300, 300)
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
