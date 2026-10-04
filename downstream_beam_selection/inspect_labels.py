from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_beam_selection.data_io import BeamCorpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect 64-beam oracle labels")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    corpus = BeamCorpus.load(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = torch.bincount(corpus.oracle_beam, minlength=64).numpy()
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.bar(np.arange(64), counts)
    axis.set_xlabel("Unshifted beam ID (row × 8 + col)")
    axis.set_ylabel("Samples")
    axis.set_title("Oracle beam class distribution")
    fig.tight_layout()
    fig.savefig(args.output_dir / "oracle_beam_distribution.png", dpi=180)
    plt.close(fig)

    example_count = min(max(args.examples, 0), len(corpus.samples))
    for index in range(example_count):
        gains = corpus.beam_gains_db[index].reshape(8, 8).numpy()
        oracle = int(corpus.oracle_beam[index])
        row, col = divmod(oracle, 8)
        fig, axis = plt.subplots(figsize=(5.5, 4.8))
        image = axis.imshow(gains, origin="upper", cmap="viridis")
        axis.scatter([col], [row], marker="x", color="red", s=100, linewidths=2)
        axis.set_title(f"sample={index}, oracle_beam={oracle}")
        axis.set_xlabel("beam column")
        axis.set_ylabel("beam row")
        fig.colorbar(image, ax=axis, label="mean gain (dBW-equivalent)")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"beam_map_{index:03d}.png", dpi=180)
        plt.close(fig)

    print(f"samples={len(corpus.samples)}")
    print(f"nonempty_classes={int((counts > 0).sum())}")
    print(f"min_nonzero_class_count={int(counts[counts > 0].min()) if (counts > 0).any() else 0}")
    print(f"max_class_count={int(counts.max())}")


if __name__ == "__main__":
    main()

