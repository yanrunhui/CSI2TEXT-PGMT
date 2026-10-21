from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import (
    _binary_threshold_metrics,
    _print_and_save_prototype_drift_metrics,
)
from scripts.pretrain import (
    freeze_prototype_bank,
    freeze_text_and_prototypes,
    semantic_anchor_freeze_policy,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text = torch.nn.Linear(2, 2)
        self.prototypes = torch.nn.Parameter(torch.eye(2))


class SemanticAnchorAblationTest(unittest.TestCase):
    def test_fast_binary_threshold_sweep_matches_naive_result(self) -> None:
        logits = torch.tensor(
            [
                [0.0, -1.0],
                [0.0, -1.0],
                [0.0, 0.5],
                [0.0, 1.5],
                [0.0, 2.0],
            ]
        )
        labels = torch.tensor([0, 1, 0, 1, 1])
        scores = logits[:, 1] - logits[:, 0]
        sorted_scores = torch.sort(scores).values
        thresholds = torch.cat(
            [
                sorted_scores[:1] - 1.0,
                (sorted_scores[:-1] + sorted_scores[1:]) * 0.5,
                sorted_scores[-1:] + 1.0,
            ]
        )
        naive = []
        for threshold in thresholds:
            predictions = (scores >= threshold).long()
            class0_acc = float((predictions[labels == 0] == 0).float().mean())
            class1_acc = float((predictions[labels == 1] == 1).float().mean())
            naive.append(
                (
                    0.5 * (class0_acc + class1_acc),
                    float((predictions == labels).float().mean()),
                    float(threshold),
                    class0_acc,
                    class1_acc,
                )
            )
        expected = max(enumerate(naive), key=lambda item: (item[1][0], -item[0]))[1]
        actual = _binary_threshold_metrics(logits, labels)

        self.assertAlmostEqual(actual["macro_top1"], expected[0])
        self.assertAlmostEqual(actual["top1"], expected[1])
        self.assertAlmostEqual(actual["threshold"], expected[2])
        self.assertAlmostEqual(actual["class0_acc"], expected[3])
        self.assertAlmostEqual(actual["class1_acc"], expected[4])

    def test_freeze_policies_match_requested_variants(self) -> None:
        self.assertEqual(semantic_anchor_freeze_policy("fixed"), (True, True))
        self.assertEqual(
            semantic_anchor_freeze_policy("adaptive_prototypes"),
            (True, False),
        )
        self.assertEqual(semantic_anchor_freeze_policy("joint"), (False, False))

    def test_text_and_prototype_freezing_are_independent(self) -> None:
        adaptive = _TinyModel()
        for parameter in adaptive.text.parameters():
            parameter.requires_grad = False
        self.assertFalse(any(parameter.requires_grad for parameter in adaptive.text.parameters()))
        self.assertTrue(adaptive.prototypes.requires_grad)

        fixed = _TinyModel()
        freeze_text_and_prototypes(fixed)
        self.assertFalse(any(parameter.requires_grad for parameter in fixed.text.parameters()))
        self.assertFalse(fixed.prototypes.requires_grad)

        prototype_only = _TinyModel()
        freeze_prototype_bank(prototype_only)
        self.assertTrue(any(parameter.requires_grad for parameter in prototype_only.text.parameters()))
        self.assertFalse(prototype_only.prototypes.requires_grad)

    def test_prototype_drift_matches_definition(self) -> None:
        prototype_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        initial_text_features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        checkpoint = {
            "prototype_initial_text_features": initial_text_features,
            "prototype_captions": ["first", "second"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "prototype_diagnostics.csv"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _print_and_save_prototype_drift_metrics(
                    prototype_features=prototype_features,
                    final_text_features=initial_text_features,
                    checkpoint=checkpoint,
                    prototype_keys=["first", "second"],
                    save_path=str(csv_path),
                )

            self.assertIn("prototype_to_initial_text_cosine_mean=0.500000", output.getvalue())
            self.assertIn("prototype_drift_Dp=0.500000", output.getvalue())
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
