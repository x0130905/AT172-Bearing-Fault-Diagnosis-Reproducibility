import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from recompute_metrics import METHODS, SEEDS, metric_from_predictions


class ArtifactMetricTests(unittest.TestCase):
    def test_all_prediction_metrics_match_json(self):
        for method in METHODS:
            for seed in SEEDS:
                run = f"{method}_{seed}"
                calculated = metric_from_predictions(ROOT / "predictions" / run / "predictions_test.npz")
                recorded = json.loads((ROOT / "metrics" / run / "metrics.json").read_text(encoding="utf-8"))["unseen_test_3"]
                for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                    self.assertAlmostEqual(calculated[metric], recorded[metric], places=12)

    def test_coral_predictions_equal_erm(self):
        for seed in SEEDS:
            erm = np.load(ROOT / "predictions" / f"baseline_{seed}" / "predictions_test.npz")["prediction"]
            coral = np.load(ROOT / "predictions" / f"coral_{seed}" / "predictions_test.npz")["prediction"]
            self.assertTrue(np.array_equal(erm, coral))


if __name__ == "__main__":
    unittest.main()
