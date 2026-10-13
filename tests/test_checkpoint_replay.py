import os
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment


class CheckpointReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("AT172_DATA_DIR")
        if not value:
            raise unittest.SkipTest("AT172_DATA_DIR not set")
        cls.xs, cls.ys, _ = experiment.load_dataset(Path(value))

    def test_all_checkpoints_reproduce_saved_test_predictions(self):
        for method in experiment.METHODS:
            for seed in experiment.SEEDS:
                run = f"{method}_{seed}"
                model = experiment.CNN(seed)
                checkpoint = np.load(ROOT / "checkpoints" / run / "checkpoint.npz")
                model.p = {key: checkpoint[key] for key in checkpoint.files}
                _, probability, prediction = experiment.scores(model, self.xs[3], self.ys[3])
                saved = np.load(ROOT / "predictions" / run / "predictions_test.npz")
                self.assertTrue(np.array_equal(prediction, saved["prediction"]), run)
                self.assertTrue(np.array_equal(self.ys[3], saved["y"]), run)
                self.assertTrue(np.allclose(probability, saved["probability"], rtol=0, atol=2e-15), run)


if __name__ == "__main__":
    unittest.main()
