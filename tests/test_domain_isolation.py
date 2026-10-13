import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment


class RejectTestAccess(dict):
    def __getitem__(self, key):
        if key == 3:
            raise AssertionError("Unseen 3-hp domain accessed during train()")
        return super().__getitem__(key)


class DomainIsolationTests(unittest.TestCase):
    def test_train_does_not_access_test_domain(self):
        rng = np.random.default_rng(7)
        xs = RejectTestAccess({
            0: rng.normal(size=(4, 16, 1)), 1: rng.normal(size=(4, 16, 1)),
            2: rng.normal(size=(4, 16, 1)), 3: rng.normal(size=(4, 16, 1)),
        })
        ys = RejectTestAccess({0: np.arange(4), 1: np.arange(4), 2: np.arange(4), 3: np.arange(4)})
        old = dict(experiment.CFG)
        experiment.CFG.update({"epochs": 1, "batch_per_domain": 4, "kernel_size": 9})
        try:
            experiment.train(42, xs, ys, "baseline")
        finally:
            experiment.CFG.clear(); experiment.CFG.update(old)

    def test_protocol_uses_load_2_only_for_selection(self):
        source = inspect.getsource(experiment.train)
        self.assertIn("xs[2]", source)
        self.assertNotIn("xs[3]", source)
        self.assertNotIn("ys[3]", source)


if __name__ == "__main__":
    unittest.main()

