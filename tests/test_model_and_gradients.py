import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment


class ModelTests(unittest.TestCase):
    def test_parameter_count_and_shapes(self):
        model = experiment.CNN(1)
        self.assertEqual(model.parameter_count, 228)
        self.assertEqual(model.p["cw"].shape, (9, 1, 16))
        self.assertEqual(model.p["cb"].shape, (16,))
        self.assertEqual(model.p["dw"].shape, (16, 4))
        self.assertEqual(model.p["db"].shape, (4,))

    def test_same_seed_same_initial_parameters(self):
        first, second = experiment.CNN(42), experiment.CNN(42)
        for key in first.p:
            self.assertTrue(np.array_equal(first.p[key], second.p[key]))

    def _finite_difference(self, method, parameter, index):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(8, 18, 1))
        y = np.asarray([0,1,2,3,0,1,2,3])
        domains = np.asarray([0,0,0,0,1,1,1,1])
        model = experiment.CNN(11)
        if method == "coral":
            objective = lambda: model.hard_gradients(x, y, domains, coral_weight=1.0)[0]
            analytic = model.hard_gradients(x, y, domains, coral_weight=1.0)[3][parameter][index]
        elif method == "mmd":
            objective = lambda: model.hard_gradients(x, y, domains, mmd_weight=1.0)[0]
            analytic = model.hard_gradients(x, y, domains, mmd_weight=1.0)[3][parameter][index]
        else:
            objective = lambda: model.hard_gradients(x, y, domains)[0]
            analytic = model.hard_gradients(x, y, domains)[3][parameter][index]
        epsilon = 1e-6
        original = model.p[parameter][index]
        model.p[parameter][index] = original + epsilon; plus = objective()
        model.p[parameter][index] = original - epsilon; minus = objective()
        model.p[parameter][index] = original
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(float(analytic), float(numerical), places=5)

    def test_erm_gradient(self):
        self._finite_difference("baseline", "cw", (2, 0, 3))

    def test_coral_gradient(self):
        self._finite_difference("coral", "cw", (2, 0, 3))

    def test_mmd_gradient(self):
        self._finite_difference("mmd", "cw", (2, 0, 3))

    def test_mixup_soft_label_gradient(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(6, 18, 1))
        labels = np.asarray([0, 1, 2, 3, 0, 1])
        onehot = np.eye(4)[labels]
        permutation = np.asarray([3, 4, 5, 0, 1, 2])
        lam = 0.37
        mixed_x = lam * x + (1 - lam) * x[permutation]
        targets = lam * onehot + (1 - lam) * onehot[permutation]
        model = experiment.CNN(9)
        parameter, index = "cw", (2, 0, 3)
        analytic = model.soft_gradients(mixed_x, targets)[3][parameter][index]
        epsilon = 1e-6
        original = model.p[parameter][index]
        model.p[parameter][index] = original + epsilon
        plus = model.soft_gradients(mixed_x, targets)[0]
        model.p[parameter][index] = original - epsilon
        minus = model.soft_gradients(mixed_x, targets)[0]
        model.p[parameter][index] = original
        self.assertAlmostEqual(float(analytic), float((plus - minus) / (2 * epsilon)), places=5)


if __name__ == "__main__":
    unittest.main()

