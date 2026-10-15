from __future__ import annotations

import unittest

import torch

from mspc.balance import balance_factors
from mspc.history import HistoryBuffer
from mspc.state import MetricSpec, StateAdapter


class StateAndHistoryTests(unittest.TestCase):
    def test_state_adapter_direction_clipping_and_degenerate_bound(self):
        adapter = StateAdapter(
            {
                "risk": MetricSpec(0.0, 10.0),
                "accuracy_as_risk": MetricSpec(0.0, 1.0, invert=True),
                "constant": MetricSpec(1.0, 1.0, neutral=0.25),
            }
        )
        state = adapter.adapt({"risk": 15.0, "accuracy_as_risk": 0.8, "constant": 1.0})
        self.assertEqual(state["risk"], 1.0)
        self.assertAlmostEqual(state["accuracy_as_risk"], 0.2)
        self.assertEqual(state["constant"], 0.25)
        self.assertEqual(len(adapter.anomalies), 1)

    def test_history_window_degradation_and_under_optimization(self):
        history = HistoryBuffer(maxlen=2)
        history.append(torch.tensor([0.2, 0.4]), torch.tensor([0.3, 0.2]))
        history.append(torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.3]))
        history.append(torch.tensor([0.1, 0.2]), torch.tensor([0.2, 0.8]))
        self.assertEqual(len(history), 2)
        self.assertTrue(torch.allclose(history.degradation(2), torch.tensor([0.1, 0.6])))
        under = history.under_optimization(torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.allclose(under, torch.tensor([0.1, 0.3]), atol=1e-6))

    def test_history_serialization_round_trip(self):
        history = HistoryBuffer(3)
        history.append(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]))
        restored = HistoryBuffer(1)
        restored.load_state_dict(history.state_dict())
        self.assertEqual(restored.maxlen, 3)
        self.assertEqual(len(restored), 1)
        self.assertTrue(
            torch.allclose(restored.degradation(2), torch.tensor([0.2, 0.2]), atol=1e-7)
        )

    def test_balance_factor_formula_and_clipping(self):
        result = balance_factors(
            torch.tensor([0.9, 0.2]),
            torch.tensor([0.3, 0.0]),
            torch.tensor([0.6, 0.4]),
            (0.5, 0.25, 0.25),
        )
        self.assertTrue(torch.allclose(result, torch.tensor([0.675, 0.2])))


if __name__ == "__main__":
    unittest.main()
