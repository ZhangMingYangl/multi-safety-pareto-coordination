from __future__ import annotations

import unittest

import torch

from mspc import MSPCConfig, MSPCCoordinator
from mspc.collector import write_flat_gradient


class CoordinatorTests(unittest.TestCase):
    def test_core_round_produces_finite_diagnostics(self):
        coordinator = MSPCCoordinator(MSPCConfig(risk_thresholds=[0.5, 0.5, 0.5]))
        gradients = torch.tensor(
            [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]], dtype=torch.float64
        )
        result = coordinator.coordinate(gradients, torch.tensor([0.7, 0.5, 0.4]))
        self.assertTrue(torch.isfinite(result.fused_gradient).all())
        self.assertAlmostEqual(float(result.alpha.sum()), 1.0, places=8)
        self.assertEqual(result.diagnostics.conflict_matrix.shape, (3, 3))
        self.assertEqual(coordinator.step, 1)

    def test_state_round_trip_preserves_history_and_weight(self):
        coordinator = MSPCCoordinator(MSPCConfig(risk_thresholds=[0.4, 0.4]))
        coordinator.coordinate(torch.tensor([[1.0], [2.0]]), torch.tensor([0.2, 0.3]))
        coordinator.record_transition(torch.tensor([0.2, 0.3]), torch.tensor([0.5, 0.1]))
        restored = MSPCCoordinator(MSPCConfig(risk_thresholds=[0.4, 0.4]))
        restored.load_state_dict(coordinator.state_dict())
        self.assertEqual(restored.step, 1)
        self.assertEqual(len(restored.history), 1)
        self.assertTrue(torch.equal(restored.previous_alpha, coordinator.previous_alpha))

    def test_two_task_linear_network_loss_decreases(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        coordinator = MSPCCoordinator()
        x = torch.ones(8, 1)

        def total_loss():
            prediction = model(x)
            return ((prediction - 2.0) ** 2).mean() + ((prediction - 4.0) ** 2).mean()

        initial = float(total_loss().item())
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            losses = [
                ((prediction - 2.0) ** 2).mean(),
                ((prediction - 4.0) ** 2).mean(),
            ]
            result = coordinator.coordinate_losses(
                losses, model.parameters(), torch.tensor([0.5, 0.5])
            )
            optimizer.step()
        self.assertLess(float(total_loss().item()), initial)
        self.assertTrue(torch.isfinite(result.fused_gradient).all())


if __name__ == "__main__":
    unittest.main()

