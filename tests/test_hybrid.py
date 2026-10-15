from __future__ import annotations

import unittest

import torch

from mspc import (
    MSPCConfig,
    MSPCCoordinator,
    common_descent_guard,
    conflict_matrix,
    min_norm_weights,
    scale_preserving_project,
)


class HybridTests(unittest.TestCase):
    def test_scale_preserving_projection_keeps_nonconflicting_magnitude(self):
        gradients = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
        projected, orders = scale_preserving_project(
            gradients, conflict_matrix(gradients)
        )
        self.assertTrue(torch.allclose(projected, gradients))
        self.assertEqual(orders, [[], []])
        self.assertAlmostEqual(float(projected[0].norm()), 3.0)

    def test_scale_preserving_projection_removes_negative_component(self):
        gradients = torch.tensor([[2.0, 1.0], [-1.0, 0.0]])
        projected, _ = scale_preserving_project(
            gradients, conflict_matrix(gradients)
        )
        reference = gradients[1] / gradients[1].norm()
        self.assertGreaterEqual(float(torch.dot(projected[0], reference)), -1e-6)
        self.assertAlmostEqual(float(projected[0, 1]), 1.0, places=6)

    def test_min_norm_weights_are_a_simplex_common_descent_solution(self):
        gradients = torch.tensor([[1.0, 0.0], [-0.25, 1.0], [0.0, 0.5]])
        weights = min_norm_weights(gradients)
        direction = torch.sum(weights[:, None] * gradients, dim=0)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(torch.all(gradients @ direction >= -1e-5))

    def test_min_norm_solver_is_stable_on_random_gradients(self):
        generator = torch.Generator().manual_seed(7)
        for _ in range(50):
            gradients = torch.randn(3, 64, generator=generator, dtype=torch.float64)
            weights = min_norm_weights(gradients)
            direction = torch.sum(weights[:, None] * gradients, dim=0)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
            self.assertTrue(torch.all(weights >= 0))
            self.assertGreaterEqual(float((gradients @ direction).min()), -2e-4)

    def test_guard_changes_only_a_non_descent_candidate(self):
        gradients = torch.tensor([[1.0, 0.0], [-0.25, 1.0], [0.0, 0.5]])
        unsafe = torch.tensor([1.0, 0.0])
        guarded = common_descent_guard(gradients, unsafe)
        self.assertTrue(guarded.triggered)
        self.assertGreater(guarded.blend, 0.0)
        self.assertTrue(torch.all(guarded.task_dots_after >= -2e-7))

        safe = torch.tensor([0.1, 1.0])
        unchanged = common_descent_guard(gradients, safe)
        self.assertFalse(unchanged.triggered)
        self.assertTrue(torch.equal(unchanged.gradient, safe))

    def test_hybrid_coordinator_reports_ablation_diagnostics(self):
        config = MSPCConfig(gradient_scaling="preserve", descent_guard="mgda")
        coordinator = MSPCCoordinator(config)
        result = coordinator.coordinate(
            torch.tensor([[1.0, 0.0], [-0.25, 1.0], [0.0, 0.5]]),
            torch.tensor([0.5, 0.5, 0.5]),
        )
        diagnostics = result.diagnostics.to_dict()
        self.assertEqual(diagnostics["gradient_scaling"], "preserve")
        self.assertEqual(diagnostics["descent_guard"], "mgda")
        self.assertIsNotNone(diagnostics["mgda_weights"])
        self.assertGreaterEqual(diagnostics["min_task_dot_after_guard"], -2e-7)

    def test_weight_floor_smoothing_and_task_count_scale(self):
        config = MSPCConfig(
            gradient_scaling="preserve",
            min_task_weight=0.25,
            alpha_smoothing=0.5,
            fused_scale="task_count",
        )
        coordinator = MSPCCoordinator(config)
        gradients = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        first = coordinator.coordinate(gradients, torch.tensor([0.9, 0.2, 0.1]))
        second = coordinator.coordinate(gradients, torch.tensor([0.1, 0.2, 0.9]))
        self.assertTrue(torch.all(first.alpha >= 0.25 - 1e-7))
        self.assertTrue(torch.all(second.alpha >= 0.25 - 1e-7))
        expected = 0.5 * first.alpha + 0.5 * second.diagnostics.selected_alpha
        self.assertTrue(torch.allclose(second.alpha, expected))
        self.assertEqual(second.diagnostics.fused_scale_factor, 3.0)
        unscaled = torch.sum(second.alpha[:, None] * second.projected_gradients, dim=0)
        self.assertTrue(torch.allclose(second.fused_gradient, 3.0 * unscaled))


if __name__ == "__main__":
    unittest.main()
