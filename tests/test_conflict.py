from __future__ import annotations

import unittest

import torch

from mspc.collector import collect_gradients
from mspc.conflict import conflict_matrix, normalize_gradients, sequential_project


class ConflictTests(unittest.TestCase):
    def test_same_direction_has_no_conflict(self):
        gradients = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        conflicts = conflict_matrix(gradients)
        self.assertTrue(torch.equal(conflicts, torch.zeros(2, 2)))
        projected, orders = sequential_project(normalize_gradients(gradients), conflicts)
        self.assertTrue(torch.allclose(projected, normalize_gradients(gradients)))
        self.assertEqual(orders, [[], []])

    def test_orthogonal_has_no_conflict(self):
        gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(float(conflict_matrix(gradients)[0, 1]), 0.0)

    def test_opposite_gradients_are_projected(self):
        gradients = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        conflicts = conflict_matrix(gradients)
        self.assertAlmostEqual(float(conflicts[0, 1]), 1.0, places=6)
        normalized = normalize_gradients(gradients)
        projected, _ = sequential_project(normalized, conflicts)
        self.assertGreaterEqual(float(torch.dot(projected[0], normalized[1])), -1e-7)
        self.assertGreaterEqual(float(torch.dot(projected[1], normalized[0])), -1e-7)

    def test_zero_gradient_is_finite_and_non_conflicting(self):
        gradients = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        conflicts = conflict_matrix(gradients)
        normalized = normalize_gradients(gradients)
        projected, _ = sequential_project(normalized, conflicts)
        self.assertTrue(torch.isfinite(conflicts).all())
        self.assertTrue(torch.isfinite(projected).all())
        self.assertEqual(float(conflicts[0, 1]), 0.0)

    def test_empty_active_intersection_has_no_conflict(self):
        gradients = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        masks = torch.tensor([[True, False], [False, True]])
        self.assertEqual(float(conflict_matrix(gradients, masks)[0, 1]), 0.0)

    def test_projection_order_is_conflict_desc_then_index(self):
        normalized = normalize_gradients(
            torch.tensor([[1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]])
        )
        conflicts = torch.tensor(
            [[0.0, 0.5, 0.8], [0.5, 0.0, 0.2], [0.8, 0.2, 0.0]]
        )
        _, orders = sequential_project(normalized, conflicts)
        self.assertEqual(orders[0], [2, 1])

    def test_collector_preserves_active_parameter_masks(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        losses = [(first - 3) ** 2, (second + 1) ** 2]
        gradients, masks = collect_gradients(losses, [first, second])
        self.assertEqual(gradients.shape, (2, 2))
        self.assertTrue(torch.equal(masks, torch.tensor([[True, False], [False, True]])))


if __name__ == "__main__":
    unittest.main()

