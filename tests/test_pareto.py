from __future__ import annotations

import unittest

import torch

from mspc.pareto import (
    ParetoWeightSolver,
    generate_candidates,
    nondominated_mask,
    objective_values,
    select_ideal,
    simplex_grid,
)


class ParetoTests(unittest.TestCase):
    def test_three_task_grid_has_documented_size(self):
        coarse = simplex_grid(3, 0.05)
        fine = simplex_grid(3, 0.02)
        self.assertEqual(coarse.shape, (231, 3))
        self.assertEqual(fine.shape, (1326, 3))
        self.assertTrue(torch.allclose(coarse.sum(dim=1), torch.ones(231)))

    def test_candidates_include_equal_vertices_and_previous(self):
        previous = torch.tensor([0.17, 0.33, 0.50])
        candidates = generate_candidates(3, 0.05, previous)
        expected = [torch.eye(3)[i] for i in range(3)]
        expected.extend([torch.full((3,), 1 / 3), previous])
        for row in expected:
            self.assertTrue(torch.any(torch.all(torch.isclose(candidates, row), dim=1)))

    def test_sampled_candidates_for_many_modules_are_on_simplex(self):
        first = generate_candidates(5, sampled_candidates=20, seed=7)
        second = generate_candidates(5, sampled_candidates=20, seed=7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first >= 0))
        self.assertTrue(torch.allclose(first.sum(dim=1), torch.ones(first.shape[0])))

    def test_candidate_floor_is_part_of_the_pareto_feasible_domain(self):
        candidates = generate_candidates(3, step=0.05, min_weight=0.25)
        self.assertTrue(torch.all(candidates >= 0.25 - 1e-7))
        self.assertTrue(torch.all(candidates <= 0.50 + 1e-7))
        self.assertTrue(torch.allclose(candidates.sum(dim=1), torch.ones(candidates.shape[0])))

    def test_impossible_candidate_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_candidates(3, min_weight=1 / 3)

    def test_nondominated_filter_matches_manual_set(self):
        objectives = torch.tensor(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        self.assertTrue(
            torch.equal(nondominated_mask(objectives), torch.tensor([True, True, True, False]))
        )

    def test_ideal_tie_uses_lexicographic_alpha(self):
        candidates = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        objectives = torch.zeros(2, 3)
        selected, distance = select_ideal(
            candidates, objectives, torch.tensor([True, True])
        )
        self.assertEqual(selected, 1)
        self.assertEqual(distance, 0.0)

    def test_solver_returns_valid_simplex_weight(self):
        conflicts = torch.tensor(
            [[0.0, 0.8, 0.0], [0.8, 0.0, 0.4], [0.0, 0.4, 0.0]]
        )
        q = torch.tensor([0.7, 0.5, 0.4])
        result = ParetoWeightSolver().solve(conflicts, q)
        self.assertTrue(torch.all(result.alpha >= 0))
        self.assertAlmostEqual(float(result.alpha.sum()), 1.0, places=6)
        self.assertGreater(result.pareto_size, 0)
        values = objective_values(result.alpha.unsqueeze(0), conflicts, q)
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
