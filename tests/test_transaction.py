from __future__ import annotations

import copy
import random
import unittest

import torch

from mspc import (
    FeedbackController,
    FullMSPCRunner,
    MSPCConfig,
    MSPCCoordinator,
    SafetyEvaluation,
    capture_rng_state,
)

from _helpers import assert_nested_equal


class TransactionTests(unittest.TestCase):
    def _fixture(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        # Materialize Adam moments before taking the transaction snapshot.
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
        coordinator = MSPCCoordinator(
            MSPCConfig(mode="full", risk_thresholds=[0.4, 0.4])
        )
        coordinator.record_transition(torch.tensor([0.2, 0.2]), torch.tensor([0.3, 0.1]))
        feedback = FeedbackController([0.4, 0.4], utility_min=0.7, stability_min=0.7)
        runner = FullMSPCRunner(
            model,
            optimizer,
            coordinator,
            feedback,
            scheduler=scheduler,
            max_retries=1,
        )
        return model, optimizer, scheduler, coordinator, runner

    def test_failed_update_restores_model_adam_scheduler_history_and_rng(self):
        model, optimizer, scheduler, coordinator, runner = self._fixture()
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        expected_coordinator = copy.deepcopy(coordinator.state_dict())
        expected_rng = capture_rng_state()

        def losses():
            value = model(torch.ones(1, 1)).squeeze()
            return [(value - 1) ** 2, (value + 1) ** 2]

        def reject():
            random.random()
            torch.rand(1)
            return SafetyEvaluation(torch.tensor([0.9, 0.9]), utility=0.1, stability=0.1)

        result = runner.run_step(losses, model.parameters(), torch.tensor([0.3, 0.3]), reject)
        self.assertFalse(result.accepted)
        self.assertTrue(result.rolled_back)
        self.assertEqual(result.attempts, 2)
        assert_nested_equal(self, model.state_dict(), expected_model)
        assert_nested_equal(self, optimizer.state_dict(), expected_optimizer)
        assert_nested_equal(self, scheduler.state_dict(), expected_scheduler)
        assert_nested_equal(self, coordinator.state_dict(), expected_coordinator)
        actual_rng = capture_rng_state()
        self.assertEqual(actual_rng["python"], expected_rng["python"])
        self.assertTrue(torch.equal(actual_rng["torch_cpu"], expected_rng["torch_cpu"]))

    def test_feedback_retry_can_commit_and_records_only_accepted_state(self):
        model, optimizer, scheduler, coordinator, runner = self._fixture()
        calls = {"count": 0}

        def losses():
            value = model(torch.ones(1, 1)).squeeze()
            return [(value - 2) ** 2, (value - 3) ** 2]

        def evaluate():
            calls["count"] += 1
            if calls["count"] == 1:
                return SafetyEvaluation(torch.tensor([0.8, 0.2]), utility=1.0, stability=1.0)
            return SafetyEvaluation(torch.tensor([0.3, 0.2]), utility=1.0, stability=1.0)

        result = runner.run_step(
            losses, model.parameters(), torch.tensor([0.3, 0.2]), evaluate
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.rolled_back)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(coordinator.history), 2)

    def test_exception_rolls_back_before_it_is_reraised(self):
        model, optimizer, scheduler, coordinator, runner = self._fixture()
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())

        def losses():
            value = model(torch.ones(1, 1)).squeeze()
            return [(value - 2) ** 2, (value - 3) ** 2]

        def broken_evaluation():
            raise RuntimeError("validation failed")

        with self.assertRaisesRegex(RuntimeError, "validation failed"):
            runner.run_step(
                losses, model.parameters(), torch.tensor([0.3, 0.2]), broken_evaluation
            )
        assert_nested_equal(self, model.state_dict(), expected_model)
        assert_nested_equal(self, optimizer.state_dict(), expected_optimizer)


if __name__ == "__main__":
    unittest.main()
