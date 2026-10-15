from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mspc import (
    MSPCConfig,
    MSPCCoordinator,
    append_diagnostics,
    load_full_checkpoint,
    save_full_checkpoint,
    save_run_config,
)


class PersistenceTests(unittest.TestCase):
    def test_checkpoint_and_json_outputs(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
        config = MSPCConfig(risk_thresholds=[0.5, 0.5])
        coordinator = MSPCCoordinator(config)
        result = coordinator.coordinate(
            torch.tensor([[1.0], [2.0]]), torch.tensor([0.2, 0.3])
        )
        expected_weight = model.weight.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "last_full.pt"
            diagnostics_path = root / "diagnostics.jsonl"
            config_path = root / "run_config.json"
            save_full_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                coordinator=coordinator,
                epoch=3,
                global_step=17,
                config=config,
            )
            append_diagnostics(diagnostics_path, result.diagnostics, epoch=3, step=17)
            save_run_config(config_path, config)

            with torch.no_grad():
                model.weight.add_(100)
            coordinator.history.clear()
            metadata = load_full_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                coordinator=coordinator,
            )
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["global_step"], 17)
            diagnostic = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["epoch"], 3)
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(run_config["history_size"], 5)


if __name__ == "__main__":
    unittest.main()
