"""Minimal training example for MSPC core mode.

Run from the repository root with::

    python -m examples.toy_training
"""

from __future__ import annotations

import torch

from mspc import MSPCConfig, MSPCCoordinator


def main() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    coordinator = MSPCCoordinator(
        MSPCConfig(
            mode="core",
            grid_step=0.1,
            risk_thresholds=[0.5, 0.5, 0.5],
        )
    )

    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    targets = [
        torch.tensor([[1.0], [0.0], [1.0], [-1.0]]),
        torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
        torch.tensor([[0.5], [0.5], [1.0], [0.0]]),
    ]
    risks = torch.tensor([0.70, 0.55, 0.40])

    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        losses = [torch.nn.functional.mse_loss(predictions, target) for target in targets]

        result = coordinator.coordinate_losses(
            losses,
            model.parameters(),
            risks,
        )
        optimizer.step()

        print(
            f"step={step} "
            f"losses={[round(loss.item(), 4) for loss in losses]} "
            f"alpha={[round(value, 4) for value in result.alpha.tolist()]} "
            f"q={[round(value, 4) for value in result.q.tolist()]} "
            f"conflict_pair_rate={result.diagnostics.conflict_pair_rate:.4f}"
        )


if __name__ == "__main__":
    main()
