"""Multi-Safety Pareto Coordination public API."""

from .balance import balance_factors
from .collector import collect_gradients, write_flat_gradient
from .config import MSPCConfig
from .conflict import (
    conflict_matrix,
    normalize_gradients,
    scale_preserving_project,
    sequential_project,
)
from .coordinator import MSPCCoordinator
from .feedback import FeedbackController, SafetyLrController
from .history import HistoryBuffer, RiskRecord
from .hybrid import DescentGuardResult, common_descent_guard, min_norm_weights
from .pareto import (
    ParetoWeightSolver,
    generate_candidates,
    nondominated_mask,
    objective_values,
    select_ideal,
    simplex_grid,
)
from .persistence import (
    append_diagnostics,
    load_full_checkpoint,
    save_full_checkpoint,
    save_run_config,
)
from .state import MetricSpec, StateAdapter
from .transaction import FullMSPCRunner, UpdateSnapshot, capture_rng_state, restore_rng_state
from .types import (
    CoordinationDiagnostics,
    CoordinationResult,
    FeedbackDecision,
    FullStepResult,
    ParetoResult,
    SafetyEvaluation,
)

__all__ = [
    "MSPCConfig",
    "MSPCCoordinator",
    "HistoryBuffer",
    "RiskRecord",
    "MetricSpec",
    "StateAdapter",
    "ParetoWeightSolver",
    "FeedbackController",
    "SafetyLrController",
    "FullMSPCRunner",
    "UpdateSnapshot",
    "SafetyEvaluation",
    "CoordinationResult",
    "CoordinationDiagnostics",
    "FeedbackDecision",
    "FullStepResult",
    "ParetoResult",
    "conflict_matrix",
    "normalize_gradients",
    "sequential_project",
    "scale_preserving_project",
    "min_norm_weights",
    "common_descent_guard",
    "DescentGuardResult",
    "balance_factors",
    "simplex_grid",
    "generate_candidates",
    "objective_values",
    "nondominated_mask",
    "select_ideal",
    "collect_gradients",
    "write_flat_gradient",
    "capture_rng_state",
    "restore_rng_state",
    "append_diagnostics",
    "save_run_config",
    "save_full_checkpoint",
    "load_full_checkpoint",
]
