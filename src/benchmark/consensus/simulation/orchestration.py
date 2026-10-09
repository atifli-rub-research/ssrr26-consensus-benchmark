"""Orchestrate consensus simulations, evaluation, and result payloads."""

from __future__ import annotations

from typing import Any

from benchmark.consensus.algorithms import algorithm_names
from benchmark.consensus.core import constants as C
from benchmark.consensus.evaluation.instance_evaluation import evaluate_multi_run
from benchmark.consensus.simulation.swarm_simulation import (
    MultiRunSimulationResult,
    SwarmSimulation,
    generate_start_combinations,
)


class ConsensusOrchestrator:
    """Run one experiment's algorithms and construct JSON-ready results.

    Algorithms are evaluated in independent simulations. Comparisons reuse
    the same start combinations for every algorithm to make runs comparable.
    File writing is handled separately by ConsensusJSONWriter.
    """

    def __init__(
        self,
        *,
        paths: list[str],
        seed: int = 42,
        n_runs: int | None = None,
        threshold: float | None = None,
        iou_threshold: float | None = None,
        wmsr_f: int | None = None,
    ):
        self.paths = list(paths)
        self.seed = seed
        self.n_runs = C.effective_simulation_runs(n_runs)
        self.threshold = threshold
        self.iou_threshold = (
            C.INSTANCE_IOU_THRESHOLD if iou_threshold is None else iou_threshold
        )
        self.wmsr_f = wmsr_f

    def run_algorithm(self, algorithm: str) -> dict[str, Any]:
        """Run one algorithm and include per-run metrics and its final state."""
        multi, evaluation = self._run_and_evaluate(algorithm)
        return {
            **self._result_metadata(),
            "wmsr_f": self.wmsr_f,
            "start_combinations": evaluation["start_combinations"],
            "metrics": evaluation["averaged"],
            "per_run_metrics": evaluation["per_run"],
            "last_run": {
                "start_positions": multi.runs[-1].start_positions,
                "flight_orders": multi.runs[-1].flight_orders,
                "final_tiles": multi.runs[-1].final_tiles,
            },
        }

    def run_comparison(self) -> dict[str, Any]:
        """Run all registered algorithms with identical Monte Carlo starts."""
        algorithms = algorithm_names()
        initial = self._simulation(algorithms[0])
        starts = generate_start_combinations(
            initial.drone_ids, initial.all_tiles, self.n_runs, self.seed
        )
        rows = []
        for algorithm in algorithms:
            _, evaluation = self._run_and_evaluate(algorithm, start_combinations=starts)
            rows.append(dict(evaluation["averaged"]))
        return {
            **self._result_metadata(),
            "start_combinations": starts,
            "comparison": rows,
        }

    def _simulation(self, algorithm: str) -> SwarmSimulation:
        return SwarmSimulation(
            self.paths, algorithm=algorithm, seed=self.seed, wmsr_f=self.wmsr_f
        )

    def _run_and_evaluate(
        self,
        algorithm: str,
        start_combinations: list[dict[str, int]] | None = None,
    ) -> tuple[MultiRunSimulationResult, dict[str, Any]]:
        simulation = self._simulation(algorithm)
        multi = simulation.run_multi(
            n_runs=self.n_runs, start_combinations=start_combinations
        )
        evaluation = evaluate_multi_run(
            multi, threshold=self.threshold, iou_threshold=self.iou_threshold
        )
        return multi, evaluation

    def _result_metadata(self) -> dict[str, Any]:
        return {
            "input_files": self.paths,
            "seed": self.seed,
            "n_runs": self.n_runs,
            "monte_carlo_averaging": self.n_runs > 1,
            "metric_level": "instance",
            "iou_threshold": self.iou_threshold,
            "threshold": self.threshold,
        }
