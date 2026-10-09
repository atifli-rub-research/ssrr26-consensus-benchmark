"""Tile-by-tile swarm simulation with perception sharing and consensus."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from benchmark.consensus.algorithms import algorithm_names, create_algorithm

from benchmark.consensus.core.agent import UAVAgent
from benchmark.consensus.data.predictions_loader import common_tile_indices, load_swarm
from benchmark.consensus.core import constants as C

def circular_tile_order(start_tile: int, tiles: list[int]) -> list[int]:
    """Visit every tile once, beginning at *start_tile*, then increasing with wrap."""
    if start_tile not in tiles:
        raise ValueError(f"start_tile {start_tile} not in tile set")
    i = tiles.index(start_tile)
    return tiles[i:] + tiles[:i]


def generate_start_combinations(
    drone_ids: list[str],
    tiles: list[int],
    n_runs: int,
    seed: int,
) -> list[dict[str, int]]:
    """Reproducible random start tile per UAV for each Monte-Carlo run."""
    rng = random.Random(seed)
    return [{did: rng.choice(tiles) for did in drone_ids} for _ in range(n_runs)]


def max_slots_for_tile(
    swarm_data: dict[str, dict[int, dict[str, Any]]],
    drone_ids: list[str],
    tile_index: int,
) -> int:
    """Number of mine slots to negotiate on this tile (one per detection rank)."""
    counts = [len(swarm_data[d][tile_index].get("detections") or []) for d in drone_ids]
    return min(max(max(counts), 1), C.MAX_MINE_SLOTS_PER_TILE)


@dataclass
class StepLog:
    step_index: int
    tiles_by_drone: dict[str, int] = field(default_factory=dict)
    observations: dict[str, float] = field(default_factory=dict)
    consensus_after_step: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SimulationResult:
    algorithm: str
    drone_ids: list[str]
    all_tiles: list[int]
    run_index: int = 0
    start_positions: dict[str, int] = field(default_factory=dict)
    flight_orders: dict[str, list[int]] = field(default_factory=dict)
    steps: list[StepLog] = field(default_factory=list)
    final_tiles: list[dict[str, Any]] = field(default_factory=list)
    per_drone_final_tiles: dict[str, list[dict[str, Any]]] | None = None
    swarm_data: dict[str, dict[int, dict[str, Any]]] | None = None


@dataclass
class MultiRunSimulationResult:
    algorithm: str
    drone_ids: list[str]
    seed: int
    n_runs: int
    start_combinations: list[dict[str, int]]
    runs: list[SimulationResult] = field(default_factory=list)


class SwarmSimulation:
    def __init__(
        self,
        prediction_paths: list[str],
        algorithm: str = "LCP",
        seed: int | None = 42,
        wmsr_f: int | None = None,
        algorithm_parameters: dict[str, Any] | None = None,
    ):
        self.algorithm = algorithm.upper()
        if self.algorithm not in algorithm_names():
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.prediction_paths = prediction_paths
        self.swarm_data = load_swarm(prediction_paths)
        self.drone_ids = sorted(self.swarm_data.keys())
        self.all_tiles = common_tile_indices(self.swarm_data)
        self.seed = seed if seed is not None else 42
        self.wmsr_f = wmsr_f if wmsr_f is not None else C.WMSR_F
        self.algorithm_parameters = dict(algorithm_parameters or {})
        if self.algorithm == "WMSR":
            self.algorithm_parameters.setdefault("f", self.wmsr_f)

    def _new_agents(self) -> dict[str, UAVAgent]:
        return {
            did: UAVAgent(
                did,
                self.swarm_data[did],
                algorithms=[create_algorithm(self.algorithm, **self.algorithm_parameters)],
            )
            for did in self.drone_ids
        }

    def run(
        self,
        start_positions: dict[str, int],
        run_index: int = 0,
        rng: random.Random | None = None,
    ) -> SimulationResult:
        """Single simulation pass with per-UAV random start and circular tile sweep."""
        if rng is not None:
            random.setstate(rng.getstate())

        agents = self._new_agents()
        flight_orders = {
            did: circular_tile_order(start_positions[did], self.all_tiles)
            for did in self.drone_ids
        }

        result = SimulationResult(
            algorithm=self.algorithm,
            drone_ids=self.drone_ids,
            all_tiles=list(self.all_tiles),
            run_index=run_index,
            start_positions=dict(start_positions),
            flight_orders=flight_orders,
        )

        n_steps = len(self.all_tiles)
        for step_idx in range(n_steps):
            step = StepLog(step_index=step_idx)
            tiles_this_step = {
                did: flight_orders[did][step_idx] for did in self.drone_ids
            }
            step.tiles_by_drone = dict(tiles_this_step)

            # 1) Each UAV observes its current tile (per mine slot)
            for drone_id in self.drone_ids:
                agent = agents[drone_id]
                tile_index = tiles_this_step[drone_id]
                n_slots = max_slots_for_tile(
                    self.swarm_data, self.drone_ids, tile_index
                )
                opinion = agent.observe_tile(tile_index, n_slots)
                step.observations[drone_id] = opinion

            # 2) Share slot opinions; receivers run consensus per slot
            for sender_id in self.drone_ids:
                tile_index = tiles_this_step[sender_id]
                record = self.swarm_data[sender_id][tile_index]
                template = self.swarm_data[self.drone_ids[0]].get(
                    tile_index, record
                )
                n_slots = max_slots_for_tile(
                    self.swarm_data, self.drone_ids, tile_index
                )
                obs_count = max(1, int(record.get("num_detections", 0)))

                for slot in range(n_slots):
                    opinion = UAVAgent.slot_opinion(record, slot)
                    for receiver_id in self.drone_ids:
                        if receiver_id == sender_id:
                            continue
                        receiver = agents[receiver_id]
                        receiver.receive_slot_perception(
                            sender_id,
                            tile_index,
                            slot,
                            opinion,
                            obs_count,
                            template,
                        )

            for _ in range(C.POST_STEP_CONSENSUS_ROUNDS):
                for agent in agents.values():
                    agent.run_consensus_on_known_mines()

            step.consensus_after_step = _snapshot_all_tiles(agents[self.drone_ids[0]])
            result.steps.append(step)

        if self.algorithm == "NO_CONSENSUS":
            result.per_drone_final_tiles = {
                did: _collect_final_state(agents[did]) for did in self.drone_ids
            }
            result.final_tiles = result.per_drone_final_tiles[self.drone_ids[0]]
        else:
            result.final_tiles = _collect_final_state(agents[self.drone_ids[0]])

        result.swarm_data = self.swarm_data
        return result

    def run_multi(
        self,
        n_runs: int | None = None,
        start_combinations: list[dict[str, int]] | None = None,
    ) -> MultiRunSimulationResult:
        """Run *n_runs* passes with reproducible start tiles; CDCI uses per-run RNG."""
        n = C.effective_simulation_runs(n_runs)
        starts = start_combinations or generate_start_combinations(
            self.drone_ids, self.all_tiles, n, self.seed
        )
        if len(starts) != n:
            raise ValueError("start_combinations length must match n_runs")

        multi = MultiRunSimulationResult(
            algorithm=self.algorithm,
            drone_ids=self.drone_ids,
            seed=self.seed,
            n_runs=n,
            start_combinations=starts,
        )

        for run_idx, start_positions in enumerate(starts):
            run_rng = random.Random(self.seed + run_idx + 1)
            multi.runs.append(
                self.run(start_positions, run_index=run_idx, rng=run_rng)
            )

        return multi


def _snapshot_all_tiles(ref_agent: UAVAgent) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for (tile_index, slot_index), mine in ref_agent._tile_slot_to_mine.items():
        out.setdefault(tile_index, {})[slot_index] = {
            "opinions": dict(mine.get("opinions", {})),
            "consensus_value": mine.get("consensus_value"),
        }
    return out


def _collect_final_state(ref_agent: UAVAgent) -> list[dict[str, Any]]:
    by_tile: dict[int, dict[str, Any]] = {}
    for mine in ref_agent.mine_objects:
        ti = mine["tile_index"]
        if ti not in by_tile:
            by_tile[ti] = {
                "tile_index": ti,
                "cell_id": mine.get("cell_id"),
                "true_mine_present": mine.get("true_mine_present"),
                "ground_truth": dict(mine.get("ground_truth") or {}),
                "slots": [],
            }
        by_tile[ti]["slots"].append(
            {
                "slot_index": mine["slot_index"],
                "opinions": dict(mine.get("opinions", {})),
                "consensus_value": mine.get("consensus_value"),
            }
        )

    rows: list[dict[str, Any]] = []
    for ti in sorted(by_tile):
        row = by_tile[ti]
        row["slots"].sort(key=lambda s: s["slot_index"])
        row["consensus_value"] = (
            row["slots"][0]["consensus_value"] if row["slots"] else None
        )
        rows.append(row)
    return rows
