"""Load per-UAV prediction JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any



def load_predictions(path: str | Path) -> tuple[str, dict[int, dict[str, Any]]]:
    """
    Load one predictions file.

    Returns:
        drone_id, mapping tile_index -> tile record
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError(f"Empty predictions file: {path}")

    drone_id = records[0].get("drone_id") or path.stem
    by_tile: dict[int, dict[str, Any]] = {}
    for rec in records:
        idx = int(rec["tile_index"])
        if idx in by_tile:
            raise ValueError(f"Duplicate tile_index {idx} in {path}")
        by_tile[idx] = rec

    return drone_id, by_tile

def load_swarm(prediction_paths: list[str | Path]) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Load multiple UAV files.

    One entry in prediction_paths = one simulated UAV agent.

    This intentionally allows the same prediction file to appear multiple times,
    because repeated paths can represent multiple UAVs with identical detector
    outputs.
    """
    swarm: dict[str, dict[int, dict[str, Any]]] = {}

    for i, p in enumerate(prediction_paths, start=1):
        drone_id, by_tile = load_predictions(p)

        agent_id = f"uav_{i:03d}__{drone_id}"

        swarm[agent_id] = by_tile

    return swarm

def common_tile_indices(swarm: dict[str, dict[int, dict[str, Any]]]) -> list[int]:
    """Sorted tile indices present in every UAV file."""
    if not swarm:
        return []
    sets = [set(tiles.keys()) for tiles in swarm.values()]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("No common tile indices across prediction files")
    return sorted(common)
