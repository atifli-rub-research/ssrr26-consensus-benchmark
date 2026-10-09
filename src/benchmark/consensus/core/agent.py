"""UAV perception state and generic execution of consensus strategies."""

from __future__ import annotations

from typing import Any

from benchmark.consensus.algorithms import ConsensusAlgorithm
from benchmark.consensus.core import constants as C


class UAVAgent:
    """Single UAV with per-tile mine slots and an ordered list of strategies."""

    def __init__(
        self,
        name: str,
        tiles_by_index: dict[int, dict[str, Any]],
        algorithms: list[ConsensusAlgorithm] | None = None,
    ):
        self.name = name
        self.tiles_by_index = tiles_by_index
        self.algorithms = list(algorithms) if algorithms is not None else []
        self.mine_objects: list[dict[str, Any]] = []
        self.comm_candidates: list[dict[str, Any]] = []
        self._tile_slot_to_mine: dict[tuple[int, int], dict[str, Any]] = {}

    @staticmethod
    def _sorted_detections(
        tile_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        dets = list(tile_record.get("detections") or [])
        return sorted(dets, key=lambda d: float(d["confidence"]), reverse=True)

    @staticmethod
    def _detection_to_opinion(tile_record: dict[str, Any]) -> float:
        """Opinion from highest-confidence detection (slot 0)."""
        dets = UAVAgent._sorted_detections(tile_record)
        if not dets:
            return C.NO_DETECTION_OPINION
        max_conf = float(dets[0]["confidence"])
        return round(2.0 * max_conf - 1.0, 3)

    @staticmethod
    def slot_opinion(tile_record: dict[str, Any], slot_index: int) -> float:
        """Return the opinion for a ranked mine slot.

        Slot s uses the (s+1)-th highest-confidence detection if present,
        otherwise NO_DETECTION_OPINION (no mine at this slot).
        """
        dets = UAVAgent._sorted_detections(tile_record)
        if slot_index >= len(dets):
            return C.NO_DETECTION_OPINION
        conf = float(dets[slot_index]["confidence"])
        return round(2.0 * conf - 1.0, 3)

    @staticmethod
    def slot_detection_box(
        tile_record: dict[str, Any], slot_index: int
    ) -> list[float] | None:
        """Return the bounding box for a ranked detection, or None."""
        dets = UAVAgent._sorted_detections(tile_record)
        if slot_index >= len(dets):
            return None
        return list(dets[slot_index]["bbox_xyxy"])

    def _get_or_create_slot_mine(
        self,
        tile_index: int,
        slot_index: int,
        template: dict[str, Any],
    ) -> dict[str, Any]:
        key = (tile_index, slot_index)
        if key in self._tile_slot_to_mine:
            return self._tile_slot_to_mine[key]

        gt = template.get("ground_truth") or {}
        mine: dict[str, Any] = {
            "tile_index": tile_index,
            "slot_index": slot_index,
            "cell_id": gt.get("cell_id"),
            "ground_truth": gt,
            "true_mine_present": bool(template.get("true_mine_present")),
            "true_mine_count": int(template.get("true_mine_count", 0)),
            "opinions": {},
            "observations": {},
            "consensus_value": None,
        }
        self.mine_objects.append(mine)
        self._tile_slot_to_mine[key] = mine
        return mine

    def observe_tile(
        self,
        tile_index: int,
        max_slots: int,
    ) -> float:
        """Register own opinions for each mine slot on this tile."""
        record = self.tiles_by_index[tile_index]
        primary = self._detection_to_opinion(record)

        for slot in range(max_slots):
            opinion = self.slot_opinion(record, slot)
            mine = self._get_or_create_slot_mine(tile_index, slot, record)
            mine["opinions"][self.name] = opinion
            n_dets = len(record.get("detections") or [])
            mine["observations"][self.name] = max(1, n_dets) if slot == 0 else n_dets
            mine["consensus_value"] = opinion
            self._maybe_run_consensus_on_slot(tile_index, slot)

        return primary

    def _update_consensus(self) -> None:
        """Apply configured strategies in order to locally observed slots.

        Strategies in this list share state and execute sequentially. Benchmark
        comparisons use one strategy per agent, in a separate simulation for
        each algorithm, to keep algorithm results independent.
        """
        saved = self.mine_objects
        self.mine_objects = [
            m for m in saved if self.name in m.get("opinions", {})
        ]
        try:
            for algorithm in self.algorithms:
                algorithm.update(self)
        finally:
            self.mine_objects = saved

    def _maybe_run_consensus_on_slot(
        self,
        tile_index: int,
        slot_index: int,
    ) -> None:
        mine = self._tile_slot_to_mine.get((tile_index, slot_index))
        if not mine or self.name not in mine.get("opinions", {}):
            return
        if len(mine.get("opinions", {})) < 2:
            return
        self.comm_candidates.clear()
        self._update_consensus()

    def receive_slot_perception(
        self,
        sender_name: str,
        tile_index: int,
        slot_index: int,
        opinion: float,
        observation_count: int,
        template: dict[str, Any],
    ) -> None:
        if sender_name == self.name:
            return

        mine = self._get_or_create_slot_mine(tile_index, slot_index, template)
        mine["opinions"][sender_name] = opinion
        mine["observations"][sender_name] = observation_count

        if self.name not in mine.get("opinions", {}):
            return

        self.comm_candidates.clear()
        self._update_consensus()
        if mine.get("consensus_value") is None and mine["opinions"]:
            mine["consensus_value"] = round(
                sum(mine["opinions"].values()) / len(mine["opinions"]), 3
            )

    def run_consensus_on_known_mines(self) -> None:
        for mine in list(self.mine_objects):
            if self.name not in mine.get("opinions", {}):
                continue
            if len(mine["opinions"]) < 2:
                continue
            self.comm_candidates.clear()
            self._update_consensus()
