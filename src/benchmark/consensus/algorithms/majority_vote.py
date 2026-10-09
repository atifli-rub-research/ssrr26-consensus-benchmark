"""Simple majority-voting multi-agent fusion baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from benchmark.consensus.core import constants as C

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class MajorityVote(ConsensusAlgorithm):
    """Non-iterative decision-level majority-voting baseline."""

    name: ClassVar[str] = "MAJORITY_VOTE"

    def update(self, uav: UAVAgent) -> None:
        for mine in uav.mine_objects:
            opinions = mine.get("opinions", {})
            if not opinions:
                continue

            yes_votes = sum(
                opinion > C.NO_DETECTION_OPINION
                for opinion in opinions.values()
            )
            accepted = yes_votes > len(opinions) / 2
            mine["consensus_value"] = 1.0 if accepted else -1.0
