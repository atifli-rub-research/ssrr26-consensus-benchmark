"""NO_CONSENSUS consensus strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class NoConsensus(ConsensusAlgorithm):

    name: ClassVar[str] = "NO_CONSENSUS"

    def update(self, uav: UAVAgent) -> None:
        """
        Baseline without swarm consensus: keep every agent's own opinion unchanged.

        Other agents' opinions may be stored for logging, but this agent's
        consensus_value is always its local opinion only.
        """
        for mine in uav.mine_objects:
            opinions = mine.get("opinions", {})
            if uav.name not in opinions:
                continue

            mine["consensus_value"] = opinions[uav.name]
