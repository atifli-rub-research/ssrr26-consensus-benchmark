"""Common interface for consensus update strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


class ConsensusAlgorithm(ABC):
    """Update an agent's locally observed mine slots in place.

    The agent exposes its name, mine_objects, and comm_candidates to the strategy.
    Algorithm parameters and any strategy-specific state belong to the strategy
    instance. Simulations create fresh instances for every agent and every run.
    """

    name: ClassVar[str]

    @abstractmethod
    def update(self, uav: UAVAgent) -> None:
        """Update opinions and consensus values on the supplied agent."""
