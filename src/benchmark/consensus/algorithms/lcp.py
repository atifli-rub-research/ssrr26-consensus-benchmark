"""LCP consensus strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class LCP(ConsensusAlgorithm):

    name: ClassVar[str] = "LCP"

    def update(self, uav: UAVAgent) -> None:
        """
        Linear consensus Protocol according to "Consensus problems in networks of agents with switching topology and time-delays", 2004
        """
        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})

            if not opinions:
                continue  # keine Meinungen vorhanden

            # Berechne Mittelwert aller bekannten Meinungen (inkl. eigener, falls vorhanden)
            total = 0
            count = 0
            for agent_name, value in opinions.items():
                total += value
                count += 1

            if count > 0:
                new_opinion = round(total / count, 3)
                current_opinion = opinions.get(uav.name, 0.0)

                # Nur aktualisieren wenn sich die Meinung signifikant ändert
                if abs(new_opinion - current_opinion) > 0.1:
                    opinions[uav.name] = new_opinion
                    mine['consensus_value'] = new_opinion
                    # Zur Kommunikation vormerken wenn sich Meinung geändert hat
                    if mine not in uav.comm_candidates:
                        uav.comm_candidates.append(mine)
