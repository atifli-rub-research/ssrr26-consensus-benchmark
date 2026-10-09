"""WMSR consensus strategy."""

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
class WMSR(ConsensusAlgorithm):

    name: ClassVar[str] = "WMSR"
    f: int = C.WMSR_F

    def update(self, uav: UAVAgent) -> None:
        """
        Weighted Mean Subsequence Reduced Algorithm according to "Resilient asymptotic consensus in robust networks", 2013
        :param f: Anzahl tolerierter fehlerhafter Nachbarn --> is to be set in constants.py
        """
        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})
            all_values = list(opinions.values())

            if len(all_values) <= 2 * self.f:
                return  # Nicht genug Meinungen zum Trimmen

            sorted_values = sorted(all_values)
            trimmed_values = sorted_values[self.f:-self.f]

            if trimmed_values:
                new_opinion = round(sum(trimmed_values) / len(trimmed_values), 3)
                # Nur aktualisieren wenn sich die Meinung signifikant ändert
                current_opinion = opinions.get(uav.name, 0.0)
                if abs(new_opinion - current_opinion) > 0.01:
                    opinions[uav.name] = new_opinion
                    mine['consensus_value'] = new_opinion
                    # Zur Kommunikation vormerken wenn sich Meinung geändert hat
                    if mine not in uav.comm_candidates:
                        uav.comm_candidates.append(mine)
