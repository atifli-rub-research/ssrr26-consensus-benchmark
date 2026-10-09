"""EEV consensus strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class EEV(ConsensusAlgorithm):

    name: ClassVar[str] = "EEV"

    def update(self, uav: UAVAgent) -> None:
        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})
            my_opinion = opinions.get(uav.name, 0)

            all_opinions = [my_opinion] + [v for k, v in opinions.items() if k != uav.name]
            expectation = sum(all_opinions) / len(all_opinions)

            # Extremwert-Logik
            if expectation > 0:
                new_opinion = 1
            elif expectation < 0:
                new_opinion = -1
            else:
                new_opinion = my_opinion  # bleibt gleich

            if new_opinion != my_opinion:
                opinions[uav.name] = new_opinion
                uav.comm_candidates.append(mine)  # neue Meinung kommunizieren
            mine['opinions'] = opinions

            # Konsenswert speichern
            mean_opinion = round(sum(opinions.values()) / len(opinions), 3)
            mine['consensus_value'] = mean_opinion
