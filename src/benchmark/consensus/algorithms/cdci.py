"""CDCI consensus strategy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from benchmark.consensus.core import constants as C

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class CDCI(ConsensusAlgorithm):

    name: ClassVar[str] = "CDCI"
    gamma: float = C.CDCI_GAMMA
    rho: float = C.CDCI_RHO
    alpha: float = C.CDCI_ALPHA
    sigma: float = C.CDCI_SIGMA

    def update(self, uav: UAVAgent) -> None:
        """
        Collective Decision through Cross-Inhibition according to "A Design Pattern for Decentralised Decision Making", 2015
        Die Meinungen liegen im Intervall [-1, 1] und werden zu Committments ∈ (0, 1) gemappt.
        gamma = social recruitment, rho = spontaneous recruitment, alpha = social withdrawal, sigma = spontaneous withdrawal
        """
        def opinion_is_committed(opinion):
            """Binäre Schwelle"""
            return 1 if opinion >= 0.3 else 0

        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})
            if not opinions:
                continue

            # Eigene Meinung (original)
            own_opinion = opinions.get(uav.name, 0.0)
            committed_self = opinion_is_committed(own_opinion)

            # Mapping aller Nachbarn
            neighbors = [k for k in opinions if k != uav.name]
            neighbor_commitments = [opinion_is_committed(opinions[k]) for k in neighbors]

            # Anteil committed Nachbarn
            if neighbors:
                committed_ratio = sum(neighbor_commitments) / len(neighbor_commitments)
            else:
                committed_ratio = 0

            # CDCI-Zustandsupdate
            if committed_self == 0:
                # Uncommitted → committed
                if random.random() < (self.rho + self.gamma * committed_ratio):
                    opinions[uav.name] = 1.0  # wird committed
            else:
                # Committed → uncommitted
                if random.random() < (self.sigma + self.alpha * (1 - committed_ratio)):
                    opinions[uav.name] = -1.0  # wird uncommitted

            # Konsenswert als Mittelwert der binären beliefs
            all_opinions = [v for v in opinions.values()]
            consensus_value = round(sum(all_opinions) / len(all_opinions), 3)
            mine['consensus_value'] = consensus_value


            # Optional zur Kommunikation vormerken
            uav.comm_candidates.append(mine)
