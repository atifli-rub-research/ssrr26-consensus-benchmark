"""DBBS consensus strategy."""

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
class DBBS(ConsensusAlgorithm):

    name: ClassVar[str] = "DBBS"
    mu: float = C.DBBS_MU

    def update(self, uav: UAVAgent) -> None:
        """
        Distributed Bayesian Belief Sharing according to "Discrete collective estimation in swarm robotics with distributed Bayesian belief sharing", 2021
        Die Meinungen liegen im Intervall [-1, 1] und werden zu Beliefs ∈ [0, 1] gemappt, kombiniert und zurückgemappt.

        :param mu: Gewichtung der sozialen Meinung (zwischen 0 und 1) --> is to be set in constants.py
        """
        def opinion_to_belief(opinion):
            """
            Wandelt eine Meinung ∈ [-1, 1] in einen belief ∈ [0, 1] um.
            """
            return (opinion + 1) / 2

        def belief_to_opinion(belief):
            """
            Wandelt einen belief ∈ [0, 1] zurück in eine Meinung ∈ [-1, 1].
            """
            return 2 * belief - 1


        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})

            # Meinungen in beliefs ∈ [0, 1] umwandeln
            local_belief = opinion_to_belief(opinions[uav.name])

            foreign_beliefs = [
                opinion_to_belief(val)
                for agent, val in opinions.items()
                if agent != uav.name
            ]

            if not foreign_beliefs:
                continue  # kein sozialer Input

            social_belief = sum(foreign_beliefs) / len(foreign_beliefs)

            # Bayessche Fusion (gewichtete Multiplikation)
            fused_belief = (local_belief ** (1 - self.mu)) * (social_belief ** self.mu)
            fused_belief = min(max(fused_belief, 0.0), 1.0)  # normalisieren

            # Zurückwandeln in Meinung
            fused_opinion = round(belief_to_opinion(fused_belief), 3)

            # Nur aktualisieren wenn sich die Meinung signifikant ändert
            current_opinion = opinions.get(uav.name, 0.0)
            if abs(fused_opinion - current_opinion) > 0.01:
                # Meinung aktualisieren
                opinions[uav.name] = fused_opinion
                mine['opinions'] = opinions
                uav.comm_candidates.append(mine)  # neue Meinung kommunizieren

            # Konsenswert speichern
            mean_opinion = round(sum(opinions.values()) / len(opinions), 3)
            mine['consensus_value'] = mean_opinion
