"""MLES consensus strategy."""

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
class MLES(ConsensusAlgorithm):

    name: ClassVar[str] = "MLES"
    lambda_value: float = C.MLES_LAMBDA

    def update(self, uav: UAVAgent) -> None:
        """
        Führt ein Maximum Likelihood Estimate Sharing (MLES) Update durch.
        Die eigene Meinung (opinion ∈ [-1, 1]) wird mit den Nachbarmeinungen verschmolzen,
        gewichtet nach Beobachtungsmenge m_j.
        """

        def to_belief(opinion):
            """Map opinion [-1, 1] → belief ∈ [0, 1]"""
            return (opinion + 1) / 2

        def to_opinion(belief):
            """Map belief ∈ [0, 1] → opinion ∈ [-1, 1]"""
            return 2 * belief - 1

        def omega(m_j, lam):
            """Gewichtung nach Paperformel"""
            return 1 / (1 + lam * (m_j ** 0.5))

        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})
            if not opinions or uav.name not in opinions:
                continue  # eigene Meinung nicht vorhanden → kein Update

            # Eigene Schätzung (belief) und Beobachtungsanzahl
            local_belief = to_belief(opinions[uav.name])

            fused_belief = local_belief  # initialer Wert
            total_weight = 1.0  # Eigengewicht ist 1

            # Nachbarn einbeziehen
            for agent, opinion in opinions.items():
                if agent == uav.name:
                    continue
                belief_j = to_belief(opinion)
                m_j = mine.get('observations', {}).get(agent, 1)
                w = omega(m_j, self.lambda_value)

                fused_belief += belief_j * w
                total_weight += w

            # Finaler belief (gewichteter Durchschnitt)
            final_belief = fused_belief / total_weight
            final_opinion = round(to_opinion(final_belief), 3)

            # Nur aktualisieren wenn sich die Meinung signifikant ändert
            current_opinion = opinions.get(uav.name, 0.0)
            if abs(final_opinion - current_opinion) > 0.01:
                # Meinung aktualisieren
                opinions[uav.name] = final_opinion
                mine['opinions'] = opinions
                # Für spätere Kommunikation vormerken
                uav.comm_candidates.append(mine)

            # Konsenswert
            consensus_value = round(sum(opinions.values()) / len(opinions), 3)
            mine['consensus_value'] = consensus_value
