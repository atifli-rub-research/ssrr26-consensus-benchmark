"""DBC consensus strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .base import ConsensusAlgorithm
from .registry import register_algorithm

if TYPE_CHECKING:
    from benchmark.consensus.core.agent import UAVAgent


@register_algorithm
@dataclass
class DBC(ConsensusAlgorithm):

    name: ClassVar[str] = "DBC"
    num_hypotheses: int = 2

    def update(self, uav: UAVAgent) -> None:

        def opinion_to_probability_distribution(opinion):
            """
            Convert opinion ∈ [-1, 1] to probability distribution over hypotheses
            For 2 hypotheses: [P(no_mine), P(mine)]
            """
            # Map opinion to probability of mine
            p_mine = (opinion + 1) / 2  # [-1,1] -> [0,1]
            p_mine = max(0.0, min(1.0, p_mine))  # Clamp to [0,1]

            if self.num_hypotheses == 2:
                return [1.0 - p_mine, p_mine]  # [P(no_mine), P(mine)]
            else:
                # For more hypotheses, distribute probability around the opinion
                probs = [0.0] * self.num_hypotheses
                if self.num_hypotheses > 2:
                    # Map to middle range for multi-hypothesis case
                    center_idx = int((opinion + 1) / 2 * (self.num_hypotheses - 1))
                    center_idx = max(0, min(self.num_hypotheses - 1, center_idx))
                    probs[center_idx] = 0.8
                    # Distribute remaining probability to neighbors
                    if center_idx > 0:
                        probs[center_idx - 1] = 0.1
                    if center_idx < self.num_hypotheses - 1:
                        probs[center_idx + 1] = 0.1
                else:
                    probs = [1.0 - p_mine, p_mine]
                return probs

        def probability_distribution_to_opinion(prob_dist):
            """
            Convert probability distribution back to opinion ∈ [-1, 1]
            """
            if self.num_hypotheses == 2:
                p_mine = prob_dist[1]  # P(mine)
                return 2 * p_mine - 1  # [0,1] -> [-1,1]
            else:
                # For multi-hypothesis, use weighted average
                weighted_sum = 0.0
                for i, prob in enumerate(prob_dist):
                    # Map hypothesis index to opinion value
                    hypothesis_value = 2 * i / (self.num_hypotheses - 1) - 1
                    weighted_sum += prob * hypothesis_value
                return weighted_sum

        def compute_shannon_entropy(prob_dist):
            """
            Compute Shannon entropy H = -∑(p_i * log2(p_i))
            """
            entropy = 0.0
            for prob in prob_dist:
                if prob > 0:  # Avoid log(0)
                    entropy -= prob * math.log2(prob)
            return entropy

        def compute_certainty(entropy):
            """
            Compute certainty Ω = 1/(1 + H)
            """
            return 1.0 / (1.0 + entropy)

        for mine in uav.mine_objects:
            opinions = mine.get('opinions', {})
            if not opinions or uav.name not in opinions:
                continue

            # Step 1: Belief Generation
            # Convert own opinion to probability distribution
            own_opinion = opinions.get(uav.name, 0.0)
            own_prob_dist = opinion_to_probability_distribution(own_opinion)
            own_entropy = compute_shannon_entropy(own_prob_dist)
            own_certainty = compute_certainty(own_entropy)

            # Collect neighbor beliefs and certainties
            neighbor_prob_dists = []
            neighbor_certainties = []

            for agent, opinion in opinions.items():
                if agent != uav.name:
                    neighbor_prob_dist = opinion_to_probability_distribution(opinion)
                    neighbor_entropy = compute_shannon_entropy(neighbor_prob_dist)
                    neighbor_certainty = compute_certainty(neighbor_entropy)

                    neighbor_prob_dists.append(neighbor_prob_dist)
                    neighbor_certainties.append(neighbor_certainty)

            if not neighbor_prob_dists:
                continue  # No neighbors to aggregate with

            # Step 3: Belief Balancing - Compute weights
            total_certainty = own_certainty + sum(neighbor_certainties)

            if total_certainty == 0:
                continue  # Avoid division by zero

            # Weight for own belief
            own_weight = own_certainty / total_certainty

            # Weights for neighbor beliefs
            neighbor_weights = [certainty / total_certainty for certainty in neighbor_certainties]

            # Step 4: Belief Aggregation - Fuse beliefs
            num_hyp = len(own_prob_dist)
            fused_prob_dist = [0.0] * num_hyp

            # Add weighted own belief
            for i in range(num_hyp):
                fused_prob_dist[i] += own_weight * own_prob_dist[i]

            # Add weighted neighbor beliefs
            for neighbor_dist, weight in zip(neighbor_prob_dists, neighbor_weights):
                for i in range(num_hyp):
                    fused_prob_dist[i] += weight * neighbor_dist[i]

            # Step 5: EEV-style Binary Decision
            # Convert fused probability distribution to expectation
            fused_opinion = probability_distribution_to_opinion(fused_prob_dist)

            current_opinion = opinions.get(uav.name, 0.0)

            # Collect all opinions for collective decision (like EEV)
            all_opinions = [current_opinion] + [opinions[agent] for agent in opinions if agent != uav.name]
            collective_expectation = sum(all_opinions) / len(all_opinions)

            certainty_weighted_opinions = []
            for agent, opinion in opinions.items():
                if agent == uav.name:
                    agent_certainty = own_certainty
                else:
                    # Find corresponding neighbor certainty
                    agent_certainty = 0.5  # Default if not found
                    for i, neighbor_opinion in enumerate([opinions[a] for a in opinions if a != uav.name]):
                        if neighbor_opinion == opinion:
                            agent_certainty = neighbor_certainties[i] if i < len(neighbor_certainties) else 0.5
                            break

                # Weight opinion by certainty
                certainty_weighted_opinions.append(opinion * agent_certainty)

            if certainty_weighted_opinions:
                weighted_collective_expectation = sum(certainty_weighted_opinions) / len(certainty_weighted_opinions)
            else:
                weighted_collective_expectation = collective_expectation

            # Use fused opinion with moderate EEV influence
            eev_influence = 0.2  # EEV influence
            eev_opinion = 1.0 if weighted_collective_expectation > 0 else (-1.0 if weighted_collective_expectation < 0 else 0.0)

            # Combine DBC fusion with EEV influence
            dbc_opinion = (fused_opinion + current_opinion) / 2
            new_opinion = (1 - eev_influence) * dbc_opinion + eev_influence * eev_opinion
            new_opinion = max(-1.0, min(1.0, new_opinion))
            new_opinion = round(new_opinion, 3)

            # Gradual convergence approach
            update_threshold = 0.01

            if abs(new_opinion - current_opinion) > update_threshold:
                opinions[uav.name] = new_opinion
                mine['opinions'] = opinions
                uav.comm_candidates.append(mine)

            # Update consensus value
            consensus_value = round(sum(opinions.values()) / len(opinions), 3)
            mine['consensus_value'] = consensus_value
