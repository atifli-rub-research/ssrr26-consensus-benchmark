"""Consensus strategies and their configuration registry.

Import new strategy modules here so their registration decorators run before
the pipeline parses configuration. No agent or simulation dispatch changes are
needed to add a strategy.
"""

from .base import ConsensusAlgorithm
from .registry import algorithm_names, create_algorithm, register_algorithm
from .lcp import LCP
from .wmsr import WMSR
from .cdci import CDCI
from .dbbs import DBBS
from .mles import MLES
from .eev import EEV
from .dbc import DBC
from .no_consensus import NoConsensus
from .majority_vote import MajorityVote

__all__ = [
    "ConsensusAlgorithm",
    "algorithm_names",
    "create_algorithm",
    "register_algorithm",
    "LCP",
    "WMSR",
    "CDCI",
    "DBBS",
    "MLES",
    "EEV",
    "DBC",
    "NoConsensus",
    "MajorityVote",
]
