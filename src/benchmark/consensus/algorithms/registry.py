"""Register and instantiate consensus strategies by their configuration names."""

from __future__ import annotations

from inspect import isabstract
from typing import Any, TypeVar

from .base import ConsensusAlgorithm

AlgorithmType = TypeVar("AlgorithmType", bound=type[ConsensusAlgorithm])
_ALGORITHMS: dict[str, type[ConsensusAlgorithm]] = {}


def register_algorithm(algorithm: AlgorithmType) -> AlgorithmType:
    """Register a concrete strategy class; also usable as a class decorator."""
    if not issubclass(algorithm, ConsensusAlgorithm) or isabstract(algorithm):
        raise TypeError("Algorithms must implement ConsensusAlgorithm.update().")
    name = getattr(algorithm, "name", "")
    if not name or name != name.upper():
        raise ValueError("An algorithm must define a non-empty uppercase name.")
    if name in _ALGORITHMS:
        raise ValueError(f"Algorithm already registered: {name}")
    _ALGORITHMS[name] = algorithm
    return algorithm


def algorithm_names() -> tuple[str, ...]:
    """Return available configuration names in registration order."""
    return tuple(_ALGORITHMS)


def create_algorithm(name: str, **parameters: Any) -> ConsensusAlgorithm:
    """Construct a fresh registered strategy with optional parameter overrides."""
    try:
        algorithm = _ALGORITHMS[name.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{name}'. Choose from: {algorithm_names()}"
        ) from None
    return algorithm(**parameters)
