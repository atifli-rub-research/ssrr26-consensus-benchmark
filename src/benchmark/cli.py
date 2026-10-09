"""Console entry points for the benchmark pipelines."""

import argparse
from pathlib import Path


def execute_observations() -> None:
    parser = argparse.ArgumentParser(description="Run an observation experiment.")
    parser.add_argument("config", type=Path, help="Path to the observation YAML config.")
    args = parser.parse_args()

    from benchmark.observations.pipeline.run_observation_pipeline import run_pipeline

    run_pipeline(str(args.config))


def execute_consensus() -> int:
    parser = argparse.ArgumentParser(description="Run consensus experiments.")
    parser.add_argument("config", type=Path, help="Path to the consensus YAML config.")
    args = parser.parse_args()

    from benchmark.consensus.pipeline.run_consensus_pipeline import run_pipeline

    return run_pipeline(args.config)


def visualize_paper_figures() -> None:
    """Generate publication-ready figures from consensus benchmark results."""
    from benchmark.visualization.paper_figures import main

    main()


def visualize_consensus_results() -> None:
    """Generate diagnostic plots from a consensus pipeline summary."""
    from benchmark.visualization.consensus_results import main

    main()


def visualize_uav_predictions() -> None:
    """Visualize per-UAV prediction records and majority-vote consensus."""
    from benchmark.visualization.uav_predictions import main

    main()
