"""Check that installed commands forward the supplied config path."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.cli import execute_consensus, execute_observations


class CommandTests(unittest.TestCase):
    def test_observations_forwards_config(self):
        config = "configs/observation_experiment_configs/example.yaml"
        with patch.object(sys, "argv", ["execute_observations", config]):
            with patch("benchmark.observations.pipeline.run_observation_pipeline.run_pipeline") as run:
                execute_observations()
        run.assert_called_once_with(config)

    def test_consensus_forwards_config(self):
        config = "configs/consensus_experiment_configs/example.yaml"
        with patch.object(sys, "argv", ["execute_consensus", config]):
            with patch("benchmark.consensus.pipeline.run_consensus_pipeline.run_pipeline", return_value=0) as run:
                self.assertEqual(execute_consensus(), 0)
        run.assert_called_once_with(Path(config))

    def test_project_scripts_are_registered(self):
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        self.assertIn('execute_observations = "benchmark.cli:execute_observations"', pyproject)
        self.assertIn('execute_consensus = "benchmark.cli:execute_consensus"', pyproject)


if __name__ == "__main__":
    unittest.main()
