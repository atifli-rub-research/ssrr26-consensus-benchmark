"""Regression checks for the import-only consensus pipeline."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmark.consensus.algorithms import (
    ConsensusAlgorithm,
    LCP,
    MajorityVote,
    WMSR,
    algorithm_names,
    create_algorithm,
    register_algorithm,
    registry,
)
from benchmark.consensus.core.agent import UAVAgent
from benchmark.consensus.simulation.orchestration import ConsensusOrchestrator
from benchmark.consensus.pipeline.config import ConsensusPipelineConfig
from benchmark.consensus.pipeline import run_consensus_pipeline
from benchmark.consensus.simulation.swarm_simulation import SwarmSimulation


def write_predictions(root: Path) -> list[str]:
    """Create two tiles with a true mine and one possible false detection."""
    paths = []
    for index, confidences in enumerate(((0.9, None), (0.7, 0.6), (None, None)), start=1):
        path = root / f"uav_{index}.json"
        rows = []
        for tile_index, confidence in enumerate(confidences):
            boxes = ([{"class_id": 0, "confidence": confidence,
                       "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
                     if confidence is not None else [])
            gt_boxes = ([{"class_id": 0, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
                        if tile_index == 0 else [])
            ground_truth = {
                "tile_index": tile_index,
                "cell_id": f"tile_{tile_index}",
                "mine_present": tile_index == 0,
                "mine_count": len(gt_boxes),
                "boxes": gt_boxes,
            }
            rows.append({
                "drone_id": f"uav_{index}",
                "tile_index": tile_index,
                "ground_truth": ground_truth,
                "true_mine_count": len(gt_boxes),
                "true_mine_present": tile_index == 0,
                "num_detections": len(boxes),
                "detections": boxes,
            })
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths.append(str(path))
    return paths


class ConsensusPipelineTests(unittest.TestCase):
    def test_reports_preserve_simulation_results_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = write_predictions(root)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "output_dir": str(root / "output"),
                "defaults": {"seed": 42, "runs": 2, "wmsr_f": 1},
                "experiments": [
                    {"name": "Single & repeated", "input_files": [inputs[0], inputs[0]],
                     "algorithms": ["LCP", "WMSR", "NO_CONSENSUS"]},
                    {"name": "Comparison", "input_files": inputs,
                     "compare": True, "runs": 1},
                ],
            }), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_consensus_pipeline.run_pipeline(config_path), 0)

            output = root / "output"
            summaries = list(output.glob("pipeline_summary_*.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(len(summary["experiments"]), 2)
            single, comparison = summary["experiments"]
            self.assertEqual(single["experiment"], "Single__repeated")
            self.assertEqual(len(single["reports"]), 3)
            self.assertEqual(len(comparison["reports"]), 1)

            expected_starts = [
                {"uav_001__uav_1": 0, "uav_002__uav_1": 0},
                {"uav_001__uav_1": 1, "uav_002__uav_1": 0},
            ]
            for algorithm, path in zip(single["algorithms"], single["reports"]):
                report_path = Path(path)
                self.assertEqual(report_path.parent.name, algorithm)
                self.assertEqual(report_path.name, f"report_{algorithm}_2uav.json")
                report = json.loads(report_path.read_text())
                self.assertEqual(report["input_files"], [inputs[0], inputs[0]])
                self.assertEqual(report["start_combinations"], expected_starts)
                self.assertEqual(
                    (report["metrics"]["tp"], report["metrics"]["fp"], report["metrics"]["fn"]),
                    (1, 0, 0),
                )
                if algorithm == "NO_CONSENSUS":
                    self.assertEqual(
                        report["per_run_metrics"][0]["metric_aggregation"],
                        "per_drone_mean_counts",
                    )

            comparison_path = Path(comparison["reports"][0])
            self.assertEqual(comparison_path.name, "comparison_3uav.json")
            report = json.loads(comparison_path.read_text())
            self.assertEqual(
                [(row["algorithm"], row["tp"], row["fp"], row["fn"])
                 for row in report["comparison"]],
                [("LCP", 1, 1, 0), ("WMSR", 1, 0, 0),
                 ("CDCI", 1, 1, 0), ("DBBS", 1, 1, 0),
                 ("MLES", 1, 1, 0), ("EEV", 1, 1, 0),
                 ("DBC", 1, 1, 0), ("NO_CONSENSUS", 1, 0, 0),
                 ("MAJORITY_VOTE", 1, 0, 0)],
            )

    def test_no_save_still_writes_log_but_no_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = write_predictions(root)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "experiments": [{"input_files": [inputs[0]], "runs": 1}],
            }), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    run_consensus_pipeline.run_pipeline(config_path, output_dir=root / "output", no_save=True),
                    0,
                )
            self.assertEqual(len(list((root / "output/logs").glob("*.log"))), 1)
            self.assertEqual(list((root / "output").rglob("*.json")), [])

    def test_paths_remain_relative_to_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "output_dir": "relative-output",
                "experiments": [{"input_files": ["relative-input.json"]}],
            }), encoding="utf-8")
            config = ConsensusPipelineConfig(config_path, run_dir=root / "working")
            self.assertEqual(config.output_root(), (root / "working/relative-output").resolve())
            with self.assertRaisesRegex(SystemExit, str((root / "working/relative-input.json").resolve())):
                config.parse_experiment(config.experiments[0], 0)

    def test_only_imported_pipeline_entry_point_remains(self):
        self.assertFalse(hasattr(run_consensus_pipeline, "main"))
        self.assertIsNone(importlib.util.find_spec("benchmark.consensus.pipeline.run_consensus_sim"))

    def test_comparison_matches_individual_runs_with_shared_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_predictions(Path(tmp))
            orchestrator = ConsensusOrchestrator(paths=inputs, seed=7, n_runs=3)
            comparison = orchestrator.run_comparison()
            self.assertEqual(
                [row["algorithm"] for row in comparison["comparison"]],
                list(algorithm_names()),
            )
            for row in comparison["comparison"]:
                single = orchestrator.run_algorithm(row["algorithm"])
                self.assertEqual(single["start_combinations"], comparison["start_combinations"])
                self.assertEqual(single["metrics"], row)

    def test_comparison_forwards_configured_wmsr_fault_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_predictions(Path(tmp))
            orchestrator = ConsensusOrchestrator(paths=inputs, n_runs=1, wmsr_f=2)
            with patch(
                "benchmark.consensus.simulation.orchestration.SwarmSimulation",
                wraps=SwarmSimulation,
            ) as simulation:
                orchestrator.run_comparison()
            for call in simulation.call_args_list:
                self.assertEqual(call.kwargs["wmsr_f"], 2)


class ConsensusStrategyTests(unittest.TestCase):
    def test_majority_vote_counts_detection_presence_and_preserves_state(self):
        cases = [
            ("three_of_five", [-1.0, -1.0, -0.5, 0.2, 0.8], 1.0),
            ("two_of_five", [-1.0, -1.0, -1.0, 0.8, 0.9], -1.0),
            ("low_confidence", [-0.5, -0.5, -0.5, -1.0, -1.0], 1.0),
            ("tie", [-0.5, 0.8, -1.0, -1.0], -1.0),
            ("single_detection", [-0.999], 1.0),
            ("single_absence", [-1.0], -1.0),
        ]
        cases.extend(
            (f"{count}_of_five", [-0.5] * count + [-1.0] * (5 - count),
             1.0 if count >= 3 else -1.0)
            for count in range(6)
        )
        strategy = create_algorithm("MAJORITY_VOTE")
        self.assertIsInstance(strategy, MajorityVote)
        for label, values, expected in cases:
            with self.subTest(label=label):
                opinions = {f"uav_{i}": value for i, value in enumerate(values, 1)}
                original = dict(opinions)
                agent = UAVAgent("uav_1", {}, algorithms=[strategy])
                mine = {"opinions": opinions, "consensus_value": None}
                agent.mine_objects = [mine]

                for _ in range(3):
                    agent._update_consensus()
                    self.assertEqual(mine, {
                        "opinions": original, "consensus_value": expected,
                    })
                    self.assertIs(mine["opinions"], opinions)
                    self.assertEqual(opinions, original)
                    self.assertEqual(agent.comm_candidates, [])

    def test_majority_vote_updates_each_slot_and_skips_empty_opinions(self):
        agent = UAVAgent("uav_1", {})
        mine = {"opinions": {"uav_1": -0.5, "uav_2": 0.4, "uav_3": -1.0},
                "consensus_value": None}
        rejected = {"opinions": {"uav_1": -1.0, "uav_2": 0.9},
                    "consensus_value": None}
        empty = {"opinions": {}, "consensus_value": None}
        missing = {"consensus_value": None}
        agent.mine_objects = [mine, rejected, empty, missing]

        MajorityVote().update(agent)

        self.assertEqual(mine["consensus_value"], 1.0)
        self.assertEqual(rejected["consensus_value"], -1.0)
        self.assertEqual(empty, {"opinions": {}, "consensus_value": None})
        self.assertEqual(missing, {"consensus_value": None})
        self.assertEqual(agent.comm_candidates, [])

    def test_majority_vote_replaces_static_baseline_in_registry(self):
        self.assertIn("MAJORITY_VOTE", algorithm_names())
        self.assertNotIn("STATIC_MEAN", algorithm_names())
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            create_algorithm("STATIC_MEAN")

    def test_majority_vote_simulation_uses_all_raw_detector_opinions(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_predictions(Path(tmp))
            simulation = SwarmSimulation(inputs, algorithm="MAJORITY_VOTE", seed=7)
            multi = simulation.run_multi(n_runs=2)

            for run in multi.runs:
                for tile in run.final_tiles:
                    for slot in tile["slots"]:
                        raw = {
                            drone_id: UAVAgent.slot_opinion(
                                simulation.swarm_data[drone_id][tile["tile_index"]],
                                slot["slot_index"],
                            )
                            for drone_id in simulation.drone_ids
                        }
                        self.assertEqual(slot["opinions"], raw)
                        self.assertEqual(
                            slot["consensus_value"],
                            1.0 if sum(value > -1.0 for value in raw.values())
                            > len(raw) / 2 else -1.0,
                        )

    def test_abstract_and_duplicate_strategies_are_rejected(self):
        with self.assertRaises(TypeError):
            ConsensusAlgorithm()
        with self.assertRaises(TypeError):
            register_algorithm(ConsensusAlgorithm)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_algorithm(LCP)
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            create_algorithm("NOT_REGISTERED")

    def test_custom_strategy_runs_from_yaml_without_agent_dispatch_changes(self):
        class RejectAll(ConsensusAlgorithm):
            name = "TEST_REJECT_ALL"

            def update(self, uav):
                for mine in uav.mine_objects:
                    mine["consensus_value"] = -1.0

        with patch.dict(registry._ALGORITHMS), tempfile.TemporaryDirectory() as tmp:
            register_algorithm(RejectAll)
            root = Path(tmp)
            inputs = write_predictions(root)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "output_dir": str(root / "output"),
                "experiments": [{
                    "input_files": inputs,
                    "algorithms": [RejectAll.name],
                    "runs": 2,
                }],
            }), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_consensus_pipeline.run_pipeline(config_path), 0)
            reports = list((root / "output").rglob("report_TEST_REJECT_ALL_3uav.json"))
            self.assertEqual(len(reports), 1)
            metrics = json.loads(reports[0].read_text())["metrics"]
            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 0, 1))
            comparison = ConsensusOrchestrator(paths=inputs, n_runs=1).run_comparison()
            self.assertEqual(comparison["comparison"][-1]["algorithm"], RejectAll.name)

    def test_strategy_instances_are_isolated_between_agents_and_runs(self):
        class CountingStrategy(ConsensusAlgorithm):
            name = "TEST_COUNTING"
            instances = []

            def __init__(self):
                self.updates = 0
                self.instances.append(self)

            def update(self, uav):
                self.updates += 1

        with patch.dict(registry._ALGORITHMS), tempfile.TemporaryDirectory() as tmp:
            register_algorithm(CountingStrategy)
            inputs = write_predictions(Path(tmp))
            SwarmSimulation(inputs, algorithm=CountingStrategy.name).run_multi(n_runs=2)
            instances = CountingStrategy.instances
            self.assertEqual(len(instances), len(inputs) * 2)
            self.assertEqual(len({id(instance) for instance in instances}), len(instances))
            self.assertTrue(all(instance.updates > 0 for instance in instances))

    def test_agent_calls_strategies_in_order(self):
        calls = []

        class SetValue(ConsensusAlgorithm):
            name = "TEST_SET"

            def update(self, uav):
                calls.append(self.name)
                uav.mine_objects[0]["consensus_value"] = 0.25

        class DoubleValue(ConsensusAlgorithm):
            name = "TEST_DOUBLE"

            def update(self, uav):
                calls.append(self.name)
                uav.mine_objects[0]["consensus_value"] *= 2

        agent = UAVAgent("own", {}, algorithms=[SetValue(), DoubleValue()])
        local = {"opinions": {"own": 0.8}, "consensus_value": None}
        remote = {"opinions": {"peer": 0.5}, "consensus_value": None}
        agent.mine_objects = [local, remote]
        agent._update_consensus()
        self.assertEqual(calls, ["TEST_SET", "TEST_DOUBLE"])
        self.assertEqual(local["consensus_value"], 0.5)
        self.assertIsNone(remote["consensus_value"])
        self.assertEqual(agent.mine_objects, [local, remote])

    def test_agent_restores_slots_if_strategy_raises(self):
        class FailingStrategy(ConsensusAlgorithm):
            name = "TEST_FAILING"

            def update(self, uav):
                raise RuntimeError("update failed")

        agent = UAVAgent("own", {}, algorithms=[FailingStrategy()])
        original = [{"opinions": {"own": 0.8}}, {"opinions": {"peer": -1.0}}]
        agent.mine_objects = original
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            agent._update_consensus()
        self.assertIs(agent.mine_objects, original)

    def test_algorithm_parameters_are_owned_by_strategy_instances(self):
        strategy = create_algorithm("wmsr", f=2)
        self.assertIsInstance(strategy, WMSR)
        self.assertEqual(strategy.f, 2)
        with tempfile.TemporaryDirectory() as tmp:
            inputs = write_predictions(Path(tmp))
            simulation = SwarmSimulation(
                inputs, algorithm="WMSR", algorithm_parameters={"f": 2}
            )
            for agent in simulation._new_agents().values():
                self.assertEqual(agent.algorithms[0].f, 2)


if __name__ == "__main__":
    unittest.main()
