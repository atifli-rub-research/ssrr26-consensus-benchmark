"""Run consensus experiments from a YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmark.consensus.output.json_writer import ConsensusJSONWriter
from benchmark.consensus.pipeline.config import ConsensusPipelineConfig
from benchmark.consensus.simulation.orchestration import ConsensusOrchestrator


def run_pipeline(
    config_path: str | Path,
    *,
    output_dir: Path | None = None,
    no_save: bool = False,
) -> int:
    """Run each configured experiment and write the existing report formats."""
    config = ConsensusPipelineConfig(config_path)
    output_root = config.output_root(output_dir)

    output = ConsensusJSONWriter(output_root)
    logger = output.logger
    logger.info("Config: {}", config.path)
    logger.info("Output root: {}", output_root)
    logger.info("Experiments: {}", len(config.experiments))

    try:
        all_summaries: list[dict[str, Any]] = []
        for exp_index, raw_exp in enumerate(
            tqdm(config.experiments, desc="Experiments", unit="experiment")
        ):
            exp = config.parse_experiment(raw_exp, exp_index)
            name = exp["name"]
            paths = exp["paths"]
            algorithms = exp["algorithms"]
            seed = exp["seed"]
            n_runs = exp["n_runs"]
            threshold = exp["threshold"]
            iou_threshold = exp["iou_threshold"]
            orchestrator = ConsensusOrchestrator(
                paths=paths,
                seed=seed,
                n_runs=n_runs,
                threshold=threshold,
                iou_threshold=iou_threshold,
                wmsr_f=exp["wmsr_f"],
            )

            logger.info("=" * 80)
            logger.info("Experiment: {}", name)
            logger.info("Experiment index: {}", exp_index)
            logger.info("UAV files: {}", len(paths))
            logger.info("Runs: {}, seed={}", n_runs, seed)
            logger.info("Algorithms: {}", ", ".join(algorithms))
            logger.info("Files: {}", ", ".join(Path(p).name for p in paths))

            exp_summary: dict[str, Any] = {
                "experiment": name,
                "input_files": paths,
                "seed": seed,
                "n_runs": n_runs,
                "algorithms": algorithms,
                "reports": [],
            }

            if exp["compare"]:
                logger.info("Running comparison for experiment: {}", name)
                report = orchestrator.run_comparison()
                output.log_start_combinations(report["start_combinations"])
                for row in report["comparison"]:
                    output.log_metrics_row(row, n_runs)

                if not no_save:
                    report_path = output.report_path(name, exp_index, None, len(paths))
                    output.write_report(report_path, report)
                    exp_summary["reports"].append(str(report_path))
            else:
                algorithm_bar = tqdm(
                    algorithms, desc=name, unit="algorithm", leave=False
                )
                for algorithm in algorithm_bar:
                    algorithm_bar.set_postfix_str(algorithm)
                    logger.info("Running algorithm: {}", algorithm)

                    report = orchestrator.run_algorithm(algorithm)
                    output.log_start_combinations(report["start_combinations"])
                    output.log_metrics_row(report["metrics"], n_runs)

                    if not no_save:
                        report_path = output.report_path(
                            name, exp_index, algorithm, len(paths)
                        )
                        output.write_report(report_path, report)
                        exp_summary["reports"].append(str(report_path))

            all_summaries.append(exp_summary)

        if not no_save:
            summary_path = output.write_summary(config.path, all_summaries)
            logger.info("=" * 80)
            logger.info("Pipeline summary: {}", summary_path)

        return 0
    finally:
        output.close()
