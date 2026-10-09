"""Write consensus JSON reports, summaries, and associated run logs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class ConsensusJSONWriter:
    """Own output paths, JSON serialization, and the pipeline's file log sink."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.timestamp = self._timestamp()
        self.logger = logger
        self._sink_id = self._setup_logger()

    def close(self) -> None:
        """Close this run's file sink without disturbing other Loguru sinks."""
        self.logger.remove(self._sink_id)

    def report_path(
        self, name: str, index: int, algorithm: str | None, n_uavs: int
    ) -> Path:
        base = self.output_root / name / self.timestamp / f"experiment_no_{index}"
        if algorithm is None:
            return base / "comparison" / f"comparison_{n_uavs}uav.json"
        return base / algorithm / f"report_{algorithm}_{n_uavs}uav.json"

    def write_report(self, path: Path, report: dict[str, Any]) -> None:
        self._write_json(path, report)
        self.logger.info("Report: {}", path)

    def write_summary(
        self, config_path: Path, experiments: list[dict[str, Any]]
    ) -> Path:
        summary_path = self.output_root / f"pipeline_summary_{self._timestamp()}.json"
        self._write_json(
            summary_path,
            {
                "config": str(config_path),
                "output_root": str(self.output_root),
                "experiments": experiments,
            },
        )
        return summary_path

    def log_start_combinations(self, combos: list[dict[str, int]]) -> None:
        self.logger.info("Start tiles per run:")
        for index, combo in enumerate(combos):
            parts = ", ".join(f"{key}={value}" for key, value in sorted(combo.items()))
            self.logger.info("  run {:2d}: {}", index, parts)

    def log_metrics_row(self, metrics: dict[str, Any], n_runs: int) -> None:
        mean_note = f", mean over {n_runs} runs" if n_runs > 1 else ""
        self.logger.info(
            "{:<8} acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} TP={} FP={} FN={}{}",
            metrics["algorithm"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
            mean_note,
        )

    @staticmethod
    def _timestamp() -> str:
        """Filesystem-friendly timestamp with seconds resolution."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _setup_logger(self) -> int:
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"consensus_pipeline_{self.timestamp}.log"
        sink_id = self.logger.add(
            log_path,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            encoding="utf-8",
        )
        self.logger.info("Logging to {}", log_path)
        return sink_id
