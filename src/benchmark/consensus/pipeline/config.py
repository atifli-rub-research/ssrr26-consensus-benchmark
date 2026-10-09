"""YAML configuration and experiment parsing for the consensus pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from benchmark.consensus.algorithms import algorithm_names
from benchmark.consensus.core import constants as C


class ConsensusPipelineConfig:
    """Load YAML, merge defaults, and resolve one experiment at a time.

    Relative paths use run_dir, which defaults to the working directory from
    which the pipeline is launched.
    """

    _DEFAULTS: dict[str, Any] = {
        "seed": 42,
        "runs": None,
        "threshold": None,
        "iou_threshold": None,
        "wmsr_f": None,
        "algorithms": ["LCP"],
        "compare": False,
    }

    def __init__(self, path: str | Path, *, run_dir: Path | None = None):
        self.path = Path(path).resolve()
        self.run_dir = Path.cwd().resolve() if run_dir is None else run_dir
        self.data = self._load_yaml()
        self.defaults = self.data.get("defaults", {}) or {}
        self.experiments = self.data.get("experiments")
        if not isinstance(self.experiments, list) or not self.experiments:
            raise SystemExit("Config YAML must contain a non-empty `experiments:` list.")

    def output_root(self, override: Path | None = None) -> Path:
        if override is not None:
            return override.resolve()
        configured_output = self.data.get("output_dir", C.OUTPUT_FILES_DIR)
        return self._resolve_path(configured_output)

    def parse_experiment(self, raw_exp: dict[str, Any], index: int) -> dict[str, Any]:
        if not isinstance(raw_exp, dict):
            raise SystemExit(f"Experiment entry {index} must be a mapping.")

        exp_cfg = self._merge_defaults(raw_exp)
        name = self._safe_name(str(exp_cfg.get("name", f"experiment_{index:03d}")))
        paths = self._collect_input_files(exp_cfg)
        algorithms = self._normalise_algorithms(exp_cfg)
        return {
            "name": name,
            "paths": paths,
            "algorithms": algorithms,
            "seed": int(exp_cfg.get("seed", 42)),
            "n_runs": C.effective_simulation_runs(exp_cfg.get("runs")),
            "threshold": exp_cfg.get("threshold"),
            "iou_threshold": exp_cfg.get("iou_threshold"),
            "wmsr_f": exp_cfg.get("wmsr_f"),
            "compare": exp_cfg.get("compare", False),
        }

    def _load_yaml(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise SystemExit(f"Config file does not exist: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise SystemExit("Config YAML must contain a mapping at the top level.")

        return data

    def _resolve_path(self, path_like: str | Path) -> Path:
        """Resolve relative paths against this configuration's run directory."""
        p = Path(path_like).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.run_dir / p).resolve()

    def _merge_defaults(self, exp: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(self._DEFAULTS)
        merged.update(self.defaults)
        merged.update(exp or {})
        return merged

    @staticmethod
    def _normalise_algorithms(exp_cfg: dict[str, Any]) -> list[str]:
        """
        Determine which algorithms to run.

        Supported config forms:
          algorithms: [LCP, WMSR]
          algorithm: WMSR
          compare: true       # expands to all algorithms
        """
        if exp_cfg.get("compare", False):
            return list(algorithm_names())

        if "algorithm" in exp_cfg and "algorithms" not in exp_cfg:
            algorithms = [exp_cfg["algorithm"]]
        else:
            algorithms = exp_cfg.get("algorithms", ["LCP"])

        if isinstance(algorithms, str):
            algorithms = [algorithms]

        if not isinstance(algorithms, list) or not algorithms:
            raise SystemExit(
                "`algorithms` must be a non-empty list, or `algorithm` must be set."
            )

        invalid = [a for a in algorithms if a not in algorithm_names()]
        if invalid:
            raise SystemExit(
                f"Unknown consensus algorithm(s): {invalid}. "
                f"Available: {list(algorithm_names())}"
            )

        return algorithms

    def _collect_input_files(self, exp_cfg: dict[str, Any]) -> list[str]:
        files = exp_cfg.get("input_files") or exp_cfg.get("predictions")
        if not files:
            raise SystemExit(
                f"Experiment `{exp_cfg.get('name', '<unnamed>')}` has no input files. "
                "Add `input_files: [...]` to the experiment."
            )

        if not isinstance(files, list):
            raise SystemExit("`input_files` must be a list of JSON file paths.")

        resolved: list[str] = []

        for f in files:
            p = self._resolve_path(f)

            if not p.is_file():
                raise SystemExit(
                    f"Input file does not exist for experiment "
                    f"`{exp_cfg.get('name', '<unnamed>')}`: {p}"
                )

            resolved.append(str(p))

        if len(resolved) > C.MAX_UAV_COUNT:
            logger.warning(
                f"Warning: {len(resolved)} UAV files found "
                f"(recommended max {C.MAX_UAV_COUNT}).",
            )

        if len(resolved) == 1:
            logger.info(
                "Note: only one UAV file. Consensus still runs, but this is not really a swarm.",
            )

        return resolved

    @staticmethod
    def _safe_name(name: str) -> str:
        keep = []

        for ch in name.strip():
            if ch.isalnum() or ch in {"-", "_"}:
                keep.append(ch)
            elif ch.isspace():
                keep.append("_")

        return "".join(keep) or "experiment"
