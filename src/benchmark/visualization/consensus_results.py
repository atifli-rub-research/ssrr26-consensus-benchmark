#!/usr/bin/env python3
"""
Visualize consensus benchmark results.

This script reads a pipeline summary JSON produced by run_consensus_pipeline.py,
loads the referenced comparison_5uav.json files, flattens the metrics into a CSV,
and generates plots showing how each consensus algorithm behaves as:

  - corruption severity increases
  - number of corrupted UAVs increases
  - cumulative corruption type set changes

Example:
  visualize_consensus_results \
      results/consensus_results/volleyballfield/pipeline_summary_2026-05-26_16-48-56.json \
      --output-dir results/consensus_results/volleyballfield/plots

All algorithms present in the reports, including MAJORITY_VOTE, are plotted by
default. Use --algorithms to select a subset.

Notes:
  - Experiment names are expected to look like:
      baseline_5uav
      k1_s01_defocus
      k2_s05_defocus_gaussian_noise
      k5_s10_defocus_gaussian_noise_motion_blur_pixelate_snow

  - Severity s01 means 0.1, ..., s10 means 1.0.
  - k means the number of corrupted UAVs.
  - Because our the experiment design is cumulative, "per corruption type"
    plots are interpreted as "experiments in which this corruption is active",
    not isolated single-corruption ablations unless those experiments exist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_RE = re.compile(r"^k(?P<k>\d+)_s(?P<s>\d+)_(?P<corruptions>.+)$")

DEFAULT_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "n_pred_instances",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_corruption_tokens(tokens: list[str]) -> list[str]:
    """
    Normalize split experiment suffix tokens into corruption names.

    Example:
      ["defocus", "gaussian", "noise", "motion", "blur", "pixelate", "snow"]
      ->
      ["defocus", "gaussian_noise", "motion_blur", "pixelate", "snow"]
    """
    known = {
        ("gaussian", "noise"): "gaussian_noise",
        ("motion", "blur"): "motion_blur",
    }

    normalized: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in known:
            normalized.append(known[(tokens[i], tokens[i + 1])])
            i += 2
        else:
            normalized.append(tokens[i])
            i += 1
    return normalized

def parse_experiment_name(name: str) -> dict[str, Any]:
    """Parse experiment metadata from the systematic experiment name.

    Supports optional field suffixes like:
      baseline_5uav__volleyballfield
      k1_s01_defocus__grassfield
      k5_s10_defocus_gaussian_noise__rubblefield
    """
    full_name = name
    base_name = name.split("__", 1)[0] # remove field from base name

    if base_name == "baseline_5uav":
        return {
            "experiment": full_name,
            "experiment_base": base_name,
            "num_corrupted": 0,
            "fraction_corrupted": 0.0,
            "severity_idx": 0,
            "severity": 0.0,
            "corruption_set": "none",
            "active_corruptions": [],
        }

    match = EXPERIMENT_RE.match(base_name)
    if not match:
        raise ValueError(
            f"Could not parse experiment name: {name!r}. "
            "Expected baseline_5uav or k<num>_s<num>_<corruptions>, "
            "optionally followed by __<field>."
        )

    k = int(match.group("k"))
    s_idx = int(match.group("s"))
    corruption_set_raw = match.group("corruptions")
    active_corruptions = normalize_corruption_tokens(corruption_set_raw.split("_"))

    return {
        "experiment": full_name,
        "experiment_base": base_name,
        "num_corrupted": k,
        "fraction_corrupted": k / 5.0,
        "severity_idx": s_idx,
        "severity": s_idx / 10.0,
        "corruption_set": "_".join(active_corruptions),
        "active_corruptions": active_corruptions,
    }


def resolve_report_path(report_path: str, summary_path: Path, repo_root: Path | None) -> Path:
    """
    Resolve report paths robustly.

    The pipeline summary may contain absolute paths from your machine.
    This function supports:
      - existing absolute paths
      - paths relative to the summary file
      - remapping through --repo-root if the absolute path moved
    """
    path = Path(report_path)

    if path.exists():
        return path

    if not path.is_absolute():
        candidate = (summary_path.parent / path).resolve()
        if candidate.exists():
            return candidate

    if repo_root is not None:
        parts = path.parts
        if "results" in parts:
            idx = parts.index("results")
            suffix = Path(*parts[idx:])
            candidate = repo_root / suffix
            if candidate.exists():
                return candidate

    return path


def load_results(summary_path: Path, repo_root: Path | None = None) -> pd.DataFrame:
    summary = load_json(summary_path)
    experiments = summary.get("experiments", [])

    rows: list[dict[str, Any]] = []
    missing_reports: list[Path] = []

    for exp in experiments:
        exp_name = exp["experiment"]
        meta = parse_experiment_name(exp_name)

        reports = exp.get("reports", [])
        if not reports:
            print(f"[WARN] No reports listed for experiment {exp_name}")
            continue

        for report_raw in reports:
            report_path = resolve_report_path(report_raw, summary_path, repo_root)

            if not report_path.exists():
                missing_reports.append(report_path)
                continue

            report = load_json(report_path)

            for entry in report.get("comparison", []):
                row = {
                    **meta,
                    "algorithm": entry.get("algorithm"),
                    "metric_level": entry.get("metric_level", report.get("metric_level")),
                    "iou_threshold": entry.get("iou_threshold", report.get("iou_threshold")),
                    "n_tiles": entry.get("n_tiles"),
                    "n_runs": entry.get("n_runs", report.get("n_runs")),
                    "report_path": str(report_path),
                }

                for key, value in entry.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        row[key] = value

                rows.append(row)

    if missing_reports:
        print(f"[WARN] Missing {len(missing_reports)} report files.")
        print("[WARN] First few missing reports:")
        for path in missing_reports[:5]:
            print(f"  - {path}")

    if not rows:
        raise RuntimeError(
            "No rows loaded. Check that the report paths in the pipeline summary exist. "
            "If paths moved, pass --repo-root /path/to/ssrr26-consensus-benchmark."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["severity", "num_corrupted", "algorithm"],
        kind="stable",
    ).reset_index(drop=True)

    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_flat_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    csv_path = output_dir / "consensus_results_flat.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")
    return csv_path


def get_baseline_by_algorithm(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the clean-baseline metric value per algorithm."""
    baseline = df[
        (df["num_corrupted"] == 0)
        & (df["severity"] == 0.0)
        & (df["corruption_set"] == "none")
    ].copy()

    if baseline.empty:
        return pd.DataFrame(columns=["algorithm", metric])

    return baseline.groupby("algorithm", as_index=False)[metric].mean()


def get_grouped_baselines(
    baseline: pd.DataFrame,
    metric: str,
    decimals: int = 4,
) -> pd.DataFrame:
    """Group algorithms whose clean-baseline metric values are effectively equal."""
    if baseline.empty:
        return pd.DataFrame(columns=["baseline_value", "algorithms"])

    grouped = baseline.copy()
    grouped["baseline_rounded"] = grouped[metric].round(decimals)

    rows: list[dict[str, Any]] = []
    for rounded_value, group in grouped.groupby("baseline_rounded", sort=True):
        rows.append(
            {
                "baseline_value": group[metric].mean(),
                "baseline_rounded": rounded_value,
                "algorithms": sorted(group["algorithm"].astype(str).tolist()),
            }
        )

    return pd.DataFrame(rows)


def pick_representative_color(
    algorithms: list[str],
    algorithm_colors: dict[str, str],
    fallback_index: int,
) -> str:
    """Pick a baseline color that matches one algorithm in the baseline group."""
    for algorithm in algorithms:
        if algorithm in algorithm_colors:
            return algorithm_colors[algorithm]

    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return fallback_colors[fallback_index % len(fallback_colors)]


def plot_metric_vs_severity_by_k(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Plot metric vs severity for each fixed number of corrupted UAVs.

    The clean baseline is included as the natural first point at severity=0.0
    for each algorithm, instead of being drawn as a separate dashed line.
    """
    plot_dir = output_dir / "curves_by_num_corrupted" / metric
    ensure_output_dir(plot_dir)

    baseline = get_baseline_by_algorithm(df, metric)

    for k in sorted(k for k in df["num_corrupted"].unique() if k > 0):
        sub = df[df["num_corrupted"] == k].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x_ticks = {0.0}

        for algorithm in sorted(sub["algorithm"].dropna().unique()):
            alg = (
                sub[sub["algorithm"] == algorithm]
                .groupby("severity", as_index=False)[metric]
                .mean()
                .sort_values("severity")
            )

            base_row = baseline[baseline["algorithm"] == algorithm]
            if not base_row.empty:
                baseline_point = pd.DataFrame(
                    [
                        {
                            "severity": 0.0,
                            metric: base_row[metric].iloc[0],
                        }
                    ]
                )

                alg = (
                    pd.concat([baseline_point, alg], ignore_index=True)
                    .drop_duplicates(subset=["severity"], keep="first")
                    .sort_values("severity")
                )

            x_ticks.update(alg["severity"].tolist())

            ax.plot(
                alg["severity"],
                alg[metric],
                marker="o",
                label=algorithm,
            )

        ax.set_title(
            f"Mean {metric} vs severity, {k}/5 corrupted UAVs\n"
            "Mean over fields and corruption types; severity 0.0 is clean baseline"
        )
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(x_ticks))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out = plot_dir / f"{metric}_vs_severity_k{k}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)


def plot_metric_vs_num_corrupted_by_severity(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Plot metric vs number of corrupted UAVs for each fixed severity.

    For every non-zero severity curve, the clean baseline is included as the
    first point at num_corrupted=0. This avoids separate dashed baseline lines
    and keeps the legend limited to consensus algorithms.
    """
    plot_dir = output_dir / "curves_by_severity" / metric
    ensure_output_dir(plot_dir)

    baseline = get_baseline_by_algorithm(df, metric)

    for severity in sorted(df["severity"].unique()):
        sub = df[df["severity"] == severity].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x_ticks = {0}

        if severity == 0.0:
            # Severity 0.0 is represented by the clean baseline only.
            # We do not invent k1..k5 corrupted experiments at s00.
            for algorithm in sorted(baseline["algorithm"].dropna().unique()):
                base_row = baseline[baseline["algorithm"] == algorithm]
                if base_row.empty:
                    continue

                ax.plot(
                    [0],
                    [base_row[metric].iloc[0]],
                    marker="o",
                    label=algorithm,
                )

            ax.set_title(
                f"Clean baseline {metric}, severity=0.0\n"
                "Baseline only: 0/5 corrupted UAVs"
            )

        else:
            for algorithm in sorted(sub["algorithm"].dropna().unique()):
                alg = (
                    sub[sub["algorithm"] == algorithm]
                    .groupby("num_corrupted", as_index=False)[metric]
                    .mean()
                    .sort_values("num_corrupted")
                )

                base_row = baseline[baseline["algorithm"] == algorithm]
                if not base_row.empty:
                    baseline_point = pd.DataFrame(
                        [
                            {
                                "num_corrupted": 0,
                                metric: base_row[metric].iloc[0],
                            }
                        ]
                    )

                    alg = (
                        pd.concat([baseline_point, alg], ignore_index=True)
                        .drop_duplicates(subset=["num_corrupted"], keep="first")
                        .sort_values("num_corrupted")
                    )

                x_ticks.update(alg["num_corrupted"].tolist())

                ax.plot(
                    alg["num_corrupted"],
                    alg[metric],
                    marker="o",
                    label=algorithm,
                )

            ax.set_title(
                f"Mean {metric} vs number of corrupted UAVs, severity={severity:.1f}\n"
                "Mean over fields and corruption types; 0/5 corrupted UAVs is clean baseline"
            )

        ax.set_xlabel("Number of corrupted UAVs")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(x_ticks))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        sev_label = f"s{int(round(severity * 10)):02d}"
        out = plot_dir / f"{metric}_vs_num_corrupted_{sev_label}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)


def plot_algorithm_heatmaps(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    plot_dir = output_dir / "heatmaps" / metric
    ensure_output_dir(plot_dir)

    for algorithm in sorted(df["algorithm"].dropna().unique()):
        sub = df[df["algorithm"] == algorithm].copy()
        if sub.empty:
            continue

        pivot = sub.pivot_table(
            index="severity",
            columns="num_corrupted",
            values=metric,
            aggfunc="mean",
        ).sort_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")

        ax.set_title(
            f"{algorithm}: mean {metric}\n"
            "Mean over fields and corruption types"
        )
        ax.set_xlabel("Number of corrupted UAVs")
        ax.set_ylabel("Corruption severity")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)

        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                val = pivot.values[y, x]
                if not math.isnan(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7)

        fig.tight_layout()

        out = plot_dir / f"heatmap_{metric}_{algorithm}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)


def plot_algorithm_ranking_by_condition(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    ranking_delta: float = 0.0,
) -> None:
    """Write a CSV showing the best algorithm(s) per condition.

    If multiple algorithms are within ``ranking_delta`` of the best value, they
    are all reported as best algorithms. This avoids arbitrary tie-breaking,
    for example CDCI being selected simply because it appears first after
    sorting.
    """
    plot_dir = output_dir / "rankings"
    ensure_output_dir(plot_dir)

    if ranking_delta < 0:
        raise ValueError("ranking_delta must be non-negative")

    rows = []
    grouped = df.groupby(["num_corrupted", "severity"], dropna=False)

    for (k, severity), group in grouped:
        # Average repeated rows first, e.g. across fields, corruption types, or runs.
        # This makes the per-algorithm columns consistent with how the plots aggregate data.
        alg_values = (
            group
            .groupby("algorithm", as_index=False)[metric]
            .mean()
            .dropna(subset=[metric])
        )

        if alg_values.empty:
            continue

        best_value = alg_values[metric].max()

        # Include all algorithms whose metric is close enough to the best value.
        # The tiny epsilon avoids annoying floating-point edge cases.
        eps = 1e-12
        best_alg_values = (
            alg_values[alg_values[metric] >= best_value - ranking_delta - eps]
            .sort_values(by=[metric, "algorithm"], ascending=[False, True])
        )
        best_algorithms = best_alg_values["algorithm"].astype(str).tolist()

        row = {
            "num_corrupted": k,
            "severity": severity,
            "best_algorithm": ", ".join(best_algorithms),
            "n_best_algorithms": len(best_algorithms),
            "ranking_delta": ranking_delta,
            f"best_{metric}": best_value,
        }

        # Add one column per algorithm containing that algorithm's metric value.
        # Example columns for precision: CDCI, DBC, MLCS, ...
        for _, alg_row in alg_values.iterrows():
            row[str(alg_row["algorithm"])] = alg_row[metric]

        rows.append(row)

    rank_df = pd.DataFrame(rows)

    # Keep metadata columns first, then algorithm columns alphabetically.
    fixed_columns = [
        "num_corrupted",
        "severity",
        "best_algorithm",
        "n_best_algorithms",
        "ranking_delta",
        f"best_{metric}",
    ]
    algorithm_columns = sorted(c for c in rank_df.columns if c not in fixed_columns)
    rank_df = rank_df[fixed_columns + algorithm_columns]

    csv_path = plot_dir / f"best_algorithm_by_condition_{metric}.csv"
    rank_df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def plot_per_corruption_type_curves(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    plot_dir = output_dir / "per_active_corruption_type" / metric
    ensure_output_dir(plot_dir)

    exploded = df.copy()
    exploded["active_corruptions"] = exploded["active_corruptions"].apply(
        lambda xs: xs if isinstance(xs, list) else []
    )
    exploded = exploded.explode("active_corruptions")
    exploded = exploded.dropna(subset=["active_corruptions"])

    if exploded.empty:
        return

    for corruption in sorted(exploded["active_corruptions"].unique()):
        sub = exploded[exploded["active_corruptions"] == corruption].copy()

        fig, ax = plt.subplots(figsize=(9, 5))

        for algorithm in sorted(sub["algorithm"].dropna().unique()):
            alg = (
                sub[sub["algorithm"] == algorithm]
                .groupby("severity", as_index=False)[metric]
                .mean()
                .sort_values("severity")
            )
            ax.plot(alg["severity"], alg[metric], marker="o", label=algorithm)

        ax.set_title(
            f"{metric} vs severity when {corruption} is active\n"
            "Cumulative design: not isolated single-corruption effect\n"
            "Mean over fields and number of corrupted UAVs"
        )
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(sub["severity"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out = plot_dir / f"{metric}_active_{corruption}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)


def plot_corruption_set_curves(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    plot_dir = output_dir / "curves_by_corruption_set" / metric
    ensure_output_dir(plot_dir)

    baseline = get_baseline_by_algorithm(df, metric)

    for corruption_set in sorted(df["corruption_set"].unique()):
        sub = df[df["corruption_set"] == corruption_set].copy()
        if sub.empty or corruption_set == "none":
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        x_ticks = {0.0}

        for algorithm in sorted(sub["algorithm"].dropna().unique()):
            alg = (
                sub[sub["algorithm"] == algorithm]
                .groupby("severity", as_index=False)[metric]
                .mean()
                .sort_values("severity")
            )

            base_row = baseline[baseline["algorithm"] == algorithm]
            if not base_row.empty:
                base_value = base_row[metric].iloc[0]

                baseline_point = pd.DataFrame(
                    [
                        {
                            "severity": 0.0,
                            metric: base_value,
                        }
                    ]
                )

                alg = (
                    pd.concat([baseline_point, alg], ignore_index=True)
                    .drop_duplicates(subset=["severity"], keep="first")
                    .sort_values("severity")
                )

            x_ticks.update(alg["severity"].tolist())

            ax.plot(
                alg["severity"],
                alg[metric],
                marker="o",
                label=algorithm,
            )

        ax.set_title(
            f"Mean {metric} vs severity, corruption set: {corruption_set}\n"
            "Mean over fields and number of corrupted UAVs; severity 0.0 is clean baseline"
        )
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(x_ticks))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        safe_name = corruption_set.replace("/", "_")
        out = plot_dir / f"{metric}_corruption_set_{safe_name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)

def plot_summary_dashboard(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    plot_dir = output_dir / "overview"
    ensure_output_dir(plot_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm in sorted(df["algorithm"].dropna().unique()):
        alg = (
            df[df["algorithm"] == algorithm]
            .groupby("severity", as_index=False)[metric]
            .mean()
            .sort_values("severity")
        )
        ax.plot(alg["severity"], alg[metric], marker="o", label=algorithm)

    ax.set_title(
        f"Mean {metric} vs corruption severity\n"
        "Mean over fields, corruption types, and number of corrupted UAVs"
    )
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = plot_dir / f"average_{metric}_vs_severity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm in sorted(df["algorithm"].dropna().unique()):
        alg = (
            df[df["algorithm"] == algorithm]
            .groupby("num_corrupted", as_index=False)[metric]
            .mean()
            .sort_values("num_corrupted")
        )
        ax.plot(alg["num_corrupted"], alg[metric], marker="o", label=algorithm)

    ax.set_title(
        f"Mean {metric} vs number of corrupted UAVs\n"
        "Mean over fields, corruption types, and severities"
    )
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel(metric)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = plot_dir / f"average_{metric}_vs_num_corrupted.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_heatmaps_by_corruption_and_algorithm(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "heatmaps_by_corruption" / metric
    ensure_output_dir(plot_dir)

    # Ignore baseline because it has no active corruption and k=0 only
    sub_all = df[df["corruption_set"] != "none"].copy()

    for corruption_set in sorted(sub_all["corruption_set"].unique()):
        sub_corr = sub_all[sub_all["corruption_set"] == corruption_set].copy()
        if sub_corr.empty:
            continue

        for algorithm in sorted(sub_corr["algorithm"].dropna().unique()):
            sub = sub_corr[sub_corr["algorithm"] == algorithm].copy()
            if sub.empty:
                continue

            pivot = sub.pivot_table(
                index="severity",
                columns="num_corrupted",
                values=metric,
                aggfunc="mean",  # averages over fields/runs if repeated
            ).sort_index()

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")

            ax.set_title(
                f"{algorithm}: mean {metric}, corruption: {corruption_set}\n"
                "Mean over fields"
            )
            ax.set_xlabel("Number of corrupted UAVs")
            ax.set_ylabel("Corruption severity")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])

            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric)

            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    val = pivot.values[y, x]
                    if not math.isnan(val):
                        ax.text(
                            x,
                            y,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

            fig.tight_layout()

            safe_corr = corruption_set.replace("/", "_")
            safe_alg = str(algorithm).replace("/", "_")
            out = plot_dir / f"heatmap_{metric}_{safe_corr}_{safe_alg}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize systematic consensus benchmark results.")
    parser.add_argument("pipeline_summary", type=Path, help="Path to pipeline_summary_*.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots and CSVs should be saved. Default: <pipeline_summary_parent>/plots",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repo root used to remap absolute report paths if the summary was generated elsewhere.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metrics to plot. Default: {' '.join(DEFAULT_METRICS)}",
    )
    parser.add_argument(
        "--ranking-delta",
        type=float,
        default=0.0,
        help=(
            "Absolute tolerance for best-algorithm CSVs. "
            "Algorithms with metric >= best_metric - ranking_delta are all listed as best. "
            "Default: 0.0, which only groups exact ties."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Algorithms to plot, e.g. MAJORITY_VOTE NO_CONSENSUS. Default: all algorithms in the reports.",
    )
    parser.add_argument("--skip-heatmaps", action="store_true", help="Skip algorithm heatmaps.")
    parser.add_argument("--skip-curves", action="store_true", help="Skip line-curve plots.")
    parser.add_argument("--skip-per-corruption", action="store_true", help="Skip per-active-corruption plots.")

    args = parser.parse_args()

    summary_path = args.pipeline_summary.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else summary_path.parent / "plots"
    ensure_output_dir(output_dir)

    df = load_results(summary_path, repo_root=args.repo_root)
    if args.algorithms:
        available = set(df["algorithm"].dropna().unique())
        missing = [name for name in args.algorithms if name not in available]
        if missing:
            print(f"[WARN] Requested algorithms not found and will be skipped: {missing}")
        df = df[df["algorithm"].isin(args.algorithms)].copy()
        if df.empty:
            parser.error("No requested algorithms are available in the reports.")
    save_flat_csv(df, output_dir)

    available_metrics = [m for m in args.metrics if m in df.columns]
    missing_metrics = [m for m in args.metrics if m not in df.columns]

    if missing_metrics:
        print(f"[WARN] These requested metrics were not found and will be skipped: {missing_metrics}")

    for metric in available_metrics:
        print(f"[INFO] Plotting metric: {metric}")

        plot_summary_dashboard(df, metric, output_dir)
        plot_algorithm_ranking_by_condition(
            df,
            metric,
            output_dir,
            ranking_delta=args.ranking_delta,
        )

        if not args.skip_heatmaps:
            plot_algorithm_heatmaps(df, metric, output_dir)
            plot_heatmaps_by_corruption_and_algorithm(df, metric, output_dir)

        if not args.skip_curves:
            plot_metric_vs_severity_by_k(df, metric, output_dir)
            plot_metric_vs_num_corrupted_by_severity(df, metric, output_dir)
            plot_corruption_set_curves(df, metric, output_dir)

        if not args.skip_per_corruption:
            plot_per_corruption_type_curves(df, metric, output_dir)

    print(f"[DONE] Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
