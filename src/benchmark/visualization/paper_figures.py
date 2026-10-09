#!/usr/bin/env python3
"""
Generate compact, paper-ready figures for the RIVA multi-UAV consensus benchmark.

This script is intended to complement visualize_consensus_results.py. The original
script is useful as a diagnostic plot factory; this one focuses on fewer, denser
figures suitable for an 8-page paper.

It reads a pipeline_summary_*.json and resolves the referenced
comparison_5uav.json files.

Recommended usage from repo root:

  visualize_paper_figures \
      results/consensus_results/all_fields_mixed_corruption/pipeline_summary_2026-06-07_18-57-02.json

By default, figures and CSV summaries are written to a paper_figures directory
next to the pipeline summary. Optional arguments can override the output path,
report-path remapping, and plotting defaults.
MAJORITY_VOTE is included by default when its results are present in the reports.

Generated figures:
  - paper_panel_vs_severity.png/pdf
  - paper_panel_vs_num_corrupted.png/pdf
  - f1_variability_heatmaps.png/pdf
  - f1_small_multiples_combined.png/pdf
  - consensus_gain_heatmap_<metric>.png/pdf
  - best_algorithm_map_<metric>.png/pdf
  - robustness_auc_<metric>.png/pdf
  - precision_recall_tradeoff.png/pdf
  - fp_fn_tradeoff.png/pdf
  - algorithm_mean_metrics_table.csv
  - robustness_auc_<metric>.csv
  - breakdown_thresholds_<metric>.csv
  - best_algorithm_by_condition_<metric>.csv

Notes:
  - The script assumes a 5-UAV benchmark by default, but can infer the maximum
    num_corrupted present in the data.
  - It saves both PNG and PDF by default.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Avoid Type 3 fonts in Matplotlib-generated PDFs.
# IEEE submission systems often reject Type 3 fonts.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
})

EXPERIMENT_RE = re.compile(r"^k(?P<k>\d+)_s(?P<s>\d+)_(?P<corruptions>.+)$")

DEFAULT_PANEL_METRICS = ["f1", "accuracy", "precision", "recall"]
DEFAULT_SELECTED_ALGORITHMS = ["CDCI", "DBBS", "DBC", "EEV", "LCP", "MLES", "WMSR", "MAJORITY_VOTE", "NO_CONSENSUS"]
DEFAULT_TABLE_METRICS = ["accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "n_pred_instances"]
HIGHER_IS_BETTER = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "tp": True,
    "n_pred_instances": False,
    "fp": False,
    "fn": False,
}

ALGORITHM_COLORS = {
    "NO_CONSENSUS": "#1C1C1C",  # dark gray
    "CDCI": "#1f77b4",          # blue
    "DBBS": "#8c564b",          # brown
    "DBC": "#2ca02c",           # green
    "EEV": "#17becf",           # cyan
    "LCP": "#9467bd",           # purple
    "MLES": "#ff7f0e",          # orange
    "WMSR": "#e377c2",          # pink
    "MAJORITY_VOTE": "#7F7F7F", # gray
}

ALGORITHM_LINESTYLES = {
    # Use distinct line styles in addition to color so line plots remain
    # distinguishable for color-blind readers and in grayscale printouts.
    "NO_CONSENSUS": "--",
    "CDCI": "-",
    "DBBS": ":",
    "DBC": "-.",
    "EEV": (0, (3, 1, 1, 1)),
    "LCP": (0, (5, 1)),
    "MLES": (0, (1, 1)),
    "WMSR": (0, (5, 2, 1, 2)),
    "MAJORITY_VOTE": (0, (7, 2)),
}

PRETTY_METRIC = {
    "accuracy": "Jaccard accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
    "tp": "True positives",
    "fp": "False positives",
    "fn": "False negatives",
    "n_pred_instances": "Predicted instances",
}

# Larger, publication-friendly text sizes. These values apply globally to all
# generated figures, including axes labels, tick labels, legends, and colorbars.
FIGURE_FONT_SIZES = {
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
}

plt.rcParams.update(FIGURE_FONT_SIZES)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 300, formats: Iterable[str] = ("png", "pdf")) -> None:
    ensure_output_dir(output_dir)
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Wrote {out}")


def normalize_corruption_tokens(tokens: list[str]) -> list[str]:
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


def parse_experiment_name(name: str, default_num_uavs: int = 5) -> dict[str, Any]:
    full_name = name
    base_name = name.split("__", 1)[0]

    if base_name == f"baseline_{default_num_uavs}uav" or base_name.startswith("baseline_"):
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
            "Expected baseline_<n>uav or k<num>_s<num>_<corruptions>, "
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
        "fraction_corrupted": k / float(default_num_uavs),
        "severity_idx": s_idx,
        "severity": s_idx / 10.0,
        "corruption_set": "_".join(active_corruptions),
        "active_corruptions": active_corruptions,
    }


def resolve_report_path(report_path: str, summary_path: Path, repo_root: Path | None) -> Path:
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


def load_results_from_summary(summary_path: Path, repo_root: Path | None = None, default_num_uavs: int = 5) -> pd.DataFrame:
    summary = load_json(summary_path)
    experiments = summary.get("experiments", [])

    rows: list[dict[str, Any]] = []
    missing_reports: list[Path] = []

    for exp in experiments:
        exp_name = exp["experiment"]
        meta = parse_experiment_name(exp_name, default_num_uavs=default_num_uavs)

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

    return clean_dataframe(pd.DataFrame(rows), default_num_uavs=default_num_uavs)


def parse_list_like_active_corruptions(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "[]"}:
            return []
        # Handles strings produced by CSV serialization of Python lists.
        stripped = stripped.strip("[]")
        parts = [p.strip().strip("'\"") for p in stripped.split(",")]
        return [p for p in parts if p]
    return []


def clean_dataframe(df: pd.DataFrame, default_num_uavs: int = 5) -> pd.DataFrame:
    df = df.copy()

    numeric_candidates = [
        "num_corrupted",
        "fraction_corrupted",
        "severity_idx",
        "severity",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "n_pred_instances",
        "n_tiles",
        "n_runs",
        "iou_threshold",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fraction_corrupted" not in df.columns and "num_corrupted" in df.columns:
        df["fraction_corrupted"] = df["num_corrupted"] / float(default_num_uavs)

    if "active_corruptions" in df.columns:
        df["active_corruptions"] = df["active_corruptions"].apply(parse_list_like_active_corruptions)
    else:
        df["active_corruptions"] = [[] for _ in range(len(df))]

    if "corruption_set" not in df.columns:
        df["corruption_set"] = df["active_corruptions"].apply(lambda xs: "_".join(xs) if xs else "none")

    if "algorithm" in df.columns:
        df["algorithm"] = df["algorithm"].astype(str)

    sort_cols = [c for c in ["severity", "num_corrupted", "algorithm"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)

    return df


def available_algorithms(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    present = sorted(df["algorithm"].dropna().astype(str).unique().tolist())
    if requested:
        chosen = [a for a in requested if a in present]
        missing = [a for a in requested if a not in present]
        if missing:
            print(f"[WARN] Requested algorithms not found and will be skipped: {missing}")
        return chosen
    return [a for a in DEFAULT_SELECTED_ALGORITHMS if a in present] or present


def metric_label(metric: str) -> str:
    return PRETTY_METRIC.get(metric, metric.replace("_", " ").title())


def is_higher_better(metric: str) -> bool:
    return HIGHER_IS_BETTER.get(metric, True)


def algorithm_color(algorithm: str) -> str:
    # Fall back to black for unexpected algorithm names instead of letting
    # Matplotlib cycle colors differently across figures.
    return ALGORITHM_COLORS.get(str(algorithm), "#333333")


def algorithm_linestyle(algorithm: str) -> str:
    return ALGORITHM_LINESTYLES.get(str(algorithm), "-")

def best_algorithm_display_label(
    best_algorithm: Any,
    all_algorithms: list[str] | set[str] | None = None,
    baseline_algorithm: str = "NO_CONSENSUS",
    max_display_algorithms: int = 3,
) -> tuple[str, str | None]:
    """Return compact main/sub labels for best-algorithm heatmap cells.

    Returns:
      (main_label, sub_label)

    Rules:
      - If every algorithm is tied, including the baseline:
          ("No improvement", None)
      - If every consensus algorithm is tied, excluding the baseline:
          ("Consensus tie", "CDCI, DBBS, ...")
      - If many algorithms are tied:
          ("N-way tie", "A, B, C, ...")
      - Otherwise:
          ("A / B / C", None)
    """
    if isinstance(best_algorithm, (list, tuple, set)):
        winners = {str(a).strip() for a in best_algorithm if str(a).strip()}
    else:
        winners = {
            part.strip()
            for part in str(best_algorithm).split(",")
            if part.strip()
        }

    if not winners:
        return "", None

    preferred_order = [
        "CDCI",
        "DBBS",
        "DBC",
        "EEV",
        "LCP",
        "MLES",
        "WMSR",
        "MAJORITY_VOTE",
        "NO_CONSENSUS",
    ]
    order = {name: i for i, name in enumerate(preferred_order)}
    ordered_winners = sorted(winners, key=lambda a: (order.get(a, 999), a))

    if all_algorithms is not None:
        all_set = {str(a).strip() for a in all_algorithms if str(a).strip()}
    else:
        all_set = set(preferred_order)

    consensus_set = all_set - {baseline_algorithm}

    if winners == all_set:
        return "No improvement", None

    if winners == consensus_set:
        return "All consensus tie", None

    if baseline_algorithm in winners and len(winners) > 1:
        return "No clear gain", ", ".join(ordered_winners)

    if len(winners) > max_display_algorithms:
        return f"{len(winners)}-way tie", ", ".join(ordered_winners)

    return ",".join(ordered_winners), None

def get_baseline_by_algorithm(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = df[
        (df["num_corrupted"] == 0)
        & (df["severity"] == 0.0)
        & (df["corruption_set"] == "none")
    ].copy()

    if baseline.empty:
        # Fallback for CSVs where corruption_set is absent or baseline labels changed.
        baseline = df[(df["num_corrupted"] == 0) & (df["severity"] == 0.0)].copy()

    if baseline.empty:
        return pd.DataFrame(columns=["algorithm", metric])

    return baseline.groupby("algorithm", as_index=False)[metric].mean()


def add_clean_baseline_points(grouped: pd.DataFrame, baseline: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if baseline.empty:
        return grouped

    baseline_points = baseline.rename(columns={y_col: y_col}).copy()
    baseline_points[x_col] = 0.0 if x_col == "severity" else 0
    baseline_points = baseline_points[["algorithm", x_col, y_col]]

    merged = pd.concat([baseline_points, grouped], ignore_index=True)
    return merged.drop_duplicates(subset=["algorithm", x_col], keep="first").sort_values(["algorithm", x_col])


def _group_panel_metric(df: pd.DataFrame, x_col: str, metric: str) -> pd.DataFrame:
    """Return mean and cross-condition standard deviation for each algorithm/x.

    Sample std describes variation across marginalized benchmark conditions,
    not run-to-run variance or statistical uncertainty. Singletons use zero std.
    """
    grouped = (
        df.groupby(["algorithm", x_col])[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    baseline = df[
        (df["num_corrupted"] == 0)
        & (df["severity"] == 0.0)
        & (df["corruption_set"] == "none")
    ]
    if baseline.empty:
        baseline = df[(df["num_corrupted"] == 0) & (df["severity"] == 0.0)]
    if not baseline.empty:
        baseline_points = (
            baseline.groupby(["algorithm", x_col])[metric]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        grouped = pd.concat([baseline_points, grouped], ignore_index=True)
        grouped = grouped.drop_duplicates(["algorithm", x_col], keep="first")
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped.sort_values(["algorithm", x_col])


def plot_metric_panel(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[str],
    output_dir: Path,
    stem: str,
    algorithms: list[str],
    title: str,
) -> None:
    n = len(metrics)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10.5, 7.2), squeeze=False)

    for ax, metric in zip(axes.ravel(), metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            print(f"[WARN] Metric {metric!r} not available for panel {stem}.")
            continue

        sub = df[df["algorithm"].isin(algorithms)].copy()
        grouped = _group_panel_metric(sub, x_col, metric)

        for algorithm in algorithms:
            alg = grouped[grouped["algorithm"] == algorithm].sort_values(x_col)
            if alg.empty:
                continue
            x = alg[x_col].to_numpy(dtype=float)
            mean = alg["mean"].to_numpy(dtype=float)
            color = algorithm_color(algorithm)
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2.2,
                markersize=5,
                label=algorithm,
                color=color,
                linestyle=algorithm_linestyle(algorithm),
                zorder=2,
            )
        ax.set_xlabel("Corruption severity" if x_col == "severity" else "Number of corrupted UAVs")
        ax.set_ylabel(metric_label(metric))
        if x_col == "num_corrupted":
            max_k = int(df["num_corrupted"].max())
            ax.set_xticks(list(range(max_k + 1)))
        else:
            ticks = sorted(df[x_col].dropna().unique())
            if 0.0 not in ticks:
                ticks = [0.0] + ticks
            ax.set_xticks(ticks)
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_paper_severity_panel(df: pd.DataFrame, output_dir: Path, algorithms: list[str], metrics: list[str]) -> None:
    plot_metric_panel(
        df=df,
        x_col="severity",
        metrics=metrics,
        output_dir=output_dir,
        stem="paper_panel_vs_severity",
        algorithms=algorithms,
        title="Consensus performance vs corruption severity",
    )


def plot_paper_num_corrupted_panel(df: pd.DataFrame, output_dir: Path, algorithms: list[str], metrics: list[str]) -> None:
    plot_metric_panel(
        df=df,
        x_col="num_corrupted",
        metrics=metrics,
        output_dir=output_dir,
        stem="paper_panel_vs_num_corrupted",
        algorithms=algorithms,
        title="Consensus performance vs number of corrupted UAVs",
    )


def _selected_f1_data(df: pd.DataFrame, algorithms: list[str]) -> pd.DataFrame:
    """Select paper algorithms, warning when F1 variability cannot be plotted."""
    selected = df[df["algorithm"].isin(algorithms)]
    if "f1" not in selected.columns or not np.isfinite(selected["f1"]).any():
        print("[WARN] F1 unavailable for selected algorithms; skipping F1 cross-condition variability figure.")
        return pd.DataFrame()
    return selected


def plot_f1_variability_heatmaps(
    df: pd.DataFrame, output_dir: Path, algorithms: list[str],
) -> None:
    """Compare cross-condition F1 standard deviation along both benchmark axes."""
    selected = _selected_f1_data(df, algorithms)
    if selected.empty:
        return
    pivots = []
    for x_col in ("severity", "num_corrupted"):
        grouped = _group_panel_metric(selected, x_col, "f1")
        # Missing algorithm/x combinations stay missing, rather than zero std.
        pivot = grouped.pivot(index="algorithm", columns=x_col, values="std")
        pivots.append(pivot.reindex(index=algorithms).sort_index(axis=1))
    finite = np.concatenate([p.to_numpy().ravel() for p in pivots])
    finite = finite[np.isfinite(finite)]
    vmax = float(finite.max()) if finite.size else 0.0
    norm = plt.Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, layout="constrained")
    for ax, pivot, label in zip(axes, pivots, ("(a) Severity", "(b) Corrupted UAVs")):
        im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), aspect="auto", cmap="viridis", norm=norm)
        ax.set_xlabel(label)
        ax.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(algorithms)), labels=algorithms)
        for y, x in np.ndindex(pivot.shape):
            value = pivot.iloc[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center",
                        fontsize=10, color=heatmap_text_color(im, value))
    fig.colorbar(im, ax=axes, shrink=0.9).set_label("Cross-condition F1 standard deviation")
    save_figure(fig, output_dir, "f1_variability_heatmaps")
    plt.close(fig)


def plot_f1_small_multiples_combined(
    df: pd.DataFrame, output_dir: Path, algorithms: list[str],
) -> None:
    """Compare both F1 relationships with ±1 cross-condition standard deviation.

    Bands describe marginalized benchmark conditions, not confidence intervals
    or run-to-run variance. Each relationship retains its own x-axis domain.
    """
    selected = _selected_f1_data(df, algorithms)
    if selected.empty:
        return
    grouped = {x_col: _group_panel_metric(selected, x_col, "f1")
               for x_col in ("severity", "num_corrupted")}
    severity_ticks = sorted(grouped["severity"]["severity"].dropna().unique())
    max_k = int(grouped["num_corrupted"]["num_corrupted"].max())
    # Enlarge content locally while reclaiming space around the subplot grid.
    content_scale = 1.10
    nrows = max(3, math.ceil(len(algorithms) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(11.5, 2.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for index, (ax, algorithm) in enumerate(zip(axes.ravel(), algorithms)):
        top = ax.twiny()
        top.patch.set_visible(False)
        top.set_xticks(range(max_k + 1))
        top.set_xlim(-0.05 * max(max_k, 1), max_k + 0.05 * max(max_k, 1))
        top.tick_params(axis="x", labelsize=10 * content_scale, labeltop=index < 3)
        color = algorithm_color(algorithm)
        for x_col, curve_ax, linestyle, alpha in (
            ("severity", ax, "-", 0.10),
            ("num_corrupted", top, "--", 0.18),
        ):
            alg = grouped[x_col][grouped[x_col]["algorithm"] == algorithm].sort_values(x_col)
            x = alg[x_col].to_numpy(dtype=float)
            mean = alg["mean"].to_numpy(dtype=float)
            std = alg["std"].to_numpy(dtype=float)
            # Draw both bands on the primary axes so both mean curves remain
            # above both fills; preserve the twin's actual data coordinates.
            ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            transform=curve_ax.transData, color=color, alpha=alpha,
                            linewidth=0, zorder=1)
            curve_ax.plot(x, mean, color=color, linestyle=linestyle, marker="o",
                          linewidth=1.6 * content_scale, markersize=4 * content_scale, zorder=2)
        ax.set_title(algorithm, fontsize=12 * content_scale, pad=20)
        ax.set_ylim(0, 1)
        ax.set_xticks(severity_ticks)
        ax.tick_params(labelsize=9 * content_scale, labelbottom=index + 3 >= len(algorithms))
        ax.grid(True, alpha=0.15)
    # Explicit limits prevent the twin-coordinate band from changing the
    # shared severity domain during autoscaling.
    severity_span = max(severity_ticks[-1] - severity_ticks[0], 0.1)
    axes[0, 0].set_xlim(severity_ticks[0] - 0.05 * severity_span,
                        severity_ticks[-1] + 0.05 * severity_span)
    for ax in axes.ravel()[len(algorithms):]:
        ax.set_visible(False)
    xlabel = fig.supxlabel("Corruption severity", y=0.09)
    ylabel = fig.supylabel("F1 score")
    for label in (xlabel, ylabel):
        label.set_fontsize(label.get_fontsize() * content_scale)
    fig.text(0.5, 0.985, "Number of corrupted UAVs", ha="center", va="top", fontsize=14 * content_scale)
    handles = [
        Line2D([], [], color="0.25", linestyle="-", marker="o",
               linewidth=1.6 * content_scale, markersize=6 * content_scale,
               label="vs corruption severity"),
        Line2D([], [], color="0.25", linestyle="--", marker="o",
               linewidth=1.6 * content_scale, markersize=6 * content_scale,
               label="vs number of corrupted UAVs"),
        Patch(facecolor="0.25", alpha=0.18, label="±1 cross-condition standard deviation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=14 * content_scale)
    fig.tight_layout(rect=(0.03, 0.125, 1, 0.965), pad=0.5, h_pad=0.6, w_pad=0.6)
    save_figure(fig, output_dir, "f1_small_multiples_combined")
    plt.close(fig)


def pivot_algorithm_metric(df: pd.DataFrame, algorithm: str, metric: str) -> pd.DataFrame:
    sub = df[df["algorithm"] == algorithm].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="severity",
        columns="num_corrupted",
        values=metric,
        aggfunc="mean",
    ).sort_index()


def compute_best_consensus_by_condition(
    df: pd.DataFrame,
    metric: str,
    baseline_algorithm: str = "NO_CONSENSUS",
    exclude_algorithms: list[str] | None = None,
) -> pd.DataFrame:
    exclude = set(exclude_algorithms or []) | {baseline_algorithm}
    sub = df[~df["algorithm"].isin(exclude)].copy()

    grouped = (
        sub.groupby(["severity", "num_corrupted", "algorithm"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
    )

    if grouped.empty:
        return pd.DataFrame(columns=["severity", "num_corrupted", "algorithm", metric])

    ascending = not is_higher_better(metric)
    idx = grouped.sort_values(metric, ascending=ascending).groupby(["severity", "num_corrupted"]).head(1).index
    return grouped.loc[idx].reset_index(drop=True)

def plot_mean_consensus_gain_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    baseline_algorithm: str = "NO_CONSENSUS",
    exclude_algorithms: list[str] | None = None,
) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create mean consensus gain heatmap: metric {metric!r} not found.")
        return

    if baseline_algorithm not in set(df["algorithm"].unique()):
        print(
            f"[WARN] Cannot create mean consensus gain heatmap: "
            f"baseline algorithm {baseline_algorithm!r} not found."
        )
        return

    exclude = set(exclude_algorithms or []) | {baseline_algorithm}

    consensus = (
        df[~df["algorithm"].isin(exclude)]
        .groupby(["severity", "num_corrupted"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"mean_consensus_{metric}"})
    )

    baseline = (
        df[df["algorithm"] == baseline_algorithm]
        .groupby(["severity", "num_corrupted"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"{baseline_algorithm}_{metric}"})
    )

    merged = consensus.merge(baseline, on=["severity", "num_corrupted"], how="inner")
    if merged.empty:
        print(f"[WARN] No overlapping conditions for mean consensus gain heatmap {metric}.")
        return

    gain_col = f"mean_consensus_gain_{metric}"

    if is_higher_better(metric):
        merged[gain_col] = (
            merged[f"mean_consensus_{metric}"]
            - merged[f"{baseline_algorithm}_{metric}"]
        )
    else:
        # For FP/FN/count-style error metrics, positive means error reduction.
        merged[gain_col] = (
            merged[f"{baseline_algorithm}_{metric}"]
            - merged[f"mean_consensus_{metric}"]
        )

    pivot = (
        merged
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values=gain_col,
            aggfunc="mean",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    vmax = np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else 1.0
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
    )
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel("Corruption severity")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([
        str(int(c)) if float(c).is_integer() else str(c)
        for c in pivot.columns
    ])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    direction = "improvement" if is_higher_better(metric) else "error reduction"
    cbar.set_label(f"Mean consensus {direction} over {baseline_algorithm}")

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            if pd.notna(val):
                ax.text(
                    x,
                    y,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

    fig.tight_layout()
    save_figure(fig, output_dir, f"mean_consensus_gain_heatmap_{metric}")
    plt.close(fig)

    csv_path = output_dir / f"mean_consensus_gain_heatmap_{metric}.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")

def plot_consensus_gain_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    baseline_algorithm: str = "NO_CONSENSUS",
) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create gain heatmap: metric {metric!r} not found.")
        return
    if baseline_algorithm not in set(df["algorithm"].unique()):
        print(f"[WARN] Cannot create gain heatmap: baseline algorithm {baseline_algorithm!r} not found.")
        return

    best = compute_best_consensus_by_condition(df, metric, baseline_algorithm=baseline_algorithm)
    baseline = (
        df[df["algorithm"] == baseline_algorithm]
        .groupby(["severity", "num_corrupted"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"{baseline_algorithm}_{metric}"})
    )

    merged = best.merge(baseline, on=["severity", "num_corrupted"], how="inner")
    if merged.empty:
        print(f"[WARN] No overlapping conditions for gain heatmap {metric}.")
        return

    gain_col = f"gain_{metric}"
    if is_higher_better(metric):
        merged[gain_col] = merged[metric] - merged[f"{baseline_algorithm}_{metric}"]
    else:
        # For error/count metrics such as FP/FN, positive gain means reduction in error.
        merged[gain_col] = merged[f"{baseline_algorithm}_{metric}"] - merged[metric]

    pivot = merged.pivot_table(index="severity", columns="num_corrupted", values=gain_col, aggfunc="mean").sort_index()
    alg_pivot = merged.pivot_table(index="severity", columns="num_corrupted", values="algorithm", aggfunc=lambda xs: sorted(xs)[0]).reindex_like(pivot)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    vmax = np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else 1.0
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel("Corruption severity")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    direction = "improvement" if is_higher_better(metric) else "error reduction"
    cbar.set_label(f"Consensus {direction} over {baseline_algorithm}")

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            alg = alg_pivot.values[y, x]
            if pd.notna(val):
                ax.text(x, y, f"{val:+.2f}\n{alg}", ha="center", va="center", fontsize=10)

    fig.tight_layout()
    save_figure(fig, output_dir, f"consensus_gain_heatmap_{metric}")
    plt.close(fig)

    merged.to_csv(output_dir / f"consensus_gain_heatmap_{metric}.csv", index=False)
    print(f"[OK] Wrote {output_dir / f'consensus_gain_heatmap_{metric}.csv'}")

def plot_best_algorithm_map(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    ranking_delta: float = 0.0,
    exclude_algorithms: list[str] | None = None,
    baseline_algorithm: str = "NO_CONSENSUS",
) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create best-algorithm map: metric {metric!r} not found.")
        return

    exclude = set(exclude_algorithms or [])
    sub = df[~df["algorithm"].isin(exclude)].copy()
    grouped = (
        sub.groupby(["severity", "num_corrupted", "algorithm"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
    )
    if grouped.empty:
        return

    all_algorithms = [
        alg for alg in DEFAULT_SELECTED_ALGORITHMS
        if alg in set(grouped["algorithm"].dropna().astype(str))
    ]

    rows: list[dict[str, Any]] = []
    for (sev, k), g in grouped.groupby(["severity", "num_corrupted"]):
        best_value = g[metric].max() if is_higher_better(metric) else g[metric].min()
        eps = 1e-12

        if is_higher_better(metric):
            best_g = g[g[metric] >= best_value - ranking_delta - eps]
        else:
            best_g = g[g[metric] <= best_value + ranking_delta + eps]

        best_algorithms = (
            best_g
            .sort_values(
                [metric, "algorithm"],
                ascending=[not is_higher_better(metric), True],
            )
            ["algorithm"]
            .astype(str)
            .tolist()
        )

        best_algorithm_text = ", ".join(best_algorithms)

        display_main, display_sub = best_algorithm_display_label(
            best_algorithm_text,
            all_algorithms=all_algorithms,
            baseline_algorithm=baseline_algorithm,
            max_display_algorithms=3,
        )

        rows.append({
            "severity": sev,
            "num_corrupted": k,
            metric: best_value,
            "best_algorithm": best_algorithm_text,
            "best_algorithm_display": display_main,
            "best_algorithm_display_sub": display_sub,
            "n_best_algorithms": len(best_algorithms),
        })

    rank_df = pd.DataFrame(rows)

    value_pivot = (
        rank_df
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
    )

    alg_pivot = (
        rank_df
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values="best_algorithm_display",
            aggfunc=lambda xs: sorted(str(x) for x in xs if pd.notna(x))[0]
            if any(pd.notna(x) for x in xs)
            else "",
        )
        .reindex_like(value_pivot)
    )

    sub_pivot = (
        rank_df
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values="best_algorithm_display_sub",
            aggfunc=lambda xs: sorted(str(x) for x in xs if pd.notna(x))[0]
            if any(pd.notna(x) for x in xs)
            else "",
        )
        .reindex_like(value_pivot)
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    im = ax.imshow(value_pivot.values, aspect="auto", origin="lower")
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel("Corruption severity")
    ax.set_xticks(range(len(value_pivot.columns)))
    ax.set_xticklabels([
        str(int(c)) if float(c).is_integer() else str(c)
        for c in value_pivot.columns
    ])
    ax.set_yticks(range(len(value_pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in value_pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label(metric))

    for y in range(value_pivot.shape[0]):
        for x in range(value_pivot.shape[1]):
            val = value_pivot.values[y, x]
            main_label = alg_pivot.values[y, x]
            sub_label = sub_pivot.values[y, x]

            if pd.notna(val):
                has_sub_label = pd.notna(sub_label) and str(sub_label).strip()

                if has_sub_label:
                    ax.text(
                        x,
                        y + 0.20,
                        str(main_label),
                        ha="center",
                        va="center",
                        fontsize=9,
                        # fontweight="semibold",
                    )
                    ax.text(
                        x,
                        y - 0.04,
                        str(sub_label),
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    ax.text(
                        x,
                        y - 0.28,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
                else:
                    ax.text(
                        x,
                        y,
                        f"{main_label}\n{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )

    fig.tight_layout()
    save_figure(fig, output_dir, f"best_algorithm_map_{metric}")
    plt.close(fig)

    csv_path = output_dir / f"best_algorithm_by_condition_{metric}.csv"
    rank_df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")

def compute_robustness_auc(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm, alg_df in df.groupby("algorithm"):
        # Average across repeated fields/corruption sets first.
        curve = (
            alg_df.groupby(["severity", "num_corrupted"], as_index=False)[metric]
            .mean()
            .dropna(subset=[metric])
        )
        if curve.empty:
            continue

        aucs = []
        for k, k_df in curve.groupby("num_corrupted"):
            k_df = k_df.sort_values("severity")
            if len(k_df) >= 2:
                auc = np.trapezoid(k_df[metric].values, k_df["severity"].values)
                span = k_df["severity"].max() - k_df["severity"].min()
                if span > 0:
                    auc = auc / span
            else:
                auc = k_df[metric].iloc[0]
            aucs.append(auc)

        rows.append({
            "algorithm": algorithm,
            f"robustness_auc_{metric}": float(np.mean(aucs)),
            f"mean_{metric}": float(curve[metric].mean()),
            "n_conditions": int(len(curve)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(f"robustness_auc_{metric}", ascending=not is_higher_better(metric)).reset_index(drop=True)


def plot_robustness_auc_bar(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create robustness AUC plot: metric {metric!r} not found.")
        return

    score_df = compute_robustness_auc(df, metric)
    if score_df.empty:
        return

    score_col = f"robustness_auc_{metric}"
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(score_df))
    ax.barh(y, score_df[score_col], color=[algorithm_color(a) for a in score_df["algorithm"]])
    ax.set_yticks(y)
    ax.set_yticklabels(score_df["algorithm"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean AUC-normalized {metric_label(metric)}")
    ax.grid(True, axis="x", alpha=0.3)

    for i, val in enumerate(score_df[score_col]):
        ax.text(val, i, f" {val:.3f}", va="center", fontsize=11)

    fig.tight_layout()
    save_figure(fig, output_dir, f"robustness_auc_{metric}")
    plt.close(fig)

    csv_path = output_dir / f"robustness_auc_{metric}.csv"
    score_df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def plot_precision_recall_tradeoff(df: pd.DataFrame, output_dir: Path) -> None:
    required = {"precision", "recall"}
    if not required.issubset(df.columns):
        print("[WARN] Cannot create precision-recall tradeoff: precision/recall columns missing.")
        return

    summary = df.groupby("algorithm", as_index=False)[["precision", "recall", "f1" if "f1" in df.columns else "precision"]].mean()
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    size_col = "f1" if "f1" in df.columns else "precision"
    sizes = 120 + 360 * (summary[size_col] - summary[size_col].min()) / max(summary[size_col].max() - summary[size_col].min(), 1e-9)
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.scatter(
            row["precision"],
            row["recall"],
            s=sizes.iloc[i] if hasattr(sizes, "iloc") else sizes[i],
            alpha=0.75,
            color=algorithm_color(row["algorithm"]),
            label=row["algorithm"],
        )
        ax.annotate(row["algorithm"], (row["precision"], row["recall"]), xytext=(5, 5), textcoords="offset points", fontsize=11)

    ax.set_xlabel("Mean precision")
    ax.set_ylabel("Mean recall")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(0, summary["precision"].min() - 0.05), min(1.02, summary["precision"].max() + 0.05))
    ax.set_ylim(max(0, summary["recall"].min() - 0.05), min(1.02, summary["recall"].max() + 0.05))
    fig.tight_layout()
    save_figure(fig, output_dir, "precision_recall_tradeoff")
    plt.close(fig)

    csv_path = output_dir / "precision_recall_tradeoff.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def plot_fp_fn_tradeoff(df: pd.DataFrame, output_dir: Path) -> None:
    required = {"fp", "fn"}
    if not required.issubset(df.columns):
        print("[WARN] Cannot create FP-FN tradeoff: fp/fn columns missing.")
        return

    summary = df.groupby("algorithm", as_index=False)[["fp", "fn", "f1" if "f1" in df.columns else "fp"]].mean()
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    size_col = "f1" if "f1" in df.columns else "fp"
    sizes = 120 + 360 * (summary[size_col] - summary[size_col].min()) / max(summary[size_col].max() - summary[size_col].min(), 1e-9)
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.scatter(
            row["fp"],
            row["fn"],
            s=sizes.iloc[i] if hasattr(sizes, "iloc") else sizes[i],
            alpha=0.75,
            color=algorithm_color(row["algorithm"]),
            label=row["algorithm"],
        )
        ax.annotate(row["algorithm"], (row["fp"], row["fn"]), xytext=(5, 5), textcoords="offset points", fontsize=11)

    ax.set_xlabel("Mean false positives")
    ax.set_ylabel("Mean false negatives")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "fp_fn_tradeoff")
    plt.close(fig)

    csv_path = output_dir / "fp_fn_tradeoff.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def write_algorithm_mean_metrics_table(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    available = [m for m in metrics if m in df.columns]
    if not available:
        return
    table = df.groupby("algorithm", as_index=False)[available].mean()
    sort_col = "f1" if "f1" in table.columns else available[0]
    table = table.sort_values(sort_col, ascending=not is_higher_better(sort_col)).reset_index(drop=True)
    csv_path = output_dir / "algorithm_mean_metrics_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")

    latex_path = output_dir / "algorithm_mean_metrics_table.tex"
    with latex_path.open("w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, float_format="%.3f"))
    print(f"[OK] Wrote {latex_path}")


def compute_breakdown_thresholds(df: pd.DataFrame, metric: str, output_dir: Path, thresholds: list[float]) -> None:
    if metric not in df.columns:
        return

    rows: list[dict[str, Any]] = []
    severity_curve = df.groupby(["algorithm", "severity"], as_index=False)[metric].mean()
    k_curve = df.groupby(["algorithm", "num_corrupted"], as_index=False)[metric].mean()

    for algorithm in sorted(df["algorithm"].unique()):
        alg_sev = severity_curve[severity_curve["algorithm"] == algorithm].sort_values("severity")
        alg_k = k_curve[k_curve["algorithm"] == algorithm].sort_values("num_corrupted")
        for threshold in thresholds:
            if is_higher_better(metric):
                sev_bad = alg_sev[alg_sev[metric] < threshold]
                k_bad = alg_k[alg_k[metric] < threshold]
            else:
                sev_bad = alg_sev[alg_sev[metric] > threshold]
                k_bad = alg_k[alg_k[metric] > threshold]
            rows.append({
                "algorithm": algorithm,
                "metric": metric,
                "threshold": threshold,
                "first_severity_breakdown": None if sev_bad.empty else float(sev_bad["severity"].iloc[0]),
                "first_num_corrupted_breakdown": None if k_bad.empty else int(alg_k[alg_k[metric] < threshold]["num_corrupted"].iloc[0]) if is_higher_better(metric) and not k_bad.empty else (None if k_bad.empty else int(k_bad["num_corrupted"].iloc[0])),
            })

    out = pd.DataFrame(rows)
    csv_path = output_dir / f"breakdown_thresholds_{metric}.csv"
    out.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def plot_per_corruption_summary(df: pd.DataFrame, metric: str, output_dir: Path, algorithms: list[str]) -> None:
    """Extra diagnostic: one compact plot comparing active corruption types."""
    if metric not in df.columns:
        return
    exploded = df.copy()
    exploded["active_corruptions"] = exploded["active_corruptions"].apply(lambda xs: xs if isinstance(xs, list) else [])
    exploded = exploded.explode("active_corruptions").dropna(subset=["active_corruptions"])
    exploded = exploded[exploded["algorithm"].isin(algorithms)].copy()
    if exploded.empty:
        return

    summary = exploded.groupby(["active_corruptions", "algorithm"], as_index=False)[metric].mean()
    corruptions = sorted(summary["active_corruptions"].unique())
    x = np.arange(len(corruptions))
    width = 0.8 / max(len(algorithms), 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for i, alg in enumerate(algorithms):
        vals = []
        for corr in corruptions:
            row = summary[(summary["active_corruptions"] == corr) & (summary["algorithm"] == alg)]
            vals.append(np.nan if row.empty else row[metric].iloc[0])
        ax.bar(
            x + (i - len(algorithms) / 2) * width + width / 2,
            vals,
            width,
            label=alg,
            color=algorithm_color(alg),
        )
    ax.set_xlabel("Active corruption type")
    ax.set_ylabel(metric_label(metric))
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, f"per_corruption_summary_{metric}")
    plt.close(fig)


def plot_mean_performance_degradation_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    baseline_algorithm: str = "NO_CONSENSUS",
    exclude_algorithms: list[str] | None = None,
) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create degradation heatmap: metric {metric!r} not found.")
        return

    exclude = set(exclude_algorithms or []) | {baseline_algorithm}
    consensus_df = df[~df["algorithm"].isin(exclude)].copy()

    if consensus_df.empty:
        print("[WARN] Cannot create degradation heatmap: no consensus algorithms found.")
        return

    clean = consensus_df[
        (consensus_df["num_corrupted"] == 0)
        & (consensus_df["severity"] == 0.0)
    ].copy()

    if clean.empty:
        print("[WARN] Cannot create degradation heatmap: clean baseline condition not found.")
        return

    clean_reference = float(clean[metric].mean())

    condition_mean = (
        consensus_df
        .groupby(["severity", "num_corrupted"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
        .rename(columns={metric: f"mean_consensus_{metric}"})
    )

    degradation_col = f"degradation_from_clean_{metric}"

    if is_higher_better(metric):
        condition_mean[degradation_col] = (
            clean_reference - condition_mean[f"mean_consensus_{metric}"]
        )
    else:
        condition_mean[degradation_col] = (
            condition_mean[f"mean_consensus_{metric}"] - clean_reference
        )

    condition_mean[f"clean_reference_{metric}"] = clean_reference

    # Keep the clean reference for the degradation calculation, but remove the
    # zero-corruption / zero-severity baseline row and column from the plotted
    # heatmap. This avoids wasting visual space on the trivial +0.00 clean cell.
    plot_condition_mean = condition_mean[
        (condition_mean["num_corrupted"] > 0)
        & (condition_mean["severity"] > 0.0)
    ].copy()

    if plot_condition_mean.empty:
        print(
            "[WARN] Cannot create degradation heatmap: no non-clean "
            "conditions remain after removing num_corrupted=0 and severity=0.0."
        )
        return

    pivot = (
        plot_condition_mean
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values=degradation_col,
            aggfunc="mean",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    vmax = np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else 1.0
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(vmax, 1e-9),
        cmap="Reds",
    )
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel("Corruption severity")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([
        str(int(c)) if float(c).is_integer() else str(c)
        for c in pivot.columns
    ])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Degradation from clean {metric_label(metric)}")

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            if pd.notna(val):
                ax.text(
                    x,
                    y,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=heatmap_text_color(im, val),
                )

    fig.tight_layout()
    save_figure(fig, output_dir, f"mean_degradation_from_clean_heatmap_{metric}")
    plt.close(fig)

    csv_path = output_dir / f"mean_degradation_from_clean_heatmap_{metric}.csv"
    plot_condition_mean.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")


def plot_mean_performance_delta_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    baseline_algorithm: str = "NO_CONSENSUS",
    exclude_algorithms: list[str] | None = None,
) -> None:
    if metric not in df.columns:
        print(f"[WARN] Cannot create delta heatmap: metric {metric!r} not found.")
        return

    exclude = set(exclude_algorithms or []) | {baseline_algorithm}
    consensus_df = df[~df["algorithm"].isin(exclude)].copy()

    if consensus_df.empty:
        print("[WARN] Cannot create delta heatmap: no consensus algorithms found.")
        return

    clean = consensus_df[
        (consensus_df["num_corrupted"] == 0)
        & (consensus_df["severity"] == 0.0)
    ].copy()

    if clean.empty:
        print("[WARN] Cannot create delta heatmap: clean baseline condition not found.")
        return

    clean_reference = float(clean[metric].mean())

    condition_mean = (
        consensus_df
        .groupby(["severity", "num_corrupted"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
        .rename(columns={metric: f"mean_consensus_{metric}"})
    )

    delta_col = f"delta_from_clean_{metric}"

    if is_higher_better(metric):
        condition_mean[delta_col] = (
            condition_mean[f"mean_consensus_{metric}"] - clean_reference
        )
    else:
        # For lower-is-better metrics, positive delta still means improvement.
        condition_mean[delta_col] = (
            clean_reference - condition_mean[f"mean_consensus_{metric}"]
        )

    condition_mean[f"clean_reference_{metric}"] = clean_reference

    # Keep the clean reference for the delta calculation, but remove the
    # zero-corruption / zero-severity baseline row and column from the plotted
    # heatmap. This avoids wasting visual space on the trivial +0.00 clean cell.
    plot_condition_mean = condition_mean[
        (condition_mean["num_corrupted"] > 0)
        & (condition_mean["severity"] > 0.0)
    ].copy()

    if plot_condition_mean.empty:
        print(
            "[WARN] Cannot create delta heatmap: no non-clean "
            "conditions remain after removing num_corrupted=0 and severity=0.0."
        )
        return

    pivot = (
        plot_condition_mean
        .pivot_table(
            index="severity",
            columns="num_corrupted",
            values=delta_col,
            aggfunc="mean",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    vmax = np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else 1.0
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu",
    )
    ax.set_xlabel("Number of corrupted UAVs")
    ax.set_ylabel("Corruption severity")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([
        str(int(c)) if float(c).is_integer() else str(c)
        for c in pivot.columns
    ])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Change in {metric_label(metric)} relative to clean condition")

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            if pd.notna(val):
                ax.text(
                    x,
                    y,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=heatmap_text_color(im, val),
                )

    fig.tight_layout()
    save_figure(fig, output_dir, f"mean_delta_from_clean_heatmap_{metric}")
    plt.close(fig)

    csv_path = output_dir / f"mean_delta_from_clean_heatmap_{metric}.csv"
    plot_condition_mean.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {csv_path}")

def heatmap_text_color(im, value: float, luminance_threshold: float = 0.60) -> str:
    """Choose black/white annotation text based on the rendered heatmap color."""
    r, g, b, _ = im.cmap(im.norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < luminance_threshold else "black"

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact paper-ready consensus benchmark figures.")
    parser.add_argument("pipeline_summary", type=Path, help="Path to pipeline_summary_*.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <pipeline_summary_parent>/paper_figures.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root for remapping report paths when the summary was generated elsewhere.")
    parser.add_argument("--num-uavs", type=int, default=5, help="Total number of UAVs. Default: 5.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_SELECTED_ALGORITHMS, help=f"Algorithms to show in panel/trade-off plots. Default: {' '.join(DEFAULT_SELECTED_ALGORITHMS)}.")
    parser.add_argument("--panel-metrics", nargs="+", default=DEFAULT_PANEL_METRICS, help="Metrics for 2x2 paper panels.")
    parser.add_argument("--heatmap-metric", default="f1", help="Metric for gain and best-algorithm heatmaps. Default: f1.")
    parser.add_argument("--baseline-algorithm", default="NO_CONSENSUS", help="Baseline algorithm for gain heatmap. Default: NO_CONSENSUS.")
    parser.add_argument("--ranking-delta", type=float, default=0.0, help="Tolerance for best algorithm map. Default: 0.0.")
    parser.add_argument("--breakdown-thresholds", nargs="+", type=float, default=[0.8, 0.7, 0.6], help="Thresholds for breakdown CSV. Default: 0.8 0.7 0.6.")
    parser.add_argument("--skip-extras", action="store_true", help="Only make core paper plots; skip extra comparison plots.")
    args = parser.parse_args()

    summary_path = args.pipeline_summary.resolve()
    if summary_path.suffix.lower() != ".json":
        parser.error("pipeline_summary must be a JSON file.")

    output_dir = args.output_dir.resolve() if args.output_dir else summary_path.parent / "paper_figures"
    ensure_output_dir(output_dir)

    df = load_results_from_summary(
        summary_path,
        repo_root=args.repo_root.resolve() if args.repo_root else None,
        default_num_uavs=args.num_uavs,
    )
    flat_csv = output_dir / "consensus_results_flat_for_paper.csv"
    df.to_csv(flat_csv, index=False)
    print(f"[OK] Wrote {flat_csv}")

    algorithms = available_algorithms(df, args.algorithms)
    if not algorithms:
        raise RuntimeError("No algorithms available for plotting.")
    print(f"[INFO] Selected algorithms for paper panels: {algorithms}")

    panel_metrics = [m for m in args.panel_metrics if m in df.columns]
    missing_panel_metrics = [m for m in args.panel_metrics if m not in df.columns]
    if missing_panel_metrics:
        print(f"[WARN] Missing panel metrics skipped: {missing_panel_metrics}")

    plot_paper_severity_panel(df, output_dir, algorithms=algorithms, metrics=panel_metrics)
    plot_paper_num_corrupted_panel(df, output_dir, algorithms=algorithms, metrics=panel_metrics)
    plot_f1_variability_heatmaps(df, output_dir, algorithms)
    plot_f1_small_multiples_combined(df, output_dir, algorithms)
    plot_mean_consensus_gain_heatmap(
        df,
        args.heatmap_metric,
        output_dir,
        baseline_algorithm=args.baseline_algorithm,
    )
    plot_consensus_gain_heatmap(df, args.heatmap_metric, output_dir, baseline_algorithm=args.baseline_algorithm)
    plot_best_algorithm_map(
        df,
        args.heatmap_metric,
        output_dir,
        ranking_delta=args.ranking_delta,
        baseline_algorithm=args.baseline_algorithm,
    )
    plot_robustness_auc_bar(df, args.heatmap_metric, output_dir)

    plot_mean_performance_degradation_heatmap(
        df,
        args.heatmap_metric,
        output_dir,
        baseline_algorithm=args.baseline_algorithm,
    )

    plot_mean_performance_delta_heatmap(
        df,
        args.heatmap_metric,
        output_dir,
        baseline_algorithm=args.baseline_algorithm,
    )

    write_algorithm_mean_metrics_table(df, output_dir, DEFAULT_TABLE_METRICS)
    compute_breakdown_thresholds(df, args.heatmap_metric, output_dir, thresholds=args.breakdown_thresholds)

    if not args.skip_extras:
        plot_precision_recall_tradeoff(df[df["algorithm"].isin(algorithms)].copy(), output_dir)
        plot_fp_fn_tradeoff(df[df["algorithm"].isin(algorithms)].copy(), output_dir)
        plot_per_corruption_summary(df, args.heatmap_metric, output_dir, algorithms=algorithms)

    print(f"[DONE] Paper figures written to: {output_dir}")


if __name__ == "__main__":
    main()
