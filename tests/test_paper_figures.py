"""Regression checks for cross-condition variability in paper figures."""

import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.visualization import paper_figures as paper


def condition_rows():
    rows = [
        ("CDCI", 0, 0.0, "none", 0.2),
        ("CDCI", 0, 0.0, "none", 0.4),
        ("CDCI", 0, 0.0, "other", 1.0),
        ("CDCI", 0, 0.5, "other", 0.9),
        ("CDCI", 1, 0.0, "other", 0.9),
        ("CDCI", 1, 0.5, "blur", 0.0),
        ("CDCI", 1, 0.5, "noise", 1.0),
        ("CDCI", 2, 0.8, "blur", 0.0),
        ("CDCI", 2, 0.8, "noise", 1.0),
        ("CDCI", 3, 1.0, "blur", 0.6),
        ("NO_CONSENSUS", 1, 0.5, "blur", 0.1),
        ("NO_CONSENSUS", 1, 0.5, "noise", 0.3),
    ]
    df = pd.DataFrame(rows, columns=[
        "algorithm", "num_corrupted", "severity", "corruption_set", "f1",
    ])
    for metric in ("accuracy", "precision", "recall"):
        df[metric] = df["f1"]
    return df


def previous_curve(df, x_col, metric):
    grouped = df.groupby(["algorithm", x_col], as_index=False)[metric].mean()
    return paper.add_clean_baseline_points(
        grouped, paper.get_baseline_by_algorithm(df, metric), x_col, metric,
    )


class PaperFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_means_preserve_previous_aggregation_and_baseline_fallback(self):
        for fallback in (False, True):
            df = condition_rows()
            if fallback:
                df["corruption_set"] = "renamed"
            for x_col in ("severity", "num_corrupted"):
                with self.subTest(fallback=fallback, x_col=x_col):
                    expected = previous_curve(df, x_col, "f1")
                    actual = paper._group_panel_metric(df, x_col, "f1")
                    pd.testing.assert_frame_equal(
                        actual[["algorithm", x_col, "mean"]]
                        .rename(columns={"mean": "f1"}).reset_index(drop=True),
                        expected.reset_index(drop=True),
                    )

    def test_sample_std_uses_same_condition_rows_and_singletons_are_zero(self):
        df = condition_rows()
        for x_col in ("severity", "num_corrupted"):
            grouped = paper._group_panel_metric(df, x_col, "f1")
            for row in grouped.itertuples(index=False):
                x = getattr(row, x_col)
                values = df[(df.algorithm == row.algorithm) & (df[x_col] == x)]
                if x == 0 and row.algorithm == "CDCI":
                    values = values[(values.num_corrupted == 0)
                                    & (values.severity == 0)
                                    & (values.corruption_set == "none")]
                expected = values.f1.std() if len(values) > 1 else 0.0
                with self.subTest(x_col=x_col, algorithm=row.algorithm, x=x):
                    self.assertAlmostEqual(row.std, expected)
            singleton = grouped[(grouped.algorithm == "CDCI")
                                & (grouped[x_col] == (1.0 if x_col == "severity" else 3))]
            self.assertEqual(singleton["std"].iloc[0], 0.0)

    def test_panels_keep_only_original_mean_curves(self):
        df = condition_rows()
        algorithms = ["NO_CONSENSUS", "CDCI"]
        for x_col in ("severity", "num_corrupted"):
            with patch.object(paper, "save_figure") as save:
                paper.plot_metric_panel(df, x_col, paper.DEFAULT_PANEL_METRICS,
                                        Path("unused"), "panel", algorithms, "Panel")
            fig = save.call_args.args[0]
            self.assertEqual([t.get_text() for t in fig.legends[0].texts], algorithms)
            for ax, metric in zip(fig.axes, paper.DEFAULT_PANEL_METRICS):
                expected = previous_curve(df, x_col, metric)
                self.assertEqual(len(ax.collections), 0)
                self.assertEqual(len(ax.containers), 0)
                self.assertEqual(len(ax.lines), len(algorithms))
                for algorithm, line in zip(algorithms, ax.lines):
                    old = expected[expected.algorithm == algorithm].sort_values(x_col)
                    np.testing.assert_allclose(line.get_xdata(), old[x_col])
                    np.testing.assert_allclose(line.get_ydata(), old[metric])
                    self.assertEqual(line.get_color(), paper.algorithm_color(algorithm))
                    self.assertEqual(line.get_marker(), "o")
                    self.assertEqual(line.get_linewidth(), 2.2)
                    self.assertEqual(line.get_markersize(), 5)
                    self.assertEqual(line.get_linestyle(), paper.algorithm_linestyle(algorithm))

    def test_heatmaps_share_scale_order_and_aggregation(self):
        df = condition_rows()
        algorithms = ["NO_CONSENSUS", "CDCI"]
        with patch.object(paper, "save_figure") as save:
            paper.plot_f1_variability_heatmaps(df, Path("unused"), algorithms)
        fig = save.call_args.args[0]
        self.assertEqual(save.call_args.args[2], "f1_variability_heatmaps")
        images = [ax.images[0] for ax in fig.axes[:2]]
        self.assertIs(images[0].norm, images[1].norm)
        expected_max = 0
        for ax, im, x_col in zip(fig.axes, images, ("severity", "num_corrupted")):
            expected = paper._group_panel_metric(df, x_col, "f1").pivot(
                index="algorithm", columns=x_col, values="std").reindex(algorithms)
            actual = im.get_array()
            np.testing.assert_allclose(actual.filled(np.nan), expected.to_numpy(), equal_nan=True)
            np.testing.assert_array_equal(np.ma.getmaskarray(actual), expected.isna())
            if ax is fig.axes[0]:
                self.assertEqual([t.get_text() for t in ax.get_yticklabels()], algorithms)
            self.assertEqual(len(ax.texts), np.isfinite(expected.to_numpy()).sum())
            expected_max = max(expected_max, np.nanmax(expected.to_numpy()))
        self.assertEqual(images[0].norm.vmin, 0)
        self.assertEqual(images[0].norm.vmax, expected_max)
        self.assertEqual(fig.axes[2].get_ylabel(), "Cross-condition F1 standard deviation")

    def test_zero_variability_has_nondegenerate_scale(self):
        df = condition_rows()
        df["f1"] = 0.5
        with patch.object(paper, "save_figure") as save:
            paper.plot_f1_variability_heatmaps(df, Path("unused"), ["CDCI"])
        norm = save.call_args.args[0].axes[0].images[0].norm
        self.assertEqual((norm.vmin, norm.vmax), (0, 1))

    def test_combined_small_multiples_preserve_both_domains_and_clipped_bands(self):
        df = condition_rows()
        algorithms = ["NO_CONSENSUS", "CDCI"]
        original = plt.Axes.fill_between
        with patch.object(paper, "save_figure") as save, patch.object(
            plt.Axes, "fill_between", autospec=True, side_effect=original,
        ) as fill:
            paper.plot_f1_small_multiples_combined(df, Path("unused"), algorithms)
        fig = save.call_args.args[0]
        self.assertEqual(len(fig.axes), 11)
        self.assertEqual(save.call_args.args[2], "f1_small_multiples_combined")
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([t.get_text() for t in fig.legends[0].texts], [
            "vs corruption severity", "vs number of corrupted UAVs",
            "±1 cross-condition standard deviation"])
        self.assertEqual(fill.call_count, 4)
        for index, algorithm in enumerate(algorithms):
            ax, top = fig.axes[index], fig.axes[9 + index]
            self.assertEqual(ax.get_title(), algorithm)
            self.assertEqual(ax.get_ylim(), (0, 1))
            self.assertTrue(ax.get_shared_y_axes().joined(ax, top))
            self.assertFalse(ax.get_shared_x_axes().joined(ax, top))
            self.assertTrue(fig.axes[0].get_shared_y_axes().joined(fig.axes[0], ax))
            self.assertTrue(fig.axes[0].get_shared_x_axes().joined(fig.axes[0], ax))
            self.assertNotEqual(ax.get_xlim(), top.get_xlim())
            np.testing.assert_allclose(ax.get_xticks(), [0, 0.5, 0.8, 1])
            np.testing.assert_allclose(top.get_xticks(), [0, 1, 2, 3])
            self.assertFalse(top.patch.get_visible())
            for offset, (x_col, curve_ax, style, alpha) in enumerate((
                ("severity", ax, "-", 0.10), ("num_corrupted", top, "--", 0.18),
            )):
                stats = paper._group_panel_metric(df, x_col, "f1")
                alg = stats[stats.algorithm == algorithm].sort_values(x_col)
                line = curve_ax.lines[0]
                call = fill.call_args_list[index * 2 + offset]
                np.testing.assert_allclose(line.get_xdata(), alg[x_col])
                np.testing.assert_allclose(line.get_ydata(), alg["mean"])
                np.testing.assert_allclose(call.args[2], np.clip(alg["mean"]-alg["std"], 0, 1))
                np.testing.assert_allclose(call.args[3], np.clip(alg["mean"]+alg["std"], 0, 1))
                self.assertIs(call.kwargs["transform"], curve_ax.transData)
                self.assertEqual(line.get_color(), paper.algorithm_color(algorithm))
                self.assertEqual(line.get_linestyle(), style)
                self.assertEqual(line.get_marker(), "o")
                self.assertAlmostEqual(line.get_linewidth(), 1.6 * 1.10)
                self.assertEqual(ax.collections[offset].get_alpha(), alpha)
                self.assertLess(ax.collections[offset].get_zorder(), line.get_zorder())
        self.assertTrue(all(not ax.get_visible() for ax in fig.axes[2:9]))
        self.assertEqual(fig._supxlabel.get_text(), "Corruption severity")
        self.assertEqual(fig._supylabel.get_text(), "F1 score")
        self.assertIn("Number of corrupted UAVs", [t.get_text() for t in fig.texts])

    def test_missing_or_nonfinite_f1_skips_new_figures(self):
        for df in (condition_rows().drop(columns="f1"), condition_rows().assign(f1=np.nan)):
            with patch.object(paper, "save_figure") as save, patch("builtins.print") as warning:
                paper.plot_f1_variability_heatmaps(df, Path("unused"), ["CDCI"])
                paper.plot_f1_small_multiples_combined(df, Path("unused"), ["CDCI"])
            save.assert_not_called()
            self.assertEqual(warning.call_count, 2)
            self.assertIn("[WARN] F1 unavailable", warning.call_args.args[0])

    def test_main_generates_variability_as_core_figures_for_selected_algorithms(self):
        algorithms = ["NO_CONSENSUS", "CDCI"]
        unrelated = (
            "plot_paper_severity_panel", "plot_paper_num_corrupted_panel",
            "plot_mean_consensus_gain_heatmap", "plot_consensus_gain_heatmap",
            "plot_best_algorithm_map", "plot_robustness_auc_bar",
            "plot_mean_performance_degradation_heatmap",
            "plot_mean_performance_delta_heatmap", "write_algorithm_mean_metrics_table",
            "compute_breakdown_thresholds",
        )
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["visualize_paper_figures", "summary.json",
                "--output-dir", directory, "--skip-extras", "--algorithms", *algorithms]))
            stack.enter_context(patch.object(paper, "load_results_from_summary", return_value=condition_rows()))
            stack.enter_context(patch("builtins.print"))
            for name in unrelated:
                stack.enter_context(patch.object(paper, name))
            heatmap = stack.enter_context(patch.object(paper, "plot_f1_variability_heatmaps"))
            multiples = stack.enter_context(patch.object(paper, "plot_f1_small_multiples_combined"))
            paper.main()
            self.assertEqual(heatmap.call_count, 1)
            self.assertEqual(heatmap.call_args.args[2], algorithms)
            self.assertEqual(multiples.call_count, 1)
            self.assertEqual(multiples.call_args.args[2], algorithms)


if __name__ == "__main__":
    unittest.main()
