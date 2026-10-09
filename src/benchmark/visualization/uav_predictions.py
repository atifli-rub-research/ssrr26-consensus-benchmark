#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


OUTCOME_TO_CODE = {
    "TN": 0,
    "TP": 1,
    "FN": 2,
    "FP": 3,
}

CODE_TO_OUTCOME = {
    0: "TN",
    1: "TP",
    2: "FN",
    3: "FP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-UAV mine detection predictions from JSON."
    )

    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to all_predictions.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prediction_visualizations"),
        help="Directory where plots and CSV summaries will be saved.",
    )

    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of prediction records.")

    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_by_drone(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for record in records:
        grouped[record["drone_id"]].append(record)

    return dict(grouped)


def group_by_tile(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for record in records:
        grouped[int(record["tile_index"])].append(record)

    return dict(grouped)


def get_grid_shape(records: list[dict[str, Any]]) -> tuple[int, int]:
    max_row = max(int(r["ground_truth"]["row"]) for r in records)
    max_col = max(int(r["ground_truth"]["col"]) for r in records)

    return max_row + 1, max_col + 1


def get_model_name(record: dict[str, Any]) -> str:
    model = record.get("model", "unknown_model")
    return Path(model).name


def get_active_corruption(record: dict[str, Any]) -> str:
    corruptions = record.get("corruptions", {})

    active = []
    for name, severity in corruptions.items():
        severity = float(severity)
        if severity > 0.0:
            active.append(f"{name}={severity:g}")

    if not active:
        return "clean"

    return ", ".join(active)


def safe_filename(text: str) -> str:
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(" ", "_")
    text = text.replace("=", "-")
    text = text.replace(",", "")
    text = text.replace("|", "_")
    return text


def predicted_mine_present(record: dict[str, Any]) -> bool:
    return int(record.get("num_detections", 0)) > 0


def classify_binary_outcome(record: dict[str, Any]) -> str:
    true_present = bool(record["true_mine_present"])
    pred_present = predicted_mine_present(record)

    if true_present and pred_present:
        return "TP"

    if true_present and not pred_present:
        return "FN"

    if not true_present and pred_present:
        return "FP"

    return "TN"


def compute_confusion(records: list[dict[str, Any]]) -> dict[str, int]:
    confusion = {
        "TP": 0,
        "FN": 0,
        "FP": 0,
        "TN": 0,
    }

    for record in records:
        outcome = classify_binary_outcome(record)
        confusion[outcome] += 1

    return confusion


def compute_rates(confusion: dict[str, int]) -> dict[str, float]:
    tp = confusion["TP"]
    fn = confusion["FN"]
    fp = confusion["FP"]
    tn = confusion["TN"]

    total = tp + fn + fp + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    headers = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    grid: np.ndarray,
    title: str,
    output_path: Path,
    colorbar_label: str,
    tick_labels: dict[int, str] | None = None,
) -> None:
    plt.figure(figsize=(10, 8))

    im = plt.imshow(grid, interpolation="nearest")

    plt.title(title)
    plt.xlabel("Grid column")
    plt.ylabel("Grid row")

    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)

    if tick_labels is not None:
        ticks = sorted(tick_labels.keys())
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([tick_labels[t] for t in ticks])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix(
    confusion: dict[str, int],
    title: str,
    output_path: Path,
) -> None:
    matrix = np.array(
        [
            [confusion["TN"], confusion["FP"]],
            [confusion["FN"], confusion["TP"]],
        ]
    )

    plt.figure(figsize=(6, 5))

    im = plt.imshow(matrix, interpolation="nearest")

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks([0, 1], ["No mine", "Mine"])
    plt.yticks([0, 1], ["No mine", "Mine"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.colorbar(im, label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_correctness_grid(
    records: list[dict[str, Any]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for record in records:
        row = int(record["ground_truth"]["row"])
        col = int(record["ground_truth"]["col"])

        outcome = classify_binary_outcome(record)
        grid[row, col] = OUTCOME_TO_CODE[outcome]

    return grid


def build_ground_truth_count_grid(
    records: list[dict[str, Any]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for record in records:
        row = int(record["ground_truth"]["row"])
        col = int(record["ground_truth"]["col"])

        grid[row, col] = int(record["true_mine_count"])

    return grid


def build_predicted_count_grid(
    records: list[dict[str, Any]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for record in records:
        row = int(record["ground_truth"]["row"])
        col = int(record["ground_truth"]["col"])

        grid[row, col] = int(record.get("num_detections", 0))

    return grid


def build_count_error_grid(
    records: list[dict[str, Any]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for record in records:
        row = int(record["ground_truth"]["row"])
        col = int(record["ground_truth"]["col"])

        true_count = int(record["true_mine_count"])
        predicted_count = int(record.get("num_detections", 0))

        grid[row, col] = predicted_count - true_count

    return grid


def plot_per_drone_visualizations(
    grouped_by_drone: dict[str, list[dict[str, Any]]],
    grid_shape: tuple[int, int],
    output_dir: Path,
) -> None:
    per_drone_dir = output_dir / "per_drone"
    ensure_dir(per_drone_dir)

    summary_rows = []

    for drone_id, records in sorted(grouped_by_drone.items()):
        records = sorted(records, key=lambda r: int(r["tile_index"]))

        first_record = records[0]
        model_name = get_model_name(first_record)
        corruption = get_active_corruption(first_record)

        label = f"{drone_id} | {model_name} | {corruption}"
        filename_prefix = safe_filename(drone_id)

        correctness_grid = build_correctness_grid(records, grid_shape)
        gt_count_grid = build_ground_truth_count_grid(records, grid_shape)
        pred_count_grid = build_predicted_count_grid(records, grid_shape)
        count_error_grid = build_count_error_grid(records, grid_shape)

        save_heatmap(
            correctness_grid,
            title=f"Binary correctness: {label}",
            output_path=per_drone_dir / f"{filename_prefix}_correctness_heatmap.png",
            colorbar_label="Outcome",
            tick_labels=CODE_TO_OUTCOME,
        )

        save_heatmap(
            gt_count_grid,
            title=f"Ground truth mine count: {label}",
            output_path=per_drone_dir / f"{filename_prefix}_ground_truth_count_heatmap.png",
            colorbar_label="True mine count",
        )

        save_heatmap(
            pred_count_grid,
            title=f"Predicted mine count: {label}",
            output_path=per_drone_dir / f"{filename_prefix}_predicted_count_heatmap.png",
            colorbar_label="Number of detections",
        )

        save_heatmap(
            count_error_grid,
            title=f"Count error: {label}",
            output_path=per_drone_dir / f"{filename_prefix}_count_error_heatmap.png",
            colorbar_label="Predicted count - true count",
        )

        confusion = compute_confusion(records)
        rates = compute_rates(confusion)

        save_confusion_matrix(
            confusion,
            title=f"Confusion matrix: {label}",
            output_path=per_drone_dir / f"{filename_prefix}_confusion_matrix.png",
        )

        summary_rows.append(
            {
                "drone_id": drone_id,
                "model": model_name,
                "corruption": corruption,
                **confusion,
                **rates,
            }
        )

    save_csv(summary_rows, output_dir / "per_drone_metrics.csv")


def plot_confidence_histograms(
    grouped_by_drone: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    confidence_dir = output_dir / "confidence"
    ensure_dir(confidence_dir)

    all_confidences_on_mine_tiles = []
    all_confidences_on_empty_tiles = []

    for drone_id, records in sorted(grouped_by_drone.items()):
        confidences_on_mine_tiles = []
        confidences_on_empty_tiles = []

        for record in records:
            true_present = bool(record["true_mine_present"])

            for detection in record.get("detections", []):
                confidence = float(detection["confidence"])

                if true_present:
                    confidences_on_mine_tiles.append(confidence)
                    all_confidences_on_mine_tiles.append(confidence)
                else:
                    confidences_on_empty_tiles.append(confidence)
                    all_confidences_on_empty_tiles.append(confidence)

        plt.figure(figsize=(8, 5))

        if confidences_on_mine_tiles:
            plt.hist(
                confidences_on_mine_tiles,
                bins=20,
                alpha=0.7,
                label="Detections on mine-present tiles",
            )

        if confidences_on_empty_tiles:
            plt.hist(
                confidences_on_empty_tiles,
                bins=20,
                alpha=0.7,
                label="Detections on mine-absent tiles",
            )

        plt.title(f"Detection confidence distribution: {drone_id}")
        plt.xlabel("Confidence")
        plt.ylabel("Number of detections")
        plt.xlim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(confidence_dir / f"{safe_filename(drone_id)}_confidence_histogram.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))

    if all_confidences_on_mine_tiles:
        plt.hist(
            all_confidences_on_mine_tiles,
            bins=20,
            alpha=0.7,
            label="Detections on mine-present tiles",
        )

    if all_confidences_on_empty_tiles:
        plt.hist(
            all_confidences_on_empty_tiles,
            bins=20,
            alpha=0.7,
            label="Detections on mine-absent tiles",
        )

    plt.title("Detection confidence distribution: all UAVs")
    plt.xlabel("Confidence")
    plt.ylabel("Number of detections")
    plt.xlim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(confidence_dir / "all_uavs_confidence_histogram.png", dpi=200)
    plt.close()


def build_votes_for_mine_grid(
    grouped_by_tile: dict[int, list[dict[str, Any]]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for records in grouped_by_tile.values():
        reference = records[0]

        row = int(reference["ground_truth"]["row"])
        col = int(reference["ground_truth"]["col"])

        votes_for_mine = sum(predicted_mine_present(r) for r in records)

        grid[row, col] = votes_for_mine

    return grid


def build_vote_fraction_grid(
    grouped_by_tile: dict[int, list[dict[str, Any]]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, np.nan)

    for records in grouped_by_tile.values():
        reference = records[0]

        row = int(reference["ground_truth"]["row"])
        col = int(reference["ground_truth"]["col"])

        votes_for_mine = sum(predicted_mine_present(r) for r in records)
        num_agents = len(records)

        grid[row, col] = votes_for_mine / num_agents if num_agents else np.nan

    return grid


def build_consensus_records(
    grouped_by_tile: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    consensus_records = []

    for tile_index, records in grouped_by_tile.items():
        reference = records[0]

        votes_for_mine = sum(predicted_mine_present(r) for r in records)
        num_agents = len(records)

        majority_threshold = (num_agents // 2) + 1
        consensus_present = votes_for_mine >= majority_threshold

        consensus_records.append(
            {
                "tile_index": tile_index,
                "ground_truth": reference["ground_truth"],
                "true_mine_present": bool(reference["true_mine_present"]),
                "true_mine_count": int(reference["true_mine_count"]),
                "num_detections": int(consensus_present),
                "votes_for_mine": votes_for_mine,
                "num_agents": num_agents,
            }
        )

    return sorted(consensus_records, key=lambda r: int(r["tile_index"]))


def plot_consensus_visualizations(
    records: list[dict[str, Any]],
    grid_shape: tuple[int, int],
    output_dir: Path,
) -> None:
    consensus_dir = output_dir / "consensus"
    ensure_dir(consensus_dir)

    grouped_by_tile = group_by_tile(records)

    votes_grid = build_votes_for_mine_grid(grouped_by_tile, grid_shape)
    vote_fraction_grid = build_vote_fraction_grid(grouped_by_tile, grid_shape)

    save_heatmap(
        votes_grid,
        title="UAV votes for mine per tile",
        output_path=consensus_dir / "votes_for_mine_heatmap.png",
        colorbar_label="Number of UAVs predicting mine",
    )

    save_heatmap(
        vote_fraction_grid,
        title="UAV vote fraction for mine per tile",
        output_path=consensus_dir / "vote_fraction_for_mine_heatmap.png",
        colorbar_label="Fraction of UAVs predicting mine",
    )

    consensus_records = build_consensus_records(grouped_by_tile)
    consensus_correctness_grid = build_correctness_grid(consensus_records, grid_shape)

    save_heatmap(
        consensus_correctness_grid,
        title="Majority-vote consensus correctness",
        output_path=consensus_dir / "majority_vote_correctness_heatmap.png",
        colorbar_label="Outcome",
        tick_labels=CODE_TO_OUTCOME,
    )

    confusion = compute_confusion(consensus_records)
    rates = compute_rates(confusion)

    save_confusion_matrix(
        confusion,
        title="Majority-vote consensus confusion matrix",
        output_path=consensus_dir / "majority_vote_confusion_matrix.png",
    )

    save_csv(
        [
            {
                "method": "majority_vote",
                **confusion,
                **rates,
            }
        ],
        output_dir / "consensus_metrics.csv",
    )


def main() -> None:
    args = parse_args()

    json_path = args.json_path
    output_dir = args.output_dir

    ensure_dir(output_dir)

    records = load_json(json_path)

    if not records:
        raise ValueError("No prediction records found in JSON file.")

    grid_shape = get_grid_shape(records)
    grouped_by_drone = group_by_drone(records)

    print(f"Loaded {len(records)} prediction records.")
    print(f"Found {len(grouped_by_drone)} UAVs.")
    print(f"Grid shape: {grid_shape[0]} rows x {grid_shape[1]} columns.")
    print(f"Saving visualizations to: {output_dir}")

    plot_per_drone_visualizations(
        grouped_by_drone=grouped_by_drone,
        grid_shape=grid_shape,
        output_dir=output_dir,
    )

    plot_confidence_histograms(
        grouped_by_drone=grouped_by_drone,
        output_dir=output_dir,
    )

    plot_consensus_visualizations(
        records=records,
        grid_shape=grid_shape,
        output_dir=output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
