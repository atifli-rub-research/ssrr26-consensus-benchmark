#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
from typing import Any

from ultralytics import YOLO


def safe_float(value: Any) -> float:
    """
    Convert Ultralytics metric values to plain Python floats.
    Handles tensors, numpy scalars, and normal numeric values.
    """
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    split: str,
    save_plots: bool,
    save_json: bool,
) -> dict:
    model_name = model_path.stem

    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print(f"Model path       : {model_path}")
    print(f"Dataset YAML     : {data_yaml}")
    print(f"Split            : {split}")
    print(f"Image size       : {imgsz}")
    print(f"Confidence       : {conf}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        project=str(output_dir),
        name=model_name,
        exist_ok=True,
        plots=save_plots,
        save_json=save_json,
    )

    precision = safe_float(metrics.box.mp)
    recall = safe_float(metrics.box.mr)
    f1 = compute_f1(precision, recall)

    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": safe_float(metrics.box.map50),
        "map50_95": safe_float(metrics.box.map),
    }

    return result


def write_json(results: list[dict], output_path: Path) -> None:
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


def write_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for r in results:
        print(
            f"{r['model_name']}: "
            f"P={r['precision']:.4f}, "
            f"R={r['recall']:.4f}, "
            f"F1={r['f1']:.4f}, "
            f"mAP50={r['map50']:.4f}, "
            f"mAP50-95={r['map50_95']:.4f}, "
            f"conf={r['conf']:.3f}, "
            f"split={r['split']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more YOLO PFM-1 detector checkpoints on a selected "
            "dataset split. Defaults to conf=0.25 to match the benchmark "
            "inference operating point."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to YOLO dataset YAML, e.g. preprocessed_tiled/pfm1.yaml",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        type=Path,
        help="One or more YOLO model .pt files.",
    )

    parser.add_argument(
        "--output-dir",
        default=Path("runs/test_eval_conf025"),
        type=Path,
        help="Output directory for validation results and summary files.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Validation image size.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "Confidence threshold for validation. "
            "Use 0.25 to match benchmark inference; use 0.001 for standard mAP-style validation."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving Ultralytics validation plots.",
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable saving Ultralytics COCO-style prediction JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_path in args.models:
        result = evaluate_model(
            model_path=model_path,
            data_yaml=args.data,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            split=args.split,
            save_plots=not args.no_plots,
            save_json=not args.no_json,
        )
        results.append(result)

    summary_json_path = args.output_dir / "summary.json"
    summary_csv_path = args.output_dir / "summary.csv"

    write_json(results, summary_json_path)
    write_csv(results, summary_csv_path)

    print_summary(results)

    print("\nSaved summaries:")
    print(f"  JSON: {summary_json_path}")
    print(f"  CSV : {summary_csv_path}")


if __name__ == "__main__":
    main()
