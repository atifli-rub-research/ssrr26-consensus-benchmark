"""Evaluate completed simulations using tile-local instance IoU matching.

Only evaluate_multi_run is public. Simulation execution and algorithm
comparisons belong to simulation.orchestration; helper functions here implement
thresholding, box selection, matching, and averaging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark.consensus.core import constants as C

if TYPE_CHECKING:
    from benchmark.consensus.simulation.swarm_simulation import (
        MultiRunSimulationResult,
        SimulationResult,
    )

__all__ = ["evaluate_multi_run"]

Box = list[float]


def _consensus_to_label(
    consensus_value: float | None,
    threshold: float | None = None,
) -> bool:
    if consensus_value is None:
        return False
    th = C.DECISION_THRESHOLD if threshold is None else threshold
    return float(consensus_value) >= th


def _slot_accepted(slot: dict[str, Any], threshold: float | None = None) -> bool:
    return _consensus_to_label(slot.get("consensus_value"), threshold=threshold)


def _bbox_iou(a: Box, b: Box) -> float:
    """Intersection over union for axis-aligned boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _match_boxes(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one matching by highest IoU first.

    Returns:
        tp, fp, fn instance counts for this tile.
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = _bbox_iou(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp += 1

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def _gt_boxes_from_row(row: dict[str, Any]) -> list[Box]:
    gt = row.get("ground_truth") or {}
    return [list(b["bbox_xyxy"]) for b in gt.get("boxes", [])]


def _best_box_for_slot(
    tile_index: int,
    slot_index: int,
    swarm_data: dict[str, dict[int, dict[str, Any]]],
    drone_ids: list[str] | None = None,
) -> Box | None:
    """Highest-confidence detection at *slot_index* among UAVs (if any)."""
    from benchmark.consensus.core.agent import UAVAgent

    best: tuple[float, Box] | None = None
    ids = drone_ids or sorted(swarm_data.keys())
    for did in ids:
        record = swarm_data[did].get(tile_index)
        if not record:
            continue
        box = UAVAgent.slot_detection_box(record, slot_index)
        if box is None:
            continue
        dets = sorted(
            record.get("detections") or [],
            key=lambda d: float(d["confidence"]),
            reverse=True,
        )
        conf = float(dets[slot_index]["confidence"])
        if best is None or conf > best[0]:
            best = (conf, box)
    return best[1] if best else None

def _collect_predicted_boxes(
    row: dict[str, Any],
    swarm_data: dict[str, dict[int, dict[str, Any]]],
    threshold: float | None,
    drone_id: str | None = None,
) -> list[Box]:
    """
    One BBox per accepted mine slot (slot 0 = first mine, slot 1 = second, …).
    """
    box_drones = [drone_id] if drone_id else None
    tile_index = int(row["tile_index"])
    pred_boxes: list[Box] = []
    for slot in row.get("slots") or []:
        if not _slot_accepted(slot, threshold):
            continue
        si = int(slot["slot_index"])
        box = _best_box_for_slot(tile_index, si, swarm_data, drone_ids=box_drones)
        if box is not None:
            pred_boxes.append(box)
    return pred_boxes

def _counts_from_rates(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def _evaluate_instance_counts(
    final_tiles: list[dict[str, Any]],
    swarm_data: dict[str, dict[int, dict[str, Any]]],
    threshold: float | None,
    iou_th: float,
    drone_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate TP/FP/FN over final_tiles, optionally using one UAV's boxes."""
    tp = fp = fn = 0
    n_gt = 0
    n_pred = 0
    per_tile: list[dict[str, Any]] = []

    for row in final_tiles:
        gt_boxes = _gt_boxes_from_row(row)
        pred_boxes = _collect_predicted_boxes(
            row, swarm_data, threshold, drone_id=drone_id
        )
        tile_tp, tile_fp, tile_fn = _match_boxes(gt_boxes, pred_boxes, iou_th)

        tp += tile_tp
        fp += tile_fp
        fn += tile_fn
        n_gt += len(gt_boxes)
        n_pred += len(pred_boxes)

        per_tile.append(
            {
                "tile_index": row["tile_index"],
                "true_mine_present": bool(row.get("true_mine_present")),
                "n_gt_instances": len(gt_boxes),
                "n_pred_instances": len(pred_boxes),
                "tp": tile_tp,
                "fp": tile_fp,
                "fn": tile_fn,
                "slots": row.get("slots"),
                "accepted_slots": [
                    s["slot_index"]
                    for s in (row.get("slots") or [])
                    if _slot_accepted(s, threshold)
                ],
            }
        )

    rates = _counts_from_rates(float(tp), float(fp), float(fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt_instances": n_gt,
        "n_pred_instances": n_pred,
        "n_tiles": len(final_tiles),
        "per_tile": per_tile,
        **rates,
    }

def _evaluate_no_consensus_per_drone_mean(
    result: SimulationResult,
    threshold: float | None,
    iou_th: float,
) -> dict[str, Any]:
    """Mean TP/FP/FN over UAVs, then precision/recall/F1 from those means."""
    per_drone = result.per_drone_final_tiles or {}
    swarm_data = result.swarm_data
    assert swarm_data is not None

    per_drone_counts: dict[str, dict[str, int]] = {}
    sum_tp = sum_fp = sum_fn = 0.0
    sum_n_gt = sum_n_pred = 0.0
    n_drones = len(result.drone_ids)

    for did in result.drone_ids:
        tiles = per_drone.get(did, [])
        counts = _evaluate_instance_counts(
            tiles, swarm_data, threshold, iou_th, drone_id=did
        )
        per_drone_counts[did] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "n_pred_instances": counts["n_pred_instances"],
        }
        sum_tp += counts["tp"]
        sum_fp += counts["fp"]
        sum_fn += counts["fn"]
        sum_n_gt += counts["n_gt_instances"]
        sum_n_pred += counts["n_pred_instances"]

    avg_tp = sum_tp / n_drones
    avg_fp = sum_fp / n_drones
    avg_fn = sum_fn / n_drones
    rates = _counts_from_rates(avg_tp, avg_fp, avg_fn)

    return {
        "algorithm": result.algorithm,
        "drones": result.drone_ids,
        "metric_level": "instance",
        "metric_aggregation": "per_drone_mean_counts",
        "iou_threshold": iou_th,
        "n_tiles": len(result.final_tiles),
        "n_gt_instances": int(round(sum_n_gt / n_drones)),
        "n_pred_instances": int(round(sum_n_pred / n_drones)),
        "tp": int(round(avg_tp)),
        "fp": int(round(avg_fp)),
        "fn": int(round(avg_fn)),
        "accuracy": round(rates["accuracy"], 4),
        "precision": round(rates["precision"], 4),
        "recall": round(rates["recall"], 4),
        "f1": round(rates["f1"], 4),
        "per_drone_counts": per_drone_counts,
    }

def _evaluate_result(
    result: SimulationResult,
    threshold: float | None = None,
    iou_threshold: float | None = None,
) -> dict[str, Any]:
    """Instance-level metrics via BBox IoU matching (tile-local coordinates)."""
    if not result.swarm_data:
        raise ValueError("SimulationResult.swarm_data required for instance metrics")

    iou_th = C.INSTANCE_IOU_THRESHOLD if iou_threshold is None else iou_threshold

    # Special case: no consensus baseline.
    # Evaluate each UAV's own local detections, then average TP/FP/FN over UAVs.
    if result.algorithm == "NO_CONSENSUS" and result.per_drone_final_tiles:
        return _evaluate_no_consensus_per_drone_mean(
            result,
            threshold,
            iou_th,
        )

    # Normal consensus algorithms:
    # evaluate the final swarm-level result.
    counts = _evaluate_instance_counts(
        result.final_tiles,
        result.swarm_data,
        threshold,
        iou_th,
    )

    return {
        "algorithm": result.algorithm,
        "drones": result.drone_ids,
        "metric_level": "instance",
        "iou_threshold": iou_th,
        "n_tiles": counts["n_tiles"],
        "n_gt_instances": counts["n_gt_instances"],
        "n_pred_instances": counts["n_pred_instances"],
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "accuracy": round(counts["accuracy"], 4),
        "precision": round(counts["precision"], 4),
        "recall": round(counts["recall"], 4),
        "f1": round(counts["f1"], 4),
        "per_tile": counts["per_tile"],
    }

_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "n_gt_instances",
    "n_pred_instances",
)


def _average_metrics(
    summaries: list[dict[str, Any]],
    algorithm: str | None = None,
    drones: list[str] | None = None,
) -> dict[str, Any]:
    """Mean of metrics over multiple simulation runs."""
    if not summaries:
        raise ValueError("No summaries to average")

    avg: dict[str, Any] = {
        "algorithm": algorithm or summaries[0].get("algorithm"),
        "drones": drones or summaries[0].get("drones"),
        "metric_level": summaries[0].get("metric_level", "instance"),
        "iou_threshold": summaries[0].get("iou_threshold"),
        "n_tiles": summaries[0].get("n_tiles"),
        "n_runs": len(summaries),
    }
    for key in _METRIC_KEYS:
        avg[key] = round(sum(s[key] for s in summaries) / len(summaries), 4)

    for key in ("tp", "fp", "fn", "n_gt_instances", "n_pred_instances"):
        if key in avg:
            avg[key] = int(round(avg[key]))

    return avg


def evaluate_multi_run(
    multi: MultiRunSimulationResult,
    threshold: float | None = None,
    iou_threshold: float | None = None,
) -> dict[str, Any]:
    """Per-run instance metrics plus averaged summary."""
    per_run = [
        _evaluate_result(r, threshold=threshold, iou_threshold=iou_threshold)
        for r in multi.runs
    ]
    averaged = _average_metrics(
        per_run,
        algorithm=multi.algorithm,
        drones=multi.drone_ids,
    )
    return {
        "averaged": averaged,
        "per_run": [{k: v for k, v in m.items() if k != "per_tile"} for m in per_run],
        "start_combinations": multi.start_combinations,
        "seed": multi.seed,
        "n_runs": multi.n_runs,
    }
