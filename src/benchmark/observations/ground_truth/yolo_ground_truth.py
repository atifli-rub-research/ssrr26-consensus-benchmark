from pathlib import Path
from typing import Any

from PIL import Image
from loguru import logger

from benchmark.observations.pipeline.config import ObservationPipelineConfig


def load_tile_ground_truths(image_path: Path, config: ObservationPipelineConfig, image_crops: list) -> dict:
    """Load ground truth data from YOLO label files and process for tile assignment.
    
    Args:
        image_path: Path to the input image (used to find label file)
        config: Pipeline configuration
        image_crops: List of tile crops for validation
        
    Returns:
        dict: Ground truth data indexed by tile index
    """
    label_path = get_label_path_for_image(image_path=image_path, config=config)
    
    # Get actual image dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    global_ground_truths = load_yolo_ground_truths(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )

    tile_ground_truths = build_tile_ground_truths_from_crops(
        global_ground_truths=global_ground_truths,
        image_crops=image_crops,
    )

    if len(tile_ground_truths) != len(image_crops):
        raise RuntimeError(
            "Ground-truth tile count does not match generated image crops: "
            f"{len(tile_ground_truths)} ground-truth tiles vs "
            f"{len(image_crops)} image crops"
        )

    ground_truth_by_tile = {
        tile_index: ground_truth
        for tile_index, ground_truth in enumerate(tile_ground_truths)
    }

    logger.info(
        f"Done ground truth loading "
        f"({len(global_ground_truths)} global mines, "
        f"{sum(gt['mine_count'] for gt in tile_ground_truths)} tile mines)"
    )
    return ground_truth_by_tile


def get_label_path_for_image(
    image_path: str | Path,
    config: ObservationPipelineConfig,
) -> Path:
    """
    Resolve the YOLO label file for the configured image.

    Default assumption:
        data/images/foo.png -> data/labels/foo.txt

    Optional config override:
        input.label_path: /path/to/foo.txt
    """

    configured_label_path = config.get(
        "input.label_path",
        None,
    )

    if configured_label_path is not None:
        label_path = Path(configured_label_path)
    else:
        image_path = Path(image_path)

        label_path = (
            image_path.parent.parent
            / "labels"
            / f"{image_path.stem}.txt"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Ground truth label file not found: {label_path}"
        )

    return label_path



def load_yolo_ground_truths(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Load YOLO-format labels from a full image.

    Expected format per line:
        class_id x_center y_center width height

    Coordinates are assumed to be normalized to the full image.
    Returned boxes are in full-image pixel coordinates.
    """

    label_path = Path(label_path)

    ground_truths = []

    with label_path.open("r") as f:
        for line_index, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_index + 1}: "
                    f"expected 5 values, got {len(parts)}"
                )

            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            box_width = width_norm * image_width
            box_height = height_norm * image_height

            x_min = x_center - box_width / 2.0
            y_min = y_center - box_height / 2.0
            x_max = x_center + box_width / 2.0
            y_max = y_center + box_height / 2.0

            ground_truths.append(
                {
                    "class_id": class_id,
                    "bbox_xyxy": [
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                    ],
                    "center_xy": [
                        float(x_center),
                        float(y_center),
                    ],
                }
            )

    return ground_truths

def _get_crop_bounds(crop: Any) -> tuple[float, float, float, float]:
    metadata = crop.metadata

    return (
        float(metadata["global_x1"]),
        float(metadata["global_y1"]),
        float(metadata["global_x2"]),
        float(metadata["global_y2"]),
    )


def _center_inside_crop(
    center_xy: list[float],
    crop: Any,
) -> bool:
    x_center, y_center = center_xy
    x1, y1, x2, y2 = _get_crop_bounds(crop)

    return (
        x1 <= x_center < x2
        and y1 <= y_center < y2
    )



def _distance_to_crop_center_sq(
    center_xy: list[float],
    crop: Any,
) -> float:
    x_center, y_center = center_xy
    x1, y1, x2, y2 = _get_crop_bounds(crop)

    crop_cx = 0.5 * (x1 + x2)
    crop_cy = 0.5 * (y1 + y2)

    return (
        (x_center - crop_cx) ** 2
        + (y_center - crop_cy) ** 2
    )



def _select_owner_crop(
    ground_truth: dict,
    candidate_crops: list,
) -> Any:
    """
    Select exactly one owner crop for a ground-truth mine.

    If a mine center lies inside multiple overlapping crops, the owner is
    the crop whose center is closest to the mine center. The tile index is
    used as a deterministic tie-breaker.
    """

    center_xy = ground_truth["center_xy"]

    return min(
        candidate_crops,
        key=lambda crop: (
            _distance_to_crop_center_sq(center_xy, crop),
            crop.metadata["tile_index"],
        ),
    )



def build_tile_ground_truths_from_crops(
    global_ground_truths: list[dict],
    image_crops: list,
) -> list[dict]:
    """
    Assign global ground-truth boxes to generated crops.

    Each mine instance is assigned to exactly one owner tile. If the mine
    center lies inside multiple overlapping crops, the owner tile is chosen
    as the crop whose center is closest to the mine center.

    Returned boxes are converted to tile-local coordinates.
    """

    tile_ground_truths = []

    for tile_index, crop in enumerate(image_crops):
        metadata = crop.metadata

        tile_x_min = metadata["global_x1"]
        tile_y_min = metadata["global_y1"]
        tile_x_max = metadata["global_x2"]
        tile_y_max = metadata["global_y2"]

        tile_ground_truths.append(
            {
                "tile_index": tile_index,
                "cell_id": crop.cell_id,
                "row": crop.row,
                "col": crop.col,
                "tile_bounds_xyxy": [
                    int(tile_x_min),
                    int(tile_y_min),
                    int(tile_x_max),
                    int(tile_y_max),
                ],
                "mine_present": False,
                "mine_count": 0,
                "boxes": [],
            }
        )

    for gt_id, ground_truth in enumerate(global_ground_truths):
        candidate_crops = [
            crop
            for crop in image_crops
            if _center_inside_crop(
                center_xy=ground_truth["center_xy"],
                crop=crop,
            )
        ]

        if not candidate_crops:
            logger.warning(
                "Ground-truth mine center was not inside any crop: "
                "gt_id={}, center={}",
                gt_id,
                ground_truth["center_xy"],
            )
            continue

        owner_crop = _select_owner_crop(
            ground_truth=ground_truth,
            candidate_crops=candidate_crops,
        )

        owner_tile_index = owner_crop.metadata["tile_index"]

        tile_x_min = owner_crop.metadata["global_x1"]
        tile_y_min = owner_crop.metadata["global_y1"]

        x_min, y_min, x_max, y_max = ground_truth["bbox_xyxy"]

        local_x_min = x_min - tile_x_min
        local_y_min = y_min - tile_y_min
        local_x_max = x_max - tile_x_min
        local_y_max = y_max - tile_y_min

        tile_ground_truths[owner_tile_index]["boxes"].append(
            {
                "class_id": ground_truth["class_id"],
                "bbox_xyxy": [
                    float(local_x_min),
                    float(local_y_min),
                    float(local_x_max),
                    float(local_y_max),
                ],
            }
        )

    for tile_ground_truth in tile_ground_truths:
        mine_count = len(tile_ground_truth["boxes"])
        tile_ground_truth["mine_count"] = mine_count
        tile_ground_truth["mine_present"] = mine_count > 0

    return tile_ground_truths
