#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
import shutil
import random


# ============================================================================
# CONFIG
# ============================================================================

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("models/training/yolo26/preprocessed_tiled")

TILE_SIZE = 1024
OVERLAP = 0.2
MIN_VISIBLE_FRACTION = 0.3

MAX_EMPTY_TILES_PER_IMAGE_PER_SPLIT = 30

RANDOM_SEED = 42

# These complete source images will be used as the test set.
# They are not split into train/val.
TEST_SOURCES = {
    "233c6f43-flight5_transparent_mosaic_group1",  # Volleyball field
    "786cda2d-Grass_F2_RGB",                       # Grass
    "42432b96-Rubble_F2_RGB",                      # Rubble
}


# ============================================================================
# CONVERSION HELPERS
# ============================================================================

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    box_w = w * img_w
    box_h = h * img_h
    x_center = xc * img_w
    y_center = yc * img_h

    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2

    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, crop_w, crop_h):
    xc = ((x1 + x2) / 2) / crop_w
    yc = ((y1 + y2) / 2) / crop_h
    w = (x2 - x1) / crop_w
    h = (y2 - y1) / crop_h

    return xc, yc, w, h


def generate_tile_positions(start, end, tile_size, stride):
    positions = []

    region_size = end - start

    if region_size <= tile_size:
        return [start]

    pos = start
    while (pos + tile_size) < end:
        positions.append(pos)
        pos += stride

    final_pos = end - tile_size
    if not positions or positions[-1] != final_pos:
        positions.append(final_pos)

    return positions


def labels_for_tile(labels, tile_box):
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_box
    tile_w = tile_x2 - tile_x1
    tile_h = tile_y2 - tile_y1

    new_labels = []

    for label in labels:
        class_id, x1, y1, x2, y2 = label

        original_area = max(0, x2 - x1) * max(0, y2 - y1)
        if original_area <= 0:
            continue

        ix1 = max(x1, tile_x1)
        iy1 = max(y1, tile_y1)
        ix2 = min(x2, tile_x2)
        iy2 = min(y2, tile_y2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        visible_area = (ix2 - ix1) * (iy2 - iy1)
        visible_fraction = visible_area / original_area

        if visible_fraction < MIN_VISIBLE_FRACTION:
            continue

        local_x1 = ix1 - tile_x1
        local_y1 = iy1 - tile_y1
        local_x2 = ix2 - tile_x1
        local_y2 = iy2 - tile_y1

        xc, yc, w, h = xyxy_to_yolo(
            local_x1,
            local_y1,
            local_x2,
            local_y2,
            tile_w,
            tile_h,
        )

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        if w <= 0 or h <= 0:
            continue

        new_labels.append(f"{class_id} {xc:.12f} {yc:.12f} {w:.12f} {h:.12f}")

    return new_labels


def save_tile(image, tile_box, labels, output_image_path, output_label_path):
    cropped = image.crop(tile_box)
    cropped.save(output_image_path)

    with open(output_label_path, "w") as f:
        f.write("\n".join(labels))
        if labels:
            f.write("\n")


def process_split_region(
    image,
    labels,
    image_path,
    split_name,
    region_box,
    output_images_dir,
    output_labels_dir,
):
    region_x1, region_y1, region_x2, region_y2 = region_box

    stride = int(TILE_SIZE * (1.0 - OVERLAP))
    stride = max(1, stride)  # just in case stride=0

    x_positions = generate_tile_positions(region_x1, region_x2, TILE_SIZE, stride)
    y_positions = generate_tile_positions(region_y1, region_y2, TILE_SIZE, stride)

    positive_tiles = []
    empty_tiles = []

    for y in y_positions:
        for x in x_positions:
            tile_x1 = x
            tile_y1 = y
            tile_x2 = min(x + TILE_SIZE, region_x2)
            tile_y2 = min(y + TILE_SIZE, region_y2)

            if tile_x2 - tile_x1 < TILE_SIZE:
                tile_x1 = max(region_x1, tile_x2 - TILE_SIZE)
            if tile_y2 - tile_y1 < TILE_SIZE:
                tile_y1 = max(region_y1, tile_y2 - TILE_SIZE)

            tile_box = (tile_x1, tile_y1, tile_x2, tile_y2)
            tile_labels = labels_for_tile(labels, tile_box)

            tile_info = (tile_box, tile_labels)

            if tile_labels:
                positive_tiles.append(tile_info)
            else:
                empty_tiles.append(tile_info)

    random.shuffle(empty_tiles)
    kept_empty_tiles = empty_tiles[:MAX_EMPTY_TILES_PER_IMAGE_PER_SPLIT]
    # kept_empty_tiles = empty_tiles

    all_tiles = positive_tiles + kept_empty_tiles

    for idx, (tile_box, tile_labels) in enumerate(all_tiles):
        tile_suffix = f"{split_name}_tile_{idx:04d}"
        output_image_name = f"{image_path.stem}_{tile_suffix}{image_path.suffix}"
        output_label_name = f"{image_path.stem}_{tile_suffix}.txt"

        save_tile(
            image=image,
            tile_box=tile_box,
            labels=tile_labels,
            output_image_path=output_images_dir / split_name / output_image_name,
            output_label_path=output_labels_dir / split_name / output_label_name,
        )

    return len(positive_tiles), len(kept_empty_tiles)


def write_yaml(output_dir):
    yaml_text = """path: .

train: images/train
val: images/val
test: images/test

names:
  0: pfm1
"""

    with open(output_dir / "pfm1.yaml", "w") as f:
        f.write(yaml_text)


def main():
    random.seed(RANDOM_SEED)

    input_images_dir = INPUT_DIR / "images"
    input_labels_dir = INPUT_DIR / "labels"

    output_images_dir = OUTPUT_DIR / "images"
    output_labels_dir = OUTPUT_DIR / "labels"

    for split in ["train", "val", "test"]:
        (output_images_dir / split).mkdir(parents=True, exist_ok=True)
        (output_labels_dir / split).mkdir(parents=True, exist_ok=True)

    classes_file = INPUT_DIR / "classes.txt"
    if classes_file.exists():
        shutil.copy(classes_file, OUTPUT_DIR / "classes.txt")

    write_yaml(OUTPUT_DIR)

    image_paths = sorted(input_images_dir.glob("*.png"))

    total_train_positive = 0
    total_train_empty = 0
    total_val_positive = 0
    total_val_empty = 0
    total_test_positive = 0
    total_test_empty = 0

    for image_path in image_paths:
        label_path = input_labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Warning: no label file for {image_path.name}, skipping")
            continue

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        labels = []

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = parts[0]
                xc, yc, w, h = map(float, parts[1:])

                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                labels.append((class_id, x1, y1, x2, y2))

        mid_x = img_w // 2

        left_region = (0, 0, mid_x, img_h)
        right_region = (mid_x, 0, img_w, img_h)
        full_region = (0, 0, img_w, img_h)

        if image_path.stem in TEST_SOURCES:
            test_pos, test_empty = process_split_region(
                image=image,
                labels=labels,
                image_path=image_path,
                split_name="test",
                region_box=full_region,
                output_images_dir=output_images_dir,
                output_labels_dir=output_labels_dir,
            )

            total_test_positive += test_pos
            total_test_empty += test_empty

            print(
                f"Processed {image_path.name}: "
                f"test-full {test_pos} positive + {test_empty} empty"
            )

        else:
            train_positive, train_empty = process_split_region(
                image=image,
                labels=labels,
                image_path=image_path,
                split_name="train",
                region_box=left_region,
                output_images_dir=output_images_dir,
                output_labels_dir=output_labels_dir,
            )

            val_positive, val_empty = process_split_region(
                image=image,
                labels=labels,
                image_path=image_path,
                split_name="val",
                region_box=right_region,
                output_images_dir=output_images_dir,
                output_labels_dir=output_labels_dir,
            )

            total_train_positive += train_positive
            total_train_empty += train_empty
            total_val_positive += val_positive
            total_val_empty += val_empty

            print(
                f"Processed {image_path.name}: "
                f"train-left {train_positive} positive + {train_empty} empty, "
                f"val-right {val_positive} positive + {val_empty} empty"
            )

    print("\nDone.")
    print(f"Dataset written to: {OUTPUT_DIR}")
    print(f"Train: {total_train_positive} positive tiles + {total_train_empty} empty tiles")
    print(f"Val:   {total_val_positive} positive tiles + {total_val_empty} empty tiles")
    print(f"Test:  {total_test_positive} positive tiles + {total_test_empty} empty tiles")


if __name__ == "__main__":
    main()
