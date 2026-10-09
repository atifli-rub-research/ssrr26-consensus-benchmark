# PFM-1 mine detector training

This folder contains the scripts for tiling the annotated minefield images,
training YOLO26 nano detectors, and evaluating their checkpoints. We compared
models trained with several epoch budgets and selected the **170-epoch version**
because it performed best in our experiments. Its exported checkpoint is named
`yolo26_170_epochs_balanced_detector.pt` and is included at
[`models/yolo26_170_epochs_balanced_detector.pt`](../../yolo26_170_epochs_balanced_detector.pt).
Retraining is optional: the root [README](../../../README.md#reproduction-quick-start)
explains how to use the supplied predictions or weights. See
[Detector Reproducibility](../../REPRODUCIBILITY.md) for the existing checksum,
hardware record, and reproduction limits.

## 1. Set up the environment

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The repository dependencies include Ultralytics, PyTorch, NumPy, Pillow, and
PyYAML. Use an Ultralytics installation that supports `yolo26n.pt`. A CUDA-capable
GPU is recommended for training; the script leaves device selection to
Ultralytics. Place the pretrained `yolo26n.pt` weights in this folder, or allow
Ultralytics to download them when training starts.

## 2. Preprocess the source images

The preprocessing script uses paths relative to the current working directory,
so run it **from the repository root**. Arrange the input as:

```text
data/
  images/<source-name>.png
  labels/<source-name>.txt
  classes.txt                 # optional
```

Each label file must share its image's stem and contain YOLO bounding boxes:
`class_id x_center y_center width height`, with normalized coordinates. The
dataset has one class: `0` = PFM-1 landmine (`pfm1` in the dataset YAML). Only PNG
images are read, and images without a matching label file are skipped.

```bash
python models/training/yolo26/minefield_data_preprocessing.py
```

The script writes `models/training/yolo26/preprocessed_tiled/`, containing
`images/{train,val,test}`, corresponding `labels/{train,val,test}`, and
`pfm1.yaml`. Its configuration is defined at the top of the script:

- Tiles are up to 1024 × 1024 pixels, with 20% overlap (819-pixel stride).
- Boxes are clipped to each tile and retained if at least 30% of their original
  area remains visible.
- All tiles with retained boxes are kept. At most 30 empty tiles per source image
  per split are sampled, using random seed 42.
- For ordinary source images, the left half becomes training data and the right
  half becomes validation data. Tiling happens within each half.
- Three complete source images are reserved exclusively for testing:
  `233c6f43-flight5_transparent_mosaic_group1` (volleyball field),
  `786cda2d-Grass_F2_RGB` (grass), and `42432b96-Rubble_F2_RGB` (rubble).

Use a fresh output directory when changing preprocessing settings: the script
creates directories and overwrites matching files but does not remove old tiles.
If the tiled dataset already exists, you can skip preprocessing.

### Set the dataset path

Before training or evaluation, set the `path` entry in `preprocessed_tiled/pfm1.yaml` to the
absolute location of the tiled dataset on your machine. The preprocessing script
writes `path: .`, which is also the checked-in value. Ultralytics may resolve this
relative to its configured dataset directory rather than your checkout.
Open the YAML in your editor and replace its `path` value. For example:

```yaml
path: /absolute/path/to/ssrr26-consensus-benchmark/models/training/yolo26/preprocessed_tiled
```

Replace `/absolute/path/to/ssrr26-consensus-benchmark` with your checkout location.
The remaining YAML entries should be `train: images/train`, `val: images/val`,
`test: images/test`, and `names: {0: pfm1}`.

## 3. Train the models

From the repository root, switch to this folder and run training with the
environment activated:

```bash
cd models/training/yolo26
python train_yolo_models.py
```

There are no command-line options. Edit the constants at the top of
`train_yolo_models.py` to change the experiment:

| Setting | Current value |
| --- | --- |
| Starting weights | `yolo26n.pt` |
| Dataset | `preprocessed_tiled/pfm1.yaml` |
| Epoch budgets | 10, 20, 40, 70, 80, 110, 130, 150, 170, 200 |
| Image size | 1024 |
| Batch size | 16 |
| Data loader workers | 0 |
| Output project | `runs/train_ensemble` |
| Seeds | 42 through 51, in epoch-budget order |

Each run creates a new model from the same starting weights; runs do not resume
from the preceding model. Python, NumPy, and PyTorch seeds are set, and the seed
is also passed to Ultralytics. Other training settings use Ultralytics defaults,
so record your installed package versions when running a new experiment. The
available historical workstation information is documented in
[Detector Reproducibility](../../REPRODUCIBILITY.md#2-workstation-hardware-and-operating-system);
it does not establish the training process's interpreter or package versions.

Run directories follow `pfm1_yolo26_epochs_<epochs>_seed_<seed>`. The 170-epoch
run in the full sweep uses seed 50 and writes checkpoints under:

```text
runs/train_ensemble/pfm1_yolo26_epochs_170_seed_50/weights/
  best.pt
  last.pt
```

`best.pt` is the checkpoint selected by Ultralytics during that run; `last.pt`
is the final checkpoint. The epoch budget in the run name does not identify the
epoch of `best.pt`. Inspect the run's logs for the actual training duration and
checkpoint selection. Repeated runs may receive a suffixed output directory
because `exist_ok=False`.

To train only the selected configuration, change `EPOCHS_LIST` to `[170]` and
`BASE_SEED` to `50`. Changing only the list would use seed 42 instead. Reduce
`BATCH_SIZE` if training exceeds available GPU memory.

## 4. Evaluate checkpoints

From this folder, evaluate the 170-epoch run's best checkpoint:

```bash
python evaluate_yolo_models.py \
  --data preprocessed_tiled/pfm1.yaml \
  --models runs/train_ensemble/pfm1_yolo26_epochs_170_seed_50/weights/best.pt \
  --output-dir runs/eval_170 \
  --split test --imgsz 1024 --conf 0.25
```

To evaluate the published selected model from this training folder, use its
existing location and a separate output directory:

```bash
python evaluate_yolo_models.py \
  --data preprocessed_tiled/pfm1.yaml \
  --models ../../yolo26_170_epochs_balanced_detector.pt \
  --output-dir /tmp/ssrr26-detector-evaluation \
  --split test --imgsz 1024 --conf 0.25
```

Pass multiple paths after `--models` to compare checkpoints. For example:

```bash
python evaluate_yolo_models.py \
  --data preprocessed_tiled/pfm1.yaml \
  --models runs/train_ensemble/*/weights/best.pt \
  --output-dir runs/eval_all --split test --conf 0.25
```

The defaults are the test split, image size 1024, confidence threshold 0.25,
and output directory `runs/test_eval_conf025`. The saved detector evaluation
uses confidence 0.25; use `--conf 0.001` for standard mAP-style validation.
`--split val` evaluates the validation split. Observation generation relies on
checkpoint/library inference settings, so this validation command does not
establish all of the observation pipeline's effective settings.

The script writes `summary.csv` and `summary.json` with precision, recall,
F1 (computed from mean precision and recall), mAP50, and mAP50–95. Ultralytics
plots and prediction JSON are enabled by default; disable them with `--no-plots`
and `--no-json`. Summaries are overwritten on each invocation. Per-model outputs
use the checkpoint filename stem, so multiple files named `best.pt` share the
same output subdirectory; use separate output directories or distinctly named
checkpoint copies when preserving each model's plots and predictions.

The saved evaluation in `runs/test_eval_conf025/summary.csv` reports the following
for the exported 170-epoch detector on the test split at confidence 0.25:

| Precision | Recall | F1 | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- |
| 0.8037 | 0.8231 | 0.8133 | 0.8054 | 0.3884 |

That summary contains only the selected model, rather than the full comparison
used to choose the 170-epoch version.
