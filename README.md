# Multi-UAV Consensus Benchmark

This repository accompanies our IEEE SSRR 2026 paper, *Benchmarking Consensus Robustness in Multi-UAV Mine Detection under Controlled Visual Perception Degradation*. The repository provides a reproducible benchmark for evaluating how different consensus strategies handle degraded perception and faulty UAV agents, using clean and corrupted detector observations from volleyball, grass, and rubble minefields.

The repository includes the benchmark code, experiment configurations, selected detector weights, training setup, saved predictions, results, and visualization tools. You can reproduce the consensus experiments and paper figures directly from the supplied predictions, without retraining the detector or downloading the source imagery.

## Reproduction quick start

After [installing the package](#installation), choose the workflow that suits what you want to reproduce. Run all commands from the repository root.

| Goal | Required inputs | Steps |
| --- | --- | --- |
| Plot the paper results | Supplied paper summary and comparison reports | Run the plotting commands below with the supplied summary. |
| Rerun consensus and evaluation | Supplied per-UAV predictions and ground truth | Run the full consensus config and plot the resulting summary. No images or detector inference needed. |
| Regenerate observations | Source imagery obtained separately, plus the supplied labels and weights | Run the three observation configs, point a copy of the consensus config to the new predictions, then run consensus and plotting. |

The summary used for the paper is [pipeline_summary_2026-10-03_09-33-21.json](runs/consensus/all_fields_mixed_corruption_all_combinations/pipeline_summary_2026-10-03_09-33-21.json), which references reports from run `2026-10-02_21-15-25`.

The released paper summary uses report paths relative to its own directory, so it works from a different checkout without editing. Other saved artifact paths are relative to the repository root. Run the following commands from that root:

```bash
SUMMARY=runs/consensus/all_fields_mixed_corruption_all_combinations/pipeline_summary_2026-10-03_09-33-21.json
visualize_consensus_results "$SUMMARY" --metrics f1 precision recall --output-dir /tmp/ssrr26-diagnostic-plots
visualize_paper_figures "$SUMMARY" --output-dir /tmp/ssrr26-paper-figures
```

Here, the explicit output directories keep your new figures separate from the archived paper figures.

To rerun consensus from the supplied predictions:

```bash
execute_consensus configs/consensus_experiment_configs/all_fields_consensus_mixed_corruption_all_combinations.yaml
```

The command prints the path to its new summary, which you can then plot as described in [Running the benchmark](#running-the-benchmark). If you also want to retrain the detector, see [Reproducing the example detector](#reproducing-the-example-detector).

## Repository structure

| Path | Contents |
| --- | --- |
| [configs/](configs/) | Observation and consensus experiment definitions. |
| [data/labels/](data/labels/) | Full-image YOLO annotations; obtain the corresponding source images separately. |
| [models/](models/) | Published detector checkpoint, checksum manifest, hardware record, and training setup under `training/yolo26/`. |
| [runs/](runs/) | Saved UAV predictions, tile ground truth, consensus comparisons, the paper summary, and example paper figures. |
| [src/benchmark/](src/benchmark/) | Observation, consensus, evaluation, and visualization code. |
| [src/benchmark/cli.py](src/benchmark/cli.py) | Entry points for the five installed commands. |
| [tests/](tests/) | Checks for CLI behavior, consensus outputs, and paper figures. |

## Installation

Use Python **3.10 or newer**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
execute_consensus --help
```

The commands above assume bash. Dependencies such as PyTorch and Ultralytics are listed in [pyproject.toml](pyproject.toml). Make sure to run the benchmark from the repository root, since relative configuration paths are resolved from your working directory rather than the YAML file's location.

The workstation used for training and evaluation had an NVIDIA RTX 4000 Ada Generation GPU (approximately 20 GiB), an Intel Core Ultra 9 285K CPU, approximately 64 GB RAM, and Ubuntu 22.04.5 LTS. Its system Python was 3.10.12. See [Detector Reproducibility](models/REPRODUCIBILITY.md#2-workstation-hardware-and-operating-system) for the available hardware and software details, including the driver-supported CUDA version and the installed toolkit version. Retraining or generating detections on another machine may produce different results.

## Published detector weights

We include the selected YOLO26 nano checkpoint at [models/yolo26_170_epochs_balanced_detector.pt](models/yolo26_170_epochs_balanced_detector.pt). This is the exported checkpoint from the selected 170-epoch configuration used to generate the archived observations. All three observation configs already point to it, so you can use the published weights without changing any model paths or running training.

To verify the checkpoint, run this from the repository root:

```bash
(cd models && sha256sum -c SHA256SUMS.txt)
```

You should see `yolo26_170_epochs_balanced_detector.pt: OK`. The expected SHA-256 is `8a291f6f002ee3c89dfbb2f743c0c21e12811645eeede411071723cdb6a581c0`. More details are available in [Detector Reproducibility](models/REPRODUCIBILITY.md), and the checkpoint's licensing considerations are discussed [below](#license-and-dataset-note).

## Dataset

The original training data for the detector came from the Scatterable Landmine Detection Project datasets listed in [Binghamton University's Earth Sciences Faculty Scholarship collection](https://orb.binghamton.edu/geology_fac/). The benchmark minefield imagery also comes from these datasets. We don't include the source images in this repository, so you'll need to obtain them separately.

Save the source images as PNG files under `data/images/`, with filenames matching their corresponding `.txt` annotations. The three held-out benchmark fields use the following paths:

| Field | Expected image |
| --- | --- |
| Volleyball | `data/images/233c6f43-flight5_transparent_mosaic_group1.png` |
| Grass | `data/images/786cda2d-Grass_F2_RGB.png` |
| Rubble | `data/images/42432b96-Rubble_F2_RGB.png` |

If you download a TIFF, export its RGB raster to PNG without resizing, rotating, or cropping it. Keep the original pixel dimensions and use the exact annotation filename stem, including its prefix, so the supplied bounding boxes still align with the image.

To retrain the detector, you'll need source images for all 23 annotation files. The full-image labels are supplied in [data/labels/](data/labels/) as normalized YOLO bounding boxes (`class_id x_center y_center width height`), with class `0` representing PFM-1 landmines. The accompanying [export metadata](models/training/yolo26/notes.json) lists Label Studio as the contributor and PFM-1 as the only category. When generating observations, the benchmark looks for labels in `data/labels/` by default; you can change this using `input.label_path`.

## Reproducing the example detector

Retraining is optional. The scripts and dataset YAML are in [models/training/yolo26/](models/training/yolo26/), including `minefield_data_preprocessing.py`, `train_yolo_models.py`, and `evaluate_yolo_models.py`. The [training README](models/training/yolo26/README.md) covers the setup in more detail. If you're only interested in the consensus experiments, you can skip this section and use the supplied prediction JSON files directly.

### Prepare training data

Once the source PNGs and matching annotations are in `data/images/` and `data/labels/`, respectively, run:

```bash
python models/training/yolo26/minefield_data_preprocessing.py
```

The script creates `models/training/yolo26/preprocessed_tiled/images/{train,val,test}` and the corresponding `labels/` directories. It produces 1024-pixel tiles with 20% overlap, keeps tiles containing mines and up to 30 empty tiles per source image per split, and reserves all three benchmark fields for testing. For the remaining source images, the left half is used for training and the right half for validation.

**Set the dataset path before training or evaluation.** Both the checked-in `pfm1.yaml` and preprocessing output use `path: .`. Ultralytics can resolve this relative to its configured dataset directory. Use the snippet below to set the dataset's absolute location on your machine:

```bash
python - <<'PY'
from pathlib import Path
import yaml

dataset = Path("models/training/yolo26/preprocessed_tiled").resolve()
config = dataset / "pfm1.yaml"
data = yaml.safe_load(config.read_text())
data["path"] = str(dataset)
config.write_text(yaml.safe_dump(data, sort_keys=False))
PY
```

### Train and evaluate

The training script expects to be run from its own directory:

```bash
cd models/training/yolo26
python train_yolo_models.py
cd ../../..
```

By default, the script trains ten models for 10, 20, 40, 70, 80, 110, 130, 150, 170, and 200 epochs, using seeds 42–51 in that order, image size 1024, and batch size 16. Checkpoints are saved under `models/training/yolo26/runs/train_ensemble/`. The selected 170-epoch run uses seed 50 and writes `weights/best.pt` and `weights/last.pt` under `pfm1_yolo26_epochs_170_seed_50/`, although repeated runs may add a suffix to the directory name. To train only that configuration, set `EPOCHS_LIST = [170]` and `BASE_SEED = 50` in your local copy of the script.

If you train a new checkpoint, update each observation agent's `model` path in a copy of the config. Otherwise, the supplied configs are already set up to use the published weights.

You can also evaluate the published detector on the generated test tiles. From the repository root, run:

```bash
python models/training/yolo26/evaluate_yolo_models.py \
  --data models/training/yolo26/preprocessed_tiled/pfm1.yaml \
  --models models/yolo26_170_epochs_balanced_detector.pt \
  --split test --imgsz 1024 --conf 0.25 \
  --output-dir /tmp/ssrr26-detector-evaluation
```

The evaluation produces `summary.json`, `summary.csv`, and model-specific validation outputs. To test your own checkpoint, replace the `--models` path. These detector-level results are separate from the consensus evaluation; the saved metrics for the selected detector are described in the training README.

## Running the benchmark

### 1. Generate corrupted observations and UAV detections

If you're using the supplied prediction JSON files, you can skip straight to consensus. To regenerate the observations yourself, first obtain the three source images, then run these commands with the supplied detector weights:

```bash
execute_observations configs/observation_experiment_configs/volleyball_field_extended_agents.yaml
execute_observations configs/observation_experiment_configs/grass_field_extended_agents.yaml
execute_observations configs/observation_experiment_configs/rubble_field_extended_agents.yaml
```

Each command divides a source image into tiles, applies the configured visual corruptions, runs the detector, and saves both the predictions and tile-level ground truth. The configs cover clean observations and severity sweeps for Gaussian noise, motion blur, snow, pixelation (`pixelate`), and defocus.

Each run gets a timestamped scenario directory under `runs/observations/`, containing a config copy, `ground_truth/tile_ground_truths.json`, `predictions/<agent_id>_predictions.json`, and `predictions/all_predictions.json`. The configs also save the corrupted image tiles under `corrupted_tiles/<agent_id>/`. If you don't need these images, set `output.save_corrupted_tiles: false` in a copy of the config.

### 2. Run consensus and evaluation

Using the supplied predictions, run the main experiment from the repository root:

```bash
execute_consensus configs/consensus_experiment_configs/all_fields_consensus_mixed_corruption_all_combinations.yaml
```

The experiment uses five per-UAV prediction files for each condition and compares nine strategies: `LCP`, `WMSR`, `CDCI`, `DBBS`, `MLES`, `EEV`, `DBC`, `MAJORITY_VOTE`, and `NO_CONSENSUS`. Across the three fields, the config covers 933 conditions: three clean baselines and, for each field, all 31 nonempty corruption combinations at ten nonzero severity levels. Each condition is evaluated over ten simulation runs, using seed 42 to generate the randomized UAV start tiles shared by the algorithms.

Predicted and ground-truth boxes are matched greedily, one to one, within each tile at IoU **0.5**. The decision threshold for accepting consensus slots is **−0.6**; leaving the thresholds as `null` in the YAML uses these defaults. The comparison reports give instance-level precision, recall, F1, TP, FP, and FN averaged over the ten runs. For `NO_CONSENSUS`, TP, FP, and FN are first averaged across UAVs within each run, and precision, recall, and F1 are calculated from those mean counts. This is the consensus evaluation, rather than the detector's separate test-split validation.

The results and logs are saved under `runs/consensus/all_fields_mixed_corruption_all_combinations/`. A `pipeline_summary_<timestamp>.json` file links to the individual experiment reports.

**If you regenerated detections:** make a copy of the consensus YAML and update each experiment's `input_files` to point to the new observation runs. The supplied YAML still points to the archived June runs, so you need to change these paths manually. Keep the agent filenames and experiment names intact, since the visualization code uses them to identify conditions.

### 3. Generate diagnostic and paper figures

After running consensus, copy the summary path printed in the terminal into `SUMMARY`, then generate the figures:

```bash
visualize_consensus_results "$SUMMARY" --metrics f1 precision recall
visualize_paper_figures "$SUMMARY"
```

Both commands read the summary and the comparison reports it references. Diagnostic plots and `consensus_results_flat.csv` are saved in a `plots/` directory alongside the summary. The `paper_figures/` directory contains PNG/PDF panels and heatmaps, `consensus_results_flat_for_paper.csv`, CSV tables, and `algorithm_mean_metrics_table.tex`. You can choose another location with `--output-dir`.

A fresh consensus run is the easiest option when moving between machines. To plot the supplied paper summary, copy it and update `experiments[].reports` to point to your checkout. As noted in the quick start, `--repo-root` currently remaps older `results/` paths but not the `runs/` layout.

## CLI reference

| Command | Purpose | Typical input | Output |
| --- | --- | --- | --- |
| `execute_observations CONFIG` | Tile imagery, apply corruption, run detection, and attach ground truth. | Observation YAML, image, labels, weights. | Timestamped observation run with JSON and optional corrupted tiles. |
| `execute_consensus CONFIG` | Run consensus strategies and evaluate decisions. | Consensus YAML referencing per-UAV prediction JSON. | Comparison/algorithm reports, pipeline summary, log. |
| `visualize_uav_predictions JSON` | Inspect UAV detections and majority vote. | New `all_predictions.json`. | PNGs and CSVs; defaults to `prediction_visualizations/` in the working directory. |
| `visualize_consensus_results SUMMARY` | Generate diagnostic robustness plots. | Pipeline summary and referenced reports. | `plots/` with PNGs and flattened/ranking CSVs. |
| `visualize_paper_figures SUMMARY` | Generate publication figures and tables. | Pipeline summary and referenced reports. | `paper_figures/` with PNG/PDF figures, CSVs, and a LaTeX table. |

Add `--help` to any command to see its arguments. The visualization commands accept `--output-dir`, while the pipeline commands take a positional config path.

## Configurations

| Configuration group | Use |
| --- | --- |
| [Observation configs](configs/observation_experiment_configs/) | `volleyball_field_extended_agents.yaml`, `grass_field_extended_agents.yaml`, and `rubble_field_extended_agents.yaml` each define 56 observation agents: a clean reference plus five corruption sweeps at severities 0.0–1.0. Each uses tile size and stride 1024. |
| [Full consensus config](configs/consensus_experiment_configs/all_fields_consensus_mixed_corruption_all_combinations.yaml) | Five-UAV clean baselines and all nonempty combinations of five corruption types at ten nonzero severity levels, across three fields. `compare: true` evaluates all registered algorithms. |

## Outputs and example results

An archived volleyball observation run is available under [runs/observations/volleyballfield_drone_run_scenario/](runs/observations/volleyballfield_drone_run_scenario/). The main consensus results and paper figures are in [runs/consensus/all_fields_mixed_corruption_all_combinations/](runs/consensus/all_fields_mixed_corruption_all_combinations/). The paper uses `pipeline_summary_2026-10-03_09-33-21.json`, referencing reports from `2026-10-02_21-15-25`. The 933 reports contain 8,397 condition/algorithm rows, matching the archived paper CSV for precision, recall, F1, TP, FP, and FN.

```text
runs/
├── observations/<field-output-dir>/<timestamp>_<scenario>/
│   ├── config.yaml
│   ├── ground_truth/tile_ground_truths.json
│   ├── predictions/<agent_id>_predictions.json
│   ├── predictions/all_predictions.json       # new observation runs
│   └── corrupted_tiles/<agent_id>/tile_*.png   # when enabled
└── consensus/all_fields_mixed_corruption_all_combinations/
    ├── logs/consensus_pipeline_<timestamp>.log
    ├── <experiment>/<timestamp>/experiment_no_<index>/
    │   └── comparison/comparison_5uav.json
    ├── pipeline_summary_<timestamp>.json
    ├── plots/                                # diagnostic visualization output
    └── paper_figures/                         # paper visualization output
```

Setting `compare: false` produces separate reports for each algorithm instead of a combined comparison. For reruns, consensus takes the per-UAV prediction JSON files as input, while the visualization tools read the pipeline summary and its reports. The detector validation summaries and exported figure CSVs are outputs for other purposes.

Archived paper artifacts include `paper_panel_vs_severity.pdf`, `paper_panel_vs_num_corrupted.pdf`, `consensus_gain_heatmap_f1.csv`, and `algorithm_mean_metrics_table.csv`. See the paper for interpretation of the results.

## Reproducing the paper experiments

**Recommended: use the supplied predictions.** Install the package, run the full consensus config from stage 2, and pass the resulting summary to `visualize_paper_figures` in stage 3. We include all 165 distinct prediction files referenced by the config, along with ground truth, so you can reproduce the consensus evaluation and paper figures without obtaining the images or retraining the detector.

**Regenerate predictions with the published weights.** Obtain the three held-out source images matching the provided annotations, verify the checkpoint, and run observation generation for each field. Then update a copy of the consensus config to use the new prediction directories before running consensus and plotting. You can skip training-data preprocessing here, since observation generation tiles the original images directly.

**Retrain the detector (optional).** You'll need all source images matching the 23 annotation files. Preprocess them, fix the dataset YAML path, train the detector, and select a checkpoint before following the observation-generation steps above.

Even with the published weights, regenerated predictions may differ from the archived files. The corruption pipeline derives seeds using Python's built-in `hash()` on a tuple containing the drone ID, which means the random corruptions can change with the process's hash seed even when the experiment seed stays the same. Detector outputs also depend on inference settings such as image size, confidence threshold, device, and precision; using 1024-pixel tiles alone doesn't fix the inference resolution. Retraining introduces further variation. For the closest reproduction of the consensus experiments reported in the paper, we therefore recommend starting from the archived prediction JSON files.

## Citation

The citation below is a placeholder until the final SSRR 2026 publication details are available.

```bibtex
@inproceedings{tifli2026benchmarking,
  author    = {Aman Tifli and Gianluca Manca and Marvin Zager and
               Felix Gehlhoff and Galyna Tabunshchyk and Alexander Fay},
  title     = {Benchmarking Consensus Robustness in Multi-UAV Mine Detection
               under Controlled Visual Perception Degradation},
  booktitle = {2026 IEEE International Symposium on Safety, Security,
               and Rescue Robotics (SSRR)},
  year      = {2026},
  publisher = {IEEE},
  note      = {To appear}
}
```

## License and dataset note

The repository code is licensed under the [Apache License 2.0](LICENSE).

**Detector weights and dependencies:** the released checkpoint is distributed under AGPL-3.0, as stated in [models/LICENSE.md](models/LICENSE.md). Apache-2.0 covers the repository code; third-party dependencies, including Ultralytics, retain their own license terms.

**Imagery:** source images and derived tiles need to be obtained separately. Check the licensing and attribution requirements for the specific image records you use, separately from the repository's code license.
