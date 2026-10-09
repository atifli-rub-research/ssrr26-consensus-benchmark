# Detector Reproducibility

This document records the YOLO26 detector checkpoint and workstation used for our SSRR 2026 multi-UAV mine-detection benchmark, along with the main steps for reproducing the experiments.

## 1. Detector checkpoint

The experiments use the 170-epoch balanced-detector checkpoint, provided at:

```text
models/yolo26_170_epochs_balanced_detector.pt
```

Its SHA-256 checksum is:

```text
8a291f6f002ee3c89dfbb2f743c0c21e12811645eeede411071723cdb6a581c0
```

### Verify the checkpoint

From the repository root, check the file against the supplied [SHA256SUMS.txt](SHA256SUMS.txt) manifest:

```bash
(cd models && sha256sum -c SHA256SUMS.txt)
```

Expected output:

```text
yolo26_170_epochs_balanced_detector.pt: OK
```

You can also calculate the checksum directly:

```bash
sha256sum models/yolo26_170_epochs_balanced_detector.pt
```

A matching checksum confirms that the checkpoint matches the file used for the published detector outputs.

The checkpoint is preserved byte for byte, including historical workstation paths in its embedded training metadata. Those paths are provenance metadata; they do not need to exist to use the published weights for inference.

## 2. Workstation hardware and operating system

The detector was trained and evaluated on the following Ubuntu workstation. The paper reports the detector's performance in Table 1.

| Component | Recorded specification |
| --- | --- |
| CPU | Intel Core Ultra 9 285K |
| Logical CPUs | 24 |
| System RAM | 62 GiB reported (approximately 64 GB nominal) |
| GPU | NVIDIA RTX 4000 Ada Generation |
| GPU memory | 20,475 MiB reported (approximately 20 GiB) |
| Motherboard | Dell Inc. 0D8XDK, revision A00 |
| Operating system | Ubuntu 22.04.5 LTS (Jammy) |
| Linux kernel | 6.8.0-124-generic, x86_64 |
| NVIDIA driver | 580.159.04 |
| CUDA version supported by the driver | 13.0 |
| Locally installed CUDA toolkit | 11.5 (`nvcc` V11.5.119) |
| System Python | 3.10.12 |

## 3. Reproducibility scope

The benchmark separates observation generation from consensus and evaluation, so the experiments can be reproduced at two levels:

1. **Perception:** Generate per-UAV detections using the supplied checkpoint, image tiling, corruptions, and inference pipeline.
2. **Consensus and evaluation:** Run the fusion methods on the supplied per-UAV prediction files and evaluate their outputs against ground truth.

For reproducing the paper's consensus results, we recommend starting with the supplied prediction files. This uses the same detector observations for every fusion method and avoids rerunning the perception pipeline.

### Observation generation

Each tile's corruption seed is calculated as:

```python
hash((base_seed, drone_id, tile_index)) % (2**32)
```

The seed depends on Python's process hash seed because `drone_id` is a string. When generating new observations across multiple processes, set a fixed `PYTHONHASHSEED` to keep the corruption seeds consistent between runs.

The observation pipeline calls `YOLO.predict()` using the inference settings provided by the checkpoint and installed Ultralytics implementation. Observation configs use 1024-pixel tiles, while the detector validation script specifies its inference settings separately.

### Paper results

The paper results are linked from [pipeline_summary_2026-10-03_09-33-21.json](../runs/consensus/all_fields_mixed_corruption_all_combinations/pipeline_summary_2026-10-03_09-33-21.json), which references the reports generated in the `2026-10-02_21-15-25` run directories.

The repository includes all 165 prediction files used across the 933 experimental conditions. The corresponding summary contains 8,397 condition/algorithm rows, matching the saved paper CSV for precision, recall, F1, TP, FP, and FN. Only the paper result set is retained in this release.

The released paper summary references comparison reports relative to the summary directory. Other saved artifact paths are relative to the repository root. Run the documented plotting commands from that root; no summary path edits are required. Newly generated summaries retain the current implementation's path conventions and may require path edits if moved to another checkout. The root [README](../README.md#reproduction-quick-start) has the plotting commands and [licensing information](../README.md#license-and-dataset-note) for the code, detector weights, annotations, and imagery.
