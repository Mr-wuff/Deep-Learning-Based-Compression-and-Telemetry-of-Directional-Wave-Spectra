# Reproducibility protocol

## Scope and evidence levels

This repository curates confirmed HF **file paths** and selected frozen-stage reports. It is **not yet a full executable reproduction** of all manuscript results. In particular, the exact 1,200-case stress-test scripts, per-case files and final-gate retest provenance were not located among the available archive items.

A stage name ending in `Final` or a `qualified_for_production` flag must be read in its *specified gate context*; it does not imply universally positive scientific performance.

## Prepare environment

Python 3.10+ is suitable for the manifest tools; the research models require the exact PyTorch/CUDA versions from the frozen training configuration when available.

```bash
python scripts/validate_repository.py
python scripts/fetch_artifacts.py --list
python scripts/fetch_artifacts.py --group reports --output downloads
python scripts/fetch_artifacts.py --group code --output downloads
# Large checkpoint download is optional:
python scripts/fetch_artifacts.py --group models --output downloads
```

The downloader retrieves **original HF bytes**, writes local SHA-256 receipts, and attempts to resolve the HF repository commit at download time. A mutable `main` reference is not an archival guarantee; cite the `resolved_revision` recorded in the local lock file.

### Baseline datasets

The research program includes [ATLAS-WDS](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS), [ATLAS-WDS-v2](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v2), and [ATLAS-WDS-v4](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v4), which have **different schemas and sizes**. The final manuscript's train/validation/test cohort must be established from an immutable dataset revision and station/time split manifest. Do not infer that all three datasets were used in the same stage.

### Scientific reproduction checklist

1. Freeze exact training data split by `(station, UTC timestamp, inversion domain)`, including duplicate detection across NDBC text/DODS.
2. Save reconstruction-domain labels: **MEM-II**, **NNSI**, **truncated Fourier**; do not call inverted spectra direct 2D measurements.
3. Freeze input frequency grid, direction convention, normalized/un-normalized energy scaling and physical-metric implementation.
4. Pin model weights, all subcomponent hashes, packet bit layout/CRC protocol, random seeds, versions and inference steps.
5. Report Direct deterministic baseline and every stochastic seed separately, including failures; retain raw per-sample residuals.
6. Compute directional errors with circular distance. Explicitly state which physical quantities are given to the baselines.
7. Evaluate revised safety gates on fresh *untouched* cases; retesting on the calibration cohort is not a new blind test.
8. Archive code commit, artifact manifest, dataset revision and raw metric files before claiming a number is independently reproducible.

## Computational safety

The HF model checkpoints use PyTorch serialization and may require pickle deserialization. **Do not load untrusted checkpoint files.** Downloaded assets are not executed by this repository's scripts.

## Original experimental notebooks

The locally available 2026 early-stage notebooks `WavePixelSeg_Training.ipynb`, `WaveCompress_Training.ipynb`, `WaveRenderer_Training.ipynb`, `Backend_Stage3_Training.ipynb`, and `WaveE2E_Reconstruction.ipynb` include obsolete model designs or unresolved `REPLACE_...` checkpoint paths. They have **not** been promoted as final training scripts. See [EXCLUSION_LOG](EXCLUSION_LOG.md).

## Published numbers

For any manuscript number, the source should provide a self-contained input cohort, source code, trained model reference, evaluation rule, output CSV, pass/fail rule, and uncertainty treatment. See [EVIDENCE_MATRIX](EVIDENCE_MATRIX.md) for current evidence coverage.
