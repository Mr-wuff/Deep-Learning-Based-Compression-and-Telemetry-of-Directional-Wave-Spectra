# Deep Learning-Based Compression and Telemetry of Directional Wave Spectra

[![Evidence status](https://img.shields.io/badge/evidence-curated%20%7C%20partial-orange)](#reproducibility-status)
[![Packet](https://img.shields.io/badge/telemetry-30%20bytes-blue)](#system-overview)
[![Grid](https://img.shields.io/badge/spectrum-47%C3%9772-blue)](#system-overview)

Research materials for **Deep Learning-Based Compression and Telemetry of Directional Wave Spectra**, prepared for a manuscript targeting the *Journal of Atmospheric and Oceanic Technology* (JTECH).

> **Research evidence, not a certified production release.** This repository distinguishes frozen artifacts from independently verified manuscript results. A frozen model or a passing protocol audit does **not** establish scientific accuracy across every data domain. The complete 1,200-case manuscript stress-test bundle has **not yet been located in the available archive** and must not be represented as independently reproducible here.

## System overview

The proposed ICWDS workflow compresses a directional wave spectrum on a **47-frequency × 72-direction** grid into an **exactly 30-byte** packet, and reconstructs a spectrum on shore:

```text
Directional spectrum (47 × 72, float32)
       │
       ▼
Buoy-side wave-system partitioning (CNN)
       │
       ▼
Per-system physical semantics + morphology + topology + residual coding
       │
       ▼
Adaptive quantization → 28-byte payload + 2-byte CRC-16
       │
       ▼
BeiDou short-message telemetry (30 bytes)
       │
       ▼
Packet decoding → deterministic Direct prior
       │
       ▼
Conditional residual LDM + reliability gate / Direct fallback
       │
       ▼
Reconstructed directional spectrum and derived wave parameters
```

The nominal matrix-to-packet compression ratio is **13,536 / 30 = 451.2:1** (uncompressed float32 matrix vs. the 30-byte packet). This is a **storage-budget ratio**, not a claim of lossless information recovery.

## Source-of-truth archives

| Role | Public archive | Interpretation |
|---|---|---|
| Updated directional-spectrum dataset | [ATLAS-WDS-v4](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v4) | Public data repository; inspect its actual splits and version before reporting sample counts |
| Spectrum + wind and Fourier/MEM data | [ATLAS-WDS-v2](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v2) | Historical dataset with station/time/source metadata |
| Early directional-spectrum dataset | [ATLAS-WDS](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS) | Earlier dataset; **not interchangeable** with v4 |
| Models, evaluation archives and checkpoints | [Wave_compress_system](https://huggingface.co/wuff-mann/Wave_compress_system) | Author-maintained, multi-stage HF model repository |

Do not transfer GB-scale datasets or model weights into Git history. Original weights, archived ZIPs and experiment files remain in Hugging Face. **The local manifest and downloader record their canonical paths**, allowing lossless retrieval. Small machine-readable reports may be mirrored under `results/` where explicitly verified.

## Repository contents

- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — documented retrieval and verification protocol.
- [docs/EVIDENCE_MATRIX.md](docs/EVIDENCE_MATRIX.md) — manuscript-to-artifact provenance, known gaps, and scientific qualifications.
- [docs/EXCLUSION_LOG.md](docs/EXCLUSION_LOG.md) — why failed and obsolete experiments were not promoted.
- [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) — proposed AMS data-availability wording and open issues.
- [data/artifact_manifest.json](data/artifact_manifest.json) — explicit source repositories, paths, statuses, and selection.
- [scripts/fetch_artifacts.py](scripts/fetch_artifacts.py) — verified-source downloader (SHA-256 manifest output).
- [scripts/validate_repository.py](scripts/validate_repository.py) — local manifest and archive consistency checks.
- [results/README.md](results/README.md) — interpretation of accepted evidence.

## Reproducibility status

| Component | Status |
|---|---|
| Public training data repositories | **Located**; HF links included |
| Model repository and C7-C5 frozen package | **Located**; official freeze JSON reports `qualified_for_production=true` for its *specified* engineering contract |
| C5-D benchmark archive | **Located**, but `QUICK_TRAIN` and `old_system_available=false`; stage-specific, not an independent final-paper proof |
| C7-C5 cross-domain scientific gain | **Mixed**; overall gain reported negative, so not a universal superiority claim |
| Later production stability audit | **Failed qualification**; expressly excluded from positive-results directory |
| Manuscript 1,200-case independent stress test and revised-gate retest | **Not yet traced to a matching raw artifact bundle**; manuscript figures alone are insufficient for independent reproduction |
| Full, exact final training notebooks | **Incomplete** in available archive; early notebooks with placeholder paths are not presented as final scripts |

See [EVIDENCE_MATRIX](docs/EVIDENCE_MATRIX.md) before using numbers in a paper. Paper claims should cite the **specific model revision, evaluation set, code, raw outputs, and scoring rule** that produced them.

## Quick start

```bash
git clone https://github.com/Mr-wuff/Deep-Learning-Based-Compression-and-Telemetry-of-Directional-Wave-Spectra.git
cd Deep-Learning-Based-Compression-and-Telemetry-of-Directional-Wave-Spectra
python scripts/validate_repository.py
python scripts/fetch_artifacts.py --list
python scripts/fetch_artifacts.py --group reports --output downloads
```

Downloading large model packages is opt-in (`--group models`). Raw assets remain associated with their HF paths and recorded revision identifiers.

## Scientific and research-integrity notes

- Public NDBC/CDIP observations and **directional spectra reconstructed from directional moments** are not interchangeable with independent direct measurements of the true 2D spectrum.
- To avoid train/test contamination, record station IDs, UTC times, reconstruction domain (MEM-II, NNSI, Fourier), deduplication keys and exact dataset revisions.
- The same 1,200 samples reportedly informed a quality-gate adjustment and its subsequent 100% retest. This **cannot be described as a new independent blind test** without additional held-out data.
- No actual float32 spectrum, checkpoint, packet, or result should be altered solely to improve a demonstration.
- The absence of a software license in this repository means no additional permission to reuse code is granted by this GitHub repository alone. Dataset licensing is governed by the source HF repositories and original providers.

## Citation

Manuscript: *Deep Learning-Based Compression and Telemetry of Directional Wave Spectra* (submission manuscript; publication/DOI pending). A citation metadata file can be added after author order and the publication record are finalized.

## Contact and contributions

Please use GitHub Issues for questions about artifact provenance and reproducibility. External contributions should link a dated experiment report, model/data revision, evaluation protocol, and explicit pass/fail gate outcome.
