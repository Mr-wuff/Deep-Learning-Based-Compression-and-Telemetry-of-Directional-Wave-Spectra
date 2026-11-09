# Deep Learning-Based Compression and Telemetry of Directional Wave Spectra

[![Evidence status](https://img.shields.io/badge/evidence-curated%20%7C%20partial-orange)](#reproducibility-status)
[![Packet](https://img.shields.io/badge/telemetry-30%20bytes-blue)](#system-overview)
[![Grid](https://img.shields.io/badge/spectrum-47%C3%9772-blue)](#system-overview)

Research materials for **Deep Learning-Based Compression and Telemetry of Directional Wave Spectra**, prepared for a manuscript targeting the *Journal of Atmospheric and Oceanic Technology* (JTECH).

> **Research evidence, not a certified production release.** This repository distinguishes frozen artifacts from independently verified manuscript results. A frozen model or a passing protocol audit does **not** establish scientific accuracy across every data domain. A matching 2006–2009, 1,200-case audit **has now been located in user-supplied R7 materials**. Its P2-A1 MSE results match the manuscript, but the formal final-production decision **FAILED**; the exact full notebook and raw NPZ have not yet been pushed to this GitHub repository.

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
| Manuscript 1,200-case initial stress test | **Located in user-supplied R7 archive**; 2006–2009, 35 sites, 1200 cases, matching 25.1% MSE gain/39 catastrophic flags; **final production audit failed**, and raw original bytes are not yet public on GitHub |
| Later audit notebooks | R5 qualified-pilot and R7 executed final-audit notebooks **recovered from user uploads**; download-ready bundle prepared, original notebook bytes not yet in GitHub |

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
- The R7 formal experiment matches the manuscript initial 39/1200 flagged cases but **fails** its NNSI severe-direction gate; it does not substantiate the manuscript's later 1200/1200 pass claim. A same-cohort post-adjustment retest is not a new blind evaluation.
- No actual float32 spectrum, checkpoint, packet, or result should be altered solely to improve a demonstration.
- The absence of a software license in this repository means no additional permission to reuse code is granted by this GitHub repository alone. Dataset licensing is governed by the source HF repositories and original providers.

## Citation

Manuscript: *Deep Learning-Based Compression and Telemetry of Directional Wave Spectra* (submission manuscript; publication/DOI pending). A citation metadata file can be added after author order and the publication record are finalized.

## Contact and contributions

Please use GitHub Issues for questions about artifact provenance and reproducibility. External contributions should link a dated experiment report, model/data revision, evaluation protocol, and explicit pass/fail gate outcome.

## Added stress-test provenance (November 2026)

The author supplied the original `r1.zip`–`r7.zip` sequence, including a full **2006–2009, 35-station, 1,200-case** audit and executed Notebook. The P2-A1 candidate's Direct-vs-model MSE improves by **25.1248%**; its catastrophic flags total 39 (11 MEM-II / 10 NNSI / 18 Fourier). However, the preregistered `tail::nnsi_dp_relative` gate failed and `FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE` is the official decision. See [stress-test provenance audit](docs/STRESS_TEST_R1_R7_AUDIT.md). This finding is **not** a successful production release. The original R7 binary results and code are available in a curated author-download bundle, but have **not yet been pushed into this GitHub tree**.
