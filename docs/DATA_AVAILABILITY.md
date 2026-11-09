# Data Availability Statement — JTECH manuscript

## Proposed wording (accurate about public repositories; requires completion)

> Processed directional wave spectrum datasets associated with the ICWDS research program are available through Hugging Face at [ATLAS-WDS](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS), [ATLAS-WDS-v2](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v2), and [ATLAS-WDS-v4](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v4). Archived model files and selected experiment reports are hosted at [Wave_compress_system](https://huggingface.co/wuff-mann/Wave_compress_system). A source-artifact inventory, retrieval procedure, and information on validation status are maintained at [this GitHub repository](https://github.com/Mr-wuff/Deep-Learning-Based-Compression-and-Telemetry-of-Directional-Wave-Spectra). **[AUTHOR TO COMPLETE: identify the immutable dataset and model revisions, exact cohort identifiers, and the final training and independent stress-test scripts/results supporting the numerical claims in this manuscript.]**

Following inspection of the complete user-supplied R7 final stress audit, the **original 1,200-case scripts and raw spectral outputs have been located locally but are not yet publicly deposited into this GitHub repository**. The audit itself **failed its production qualification**. This text remains a **working draft only**, not a submission-ready final data availability statement. The bracketed clause must be replaced by actual repository locations before submission. Do not claim all article results are accessible until the matching raw source material has been deposited and verified.

## Data provenance requirements before a definitive statement

- Identify which dataset version (ATLAS-WDS, v2, v4) produced each training phase; record HF commit SHA and splits.
- Deposit a machine-readable *train/validation/test* manifest keyed by station, UTC timestamp and spectral inversion method.
- Resolve manuscript station/sample count conflicts (117/110 sites; five million/1.77 million observations).
- Deposit final code, configuration and original R7 full archive for §3.4. Its 39 catastrophic flags and P2-A1 MSE +25.1% match the manuscript's initial counts, but **the formal R7 audit failed**, and it does not substantiate a 1200/1200 post-gate success. Obtain separate supporting provenance if that later claim is retained. Resolve missing §3.1–§3.3 final scripts as well.
- If buoy observations are governed by upstream NDBC or CDIP conditions, retain required source attributions.
- Identify whether any GNSS offshore field observations are already public; if not, state restrictions and the availability route without implying that unpublished data have been deposited.
- Document and reconcile the formal HF production-stability failure and mixed C7C5 model performance with manuscript claims.

## Model and code availability

Hugging Face contains final and intermediate model packages; **not all paths are promoted**. The GitHub manifest references frozen C7C5 material and explicitly excludes non-qualified or older experimental paths. Only independently traceable artifacts should appear in the final statement.

## AMS note

AMS requires a data availability statement. This repository is not a substitute for depositing missing reproducibility artifacts, and an annotated draft statement is not itself sufficient for final submission.
