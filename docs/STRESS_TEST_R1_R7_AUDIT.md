# ICWDS stress testing: forensic audit of original R1–R7 packages

This document records the content and status of seven research ZIP packages provided by the author. **No failed production qualification is presented as an approved release.**

## What the paper's 1,200-case result corresponds to

- R7 original Notebook: `ICWDS_FINAL_FIX1_函数别名冲突修复_最终生产压力测试与冻结.ipynb`.
- R7 full result: `ICWDS_FINAL_ProductionStressFreeze_FULL.zip`.
- R7 formal decision: `FINAL_正式判定.json` → **`FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE`**; `selected_model: null`.
- Evaluated years: **2006–2009**; **35 stations**; **1,200 spectra** (MEM-II 480; NNSI 240; direct Fourier 480).
- The P2-A1 method has `direct_mse=0.013137457967277312` and `model_mse=0.009836698482474022`, corresponding to **25.1248%** improvement of mean MSE. The reported bootstrap gain is **25.12798%**, with **95% CI [23.12996%, 26.93848%]** and 1,000 repetitions.
- The P2-A1 catastrophic-flag counts are **11 MEM-II, 10 NNSI, 18 Fourier**, totaling **39/1,200 = 3.25%**, so the complement is **1,161/1,200 = 96.75%**. This aligns numerically with the paper's initial reported acceptance counts, but catastrophic flags alone are **not proof of the complete manuscript composite usability gate**.
- Recalculation from original `FINAL_selected_outputs.npz`, using `float64` on the stored `float16` arrays with shapes `(1200,47,72)`, independently reproduces Direct MSE **0.0131374579673**, P2-A1 candidate MSE **0.0098366997424**, and **25.124786%** relative gain (minor difference due to stored precision).

## Why R7 was not qualified

- **P2-A1** failed its preregistered NNSI severe peak-direction excess gate: **0.0125 > 0.0100** (1.25 vs 1.0 percentage points), despite improvements in average MSE.
- **FIX2** fallback also failed this gate: **0.016667 > 0.0100**.
- Exact-30B and CRC requirements passed on the 1,200 packets, but engineering integrity alone cannot establish the physical reliability of the reconstructions.
- The archive reports **no selected production model**. Therefore the R7 tests do **not** verify a successful final production freeze.
- The stated **1200/1200 post-gate 100% quality pass** in the manuscript is **not present** as a passing final qualification in R1–R7. A separate report and matched source revision are still needed.

## Chronology and selection

| Round | Original material | Purpose | Audited verdict | Final manuscript evidence? |
|---|---|---|---|---|
| R1 | NDBC diagnostic dataset + Notebook | 2024–2025 reference sampling, 6,000 total | Dataset created; **different period** from final R7 | Not the 2006–2009 final cohort |
| R2 | P1 frozen-main-LDM production stress | 4,499 primary / 5,406 extended | `qualified_for_production_stability=false` | Safety failure, do not promote |
| R3 | P1R1 forensic safety router | Learn a safe selector | `DIRECT_ONLY` selected, no useful LDM selection | Diagnosis only |
| R4 | P2A0 multi-morphology pilot | Preliminary adaptation | Follow-up qualification incomplete/failed | Prototype only |
| R5 | P2A0R1_FIX2 twin-gradient pilot | Tail-safe qualification | **Qualified to proceed P2A1**, **pilot not production** | Validated intermediate parent |
| R6 | P2A1 candidate | 2010–2013 qualification | **Not qualified**; NNSI catastrophic-rate gate failed | Needed to interpret R7 candidate, not final production |
| R7 | Final independent 2006–2009 stress | Held-out production freeze audit | **FAILED_NO_FREEZE** | Exactly matching initial paper MSE/cat numbers, not a successful release |

## Public availability status

The original R7 Notebook, 15 MB `FULL.zip`, report JSON/CSV and the R5 qualified pilot were recovered from the author uploads and placed in a **downloadable curated archive in the conversation**. They have **not yet been pushed into this GitHub repository** because the GitHub connector in this session cannot transfer local binary archives into GitHub.

The repository includes this evidence audit, an expected JSON report, and a local verification script to be run after the original archived ZIP is publicly deposited. Please do not claim those raw materials are in GitHub before their actual upload and SHA-256 verification.

Source model checkpoint paths:
- R5 FIX2 pilot: `wuff-mann/Wave_compress_system`, `WaveCleanLDM/P2A0R1_FIX2_TwinGradientSafeNNSITail/latest/PILOT_PACKAGE.pt`.
- R6 P2-A1 candidate (not qualified): `wuff-mann/Wave_compress_system`, `WaveCleanLDM/P2A1_FinalMultiMorphologyProductionCandidate/latest/CANDIDATE_PACKAGE.pt`.

## Suggested manuscript action

Preserve the reported mean MSE and engineering packet audit **as observations**, but describe the failed NNSI Dp-severe gate, distinguish pilot from production, and avoid using the 1200/1200 claim without a traceable separate final report. This is a reporting qualification, not a suggestion to alter the measured data.
