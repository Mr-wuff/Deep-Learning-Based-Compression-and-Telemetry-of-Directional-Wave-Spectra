# Manuscript evidence matrix and unresolved discrepancies

Only **located public assets** are marked 'located.' Values taken from the paper are explicitly labeled as manuscript claims, not newly independently reproduced results.

| Manuscript section / claim | Located primary evidence | Status and limitation |
|---|---|---|
| §2.1 training data and station map | [ATLAS-WDS-v4](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v4) and v2 datasets | Datasets located; **117 sites / ~5 million** in prose vs **110 sites / ~1.77 million** on map requires an immutable cohort manifest and reconciliation. |
| §2.2 CNN wave-system parser | [HF `CNN.py`](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/CNN.py), [C7C3 protocol artifacts](https://huggingface.co/wuff-mann/Wave_compress_system/tree/main/WaveCleanLDM/C7C3_ProtocolV3_DirectV2/latest) | Public source and frozen Direct package located; final source-to-checkpoint mapping not fully re-audited. |
| §2.2 VAE semantics | [HF `VAE.py`](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/VAE.py) | Public reference source located; final model's training notebook and all flags not yet matched. |
| §2.2 exact 30-byte protocol | [C7C5 packet reliability report](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/WaveCleanLDM/C7C5_FinalSafeFreeze/latest/C7C5_%E6%8A%A5%E6%96%87%E5%8F%AF%E9%9D%A0%E6%80%A7%E6%91%98%E8%A6%81.json) | **Located engineering audit**; packet checks do not establish physical spectral accuracy. |
| §2.3 / §3.3 end-to-end model | [C7C5 frozen package](https://huggingface.co/wuff-mann/Wave_compress_system/tree/main/WaveCleanLDM/C7C5_FinalSafeFreeze/latest) | Formal C7C5 freeze marked `qualified_for_production=true` within its stated configuration; **overall relative MSE gain is negative** in its reported mixed-domain audit. |
| §3.1 synthetic partitioning comparison | Figure in paper, early prototypes locally | Full final evaluation code and sample-level metrics **not located**; cannot claim raw scientific audit was deposited. |
| §3.2 synthetic artifact-removal comparison | Figure in paper | Final dataset generator, benchmark labels and per-case results **not located**. |
| §3.3 physical parameter scatter plots & classical comparisons | Figures/table in paper | Final evaluation script, case manifest and per-case scores **not located**. |
| §3.4 initial blind stress test (1,161/1,200; 96.75%; 25.1% MSE improvement; 95% CI 23.1–26.9%) | Manuscript only | **LOCATED in the author's uploaded R7 archive**, which contains an executed Notebook, final decision, 1,200 reference/Direct/candidate spectrum arrays (47×72), and raw method summaries. Relative MSE improvement **25.1248%** and reported bootstrap CI **23.13%–26.94%** match the manuscript. However, the formal audit **FAILED** its NNSI Dp-severe gate; the original full files are not yet deposited publicly in this GitHub repo. |
| §3.4 revised gate (1,200/1,200 on same cohort) | Manuscript only | The R7 archive documents **no successful production freeze** (selected_model=null). A subsequent 1200/1200 passing report matching the manuscript's wording **was not located** among R1–R7. Same-cohort gate tuning, if used, is not a new independent blind evaluation. |
| Later production-stability qualification | [HF final production-stress audit](https://huggingface.co/wuff-mann/Wave_compress_system/blob/main/StressResults/ICWDS_FinalMainLDM_ProductionStress_v1/%E6%AD%A3%E5%BC%8F%E7%94%9F%E4%BA%A7%E7%A8%B3%E5%AE%9A%E6%80%A7%E5%88%A4%E5%AE%9A.json) | Explicitly **FAILED** (`qualified_for_production_stability=false`); excluded as positive result, but must be reconciled before strong operational claims. |

## Exact, potentially conflicting findings

1. The public **C7C5** freeze report records `audit_n=1824`, `overall_MSE_gain_vs_direct=-0.008497...`, and `REAL_ATLAS` bootstrap median approximately `-0.04142`. The package's affirmative freeze status concerns its contract, not unrestricted scientific superiority.
2. The **C7C3** release decision reports `qualified_for_C7C4=false` while also `scientific_path_frozen=true`. A frozen stage can have an unpassed follow-up gate.
3. The **C5D** report identifies `profile=QUICK_TRAIN`, `old_system_available=false`, and `best_method=raw_input`; it cannot be presented as an independent, final old-versus-new comparison.
4. The later **production-stability audit** reports catastrophic/major error rates exceeding its thresholds. Do not hide this when writing an operational reliability claim.
5. Values of 117 vs 110 stations, five million vs approximately 1.77 million observations, and inverted versus directly observed 2D spectra should be traced to dataset manifests rather than harmonized silently.

## Evidence promotion policy

An experiment goes into the verified manuscript-results directory only after its original code, exact HF revision, data cohort, complete result files and noncontradictory gate decision are matched. Failed/diagnostic or older prototype results remain only in the evidence matrix or exclusion notes, not the recommended reproduction manifest.

### New evidence from original R1–R7 stress-test packages

- R1: diagnostic NDBC First5-based dataset, **2024–2025**, 6,000 samples / 48 stations; NOT the final 2006–2009 cohort.
- R2: **production stability failed** for a C8A2R1 frozen main-LDM on 4,499 primary cases and 5,406 broader cases.
- R3: forensic safety selection returned **DIRECT_ONLY** (safe but no useful LDM selection).
- R4: P2A0 pilot did not pass its complete follow-up qualification.
- R5: P2A0R1_FIX2 pilot **qualified for P2A1** on an independent 2014–2017 holdout; **pilot_not_production=true**.
- R6: P2A1 candidate **not qualified** on 2010–2013 due NNSI catastrophic-rate gate; P2-A1 remained eligible for inspection as a candidate, not a certified model.
- R7: final 2006–2009, **35-site/1,200-case** test explicitly **FAILED_NO_FREEZE** for both P2-A1 and fallback FIX2, due NNSI Dp severe excess relative to Direct; no final model selected.

The 39 catastrophic flags in the P2-A1 candidate (11 MEM-II, 10 NNSI, 18 Fourier) numerically match the manuscript's 96.75% initial pass figure. Catastrophic-flag complement alone does not independently reproduce the manuscript's complete composite usability criteria. Sample-level 47×72 arrays confirm Direct MSE 0.01313746 and candidate MSE 0.00983670, giving **25.1248%** relative improvement. See [the R7 audit](STRESS_TEST_R1_R7_AUDIT.md).
