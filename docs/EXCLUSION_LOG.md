# Excluded, incomplete or nonpromoted experiments

The public HF repository contains **many historical experiments**, not a single linear series of successful releases. The following are *not* promoted into the manuscript's final experiment bundle.

| Material or stage | Reason not promoted |
|---|---|
| `WavePixelSeg_Training.ipynb` | Early K=4 segmentation prototype; outputs are from a past design and cannot establish the final C7/C8 model. |
| `WaveCompress_Training.ipynb` | References `REPLACE_WITH_YOUR_SEG_RUN`; training path unresolved and designed for an earlier 4–5D codec. |
| `WaveRenderer_Training.ipynb` | Old cLDM renderer bridge, unresolved checkpoint placeholders. |
| `Backend_Stage3_Training.ipynb` | Early condition encoder/ControlNet design and missing checkpoints. |
| `WaveE2E_Reconstruction.ipynb` | Explicit 26-byte simulation and unresolved checkpoint locations; conflicts with final 30-byte protocol. |
| `ATLAS_WDS_Full_Training_v4.ipynb` | Historical ATLAS training, not a verified final ICWDS exact-30B evaluation notebook. |
| HF `C7C4` topology side-branch and `C7C4R1` | C7C5 freeze explicitly disables deployment of these research branches. |
| HF `C7C3` direct stage | Scientifically frozen, but `qualified_for_C7C4=false`; only its baseline/protocol artifacts are indexed with the caveat. |
| HF `C5D` results | Stage-final / QUICK_TRAIN archive; old model unavailable; not the paper's ultimate final LDM benchmark. |
| HF `StressResults/ICWDS_FinalMainLDM_ProductionStress_v1` | **Formal production-stability qualification failed.** Can be cited as a safety audit but not as a successful scientific outcome. |
| Manuscript revised-gate 1200/1200 success claim | Underlying new independent qualification sample not identified; the cited retest used the same cases. It remains a manuscript claim requiring traceable raw outputs. |

**Exclusion does not mean deletion:** original HF assets remain in their source repository under their existing names. We do not overwrite or re-label them. Failed audits are documented candidly in `EVIDENCE_MATRIX.md`.

| User-supplied R7 final 2006–2009 audit | **Completed, auditable, but FAILED** `FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE`; scientifically important and indexed honestly as an adverse final qualification, not a successful model release. |
| User-supplied R6 P2A1 ProductionCandidate | Trained and evaluated but **failed** NNSI catastrophic-rate qualification; not promoted as production-ready. |
