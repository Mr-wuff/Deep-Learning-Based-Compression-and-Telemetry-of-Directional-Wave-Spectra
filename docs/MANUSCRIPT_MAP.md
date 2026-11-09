# Manuscript section → experimental evidence

This file maps sections in the JTECH-formatted English manuscript to the exact source-artifact requirements. **Missing** means the item was not found/qualified in the accessible archive, not that the experiment necessarily was never run.

| Section | Experimental question | Required archival material | Currently verifiable |
|---|---|---|---|
| §2.1 | Wave-spectrum data provenance | dataset revision(s), station IDs, UTC splits, quality filters, schema, inversion method | HF datasets exist; cohort manifest missing |
| §2.2 | Wave-system segmentation model | final model source, checkpoint hash, inference contract, training configuration | HF `CNN.py`, C7C5 frozen hash references |
| §2.2 | Adaptive 30-byte semantic codec | bit layout, encoder/decoder, CRC test, weights | C7C3 and C7C5 release records; full code path mapping pending |
| §2.2–2.3 | Conditional LDM and quality gate | final checkpoints, conditioner, generator sampling settings, model hashes | C7C5 package and frozen hash audit; later stability qualification failed |
| §3.1 | Synthetic partitioning | synthetic generator + labels + all baselines + per-sample scores | manuscript figure only |
| §3.2 | Inverse-Fourier artifact suppression | original simulation + Fourier/MEM/NNSI baselines + conservation audit | manuscript figure only |
| §3.3 | E2E physical plots & classical model results | exact benchmark cases, scripts, per-case CSV, evaluation metrics | manuscript figures + C5D older stage results not equivalent |
| §3.4 | Historical 2006–2009, 35-site / 1200-case stress | cohort list, training exclusion proof, 3 seeds, packet+CRC, results before/after gate, confidence intervals | R7 source archive located in author uploads; metrics and 1,200-case spectra verified; **FAILED production gate**, original bytes not yet publicly deposited |

**Do not replace** the final 1200-case audit with C7C5's 1824-case freeze or the later 4499/5406-case failed production audit: these are different cohorts and model contracts.
