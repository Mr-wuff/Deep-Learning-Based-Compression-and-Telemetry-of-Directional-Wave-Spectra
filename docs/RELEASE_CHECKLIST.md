# Release readiness checklist for the manuscript dataset and code

The GitHub repository is a research-evidence index and partial collection of confirmed source files. **It is not yet a complete, article-reproduction package.**

## Required before declaring the journal submission fully reproducible

- [ ] Dataset version, station count and training/sample count reconciled against the same immutable manifest.
- [ ] Final train/validation/test cohorts available as machine-readable station–UTC–inversion-domain lists.
- [ ] Exact final CNN segmentation training and validation Notebook matched to archived weight hashes.
- [ ] Final exact-30B VAE/codec and LDM/gate training code, run configuration and environment pinned.
- [ ] Synthetic segmentation and artifact-removal test generators and per-sample baseline CSVs released.
- [ ] Classical spectrum comparison benchmark code, all input features and per-sample errors released.
- [ ] The **1,200-case** R7 original notebook, full report and NPZ have now been **found in author uploads**; publish these original bytes to a stable GitHub release or HF revision and reconcile the failed final qualification before describing availability.
- [ ] The gate-revised **same-sample retest** clearly labeled as *post-adjustment within the same cohort*; a new independent test collected if making a fresh generalization claim.
- [ ] C7C5 mixed negative gains and later formal failed production-stability audit reconciled with any manuscript statements of overall superiority or operational readiness.
- [ ] The author-submitted AMS Data Availability Statement names the specific immutable source revisions and exact experimental artifacts, with no unresolved brackets.

## Current checks

The repository's `Manifest checks` GitHub Action validates that artifact paths are unique and code tools compile. This CI passing **does not validate scientific claims**.

See [EVIDENCE_MATRIX.md](EVIDENCE_MATRIX.md), [EXCLUSION_LOG.md](EXCLUSION_LOG.md) and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
