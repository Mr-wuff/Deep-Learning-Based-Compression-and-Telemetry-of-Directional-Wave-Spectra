# Contributing to ICWDS reproducibility

Thank you for helping make the published directional-wave-spectrum compression experiments independently verifiable.

## Evidence acceptance rules

Pull requests adding an experiment must contain or link:

1. The exact **paper section, figure or table** and the scientific claim under test.
2. Original, **complete experiment code**, frozen configuration, environment and reproducible command.
3. Immutable HF repository **revision SHA**, checkpoint SHA-256 and packet/protocol version.
4. Data cohort manifest identifying station, observation UTC timestamp, source and spectral inversion method.
5. Per-sample outputs, not solely an average or a figure; include **all failures** and run seeds.
6. An explicit method of deriving report metrics and uncertainty intervals from raw outputs.
7. A machine-readable and truthful stage decision with precise qualification criteria.

Do not label a checkpoint or script final solely because its filename includes `FINAL`; review its associated gate report. Do not replace an unqualified experiment with a passing training-loss plot. Excluded experiments stay at their public source HF URLs; never rewrite their original results.

## Running checks

```bash
python scripts/validate_repository.py
python -m py_compile scripts/*.py
```

For large datasets or weights, use a HF source reference in `data/artifact_manifest.json`, instead of committing binary checkpoints directly. The downloader will record SHA-256 receipts.

## Pull request checklist

- [ ] The source is genuinely a final, fully completed experiment for the specified manuscript result.
- [ ] Data/test split and model/protocol versions are frozen.
- [ ] Statistical metrics match the original raw report.
- [ ] No tokens, credentials, personal paths or private data are included.
- [ ] All provenance links resolve and the format validation workflow passes.
- [ ] No false claims of independent validation or successful production qualification.

## Citation, authorship and license

Submit only material you have the right to share, retaining all third-party notices. No general code reuse license has been chosen for this repository; source HF datasets may have independent license conditions. Manuscript author order and bibliographic publication information are not finalized here.
