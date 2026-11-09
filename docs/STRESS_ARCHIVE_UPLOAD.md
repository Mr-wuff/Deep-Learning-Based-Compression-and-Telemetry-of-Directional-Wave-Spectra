# Publishing the completed author-supplied R7 experiment

The source files for this evidence were provided as seven ZIP uploads to the assistant. The corresponding full R7 raw experiment materials are **not yet on GitHub**. To finish public deposit:

1. Obtain `ICWDS_1200Case_StressTest_R5_R7_Curated_Archive.zip` from the original conversation.
2. Validate the archive: `python scripts/verify_r7_stress.py <path-to-archive.zip>` (requires NumPy).
3. Extract the ZIP into `experiments/stress_2006_2009/` of a working clone. Keep `docs/source_archive_manifest.json` and original Notebook bytes.
4. Push via Git or `gh`. ZIP files in this repository are configured for Git LFS; install and initialize Git LFS before tracking `*.zip`, or publish the whole archival ZIP as a GitHub Release asset and reference its permanent link and checksum.
5. Confirm the resulting public URL, commit SHA, file SHA-256, all 1,200-case arrays and notebook availability before editing the AMS Data Availability Statement.

Do not promote `FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE` to a successful release. GitHub CI checks only manifest syntax and may not establish scientific accuracy.
