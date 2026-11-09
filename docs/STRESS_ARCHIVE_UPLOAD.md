# Publish the ICWDS R5–R7 stress-test archive

## Recommended: GitHub Releases (no Git/Git LFS needed)

The archival package is approximately 15 MB. GitHub Releases can host the original ZIP as a **release asset**, leaving the Git commit history lightweight and avoiding Git LFS entirely. The user-uploaded FIX3 bundle contains a PowerShell 5.1 script named `scripts/Publish_Stress_Archive_NoGit.ps1`.

1. Verify the local extracted archive's 32 original SHA-256 entries with the FIX3 script.
2. Run the script (without `-AutoUpload`) to build an uploadable ZIP in a temporary directory and open [New release](https://github.com/Mr-wuff/Deep-Learning-Based-Compression-and-Telemetry-of-Directional-Wave-Spectra/releases/new).
3. Create tag `icwds-r5-r7-stress-2006-2009-audit-v1`.
4. Use title **ICWDS R5–R7 stress test (2006–2009) — FAILED production qualification**.
5. Attach the generated ZIP as a Release asset, explicitly noting the official decision **FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE**, and publish.
6. Verify the public Release download and SHA-256; only then update the AMS Data Availability Statement with the published URL.

For optional automatic API upload, run the FIX3 script with `-AutoUpload` and use a fine-grained personal access token limited to this repository and `Contents: Read and write`. Never paste the token in a chat, and do not commit it to Git. The script prompts for the token in a hidden input.

**The Release is not present until the asset actually uploads and is published.** The original data should never be relabeled as a successful production freeze.

## Git + Git LFS (alternative)

The source files for this evidence were provided as seven ZIP uploads to the assistant. The corresponding full R7 raw experiment materials are **not yet on GitHub**. To finish public deposit:

1. Obtain `ICWDS_1200Case_StressTest_R5_R7_Curated_Archive.zip` from the original conversation.
2. Validate the archive: `python scripts/verify_r7_stress.py <path-to-archive.zip>` (requires NumPy).
3. Extract the ZIP into `experiments/stress_2006_2009/` of a working clone. Keep `docs/source_archive_manifest.json` and original Notebook bytes.
4. Push via Git or `gh`. ZIP files in this repository are configured for Git LFS; install and initialize Git LFS before tracking `*.zip`, or publish the whole archival ZIP as a GitHub Release asset and reference its permanent link and checksum.
5. Confirm the resulting public URL, commit SHA, file SHA-256, all 1,200-case arrays and notebook availability before editing the AMS Data Availability Statement.

Do not promote `FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE` to a successful release. GitHub CI checks only manifest syntax and may not establish scientific accuracy.
