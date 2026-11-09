# Curated result evidence

Original **byte-exact** HF result files are obtained with `python scripts/fetch_artifacts.py --group reports`. Do not treat GitHub metadata mirrors as replacements for originals.

- `c7c5/` — formal freeze reports mirrored as human-readable JSON, originally held at `WaveCleanLDM/C7C5_FinalSafeFreeze/latest/`. They describe a *specific contract*, including mixed-domain negative results.
- `c5d/` — optional historical stage-final QUICK_TRAIN report. Not the final model or a verified old-system comparator.
- No local copy of the manuscript 1,200-case stress test is claimed. That raw cohort/report remains **unverified and unlocated** in the available public source tree.

The formal HF production-stability audit says **`qualified_for_production_stability=false`** and is *not* mirrored as a successful result. A link and caution remain in `docs/EVIDENCE_MATRIX.md`.

Each mirrored JSON has a source URL in its provenance ledger. Treat these as **parsed-and-reformatted metadata**, not original-file byte-identical mirrors; use the downloader for archival integrity.
