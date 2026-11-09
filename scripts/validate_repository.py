#!/usr/bin/env python3
"""Offline checks for ICWDS artifact manifest and evidence categorization."""
from __future__ import annotations
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "artifact_manifest.json"
ALLOWED = {
    "frozen_contract_qualified", "frozen_with_unpassed_followup_gate",
    "stage_final_not_paper_final", "public_reference_source",
}


def main() -> int:
    m = json.loads(FILE.read_text(encoding="utf-8"))
    failures = []
    keys = set()
    for a in m["artifacts"]:
        key = (a["repo_id"], a["path"])
        if key in keys:
            failures.append("duplicate artifact: " + str(key))
        keys.add(key)
        if a.get("status") not in ALLOWED:
            failures.append("unapproved result status: " + str(key))
        if a.get("group") not in {"models", "reports", "code"}:
            failures.append("unexpected download group: " + str(key))
        if "REPLACE_" in a.get("path", ""):
            failures.append("unresolved checkpoint path: " + str(key))
        if a.get("status") == "frozen_contract_qualified" and "C7C4" in a["path"]:
            failures.append("disabled topology side branch mislabeled as qualified")
    if not m.get("observational_datasets"):
        failures.append("dataset inventory missing")
    for path in ["README.md", "docs/EVIDENCE_MATRIX.md", "docs/DATA_AVAILABILITY.md"]:
        if not (ROOT / path).is_file():
            failures.append("missing: " + path)
    if failures:
        for e in failures:
            print("FAIL:", e)
        return 1
    print(f"PASS: {len(m['artifacts'])} unique manifest entries; "
          f"{len(m['observational_datasets'])} dataset sources")
    print("NOTE: Does not validate paper numerical claims or raw HF checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
