#!/usr/bin/env python3
"""Verify the complete author-uploaded R7 stress-test archive without running the model.

Usage:
    python scripts/verify_r7_stress.py ICWDS_1200Case_StressTest_R5_R7_Curated_Archive.zip

Requires numpy. Reads nested ZIP+NPZ with allow_pickle=False, verifies mean MSE
and formal no-freeze decision, then prints the independent stress-cohort summary.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile

import numpy as np


def read_source(archive: zipfile.ZipFile):
    names = archive.namelist()
    if "artifacts/ICWDS_FINAL_ProductionStressFreeze_FULL.zip" in names:
        return io.BytesIO(archive.read("artifacts/ICWDS_FINAL_ProductionStressFreeze_FULL.zip"))
    if "ICWDS_FINAL_ProductionStressFreeze_FULL.zip" in names:
        return io.BytesIO(archive.read("ICWDS_FINAL_ProductionStressFreeze_FULL.zip"))
    if "r7.zip" in names:
        with zipfile.ZipFile(io.BytesIO(archive.read("r7.zip"))) as outer:
            return io.BytesIO(outer.read("ICWDS_FINAL_ProductionStressFreeze_FULL.zip"))
    raise ValueError("Full original R7 audit ZIP was not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", help="Path to curated archive ZIP or original r7.zip")
    opts = ap.parse_args()
    with zipfile.ZipFile(opts.archive) as z:
        if "ICWDS_FINAL_ProductionStressFreeze_FULL.zip" in z.namelist():
            full_bytes = io.BytesIO(z.read("ICWDS_FINAL_ProductionStressFreeze_FULL.zip"))
        else:
            full_bytes = read_source(z)
    with zipfile.ZipFile(full_bytes) as full:
        final = json.loads(full.read("FINAL_正式判定.json").decode("utf-8-sig"))
        with np.load(io.BytesIO(full.read("FINAL_selected_outputs.npz")), allow_pickle=False) as npz:
            ref = npz["reference"].astype(np.float64)
            direct = npz["direct"].astype(np.float64)
            model = npz["model"].astype(np.float64)
    assert ref.shape == direct.shape == model.shape == (1200, 47, 72)
    baseline = float(np.mean((ref - direct) ** 2))
    candidate = float(np.mean((ref - model) ** 2))
    gain = 1.0 - candidate / baseline
    report = final["audits"]["P2-A1"]
    assert abs(baseline - report["overall"]["direct_mse"]) < 1e-7
    assert abs(candidate - report["overall"]["model_mse"]) < 5e-6
    assert abs(gain - report["bootstrap"]["mean"]) < 5e-4
    counts = {x["method"]: round(x["model_cat"] * x["n"]) for x in report["methods"]}
    assert counts == {"mem2": 11, "nnsi": 10, "fourier": 18}
    assert final["formal_decision"] == "FINAL_PRODUCTION_AUDIT_FAILED_NO_FREEZE"
    assert final["selected_model"] is None
    assert report["gates"]["tail::nnsi_dp_relative"]["pass"] is False
    print(f"R7 verified: N={len(ref)}, 47×72, {final['final_years']}")
    print(f"Catastrophic flags: {counts} (sum={sum(counts.values())})")
    print(f"Direct MSE: {baseline:.9f}; candidate MSE: {candidate:.9f}; mean improvement: {gain:.4%}")
    print(f"Bootstrap reported CI: {report['bootstrap']['ci95']}")
    print("FORMAL VERDICT:", final["formal_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
