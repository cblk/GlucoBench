#!/usr/bin/env python3
"""Regression checks for the external five-dimensional validation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def main():
    results = json.loads((OUTPUT / "external_base5_validation_results.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "external_base5_export_audit.json").read_text(encoding="utf-8"))
    model = json.loads((OUTPUT / "stability_base5_results.json").read_text(encoding="utf-8"))["final_formula"]
    frame = pd.read_csv(OUTPUT / "external_base5_validation_scores.csv")
    eligible = frame[frame["eligible"].astype(bool)].copy()

    assert audit["shanghai_t2dm"]["mapped_cgm_records"] == 109
    assert audit["shanghai_t2dm"]["xls_records"] == 89
    assert audit["shanghai_t2dm"]["xlsx_records"] == 20
    assert audit["stanford"]["direct_phenotype_overlap_n"] == 5
    assert audit["kobe"]["paper_example_checks"]["participant_13"]["clamp_di"] == 65.68
    assert audit["kobe"]["paper_example_checks"]["participant_46"]["retained_row_one_based"] == 43
    assert audit["kobe"]["paper_example_checks"]["participant_46"]["clamp_di"] == 11.53

    forbidden = {"a1c", "fpg", "ogtt", "sspg", "insulin", "diagnosis", "ccas", "clamp", "oral_di"}
    assert not any(any(token in feature.lower() for token in forbidden) for feature in model["features"])

    errors = []
    for _, row in eligible.iterrows():
        logit = float(model["intercept"])
        for feature in model["features"]:
            value = float(row[feature])
            if model["transform"][feature].startswith("log1p"):
                value = math.log1p(max(value, 0.0))
            reference = model["standardization"][feature]
            logit += float(model["weights"][feature]) * (value - float(reference["center"])) / float(reference["scale"])
        expected = 100.0 * sigmoid(logit)
        errors.append(abs(expected - float(row["frozen_base5"])))
    assert max(errors) < 1e-10
    assert eligible["frozen_base5"].between(0, 100, inclusive="both").all()

    shanghai = eligible[eligible["cohort"].eq("shanghai_t2dm_w48")].dropna(
        subset=["a1c_ifcc_mmol_mol", "a1c_percent"]
    )
    converted_a1c = 0.09148 * shanghai["a1c_ifcc_mmol_mol"].to_numpy(float) + 2.152
    assert np.max(np.abs(converted_a1c - shanghai["a1c_percent"].to_numpy(float))) < 1e-12

    v87_errors = []
    for _, row in eligible.iterrows():
        night = float(row["v87NightMean"])
        if row["cohort"].startswith("kobe_"):
            z = (
                -6.772379 + 0.701805 * night
                + 0.008823 * float(row["workIntegral"])
                + 0.103079 * float(row["ascendFriction"])
            )
        elif row["cohort"].startswith("shanghai_"):
            z = (
                -2.103627 + 0.404453 * night
                - 0.050432 * float(row["nightFriction"])
                - 0.102868 * float(row["ascendFriction"])
            )
        else:
            z = 1.064314 * night - 6.746364
        v87_errors.append(abs(sigmoid(z) - float(row["v87Risk"])))
    assert max(v87_errors) < 1e-12

    assert results["candidate_freeze_eligible"] is False
    assert results["deployment_eligible"] is False
    assert "unchanged" in results["html_decision"]
    assert results["protocol"]["no_refitting_on_external_cohorts"] is True

    summary = {
        "eligible_rows": int(len(eligible)),
        "maximum_base5_formula_error": max(errors),
        "maximum_v87_formula_error": max(v87_errors),
        "base5_bounds_ok": True,
        "ifcc_ngsp_conversion_ok": True,
        "endpoint_leakage_check": True,
        "candidate_freeze_eligible": False,
        "deployment_eligible": False,
    }
    (OUTPUT / "external_base5_validation_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
