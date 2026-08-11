#!/usr/bin/env python3
"""Regression checks for the bounded five-dimensional research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "output" / "stability_base5_results.json"
FEATURES = ROOT / "output" / "stability_base5_features.csv"
BASE5 = {
    "hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "anchor_level"
}


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    frame = pd.read_csv(FEATURES)
    formula = result["final_formula"]

    assert result["protocol"]["primary_model"] == "stability_base5"
    assert set(formula["features"]) == BASE5
    assert not (set(formula["features"]) & {"a1c", "fpg", "ogtt", "SSPG", "insulin", "homa_ir"})
    assert all(value >= 0 for value in formula["weights"].values())
    assert set(frame["analysis_cohort"]) == {"cgmacros", "colas", "hall"}
    assert frame.groupby("analysis_cohort").size().to_dict() == {
        "cgmacros": 44, "colas": 191, "hall": 55,
    }

    eta = np.full(len(frame), formula["intercept"], dtype=float)
    for feature, weight in formula["weights"].items():
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if formula["transform"][feature].startswith("log1p"):
            values = np.log1p(np.maximum(values, 0.0))
        center = formula["standardization"][feature]["center"]
        scale = formula["standardization"][feature]["scale"]
        values = np.where(np.isfinite(values), values, center)
        eta += weight * (values - center) / scale
    reconstructed = 100.0 * sigmoid(eta)
    observed = frame["stability_base5_full_fit"].to_numpy(float)
    assert np.max(np.abs(reconstructed - observed)) < 1e-10
    assert np.all((observed > 0.0) & (observed < 100.0))

    for column in [name for name in frame.columns if name.startswith("loco_")]:
        values = frame[column].to_numpy(float)
        assert np.all(np.isfinite(values))
        assert np.all((values > 0.0) & (values < 100.0))

    default_sensitivity = result["ccas_core_sensitivity"]["slope_4_max_0.60"]
    primary = result["loco_evaluation"]["stability_base5"]
    assert abs(default_sensitivity["macro_mean_spearman"] - primary["macro_mean_spearman"]) < 1e-12
    assert result["data"]["pair_groups"]["sensor_cgmacros_w48"] == 44
    assert result["data"]["pair_groups"]["time_colas_default"] == 191
    assert result["hall_exploratory"]["hidden_abnormal"]["n"] == 30
    assert result["candidate_freeze_eligible"] is False
    assert result["deployment_eligible"] is False
    assert result["html_decision"].startswith("retain index.html")
    assert any(value is False for value in result["candidate_freeze_gates"].values())
    print("stability base5 artifact checks: PASS")


if __name__ == "__main__":
    main()
