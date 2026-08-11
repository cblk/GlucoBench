#!/usr/bin/env python3
"""Regression checks for CCAS and the frozen nine-dimensional formula."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "output" / "clinical_continuum_results.json"
FEATURES = ROOT / "output" / "clinical_continuum_features.csv"


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def component(value, lower, upper):
    return sigmoid(4.0 * (value - lower) / (upper - lower))


def max_mean(values):
    return 0.6 * np.max(values, axis=1) + 0.4 * np.mean(values, axis=1)


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    frame = pd.read_csv(FEATURES)
    formula = result["primary_formula"]

    assert result["protocol"]["primary_model"] == "full9 fixed before Hall"
    assert set(formula["features"]) == {
        "hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "anchor_level",
        "volume", "lyapunov", "det", "entr",
    }
    assert not (set(formula["features"]) & {"a1c", "fpg", "ogtt", "SSPG", "insulin", "homa_ir"})

    predicted = np.full(len(frame), 100.0 * formula["intercept"], dtype=float)
    for feature, weight in formula["weights"].items():
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if formula["transform"][feature].startswith("log1p"):
            values = np.log1p(np.maximum(values, 0.0))
        center = formula["standardization"][feature]["center"]
        scale = formula["standardization"][feature]["scale"]
        values = np.where(np.isfinite(values), values, center)
        predicted += 100.0 * weight * (values - center) / scale
    observed = frame["predicted_ccas_full9"].to_numpy(float)
    assert np.max(np.abs(predicted - observed)) < 1e-10

    core_rows = frame[["a1c", "fpg", "ccas_core"]].notna().all(axis=1)
    a1c = frame.loc[core_rows, "a1c"].to_numpy(float)
    fpg = frame.loc[core_rows, "fpg"].to_numpy(float)
    reconstructed_core = 100.0 * max_mean(np.column_stack([
        component(a1c, 5.7, 6.5), component(fpg, 100.0, 126.0)
    ]))
    assert np.max(np.abs(reconstructed_core - frame.loc[core_rows, "ccas_core"])) < 1e-10

    hall = frame[frame["analysis_cohort"] == "hall"].copy()
    full_rows = hall[["a1c", "fpg", "ogtt", "SSPG", "ccas_full"]].notna().all(axis=1)
    h = hall.loc[full_rows]
    glycemic = max_mean(np.column_stack([
        component(h["a1c"], 5.7, 6.5),
        component(h["fpg"], 100.0, 126.0),
        component(h["ogtt"], 140.0, 200.0),
    ]))
    ir = sigmoid((h["SSPG"].to_numpy(float) - 150.0) / 25.0)
    reconstructed_full = 100.0 * max_mean(np.column_stack([glycemic, ir]))
    assert len(h) == result["data"]["hall_full_complete"] == 41
    assert np.max(np.abs(reconstructed_full - h["ccas_full"])) < 1e-10

    near_example = 100.0 * max_mean(np.array([[
        component(np.array([5.5]), 5.7, 6.5)[0],
        component(np.array([95.0]), 100.0, 126.0)[0],
    ]]))[0]
    assert 30.0 < near_example < 32.0

    assert result["deployment_eligible"] is False
    assert result["html_decision"].startswith("retain index.html")
    assert any(value is False for value in result["deployment_gates"].values())
    print("clinical continuum artifact checks: PASS")


if __name__ == "__main__":
    main()
