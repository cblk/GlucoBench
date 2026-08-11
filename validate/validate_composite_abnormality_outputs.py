#!/usr/bin/env python3
"""Regression checks for the composite abnormality experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "output" / "composite_abnormality_results.json"
FEATURES = ROOT / "output" / "composite_abnormality_features.csv"


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    frame = pd.read_csv(FEATURES)

    assert result["protocol"]["discovery"] == ["CGMacros fasting insulin", "Colas T2DM"]
    assert result["protocol"]["untouched_external_validation"] == [
        "Hall diagnosis-normal SSPG", "Hall diagnosis"
    ]
    assert set(result["protocol"]["model_families"]["full9"]) == {
        "hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "anchor_level",
        "volume", "lyapunov", "det", "entr",
    }
    assert set(result["frozen_family_fits"]["full9"]["weights"]) == set(
        result["protocol"]["model_families"]["full9"]
    )

    formula = result["formula"]
    reconstructed = np.zeros(len(frame), dtype=float)
    for feature, weight in formula["weights"].items():
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if formula["feature_transform"][feature].startswith("log1p"):
            values = np.log1p(np.maximum(values, 0.0))
        center = formula["standardization"][feature]["center"]
        scale = formula["standardization"][feature]["scale"]
        values = np.where(np.isfinite(values), values, center)
        reconstructed += weight * (values - center) / scale

    observed = frame["omega_g"].to_numpy(float)
    assert np.max(np.abs(reconstructed - observed)) < 1e-10
    display = frame["omega_g_0_100_display"].to_numpy(float)
    assert np.isfinite(display).all() and np.all((display > 0) & (display < 100))

    assert result["deployment_eligible"] is False
    assert result["html_decision"].startswith("retain existing index.html")
    assert any(value is False for value in result["deployment_gates"].values())
    assert result["selected_family"] in result["protocol"]["model_families"]

    forbidden = {"insulin", "SSPG", "diagnosis", "T2DM", "a1c"}
    assert not (set(formula["weights"]) & forbidden)
    print("composite abnormality artifact checks: PASS")


if __name__ == "__main__":
    main()
