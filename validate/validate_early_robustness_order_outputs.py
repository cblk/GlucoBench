#!/usr/bin/env python3
"""Regression checks for the early sample-robustness research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "output" / "early_robustness_order_results.json"
ROWS = ROOT / "output" / "early_robustness_order_rows.csv"


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = pd.read_csv(ROWS)
    prefix = rows[rows["analysis_type"] == "night_prefix"].copy()
    pairs = rows[rows["analysis_type"] == "window_24_to_48"].copy()

    assert len(prefix) >= 600
    assert len(pairs) >= 500
    assert set(prefix["cohort"]) == {"hall", "colas"}
    assert set(prefix[prefix["cohort"] == "hall"]["prefix_nights"].astype(int)) == {1, 2, 3, 5}
    assert set(prefix[prefix["cohort"] == "colas"]["prefix_nights"].astype(int)) == {1, 2}

    recomputed = (prefix["risk_abs_error"] <= 0.05) & prefix["tier_agreement"].astype(bool)
    assert np.array_equal(recomputed.to_numpy(bool), prefix["stable_0.05"].astype(bool).to_numpy())
    assert np.all(prefix["q_eff_60"] > 0)
    assert np.all(prefix["q_eff_60"] <= prefix["qualified_nights"] + 1e-12)

    integrity = result["integrity"]
    assert integrity["raw_data_zip_sha256_before"] == integrity["raw_data_zip_sha256_after"]
    assert integrity["index_html_sha256_before"] == integrity["index_html_sha256_after"]
    assert result["protocol"]["source_data_read_only"] is True
    assert result["deployment"]["index_changed"] is False
    assert result["deployment"]["candidate_ready_for_frontend"] is False

    allowed = {
        "qualified_nights", "log_n_raw", "q_eff_60", "log_n_resampled",
        "overall_coverage", "all_longest_gap_hours", "all_jump_fraction",
    }
    forbidden = {"y", "diagnosis", "SSPG", "insulin", "a1c", "fpg", "ogtt"}
    for model in result["model_comparison"]["hall_grouped_oof"].values():
        features = set(model["features"])
        assert features <= allowed
        assert not features & forbidden

    assert result["interpretation"]["main_order_parameter"] in {"qualified_nights", "q_eff_60"}
    assert result["protocol"]["unit"] == "subject"
    print("early robustness order artifact checks: PASS")


if __name__ == "__main__":
    main()
