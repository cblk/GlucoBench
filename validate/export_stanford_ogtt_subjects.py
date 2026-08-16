#!/usr/bin/env python3
"""Export read-only Stanford OGTT-CGM traces (standardized 75g glucose challenge)
for the wind-tunnel meal-perturbation harness cross-validation.

Source: output/stanford_repo/data/filtered_ogtt_glucose_timeseries_ctru_athome_venous_cgm_03222026.csv
Data integrity & doctrine notes:
  - Uses CTRU_CGM (in-clinic, most standardized) samples as primary source;
    falls back to Home_CGM_1/Home_CGM_2 if CTRU_CGM is unavailable for a
    subject, so long as at least one CGM-based OGTT trace exists.
  - Glucose values are converted from mg/dL to mmol/L (MGDL_PER_MMOLL = 18.0182).
  - The OGTT load is a FIXED, protocol-defined 75g -- this is a controlled
    external perturbation (unlike CGMacros' variable free-living meals),
    making it a genuinely INDEPENDENT perturbation source for cross-validation
    per the Staging Matrix's "Cross-Source Replication" criterion.
  - SSPG / sspg_2_classes / di are attached purely as prism metadata
    (Section 9.1.2), never used as fit targets.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "output" / "stanford_ogtt_subjects.json"
STANFORD_DATA = ROOT / "output" / "stanford_repo" / "data"
MGDL_PER_MMOLL = 18.0182
OGTT_CARBS_G = 75.0  # Standardized 75g oral glucose tolerance test load


def export_stanford_ogtt_subjects() -> Path:
    ogtt = pd.read_csv(STANFORD_DATA / "filtered_ogtt_glucose_timeseries_ctru_athome_venous_cgm_03222026.csv")
    tests = pd.read_csv(STANFORD_DATA / "filtered_metabolic_tests.csv")
    tests = tests.drop_duplicates(subset=["SubjectID"], keep="first")

    # Priority: CTRU_CGM (in-clinic standardized) > Home_CGM_1 > Home_CGM_2
    priority = ["CTRU_CGM", "Home_CGM_1", "Home_CGM_2"]

    subjects_list = []
    for subj_id, t_row in tests.set_index("SubjectID").iterrows():
        trace = None
        source_used = None
        for loc in priority:
            candidate = ogtt[(ogtt["SubjectID"] == subj_id) & (ogtt["SampleLocation_ExtractionMethod"] == loc)]
            if len(candidate) >= 15:  # Require a reasonably complete trace
                trace = candidate.sort_values("Timepoint")
                source_used = loc
                break
        if trace is None:
            continue

        sspg = float(t_row["sspg"]) if pd.notna(t_row["sspg"]) else None
        sspg_class = str(t_row["sspg_2_classes"]) if pd.notna(t_row["sspg_2_classes"]) else None
        di = float(t_row["di"]) if pd.notna(t_row["di"]) else None
        if sspg is None:
            continue  # No prism label, exclude from this cross-validation cohort

        timepoints_min = trace["Timepoint"].tolist()
        values_mmol = (pd.to_numeric(trace["Glucose"], errors="coerce") / MGDL_PER_MMOLL).tolist()

        subjects_list.append({
            "cohort": "stanford_ogtt",
            "id": str(subj_id),
            "source_location": source_used,
            "sspg": sspg,
            "sspg_class": sspg_class,
            "di": di,
            "timepoints_min": timepoints_min,
            "values_mmol": values_mmol,
            "carbs_g": OGTT_CARBS_G,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "stanford_ogtt",
            "n_subjects": len(subjects_list),
            "carbs_g": OGTT_CARBS_G,
            "subjects": subjects_list,
        }, f, indent=2)

    print(f"Exported {len(subjects_list)} Stanford OGTT-CGM subjects (with SSPG label) to {OUTPUT_FILE}")
    src_counts = pd.Series([s["source_location"] for s in subjects_list]).value_counts()
    print("Source location breakdown:", src_counts.to_dict())
    return OUTPUT_FILE


if __name__ == "__main__":
    export_stanford_ogtt_subjects()
