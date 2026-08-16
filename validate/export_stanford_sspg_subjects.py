#!/usr/bin/env python3
"""Export read-only Stanford Home CGM + SSPG subjects for the wind-tunnel harness.

Source: Snyder Lab / Nature Biomedical Engineering 2025 (output/stanford_repo)
Data integrity & doctrine notes:
  - Glucose values are converted from mg/dL to mmol/L (MGDL_PER_MMOLL = 18.0182)
    to match the production index_v4.html ingestion boundary.
  - Timestamps are deduplicated by averaging any simultaneous sensor readings.
  - Labels (sspg, sspg_class, di, hba1c, fpg, bmi) are stored purely as metadata
    for downstream prism stratification (Section 9.1.2) and are never used as
    computational targets or weights.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "output" / "stanford_sspg_subjects.json"
STANFORD_DATA = ROOT / "output" / "stanford_repo" / "data"
MGDL_PER_MMOLL = 18.0182


def export_stanford_subjects() -> Path:
    cgm = pd.read_csv(STANFORD_DATA / "filtered_cgm_03222026.csv")
    tests = pd.read_csv(STANFORD_DATA / "filtered_metabolic_tests.csv")
    chars = pd.read_csv(STANFORD_DATA / "filtered_study_participants_characteristics.csv")

    common = sorted(set(cgm["subject"].unique()).intersection(set(tests["SubjectID"].unique())))
    print(f"Exporting {len(common)} Stanford subjects with matched CGM and SSPG...")

    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    cgm["glucose_value"] = pd.to_numeric(cgm["glucose_value"], errors="coerce")
    cgm = cgm.dropna(subset=["timestamp", "glucose_value", "subject"])

    subjects_list = []
    for subj_id in common:
        df_s = cgm[cgm["subject"] == subj_id].copy()
        df_s = df_s.groupby("timestamp", as_index=False)["glucose_value"].mean()
        df_s = df_s.sort_values("timestamp").reset_index(drop=True)

        t_row = tests[tests["SubjectID"] == subj_id].iloc[0]
        c_rows = chars[chars["SubjectID"] == subj_id]
        c_row = c_rows.iloc[0] if len(c_rows) > 0 else None

        sspg = float(t_row["sspg"]) if pd.notna(t_row["sspg"]) else None
        sspg_class = str(t_row["sspg_2_classes"]) if pd.notna(t_row["sspg_2_classes"]) else None
        di = float(t_row["di"]) if pd.notna(t_row["di"]) else None

        hba1c = float(c_row["HbA1c"]) if (c_row is not None and pd.notna(c_row["HbA1c"])) else None
        fpg = float(c_row["FPG"]) if (c_row is not None and pd.notna(c_row["FPG"])) else None
        bmi = str(c_row["BMI"]) if (c_row is not None and pd.notna(c_row["BMI"])) else None

        timestamps = [t.isoformat() for t in df_s["timestamp"]]
        values = (df_s["glucose_value"].astype(float) / MGDL_PER_MMOLL).tolist()

        subjects_list.append({
            "cohort": "stanford",
            "id": str(subj_id),
            "sspg": sspg,
            "sspg_class": sspg_class,
            "di": di,
            "hba1c": hba1c,
            "fpg": fpg,
            "bmi": bmi,
            "timestamps": timestamps,
            "values": values,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "stanford",
            "n_subjects": len(subjects_list),
            "subjects": subjects_list,
        }, f, indent=2)

    print(f"Wrote {len(subjects_list)} Stanford subject records to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_stanford_subjects()
