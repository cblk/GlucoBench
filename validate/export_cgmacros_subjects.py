#!/usr/bin/env python3
"""Export read-only CGMacros subjects for the wind-tunnel harness.

Source: CGMacros / PhysioNet 2026 (output/cgmacros_subset)
Data integrity & doctrine notes:
  - Glucose values are converted from mg/dL to mmol/L (MGDL_PER_MMOLL = 18.0182)
    to match the production index_v4.html ingestion boundary.
  - Primary sensor: Dexcom GL (Dexcom G6 5-min native resolution).
  - Subsampled at 5-minute intervals to match the standard CGM temporal grid
    across Hall, Colas, and Stanford cohorts.
  - Labels (A1c, Fasting GLU, Insulin, HOMA-IR, BMI, Age, Gender, group_a1c)
    are stored strictly as prism metadata (Section 9.1.2) for post-hoc grouping.
  - Structured meal events are extracted for Vector 1/2 perturbation dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "output" / "cgmacros_subjects.json"
CGMACROS_DIR = ROOT / "output" / "cgmacros_subset"
MGDL_PER_MMOLL = 18.0182


def export_cgmacros_subjects() -> Path:
    bio_path = CGMACROS_DIR / "bio.csv"
    subjects_dir = CGMACROS_DIR / "subjects"

    bio = pd.read_csv(bio_path)
    print(f"Loaded bio.csv with {len(bio)} participants.")

    a1c_col = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    fpg_col = pd.to_numeric(bio["Fasting GLU - PDL (Lab)"], errors="coerce")
    ins_col = pd.to_numeric(bio["Insulin "], errors="coerce")
    bmi_col = pd.to_numeric(bio["BMI"], errors="coerce")
    age_col = pd.to_numeric(bio["Age"], errors="coerce")

    homa_ir = (fpg_col * ins_col) / 405.0

    def get_a1c_group(val: float) -> str:
        if pd.isna(val):
            return "Unknown"
        if val < 5.7:
            return "Normal"
        elif val < 6.5:
            return "Pre-diabetes"
        else:
            return "T2D"

    bio["group_a1c"] = a1c_col.apply(get_a1c_group)
    bio["homa_ir"] = homa_ir

    subject_files = sorted(subjects_dir.glob("CGMacros-*.csv"))
    print(f"Found {len(subject_files)} subject time series files.")

    subjects_list = []
    total_meals = 0

    for f in subject_files:
        sid_num = int(f.stem.split("-")[-1])
        bio_row = bio[bio["subject"] == sid_num]
        if bio_row.empty:
            print(f"Warning: Subject {sid_num} not found in bio.csv, skipping.")
            continue
        bio_row = bio_row.iloc[0]

        df = pd.read_csv(f)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        # Primary sensor: Dexcom GL, fallback to Libre GL if Dexcom is NaN
        gl_mgdl = df["Dexcom GL"].copy()
        gl_mgdl = gl_mgdl.fillna(df["Libre GL"])

        df["gl_mmol"] = gl_mgdl / MGDL_PER_MMOLL

        # Extract structured meal events before subsampling
        meal_rows = df.dropna(subset=["Meal Type"])
        meals = []
        for _, m in meal_rows.iterrows():
            carbs = float(m["Carbs"]) if pd.notna(m["Carbs"]) else 0.0
            calories = float(m["Calories"]) if pd.notna(m["Calories"]) else 0.0
            protein = float(m["Protein"]) if pd.notna(m["Protein"]) else 0.0
            fat = float(m["Fat"]) if pd.notna(m["Fat"]) else 0.0
            fiber = float(m["Fiber"]) if pd.notna(m["Fiber"]) else 0.0
            meals.append({
                "timestamp": m["Timestamp"].isoformat(),
                "meal_type": str(m["Meal Type"]),
                "calories": calories,
                "carbs": carbs,
                "protein": protein,
                "fat": fat,
                "fiber": fiber,
            })
        total_meals += len(meals)

        # 5-minute subsampling to align with standard CGM temporal grid
        df_5min = df.iloc[::5].copy().reset_index(drop=True)

        timestamps = [t.isoformat() for t in df_5min["Timestamp"]]
        values = [float(v) if pd.notna(v) else None for v in df_5min["gl_mmol"]]

        record = {
            "cohort": "cgmacros",
            "id": f"CGMacros-{sid_num:03d}",
            "subject_num": sid_num,
            "group_a1c": bio_row["group_a1c"],
            "a1c": float(bio_row["A1c PDL (Lab)"]) if pd.notna(bio_row["A1c PDL (Lab)"]) else None,
            "fpg": float(bio_row["Fasting GLU - PDL (Lab)"]) if pd.notna(bio_row["Fasting GLU - PDL (Lab)"]) else None,
            "insulin": float(bio_row["Insulin "]) if pd.notna(bio_row["Insulin "]) else None,
            "homa_ir": float(bio_row["homa_ir"]) if pd.notna(bio_row["homa_ir"]) else None,
            "bmi": float(bio_row["BMI"]) if pd.notna(bio_row["BMI"]) else None,
            "age": int(bio_row["Age"]) if pd.notna(bio_row["Age"]) else None,
            "gender": str(bio_row["Gender"]) if pd.notna(bio_row["Gender"]) else None,
            "timestamps": timestamps,
            "values": values,
            "meals": meals,
        }
        subjects_list.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump({
            "cohort": "cgmacros",
            "n_subjects": len(subjects_list),
            "total_meals": total_meals,
            "subjects": subjects_list,
        }, out_f, indent=2)

    print(f"Export complete: {len(subjects_list)} subjects, {total_meals} meals -> {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_cgmacros_subjects()
