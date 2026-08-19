#!/usr/bin/env python3
"""Export read-only BIG IDEAs subjects for the wind-tunnel harness.

Source: BIG IDEAs Lab Glycemic Variability and Wearable Device Data (PhysioNet),
  raw files at output/external_datasets/raw/big_ideas/<subject_id>/Dexcom_<id>.csv
  + output/external_datasets/raw/big_ideas/Demographics.csv.

This is a baseline extraction ("跑一遍数据集" -- run the standard pipeline once
before any deep single-operator investigation), mirroring the shape used for
every prior cohort: glucose series + one static per-subject clinical prism
(HbA1c), nothing more elaborate yet.

Data integrity & doctrine notes:
  - Dexcom Clarity exports interleave several `Event Type` rows (FirstName,
    LastName, PatientIdentifier, Device, Alert, EGV, ...) in one CSV. Only
    `Event Type == 'EGV'` rows carry an actual timestamped glucose reading;
    all others are metadata/alert rows and are dropped here (Section 8.1 No
    Inference & No Fabrication -- these are not "missing data", they are a
    different row type that was never a glucose measurement).
  - Glucose Value (mg/dL) -> mmol/L via MGDL_PER_MMOLL, matching every other
    cohort's ingestion boundary.
  - HbA1c is attached purely as metadata (Section 9.1.2 Labels as Prisms).
  - Known cohort caveat (registry-documented, repeated here for traceability):
    this cohort's HbA1c band is narrow (5.3-6.4), i.e. no participant is in
    the diabetic range -- any HbA1c-based grouping here tests a much subtler
    contrast than Hall/CGMacros/Shanghai's diabetic-vs-non-diabetic splits.

[2026-08-19 11:46 addendum] Structured meal extraction for Vector 1/2
perturbation dynamics (`w_carb`/`strain_per_carb` cross-source replication,
candidate #1/#2 in candidate_tensor_staging_matrix.md), added to answer the
"deep-dive Food_Log" action item from the baseline scan report. Meal schema
mirrors export_cgmacros_subjects.py's `meals` list exactly (Section 9.4
Bit-for-Bit Truth Across Tracks: same downstream consumer,
analyze_subject_meals() in wind_tunnel_v4_cgmacros_meals.py, is reused
verbatim rather than reimplemented).

Food_Log structural notes (honest, not silently patched):
  - Column naming varies across subjects: most use `time`, four (007/013/
    015/016) use `time_of_day` for the same semantic field. Handled via
    fallback lookup, not a guess.
  - Multiple logged_food rows sharing the exact same (date, time) are
    treated as ONE meal event and their total_carb/calorie/protein/
    total_fat/dietary_fiber are summed (Food_Log's own granularity is the
    ground truth here; no finer time-window merge is invented).
  - Subject 003's Food_Log_003.csv is missing its header row AND three
    columns entirely (time_end, sugar, total_fat -- confirmed by field-count
    mismatch: 11 fields present vs the 14-column schema every other subject
    uses). Reconstructing which 3 of the 14 columns were dropped would be a
    guess, not a fact -- Section 8.1 No Inference & No Fabrication forbids
    silently reassigning columns. Subject 003 is therefore EXCLUDED from
    meal extraction (its glucose-only baseline-scan record is unaffected).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "output" / "external_datasets" / "raw" / "big_ideas"
OUTPUT_FILE = ROOT / "output" / "big_ideas_subjects.json"
MGDL_PER_MMOLL = 18.0182

# The 14-column schema used by 15/16 subjects' Food_Log files.
EXPECTED_FOOD_LOG_COLS = {
    "date", "time_begin", "logged_food", "amount", "unit", "searched_food",
    "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat",
}


def _extract_meals(sdir: Path, sid: str) -> list:
    food_path = sdir / f"Food_Log_{sid}.csv"
    if not food_path.exists():
        print(f"  {sid}: no Food_Log_{sid}.csv found, meals=[].")
        return []

    df = pd.read_csv(food_path)
    cols = set(df.columns)
    if not EXPECTED_FOOD_LOG_COLS.issubset(cols):
        print(f"  SKIP meals for {sid}: Food_Log columns do not match the "
              f"expected 14-column schema (missing {EXPECTED_FOOD_LOG_COLS - cols}) "
              f"-- likely a malformed export (e.g. missing header). Not guessing "
              f"a column mapping; excluding from meal extraction (Section 8.1).")
        return []

    time_col = "time" if "time" in cols else ("time_of_day" if "time_of_day" in cols else None)
    if time_col is None:
        print(f"  SKIP meals for {sid}: neither 'time' nor 'time_of_day' column present.")
        return []

    df["meal_ts"] = pd.to_datetime(df["date"].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["meal_ts"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  {sid}: dropped {n_dropped} Food_Log rows with unparseable date/time.")

    for c in ("total_carb", "calorie", "protein", "total_fat", "dietary_fiber"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = df.groupby("meal_ts", as_index=False).agg(
        total_carb=("total_carb", "sum"),
        calorie=("calorie", "sum"),
        protein=("protein", "sum"),
        total_fat=("total_fat", "sum"),
        dietary_fiber=("dietary_fiber", "sum"),
        n_items=("logged_food", "count"),
    ).sort_values("meal_ts")

    meals = []
    for _, m in grouped.iterrows():
        meals.append({
            "timestamp": m["meal_ts"].isoformat(),
            "meal_type": None,  # Food_Log has no breakfast/lunch/dinner tag; honest None, not guessed.
            "calories": float(m["calorie"]) if pd.notna(m["calorie"]) else 0.0,
            "carbs": float(m["total_carb"]) if pd.notna(m["total_carb"]) else 0.0,
            "protein": float(m["protein"]) if pd.notna(m["protein"]) else 0.0,
            "fat": float(m["total_fat"]) if pd.notna(m["total_fat"]) else 0.0,
            "fiber": float(m["dietary_fiber"]) if pd.notna(m["dietary_fiber"]) else 0.0,
            "n_logged_items": int(m["n_items"]),
        })
    return meals


def export_big_ideas_subjects() -> Path:
    demo = pd.read_csv(RAW_DIR / "Demographics.csv")
    demo["ID"] = demo["ID"].astype(str).str.zfill(3)
    demo_lookup = demo.set_index("ID").to_dict(orient="index")

    subject_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(subject_dirs)} BIG IDEAs subject directories.")

    subjects_list = []
    for sdir in subject_dirs:
        sid = sdir.name
        dexcom_path = sdir / f"Dexcom_{sid}.csv"
        if not dexcom_path.exists():
            print(f"  SKIP {sid}: no Dexcom_{sid}.csv found.")
            continue

        df = pd.read_csv(dexcom_path)
        egv = df[df["Event Type"] == "EGV"].copy()
        egv["ts"] = pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"], errors="coerce")
        egv["glucose_mgdl"] = pd.to_numeric(egv["Glucose Value (mg/dL)"], errors="coerce")
        n_before = len(egv)
        egv = egv.dropna(subset=["ts", "glucose_mgdl"])
        n_dropped = n_before - len(egv)
        egv = egv.groupby("ts", as_index=False)["glucose_mgdl"].mean().sort_values("ts").reset_index(drop=True)

        if len(egv) < 60:
            print(f"  SKIP {sid}: only {len(egv)} valid EGV rows.")
            continue

        meta = demo_lookup.get(sid, {})
        hba1c = meta.get("HbA1c")
        gender = meta.get("Gender")

        duration_days = round((egv["ts"].iloc[-1] - egv["ts"].iloc[0]).total_seconds() / 86400.0, 3)
        meals = _extract_meals(sdir, sid)

        subjects_list.append({
            "cohort": "big_ideas",
            "id": f"big_ideas_{sid}",
            "subject_num": sid,
            "gender": str(gender) if gender is not None and pd.notna(gender) else None,
            "hba1c_pct": float(hba1c) if hba1c is not None and pd.notna(hba1c) else None,
            "duration_days": duration_days,
            "n_raw_points": int(len(egv)),
            "n_rows_dropped_non_egv_or_unparseable": int(n_dropped),
            "timestamps": [t.isoformat() for t in egv["ts"]],
            "values": (egv["glucose_mgdl"].astype(float) / MGDL_PER_MMOLL).round(4).tolist(),
            "meals": meals,
        })
        print(f"  {sid}: {len(egv)} EGV pts over {duration_days} days, HbA1c={hba1c}, gender={gender}, meals={len(meals)}.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "big_ideas",
            "n_subjects": len(subjects_list),
            "subjects": subjects_list,
        }, f)

    print(f"Wrote {len(subjects_list)} BIG IDEAs subjects to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_big_ideas_subjects()
