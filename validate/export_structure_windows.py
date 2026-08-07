#!/usr/bin/env python3
"""Export read-only early CGM windows and later outcomes for structure research.

Predictor windows are the first 1, 2, and 3 elapsed days after each subject's
first timestamp. The frozen future window is elapsed days 3-5. Glucose is
converted to mmol/L at the browser ingestion boundary; raw ZIP data are never
extracted or modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "structure_windows.json"
MGDL_PER_MMOLL = 18.0182
WINDOW_DAYS = (1, 2, 3)
FUTURE_START_DAY = 3
FUTURE_END_DAY = 5
MIN_POINTS_PER_DAY = 144  # >=50% of a nominal 5-minute day


HALL_FIELDS = [
    "Age", "BMI", "A1C", "FBG", "ogtt.2hr", "insulin", "SSPG",
    "hs.CRP", "Trg", "HDL", "LDL", "mage", "modd", "coef_variation",
    "freq_severe", "glucotype", "Insulin_rate_dd", "diagnosis",
]
WEINSTOCK_FIELDS = [
    "Gender", "Race", "T1DDiagAge", "NumHospDKA", "NumSHSinceT1DDiag",
    "InsDeliveryMethod", "UnitsInsTotal", "NumMeterCheckDay", "Height", "Weight",
    "Hypertension", "Hyperlipidemia", "Depression", "Coronary artery disease",
    "Diabetic peripheral neuropathy", "Chronic kidney disease",
    "Proliferative diabetic retinopathy",
]


def scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def glucose_stats(values):
    values = np.asarray(values, float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=0))
    return {
        "mean": mean,
        "sd": sd,
        "cv": float(sd / mean) if mean > 1e-8 else None,
        "outOfRangeFraction": float(np.mean((values < 3.9) | (values > 10.0))),
        "lowFraction": float(np.mean(values < 3.9)),
        "highFraction": float(np.mean(values > 10.0)),
        "p95MinusP05": float(np.quantile(values, 0.95) - np.quantile(values, 0.05)),
        "n": int(len(values)),
    }


def serialize(frame, cohort, fields):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["id", "time", "gl"])
    arrays = {f"{cohort}_k{k}": [] for k in WINDOW_DAYS}
    excluded = {f"k{k}": 0 for k in WINDOW_DAYS}
    excluded["future"] = 0

    for subject_id, group in frame.groupby("id", sort=True):
        group = group.sort_values("time")
        start = group["time"].iloc[0]
        values = group["gl"].to_numpy(float)
        if np.median(values) > 30:
            values = values / MGDL_PER_MMOLL
        group = group.assign(gl_mmol=values)

        future = group[
            (group["time"] >= start + pd.Timedelta(days=FUTURE_START_DAY))
            & (group["time"] < start + pd.Timedelta(days=FUTURE_END_DAY))
        ]
        if len(future) < MIN_POINTS_PER_DAY * (FUTURE_END_DAY - FUTURE_START_DAY):
            excluded["future"] += 1
            continue

        first = group.iloc[0]
        clinical = {field: scalar(first[field]) for field in fields}
        future_stats = glucose_stats(future["gl_mmol"].to_numpy(float))
        for k in WINDOW_DAYS:
            early = group[group["time"] < start + pd.Timedelta(days=k)]
            if len(early) < MIN_POINTS_PER_DAY * k:
                excluded[f"k{k}"] += 1
                continue
            record = {
                "cohort": cohort,
                "id": str(subject_id),
                "windowDays": k,
                "timestamps": early["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
                "values": early["gl_mmol"].round(12).tolist(),
                "earlyConventional": glucose_stats(early["gl_mmol"].to_numpy(float)),
                "future": future_stats,
                "clinical": clinical,
            }
            arrays[f"{cohort}_k{k}"].append(record)
    return arrays, excluded


def main():
    with ZipFile(ROOT / "raw_data.zip") as archive:
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle)
        with archive.open("raw_data/weinstock.csv") as handle:
            weinstock = pd.read_csv(handle)

    hall_arrays, hall_excluded = serialize(hall, "hall", HALL_FIELDS)
    weinstock_arrays, weinstock_excluded = serialize(weinstock, "weinstock", WEINSTOCK_FIELDS)
    payload = {
        "metadata": {
            "glucose_unit": "mmol/L",
            "source": "raw_data.zip (read-only)",
            "predictor_windows_elapsed_days": list(WINDOW_DAYS),
            "future_window_elapsed_days": [FUTURE_START_DAY, FUTURE_END_DAY],
            "minimum_points_per_elapsed_day": MIN_POINTS_PER_DAY,
            "out_of_range_definition_mmol_L": "<3.9 or >10.0",
            "excluded": {"hall": hall_excluded, "weinstock": weinstock_excluded},
        },
        **hall_arrays,
        **weinstock_arrays,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for key, rows in payload.items():
        if isinstance(rows, list):
            print(f"{key}: n={len(rows)}")
    print(f"excluded={payload['metadata']['excluded']}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
