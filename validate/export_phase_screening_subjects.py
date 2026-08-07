#!/usr/bin/env python3
"""Export read-only Hall/Colas subject time series for the exact JS pipeline.

The source ZIP is never extracted or modified. Glucose is converted from mg/dL
to mmol/L before serialization, matching index.html's ingestion boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "phase_screening_subjects.json"
MGDL_PER_MMOLL = 18.0182


def valid_clinical(value):
    if pd.isna(value) or float(value) < 0:
        return None
    return float(value)


def serialize_cohort(frame: pd.DataFrame, cohort: str):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["id", "time", "gl"])
    frame["gl_mmol"] = frame["gl"].astype(float) / MGDL_PER_MMOLL
    subjects = []
    for subject_id, group in frame.groupby("id", sort=True):
        group = group.sort_values("time")
        first = group.iloc[0]
        record = {
            "cohort": cohort,
            "id": str(subject_id),
            "timestamps": group["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "values": group["gl_mmol"].round(12).tolist(),
        }
        if cohort == "hall":
            record.update({
                "diagnosis": int(first["diagnosis"]),
                "y": int(first["diagnosis"] >= 1),
                "insulin": valid_clinical(first["insulin"]),
                "SSPG": valid_clinical(first["SSPG"]),
            })
        else:
            label = first["T2DM"]
            if isinstance(label, str):
                label = label.strip().lower() in {"true", "1", "yes"}
            record.update({"y": int(bool(label)), "diagnosis": None, "insulin": None, "SSPG": None})
        subjects.append(record)
    return subjects


def main():
    with ZipFile(ROOT / "raw_data.zip") as archive:
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle)
        with archive.open("raw_data/colas.csv") as handle:
            colas = pd.read_csv(handle)

    payload = {
        "metadata": {
            "glucose_unit": "mmol/L",
            "source": "raw_data.zip (read-only)",
            "mgdl_per_mmoll": MGDL_PER_MMOLL,
        },
        "hall": serialize_cohort(hall, "hall"),
        "colas": serialize_cohort(colas, "colas"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Hall={len(payload['hall'])}; Colas={len(payload['colas'])}; wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
