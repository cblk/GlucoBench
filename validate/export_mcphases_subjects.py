#!/usr/bin/env python3
"""Export read-only mcPHASES subjects for the wind-tunnel harness.

Source: mcPHASES (PhysioNet, Restricted Health Data, DUA-gated) 2025.
  https://physionet.org/content/mcphases/1.0.0/
  Raw files placed by the user at output/external_datasets/raw/mcphases/.

Data integrity & doctrine notes:
  - CGM exists ONLY in study_interval=2022 (the paper's "Interval 1"); Interval 2
    subjects have no glucose channel at all. This export is Interval-1-only by
    construction (glucose.csv itself only contains those rows).
  - Timestamps are NOT absolute calendar dates in the raw file -- only
    `day_in_study` (relative day offset) + a time-of-day string. We synthesize a
    monotonic datetime axis from an arbitrary anchor date (2022-01-01). This is
    safe because every downstream operator (extract_tau, estimate_dimension,
    RQA, Work Integral) only consumes RELATIVE time deltas between samples, never
    absolute calendar dates -- confirmed against _wind_tunnel_common.py /
    _extracted_tensor_engine_v4.py. No physical quantity depends on which real
    calendar date is used.
  - UNIT ANOMALY (Section 8.2 Honest Fail-Closed -- recorded, not silently
    patched): subjects 6 and 11 have glucose_value recorded in mg/dL (raw means
    ~115/~118) while all other 40 subjects are in mmol/L (raw means ~5.5-7).
    This was verified by per-subject min/max inspection, not assumed. Only ids 6
    and 11 are divided by MGDL_PER_MMOLL; every other subject is passed through
    unconverted. This asymmetric fix is deliberately id-scoped, not global, to
    avoid silently double-converting the other 40 subjects.
  - `phase` (menstrual cycle phase: Follicular / Fertility / Luteal / Menstrual)
    is a TIME-VARYING prism, unlike the static per-subject labels (SSPG, A1C,
    diagnosis) used by every other cohort so far. It is attached here as a
    per-timestamp-aligned parallel array (same length as `timestamps`/`values`),
    NOT as a single subject-level scalar, so a future wind-tunnel driver can
    segment each subject's series by cycle phase (natural same-body Epoch0 vs
    Epoch1 contrast) instead of only doing cross-subject grouping. It is never
    fed into extract_tau / estimate_dimension / compute_rqa / compute_work_integral
    (Section 9.1.1 Calculation Firewall) -- those still only ever see `values`.
  - Fitbit high-frequency channels (heart_rate.csv 2GB, calories.csv 646MB, etc.)
    are intentionally NOT touched by this script. They are a separate, much
    larger extraction to be scoped only if/when cross-modal compensation
    research is explicitly authorized.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "output" / "external_datasets" / "raw" / "mcphases"
OUTPUT_FILE = ROOT / "output" / "mcphases_subjects.json"
MGDL_PER_MMOLL = 18.0182
ANCHOR_DATE = dt.date(2022, 1, 1)
MGDL_SUBJECT_IDS = {6, 11}  # verified by per-subject min/max inspection, not assumed


def _to_datetime(day_in_study: int, time_of_day: str) -> dt.datetime:
    date_part = ANCHOR_DATE + dt.timedelta(days=int(day_in_study) - 1)
    h, m, s = (int(x) for x in time_of_day.split(":"))
    return dt.datetime.combine(date_part, dt.time(h, m, s))


def export_mcphases_subjects() -> Path:
    glucose_path = RAW_DIR / "glucose.csv"
    subject_info_path = RAW_DIR / "subject-info.csv"
    hormones_path = RAW_DIR / "hormones_and_selfreport.csv"

    glucose = pd.read_csv(glucose_path)
    subject_info = pd.read_csv(subject_info_path)
    hormones = pd.read_csv(hormones_path)
    print(f"Loaded glucose.csv: {len(glucose)} rows, {glucose['id'].nunique()} subjects.")

    phase_lookup = (
        hormones.dropna(subset=["phase"])
        .drop_duplicates(subset=["id", "day_in_study"])
        .set_index(["id", "day_in_study"])["phase"]
        .to_dict()
    )
    info_lookup = subject_info.set_index("id").to_dict(orient="index")

    subjects_list = []
    n_mgdl_fixed = 0

    for sid, grp in glucose.groupby("id"):
        grp = grp.sort_values(["day_in_study", "timestamp"]).reset_index(drop=True)

        is_mgdl_subject = int(sid) in MGDL_SUBJECT_IDS
        raw_values = pd.to_numeric(grp["glucose_value"], errors="coerce")
        if is_mgdl_subject:
            values = (raw_values / MGDL_PER_MMOLL).round(4)
            n_mgdl_fixed += 1
        else:
            values = raw_values.round(4)

        timestamps = [
            _to_datetime(d, t).isoformat()
            for d, t in zip(grp["day_in_study"], grp["timestamp"])
        ]
        phases = [phase_lookup.get((sid, d)) for d in grp["day_in_study"]]

        info = info_lookup.get(sid, {})
        birth_year = info.get("birth_year")

        record = {
            "cohort": "mcphases",
            "id": f"mcphases-{int(sid):03d}",
            "subject_num": int(sid),
            "birth_year": int(birth_year) if pd.notna(birth_year) else None,
            "gender": str(info.get("gender")) if pd.notna(info.get("gender")) else None,
            "glucose_unit_source": "mg/dL (converted)" if is_mgdl_subject else "mmol/L (native)",
            "timestamps": timestamps,
            "values": [float(v) if pd.notna(v) else None for v in values],
            "phase": phases,
        }
        subjects_list.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump({
            "cohort": "mcphases",
            "n_subjects": len(subjects_list),
            "n_mgdl_unit_fixed": n_mgdl_fixed,
            "subjects": subjects_list,
        }, out_f)

    print(
        f"Export complete: {len(subjects_list)} subjects "
        f"({n_mgdl_fixed} unit-corrected from mg/dL) -> {OUTPUT_FILE}"
    )
    return OUTPUT_FILE


if __name__ == "__main__":
    export_mcphases_subjects()
