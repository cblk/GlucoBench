#!/usr/bin/env python3
"""Export read-only T1D-UOM subjects for the wind-tunnel harness.

Source: T1D-UOM -- A Longitudinal Multimodal Dataset of Type 1 Diabetes
  (University of Manchester, Zenodo DOI 10.5281/zenodo.15169263), raw archive
  at output/external_datasets/raw/t1d_uom/T1D-UOM.zip.

Why this cohort, and why NOT a cross-subject comparison (AGENTS.md Section 4
strong-warning + dataset_fleet_registry.md Section 4): T1D-UOM's glucose
dynamics are dominated by exogenous insulin dosing, not pure endogenous
feedback, so it must NEVER be pooled cross-subject against T2D/non-diabetic
cohorts (that mistake already collapsed once in the 2026-08-11 historical
report). The registry explicitly reserves T1D-UOM's real value for a
same-body Epoch0/Epoch1 contrast (AGENTS.md Section 7), and the
mcPHASES paired-analysis report (2026-08-16) explicitly named this cohort's
90-day longitudinal span as the next place to try to reproduce the `dim`
(embedding dimension) signal that leaked through when Work Integral failed
on Stanford SSPG's resting-state comparison.

This script performs NO physical computation. It reshapes three raw CSV
families per subject into the standard wind-tunnel subject shape, PLUS
per-calendar-week aggregate metadata used ONLY as a same-body activity
prism (never as a fit target, Section 9.1.2):
  - Glucose Data/UoMGlucose<ID>.csv  -> timestamps/values (mmol/L, already
    native units, no conversion needed).
  - Activity Data/UoMActivity<ID>.csv -> weekly step_count / active_Kcal
    totals (the "high activity week vs low activity week" prism).
  - Insulin Data/Basal Data/UoMBasal<ID>.csv -> weekly total basal dose
    (attached PURELY as an honest confound-disclosure field, per Wind-Tunnel
    Doctrine v1.1's Thermodynamic Bill -- if basal dosing itself co-varies
    with activity level, that is a real physiological confound the report
    must disclose, not hide).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: no glucose computation happens here.
  - Section 9.1.2 Labels as Prisms, Not Targets: weekly activity totals are
    a behavioral/environmental exposure variable, not a clinical diagnostic
    label, and are used ONLY for within-subject relative (median-split)
    ranking downstream, never as a regression target.
  - Section 8.1 No Inference & No Fabrication: the README's data dictionary
    claims `MM/DD/YYYY HH:MM:SS` timestamps, but manual inspection of
    UoMGlucose2301.csv found a value "13/10/2023" that cannot be a month --
    the actual format is DD/MM/YYYY (day-first). This mismatch is corrected
    here explicitly (dayfirst format string), not silently guessed via
    pandas' error-coercing fallback.
  - Subjects missing Basal Data (3 of 17) get `weekly_basal_dose_total=None`
    for every week, never a fabricated 0.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "output" / "external_datasets" / "raw" / "t1d_uom" / "T1D-UOM.zip"
OUTPUT_FILE = ROOT / "output" / "t1d_uom_subjects.json"
ARCHIVE_ROOT = "ManchesterCSCoordinatedDiabetesStudy-1.0.3"

# README's data dictionary claims MM/DD/YYYY; verified-actual format is DD/MM/YYYY
# (day-first) -- see module docstring "No Inference & No Fabrication" note.
TS_FORMAT_MINUTE = "%d/%m/%Y %H:%M"
TS_FORMAT_SECOND = "%d/%m/%Y %H:%M:%S"


def _subject_ids(zf: zipfile.ZipFile) -> list[str]:
    names = zf.namelist()
    ids = sorted({
        re.search(r"UoMGlucose(\d+)\.csv", n).group(1)
        for n in names if re.search(r"UoMGlucose(\d+)\.csv", n)
    })
    return ids


def _load_glucose(zf: zipfile.ZipFile, sid: str) -> pd.DataFrame:
    path = f"{ARCHIVE_ROOT}/Glucose Data/UoMGlucose{sid}.csv"
    with zf.open(path) as f:
        df = pd.read_csv(f)
    df["bg_ts"] = pd.to_datetime(df["bg_ts"], format=TS_FORMAT_MINUTE, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["bg_ts", "value"])
    n_dropped = n_before - len(df)
    df = df.groupby("bg_ts", as_index=False)["value"].mean().sort_values("bg_ts").reset_index(drop=True)
    return df, n_dropped


def _load_activity(zf: zipfile.ZipFile, sid: str) -> pd.DataFrame | None:
    path = f"{ARCHIVE_ROOT}/Activity Data/UoMActivity{sid}.csv"
    if path not in zf.namelist():
        return None
    with zf.open(path) as f:
        df = pd.read_csv(f)
    df["activity_ts"] = pd.to_datetime(df["activity_ts"], format=TS_FORMAT_MINUTE, errors="coerce")
    for col in ("step_count", "active_Kcal"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["activity_ts"])


def _load_basal(zf: zipfile.ZipFile, sid: str) -> pd.DataFrame | None:
    path = f"{ARCHIVE_ROOT}/Insulin Data/Basal Data/UoMBasal{sid}.csv"
    if path not in zf.namelist():
        return None
    with zf.open(path) as f:
        df = pd.read_csv(f)
    # basal_ts observed at both minute and (rarely) second resolution across subjects;
    # try minute format first, fall back to second format, never silently coerce to NaT en masse.
    ts_min = pd.to_datetime(df["basal_ts"], format=TS_FORMAT_MINUTE, errors="coerce")
    ts_sec = pd.to_datetime(df["basal_ts"], format=TS_FORMAT_SECOND, errors="coerce")
    df["basal_ts"] = ts_min.fillna(ts_sec)
    df["basal_dose"] = pd.to_numeric(df["basal_dose"], errors="coerce")
    return df.dropna(subset=["basal_ts", "basal_dose"])


def _weekly_bins(start, end):
    """Fixed 7-day calendar bins anchored at this subject's own first glucose
    sample (Section 8.3 Zero Magic-Constant: no externally-imposed calendar
    week boundary is used, only the subject's own recording start)."""
    bins = []
    cur = start
    while cur < end:
        bins.append((cur, cur + pd.Timedelta(days=7)))
        cur += pd.Timedelta(days=7)
    return bins


def export_t1d_uom_subjects() -> Path:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Raw archive not found at {ZIP_PATH}.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        sids = _subject_ids(zf)
        print(f"Found {len(sids)} T1D-UOM subjects: {sids}")

        subjects_list = []
        for sid in sids:
            glucose, n_dropped_glucose = _load_glucose(zf, sid)
            if len(glucose) < 60:
                print(f"  SKIP {sid}: only {len(glucose)} valid glucose rows.")
                continue

            activity = _load_activity(zf, sid)
            basal = _load_basal(zf, sid)

            weeks = []
            for w_start, w_end in _weekly_bins(glucose["bg_ts"].iloc[0], glucose["bg_ts"].iloc[-1]):
                g_mask = (glucose["bg_ts"] >= w_start) & (glucose["bg_ts"] < w_end)
                week_glucose = glucose[g_mask]
                if len(week_glucose) < 30:
                    continue  # too sparse to even attempt run_subject downstream

                if activity is not None:
                    a_mask = (activity["activity_ts"] >= w_start) & (activity["activity_ts"] < w_end)
                    week_activity = activity[a_mask]
                    weekly_steps = float(week_activity["step_count"].sum()) if len(week_activity) else None
                    weekly_kcal = float(week_activity["active_Kcal"].sum()) if len(week_activity) else None
                    n_activity_records = int(len(week_activity))
                else:
                    weekly_steps = None
                    weekly_kcal = None
                    n_activity_records = 0

                if basal is not None:
                    b_mask = (basal["basal_ts"] >= w_start) & (basal["basal_ts"] < w_end)
                    weekly_basal = float(basal.loc[b_mask, "basal_dose"].sum()) if b_mask.sum() else None
                else:
                    weekly_basal = None

                weeks.append({
                    "week_start": w_start.isoformat(),
                    "week_end": w_end.isoformat(),
                    "n_glucose_points": int(len(week_glucose)),
                    "weekly_step_count_total": weekly_steps,
                    "weekly_active_kcal_total": weekly_kcal,
                    "n_activity_records": n_activity_records,
                    "weekly_basal_dose_total": weekly_basal,
                    "timestamps": [t.isoformat() for t in week_glucose["bg_ts"]],
                    "values": week_glucose["value"].astype(float).round(4).tolist(),
                })

            subjects_list.append({
                "cohort": "t1d_uom",
                "id": f"t1d_uom_{sid}",
                "subject_num": sid,
                "n_total_glucose_points": int(len(glucose)),
                "n_glucose_rows_dropped_unparseable": int(n_dropped_glucose),
                "recording_start": glucose["bg_ts"].iloc[0].isoformat(),
                "recording_end": glucose["bg_ts"].iloc[-1].isoformat(),
                "duration_days": round((glucose["bg_ts"].iloc[-1] - glucose["bg_ts"].iloc[0]).total_seconds() / 86400.0, 2),
                "has_activity_data": activity is not None,
                "has_basal_data": basal is not None,
                "n_weeks": len(weeks),
                "weeks": weeks,
            })
            print(f"  {sid}: {len(glucose)} glucose pts over "
                  f"{(glucose['bg_ts'].iloc[-1] - glucose['bg_ts'].iloc[0]).days} days, "
                  f"{len(weeks)} usable weekly segments "
                  f"(activity={'Y' if activity is not None else 'N'}, basal={'Y' if basal is not None else 'N'}).")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "t1d_uom",
            "n_subjects": len(subjects_list),
            "subjects": subjects_list,
        }, f)

    print(f"Wrote {len(subjects_list)} T1D-UOM subjects to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_t1d_uom_subjects()
