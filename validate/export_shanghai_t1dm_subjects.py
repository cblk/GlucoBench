#!/usr/bin/env python3
"""Export read-only ShanghaiT1DM subjects for the wind-tunnel harness.

Source: SAME archive as ShanghaiT2DM (Wang et al. 2023), the Shanghai_T1DM/
  subfolder + Shanghai_T1DM_Summary.xlsx, both already sitting in
  output/external_datasets/raw/shanghai_t2dm/diabetes_datasets.zip.

Why this cohort is being extracted now, and why it was deliberately SKIPPED
when export_shanghai_subjects.py was written (see that file's "T1DM
exclusion" docstring section): AGENTS.md's T1D strong-warning doctrine
forbids cross-pooling T1D subjects with T2D/non-diabetic cohorts for
topological comparison (exogenous insulin dosing is a confound that has no
analog in T2D/non-diabetic physiology). This script extracts T1DM in
complete isolation -- its output JSON is never merged with
shanghai_t2dm_subjects.json, and any downstream analysis of this file must
stay strictly within-cohort (e.g. HbA1c severity split among T1DM patients
themselves), never T1DM-vs-T2DM group comparison.

Structural note (small-n, mirrors dataset_fleet_registry.md's pre-existing
"强警示" framing): only 12 unique patients, 16 total recordings. Two patients
(1002, 1006) have 3 admissions each spread over 4-19 months -- these are
flagged via `patient_base_id`/`visit_index` (same convention as the T2DM
script) so a downstream analysis can optionally treat them as same-body
Epoch0/Epoch1/Epoch2 case observations, but n=2 patients is far too small
for a formal statistical claim; this is documented as a descriptive-only
supplement, not a primary result.

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: no physical computation here.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c / diabetes duration /
    complications / insulin regimen are attached as pure metadata, never fed
    into the glucose values themselves.
  - Section 8.1 No Inference & No Fabrication: the same "/" missing-value
    sentinel fix from export_shanghai_subjects.py is reused verbatim here
    (same raw Excel format, same publisher, same sentinel convention).
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "output" / "external_datasets" / "raw" / "shanghai_t2dm" / "diabetes_datasets.zip"
OUTPUT_FILE = ROOT / "output" / "shanghai_t1dm_subjects.json"
MGDL_PER_MMOLL = 18.0182

PATIENT_ID_RE = re.compile(r"^(\d+)_(\d+)_(\d{8})$")


def _summary_lookup(zf: zipfile.ZipFile) -> dict:
    with zf.open("Shanghai_T1DM_Summary.xlsx") as f:
        df = pd.read_excel(f)
    df = df.set_index("Patient Number")
    return df.to_dict(orient="index")


def _get(meta, key, cast=float):
    v = meta.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str) and v.strip() == "/":
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return str(v)


def export_shanghai_t1dm_subjects() -> Path:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Raw archive not found at {ZIP_PATH}.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        summary = _summary_lookup(zf)
        names = [
            n for n in zf.namelist()
            if n.startswith("Shanghai_T1DM/") and n.endswith((".xls", ".xlsx"))
            and "__MACOSX" not in n and not n.endswith("/")
        ]
        print(f"Found {len(names)} Shanghai_T1DM CGM recording files.")

        subjects_list = []
        n_skipped_no_summary = 0
        for name in sorted(names):
            filename = name.split("/")[-1].rsplit(".", 1)[0]
            m = PATIENT_ID_RE.match(filename)
            if not m:
                print(f"  SKIP (unrecognized filename pattern): {name}")
                continue
            patient_base_id, visit_index, admission_date = m.groups()

            meta = summary.get(filename)
            if meta is None:
                n_skipped_no_summary += 1
                print(f"  WARN: no summary row for {filename}, exporting CGM with metadata=None.")
                meta = {}

            with zf.open(name) as f:
                df = pd.read_excel(f, usecols=[0, 1])
            df.columns = ["timestamp", "cgm_mgdl"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["cgm_mgdl"] = pd.to_numeric(df["cgm_mgdl"], errors="coerce")
            df = df.dropna(subset=["timestamp", "cgm_mgdl"])
            if df.empty:
                print(f"  SKIP (no valid CGM rows after parsing): {name}")
                continue
            df = df.groupby("timestamp", as_index=False)["cgm_mgdl"].mean()
            df = df.sort_values("timestamp").reset_index(drop=True)

            duration_days = round(
                (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 3
            )

            record = {
                "cohort": "shanghai_t1dm",
                "id": filename,
                "patient_base_id": patient_base_id,
                "visit_index": int(visit_index),
                "admission_date": admission_date,
                "duration_days": duration_days,
                "n_raw_points": int(len(df)),
                "gender": _get(meta, "Gender (Female=1, Male=2)", int),
                "age_years": _get(meta, "Age (years)", float),
                "bmi": _get(meta, "BMI (kg/m2)", float),
                "diabetes_duration_years": _get(meta, "Duration of Diabetes  (years)", float),
                "hba1c_mmol_mol": _get(meta, "HbA1c (mmol/mol)", float),
                "fasting_plasma_glucose_mgdl": _get(meta, "Fasting Plasma Glucose (mg/dl)", float),
                "fasting_cpeptide_nmol_l": _get(meta, "Fasting C-peptide (nmol/L)", float),
                "fasting_insulin_pmol_l": _get(meta, "Fasting Insulin (pmol/L)", float),
                "acute_complications": _get(meta, "Acute Diabetic Complications", str),
                "macrovascular_complications": _get(meta, "Diabetic Macrovascular  Complications", str),
                "microvascular_complications": _get(meta, "Diabetic Microvascular Complications", str),
                "hypoglycemic_agents": _get(meta, "Hypoglycemic Agents", str),
                "hypoglycemia_yesno": _get(meta, "Hypoglycemia (yes/no)", str),
                "timestamps": [t.isoformat() for t in df["timestamp"]],
                "values": (df["cgm_mgdl"].astype(float) / MGDL_PER_MMOLL).round(4).tolist(),
            }
            subjects_list.append(record)

    n_unique_patients = len({s["patient_base_id"] for s in subjects_list})
    n_multi_visit = sum(
        1 for pid in {s["patient_base_id"] for s in subjects_list}
        if sum(1 for s in subjects_list if s["patient_base_id"] == pid) > 1
    )
    print(f"Exported {len(subjects_list)} recordings from {n_unique_patients} unique patients "
          f"({n_multi_visit} with >1 visit). {n_skipped_no_summary} recordings had no summary metadata match.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "shanghai_t1dm",
            "n_subjects": len(subjects_list),
            "n_unique_patients": n_unique_patients,
            "subjects": subjects_list,
        }, f)

    print(f"Wrote {len(subjects_list)} Shanghai_T1DM recordings to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_shanghai_t1dm_subjects()
