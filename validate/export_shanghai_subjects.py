#!/usr/bin/env python3
"""Export read-only ShanghaiT2DM subjects for the wind-tunnel harness.

Source: Shanghai_T2DM cohort (Wang et al. 2023, PhysioNet/figshare mirror),
  raw archive at output/external_datasets/raw/shanghai_t2dm/diabetes_datasets.zip.

Why this cohort is being extracted now (AGENTS.md Section 9.2 Wind-Tunnel
Trigger, driven by dataset_fleet_registry.md Section 5's pre-registered
open question): Colas (n=208, ~2-day recordings) showed Work Integral's
group-separation power COLLAPSE as sample size grew (rank-sep 0.699 -> 0.563
vs Hall). The registry explicitly flagged this as possibly a "short-cycle
collapse" artifact rather than a real property of the operator, and proposed
testing it on a cohort with a NATURAL mix of short and long recordings from
the SAME hospital/protocol/disease-type -- removing the cross-dataset
confounds (different populations, different CGM devices, different eras)
that made the original Kobe/Shanghai/Colas comparisons uninterpretable back
in the 2026-08-11 report. Shanghai_T2DM is exactly this natural experiment:
its 109 CGM recordings span 2.6 to 13.9 days for the SAME disease type and
extraction protocol.

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: this script performs NO physical
    computation. It only reshapes raw Excel rows into the standard
    (timestamps, values) + passthrough-metadata JSON shape every other
    wind-tunnel driver consumes.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c / diabetes duration /
    complications / recording-duration-days are attached as pure metadata.
    None of them are computed from, or fed into, the glucose values.
  - Section 8.1 No Inference & No Fabrication: rows with an unparseable date
    or non-numeric CGM value are dropped (never zero-filled or interpolated
    here -- that is _wind_tunnel_common.resample_raw's job downstream, on
    the raw series, not this script's).
  - T1DM exclusion (deliberate, scoped): the same archive also contains a
    Shanghai_T1DM cohort (n=12). It is INTENTIONALLY NOT extracted here.
    dataset_fleet_registry.md Section 4 already flags T1D cohorts (Weinstock,
    T1D-UOM) as exogenous-insulin-confounded and forbidden from cross-pooling
    with T2D/non-diabetic cohorts for topological comparison. Extracting it
    would invite exactly that mistake later; it is left for a dedicated,
    separately-scoped T1D-only same-body analysis if ever authorized.
  - Multi-visit patients (8 of 100 unique "Patient Number" prefixes have 2-3
    separate hospital admissions, e.g. "2001_0_..." and "2001_1_..."): ALL
    visits are exported here (mechanical extraction, no filtering). Which
    visit(s) to use for a cross-sectional (independence-preserving) group
    comparison vs. a same-body longitudinal comparison is an ANALYSIS-TIME
    decision (Section 9.1.2), not an extraction-time one -- so `visit_index`
    and `patient_base_id` are exported as explicit metadata to make that
    downstream choice possible and auditable.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "output" / "external_datasets" / "raw" / "shanghai_t2dm" / "diabetes_datasets.zip"
OUTPUT_FILE = ROOT / "output" / "shanghai_t2dm_subjects.json"
MGDL_PER_MMOLL = 18.0182

PATIENT_ID_RE = re.compile(r"^(\d+)_(\d+)_(\d{8})$")


def _summary_lookup(zf: zipfile.ZipFile) -> dict:
    with zf.open("Shanghai_T2DM_Summary.xlsx") as f:
        df = pd.read_excel(f)
    df = df.set_index("Patient Number")
    return df.to_dict(orient="index")


def export_shanghai_t2dm_subjects() -> Path:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Raw archive not found at {ZIP_PATH}.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        summary = _summary_lookup(zf)
        names = [
            n for n in zf.namelist()
            if n.startswith("Shanghai_T2DM/") and n.endswith((".xls", ".xlsx"))
            and "__MACOSX" not in n and not n.endswith("/")
        ]
        print(f"Found {len(names)} Shanghai_T2DM CGM recording files.")

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

            def _get(key, cast=float):
                v = meta.get(key)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return None
                # The raw Excel encodes missing numeric fields with the literal
                # sentinel "/" (verified across HbA1c/FPG/C-peptide/Insulin columns).
                # Section 8.1 No Inference & No Fabrication: this must become an
                # honest None, never survive as the string "/" masquerading as data.
                if isinstance(v, str) and v.strip() == "/":
                    return None
                try:
                    return cast(v)
                except (TypeError, ValueError):
                    return str(v)

            record = {
                "cohort": "shanghai_t2dm",
                "id": filename,
                "patient_base_id": patient_base_id,
                "visit_index": int(visit_index),
                "admission_date": admission_date,
                "duration_days": duration_days,
                "n_raw_points": int(len(df)),
                "gender": _get("Gender (Female=1, Male=2)", int),
                "age_years": _get("Age (years)", float),
                "bmi": _get("BMI (kg/m2)", float),
                "diabetes_duration_years": _get("Duration of diabetes (years)", float),
                "hba1c_mmol_mol": _get("HbA1c (mmol/mol)", float),
                "fasting_plasma_glucose_mgdl": _get("Fasting Plasma Glucose (mg/dl)", float),
                "fasting_cpeptide_nmol_l": _get("Fasting C-peptide (nmol/L)", float),
                "fasting_insulin_pmol_l": _get("Fasting Insulin (pmol/L)", float),
                "acute_complications": _get("Acute Diabetic Complications", str),
                "macrovascular_complications": _get("Diabetic Macrovascular  Complications", str),
                "microvascular_complications": _get("Diabetic Microvascular Complications", str),
                "hypoglycemic_agents": _get("Hypoglycemic Agents", str),
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
            "cohort": "shanghai_t2dm",
            "n_subjects": len(subjects_list),
            "n_unique_patients": n_unique_patients,
            "subjects": subjects_list,
        }, f)

    print(f"Wrote {len(subjects_list)} Shanghai_T2DM recordings to {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    export_shanghai_t2dm_subjects()
