#!/usr/bin/env python3
"""Export frozen-window CGM features from newly downloaded public datasets.

The source archives and workbooks are read only.  The script writes derived
participant-level primitives and a JSON bundle that can be passed through the
production JavaScript metric pipeline.  Clinical fields are endpoints only.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validate"))

from export_composite_abnormality_subjects import fixed_window, primitive_features  # noqa: E402
from validate_agent_nonclassic import MGDL_PER_MMOL  # noqa: E402


EXTERNAL = ROOT / "output" / "external_datasets" / "raw"
OUTPUT = ROOT / "output"
PRIMITIVE_OUTPUT = OUTPUT / "external_base5_primitives.csv"
SUBJECT_OUTPUT = OUTPUT / "external_base5_subjects.json"
STANFORD_OUTPUT = OUTPUT / "external_stanford_intervention_features.csv"
AUDIT_OUTPUT = OUTPUT / "external_base5_export_audit.json"
WINDOW_HOURS = 48
KOBE_PHASE_SHIFTS = (0, 6, 12, 18)


def clean_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def soft_component(value, lower, upper, slope=4.0):
    value = clean_number(value)
    if not np.isfinite(value):
        return np.nan
    z = slope * (value - lower) / (upper - lower)
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z)))))


def weighted_max_mean(values, max_weight=0.60):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(max_weight * np.max(values) + (1.0 - max_weight) * np.mean(values))


def ccas_core(a1c_percent, fpg_mgdl):
    values = [
        soft_component(a1c_percent, 5.7, 6.5),
        soft_component(fpg_mgdl, 100.0, 126.0),
    ]
    if not all(np.isfinite(values)):
        return np.nan
    return 100.0 * weighted_max_mean(values)


def ccas_full(a1c_percent, fpg_mgdl, ogtt_2h_mgdl):
    values = [
        soft_component(a1c_percent, 5.7, 6.5),
        soft_component(fpg_mgdl, 100.0, 126.0),
        soft_component(ogtt_2h_mgdl, 140.0, 200.0),
    ]
    if not all(np.isfinite(values)):
        return np.nan
    return 100.0 * weighted_max_mean(values)


def a1c_ifcc_to_ngsp(value):
    value = clean_number(value)
    return 0.09148 * value + 2.152 if np.isfinite(value) else np.nan


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def add_record(rows, payload, cohort, subject_id, frame, frequency, endpoints):
    data = fixed_window(frame, WINDOW_HOURS)
    features = primitive_features(data, frequency, WINDOW_HOURS)
    row = {
        "cohort": cohort,
        "id": str(subject_id),
        "window_hours": WINDOW_HOURS,
        **endpoints,
        **features,
    }
    rows.append(row)
    if features.get("eligible"):
        payload.setdefault(cohort, []).append({
            "cohort": cohort,
            "id": str(subject_id),
            "timestamps": data["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "values": data["gl_mmol"].round(12).tolist(),
            "y": None,
            "diagnosis": None,
            "insulin": endpoints.get("fasting_insulin_pmol_l"),
            "SSPG": endpoints.get("sspg"),
        })


def export_kobe(rows, payload):
    root = EXTERNAL / "kobe_cgm_ac"
    cgm = pd.read_csv(root / "CGM_data.csv")
    disposition = pd.read_excel(root / "SourceData.xlsx", sheet_name="Fig.1d")
    if len(cgm) != 64 or len(disposition) != 64:
        raise RuntimeError("Kobe row count changed; refusing inferred linkage")

    ids = pd.to_numeric(cgm["ID"], errors="raise").astype(int).tolist()
    missing = [value for value in range(1, 71) if value not in set(ids)]
    linkage = pd.DataFrame({
        "id": ids,
        "cluster": pd.to_numeric(disposition["Cluster"], errors="coerce"),
        "oral_di": pd.to_numeric(disposition["Oral DI"], errors="coerce"),
        "clamp_di": pd.to_numeric(disposition["Clamp DI"], errors="coerce"),
    })
    linkage = linkage.set_index("id")

    minute_columns = [column for column in cgm.columns if str(column) != "ID"]
    minute_offsets = np.asarray([int(column) for column in minute_columns], int)
    for _, source in cgm.iterrows():
        subject_id = int(source["ID"])
        values = pd.to_numeric(source[minute_columns], errors="coerce").to_numpy(float) / MGDL_PER_MMOL
        endpoints = {
            "source_role": "direct_mechanism_external",
            "base_subject_id": str(subject_id),
            "phase_shift_hours": 0,
            "cluster": clean_number(linkage.loc[subject_id, "cluster"]),
            "oral_di": clean_number(linkage.loc[subject_id, "oral_di"]),
            "clamp_di": clean_number(linkage.loc[subject_id, "clamp_di"]),
            "ccas_core": np.nan,
            "a1c_percent": np.nan,
            "fpg_mgdl": np.nan,
            "treatment_status": "untreated",
        }
        for shift in KOBE_PHASE_SHIFTS:
            frame = pd.DataFrame({
                "time": pd.Timestamp("2000-01-01") + pd.to_timedelta(minute_offsets + shift * 60, unit="m"),
                "gl_mmol": values,
            }).dropna()
            shifted = dict(endpoints)
            shifted["phase_shift_hours"] = shift
            add_record(rows, payload, f"kobe_shift{shift}_w48", subject_id, frame, 5, shifted)

    return {
        "linkage_basis": "Fig.1d row order paired to CGM_data retained-ID row order",
        "retained_ids": ids,
        "missing_original_ids": missing,
        "paper_example_checks": {
            "participant_13": {
                "cgm_first_four": pd.to_numeric(cgm.loc[cgm["ID"].eq(13), minute_columns[:4]].iloc[0]).tolist(),
                "cgm_mean_mgdl": float(pd.to_numeric(cgm.loc[cgm["ID"].eq(13), minute_columns].iloc[0]).mean()),
                "clamp_di": float(linkage.loc[13, "clamp_di"]),
            },
            "participant_46": {
                "retained_row_one_based": ids.index(46) + 1,
                "cgm_first_four": pd.to_numeric(cgm.loc[cgm["ID"].eq(46), minute_columns[:4]].iloc[0]).tolist(),
                "cgm_mean_mgdl": float(pd.to_numeric(cgm.loc[cgm["ID"].eq(46), minute_columns].iloc[0]).mean()),
                "clamp_di": float(linkage.loc[46, "clamp_di"]),
            },
        },
        "clock_assumption": "minute 0 treated as 00:00 only for shift0; 6/12/18 h shifts are mandatory sensitivity analyses",
    }


def find_column(columns, prefix):
    for column in columns:
        if str(column).strip().lower().startswith(prefix.lower()):
            return column
    return None


def load_shanghai_xlsx(archive):
    frames = {}
    names = [name for name in archive.namelist() if name.startswith("Shanghai_T2DM/") and name.lower().endswith(".xlsx")]
    for name in names:
        record_id = Path(name).stem
        frame = pd.read_excel(io.BytesIO(archive.read(name)))
        date_column = find_column(frame.columns, "date")
        cgm_column = find_column(frame.columns, "cgm")
        if date_column is None or cgm_column is None:
            continue
        frames[record_id] = pd.DataFrame({
            "time": pd.to_datetime(frame[date_column], errors="coerce"),
            "gl_mmol": pd.to_numeric(frame[cgm_column], errors="coerce") / MGDL_PER_MMOL,
        }).dropna()
    return frames


INSULIN_PATTERN = re.compile(
    r"insulin|novolin|humulin|gansulin|degludec|detemir|glargine|aspart|lispro|30r|40r|50/50|70/30",
    re.IGNORECASE,
)


def export_shanghai(rows, payload):
    archive_path = EXTERNAL / "shanghai_t2dm" / "diabetes_datasets.zip"
    with ZipFile(archive_path) as archive:
        summary_name = next(name for name in archive.namelist() if name.endswith("Shanghai_T2DM_Summary.xlsx"))
        summary = pd.read_excel(io.BytesIO(archive.read(summary_name)))
        frames = load_shanghai_xlsx(archive)

    legacy = pd.read_csv(OUTPUT / "external_shanghai_xls_cgm.csv")
    legacy["time"] = pd.to_datetime(legacy["timestamp"], errors="coerce")
    legacy["gl_mmol"] = pd.to_numeric(legacy["glucose_mgdl"], errors="coerce") / MGDL_PER_MMOL
    for record_id, frame in legacy.groupby("record_id", sort=False):
        frames[str(record_id)] = frame[["time", "gl_mmol"]].dropna().copy()

    mapped = 0
    for _, clinical in summary.iterrows():
        record_id = str(clinical["Patient Number"]).strip()
        frame = frames.get(record_id)
        if frame is None or frame.empty:
            continue
        mapped += 1
        parts = record_id.split("_")
        base_id = parts[0]
        visit_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        treatment_text = str(clinical.get("Hypoglycemic Agents", ""))
        uses_insulin = bool(INSULIN_PATTERN.search(treatment_text))
        a1c_ifcc = clean_number(clinical.get("HbA1c (mmol/mol)"))
        a1c_percent = a1c_ifcc_to_ngsp(a1c_ifcc)
        fpg = clean_number(clinical.get("Fasting Plasma Glucose (mg/dl)"))
        ppg = clean_number(clinical.get("2-hour Postprandial Plasma Glucose (mg/dl)"))
        fasting_insulin = clean_number(clinical.get("Fasting Insulin (pmol/L)"))
        fasting_cpeptide = clean_number(clinical.get("Fasting C-peptide (nmol/L)"))
        homa_ir = np.nan
        if not uses_insulin and np.isfinite(fpg) and np.isfinite(fasting_insulin):
            insulin_uuml = fasting_insulin / 6.0
            homa_ir = (fpg / MGDL_PER_MMOL) * insulin_uuml / 22.5
        endpoints = {
            "source_role": "treated_stress",
            "base_subject_id": base_id,
            "visit_index": visit_index,
            "phase_shift_hours": 0,
            "a1c_ifcc_mmol_mol": a1c_ifcc,
            "a1c_percent": a1c_percent,
            "fpg_mgdl": fpg,
            "postprandial_2h_mgdl_not_confirmed_ogtt": ppg,
            "fasting_insulin_pmol_l": fasting_insulin,
            "fasting_cpeptide_nmol_l": fasting_cpeptide,
            "homa_ir_exploratory": homa_ir,
            "ccas_core": ccas_core(a1c_percent, fpg),
            "treatment_status": "treated",
            "uses_insulin_agent": uses_insulin,
            "hypoglycemic_agents": treatment_text,
        }
        add_record(rows, payload, "shanghai_t2dm_w48", record_id, frame, 15, endpoints)
    return {
        "summary_rows": int(len(summary)),
        "mapped_cgm_records": mapped,
        "xlsx_records": len(frames) - int(legacy["record_id"].nunique()),
        "xls_records": int(legacy["record_id"].nunique()),
    }


def export_big_ideas(rows, payload):
    root = EXTERNAL / "big_ideas_physionet" / "research_relevant_v1.1.3"
    demographics = pd.read_csv(root / "Demographics.csv").set_index("ID")
    mapped = 0
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        subject_id = int(directory.name)
        dexcom_path = next(directory.glob("Dexcom_*.csv"), None)
        if dexcom_path is None or subject_id not in demographics.index:
            continue
        data = pd.read_csv(dexcom_path)
        data = data[data["Event Type"].eq("EGV")].copy()
        frame = pd.DataFrame({
            "time": pd.to_datetime(data["Timestamp (YYYY-MM-DDThh:mm:ss)"], errors="coerce"),
            "gl_mmol": pd.to_numeric(data["Glucose Value (mg/dL)"], errors="coerce") / MGDL_PER_MMOL,
        }).dropna()
        a1c = clean_number(demographics.loc[subject_id, "HbA1c"])
        endpoints = {
            "source_role": "a1c_only_transfer",
            "base_subject_id": str(subject_id),
            "phase_shift_hours": 0,
            "a1c_percent": a1c,
            "a1c_component": 100.0 * soft_component(a1c, 5.7, 6.5),
            "fpg_mgdl": np.nan,
            "ccas_core": np.nan,
            "treatment_status": "unknown",
        }
        add_record(rows, payload, "big_ideas_w48", subject_id, frame, 5, endpoints)
        mapped += 1
    return {"subjects": mapped, "clinical_endpoint": "HbA1c only"}


def curve_features(frame):
    frame = frame.sort_values("Timepoint").dropna(subset=["Timepoint", "Glucose"])
    baseline_values = frame.loc[frame["Timepoint"] <= 0, "Glucose"].to_numpy(float)
    if not len(baseline_values):
        baseline_values = frame.head(2)["Glucose"].to_numpy(float)
    baseline = float(np.mean(baseline_values))
    post = frame[frame["Timepoint"] >= 0]
    if len(post) < 5:
        return None
    time = post["Timepoint"].to_numpy(float)
    glucose = post["Glucose"].to_numpy(float)
    peak_index = int(np.argmax(glucose))
    peak_delta = float(glucose[peak_index] - baseline)
    residual_180 = float(glucose[-1] - baseline)
    iauc = float(np.trapezoid(np.maximum(glucose - baseline, 0.0), x=time))
    recovery_fraction = (peak_delta - residual_180) / peak_delta if peak_delta > 1e-8 else np.nan
    return {
        "baseline_mgdl": baseline,
        "peak_delta_mgdl": peak_delta,
        "time_to_peak_min": float(time[peak_index]),
        "iauc_mgdl_min": iauc,
        "residual_180_mgdl": residual_180,
        "recovery_fraction": recovery_fraction,
    }


def export_stanford():
    archive_path = EXTERNAL / "stanford_metabolic_subphenotype" / "metabolic_subphenotypes_db.zip"
    prefix = "metabolic_subphenotypes_db/"
    with ZipFile(archive_path) as archive:
        read = lambda name: pd.read_csv(io.BytesIO(archive.read(prefix + name)))
        glucose = read("glucose_values_cgm_and_venous.csv")
        demographics = read("demographics.csv")
        phenotype = read("initial_cohort_metabolic_phenotypes.csv")

    experiment = "venous_with_matching_cgm_and_with_planned_athome_cgm"
    demographics = demographics[demographics["ExperimentType"].eq(experiment)].drop_duplicates("SubjectID")
    home = glucose[
        glucose["ExperimentType"].eq(experiment)
        & glucose["SampleLocation_ExtractionMethod"].isin(["Home_CGM_1", "Home_CGM_2"])
    ].copy()
    curve_rows = []
    for (subject_id, method), frame in home.groupby(["SubjectID", "SampleLocation_ExtractionMethod"]):
        features = curve_features(frame)
        if features:
            curve_rows.append({"SubjectID": subject_id, "curve": method, **features})
    curves = pd.DataFrame(curve_rows)
    aggregated = curves.groupby("SubjectID", as_index=False).agg({
        "baseline_mgdl": "mean",
        "peak_delta_mgdl": "mean",
        "time_to_peak_min": "mean",
        "iauc_mgdl_min": "mean",
        "residual_180_mgdl": "mean",
        "recovery_fraction": "mean",
        "curve": "count",
    }).rename(columns={"curve": "home_curve_count"})
    aggregated = aggregated.merge(demographics, on="SubjectID", how="left")
    aggregated = aggregated.merge(phenotype, on="SubjectID", how="left")
    aggregated["ccas_core"] = [ccas_core(a1c, fpg) for a1c, fpg in zip(aggregated["HbA1c"], aggregated["FPG"])]
    aggregated["ccas_full"] = [
        ccas_full(a1c, fpg, ogtt) for a1c, fpg, ogtt in zip(aggregated["HbA1c"], aggregated["FPG"], aggregated["OGTT_2h"])
    ]
    aggregated.to_csv(STANFORD_OUTPUT, index=False)
    overlap = sorted(set(aggregated["SubjectID"]) & set(phenotype["SubjectID"]))
    return {
        "home_cgm_subjects": int(aggregated["SubjectID"].nunique()),
        "home_curves": int(len(curves)),
        "direct_phenotype_overlap_subjects": overlap,
        "direct_phenotype_overlap_n": len(overlap),
        "not_eligible_for_48h_score": True,
    }


def main():
    payload = {
        "metadata": {
            "window_hours": WINDOW_HOURS,
            "glucose_unit": "mmol/L",
            "clinical_fields_are_endpoints_only": True,
            "source_data_read_only": True,
        }
    }
    rows = []
    audit = {
        "kobe": export_kobe(rows, payload),
        "shanghai_t2dm": export_shanghai(rows, payload),
        "big_ideas": export_big_ideas(rows, payload),
        "stanford": export_stanford(),
    }
    frame = pd.DataFrame(rows)
    frame.to_csv(PRIMITIVE_OUTPUT, index=False)
    SUBJECT_OUTPUT.write_text(json.dumps(json_ready(payload), ensure_ascii=False), encoding="utf-8")
    audit["derived_rows"] = int(len(frame))
    audit["eligible_by_cohort"] = {
        str(key): {"rows": int(len(group)), "eligible": int(group["eligible"].sum())}
        for key, group in frame.groupby("cohort")
    }
    AUDIT_OUTPUT.write_text(json.dumps(json_ready(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(audit), ensure_ascii=False, indent=2))
    print(f"wrote {PRIMITIVE_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {SUBJECT_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {STANFORD_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
