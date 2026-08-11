#!/usr/bin/env python3
"""Export common CGM-only features and fixed-window subject series.

The experiment intentionally uses only subject id, timestamp, and CGM glucose
as candidate inputs. Clinical fields are copied solely as downstream endpoints.
All source data remain read-only; the GlucoBench ZIP is streamed in place.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validate"))

from validate_agent_nonclassic import (  # noqa: E402
    MGDL_PER_MMOL,
    detect_events,
    extract_features,
    quantile,
    regularize,
)


RAW_ZIP = ROOT / "raw_data.zip"
CGM_ROOT = ROOT / "output" / "cgmacros_subset"
SUBJECT_OUTPUT = ROOT / "output" / "composite_abnormality_subjects.json"
PRIMITIVE_OUTPUT = ROOT / "output" / "composite_abnormality_primitives.csv"

WINDOW_HOURS = (24, 48)
PRIMARY_WINDOW_HOURS = 48
MIN_SPAN_FRACTION = 0.75
MIN_COVERAGE = 0.60
VARIATION_NOISE_FLOOR = 0.15  # mmol/L per standardized 5-minute step


def clean_endpoint(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def bool_label(value) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def fixed_window(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    data = frame[["time", "gl_mmol"]].dropna().copy()
    data = data.sort_values("time").drop_duplicates("time", keep="last")
    if data.empty:
        return data
    start = data["time"].iloc[0]
    return data[(data["time"] >= start) & (data["time"] < start + pd.Timedelta(hours=hours))]


def primitive_features(frame: pd.DataFrame, frequency: int, hours: int) -> dict:
    data = fixed_window(frame, hours)
    if data.empty:
        return {"eligible": False, "exclusion": "no valid CGM rows"}

    series = regularize(data, frequency)
    if series.empty:
        return {"eligible": False, "exclusion": "regularization produced no rows"}

    valid = series.dropna().to_numpy(float)
    span_hours = float((data["time"].iloc[-1] - data["time"].iloc[0]).total_seconds() / 3600)
    expected = max(1, int(hours * 60 / frequency))
    coverage = min(1.0, len(valid) / expected)

    hyper = float(np.mean(np.maximum(valid - 7.8, 0.0) + 2.0 * np.maximum(valid - 10.0, 0.0)))
    hypo = float(np.mean(np.maximum(3.9 - valid, 0.0) + 2.0 * np.maximum(3.0 - valid, 0.0)))

    raw = series.to_numpy(float)
    adjacent = np.isfinite(raw[:-1]) & np.isfinite(raw[1:])
    if adjacent.any():
        standardized_change = np.abs(np.diff(raw)[adjacent]) * 5.0 / frequency
        variation = float(np.mean(np.maximum(standardized_change - VARIATION_NOISE_FLOOR, 0.0)))
    else:
        variation = float("nan")

    events = detect_events(series, frequency)
    recovery = quantile([row["recovery_debt"] for row in events], 0.75) if events else 0.0
    legacy, _ = extract_features(data, frequency)

    cv = float(np.std(valid, ddof=1) / np.mean(valid)) if len(valid) > 1 and np.mean(valid) > 1e-8 else np.nan
    result = {
        "hyper_burden": hyper,
        "hypo_burden": hypo,
        "variation_load": variation,
        "recovery_debt": recovery,
        "anchor_level": legacy.get("anchor_level", np.nan),
        "night_mean": legacy.get("night_mean", np.nan),
        "tir_70_180": float(np.mean((valid >= 3.9) & (valid <= 10.0))),
        "tar_180": float(np.mean(valid > 10.0)),
        "tbr_70": float(np.mean(valid < 3.9)),
        "cv": cv,
        "event_count": len(events),
        "valid_nights": legacy.get("valid_nights", 0),
        "valid_points": len(valid),
        "span_hours": span_hours,
        "coverage": coverage,
    }
    result["eligible"] = bool(
        span_hours >= hours * MIN_SPAN_FRACTION
        and coverage >= MIN_COVERAGE
        and np.isfinite(result["anchor_level"])
        and np.isfinite(result["variation_load"])
    )
    result["exclusion"] = None if result["eligible"] else "failed span/coverage/night-anchor gate"
    return result


def serialize_record(
    cohort_key: str,
    source_cohort: str,
    subject_id,
    frame: pd.DataFrame,
    frequency: int,
    hours: int,
    endpoints: dict,
    sensor: str | None = None,
):
    data = fixed_window(frame, hours)
    features = primitive_features(data, frequency, hours)
    row = {
        "cohort": cohort_key,
        "source_cohort": source_cohort,
        "sensor": sensor,
        "id": str(subject_id),
        "window_hours": hours,
        **endpoints,
        **features,
    }
    if not features["eligible"]:
        return row, None

    subject = {
        "cohort": cohort_key,
        "id": str(subject_id),
        "timestamps": data["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "values": data["gl_mmol"].round(12).tolist(),
        "y": endpoints.get("y"),
        "diagnosis": endpoints.get("diagnosis"),
        "insulin": endpoints.get("insulin"),
        "SSPG": endpoints.get("SSPG"),
    }
    return row, subject


def load_zip_dataset(name: str) -> pd.DataFrame:
    with ZipFile(RAW_ZIP) as archive:
        with archive.open(f"raw_data/{name}.csv") as handle:
            frame = pd.read_csv(handle)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["gl_mmol"] = pd.to_numeric(frame["gl"], errors="coerce") / MGDL_PER_MMOL
    return frame.dropna(subset=["id", "time", "gl_mmol"])


def raw_endpoints(name: str, first: pd.Series) -> dict:
    if name == "colas":
        return {
            "y": bool_label(first["T2DM"]),
            "diagnosis": None,
            "insulin": None,
            "SSPG": None,
            "a1c": clean_endpoint(first.get("HbA1c")),
        }
    if name == "hall":
        diagnosis = int(first["diagnosis"])
        return {
            "y": int(diagnosis >= 1),
            "diagnosis": diagnosis,
            "insulin": clean_endpoint(first.get("insulin")),
            "SSPG": clean_endpoint(first.get("SSPG")),
            "a1c": clean_endpoint(first.get("A1C")),
        }
    return {"y": None, "diagnosis": None, "insulin": None, "SSPG": None, "a1c": None}


def append_zip_cohort(name: str, payload: dict, rows: list[dict]):
    frame = load_zip_dataset(name)
    for subject_id, group in frame.groupby("id", sort=True):
        first = group.iloc[0]
        endpoints = raw_endpoints(name, first)
        base = group[["time", "gl_mmol"]]
        for hours in WINDOW_HOURS:
            key = f"{name}_w{hours}"
            row, subject = serialize_record(key, name, subject_id, base, 5, hours, endpoints)
            rows.append(row)
            if subject is not None:
                payload.setdefault(key, []).append(subject)


def load_cgmacros_bio() -> pd.DataFrame:
    bio = pd.read_csv(CGM_ROOT / "bio.csv")
    bio.columns = [str(column).strip() for column in bio.columns]
    bio = bio.rename(columns={
        "subject": "id",
        "A1c PDL (Lab)": "a1c",
        "Fasting GLU - PDL (Lab)": "fasting_glucose",
        "Insulin": "insulin",
    })
    bio["id"] = pd.to_numeric(bio["id"], errors="raise").astype(int)
    return bio.set_index("id")


def append_cgmacros(payload: dict, rows: list[dict]):
    bio = load_cgmacros_bio()
    for path in sorted((CGM_ROOT / "subjects").glob("CGMacros-*.csv")):
        subject_id = int(path.stem.split("-")[-1])
        data = pd.read_csv(path)
        timestamps = pd.to_datetime(data["Timestamp"], errors="coerce")
        clinical = bio.loc[subject_id]
        endpoints = {
            "y": None,
            "diagnosis": None,
            "insulin": clean_endpoint(clinical.get("insulin")),
            "SSPG": None,
            "a1c": clean_endpoint(clinical.get("a1c")),
        }
        for sensor, column, frequency in (("libre", "Libre GL", 15), ("dexcom", "Dexcom GL", 5)):
            frame = pd.DataFrame({
                "time": timestamps,
                "gl_mmol": pd.to_numeric(data[column], errors="coerce") / MGDL_PER_MMOL,
            }).dropna()
            for hours in WINDOW_HOURS:
                key = f"cgmacros_{sensor}_w{hours}"
                row, subject = serialize_record(
                    key, "cgmacros", subject_id, frame, frequency, hours, endpoints, sensor=sensor
                )
                rows.append(row)
                if subject is not None:
                    payload.setdefault(key, []).append(subject)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main():
    payload = {
        "metadata": {
            "glucose_unit": "mmol/L",
            "source_fields": ["subject_id", "timestamp", "CGM glucose"],
            "windows_hours": list(WINDOW_HOURS),
            "primary_window_hours": PRIMARY_WINDOW_HOURS,
            "minimum_span_fraction": MIN_SPAN_FRACTION,
            "minimum_coverage": MIN_COVERAGE,
            "variation_noise_floor_mmol_per_5min": VARIATION_NOISE_FLOOR,
        }
    }
    rows: list[dict] = []
    for name in ("colas", "hall", "iglu", "dubosson", "weinstock"):
        append_zip_cohort(name, payload, rows)
    append_cgmacros(payload, rows)

    SUBJECT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUBJECT_OUTPUT.write_text(json.dumps(json_ready(payload), ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(PRIMITIVE_OUTPUT, index=False)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["cohort", "window_hours"])["eligible"].agg(["count", "sum"])
    print(summary.to_string())
    print(f"wrote {SUBJECT_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {PRIMITIVE_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
