#!/usr/bin/env python3
"""Export matched natural/intervention state windows for structure-reserve research.

The raw ZIP is opened read-only and never extracted. Hall and Weinstock subjects
are restricted to their first four calendar days containing both a sufficiently
sampled night (00:00-06:00) and daytime (06:00-18:00) state. Full, odd-day and
even-day records support an independent split-half reliability check.

Dubosson records are exported as isolated intervention events. An event is the
onset of a positive fast-insulin or calorie trajectory; onsets within 30 minutes
are merged, and events with another onset in the pre/post observation interval
are excluded. These event records are mechanistic only, never clinical labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "structure_reserve_windows.json"
MGDL_PER_MMOLL = 18.0182
MIN_NIGHT_POINTS = 36
MIN_DAY_POINTS = 72
N_PAIRED_DAYS = 4

HALL_FIELDS = [
    "Age", "BMI", "A1C", "FBG", "ogtt.2hr", "insulin", "SSPG",
    "diagnosis", "glucotype", "Insulin_rate_dd",
]
WEINSTOCK_FIELDS = [
    "Gender", "T1DDiagAge", "NumHospDKA", "NumSHSinceT1DDiag",
    "InsDeliveryMethod", "UnitsInsTotal", "NumMeterCheckDay", "Height",
    "Weight", "Hypertension", "Hyperlipidemia", "Depression",
]


def scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def convert_glucose(values):
    values = np.asarray(values, float)
    if np.nanmedian(values) > 30:
        values = values / MGDL_PER_MMOLL
    return values


def serialize_state_cohort(frame, cohort, fields, paired_day_count=N_PAIRED_DAYS):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["gl"] = pd.to_numeric(frame["gl"], errors="coerce")
    frame = frame.dropna(subset=["id", "time", "gl"]).sort_values(["id", "time"])

    records = []
    exclusion_key = f"insufficient_{paired_day_count}_paired_days"
    excluded = {exclusion_key: 0}
    for subject_id, group in frame.groupby("id", sort=True):
        group = group.copy()
        group["date"] = group["time"].dt.date
        hour = group["time"].dt.hour + group["time"].dt.minute / 60 + group["time"].dt.second / 3600
        group["hour"] = hour
        counts = group.groupby("date").agg(
            night=("hour", lambda x: int(((x >= 0) & (x < 6)).sum())),
            daytime=("hour", lambda x: int(((x >= 6) & (x < 18)).sum())),
        )
        paired_dates = counts.index[
            (counts["night"] >= MIN_NIGHT_POINTS) & (counts["daytime"] >= MIN_DAY_POINTS)
        ].tolist()
        if len(paired_dates) < paired_day_count:
            excluded[exclusion_key] += 1
            continue

        selected = paired_dates[:paired_day_count]
        first = group.iloc[0]
        clinical = {field: scalar(first[field]) for field in fields}
        subsets = {
            "full": selected,
            "odd": selected[::2],
            "even": selected[1::2],
        }
        for split, dates in subsets.items():
            state = group[group["date"].isin(dates) & (group["hour"] >= 0) & (group["hour"] < 18)].copy()
            values = convert_glucose(state["gl"].to_numpy(float))
            records.append({
                "cohort": cohort,
                "id": str(subject_id),
                "split": split,
                "pairedDayCount": paired_day_count,
                "pairedDates": [str(value) for value in dates],
                "timestamps": state["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
                "values": np.round(values, 12).tolist(),
                "clinical": clinical,
            })
    return records, excluded


def onset_indices(values):
    values = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(float)
    return np.flatnonzero((values > 0) & ~np.r_[False, values[:-1] > 0])


def serialize_dubosson_events(frame):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["gl"] = pd.to_numeric(frame["gl"], errors="coerce")
    frame = frame.dropna(subset=["id", "time", "gl"]).sort_values(["id", "time"])

    events = []
    subject_summary = []
    for subject_id, group in frame.groupby("id", sort=True):
        group = group.reset_index(drop=True)
        candidates = []
        for field in ("fast_insulin", "calories"):
            for index in onset_indices(group[field]):
                candidates.append((group.loc[index, "time"], field))
        candidates.sort(key=lambda item: item[0])

        merged = []
        for timestamp, field in candidates:
            if merged and (timestamp - merged[-1]["time"]).total_seconds() <= 30 * 60:
                merged[-1]["types"].add(field)
            else:
                merged.append({"time": timestamp, "types": {field}})

        accepted = 0
        for index, event in enumerate(merged):
            timestamp = event["time"]
            isolated = all(
                other_index == index
                or not (timestamp - pd.Timedelta(minutes=120) < other["time"] < timestamp + pd.Timedelta(minutes=180))
                for other_index, other in enumerate(merged)
            )
            if not isolated:
                continue

            pre = group[
                (group["time"] >= timestamp - pd.Timedelta(minutes=120))
                & (group["time"] < timestamp - pd.Timedelta(minutes=15))
            ].copy()
            post = group[
                (group["time"] >= timestamp + pd.Timedelta(minutes=15))
                & (group["time"] <= timestamp + pd.Timedelta(minutes=180))
            ].copy()
            if len(pre) < 15 or len(post) < 24:
                continue

            load_window = group[
                (group["time"] >= timestamp) & (group["time"] <= timestamp + pd.Timedelta(minutes=30))
            ]
            pre_values = convert_glucose(pre["gl"].to_numpy(float))
            post_values = convert_glucose(post["gl"].to_numpy(float))
            accepted += 1
            events.append({
                "cohort": "dubosson",
                "id": str(subject_id),
                "eventId": f"{subject_id}_{accepted}",
                "eventTime": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                "eventTypes": sorted(event["types"]),
                "fastInsulinPeak30m": float(pd.to_numeric(load_window["fast_insulin"], errors="coerce").max()),
                "caloriesPeak30m": float(pd.to_numeric(load_window["calories"], errors="coerce").max()),
                "preTimestamps": pre["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
                "preValues": np.round(pre_values, 12).tolist(),
                "postTimestamps": post["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
                "postValues": np.round(post_values, 12).tolist(),
            })
        subject_summary.append({
            "id": str(subject_id),
            "rawOnsets": len(candidates),
            "mergedOnsets": len(merged),
            "isolatedCompleteEvents": accepted,
        })
    return events, subject_summary


def main():
    with ZipFile(ROOT / "raw_data.zip") as archive:
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle)
        with archive.open("raw_data/weinstock.csv") as handle:
            weinstock = pd.read_csv(handle)
        with archive.open("raw_data/dubosson.csv") as handle:
            dubosson = pd.read_csv(handle)

    hall_records, hall_excluded = serialize_state_cohort(hall, "hall", HALL_FIELDS)
    weinstock_records, weinstock_excluded = serialize_state_cohort(
        weinstock, "weinstock", WEINSTOCK_FIELDS,
    )
    hall_six_day, hall_six_excluded = serialize_state_cohort(
        hall, "hall", HALL_FIELDS, paired_day_count=6,
    )
    weinstock_six_day, weinstock_six_excluded = serialize_state_cohort(
        weinstock, "weinstock", WEINSTOCK_FIELDS, paired_day_count=6,
    )
    dubosson_events, dubosson_summary = serialize_dubosson_events(dubosson)
    payload = {
        "metadata": {
            "source": "raw_data.zip (read-only)",
            "glucoseUnit": "mmol/L",
            "pairedDaysPerSubject": N_PAIRED_DAYS,
            "nightDefinition": "00:00-06:00, >=36 raw points/day",
            "daytimeDefinition": "06:00-18:00, >=72 raw points/day",
            "splits": {"full": [1, 2, 3, 4], "odd": [1, 3], "even": [2, 4]},
            "excluded": {"hall": hall_excluded, "weinstock": weinstock_excluded},
            "sixDaySensitivitySplits": {"full": [1, 2, 3, 4, 5, 6], "odd": [1, 3, 5], "even": [2, 4, 6]},
            "sixDaySensitivityExcluded": {"hall": hall_six_excluded, "weinstock": weinstock_six_excluded},
            "dubossonEventRule": "zero-to-positive fast_insulin/calories onset; merge <=30m; no other onset in [-120m,+180m]",
            "dubossonSubjects": dubosson_summary,
        },
        "stateRecords": hall_records + weinstock_records,
        "stateRecordsSixDay": hall_six_day + weinstock_six_day,
        "dubossonEvents": dubosson_events,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for cohort in ("hall", "weinstock"):
        rows = [row for row in payload["stateRecords"] if row["cohort"] == cohort]
        subjects = len({row["id"] for row in rows})
        print(f"{cohort}: subjects={subjects}, records={len(rows)}")
        six_rows = [row for row in payload["stateRecordsSixDay"] if row["cohort"] == cohort]
        six_subjects = len({row["id"] for row in six_rows})
        print(f"{cohort} six-day sensitivity: subjects={six_subjects}, records={len(six_rows)}")
    print(f"dubosson: events={len(dubosson_events)}, subjects={len({row['id'] for row in dubosson_events})}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
