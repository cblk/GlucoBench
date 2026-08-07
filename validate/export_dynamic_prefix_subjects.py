#!/usr/bin/env python3
"""Create deterministic 1/2/3/5-night prefixes without altering source data.

A usable calendar night has at least 48 raw observations between 00:00 and
06:00. For each requested k, the prefix ends at 23:59:59 on the kth usable
night's date and starts with the subject's first observation. This preserves
the browser pipeline's natural longitudinal context while preventing future
nights from leaking into a shorter-record evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "phase_screening_subjects.json"
OUTPUT = ROOT / "output" / "dynamic_prefix_subjects.json"
REQUESTED_NIGHTS = (1, 2, 3, 5)
MIN_NIGHT_POINTS = 48


def usable_night_dates(timestamps):
    times = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    night = times[(times.dt.hour >= 0) & (times.dt.hour < 6)].dropna()
    counts = night.dt.normalize().value_counts().sort_index()
    return [date for date, count in counts.items() if count >= MIN_NIGHT_POINTS]


def prefix_record(subject, k, end_date):
    times = pd.to_datetime(pd.Series(subject["timestamps"]), errors="coerce")
    end = end_date + pd.Timedelta(days=1)
    keep = (times < end).to_numpy(bool)
    record = dict(subject)
    record["timestamps"] = [value for value, selected in zip(subject["timestamps"], keep) if selected]
    record["values"] = [value for value, selected in zip(subject["values"], keep) if selected]
    record["prefixNights"] = k
    return record


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = {
        "metadata": {
            **source.get("metadata", {}),
            "derivation": "first k usable 00:00-06:00 calendar nights; >=48 raw observations/night",
            "requested_nights": list(REQUESTED_NIGHTS),
        }
    }
    for cohort in ("hall", "colas"):
        availability = {k: [] for k in REQUESTED_NIGHTS}
        for subject in source[cohort]:
            dates = usable_night_dates(subject["timestamps"])
            for k in REQUESTED_NIGHTS:
                if len(dates) >= k:
                    availability[k].append(prefix_record(subject, k, dates[k - 1]))
        for k, records in availability.items():
            if records:
                payload[f"{cohort}_k{k}"] = records
                positives = sum(int(record["y"]) for record in records)
                print(f"{cohort}_k{k}: n={len(records)}, positives={positives}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
