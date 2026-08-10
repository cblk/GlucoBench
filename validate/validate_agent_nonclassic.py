"""Agent-guided validation of non-classic CGM features.

The protocol was frozen on 2026-08-10 before the Hall SSPG endpoint was used
for feature selection.  Candidate generation sees only subject id, timestamp,
and CGM glucose.  CGMacros fasting insulin is the discovery endpoint; the Hall
``diagnosis == 0`` SSPG cohort is a separate, untouched validation endpoint.

Formal candidates are deliberately white-box and client-portable:

* low-entropy nocturnal anchor from the bottom 1% of fixed-opportunity
  90-minute windows (MAD + five-minute-normalized total variation + jump
  penalty), using at most the first six eligible nights;
* nocturnal anchor dwell and Theil-Sen drift;
* causally triggered excursion rise/fall asymmetry;
* next-night anchor carry-over after high versus low daily excursion burden.

Previously tested recovery debt is recomputed only as a historical benchmark.
All inference is at subject level.  Overlapping windows, nights, events and the
two CGMacros sensors are never treated as independent observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "raw_data.zip"
CGM_ROOT = ROOT / "output" / "cgmacros_subset"
OUTPUT_DIR = ROOT / "output"
RESULT_PATH = OUTPUT_DIR / "agent_nonclassic_results.json"
HALL_FEATURE_PATH = OUTPUT_DIR / "agent_nonclassic_hall_features.csv"
CGM_FEATURE_PATH = OUTPUT_DIR / "agent_nonclassic_cgmacros_features.csv"

SEED = 20260810
MGDL_PER_MMOL = 18.0182
MAX_NIGHTS = 6
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6
WINDOW_MINUTES = 90
WINDOW_STEP_MINUTES = 15
MIN_NIGHT_COVERAGE = 0.80
MIN_WINDOW_COVERAGE = 0.80
LOW_ENTROPY_FRACTION = 0.01
PHYSICAL_SLOPE_LIMIT = 0.30  # mmol/L/min; penalty, not a disease threshold
ANCHOR_ENVELOPE = 0.30  # mmol/L; engineering persistence envelope
EVENT_BASELINE_MINUTES = 30
EVENT_TRIGGER_LOOKBACK = 15
EVENT_TRIGGER_DELTA = 0.30  # mmol/L over the lookback
EVENT_TRIGGER_ABOVE_BASELINE = 0.25
EVENT_MIN_AMPLITUDE = 0.80
EVENT_PEAK_HORIZON = 120
EVENT_RECOVERY_HORIZON = 180
EVENT_COOLDOWN_MINUTES = 180
MIN_EVENTS = 3
N_DISCOVERY_PERMUTATIONS = 5000
N_HALL_PERMUTATIONS = 10000
N_BOOTSTRAP = 5000

CANDIDATE_GROUPS = {
    "anchor": [
        "anchor_level",
        "anchor_entropy",
        "anchor_dwell",
        "anchor_drift",
        "anchor_rise_fraction",
    ],
    "kinetics": [
        "recovery_asymmetry_q75",
        "next_night_carryover",
    ],
}
CANDIDATES = [name for names in CANDIDATE_GROUPS.values() for name in names]
HISTORICAL_BENCHMARKS = ["night_mean", "recovery_debt_q75"]

PROTOCOL = {
    "frozen_date": "2026-08-10",
    "candidate_inputs": ["subject_id", "timestamp", "CGM glucose"],
    "clinical_fields_are_endpoints_only": True,
    "natural_state": "00:00-06:00 local clock; first six nights with >=80% coverage",
    "low_entropy_window": {
        "duration_minutes": WINDOW_MINUTES,
        "stride_minutes": WINDOW_STEP_MINUTES,
        "score": "MAD(glucose) + median_abs_5min_change + 2*jump_fraction",
        "jump_penalty_threshold_mmol_l_min": PHYSICAL_SLOPE_LIMIT,
        "selected_fraction": LOW_ENTROPY_FRACTION,
    },
    "primary_discovery_endpoint": "CGMacros log1p(fasting insulin)",
    "primary_validation_endpoints": [
        "continuous SSPG in Hall diagnosis=0 complete cases",
        "SSPG>=150 mg/dL in Hall diagnosis=0 complete cases",
    ],
    "validation_unit": "subject",
    "formal_hall_candidate": "single scalar selected only in CGMacros; direction frozen before Hall endpoints",
    "binary_auc_semantics": "pairwise concordance across 9x19 positive-negative pairs",
    "threshold_protocol": "leave-one-subject-out; training-only threshold with sensitivity>=0.80 then maximum specificity",
    "notes": [
        "SSPG measures insulin resistance/glucose disposal, not fasting insulin concentration.",
        "CGMacros one-minute rows are interpolated; Libre is downsampled to 15 min and Dexcom to 5 min.",
        "Means/SD are not used to define health; nightMean is a comparison baseline only.",
        "Short-record drift is a compensation-fatigue proxy, not longitudinal collapse prediction.",
    ],
}


def clean_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(float)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def roc_auc(y, scores) -> float:
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    keep = np.isfinite(scores)
    y, scores = y[keep], scores[keep]
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, scores) -> float:
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    keep = np.isfinite(scores)
    y, scores = y[keep], scores[keep]
    order = np.argsort(-scores, kind="mergesort")
    ranked_y = y[order]
    n_pos = int(ranked_y.sum())
    if n_pos == 0:
        return float("nan")
    precision = np.cumsum(ranked_y) / np.arange(1, len(ranked_y) + 1)
    return float(precision[ranked_y == 1].sum() / n_pos)


def mad(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def theil_sen(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[j] - x[i]) > 1e-12:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else float("nan")


def quantile(values, q: float) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("nan")


def restore_long_gaps(series: pd.Series, max_gap_points: int) -> pd.Series:
    """Interpolate only bounded gaps no longer than ``max_gap_points``."""
    series = series.astype(float).copy()
    missing = series.isna().to_numpy()
    interpolated = series.interpolate(method="linear", limit_area="inside")
    start = 0
    while start < len(missing):
        if not missing[start]:
            start += 1
            continue
        end = start
        while end < len(missing) and missing[end]:
            end += 1
        bounded = start > 0 and end < len(missing)
        if not bounded or end - start > max_gap_points:
            interpolated.iloc[start:end] = np.nan
        start = end
    return interpolated


def regularize(frame: pd.DataFrame, freq_minutes: int) -> pd.Series:
    data = frame[["time", "gl_mmol"]].dropna().copy()
    data = data.sort_values("time").drop_duplicates("time", keep="last")
    if data.empty:
        return pd.Series(dtype=float)
    series = data.set_index("time")["gl_mmol"].resample(f"{freq_minutes}min").first()
    series = restore_long_gaps(series, max(1, 15 // freq_minutes))
    return series


def window_record(values: np.ndarray, freq_minutes: int, date, start_minute: int) -> dict:
    center = float(np.median(values))
    dispersion = mad(values)
    diffs = np.diff(values)
    five_minute_change = float(np.median(np.abs(diffs)) * 5.0 / freq_minutes)
    slope = np.abs(diffs) / freq_minutes
    violation = float(np.mean(slope > PHYSICAL_SLOPE_LIMIT)) if len(slope) else 0.0
    entropy = dispersion + five_minute_change + 2.0 * violation
    return {
        "date": pd.Timestamp(date),
        "start_minute": int(start_minute),
        "median": center,
        "mad": dispersion,
        "five_minute_change": five_minute_change,
        "jump_fraction": violation,
        "entropy": entropy,
    }


def nocturnal_windows(series: pd.Series, freq_minutes: int) -> tuple[list[dict], list[pd.Timestamp]]:
    if series.empty:
        return [], []
    records: list[dict] = []
    eligible_dates = []
    dates = sorted({stamp.normalize() for stamp in series.index})
    expected_night = int((NIGHT_END_HOUR - NIGHT_START_HOUR) * 60 / freq_minutes)
    expected_window = int(WINDOW_MINUTES / freq_minutes)
    for date in dates:
        start = date + pd.Timedelta(hours=NIGHT_START_HOUR)
        end = date + pd.Timedelta(hours=NIGHT_END_HOUR)
        night = series[(series.index >= start) & (series.index < end)]
        if night.notna().sum() < math.ceil(MIN_NIGHT_COVERAGE * expected_night):
            continue
        eligible_dates.append(date)
        for start_minute in range(0, 360 - WINDOW_MINUTES + 1, WINDOW_STEP_MINUTES):
            w_start = start + pd.Timedelta(minutes=start_minute)
            w_end = w_start + pd.Timedelta(minutes=WINDOW_MINUTES)
            values = night[(night.index >= w_start) & (night.index < w_end)].dropna().to_numpy(float)
            if len(values) < math.ceil(MIN_WINDOW_COVERAGE * expected_window):
                continue
            records.append(window_record(values, freq_minutes, date, start_minute))
    eligible_dates = eligible_dates[:MAX_NIGHTS]
    allowed = set(eligible_dates)
    return [row for row in records if row["date"] in allowed], eligible_dates


def aggregate_anchor(windows: list[dict], dates: list[pd.Timestamp], series: pd.Series) -> dict:
    date_set = set(dates)
    rows = [row for row in windows if row["date"] in date_set]
    if not rows:
        return {name: np.nan for name in (
            "anchor_level", "anchor_entropy", "anchor_dwell", "anchor_drift", "anchor_rise_fraction"
        )}
    selected_n = max(1, int(math.ceil(LOW_ENTROPY_FRACTION * len(rows))))
    selected = sorted(rows, key=lambda row: (row["entropy"], row["date"], row["start_minute"]))[:selected_n]
    anchor_level = float(np.median([row["median"] for row in selected]))
    anchor_entropy = float(np.median([row["entropy"] for row in selected]))

    night_values = []
    nightly_anchor = []
    for date in dates:
        day_rows = [row for row in rows if row["date"] == date]
        if not day_rows:
            continue
        best = min(day_rows, key=lambda row: (row["entropy"], row["start_minute"]))
        nightly_anchor.append((date, best["median"]))
        start = date + pd.Timedelta(hours=NIGHT_START_HOUR)
        end = date + pd.Timedelta(hours=NIGHT_END_HOUR)
        night_values.extend(series[(series.index >= start) & (series.index < end)].dropna().to_numpy(float))
    night_values = np.asarray(night_values, float)
    anchor_dwell = float(np.mean(np.abs(night_values - anchor_level) <= ANCHOR_ENVELOPE)) if len(night_values) else np.nan
    if len(nightly_anchor) >= 4:
        x = np.arange(len(nightly_anchor), dtype=float)
        y = np.array([row[1] for row in nightly_anchor], dtype=float)
        drift = theil_sen(x, y)
        rise_fraction = float(np.mean(np.diff(y) > 0))
    else:
        drift, rise_fraction = np.nan, np.nan
    return {
        "anchor_level": anchor_level,
        "anchor_entropy": anchor_entropy,
        "anchor_dwell": anchor_dwell,
        "anchor_drift": drift,
        "anchor_rise_fraction": rise_fraction,
    }


def detect_events(series: pd.Series, freq_minutes: int) -> list[dict]:
    if series.empty:
        return []
    events = []
    cooldown_until = series.index.min()
    lookback_steps = max(1, EVENT_TRIGGER_LOOKBACK // freq_minutes)
    baseline_steps = max(2, EVENT_BASELINE_MINUTES // freq_minutes)
    peak_steps = max(1, EVENT_PEAK_HORIZON // freq_minutes)
    recovery_steps = max(1, EVENT_RECOVERY_HORIZON // freq_minutes)
    consecutive = max(1, 15 // freq_minutes)
    values = series.to_numpy(float)
    times = series.index
    for i in range(max(baseline_steps, lookback_steps), len(series) - peak_steps - recovery_steps):
        time = times[i]
        if time < cooldown_until or not (6 <= time.hour < 22) or not np.isfinite(values[i]):
            continue
        baseline_values = values[i - baseline_steps:i]
        if np.isfinite(baseline_values).sum() < math.ceil(0.8 * baseline_steps):
            continue
        baseline = float(np.nanmedian(baseline_values))
        past = values[i - lookback_steps]
        if not np.isfinite(past):
            continue
        if values[i] - past < EVENT_TRIGGER_DELTA or values[i] - baseline < EVENT_TRIGGER_ABOVE_BASELINE:
            continue
        future = values[i:i + peak_steps + 1]
        if np.isfinite(future).sum() < math.ceil(0.8 * len(future)):
            continue
        peak_offset = int(np.nanargmax(future))
        peak_value = float(future[peak_offset])
        amplitude = peak_value - baseline
        if amplitude < EVENT_MIN_AMPLITUDE:
            continue
        peak_i = i + peak_offset
        threshold = baseline + 0.25 * amplitude
        recovery_slice = values[peak_i:peak_i + recovery_steps + 1]
        recovery_minutes = EVENT_RECOVERY_HORIZON
        recovered = False
        for j in range(0, max(0, len(recovery_slice) - consecutive + 1)):
            block = recovery_slice[j:j + consecutive]
            if len(block) == consecutive and np.all(np.isfinite(block)) and np.all(block <= threshold):
                recovery_minutes = j * freq_minutes
                recovered = True
                break
        x_minutes = np.arange(len(recovery_slice), dtype=float) * freq_minutes
        excess = np.maximum(np.nan_to_num(recovery_slice, nan=threshold) - threshold, 0.0)
        debt = float(np.trapezoid(excess, x=x_minutes) / max(amplitude, 1e-6))
        time_to_peak = max(freq_minutes, peak_offset * freq_minutes)
        residual = float((recovery_slice[-1] - baseline) / amplitude) if np.isfinite(recovery_slice[-1]) else np.nan
        events.append({
            "start": time,
            "date": time.normalize(),
            "baseline": baseline,
            "amplitude": amplitude,
            "time_to_peak": time_to_peak,
            "recovery_minutes": recovery_minutes,
            "recovered": recovered,
            "recovery_debt": debt,
            "recovery_asymmetry": recovery_minutes / time_to_peak,
            "residual_180": residual,
            "burden": debt * amplitude,
        })
        cooldown_until = time + pd.Timedelta(minutes=EVENT_COOLDOWN_MINUTES)
    return events


def event_summary(events: list[dict], nightly_anchor: dict[pd.Timestamp, float]) -> dict:
    if len(events) < MIN_EVENTS:
        return {
            "recovery_debt_q75": np.nan,
            "recovery_asymmetry_q75": np.nan,
            "next_night_carryover": np.nan,
            "event_count": len(events),
        }
    debt = quantile([row["recovery_debt"] for row in events], 0.75)
    asymmetry = quantile([row["recovery_asymmetry"] for row in events], 0.75)
    daily_burden: dict[pd.Timestamp, float] = {}
    for row in events:
        daily_burden[row["date"]] = daily_burden.get(row["date"], 0.0) + row["burden"]
    pairs = []
    for date, burden in daily_burden.items():
        next_date = date + pd.Timedelta(days=1)
        if next_date in nightly_anchor:
            pairs.append((burden, nightly_anchor[next_date]))
    carryover = np.nan
    if len(pairs) >= 4:
        burdens = np.array([row[0] for row in pairs], float)
        anchors = np.array([row[1] for row in pairs], float)
        split = np.median(burdens)
        low, high = anchors[burdens <= split], anchors[burdens > split]
        if len(low) >= 2 and len(high) >= 2:
            carryover = float(np.median(high) - np.median(low))
    return {
        "recovery_debt_q75": debt,
        "recovery_asymmetry_q75": asymmetry,
        "next_night_carryover": carryover,
        "event_count": len(events),
    }


def extract_features(frame: pd.DataFrame, freq_minutes: int, meal_times=None) -> tuple[dict, dict]:
    series = regularize(frame, freq_minutes)
    windows, dates = nocturnal_windows(series, freq_minutes)
    anchor = aggregate_anchor(windows, dates, series)
    nightly_anchor = {}
    for date in dates:
        day_rows = [row for row in windows if row["date"] == date]
        if day_rows:
            nightly_anchor[date] = float(min(day_rows, key=lambda row: row["entropy"])["median"])
    events = detect_events(series, freq_minutes)
    summary = {**anchor, **event_summary(events, nightly_anchor)}
    night_mask = (series.index.hour >= NIGHT_START_HOUR) & (series.index.hour < NIGHT_END_HOUR)
    night_values = series[night_mask].dropna().to_numpy(float)
    summary.update({
        "night_mean": float(np.mean(night_values)) if len(night_values) else np.nan,
        "valid_nights": len(dates),
        "valid_points": int(series.notna().sum()),
        "record_days": float((series.index.max() - series.index.min()).total_seconds() / 86400) if len(series) else 0.0,
    })

    odd_dates, even_dates = dates[::2], dates[1::2]
    odd_anchor = aggregate_anchor(windows, odd_dates, series)
    even_anchor = aggregate_anchor(windows, even_dates, series)
    odd_events, even_events = events[::2], events[1::2]
    odd_event = event_summary(odd_events, nightly_anchor)
    even_event = event_summary(even_events, nightly_anchor)
    for name in CANDIDATES:
        summary[f"odd_{name}"] = odd_anchor.get(name, odd_event.get(name, np.nan))
        summary[f"even_{name}"] = even_anchor.get(name, even_event.get(name, np.nan))

    detector = {"events": len(events), "meals": 0, "matched_events": np.nan, "matched_meals": np.nan}
    if meal_times is not None:
        meals = sorted(pd.Timestamp(value) for value in meal_times if pd.notna(value))
        event_times = [row["start"] for row in events]
        matched_events = sum(
            any(meal - pd.Timedelta(minutes=15) <= event <= meal + pd.Timedelta(minutes=60) for meal in meals)
            for event in event_times
        )
        matched_meals = sum(
            any(meal - pd.Timedelta(minutes=15) <= event <= meal + pd.Timedelta(minutes=60) for event in event_times)
            for meal in meals
        )
        detector = {
            "events": len(events),
            "meals": len(meals),
            "matched_events": matched_events,
            "matched_meals": matched_meals,
        }
    return summary, detector


def load_cgmacros_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    bio = pd.read_csv(CGM_ROOT / "bio.csv")
    bio.columns = [str(column).strip() for column in bio.columns]
    bio = bio.rename(columns={
        "subject": "id",
        "A1c PDL (Lab)": "a1c",
        "Fasting GLU - PDL (Lab)": "fasting_glucose",
        "Insulin": "insulin",
        "Triglycerides": "triglycerides",
        "HDL": "hdl",
    })
    bio["id"] = pd.to_numeric(bio["id"], errors="raise").astype(int)
    rows = {"libre": [], "dexcom": []}
    detector_totals = {sensor: {"events": 0, "meals": 0, "matched_events": 0, "matched_meals": 0} for sensor in rows}
    for path in sorted((CGM_ROOT / "subjects").glob("CGMacros-*.csv")):
        subject_id = int(path.stem.split("-")[-1])
        data = pd.read_csv(path)
        data["time"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        meal_times = data.loc[data["Meal Type"].notna(), "time"].dropna().tolist()
        for sensor, column, frequency in (("libre", "Libre GL", 15), ("dexcom", "Dexcom GL", 5)):
            frame = pd.DataFrame({
                "time": data["time"],
                "gl_mmol": pd.to_numeric(data[column], errors="coerce") / MGDL_PER_MMOL,
            })
            features, detector = extract_features(frame, frequency, meal_times)
            rows[sensor].append({"id": subject_id, **features})
            for key in detector_totals[sensor]:
                detector_totals[sensor][key] += int(detector[key])
    libre, dexcom = pd.DataFrame(rows["libre"]), pd.DataFrame(rows["dexcom"])
    return bio, libre, dexcom, detector_totals


def load_hall_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    with ZipFile(RAW_ZIP) as archive:
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle)
    hall["time"] = pd.to_datetime(hall["time"], errors="coerce")
    hall["gl_mmol"] = pd.to_numeric(hall["gl"], errors="coerce") / MGDL_PER_MMOL
    feature_rows = []
    for subject_id, group in hall.groupby("id", sort=True):
        features, _ = extract_features(group[["time", "gl_mmol"]], 5)
        feature_rows.append({"id": str(subject_id), **features})
    features = pd.DataFrame(feature_rows)
    clinical = hall.groupby("id", sort=True).first().reset_index()
    clinical["id"] = clinical["id"].astype(str)
    return clinical, features


def discover_candidate(bio, libre, dexcom, hall_features) -> dict:
    left = bio.merge(libre, on="id", how="inner", suffixes=("", "_libre"))
    both = left.merge(dexcom, on="id", how="inner", suffixes=("_libre", "_dexcom"))
    outcome = np.log1p(pd.to_numeric(both["insulin"], errors="coerce").to_numpy(float))
    rows = []
    for name in CANDIDATES:
        x_l = pd.to_numeric(both[f"{name}_libre"], errors="coerce").to_numpy(float)
        x_d = pd.to_numeric(both[f"{name}_dexcom"], errors="coerce").to_numpy(float)
        keep = np.isfinite(outcome) & np.isfinite(x_l) & np.isfinite(x_d)
        rho_sensor = spearman(x_l[keep], x_d[keep])
        rho_l = spearman(x_l[keep], outcome[keep])
        rho_d = spearman(x_d[keep], outcome[keep])
        hall_calc = int(pd.to_numeric(hall_features[name], errors="coerce").notna().sum())
        same_direction = bool(np.isfinite(rho_l) and np.isfinite(rho_d) and rho_l * rho_d > 0)
        min_abs = min(abs(rho_l), abs(rho_d)) if same_direction else 0.0
        eligible = bool(
            keep.sum() >= 36 and rho_sensor >= 0.60 and same_direction and min_abs >= 0.10 and hall_calc >= 26
        )
        rows.append({
            "feature": name,
            "complete_both_sensors": int(keep.sum()),
            "sensor_spearman": rho_sensor,
            "insulin_spearman_libre": rho_l,
            "insulin_spearman_dexcom": rho_d,
            "same_direction": same_direction,
            "minimum_absolute_insulin_spearman": min_abs,
            "hall_calculable": hall_calc,
            "eligible": eligible,
        })
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: (row["minimum_absolute_insulin_spearman"], row["sensor_spearman"])) if eligible_rows else None
    redundancy = None
    if selected is not None:
        name = selected["feature"]
        redundancy = {
            "selected_vs_nightMean_libre_spearman": spearman(
                pd.to_numeric(both[f"{name}_libre"], errors="coerce"),
                pd.to_numeric(both["night_mean_libre"], errors="coerce"),
            ),
            "selected_vs_nightMean_dexcom_spearman": spearman(
                pd.to_numeric(both[f"{name}_dexcom"], errors="coerce"),
                pd.to_numeric(both["night_mean_dexcom"], errors="coerce"),
            ),
        }

    observed_max = max((row["minimum_absolute_insulin_spearman"] for row in rows), default=np.nan)
    rng = np.random.default_rng(SEED + 11)
    null = np.zeros(N_DISCOVERY_PERMUTATIONS, float)
    for b in range(N_DISCOVERY_PERMUTATIONS):
        perm = rng.permutation(outcome)
        best = 0.0
        for name in CANDIDATES:
            x_l = pd.to_numeric(both[f"{name}_libre"], errors="coerce").to_numpy(float)
            x_d = pd.to_numeric(both[f"{name}_dexcom"], errors="coerce").to_numpy(float)
            keep = np.isfinite(perm) & np.isfinite(x_l) & np.isfinite(x_d)
            if keep.sum() < 3:
                continue
            r_l, r_d = spearman(x_l[keep], perm[keep]), spearman(x_d[keep], perm[keep])
            if np.isfinite(r_l) and np.isfinite(r_d) and r_l * r_d > 0:
                best = max(best, min(abs(r_l), abs(r_d)))
        null[b] = best
    max_t_p = float((1 + np.sum(null >= observed_max)) / (N_DISCOVERY_PERMUTATIONS + 1))
    return {
        "feature_audit": rows,
        "selected": selected,
        "redundancy_audit": redundancy,
        "selection_adjusted_maxT": {
            "observed_max_minimum_absolute_sensor_replicated_spearman": observed_max,
            "p_value": max_t_p,
            "permutations": N_DISCOVERY_PERMUTATIONS,
        },
    }


def bootstrap_spearman(x, y, replicates=N_BOOTSTRAP, seed=SEED + 20) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = spearman(x, y)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, float)
    for b in range(replicates):
        idx = rng.integers(0, len(x), len(x))
        values[b] = spearman(x[idx], y[idx])
    values = values[np.isfinite(values)]
    return {
        "estimate": observed,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "replicates_valid": int(len(values)),
    }


def permutation_spearman(x, y, replicates=N_HALL_PERMUTATIONS, seed=SEED + 30) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = spearman(x, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(replicates):
        if abs(spearman(x, rng.permutation(y))) >= abs(observed):
            count += 1
    return {"estimate": observed, "two_sided_p": (count + 1) / (replicates + 1), "permutations": replicates}


def stratified_auc_bootstrap(y, candidate, baseline, replicates=N_BOOTSTRAP, seed=SEED + 40) -> dict:
    y, candidate, baseline = np.asarray(y, int), np.asarray(candidate, float), np.asarray(baseline, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    observed_auc = roc_auc(y, candidate)
    observed_delta = observed_auc - roc_auc(y, baseline)
    rng = np.random.default_rng(seed)
    aucs, deltas = np.empty(replicates), np.empty(replicates)
    for b in range(replicates):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        aucs[b] = roc_auc(y[idx], candidate[idx])
        deltas[b] = aucs[b] - roc_auc(y[idx], baseline[idx])
    return {
        "auc": {"estimate": observed_auc, "ci95": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))]},
        "delta_auc_vs_nightMean": {"estimate": observed_delta, "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]},
        "replicates": replicates,
    }


def permutation_auc(y, candidate, baseline, replicates=N_HALL_PERMUTATIONS, seed=SEED + 50) -> dict:
    y, candidate, baseline = np.asarray(y, int), np.asarray(candidate, float), np.asarray(baseline, float)
    observed_auc = roc_auc(y, candidate)
    observed_delta = observed_auc - roc_auc(y, baseline)
    rng = np.random.default_rng(seed)
    count_auc = count_delta = 0
    for _ in range(replicates):
        perm = rng.permutation(y)
        p_auc = roc_auc(perm, candidate)
        p_delta = p_auc - roc_auc(perm, baseline)
        count_auc += p_auc >= observed_auc
        count_delta += p_delta >= observed_delta
    return {
        "auc_gt_random_one_sided_p": (count_auc + 1) / (replicates + 1),
        "delta_auc_vs_nightMean_one_sided_p": (count_delta + 1) / (replicates + 1),
        "permutations": replicates,
    }


def choose_threshold(y, scores, min_sensitivity=0.80) -> float:
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    candidates = np.unique(np.r_[-np.inf, scores, np.inf])
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fn = int(np.sum((~predicted) & (y == 1)))
        tn = int(np.sum((~predicted) & (y == 0)))
        fp = int(np.sum(predicted & (y == 0)))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        if sensitivity >= min_sensitivity:
            key = (specificity, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold))
    return best[1] if best else -np.inf


def binomial_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def binomial_survival(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def clopper_pearson(k: int, n: int, alpha=0.05) -> list[float]:
    if n == 0:
        return [np.nan, np.nan]
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, k / n
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_survival(k, n, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.0
    else:
        lo, hi = k / n, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_cdf(k, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return [float(lower), float(upper)]


def loso_threshold_metrics(y, scores) -> dict:
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    decisions = np.zeros(len(y), bool)
    thresholds = np.empty(len(y), float)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        thresholds[i] = choose_threshold(y[keep], scores[keep], 0.80)
        decisions[i] = scores[i] >= thresholds[i]
    tp = int(np.sum(decisions & (y == 1)))
    fn = int(np.sum((~decisions) & (y == 1)))
    tn = int(np.sum((~decisions) & (y == 0)))
    fp = int(np.sum(decisions & (y == 0)))
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / (tp + fn),
        "sensitivity_ci95_exact": clopper_pearson(tp, tp + fn),
        "specificity": tn / (tn + fp),
        "specificity_ci95_exact": clopper_pearson(tn, tn + fp),
        "threshold_median": float(np.median(thresholds[np.isfinite(thresholds)])),
    }


def robust_missingness_audit(clinical: pd.DataFrame, features: pd.DataFrame) -> dict:
    merged = clinical.merge(features[["id", "record_days", "valid_points"]], on="id", how="left")
    normal = merged[pd.to_numeric(merged["diagnosis"], errors="coerce") == 0].copy()
    normal["sspg_valid"] = pd.to_numeric(normal["SSPG"], errors="coerce") >= 0
    variables = ["Age", "BMI", "A1C", "FBG", "ogtt.2hr", "record_days", "valid_points"]
    rows = {}
    for name in variables:
        included = pd.to_numeric(normal.loc[normal["sspg_valid"], name], errors="coerce").dropna().to_numpy(float)
        missing = pd.to_numeric(normal.loc[~normal["sspg_valid"], name], errors="coerce").dropna().to_numpy(float)
        scale = np.nanmedian([mad(included), mad(missing)])
        difference = float(np.median(included) - np.median(missing)) if len(included) and len(missing) else np.nan
        rows[name] = {
            "included_n": len(included), "missing_n": len(missing),
            "included_median": float(np.median(included)) if len(included) else np.nan,
            "missing_median": float(np.median(missing)) if len(missing) else np.nan,
            "robust_standardized_median_difference": difference / (1.4826 * scale) if scale and np.isfinite(scale) else np.nan,
        }
    return {"diagnosis0_sspg_complete": int(normal["sspg_valid"].sum()), "diagnosis0_sspg_missing": int((~normal["sspg_valid"]).sum()), "variables": rows}


def hall_validation(clinical, features, discovery) -> dict:
    merged = clinical.merge(features, on="id", how="inner")
    normal = merged[pd.to_numeric(merged["diagnosis"], errors="coerce") == 0].copy()
    normal["sspg"] = pd.to_numeric(normal["SSPG"], errors="coerce")
    normal = normal[normal["sspg"] >= 0].copy().reset_index(drop=True)
    selected = discovery["selected"]
    if selected is None:
        return {"n": len(normal), "selected_feature": None, "reason": "No CGMacros candidate passed frozen replication/calculability gates"}
    feature = selected["feature"]
    direction = 1.0 if selected["insulin_spearman_libre"] + selected["insulin_spearman_dexcom"] > 0 else -1.0
    normal["candidate_score"] = direction * pd.to_numeric(normal[feature], errors="coerce")
    normal["night_score"] = pd.to_numeric(normal["night_mean"], errors="coerce")
    valid = normal[["candidate_score", "night_score", "sspg"]].notna().all(axis=1)
    analysis = normal[valid].copy().reset_index(drop=True)
    y_cont = analysis["sspg"].to_numpy(float)
    y = (y_cont >= 150).astype(int)
    candidate = analysis["candidate_score"].to_numpy(float)
    baseline = analysis["night_score"].to_numpy(float)
    v84_payload = json.loads((OUTPUT_DIR / "v84_expected.json").read_text(encoding="utf-8"))
    v84_by_id = v84_payload["hall"]["probability_by_id"]
    v84_score = np.array([clean_number(v84_by_id.get(subject_id)) for subject_id in analysis["id"]], float)
    binary = {
        "n": len(analysis), "positive": int(y.sum()), "negative": int((1-y).sum()),
        "candidate_auc_pairwise_9x19": roc_auc(y, candidate),
        "candidate_pr_auc": average_precision(y, candidate),
        "nightMean_auc": roc_auc(y, baseline),
        "current_v84_auc": roc_auc(y, v84_score),
        "delta_auc_vs_nightMean": roc_auc(y, candidate) - roc_auc(y, baseline),
        "delta_auc_vs_current_v84": roc_auc(y, candidate) - roc_auc(y, v84_score),
        "bootstrap": stratified_auc_bootstrap(y, candidate, baseline),
        "permutation": permutation_auc(y, candidate, baseline),
        "loso_threshold": loso_threshold_metrics(y, candidate),
    }
    continuous = {
        "candidate_vs_sspg": bootstrap_spearman(candidate, y_cont),
        "candidate_vs_sspg_permutation": permutation_spearman(candidate, y_cont),
        "nightMean_vs_sspg": bootstrap_spearman(baseline, y_cont, seed=SEED + 21),
        "candidate_vs_nightMean_spearman": spearman(candidate, baseline),
    }
    odd = direction * pd.to_numeric(analysis[f"odd_{feature}"], errors="coerce").to_numpy(float)
    even = direction * pd.to_numeric(analysis[f"even_{feature}"], errors="coerce").to_numpy(float)
    reliability = bootstrap_spearman(odd, even, seed=SEED + 22)
    cross_domain = {}
    for index, (name, raw) in enumerate((
        ("fasting_insulin", pd.to_numeric(analysis["insulin"], errors="coerce").to_numpy(float)),
        ("hs_crp", pd.to_numeric(analysis["hs.CRP"], errors="coerce").to_numpy(float)),
        ("triglyceride_hdl_ratio", pd.to_numeric(analysis["Trg"], errors="coerce").to_numpy(float) / pd.to_numeric(analysis["HDL"], errors="coerce").to_numpy(float)),
    )):
        cross_domain[name] = permutation_spearman(candidate, raw, replicates=5000, seed=SEED + 60 + index)
    positive_domains = sum(row["estimate"] > 0 for row in cross_domain.values() if np.isfinite(row["estimate"]))
    significant_reverse = any(row["estimate"] < 0 and row["two_sided_p"] <= 0.05 for row in cross_domain.values())
    bootstrap_gate = binary["bootstrap"]
    threshold = binary["loso_threshold"]
    gates = {
        "cgmacros_selection_maxT_p_lte_0_05": discovery["selection_adjusted_maxT"]["p_value"] <= 0.05,
        "hall_calculable_gte_26": len(analysis) >= 26,
        "auc_gte_0_70": binary["candidate_auc_pairwise_9x19"] >= 0.70,
        "auc_ci_lower_gt_0_50": bootstrap_gate["auc"]["ci95"][0] > 0.50,
        "delta_auc_gte_0_05": binary["delta_auc_vs_nightMean"] >= 0.05,
        "delta_auc_ci_lower_gt_0": bootstrap_gate["delta_auc_vs_nightMean"]["ci95"][0] > 0,
        "delta_auc_permutation_p_lte_0_05": binary["permutation"]["delta_auc_vs_nightMean_one_sided_p"] <= 0.05,
        "threshold_tp_gte_8_of_9": threshold["tp"] >= 8,
        "threshold_tn_gte_10_of_19": threshold["tn"] >= 10,
        "odd_even_reliability_gte_0_70": reliability["estimate"] >= 0.70,
        "continuous_sspg_positive_p_lte_0_05": continuous["candidate_vs_sspg_permutation"]["estimate"] > 0 and continuous["candidate_vs_sspg_permutation"]["two_sided_p"] <= 0.05,
        "cross_domain_at_least_2_of_3_positive_no_significant_reverse": positive_domains >= 2 and not significant_reverse,
    }
    return {
        "n": len(analysis), "positive": int(y.sum()), "negative": int((1-y).sum()),
        "selected_feature": feature, "direction_from_cgmacros": direction,
        "binary": binary, "continuous": continuous, "odd_even_reliability": reliability,
        "cross_domain_waste_heat": cross_domain,
        "gates": gates, "deployment_eligible": bool(all(gates.values())),
        "analysis_ids": analysis["id"].tolist(),
    }


def all_hall_exploratory_maxT(clinical, features) -> dict:
    merged = clinical.merge(features, on="id", how="inner")
    normal = merged[pd.to_numeric(merged["diagnosis"], errors="coerce") == 0].copy()
    normal["sspg"] = pd.to_numeric(normal["SSPG"], errors="coerce")
    normal = normal[normal["sspg"] >= 0].reset_index(drop=True)
    outcome = normal["sspg"].to_numpy(float)
    observed = {}
    matrices = {}
    for name in CANDIDATES:
        values = pd.to_numeric(normal[name], errors="coerce").to_numpy(float)
        matrices[name] = values
        observed[name] = abs(spearman(values, outcome))
    observed_max = max(value for value in observed.values() if np.isfinite(value))
    rng = np.random.default_rng(SEED + 70)
    count = 0
    for _ in range(N_HALL_PERMUTATIONS):
        perm = rng.permutation(outcome)
        maximum = max(abs(spearman(values, perm)) for values in matrices.values() if np.isfinite(spearman(values, perm)))
        count += maximum >= observed_max
    best = max(observed, key=lambda name: observed[name] if np.isfinite(observed[name]) else -1)
    return {
        "exploratory_only": True,
        "feature_absolute_spearman": observed,
        "best_feature": best,
        "observed_max_absolute_spearman": observed_max,
        "maxT_p_value": (count + 1) / (N_HALL_PERMUTATIONS + 1),
        "permutations": N_HALL_PERMUTATIONS,
    }


def detector_summary(totals: dict) -> dict:
    result = {}
    for sensor, row in totals.items():
        result[sensor] = {
            **row,
            "event_precision_within_minus15_plus60_of_meal": row["matched_events"] / row["events"] if row["events"] else np.nan,
            "meal_recall_with_event_minus15_plus60": row["matched_meals"] / row["meals"] if row["meals"] else np.nan,
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (CGM_ROOT / "manifest.json").exists():
        raise FileNotFoundError("Run validate/fetch_cgmacros_subset.py first")
    bio, libre, dexcom, detector = load_cgmacros_features()
    clinical, hall_features = load_hall_features()

    # Discovery is completed before any Hall endpoint enters selection.
    hall_dx0_ids = set(clinical.loc[pd.to_numeric(clinical["diagnosis"], errors="coerce") == 0, "id"])
    hall_calc_only = hall_features[hall_features["id"].isin(hall_dx0_ids)].copy()
    discovery = discover_candidate(bio, libre, dexcom, hall_calc_only)
    validation = hall_validation(clinical, hall_features, discovery)
    exploratory = all_hall_exploratory_maxT(clinical, hall_features)
    missingness = robust_missingness_audit(clinical, hall_features)

    combined_cgm = libre.add_suffix("_libre").rename(columns={"id_libre": "id"}).merge(
        dexcom.add_suffix("_dexcom").rename(columns={"id_dexcom": "id"}), on="id", how="outer"
    )
    combined_cgm.to_csv(CGM_FEATURE_PATH, index=False)
    hall_features.to_csv(HALL_FEATURE_PATH, index=False)

    results = {
        "protocol": PROTOCOL,
        "data": {
            "raw_data_zip_sha256": sha256(RAW_ZIP),
            "cgmacros_manifest_sha256": sha256(CGM_ROOT / "manifest.json"),
            "cgmacros_subjects": len(bio),
            "hall_subjects": len(clinical),
            "hall_diagnosis0": int((pd.to_numeric(clinical["diagnosis"], errors="coerce") == 0).sum()),
            "hall_diagnosis0_sspg_complete": missingness["diagnosis0_sspg_complete"],
        },
        "candidate_families": CANDIDATE_GROUPS,
        "historical_benchmarks": HISTORICAL_BENCHMARKS,
        "cgmacros_meal_detector": detector_summary(detector),
        "cgmacros_discovery": discovery,
        "hall_hidden_sspg_validation": validation,
        "hall_exploratory_all_candidate_maxT": exploratory,
        "hall_sspg_missingness_audit": missingness,
        "deployment_decision": (
            "update index.html" if validation.get("deployment_eligible") else
            "retain index.html; non-classic candidate did not pass every frozen gate"
        ),
        "limitations": [
            "Hall hidden endpoint has only 28 complete cases (9 SSPG>=150); all intervals are necessarily wide.",
            "Ten of 38 diagnosis-normal Hall participants lack SSPG; complete-case selection may bias transportability.",
            "CGMacros fasting insulin and Hall SSPG are related but distinct physiological endpoints.",
            "The Hall dataset is the only SSPG validation cohort; passing internal gates would still require independent confirmation.",
            "Fixed clock 00:00-06:00 cannot prove sleep or fasting, and short drift cannot establish long-term collapse.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(json_clean(results), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Agent non-classic validation ===")
    print(f"CGMacros n={len(bio)}; Hall n={len(clinical)}; dx0 SSPG complete={missingness['diagnosis0_sspg_complete']}")
    print(f"Selected in CGMacros: {discovery['selected']}")
    if validation.get("selected_feature"):
        binary = validation["binary"]
        print(
            f"Hall hidden SSPG: feature={validation['selected_feature']}, "
            f"AUC={binary['candidate_auc_pairwise_9x19']:.3f}, "
            f"nightMean={binary['nightMean_auc']:.3f}, delta={binary['delta_auc_vs_nightMean']:+.3f}"
        )
        print(f"Deployment eligible: {validation['deployment_eligible']}")
    else:
        print("No formal Hall candidate: discovery gates failed")
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
