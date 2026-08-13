#!/usr/bin/env python3
"""Round-2 label-blind reliability study for five frozen CGM metrics.

Only cohort/id/split/pairedDates/timestamps/values are copied from the source
export. Clinical fields present in the JSON are never loaded into analysis
records or written to outputs. The script does not edit index.html or raw data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "structure_reserve_windows.json"
OUT = ROOT / "output" / "cgm_metric_round2_five"
BASE_SCRIPT = ROOT / "validate" / "analyze_cgm_raw_metrics.py"
REPORT = ROOT / "reports" / "cgm_metric_round2_five_20260812.md"
SEED = 20260812

CANDIDATES = [
    "rate_mean_abs",
    "log_volume",
    "work_integral",
    "lyapunov_rosenstein",
    "permutation_entropy",
]
BASELINES = ["mean_glucose", "cv_pct", "tir_70_180"]
METRICS = BASELINES + CANDIDATES
LABELS = {
    "mean_glucose": "Mean glucose",
    "cv_pct": "CV",
    "tir_70_180": "TIR 70-180",
    "rate_mean_abs": "Mean absolute rate",
    "log_volume": "Log volume",
    "work_integral": "Work integral",
    "lyapunov_rosenstein": "Rosenstein proxy",
    "permutation_entropy": "Permutation entropy",
}
UNITS = {
    "mean_glucose": "mmol/L",
    "cv_pct": "%",
    "tir_70_180": "%",
    "rate_mean_abs": "mmol/L/hour",
    "log_volume": "log scale",
    "work_integral": "arbitrary",
    "lyapunov_rosenstein": "1/hour",
    "permutation_entropy": "normalized",
}
PROFILES = [
    ("grid5", 5, None),
    ("grid10", 10, None),
    ("grid15", 15, None),
    ("ema03", 5, 0.3),
]


def load_base_module():
    spec = importlib.util.spec_from_file_location("cgm_metric_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuity_ok(dates: list[pd.Timestamp]) -> bool:
    ordered = sorted(pd.Timestamp(item).normalize() for item in dates)
    gaps = [(ordered[i] - ordered[i - 1]).days for i in range(1, len(ordered))]
    return len(ordered) == 6 and (ordered[-1] - ordered[0]).days <= 7 and max(gaps, default=0) <= 2


def qualify_day(frame: pd.DataFrame, day) -> dict:
    local = frame[frame["time"].dt.date == day].dropna(subset=["glucose"]).copy()
    night = local[(local["time"].dt.hour >= 0) & (local["time"].dt.hour < 6)]
    daytime = local[(local["time"].dt.hour >= 6) & (local["time"].dt.hour < 18)]
    return {
        "date": str(day),
        "raw_points": int(len(local)),
        "night_points": int(len(night)),
        "daytime_points": int(len(daytime)),
        "qualified": bool(len(night) >= 48 and len(daytime) >= 72),
    }


def load_whitelisted_episodes() -> tuple[list[dict], pd.DataFrame]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_rows = payload.get("stateRecordsSixDay", [])
    episodes, qc_rows = [], []
    for source in source_rows:
        if source.get("split") != "full":
            continue
        # Explicit whitelist: clinical and all other fields are intentionally ignored.
        cohort = str(source.get("cohort", ""))
        subject_id = str(source.get("id", ""))
        paired_dates = [pd.Timestamp(item) for item in source.get("pairedDates", [])]
        timestamps = list(source.get("timestamps", []))
        values = list(source.get("values", []))
        if cohort not in {"hall", "weinstock"} or len(timestamps) != len(values) or not continuity_ok(paired_dates):
            continue
        frame = pd.DataFrame({
            "time": pd.to_datetime(timestamps, errors="coerce"),
            "glucose": pd.to_numeric(pd.Series(values), errors="coerce"),
        }).dropna(subset=["time"]).sort_values("time")
        dates = [item.date() for item in paired_dates]
        day_qc = [qualify_day(frame, day) for day in dates]
        for item in day_qc:
            qc_rows.append({"cohort": cohort, "subject_id": subject_id, **item})
        episodes.append({
            "cohort": cohort,
            "subject_id": subject_id,
            "paired_dates": dates,
            "frame": frame,
            "day_qc": day_qc,
        })
    return episodes, pd.DataFrame(qc_rows)


def selected_dates(episode: dict, nq: int, split: str) -> list:
    indices = [0, 2, 4] if split == "odd" else [1, 3, 5]
    return [episode["paired_dates"][index] for index in indices[:nq]]


def make_record(episode: dict, nq: int, split: str):
    dates = selected_dates(episode, nq, split)
    frame = episode["frame"]
    local = frame[frame["time"].dt.date.isin(dates)].copy()
    qc_lookup = {item["date"]: item for item in episode["day_qc"]}
    qualified = all(qc_lookup[str(day)]["qualified"] for day in dates)
    record = BASE.SeriesRecord(
        role="round2",
        cohort=episode["cohort"],
        subject_id=episode["subject_id"],
        timestamps=[item.isoformat() for item in local["time"]],
        values=local["glucose"].tolist(),
        window="24h" if nq == 1 else f"{nq}d_disjoint",
        split=split,
        device="cohort_native",
    )
    return record, dates, qualified


def compute_windows(episodes: list[dict], profiles: Iterable[tuple[str, int, float | None]], sensitivity_ids: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for episode_index, episode in enumerate(episodes, 1):
        key = f"{episode['cohort']}:{episode['subject_id']}"
        for nq in (1, 2, 3):
            for split in ("odd", "even"):
                record, dates, qualified = make_record(episode, nq, split)
                for profile, grid_minutes, ema_alpha in profiles:
                    if profile != "grid5" and nq != 3:
                        continue
                    if profile != "grid5" and sensitivity_ids is not None and key not in sensitivity_ids:
                        continue
                    result = BASE.compute_metric_row(record, grid_minutes=grid_minutes, ema_alpha=ema_alpha)
                    output = {
                        "cohort": episode["cohort"],
                        "subject_id": episode["subject_id"],
                        "nq": nq,
                        "split": split,
                        "profile": profile,
                        "observation_clock": "00:00-18:00 source paired window",
                        "grid_minutes": grid_minutes,
                        "ema_alpha": "" if ema_alpha is None else ema_alpha,
                        "selected_dates": "|".join(str(day) for day in dates),
                        "qualified_days": qualified,
                        "source_points": result.get("source_points"),
                        "valid_points": result.get("valid_points"),
                        "coverage": result.get("coverage"),
                        "tau_minutes": result.get("tau_minutes"),
                    }
                    output.update({metric: result.get(metric, float("nan")) for metric in METRICS})
                    rows.append(output)
        if episode_index % 20 == 0 or episode_index == len(episodes):
            print(f"computed episodes {episode_index}/{len(episodes)}", flush=True)
    return pd.DataFrame(rows)


def icc_components(pairs: np.ndarray) -> tuple[float, float]:
    pairs = np.asarray(pairs, dtype=float)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    return BASE.icc_a1(pairs)


def bootstrap_icc(pairs: np.ndarray, seed: int, repeats: int = 1000) -> tuple[float, float]:
    pairs = np.asarray(pairs, dtype=float)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if len(pairs) < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = pairs[rng.integers(0, len(pairs), len(pairs))]
        estimate, _ = icc_components(sample)
        if finite(estimate):
            estimates.append(estimate)
    return (float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)))


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def reliability_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_primary = primary[primary["profile"] == "grid5"].copy()
    eligible = all_primary[all_primary["qualified_days"]].copy()
    for cohort in ("hall", "weinstock"):
        for nq in (1, 2, 3):
            local = eligible[(eligible["cohort"] == cohort) & (eligible["nq"] == nq)]
            source_subjects = all_primary[(all_primary["cohort"] == cohort) & (all_primary["nq"] == nq)]["subject_id"].nunique()
            split_counts = local.groupby("subject_id")["split"].nunique()
            eligible_ids = split_counts[split_counts == 2].index
            local = local[local["subject_id"].isin(eligible_ids)]
            for metric_index, metric in enumerate(METRICS):
                pivot = local.pivot_table(index="subject_id", columns="split", values=metric, aggfunc="first")
                total_subjects = len(eligible_ids)
                if not {"odd", "even"}.issubset(pivot.columns):
                    pairs = np.empty((0, 2))
                else:
                    pairs = pivot[["odd", "even"]].to_numpy(float)
                    pairs = pairs[np.isfinite(pairs).all(axis=1)]
                icc, mse = icc_components(pairs)
                ci_low, ci_high = bootstrap_icc(pairs, SEED + metric_index * 31 + nq * 7 + (0 if cohort == "hall" else 1000))
                difference = pairs[:, 1] - pairs[:, 0] if len(pairs) else np.asarray([])
                pair_mean = pairs.mean(axis=1) if len(pairs) else np.asarray([])
                proportional_slope = float(np.polyfit(pair_mean, difference, 1)[0]) if len(pairs) >= 3 and np.std(pair_mean) > 1e-12 else float("nan")
                proportional_r = float(np.corrcoef(pair_mean, difference)[0, 1]) if len(pairs) >= 3 and np.std(pair_mean) > 1e-12 and np.std(difference) > 1e-12 else float("nan")
                sem = math.sqrt(max(mse, 0.0)) if finite(mse) else float("nan")
                mdc95 = 1.96 * math.sqrt(2.0) * sem if finite(sem) else float("nan")
                median_abs = float(np.median(np.abs(pairs))) if len(pairs) else float("nan")
                rows.append({
                    "cohort": cohort,
                    "nq": nq,
                    "metric_id": metric,
                    "label": LABELS[metric],
                    "unit": UNITS[metric],
                    "source_episode_subjects": source_subjects,
                    "eligible_subjects": total_subjects,
                    "qc_pair_rate": total_subjects / source_subjects if source_subjects else float("nan"),
                    "n_pairs": len(pairs),
                    "compute_rate": len(pairs) / total_subjects if total_subjects else float("nan"),
                    "icc_a1": icc,
                    "icc_ci_low": ci_low,
                    "icc_ci_high": ci_high,
                    "spearman": BASE.spearman(pairs[:, 0], pairs[:, 1]) if len(pairs) else float("nan"),
                    "ccc": concordance_correlation(pairs[:, 0], pairs[:, 1]) if len(pairs) else float("nan"),
                    "mean_even_minus_odd": float(np.mean(difference)) if len(difference) else float("nan"),
                    "ba_lower_loa": float(np.mean(difference) - 1.96 * np.std(difference, ddof=1)) if len(difference) > 1 else float("nan"),
                    "ba_upper_loa": float(np.mean(difference) + 1.96 * np.std(difference, ddof=1)) if len(difference) > 1 else float("nan"),
                    "ba_proportional_slope": proportional_slope,
                    "ba_proportional_r": proportional_r,
                    "sem_agreement": sem,
                    "mdc95": mdc95,
                    "mdc95_pct_median_abs": 100.0 * mdc95 / median_abs if median_abs > 1e-9 else float("nan"),
                })
    return pd.DataFrame(rows)


def icc_improvement(primary: pd.DataFrame) -> pd.DataFrame:
    eligible = primary[(primary["profile"] == "grid5") & primary["qualified_days"]].copy()
    rows = []
    for cohort in ("hall", "weinstock"):
        for metric_index, metric in enumerate(METRICS):
            pivots = {}
            for nq in (2, 3):
                local = eligible[(eligible["cohort"] == cohort) & (eligible["nq"] == nq)]
                pivots[nq] = local.pivot_table(index="subject_id", columns="split", values=metric, aggfunc="first")
            common = sorted(set(pivots[2].index) & set(pivots[3].index))
            arrays = {}
            valid_subjects = []
            for subject_id in common:
                values = []
                ok = True
                for nq in (2, 3):
                    if not {"odd", "even"}.issubset(pivots[nq].columns):
                        ok = False
                        break
                    pair = pivots[nq].loc[subject_id, ["odd", "even"]].to_numpy(float)
                    if not np.isfinite(pair).all():
                        ok = False
                        break
                    values.append(pair)
                if ok:
                    valid_subjects.append(subject_id)
                    arrays.setdefault(2, []).append(values[0])
                    arrays.setdefault(3, []).append(values[1])
            pair2 = np.asarray(arrays.get(2, []), dtype=float)
            pair3 = np.asarray(arrays.get(3, []), dtype=float)
            icc2, _ = icc_components(pair2)
            icc3, _ = icc_components(pair3)
            rng = np.random.default_rng(SEED + metric_index * 37 + (0 if cohort == "hall" else 5000))
            deltas = []
            for _ in range(1000):
                if len(pair2) < 20:
                    break
                indices = rng.integers(0, len(pair2), len(pair2))
                boot2, _ = icc_components(pair2[indices])
                boot3, _ = icc_components(pair3[indices])
                if finite(boot2) and finite(boot3):
                    deltas.append(boot3 - boot2)
            rows.append({
                "cohort": cohort,
                "metric_id": metric,
                "n_common_subjects": len(valid_subjects),
                "icc_nq2": icc2,
                "icc_nq3": icc3,
                "delta_icc_3_minus_2": icc3 - icc2 if finite(icc2) and finite(icc3) else float("nan"),
                "delta_ci_low": float(np.quantile(deltas, 0.025)) if deltas else float("nan"),
                "delta_ci_high": float(np.quantile(deltas, 0.975)) if deltas else float("nan"),
                "bootstrap_probability_delta_positive": float(np.mean(np.asarray(deltas) > 0)) if deltas else float("nan"),
            })
    return pd.DataFrame(rows)


def deterministic_fold(cohort: str, subject_id: str, folds: int = 5) -> int:
    key = f"{cohort}:{subject_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % folds


def oof_predictions(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    local = frame[["cohort", "subject_id", "split", target] + BASELINES].dropna().copy()
    local["row_id"] = np.arange(len(local))
    predictions = np.full(len(local), np.nan)
    folds = np.asarray([deterministic_fold(row.cohort, row.subject_id) for row in local.itertuples()])
    features = local[BASELINES].to_numpy(float)
    design = np.column_stack([np.ones(len(local)), features])
    target_values = local[target].to_numpy(float)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if train.sum() < 20 or test.sum() == 0:
            continue
        coefficients, *_ = np.linalg.lstsq(design[train], target_values[train], rcond=None)
        predictions[test] = design[test] @ coefficients
    return local, predictions


def incremental_table(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = primary[(primary["profile"] == "grid5") & primary["qualified_days"] & (primary["nq"] == 3)].copy()
    summary_rows, residual_rows = [], []
    for metric in CANDIDATES:
        data, predictions = oof_predictions(local, metric)
        actual = data[metric].to_numpy(float)
        mask = np.isfinite(actual) & np.isfinite(predictions)
        denominator = float(np.sum((actual[mask] - actual[mask].mean()) ** 2)) if mask.any() else float("nan")
        r2 = 1.0 - float(np.sum((actual[mask] - predictions[mask]) ** 2)) / denominator if mask.sum() >= 20 and denominator > 0 else float("nan")
        data["oof_prediction"] = predictions
        data["residual"] = actual - predictions
        summary = {"metric_id": metric, "n_rows": int(mask.sum()), "oof_r2_from_mean_cv_tir": r2}
        for cohort in ("hall", "weinstock"):
            cohort_data = data[data["cohort"] == cohort]
            pivot = cohort_data.pivot_table(index="subject_id", columns="split", values="residual", aggfunc="first")
            if {"odd", "even"}.issubset(pivot.columns):
                pairs = pivot[["odd", "even"]].to_numpy(float)
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
            else:
                pairs = np.empty((0, 2))
            residual_icc, _ = icc_components(pairs)
            summary[f"residual_icc_{cohort}"] = residual_icc
            summary[f"residual_pairs_{cohort}"] = len(pairs)
        summary_rows.append(summary)
        residual_rows.append(data.assign(metric_id=metric))
    return pd.DataFrame(summary_rows), pd.concat(residual_rows, ignore_index=True)


def sensitivity_table(metrics: pd.DataFrame) -> pd.DataFrame:
    local = metrics[(metrics["nq"] == 3) & metrics["qualified_days"]].copy()
    rows = []
    for cohort in ("hall", "weinstock"):
        cohort_data = local[local["cohort"] == cohort]
        primary = cohort_data[cohort_data["profile"] == "grid5"]
        for profile, _, _ in PROFILES[1:]:
            comparison = cohort_data[cohort_data["profile"] == profile]
            merged = primary.merge(comparison, on=["cohort", "subject_id", "nq", "split"], suffixes=("_primary", "_comparison"))
            for metric in METRICS:
                a = pd.to_numeric(merged[f"{metric}_primary"], errors="coerce").to_numpy(float)
                b = pd.to_numeric(merged[f"{metric}_comparison"], errors="coerce").to_numpy(float)
                mask = np.isfinite(a) & np.isfinite(b)
                rows.append({
                    "cohort": cohort,
                    "profile": profile,
                    "metric_id": metric,
                    "n_pairs": int(mask.sum()),
                    "spearman": BASE.spearman(a[mask], b[mask]) if mask.sum() else float("nan"),
                    "median_relative_abs_difference": float(np.median(np.abs(b[mask] - a[mask]) / np.maximum(np.abs(a[mask]), 1e-6))) if mask.sum() else float("nan"),
                })
    return pd.DataFrame(rows)


def device_evidence() -> pd.DataFrame:
    prior = pd.read_csv(ROOT / "output" / "cgm_metric_research" / "agreement_summary.csv")
    return prior[(prior["context"] == "libre_vs_dexcom") & prior["metric_id"].isin(METRICS)].copy()


def metric_decisions(reliability: pd.DataFrame, improvements: pd.DataFrame, incremental: pd.DataFrame, sensitivity: pd.DataFrame, device: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for metric in CANDIDATES:
        rel3 = reliability[(reliability["nq"] == 3) & (reliability["metric_id"] == metric)]
        imp = improvements[improvements["metric_id"] == metric]
        inc = incremental[incremental["metric_id"] == metric].iloc[0]
        sens = sensitivity[sensitivity["metric_id"] == metric]
        dev = device[device["metric_id"] == metric]
        get_min = lambda series: float(pd.to_numeric(series, errors="coerce").min()) if len(series) else float("nan")
        get_max = lambda series: float(pd.to_numeric(series, errors="coerce").max()) if len(series) else float("nan")
        rows.append({
            "metric_id": metric,
            "minimum_nq3_compute_rate": get_min(rel3["compute_rate"]),
            "minimum_nq3_icc": get_min(rel3["icc_a1"]),
            "minimum_nq3_icc_ci_low": get_min(rel3["icc_ci_low"]),
            "minimum_delta_icc_3_minus_2": get_min(imp["delta_icc_3_minus_2"]),
            "minimum_sensitivity_spearman": get_min(sens["spearman"]),
            "oof_r2_from_mean_cv_tir": float(inc["oof_r2_from_mean_cv_tir"]),
            "minimum_residual_icc": min(float(inc["residual_icc_hall"]), float(inc["residual_icc_weinstock"])),
            "device_icc": float(dev["icc_a1"].iloc[0]) if len(dev) else float("nan"),
            "same_device_measurement_ready": False,
        })
    decisions = pd.DataFrame(rows)
    rate_index = decisions.index[decisions["metric_id"] == "rate_mean_abs"][0]
    rate = decisions.loc[rate_index]
    passes = {
        "compute_rate_ge_0_95": bool(rate["minimum_nq3_compute_rate"] >= 0.95),
        "icc_ge_0_60": bool(rate["minimum_nq3_icc"] >= 0.60),
        "icc_ci_low_ge_0_45": bool(rate["minimum_nq3_icc_ci_low"] >= 0.45),
        "delta_icc_ge_0_05": bool(rate["minimum_delta_icc_3_minus_2"] >= 0.05),
        "sensitivity_spearman_ge_0_90": bool(rate["minimum_sensitivity_spearman"] >= 0.90),
        "simple_oof_r2_lt_0_80": bool(rate["oof_r2_from_mean_cv_tir"] < 0.80),
        "residual_icc_ge_0_40": bool(rate["minimum_residual_icc"] >= 0.40),
    }
    same_device_ready = all(passes.values())
    decisions.loc[rate_index, "same_device_measurement_ready"] = same_device_ready
    decision = {
        "schema": "glucobench.cgm-round2-five.decision.v1",
        "analysis_date": "2026-08-12",
        "label_blind": True,
        "primary_metric": "rate_mean_abs",
        "same_device_gate_results": passes,
        "same_device_measurement_ready": same_device_ready,
        "cross_device_icc_ge_0_60": bool(rate["device_icc"] >= 0.60),
        "interpretation": "measurement reliability only; no health, diagnosis, or intervention-effect direction",
    }
    return decisions, decision


def save_svg(name: str, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        '<style>text{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;fill:#17212b}'
        '.title{font-size:20px;font-weight:700}.sub{font-size:12px;fill:#52606d}.label{font-size:11px}'
        '.grid{stroke:#d9e2ec;stroke-width:1}.axis{stroke:#7b8794;stroke-width:1}</style>'
        f'{body}</svg>'
    )
    (OUT / name).write_text(svg, encoding="utf-8")


def render_reliability(reliability: pd.DataFrame) -> None:
    width, height, left, top, plot_w, plot_h = 980, 560, 110, 85, 790, 390
    colors = {"hall": "#2563eb", "weinstock": "#f59e0b"}
    pieces = [
        '<text x="30" y="35" class="title">五项候选的合格日数—重复性曲线</text>',
        '<text x="30" y="58" class="sub">ICC(A,1)，Hall 与 Weinstock 分层；虚线为冻结门槛 0.60</text>',
    ]
    for tick in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = top + (1.0 - tick) / 1.2 * plot_h
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
        pieces.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="label">{tick:.1f}</text>')
    threshold_y = top + (1.0 - 0.6) / 1.2 * plot_h
    pieces.append(f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left+plot_w}" y2="{threshold_y:.1f}" stroke="#b91c1c" stroke-dasharray="6 5"/>')
    metric_spacing = plot_w / len(CANDIDATES)
    for metric_index, metric in enumerate(CANDIDATES):
        center = left + metric_spacing * (metric_index + 0.5)
        pieces.append(f'<text x="{center:.1f}" y="{height-42}" text-anchor="middle" class="label">{html.escape(LABELS[metric])}</text>')
        for cohort_index, cohort in enumerate(("hall", "weinstock")):
            local = reliability[(reliability["metric_id"] == metric) & (reliability["cohort"] == cohort)].sort_values("nq")
            points = []
            for row in local.itertuples():
                x = center + (row.nq - 2) * 34 + (cohort_index - 0.5) * 8
                y = top + (1.0 - row.icc_a1) / 1.2 * plot_h
                points.append((x, y))
            if points:
                pieces.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(" ".join(f"{x:.1f},{y:.1f}" for x, y in points), colors[cohort]))
                for x, y in points:
                    pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[cohort]}"/>')
    pieces.extend([
        f'<rect x="{left}" y="{height-18}" width="12" height="7" fill="{colors["hall"]}"/><text x="{left+18}" y="{height-11}" class="label">Hall</text>',
        f'<rect x="{left+80}" y="{height-18}" width="12" height="7" fill="{colors["weinstock"]}"/><text x="{left+98}" y="{height-11}" class="label">Weinstock</text>',
    ])
    save_svg("reliability_by_nq.svg", width, height, "".join(pieces))


def render_decisions(decisions: pd.DataFrame) -> None:
    width, height, left, top, row_h, plot_w = 980, 360, 245, 82, 45, 650
    pieces = [
        '<text x="30" y="35" class="title">五项候选的最弱三日 ICC 与双设备 ICC</text>',
        '<text x="30" y="58" class="sub">同一指标在不同语境下可能表现相反；红线为 0.60</text>',
    ]
    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + tick * plot_w
        pieces.append(f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-45}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height-26}" text-anchor="middle" class="label">{tick:.1f}</text>')
    pieces.append(f'<line x1="{left+0.6*plot_w:.1f}" y1="{top-12}" x2="{left+0.6*plot_w:.1f}" y2="{height-45}" stroke="#b91c1c" stroke-dasharray="6 5"/>')
    for index, row in enumerate(decisions.itertuples()):
        y = top + index * row_h
        pieces.append(f'<text x="{left-15}" y="{y+5}" text-anchor="end" class="label">{html.escape(LABELS[row.metric_id])}</text>')
        for value, color, offset in ((row.minimum_nq3_icc, "#0f766e", -6), (row.device_icc, "#7c3aed", 6)):
            if finite(value):
                pieces.append(f'<rect x="{left}" y="{y+offset-4}" width="{max(0,float(value))*plot_w:.1f}" height="8" fill="{color}"/>')
    pieces.extend([
        f'<rect x="{left}" y="{height-17}" width="12" height="7" fill="#0f766e"/><text x="{left+18}" y="{height-10}" class="label">队列内最弱三日 ICC</text>',
        f'<rect x="{left+180}" y="{height-17}" width="12" height="7" fill="#7c3aed"/><text x="{left+198}" y="{height-10}" class="label">双设备 ICC</text>',
    ])
    save_svg("candidate_context_icc.svg", width, height, "".join(pieces))


def write_manifest() -> None:
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "analysis_manifest.json")
    code_files = [BASE_SCRIPT, Path(__file__).resolve()]
    manifest = {
        "schema": "glucobench.cgm-round2-five.manifest.v1",
        "analysis_date": "2026-08-12",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "label_blind": True,
        "allowed_source_fields": ["cohort", "id", "split", "pairedDates", "timestamps", "values"],
        "clinical_fields_loaded": False,
        "observation_clock": "00:00-18:00 source paired window; not a full 24-hour day",
        "raw_data_modified": False,
        "index_html_modified": False,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "code": {str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in code_files},
        "report": ({"path": str(REPORT.relative_to(ROOT)).replace("\\", "/"), "bytes": REPORT.stat().st_size, "sha256": sha256(REPORT)} if REPORT.exists() else None),
        "index_html_sha256": sha256(ROOT / "index.html"),
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small smoke-test subset")
    parser.add_argument("--postprocess", action="store_true", help="Rebuild summaries from existing window_metrics_long.csv")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.postprocess:
        metrics = pd.read_csv(OUT / "window_metrics_long.csv")
        if "observation_clock" not in metrics.columns:
            metrics.insert(5, "observation_clock", "00:00-18:00 source paired window")
        day_qc = pd.read_csv(OUT / "day_quality.csv")
        print("postprocess rows", len(metrics), flush=True)
    else:
        episodes, day_qc = load_whitelisted_episodes()
        if args.quick:
            episodes = [row for row in episodes if row["cohort"] == "hall"][:8] + [row for row in episodes if row["cohort"] == "weinstock"][:12]
        sensitivity_ids = {
            f"{row['cohort']}:{row['subject_id']}"
            for row in episodes
            if row["cohort"] == "hall"
        }
        sensitivity_ids.update(
            f"weinstock:{row['subject_id']}"
            for row in [item for item in episodes if item["cohort"] == "weinstock"][: (12 if args.quick else 60)]
        )
        print("episodes", len(episodes), "sensitivity subjects", len(sensitivity_ids), flush=True)
        metrics = compute_windows(episodes, PROFILES, sensitivity_ids)
    primary = metrics[metrics["profile"] == "grid5"].copy()
    reliability = reliability_table(primary)
    improvements = icc_improvement(primary)
    incremental, residuals = incremental_table(primary)
    sensitivity = sensitivity_table(metrics)
    device = device_evidence()
    decisions, decision = metric_decisions(reliability, improvements, incremental, sensitivity, device)

    if not args.postprocess:
        write_csv(day_qc, OUT / "day_quality.csv")
    write_csv(metrics, OUT / "window_metrics_long.csv")
    write_csv(reliability, OUT / "reliability_by_nq.csv")
    write_csv(improvements, OUT / "icc_improvement_2_to_3_days.csv")
    write_csv(incremental, OUT / "incremental_summary.csv")
    write_csv(residuals, OUT / "incremental_residuals_long.csv")
    write_csv(sensitivity, OUT / "sensitivity_summary.csv")
    write_csv(device, OUT / "device_agreement_prior.csv")
    write_csv(decisions, OUT / "candidate_decisions.csv")
    (OUT / "selection_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    render_reliability(reliability)
    render_decisions(decisions)
    write_manifest()
    print(json.dumps(decision, ensure_ascii=False, default=json_default), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
