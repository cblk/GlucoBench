#!/usr/bin/env python3
"""Identify early, actionable order parameters for CGM sample robustness.

The analysis is intentionally about acquisition and output stability, not
clinical treatment response. Source archives, CGMacros files, and index.html
are read-only. The subject is always the statistical unit; repeated prefixes
are kept together during validation and bootstrap resampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
RESULT_PATH = OUTPUT / "early_robustness_order_results.json"
ROW_PATH = OUTPUT / "early_robustness_order_rows.csv"

PREFIX_SOURCE = OUTPUT / "dynamic_prefix_subjects.json"
PRIMITIVE_SOURCE = OUTPUT / "composite_abnormality_primitives.csv"
FORMULA_SOURCE = OUTPUT / "stability_base5_results.json"

SEED = 20260810 + 900
BOOTSTRAPS = 2000
RISK_TOLERANCE = 0.05
RISK_TOLERANCE_SENSITIVITY = (0.03, 0.05, 0.10)
NIGHT_EXPECTED_POINTS = 72
MIN_QUALIFIED_NIGHT_POINTS = 48
EXPECTED_INTERVAL_MINUTES = 5.0
QUALITY_GAP_DECAY_MINUTES = (30.0, 60.0, 120.0)
RIDGE_ALPHAS = (0.0, 0.1, 1.0, 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def expit(value):
    value = np.asarray(value, float)
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def safe_spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return np.nan
    ranked_a = pd.Series(a[keep]).rank(method="average").to_numpy(float)
    ranked_b = pd.Series(b[keep]).rank(method="average").to_numpy(float)
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def icc_absolute_agreement(a, b):
    matrix = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    n, k = matrix.shape
    if n < 3:
        return float("nan")
    grand = matrix.mean()
    row_mean = matrix.mean(axis=1)
    col_mean = matrix.mean(axis=0)
    ms_row = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    ms_col = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    residual = matrix - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n
    return float((ms_row - ms_error) / denominator) if abs(denominator) > 1e-12 else float("nan")


def group_splits(groups, n_splits: int, seed: int):
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, min(n_splits, len(shuffled)))
    for test_groups in folds:
        test = np.isin(groups, test_groups)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def risk_tier(probability: float) -> str:
    if probability >= 0.50:
        return "high"
    if probability >= 0.30:
        return "middle"
    return "low"


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def cluster_bootstrap(frame: pd.DataFrame, statistic, seed: int, draws=BOOTSTRAPS):
    rng = np.random.default_rng(seed)
    ids = frame["id"].astype(str).unique()
    id_values = frame["id"].astype(str).to_numpy()
    grouped_indices = {subject_id: np.flatnonzero(id_values == subject_id) for subject_id in ids}
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        indices = np.concatenate([grouped_indices[subject_id] for subject_id in sampled])
        current = frame.iloc[indices]
        estimate = statistic(current)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    return {
        "estimate": float(statistic(frame)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]
        if estimates else [None, None],
        "draws": len(estimates),
    }


def quality_features(record: dict) -> dict:
    frame = pd.DataFrame({
        "time": pd.to_datetime(record["timestamps"], errors="coerce"),
        "glucose": pd.to_numeric(pd.Series(record["values"]), errors="coerce"),
    }).dropna().sort_values("time").drop_duplicates("time", keep="last")
    if frame.empty:
        raise ValueError(f"empty prefix record: {record.get('cohort')} {record.get('id')}")

    times = frame["time"].to_numpy(dtype="datetime64[ns]")
    gaps = np.diff(times).astype("timedelta64[s]").astype(float) / 60.0
    slopes = np.abs(np.diff(frame["glucose"].to_numpy(float))) / np.maximum(gaps, 1e-9)
    jump_fraction = float(np.mean(slopes > 0.30)) if len(slopes) else 0.0
    span_minutes = float((frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds() / 60.0)
    expected_points = max(1.0, math.floor(span_minutes / EXPECTED_INTERVAL_MINUTES) + 1.0)
    overall_coverage = min(1.0, len(frame) / expected_points)

    night = frame[(frame["time"].dt.hour >= 0) & (frame["time"].dt.hour < 6)].copy()
    night["date"] = night["time"].dt.normalize()
    qualified = []
    for _, group in night.groupby("date", sort=True):
        count = int(len(group))
        if count < MIN_QUALIFIED_NIGHT_POINTS:
            continue
        night_gaps = group["time"].sort_values().diff().dt.total_seconds().dropna().to_numpy(float) / 60.0
        longest_gap = float(np.max(night_gaps)) if len(night_gaps) else EXPECTED_INTERVAL_MINUTES
        night_slopes = (
            np.abs(np.diff(group.sort_values("time")["glucose"].to_numpy(float)))
            / np.maximum(night_gaps, 1e-9)
        )
        night_jump = float(np.mean(night_slopes > 0.30)) if len(night_slopes) else 0.0
        coverage = min(1.0, count / NIGHT_EXPECTED_POINTS)
        qualities = {
            int(decay): coverage
            * math.exp(-max(0.0, longest_gap - EXPECTED_INTERVAL_MINUTES) / decay)
            * (1.0 - night_jump)
            for decay in QUALITY_GAP_DECAY_MINUTES
        }
        qualified.append({
            "count": count,
            "coverage": coverage,
            "longest_gap": longest_gap,
            "jump_fraction": night_jump,
            "mean": float(group["glucose"].mean()),
            "qualities": qualities,
        })

    if len(qualified) != int(record["prefixNights"]):
        raise AssertionError(
            f"qualified-night mismatch for {record.get('cohort')} {record.get('id')}: "
            f"derived={len(qualified)} declared={record['prefixNights']}"
        )
    means = np.asarray([row["mean"] for row in qualified], float)
    median = float(np.median(means))
    night_mad = float(np.median(np.abs(means - median))) if len(means) > 1 else 0.0
    result = {
        "qualified_nights": len(qualified),
        "night_coverage_mean": float(np.mean([row["coverage"] for row in qualified])),
        "night_coverage_min": float(np.min([row["coverage"] for row in qualified])),
        "night_longest_gap_minutes": float(np.max([row["longest_gap"] for row in qualified])),
        "night_jump_fraction": float(np.mean([row["jump_fraction"] for row in qualified])),
        "night_level_mad": night_mad,
        "overall_coverage": overall_coverage,
        "all_longest_gap_hours": float(np.max(gaps) / 60.0) if len(gaps) else 0.0,
        "all_jump_fraction": jump_fraction,
        "span_hours": span_minutes / 60.0,
        "valid_points": int(len(frame)),
    }
    for decay in QUALITY_GAP_DECAY_MINUTES:
        result[f"q_eff_{int(decay)}"] = float(sum(row["qualities"][int(decay)] for row in qualified))
    return result


def load_prefix_metrics(cohort: str, k: int) -> pd.DataFrame:
    path = OUTPUT / f"dynamic_prefix_metrics_{cohort}_k{k}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["id"] = frame["id"].astype(str)
    return frame


def load_full_metrics(cohort: str) -> pd.DataFrame:
    path = OUTPUT / f"phase_screening_metrics_{cohort}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["id"] = frame["id"].astype(str)
    return frame


def build_prefix_rows() -> pd.DataFrame:
    payload = json.loads(PREFIX_SOURCE.read_text(encoding="utf-8"))
    records = {}
    for cohort, ks in {"hall": (1, 2, 3, 5), "colas": (1, 2)}.items():
        for k in ks:
            records[(cohort, k)] = {
                str(record["id"]): quality_features(record)
                for record in payload[f"{cohort}_k{k}"]
            }

    metric_frames = {
        (cohort, k): load_prefix_metrics(cohort, k)
        for cohort, ks in {"hall": (1, 2, 3, 5), "colas": (1, 2)}.items()
        for k in ks
    }
    rows = []
    for cohort, prefix_ks in (("hall", (1, 2, 3, 5)), ("colas", (1, 2))):
        reference = load_full_metrics(cohort).set_index("id")
        previous_risk = {}
        for k in prefix_ks:
            current = metric_frames[(cohort, k)]
            for metric in current.to_dict("records"):
                subject_id = str(metric["id"])
                if subject_id not in reference.index:
                    continue
                ref = reference.loc[subject_id]
                current_risk = finite(metric.get("currentRisk"))
                reference_risk = finite(ref.get("currentRisk"))
                if not np.isfinite(current_risk) or not np.isfinite(reference_risk):
                    continue
                quality = records[(cohort, k)][subject_id]
                row = {
                    "analysis_type": "night_prefix",
                    "cohort": cohort,
                    "id": subject_id,
                    "prefix_nights": k,
                    "reference_window": "full_available_record",
                    "current_risk": current_risk,
                    "reference_risk": reference_risk,
                    "y": metric.get("y"),
                    "current_tier": risk_tier(current_risk),
                    "reference_tier": risk_tier(reference_risk),
                    "reference_boundary_distance": min(abs(reference_risk - 0.30), abs(reference_risk - 0.50)),
                    "risk_abs_error": abs(current_risk - reference_risk),
                    "log_risk_error": math.log(abs(current_risk - reference_risk) + 0.005),
                    "tier_agreement": risk_tier(current_risk) == risk_tier(reference_risk),
                    "night_mean_abs_error": abs(finite(metric.get("nightMean")) - finite(ref.get("nightMean"))),
                    "n_raw": int(metric.get("nRaw", quality["valid_points"])),
                    "n_resampled": int(metric.get("nResampled", 0)),
                    "log_n_raw": math.log1p(int(metric.get("nRaw", quality["valid_points"]))),
                    "log_n_resampled": math.log1p(int(metric.get("nResampled", 0))),
                    "score_drift": abs(current_risk - previous_risk.get(subject_id, current_risk))
                    if k > 1 else np.nan,
                    **quality,
                }
                for tolerance in RISK_TOLERANCE_SENSITIVITY:
                    row[f"stable_{tolerance:.2f}"] = bool(
                        row["risk_abs_error"] <= tolerance and row["tier_agreement"]
                    )
                rows.append(row)
            previous_risk = dict(zip(current["id"].astype(str), current["currentRisk"].astype(float)))
    return pd.DataFrame(rows)


MODEL_FAMILIES = {
    "qualified_nights": ("qualified_nights",),
    "raw_points": ("log_n_raw",),
    "q_eff": ("q_eff_60",),
    "q_eff_phase": ("q_eff_60", "log_n_resampled"),
    "quality_panel": (
        "q_eff_60", "log_n_resampled", "overall_coverage",
        "all_longest_gap_hours", "all_jump_fraction",
    ),
}


def fit_ridge(frame: pd.DataFrame, features, alpha: float):
    x = frame[list(features)].to_numpy(float)
    y = frame["log_risk_error"].to_numpy(float)
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (x - center) / scale
    intercept = float(np.mean(y))
    centered_y = y - intercept
    penalty = float(alpha) * np.eye(z.shape[1])
    coefficient = np.linalg.pinv(z.T @ z + penalty) @ z.T @ centered_y
    return {
        "coefficient": coefficient,
        "intercept": intercept,
        "center": center,
        "scale": scale,
        "features": tuple(features),
        "alpha": alpha,
    }


def predict_ridge(fit, frame: pd.DataFrame):
    x = frame[list(fit["features"])].to_numpy(float)
    return fit["intercept"] + ((x - fit["center"]) / fit["scale"]) @ fit["coefficient"]


def select_alpha(frame: pd.DataFrame, features, groups, folds=3):
    scores = []
    for alpha in RIDGE_ALPHAS:
        losses = []
        for train, test in group_splits(groups, folds, SEED + 410):
            fit = fit_ridge(frame.iloc[train], features, alpha)
            prediction = predict_ridge(fit, frame.iloc[test])
            losses.append(float(np.mean((prediction - frame.iloc[test]["log_risk_error"].to_numpy(float)) ** 2)))
        scores.append((float(np.mean(losses)), alpha))
    return min(scores)[1]


def group_oof(frame: pd.DataFrame, features):
    groups = frame["id"].astype(str).to_numpy()
    prediction = np.full(len(frame), np.nan)
    selected = []
    for fold, (train, test) in enumerate(group_splits(groups, 5, SEED + 420)):
        alpha = select_alpha(frame.iloc[train], features, groups[train])
        fit = fit_ridge(frame.iloc[train], features, alpha)
        prediction[test] = predict_ridge(fit, frame.iloc[test])
        selected.append(alpha)
    truth = frame["log_risk_error"].to_numpy(float)
    return prediction, {
        "features": list(features),
        "log_error_mse": float(np.mean((prediction - truth) ** 2)),
        "log_error_mae": float(np.mean(np.abs(prediction - truth))),
        "predicted_vs_observed_spearman": safe_spearman(prediction, truth),
        "selected_alphas": selected,
    }


def model_analysis(prefix_rows: pd.DataFrame):
    hall = prefix_rows[prefix_rows["cohort"] == "hall"].copy().reset_index(drop=True)
    comparisons = {}
    oof_predictions = {}
    for name, features in MODEL_FAMILIES.items():
        prediction, metrics = group_oof(hall, features)
        comparisons[name] = metrics
        oof_predictions[name] = prediction

    def mse_difference(sample):
        index = sample["_row_index"].to_numpy(int)
        y = hall.loc[index, "log_risk_error"].to_numpy(float)
        return float(np.mean((oof_predictions["q_eff"][index] - y) ** 2)
                     - np.mean((oof_predictions["qualified_nights"][index] - y) ** 2))

    indexed = hall.copy()
    indexed["_row_index"] = np.arange(len(indexed))
    q_vs_nights = cluster_bootstrap(indexed, mse_difference, SEED + 1)

    final_fits = {}
    transfer = {}
    colas = prefix_rows[
        (prefix_rows["cohort"] == "colas") & (prefix_rows["prefix_nights"] == 1)
    ].copy().reset_index(drop=True)
    for name, features in MODEL_FAMILIES.items():
        alpha = select_alpha(hall, features, hall["id"].astype(str).to_numpy(), folds=5)
        fit = fit_ridge(hall, features, alpha)
        prediction = predict_ridge(fit, colas)
        transfer[name] = {
            "n": len(colas),
            "predicted_vs_observed_spearman": safe_spearman(prediction, colas["log_risk_error"]),
            "log_error_mse": float(np.mean((prediction - colas["log_risk_error"].to_numpy(float)) ** 2)),
            "log_error_mae": float(np.mean(np.abs(prediction - colas["log_risk_error"].to_numpy(float)))),
        }
        final_fits[name] = {
            "features": list(features),
            "alpha": alpha,
            "intercept": float(fit["intercept"]),
            "coefficients": {feature: float(value) for feature, value in zip(features, fit["coefficient"])},
            "center": {feature: float(value) for feature, value in zip(features, fit["center"])},
            "scale": {feature: float(value) for feature, value in zip(features, fit["scale"])},
        }

    for name, prediction in oof_predictions.items():
        hall[f"pred_log_error_{name}"] = prediction
        hall[f"pred_error_{name}"] = np.maximum(0.0, np.exp(prediction) - 0.005)

    q_mse = comparisons["q_eff"]["log_error_mse"]
    night_mse = comparisons["qualified_nights"]["log_error_mse"]
    q_improvement = (night_mse - q_mse) / night_mse if night_mse > 0 else np.nan
    selected_primary = bool(
        q_improvement >= 0.05
        and q_vs_nights["ci95"][1] is not None
        and q_vs_nights["ci95"][1] < 0
        and finite(transfer["q_eff"]["predicted_vs_observed_spearman"]) > 0
    )
    return hall, {
        "hall_grouped_oof": comparisons,
        "q_eff_relative_mse_improvement_vs_nights": q_improvement,
        "q_eff_minus_nights_mse_cluster_bootstrap": q_vs_nights,
        "colas_k1_to_k2_transfer": transfer,
        "final_hall_fits_for_transfer_only": final_fits,
        "selection_rule": (
            "select q_eff over qualified-night count only if grouped-OOF log-error MSE improves >=5%, "
            "cluster-bootstrap upper CI for q_eff-minus-night MSE is <0, and Colas transfer correlation is positive"
        ),
        "q_eff_selected_as_primary": selected_primary,
    }


def prefix_descriptives(prefix_rows: pd.DataFrame):
    rows = []
    for (cohort, k), group in prefix_rows.groupby(["cohort", "prefix_nights"]):
        item = {
            "cohort": cohort,
            "prefix_nights": int(k),
            "reference_window": str(group["reference_window"].iloc[0]),
            "n": len(group),
            "risk_error_mean": float(group["risk_abs_error"].mean()),
            "risk_error_median": float(group["risk_abs_error"].median()),
            "tier_agreement": float(group["tier_agreement"].mean()),
            "q_eff_60_median": float(group["q_eff_60"].median()),
        }
        for tolerance in RISK_TOLERANCE_SENSITIVITY:
            item[f"stable_fraction_{tolerance:.2f}"] = float(group[f"stable_{tolerance:.2f}"].mean())
        rows.append(item)
    return rows


def order_parameter_correlations(prefix_rows: pd.DataFrame):
    hall = prefix_rows[prefix_rows["cohort"] == "hall"].copy()
    candidates = (
        "qualified_nights", "n_raw", "q_eff_30", "q_eff_60", "q_eff_120",
        "overall_coverage", "all_longest_gap_hours", "all_jump_fraction",
        "log_n_resampled", "night_level_mad",
    )
    results = {}
    for index, candidate in enumerate(candidates):
        results[candidate] = cluster_bootstrap(
            hall,
            lambda frame, name=candidate: safe_spearman(frame[name], frame["risk_abs_error"]),
            SEED + 100 + index,
        )

    adjusted = hall.copy()
    adjusted["log_error_within_k"] = adjusted["log_risk_error"] - adjusted.groupby("prefix_nights")["log_risk_error"].transform("median")
    for candidate in ("q_eff_30", "q_eff_60", "q_eff_120", "overall_coverage", "all_longest_gap_hours", "all_jump_fraction", "log_n_resampled"):
        adjusted[f"{candidate}_within_k"] = adjusted[candidate] - adjusted.groupby("prefix_nights")[candidate].transform("median")
        results[f"{candidate}_within_prefix"] = cluster_bootstrap(
            adjusted,
            lambda frame, name=candidate: safe_spearman(frame[f"{name}_within_k"], frame["log_error_within_k"]),
            SEED + 200 + len(results),
        )

    drift = hall[hall["prefix_nights"] >= 2].copy()
    results["score_drift_remaining_error"] = cluster_bootstrap(
        drift,
        lambda frame: safe_spearman(frame["score_drift"], frame["risk_abs_error"]),
        SEED + 299,
    )
    return results


def robustness_bias_audit(prefix_rows: pd.DataFrame):
    rows = []
    for (cohort, k, label), group in prefix_rows.groupby(["cohort", "prefix_nights", "y"], dropna=False):
        rows.append({
            "cohort": cohort,
            "prefix_nights": int(k),
            "label": None if pd.isna(label) else int(label),
            "n": len(group),
            "risk_error_mean": float(group["risk_abs_error"].mean()),
            "tier_agreement": float(group["tier_agreement"].mean()),
            "strict_stability_0_05": float(group["stable_0.05"].mean()),
        })
    return {
        "by_observed_label_descriptive_only": rows,
        "reference_boundary_distance_vs_tier_agreement": safe_spearman(
            prefix_rows["reference_boundary_distance"], prefix_rows["tier_agreement"].astype(float)
        ),
        "boundary_note": "tier agreement is mechanically harder near 0.30/0.50 boundaries; label summaries are descriptive and not inputs to the robustness models",
    }


def adaptive_policy(hall_with_predictions: pd.DataFrame, model_name: str, tolerance=RISK_TOLERANCE):
    decisions = []
    for subject_id, group in hall_with_predictions.groupby("id"):
        group = group.sort_values("prefix_nights")
        selected = None
        for row in group.to_dict("records"):
            if row[f"pred_error_{model_name}"] <= tolerance:
                selected = row
                break
        if selected is None:
            reference = group.iloc[-1]
            decisions.append({
                "id": subject_id,
                "selected_nights": int(reference["prefix_nights"]),
                "risk_abs_error": float(reference["risk_abs_error"]),
                "tier_agreement": bool(reference["tier_agreement"]),
            })
        else:
            decisions.append({
                "id": subject_id,
                "selected_nights": int(selected["prefix_nights"]),
                "risk_abs_error": float(selected["risk_abs_error"]),
                "tier_agreement": bool(selected["tier_agreement"]),
            })
    frame = pd.DataFrame(decisions)
    return {
        "model": model_name,
        "prediction_tolerance": tolerance,
        "n": len(frame),
        "average_nights": float(frame["selected_nights"].mean()),
        "median_nights": float(frame["selected_nights"].median()),
        "risk_error_mean": float(frame["risk_abs_error"].mean()),
        "risk_error_median": float(frame["risk_abs_error"].median()),
        "tier_agreement": float(frame["tier_agreement"].mean()),
        "selected_nights_counts": {str(key): int(value) for key, value in frame["selected_nights"].value_counts().sort_index().items()},
    }


def fixed_policy(prefix_rows: pd.DataFrame, k: int):
    group = prefix_rows[(prefix_rows["cohort"] == "hall") & (prefix_rows["prefix_nights"] == k)]
    return {
        "nights": k,
        "n": len(group),
        "risk_error_mean": float(group["risk_abs_error"].mean()),
        "risk_error_median": float(group["risk_abs_error"].median()),
        "tier_agreement": float(group["tier_agreement"].mean()),
    }


def apply_s5_formula(frame: pd.DataFrame, formula: dict) -> np.ndarray:
    linear = np.full(len(frame), float(formula["intercept"]))
    for feature in formula["features"]:
        raw = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if feature != "anchor_level":
            raw = np.log1p(np.maximum(raw, 0.0))
        center = float(formula["standardization"][feature]["center"])
        scale = float(formula["standardization"][feature]["scale"])
        linear += float(formula["weights"][feature]) * (raw - center) / scale
    return 100.0 * expit(linear)


def fixed_window_analysis():
    primitives = pd.read_csv(PRIMITIVE_SOURCE)
    primitives["id"] = primitives["id"].astype(str)
    primitives["sensor_key"] = primitives["sensor"].fillna("default").astype(str)
    primitives["eligible"] = primitives["eligible"].astype(str).str.lower().eq("true")
    formula = json.loads(FORMULA_SOURCE.read_text(encoding="utf-8"))["final_formula"]
    primitives["s5_score"] = apply_s5_formula(primitives, formula)

    pairs = []
    summaries = []
    for (source, sensor), group in primitives.groupby(["source_cohort", "sensor_key"]):
        a = group[(group["window_hours"] == 24) & group["eligible"]]
        b = group[(group["window_hours"] == 48) & group["eligible"]]
        merged = a.merge(b, on="id", suffixes=("_24", "_48"), validate="one_to_one")
        if merged.empty:
            continue
        current = pd.DataFrame({
            "analysis_type": "window_24_to_48",
            "cohort": source,
            "sensor": sensor,
            "id": merged["id"].astype(str),
            "coverage_24": pd.to_numeric(merged["coverage_24"], errors="coerce"),
            "valid_points_24": pd.to_numeric(merged["valid_points_24"], errors="coerce"),
            "valid_nights_24": pd.to_numeric(merged["valid_nights_24"], errors="coerce"),
            "opportunity_mass_24": (
                pd.to_numeric(merged["coverage_24"], errors="coerce")
                * pd.to_numeric(merged["valid_nights_24"], errors="coerce").clip(lower=1)
            ),
            "s5_24": merged["s5_score_24"].astype(float),
            "s5_48": merged["s5_score_48"].astype(float),
            "night_mean_24": pd.to_numeric(merged["night_mean_24"], errors="coerce"),
            "night_mean_48": pd.to_numeric(merged["night_mean_48"], errors="coerce"),
        })
        current["s5_abs_change"] = np.abs(current["s5_24"] - current["s5_48"])
        current["night_mean_abs_change"] = np.abs(current["night_mean_24"] - current["night_mean_48"])
        pairs.append(current)
        summaries.append({
            "cohort": source,
            "sensor": sensor,
            "n": len(current),
            "s5_spearman_24_48": safe_spearman(current["s5_24"], current["s5_48"]),
            "s5_icc_absolute_24_48": icc_absolute_agreement(current["s5_24"], current["s5_48"]),
            "s5_absolute_change_median": float(current["s5_abs_change"].median()),
            "night_mean_spearman_24_48": safe_spearman(current["night_mean_24"], current["night_mean_48"]),
            "night_mean_absolute_change_median": float(current["night_mean_abs_change"].median()),
            "opportunity_mass_vs_s5_change_spearman": safe_spearman(current["opportunity_mass_24"], current["s5_abs_change"]),
        })
    pair_frame = pd.concat(pairs, ignore_index=True)

    cgm = primitives[(primitives["source_cohort"] == "cgmacros") & (primitives["window_hours"] == 48) & primitives["eligible"]]
    libre = cgm[cgm["sensor_key"] == "libre"]
    dexcom = cgm[cgm["sensor_key"] == "dexcom"]
    sensors = libre.merge(dexcom, on="id", suffixes=("_libre", "_dexcom"), validate="one_to_one")
    high_quality = sensors[
        (pd.to_numeric(sensors["coverage_libre"], errors="coerce") >= 0.80)
        & (pd.to_numeric(sensors["coverage_dexcom"], errors="coerce") >= 0.80)
    ]

    def device_summary(frame):
        return {
            "n": len(frame),
            "s5_spearman": safe_spearman(frame["s5_score_libre"], frame["s5_score_dexcom"]),
            "s5_icc_absolute": icc_absolute_agreement(frame["s5_score_libre"], frame["s5_score_dexcom"]),
            "s5_absolute_difference_median": float(np.median(np.abs(frame["s5_score_libre"] - frame["s5_score_dexcom"]))),
            "s5_bias_libre_minus_dexcom": float(np.mean(frame["s5_score_libre"] - frame["s5_score_dexcom"])),
            "night_mean_spearman": safe_spearman(frame["night_mean_libre"], frame["night_mean_dexcom"]),
            "night_mean_absolute_difference_median": float(np.median(np.abs(frame["night_mean_libre"] - frame["night_mean_dexcom"]))),
            "night_mean_bias_libre_minus_dexcom": float(np.mean(frame["night_mean_libre"] - frame["night_mean_dexcom"])),
        }

    return pair_frame, {
        "pair_summaries": summaries,
        "all_pairs": {
            "n": len(pair_frame),
            "opportunity_mass_vs_s5_change_spearman": safe_spearman(pair_frame["opportunity_mass_24"], pair_frame["s5_abs_change"]),
            "coverage_vs_s5_change_spearman": safe_spearman(pair_frame["coverage_24"], pair_frame["s5_abs_change"]),
        },
        "cgmacros_device_48h": {
            "all_eligible": device_summary(sensors),
            "both_coverage_gte_0_80": device_summary(high_quality),
        },
        "score_boundary": "S5 is an undeployed research coordinate; 24/48 and device analyses test measurement robustness only",
    }


def main():
    raw_hash_before = sha256(ROOT / "raw_data.zip")
    index_hash_before = sha256(ROOT / "index.html")

    prefix_rows = build_prefix_rows()
    prefix_stats = prefix_descriptives(prefix_rows)
    correlations = order_parameter_correlations(prefix_rows)
    bias_audit = robustness_bias_audit(prefix_rows)
    hall_oof, models = model_analysis(prefix_rows)
    fixed_rows, fixed_results = fixed_window_analysis()

    policies = {
        "fixed": [fixed_policy(prefix_rows, k) for k in (1, 2, 3, 5)],
        "adaptive_q_eff": adaptive_policy(hall_oof, "q_eff"),
        "adaptive_quality_panel": adaptive_policy(hall_oof, "quality_panel"),
    }

    q_selected = models["q_eff_selected_as_primary"]
    drift = correlations["score_drift_remaining_error"]
    drift_selected = bool(drift["ci95"][0] is not None and drift["ci95"][0] > 0 and drift["estimate"] >= 0.30)
    main_parameter = "q_eff_60" if q_selected else "qualified_nights"
    interpretation = {
        "main_order_parameter": main_parameter,
        "q_eff_status": "primary" if q_selected else "quality correction, not proven superior to qualified-night count",
        "prefix_drift_status": "secondary controller" if drift_selected else "exploratory only",
        "device_offset_status": "orthogonal robustness axis; cannot be repaired by collecting more points alone",
        "clinical_boundary": "acquisition/QC robustness only; not a treatment recommendation or physiological health score",
    }

    combined_rows = pd.concat([prefix_rows, fixed_rows], ignore_index=True, sort=False)
    combined_rows.to_csv(ROW_PATH, index=False)

    result = {
        "protocol": {
            "research_question": "which early, actionable order parameter predicts whether more CGM data will materially change the sample output",
            "unit": "subject",
            "prefix_reference": {"hall": "full available record", "colas": "full available record"},
            "primary_error": "absolute difference in current v8.4 screening output versus subject's longer reference",
            "strict_stability": "absolute risk difference <=0.05 and unchanged 0.30/0.50 risk tier",
            "q_eff": "sum over qualified nights of coverage*exp(-max(longest_gap-5,0)/60)*(1-jump_fraction)",
            "q_eff_gap_decay_sensitivity_minutes": list(QUALITY_GAP_DECAY_MINUTES),
            "validation": "Hall five-fold grouped OOF plus Colas k1-to-k2 transfer; clustered subject bootstrap",
            "fixed_window_secondary": "24-to-48-hour S5 and nightMean stability across all available sources; CGMacros dual-sensor audit",
            "source_data_read_only": True,
        },
        "prefix_descriptives": prefix_stats,
        "order_parameter_correlations": correlations,
        "robustness_bias_audit": bias_audit,
        "model_comparison": models,
        "collection_policy_simulation": policies,
        "fixed_window_and_device": fixed_results,
        "interpretation": interpretation,
        "deployment": {
            "candidate_ready_for_frontend": False,
            "index_changed": False,
            "reason": "research-only cycle; no independent unseen cohort and device offset remains a separate limitation",
        },
        "integrity": {
            "raw_data_zip_sha256_before": raw_hash_before,
            "raw_data_zip_sha256_after": sha256(ROOT / "raw_data.zip"),
            "index_html_sha256_before": index_hash_before,
            "index_html_sha256_after": sha256(ROOT / "index.html"),
        },
        "artifacts": {
            "rows_csv": str(ROW_PATH.relative_to(ROOT)),
            "results_json": str(RESULT_PATH.relative_to(ROOT)),
        },
    }
    RESULT_PATH.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"prefix rows: {len(prefix_rows)}; fixed 24/48 pairs: {len(fixed_rows)}")
    print(f"main order parameter: {main_parameter}")
    print(f"q_eff selected: {q_selected}; drift selected: {drift_selected}")
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"wrote {ROW_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
