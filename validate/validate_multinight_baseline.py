#!/usr/bin/env python3
"""Validate multi-night nocturnal-baseline features on the Hall cohort.

Research question
-----------------
Does persistent elevation across complete nights improve the v8.3 deployed
screen, which pools every 00:00-06:00 point into one mean?

Pre-registered candidate rules (fixed before examining outcomes)
----------------------------------------------------------------
1. current_pooled: pooled 00:00-06:00 mean (v8.3 comparator)
2. robust_median: median of complete-night 00:00-06:00 means
3. split_baseline: median 00:00-03:00 mean + median 03:00-06:00 mean
4. late_plus_dawn: median 03:00-06:00 mean + median within-night rise
5. level_plus_iqr: median complete-night mean + between-night IQR

Validation protocol
-------------------
* Subject-level repeated 10x5 stratified outer CV.
* Standardization and logistic fitting occur inside each training fold.
* A nested selector chooses among the four new rules using training data only.
* Screening thresholds target sensitivity >= 0.75 and are selected from inner
  OOF predictions only.
* Paired stratified bootstrap quantifies AUC differences.
* A max-statistic label-permutation test adjusts for choosing the best of four
  new candidates.
* Fasting insulin (n=53) and SSPG (n=42) are secondary mechanistic endpoints;
  they are never used to choose the diagnosis model.

Deployment gate (fixed before running)
--------------------------------------
The risk formula is eligible for deployment only if all conditions hold:
* best new candidate AUC improves on current_pooled by >= 0.02;
* paired bootstrap 95% CI lower bound for delta AUC is > 0;
* selection-adjusted permutation p <= 0.05;
* nested-selector AUC improves on current_pooled by >= 0.01;
* >= 90% of subjects have at least three complete nights.

The script intentionally uses only numpy/pandas from the bundled runtime. It
reads raw_data.zip without extracting or modifying source healthcare data and
writes auditable derived results to output/.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "raw_data.zip"
OUTPUT_DIR = ROOT / "output"
RNG_SEED = 20260807
MGDL_PER_MMOLL = 18.0182
MIN_HALF_NIGHT_POINTS = 48  # >=2 hours at the JS pipeline's typical 2.5-min grid
N_OUTER_REPEATS = 10
N_SPLITS = 5
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 300

CANDIDATES = {
    "current_pooled": ["pooled_night_mean"],
    "robust_median": ["median_night_mean"],
    "split_baseline": ["median_early_mean", "median_late_mean"],
    "late_plus_dawn": ["median_late_mean", "median_dawn_delta"],
    "level_plus_iqr": ["median_night_mean", "night_iqr"],
}
NEW_CANDIDATES = [k for k in CANDIDATES if k != "current_pooled"]


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranked_y = y[order]
    n_pos = int(ranked_y.sum())
    if n_pos == 0:
        return float("nan")
    precision = np.cumsum(ranked_y) / np.arange(1, len(y) + 1)
    return float(precision[ranked_y == 1].sum() / n_pos)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    chunks: dict[int, list[np.ndarray]] = {}
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        chunks[label] = list(np.array_split(idx, n_splits))
    all_idx = np.arange(len(y))
    folds = []
    for fold in range(n_splits):
        test = np.sort(np.concatenate([chunks[0][fold], chunks[1][fold]]))
        train = np.setdiff1d(all_idx, test, assume_unique=True)
        folds.append((train, test))
    return folds


def fit_logistic_standardized(X: np.ndarray, y: np.ndarray, l2: float = 1.0):
    """Fit L2 logistic regression by damped Newton steps.

    The intercept is not penalized. Returned raw coefficients can be hard-coded
    directly into JavaScript.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    design = np.column_stack([np.ones(len(Z)), Z])
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag(np.r_[0.0, np.full(Z.shape[1], l2)])

    for _ in range(100):
        p = sigmoid(design @ beta)
        weights = np.maximum(p * (1.0 - p), 1e-8)
        grad = design.T @ (p - y) + penalty @ beta
        hess = design.T @ (design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break

    raw_coef = beta[1:] / scale
    raw_intercept = beta[0] - float(np.sum(beta[1:] * mean / scale))
    return {
        "mean": mean,
        "scale": scale,
        "beta_standardized": beta,
        "raw_intercept": float(raw_intercept),
        "raw_coef": raw_coef,
    }


def predict_logistic(model, X: np.ndarray) -> np.ndarray:
    return sigmoid(model["raw_intercept"] + np.asarray(X, dtype=float) @ model["raw_coef"])


def repeated_oof(X: np.ndarray, y: np.ndarray, repeats: int, seed: int):
    total = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    rep_aucs = []
    for rep in range(repeats):
        rep_pred = np.zeros(len(y), dtype=float)
        for train, test in stratified_folds(y, N_SPLITS, seed + rep):
            model = fit_logistic_standardized(X[train], y[train])
            rep_pred[test] = predict_logistic(model, X[test])
        total += rep_pred
        counts += 1
        rep_aucs.append(roc_auc(y, rep_pred))
    return total / counts, np.asarray(rep_aucs)


def screening_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    positive = np.asarray(pred, dtype=float) >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fn = int(np.sum((~positive) & (y == 1)))
    tn = int(np.sum((~positive) & (y == 0)))
    fp = int(np.sum(positive & (y == 0)))
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "balanced_accuracy": 0.5 * (
            tp / (tp + fn) if tp + fn else 0.0
        ) + 0.5 * (
            tn / (tn + fp) if tn + fp else 0.0
        ),
        "flag_rate": float(positive.mean()),
    }


def choose_screen_threshold(y: np.ndarray, pred: np.ndarray, min_sensitivity: float = 0.75) -> float:
    candidates = np.unique(np.r_[0.0, np.asarray(pred, dtype=float), 1.0])
    eligible = []
    for threshold in candidates:
        metrics = screening_metrics(y, pred, float(threshold))
        if metrics["sensitivity"] >= min_sensitivity:
            eligible.append((metrics["specificity"], float(threshold)))
    if not eligible:
        return 0.0
    # Maximize specificity, then choose the highest threshold for deterministic ties.
    eligible.sort()
    return eligible[-1][1]


def resample_like_js(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror index.html resampleDataImpl(..., smooth=false)."""
    frame = frame.sort_values("time")[["time", "gl_mmol"]].dropna().reset_index(drop=True)
    if frame.empty:
        return frame
    times = [frame.loc[0, "time"]]
    values: list[float | None] = [float(frame.loc[0, "gl_mmol"])]
    for i in range(1, len(frame)):
        prev_t = frame.loc[i - 1, "time"]
        curr_t = frame.loc[i, "time"]
        prev_v = float(frame.loc[i - 1, "gl_mmol"])
        curr_v = float(frame.loc[i, "gl_mmol"])
        gap = (curr_t - prev_t).total_seconds() / 60.0
        if 4 < gap <= 15:
            steps = int(math.floor(gap / 3.0 + 0.5))
            t_step = (curr_t - prev_t) / steps
            v_step = (curr_v - prev_v) / steps
            for j in range(1, steps):
                times.append(prev_t + t_step * j)
                values.append(prev_v + v_step * j)
        elif gap > 15:
            times.append(prev_t + pd.Timedelta(minutes=15))
            values.append(None)
        times.append(curr_t)
        values.append(curr_v)
    return pd.DataFrame({"time": times, "gl_mmol": values})


def extract_subject_features(subject: pd.DataFrame):
    resampled = resample_like_js(subject)
    resampled["hour"] = (
        resampled["time"].dt.hour
        + resampled["time"].dt.minute / 60.0
        + resampled["time"].dt.second / 3600.0
    )
    night_all = resampled[(resampled["hour"] >= 0) & (resampled["hour"] < 6)]
    pooled = float(night_all["gl_mmol"].dropna().mean()) if night_all["gl_mmol"].notna().sum() >= 6 else np.nan

    night_rows = []
    for date, group in night_all.groupby(night_all["time"].dt.date):
        early = group[group["hour"] < 3]["gl_mmol"].dropna().to_numpy(float)
        late = group[group["hour"] >= 3]["gl_mmol"].dropna().to_numpy(float)
        if len(early) < MIN_HALF_NIGHT_POINTS or len(late) < MIN_HALF_NIGHT_POINTS:
            continue
        combined = np.r_[early, late]
        night_rows.append({
            "date": str(date),
            "night_mean": float(combined.mean()),
            "early_mean": float(early.mean()),
            "late_mean": float(late.mean()),
            "dawn_delta": float(late.mean() - early.mean()),
            "n_early": int(len(early)),
            "n_late": int(len(late)),
        })

    nights = pd.DataFrame(night_rows)
    if nights.empty:
        summaries = {
            "median_night_mean": np.nan,
            "median_early_mean": np.nan,
            "median_late_mean": np.nan,
            "median_dawn_delta": np.nan,
            "night_iqr": np.nan,
            "valid_night_pooled_mean": np.nan,
        }
    else:
        summaries = {
            "median_night_mean": float(nights["night_mean"].median()),
            "median_early_mean": float(nights["early_mean"].median()),
            "median_late_mean": float(nights["late_mean"].median()),
            "median_dawn_delta": float(nights["dawn_delta"].median()),
            "night_iqr": float(nights["night_mean"].quantile(0.75) - nights["night_mean"].quantile(0.25)),
            "valid_night_pooled_mean": float(nights["night_mean"].mean()),
        }
    summaries.update({
        "pooled_night_mean": pooled,
        "n_valid_nights": int(len(nights)),
        "n_resampled_points": int(resampled["gl_mmol"].notna().sum()),
    })
    return summaries, night_rows


def load_hall_features():
    with ZipFile(ZIP_PATH) as archive:
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle)
    hall["time"] = pd.to_datetime(hall["time"], errors="coerce")
    hall["gl_mmol"] = hall["gl"].astype(float) / MGDL_PER_MMOLL
    rows = []
    night_records: dict[str, list[dict]] = {}
    clinical_columns = ["diagnosis", "A1C", "FBG", "ogtt.2hr", "insulin", "SSPG"]
    for subject_id, group in hall.groupby("id", sort=True):
        features, nights = extract_subject_features(group)
        first = group.iloc[0]
        row = {"id": str(subject_id), **features}
        for column in clinical_columns:
            value = float(first[column]) if pd.notna(first[column]) else np.nan
            row[column] = value if value >= 0 else np.nan
        rows.append(row)
        night_records[str(subject_id)] = nights
    return pd.DataFrame(rows), night_records


def candidate_matrix(features: pd.DataFrame, name: str) -> np.ndarray:
    return features[CANDIDATES[name]].to_numpy(float)


def evaluate_candidates(features: pd.DataFrame, y: np.ndarray):
    results = {}
    predictions = {}
    for name, columns in CANDIDATES.items():
        X = candidate_matrix(features, name)
        pred, rep_aucs = repeated_oof(X, y, N_OUTER_REPEATS, RNG_SEED + 100)
        predictions[name] = pred
        full_model = fit_logistic_standardized(X, y)
        results[name] = {
            "features": columns,
            "auc": roc_auc(y, pred),
            "rep_auc_mean": float(rep_aucs.mean()),
            "rep_auc_sd": float(rep_aucs.std(ddof=0)),
            "pr_auc": average_precision(y, pred),
            "brier": float(np.mean((pred - y) ** 2)),
            "raw_intercept": full_model["raw_intercept"],
            "raw_coefficients": {
                column: float(coef) for column, coef in zip(columns, full_model["raw_coef"])
            },
        }
    return results, predictions


def bootstrap_auc_delta(y: np.ndarray, new_pred: np.ndarray, base_pred: np.ndarray, n_boot: int):
    rng = np.random.default_rng(RNG_SEED + 200)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        sampled_y = y[sample]
        deltas[i] = roc_auc(sampled_y, new_pred[sample]) - roc_auc(sampled_y, base_pred[sample])
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def selection_adjusted_permutation(features: pd.DataFrame, y: np.ndarray):
    """Max-statistic test for choosing the best of four new candidates."""
    rng = np.random.default_rng(RNG_SEED + 300)
    observed = {}
    for name in NEW_CANDIDATES:
        pred, _ = repeated_oof(candidate_matrix(features, name), y, 2, RNG_SEED + 310)
        observed[name] = roc_auc(y, pred)
    observed_max = max(observed.values())
    null_max = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        scores = []
        for name in NEW_CANDIDATES:
            pred, _ = repeated_oof(
                candidate_matrix(features, name), permuted, 2, RNG_SEED + 400 + i * 7
            )
            scores.append(roc_auc(permuted, pred))
        null_max[i] = max(scores)
    p_value = (1 + int(np.sum(null_max >= observed_max))) / (N_PERMUTATIONS + 1)
    return {
        "observed_two_repeat_auc_by_candidate": observed,
        "observed_max_auc": float(observed_max),
        "null_mean": float(null_max.mean()),
        "null_p95": float(np.quantile(null_max, 0.95)),
        "p_value": float(p_value),
        "n_permutations": N_PERMUTATIONS,
    }


def nested_selection(features: pd.DataFrame, y: np.ndarray):
    total_pred = np.zeros(len(y), dtype=float)
    pred_count = np.zeros(len(y), dtype=int)
    choices = Counter()
    thresholds = []
    fold_metrics = []

    for rep in range(N_OUTER_REPEATS):
        for fold_no, (train, test) in enumerate(stratified_folds(y, N_SPLITS, RNG_SEED + 500 + rep)):
            inner_scores = {}
            inner_predictions = {}
            for name in NEW_CANDIDATES:
                X_train = candidate_matrix(features.iloc[train], name)
                pred, _ = repeated_oof(
                    X_train, y[train], 3, RNG_SEED + 600 + rep * 31 + fold_no
                )
                inner_predictions[name] = pred
                inner_scores[name] = roc_auc(y[train], pred)
            # Prefer the simpler one-feature rule on exact ties.
            chosen = max(
                NEW_CANDIDATES,
                key=lambda name: (inner_scores[name], -len(CANDIDATES[name]), name),
            )
            choices[chosen] += 1
            threshold = choose_screen_threshold(y[train], inner_predictions[chosen])
            thresholds.append(threshold)

            X_train = candidate_matrix(features.iloc[train], chosen)
            X_test = candidate_matrix(features.iloc[test], chosen)
            model = fit_logistic_standardized(X_train, y[train])
            test_pred = predict_logistic(model, X_test)
            total_pred[test] += test_pred
            pred_count[test] += 1
            fold_metrics.append(screening_metrics(y[test], test_pred, threshold))

    pred = total_pred / pred_count
    return {
        "auc": roc_auc(y, pred),
        "pr_auc": average_precision(y, pred),
        "brier": float(np.mean((pred - y) ** 2)),
        "choice_counts": dict(choices),
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "mean_fold_screening_metrics": {
            key: float(np.mean([row[key] for row in fold_metrics]))
            for key in fold_metrics[0]
        },
        "predictions": pred,
    }


def fixed_model_nested_threshold(features: pd.DataFrame, y: np.ndarray, name: str):
    fold_metrics = []
    thresholds = []
    for rep in range(N_OUTER_REPEATS):
        for fold_no, (train, test) in enumerate(stratified_folds(y, N_SPLITS, RNG_SEED + 700 + rep)):
            X_train = candidate_matrix(features.iloc[train], name)
            inner_pred, _ = repeated_oof(
                X_train, y[train], 3, RNG_SEED + 800 + rep * 31 + fold_no
            )
            threshold = choose_screen_threshold(y[train], inner_pred)
            thresholds.append(threshold)
            model = fit_logistic_standardized(X_train, y[train])
            test_pred = predict_logistic(model, candidate_matrix(features.iloc[test], name))
            fold_metrics.append(screening_metrics(y[test], test_pred, threshold))
    return {
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "mean_fold_screening_metrics": {
            key: float(np.mean([row[key] for row in fold_metrics]))
            for key in fold_metrics[0]
        },
    }


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 1000):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    observed = spearman(x, y)
    rng = np.random.default_rng(RNG_SEED + len(x) * 13)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(n_perm)])
    p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_perm + 1)
    return {"n": int(len(x)), "rho": observed, "p_value": float(p)}


def loo_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.zeros(len(y), dtype=float)
    for test in range(len(y)):
        train = np.arange(len(y)) != test
        mean = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        Z = (X[train] - mean) / scale
        y_mean = y[train].mean()
        beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y[train] - y_mean))
        pred[test] = y_mean + ((X[test] - mean) / scale) @ beta
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "loo_r2": float(1.0 - np.sum((y - pred) ** 2) / denominator),
        "loo_mae": float(np.mean(np.abs(y - pred))),
        "spearman_predicted_observed": spearman(pred, y),
    }


def mechanistic_endpoints(features: pd.DataFrame):
    individual_features = sorted({column for columns in CANDIDATES.values() for column in columns})
    results = {}
    for endpoint in ("insulin", "SSPG"):
        valid_endpoint = np.isfinite(features[endpoint].to_numpy(float))
        endpoint_result = {"feature_correlations": {}, "candidate_loo_ridge": {}}
        y_endpoint = features.loc[valid_endpoint, endpoint].to_numpy(float)
        for column in individual_features:
            x = features.loc[valid_endpoint, column].to_numpy(float)
            endpoint_result["feature_correlations"][column] = permutation_spearman(x, y_endpoint)
        for name in CANDIDATES:
            X = candidate_matrix(features.loc[valid_endpoint], name)
            endpoint_result["candidate_loo_ridge"][name] = loo_ridge(X, y_endpoint)
        results[endpoint] = endpoint_result
    return results


def risk_tier(probability: float) -> str:
    if probability >= 0.50:
        return "high"
    if probability >= 0.30:
        return "medium"
    return "low"


def prefix_stability(features: pd.DataFrame, night_records: dict[str, list[dict]], y: np.ndarray):
    full_by_id = features.set_index("id")
    result = {}
    for k in (1, 2, 3, 5):
        rows = []
        for subject_id, nights in night_records.items():
            if len(nights) < k:
                continue
            prefix = nights[:k]
            prefix_means = np.array([row["night_mean"] for row in prefix], dtype=float)
            full_means = np.array([row["night_mean"] for row in nights], dtype=float)
            rows.append({
                "id": subject_id,
                "prefix_pooled": float(prefix_means.mean()),
                "prefix_median": float(np.median(prefix_means)),
                "full_valid_pooled": float(full_means.mean()),
                "full_median": float(np.median(full_means)),
                "full_deployed_pooled": float(full_by_id.loc[subject_id, "pooled_night_mean"]),
            })
        table = pd.DataFrame(rows).set_index("id")
        ids = table.index
        subset_y = full_by_id.loc[ids, "diagnosis"].to_numpy(float) >= 1
        prefix_risk = sigmoid(1.064314 * table["prefix_pooled"].to_numpy() - 6.746364)
        full_risk = sigmoid(1.064314 * table["full_deployed_pooled"].to_numpy() - 6.746364)
        tier_agreement = np.mean([
            risk_tier(float(a)) == risk_tier(float(b)) for a, b in zip(prefix_risk, full_risk)
        ])
        result[str(k)] = {
            "n_subjects": int(len(table)),
            "pooled_auc": roc_auc(subset_y.astype(int), table["prefix_pooled"].to_numpy()),
            "median_auc": roc_auc(subset_y.astype(int), table["prefix_median"].to_numpy()),
            "pooled_spearman_vs_full": spearman(
                table["prefix_pooled"].to_numpy(), table["full_valid_pooled"].to_numpy()
            ),
            "median_spearman_vs_full": spearman(
                table["prefix_median"].to_numpy(), table["full_median"].to_numpy()
            ),
            "deployed_risk_mae_vs_full": float(np.mean(np.abs(prefix_risk - full_risk))),
            "deployed_tier_agreement_vs_full": float(tier_agreement),
        }
    return result


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, night_records = load_hall_features()
    required = sorted({column for columns in CANDIDATES.values() for column in columns})
    complete = features[required].notna().all(axis=1)
    analysis = features.loc[complete].reset_index(drop=True)
    y = (analysis["diagnosis"].to_numpy(float) >= 1).astype(int)

    candidate_results, predictions = evaluate_candidates(analysis, y)
    baseline_auc = candidate_results["current_pooled"]["auc"]
    best_new = max(NEW_CANDIDATES, key=lambda name: candidate_results[name]["auc"])
    best_auc = candidate_results[best_new]["auc"]

    bootstrap = {}
    for name in NEW_CANDIDATES:
        bootstrap[name] = bootstrap_auc_delta(
            y, predictions[name], predictions["current_pooled"], N_BOOTSTRAP
        )

    permutation = selection_adjusted_permutation(analysis, y)
    nested = nested_selection(analysis, y)
    baseline_threshold = fixed_model_nested_threshold(analysis, y, "current_pooled")
    mechanisms = mechanistic_endpoints(analysis)
    stability = prefix_stability(analysis, night_records, y)

    n_three_nights = int((analysis["n_valid_nights"] >= 3).sum())
    three_night_coverage = n_three_nights / len(analysis)
    best_bootstrap = bootstrap[best_new]
    deploy = bool(
        best_auc - baseline_auc >= 0.02
        and best_bootstrap["ci95"][0] > 0
        and permutation["p_value"] <= 0.05
        and nested["auc"] - baseline_auc >= 0.01
        and three_night_coverage >= 0.90
    )

    # A mechanistic association requires at least one nominally significant
    # positive correlation with fasting insulin or SSPG. Individual-level
    # prediction is a stricter claim and additionally requires useful LOO R².
    best_columns = CANDIDATES[best_new]
    mechanism_hits = []
    for endpoint in ("insulin", "SSPG"):
        for column in best_columns:
            corr = mechanisms[endpoint]["feature_correlations"][column]
            if corr["rho"] is not None and corr["rho"] > 0 and corr["p_value"] <= 0.05:
                mechanism_hits.append({"endpoint": endpoint, "feature": column, **corr})

    deployed_formula_risk = sigmoid(1.064314 * analysis["pooled_night_mean"].to_numpy() - 6.746364)
    best_insulin_loo = mechanisms["insulin"]["candidate_loo_ridge"][best_new]["loo_r2"]
    best_sspg_loo = mechanisms["SSPG"]["candidate_loo_ridge"][best_new]["loo_r2"]
    individual_insulin_prediction_supported = bool(max(best_insulin_loo, best_sspg_loo) >= 0.05)

    results = {
        "protocol": {
            "cohort": "Hall/PLOS Biol untreated cohort",
            "n_total": int(len(features)),
            "n_complete_case": int(len(analysis)),
            "n_positive_diagnosis_ge_1": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "min_half_night_points": MIN_HALF_NIGHT_POINTS,
            "outer_cv": f"{N_OUTER_REPEATS}x{N_SPLITS} subject-level stratified CV",
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutations": N_PERMUTATIONS,
        },
        "data_quality": {
            "valid_nights_median": float(analysis["n_valid_nights"].median()),
            "valid_nights_range": [int(analysis["n_valid_nights"].min()), int(analysis["n_valid_nights"].max())],
            "subjects_with_at_least_3_nights": n_three_nights,
            "three_night_coverage": three_night_coverage,
        },
        "candidate_results": candidate_results,
        "best_new_candidate": best_new,
        "best_new_delta_auc": best_auc - baseline_auc,
        "paired_bootstrap_delta_auc": bootstrap,
        "selection_adjusted_permutation": permutation,
        "nested_candidate_selector": {k: v for k, v in nested.items() if k != "predictions"},
        "baseline_nested_threshold": baseline_threshold,
        "deployed_v83_formula": {
            "auc": roc_auc(y, deployed_formula_risk),
            "pr_auc": average_precision(y, deployed_formula_risk),
            "brier": float(np.mean((deployed_formula_risk - y) ** 2)),
            "threshold_0_30": screening_metrics(y, deployed_formula_risk, 0.30),
            "threshold_0_50": screening_metrics(y, deployed_formula_risk, 0.50),
            "coefficient_replication": {
                "fitted_intercept": candidate_results["current_pooled"]["raw_intercept"],
                "deployed_intercept": -6.746364,
                "absolute_intercept_difference": abs(
                    candidate_results["current_pooled"]["raw_intercept"] + 6.746364
                ),
                "fitted_coefficient": candidate_results["current_pooled"]["raw_coefficients"]["pooled_night_mean"],
                "deployed_coefficient": 1.064314,
                "absolute_coefficient_difference": abs(
                    candidate_results["current_pooled"]["raw_coefficients"]["pooled_night_mean"] - 1.064314
                ),
            },
            "best_new_oof_auc_minus_deployed_apparent_auc": best_auc - roc_auc(y, deployed_formula_risk),
        },
        "prefix_night_stability": stability,
        "mechanistic_endpoints": mechanisms,
        "mechanistic_support_for_insulin_claim": {
            "association_supported": bool(mechanism_hits),
            "individual_prediction_supported": individual_insulin_prediction_supported,
            "nominal_positive_hits": mechanism_hits,
            "best_new_candidate_loo_r2": {
                "insulin": best_insulin_loo,
                "SSPG": best_sspg_loo,
            },
            "interpretation": (
                "Night-level features show mechanistic association, but do not support individual insulin prediction"
                if mechanism_hits and not individual_insulin_prediction_supported
                else "No adequate mechanistic support for an insulin-abnormality prediction claim"
            ),
        },
        "deployment_gate": {
            "eligible": deploy,
            "requirements": {
                "best_new_delta_auc_gte_0_02": bool(best_auc - baseline_auc >= 0.02),
                "bootstrap_ci_lower_gt_0": bool(best_bootstrap["ci95"][0] > 0),
                "selection_adjusted_permutation_p_lte_0_05": bool(permutation["p_value"] <= 0.05),
                "nested_selector_delta_auc_gte_0_01": bool(nested["auc"] - baseline_auc >= 0.01),
                "three_night_coverage_gte_0_90": bool(three_night_coverage >= 0.90),
            },
            "decision": (
                "Deploy best new multi-night rule" if deploy
                else "Retain v8.3 risk formula; multi-night optimization not validated"
            ),
        },
    }

    feature_path = OUTPUT_DIR / "multinight_subject_features.csv"
    result_path = OUTPUT_DIR / "multinight_baseline_results.json"
    analysis.to_csv(feature_path, index=False)
    result_path.write_text(
        json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Multi-night baseline validation ===")
    print(f"Hall complete-case n={len(analysis)}; positives={y.sum()}; median valid nights={analysis['n_valid_nights'].median():.1f}")
    for name, row in candidate_results.items():
        print(
            f"  {name:<18} AUC={row['auc']:.3f} | rep={row['rep_auc_mean']:.3f}±{row['rep_auc_sd']:.3f} "
            f"| PR-AUC={row['pr_auc']:.3f} | Brier={row['brier']:.3f}"
        )
    print(f"Best new: {best_new}; delta AUC={best_auc - baseline_auc:+.3f}; bootstrap 95% CI={best_bootstrap['ci95']}")
    print(
        f"Selection-adjusted permutation p={permutation['p_value']:.4f}; "
        f"nested-selector AUC={nested['auc']:.3f}"
    )
    print(
        f"Mechanistic association: {bool(mechanism_hits)}; "
        f"individual insulin prediction: {individual_insulin_prediction_supported}"
    )
    print(f"DEPLOYMENT ELIGIBLE: {deploy} — {results['deployment_gate']['decision']}")
    print(f"Wrote {result_path.relative_to(ROOT)} and {feature_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
