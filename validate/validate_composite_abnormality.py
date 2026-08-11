#!/usr/bin/env python3
"""Fit and validate the nine-feature composite glycemic abnormality parameter.

Discovery is restricted to CGMacros fasting insulin and Colas T2DM. Hall is
untouched until the feature family and regularization are frozen. Iglu,
Dubosson, and Weinstock are used only for calculability and stability audits.
"""

from __future__ import annotations

import glob
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PRIMITIVE_PATH = OUTPUT / "composite_abnormality_primitives.csv"
RESULT_PATH = OUTPUT / "composite_abnormality_results.json"
FEATURE_PATH = OUTPUT / "composite_abnormality_features.csv"

SEED = 20260810
OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 3
LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)
ELASTIC_ALPHA = 0.50
BOOTSTRAPS = 5000
PERMUTATIONS = 5000

BASE_FEATURES = (
    "hyper_burden",
    "hypo_burden",
    "variation_load",
    "recovery_debt",
    "anchor_level",
)
PHASE_FEATURES = ("volume", "lyapunov", "det", "entr")
ALL_FEATURES = BASE_FEATURES + PHASE_FEATURES
MODEL_FAMILIES = {
    "base5": BASE_FEATURES,
    "phase4": PHASE_FEATURES,
    "full9": ALL_FEATURES,
}
BASELINES = ("night_mean", "tar_180", "tbr_70", "cv")


def id_key(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_features() -> pd.DataFrame:
    primitive = pd.read_csv(PRIMITIVE_PATH)
    primitive["id"] = primitive["id"].map(id_key)
    primitive = primitive[primitive["eligible"].astype(str).str.lower().isin({"true", "1"})].copy()

    phase_rows = []
    for path in sorted(glob.glob(str(OUTPUT / "composite_phase_*_w*.json"))):
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["id"] = frame["id"].map(id_key)
        phase_rows.append(frame[["cohort", "id", *PHASE_FEATURES, "tau", "embeddingDim", "nResampled"]])
    phase = pd.concat(phase_rows, ignore_index=True)
    merged = primitive.merge(phase, on=["cohort", "id"], how="inner", validate="one_to_one")
    for column in (*ALL_FEATURES, *BASELINES, "insulin", "SSPG", "diagnosis", "y"):
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def cgmacros_sensor_average(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    subset = frame[(frame["source_cohort"] == "cgmacros") & (frame["window_hours"] == hours)].copy()
    numeric = [*ALL_FEATURES, *BASELINES, "insulin", "a1c"]
    result = subset.groupby("id", as_index=False)[numeric].mean(numeric_only=True)
    result["source_cohort"] = "cgmacros"
    result["sensor"] = "sensor_average"
    result["window_hours"] = hours
    result["cohort"] = f"cgmacros_average_w{hours}"
    return result


def raw_transform(frame: pd.DataFrame, features) -> np.ndarray:
    columns = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if feature in {"hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "volume"}:
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    return np.column_stack(columns)


class RobustTransformer:
    def __init__(self, features, center=None, scale=None):
        self.features = tuple(features)
        self.center = None if center is None else np.asarray(center, float)
        self.scale = None if scale is None else np.asarray(scale, float)

    def fit(self, frames):
        raw = np.vstack([raw_transform(frame, self.features) for frame in frames])
        self.center = np.nanmedian(raw, axis=0)
        q25 = np.nanquantile(raw, 0.25, axis=0)
        q75 = np.nanquantile(raw, 0.75, axis=0)
        robust = (q75 - q25) / 1.349
        fallback = np.nanstd(raw, axis=0, ddof=1)
        self.scale = np.where(robust > 1e-6, robust, np.where(fallback > 1e-6, fallback, 1.0))
        return self

    def transform(self, frame):
        raw = raw_transform(frame, self.features)
        raw = np.where(np.isfinite(raw), raw, self.center)
        return (raw - self.center) / self.scale

    def export(self):
        return {
            feature: {"center": float(self.center[i]), "scale": float(self.scale[i])}
            for i, feature in enumerate(self.features)
        }


def fit_joint(cgm: pd.DataFrame, colas: pd.DataFrame, features, lam: float):
    transformer = RobustTransformer(features).fit([cgm, colas])
    xc, xk = transformer.transform(cgm), transformer.transform(colas)
    yc_raw = np.log1p(cgm["insulin"].to_numpy(float))
    yc_center = float(np.mean(yc_raw))
    yc_scale = float(np.std(yc_raw, ddof=1)) or 1.0
    yc = (yc_raw - yc_center) / yc_scale
    yk = colas["y"].to_numpy(float)
    p = len(features)

    def objective(params):
        weights = params[:p]
        c_intercept, k_intercept = params[p], params[p + 1]
        c_residual = c_intercept + xc @ weights - yc
        k_eta = k_intercept + xk @ weights
        k_prob = expit(k_eta)
        mse = 0.5 * np.mean(c_residual ** 2)
        logloss = np.mean(np.logaddexp(0.0, k_eta) - yk * k_eta)
        smooth_l1 = np.sqrt(weights ** 2 + 1e-8)
        penalty = lam * (
            ELASTIC_ALPHA * smooth_l1.sum()
            + 0.5 * (1.0 - ELASTIC_ALPHA) * np.dot(weights, weights)
        )
        value = mse + logloss + penalty

        grad_w = xc.T @ c_residual / len(xc) + xk.T @ (k_prob - yk) / len(xk)
        grad_w += lam * (
            ELASTIC_ALPHA * weights / smooth_l1
            + (1.0 - ELASTIC_ALPHA) * weights
        )
        gradient = np.r_[grad_w, np.mean(c_residual), np.mean(k_prob - yk)]
        return value, gradient

    initial = np.zeros(p + 2)
    prevalence = np.clip(yk.mean(), 1e-5, 1 - 1e-5)
    initial[p + 1] = math.log(prevalence / (1 - prevalence))
    bounds = []
    for feature in features:
        bounds.append((0.0, None) if feature in BASE_FEATURES else (None, None))
    bounds.extend([(None, None), (None, None)])
    fit = minimize(
        lambda params: objective(params),
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not fit.success:
        raise RuntimeError(f"joint optimization failed: {fit.message}")
    return {
        "features": tuple(features),
        "transformer": transformer,
        "weights": fit.x[:p],
        "cgm_intercept": float(fit.x[p]),
        "colas_intercept": float(fit.x[p + 1]),
        "cgm_y_center": yc_center,
        "cgm_y_scale": yc_scale,
        "lambda": float(lam),
        "objective": float(fit.fun),
    }


def model_score(model, frame) -> np.ndarray:
    return model["transformer"].transform(frame) @ model["weights"]


def safe_spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) < 1e-12 or np.std(y[keep]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def safe_auc(y, score) -> float:
    y, score = np.asarray(y, float), np.asarray(score, float)
    keep = np.isfinite(y) & np.isfinite(score)
    y, score = y[keep].astype(int), score[keep]
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def combined_metric(cgm_y, cgm_score, colas_y, colas_score) -> float:
    rho = safe_spearman(cgm_y, cgm_score)
    auc = safe_auc(colas_y, colas_score)
    return 0.5 * ((rho + 1.0) / 2.0) + 0.5 * auc


def simultaneous_splits(cgm, colas, folds, seed):
    c_split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    k_split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return zip(c_split.split(cgm), k_split.split(colas, colas["y"]))


def select_lambda(cgm, colas, features, seed) -> float:
    performance = {lam: [] for lam in LAMBDAS}
    for (c_train, c_test), (k_train, k_test) in simultaneous_splits(cgm, colas, INNER_FOLDS, seed):
        for lam in LAMBDAS:
            model = fit_joint(cgm.iloc[c_train], colas.iloc[k_train], features, lam)
            performance[lam].append(combined_metric(
                cgm.iloc[c_test]["insulin"], model_score(model, cgm.iloc[c_test]),
                colas.iloc[k_test]["y"], model_score(model, colas.iloc[k_test]),
            ))
    return max(LAMBDAS, key=lambda lam: (float(np.nanmean(performance[lam])), lam))


def repeated_nested_cv(cgm, colas, features):
    fold_rows = []
    c_predictions = {subject_id: [] for subject_id in cgm["id"]}
    k_predictions = {subject_id: [] for subject_id in colas["id"]}
    lambdas = []
    for repeat in range(OUTER_REPEATS):
        seed = SEED + repeat * 101
        for fold, ((c_train, c_test), (k_train, k_test)) in enumerate(
            simultaneous_splits(cgm, colas, OUTER_FOLDS, seed)
        ):
            lam = select_lambda(cgm.iloc[c_train], colas.iloc[k_train], features, seed + fold + 17)
            model = fit_joint(cgm.iloc[c_train], colas.iloc[k_train], features, lam)
            c_score = model_score(model, cgm.iloc[c_test])
            k_score = model_score(model, colas.iloc[k_test])
            for subject_id, score in zip(cgm.iloc[c_test]["id"], c_score):
                c_predictions[subject_id].append(float(score))
            for subject_id, score in zip(colas.iloc[k_test]["id"], k_score):
                k_predictions[subject_id].append(float(score))
            rho = safe_spearman(cgm.iloc[c_test]["insulin"], c_score)
            auc = safe_auc(colas.iloc[k_test]["y"], k_score)
            fold_rows.append({"repeat": repeat, "fold": fold, "lambda": lam, "cgm_rho": rho, "colas_auc": auc})
            lambdas.append(lam)

    c_oof = np.array([np.mean(c_predictions[subject_id]) for subject_id in cgm["id"]])
    k_oof = np.array([np.mean(k_predictions[subject_id]) for subject_id in colas["id"]])
    fold = pd.DataFrame(fold_rows)
    return {
        "fold_cgm_rho_mean": float(fold["cgm_rho"].mean()),
        "fold_cgm_rho_sd": float(fold["cgm_rho"].std(ddof=1)),
        "fold_colas_auc_mean": float(fold["colas_auc"].mean()),
        "fold_colas_auc_sd": float(fold["colas_auc"].std(ddof=1)),
        "oof_cgm_rho": safe_spearman(cgm["insulin"], c_oof),
        "oof_colas_auc": safe_auc(colas["y"], k_oof),
        "oof_colas_pr_auc": float(average_precision_score(colas["y"], k_oof)),
        "combined_oof": combined_metric(cgm["insulin"], c_oof, colas["y"], k_oof),
        "lambda_counts": {str(key): int(value) for key, value in sorted(Counter(lambdas).items())},
        "oof_cgm_scores": c_oof,
        "oof_colas_scores": k_oof,
    }


def tune_full_lambda(cgm, colas, features):
    values = []
    for repeat in range(OUTER_REPEATS):
        values.append(select_lambda(cgm, colas, features, SEED + 800 + repeat))
    counts = Counter(values)
    chosen = max(LAMBDAS, key=lambda lam: (counts[lam], lam))
    return chosen, {str(key): int(value) for key, value in sorted(counts.items())}


def percentile_interval(values, alpha=0.05):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def bootstrap_auc(y, score, baseline=None, seed=SEED + 1000):
    y, score = np.asarray(y, int), np.asarray(score, float)
    baseline = None if baseline is None else np.asarray(baseline, float)
    rng = np.random.default_rng(seed)
    aucs, deltas = [], []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) != 2:
            continue
        auc = safe_auc(y[index], score[index])
        aucs.append(auc)
        if baseline is not None:
            deltas.append(auc - safe_auc(y[index], baseline[index]))
    result = {"estimate": safe_auc(y, score), "ci95": percentile_interval(aucs), "replicates": len(aucs)}
    if baseline is not None:
        result["delta_estimate"] = safe_auc(y, score) - safe_auc(y, baseline)
        result["delta_ci95"] = percentile_interval(deltas)
    return result


def bootstrap_spearman(x, y, seed=SEED + 1100):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(x), len(x))
        value = safe_spearman(x[index], y[index])
        if np.isfinite(value):
            values.append(value)
    return {"estimate": safe_spearman(x, y), "ci95": percentile_interval(values), "replicates": len(values)}


def permutation_p(y, score, metric, alternative="greater", seed=SEED + 1200):
    y, score = np.asarray(y), np.asarray(score, float)
    observed = metric(y, score)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(PERMUTATIONS):
        null.append(metric(rng.permutation(y), score))
    null = np.asarray(null, float)
    if alternative == "greater":
        count = int(np.sum(null >= observed))
    else:
        count = int(np.sum(np.abs(null) >= abs(observed)))
    return {"estimate": float(observed), "p_value": float((count + 1) / (len(null) + 1)), "permutations": len(null)}


def choose_threshold(y, score, minimum_sensitivity=0.80):
    y, score = np.asarray(y, int), np.asarray(score, float)
    candidates = np.unique(score)
    best = None
    for threshold in candidates:
        pred = score >= threshold
        sensitivity = float(np.mean(pred[y == 1]))
        specificity = float(np.mean(~pred[y == 0]))
        if sensitivity + 1e-12 < minimum_sensitivity:
            continue
        key = (specificity, sensitivity, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return float(np.min(score)) if best is None else best[1]


def threshold_metrics(y, score, threshold):
    y, pred = np.asarray(y, int), np.asarray(score, float) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    return {
        "threshold": float(threshold), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


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


def sensor_reliability(frame, model):
    libre = frame[frame["cohort"] == "cgmacros_libre_w48"].copy()
    dexcom = frame[frame["cohort"] == "cgmacros_dexcom_w48"].copy()
    paired = libre.merge(dexcom, on="id", suffixes=("_libre", "_dexcom"), validate="one_to_one")
    libre_model = libre.set_index("id").loc[paired["id"]].reset_index()
    dexcom_model = dexcom.set_index("id").loc[paired["id"]].reset_index()
    left, right = model_score(model, libre_model), model_score(model, dexcom_model)
    per_feature = {}
    for feature in ALL_FEATURES:
        per_feature[feature] = safe_spearman(paired[f"{feature}_libre"], paired[f"{feature}_dexcom"])
    return {
        "n": len(paired),
        "score_spearman": safe_spearman(left, right),
        "score_icc_absolute_agreement": icc_absolute_agreement(left, right),
        "median_absolute_score_difference": float(np.median(np.abs(left - right))),
        "feature_spearman": per_feature,
    }


def window_stability(frame, model):
    rows = {}
    for source in ("cgmacros_libre", "cgmacros_dexcom", "colas", "hall", "iglu", "dubosson", "weinstock"):
        short = frame[frame["cohort"] == f"{source}_w24"].copy()
        long = frame[frame["cohort"] == f"{source}_w48"].copy()
        common = sorted(set(short["id"]) & set(long["id"]))
        if len(common) < 3:
            continue
        short = short.set_index("id").loc[common].reset_index()
        long = long.set_index("id").loc[common].reset_index()
        a, b = model_score(model, short), model_score(model, long)
        rows[source] = {
            "n": len(common),
            "spearman": safe_spearman(a, b),
            "icc_absolute_agreement": icc_absolute_agreement(a, b),
            "median_absolute_score_difference": float(np.median(np.abs(a - b))),
        }
    return rows


def phase_direction_audit(cgm, colas, hall):
    normal = hall[(hall["diagnosis"] == 0) & hall["SSPG"].notna()]
    result = {}
    for feature in PHASE_FEATURES:
        c_assoc = safe_spearman(cgm[feature], cgm["insulin"])
        k_auc = safe_auc(colas["y"], colas[feature])
        h_assoc = safe_spearman(normal[feature], normal["SSPG"])
        result[feature] = {
            "cgmacros_insulin_spearman": c_assoc,
            "colas_t2dm_auc": k_auc,
            "colas_signed_association": 2.0 * (k_auc - 0.5),
            "hall_normal_sspg_spearman": h_assoc,
        }
    return result


def score_summary(frame, model):
    result = {}
    subset = frame[frame["window_hours"] == 48].copy()
    for cohort, group in subset.groupby("cohort"):
        score = model_score(model, group)
        result[cohort] = {
            "n": len(group),
            "median": float(np.median(score)),
            "q25": float(np.quantile(score, 0.25)),
            "q75": float(np.quantile(score, 0.75)),
        }
    return result


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main():
    frame = load_features()
    cgm = cgmacros_sensor_average(frame, 48)
    cgm = cgm[cgm["insulin"].notna()].reset_index(drop=True)
    colas = frame[(frame["cohort"] == "colas_w48") & frame["y"].notna()].reset_index(drop=True)
    hall = frame[frame["cohort"] == "hall_w48"].reset_index(drop=True)

    family_results = {}
    family_cv = {}
    for name, features in MODEL_FAMILIES.items():
        print(f"nested CV: {name} ({len(features)} features)", flush=True)
        cv = repeated_nested_cv(cgm, colas, features)
        family_cv[name] = cv
        family_results[name] = {key: value for key, value in cv.items() if not key.startswith("oof_")}
        family_results[name].update({
            "oof_cgm_rho": cv["oof_cgm_rho"],
            "oof_colas_auc": cv["oof_colas_auc"],
            "oof_colas_pr_auc": cv["oof_colas_pr_auc"],
        })

    best_value = max(item["combined_oof"] for item in family_results.values())
    eligible_family = [
        name for name, item in family_results.items() if item["combined_oof"] >= best_value - 0.01
    ]
    selected_name = min(eligible_family, key=lambda name: len(MODEL_FAMILIES[name]))
    selected_features = MODEL_FAMILIES[selected_name]
    print(f"selected family: {selected_name}", flush=True)

    hall_normal = hall[(hall["diagnosis"] == 0) & hall["SSPG"].notna()].copy()
    hall_sspg_y = (hall_normal["SSPG"].to_numpy(float) >= 150).astype(int)
    frozen_models = {}
    frozen_family_results = {}
    for name, features in MODEL_FAMILIES.items():
        family_lambda, family_lambda_counts = tune_full_lambda(cgm, colas, features)
        family_model = fit_joint(cgm, colas, features, family_lambda)
        frozen_models[name] = family_model
        c_score = model_score(family_model, cgm)
        k_score = model_score(family_model, colas)
        h_score = model_score(family_model, hall_normal)
        family_sensor = sensor_reliability(frame, family_model)
        frozen_family_results[name] = {
            "lambda": family_lambda,
            "lambda_selection_counts": family_lambda_counts,
            "weights": {
                feature: float(weight) for feature, weight in zip(features, family_model["weights"])
            },
            "standardization": family_model["transformer"].export(),
            "discovery_cgmacros_insulin_spearman": safe_spearman(cgm["insulin"], c_score),
            "discovery_colas_auc": safe_auc(colas["y"], k_score),
            "hall_sspg_binary_auc_descriptive": safe_auc(hall_sspg_y, h_score),
            "hall_continuous_sspg_spearman_descriptive": safe_spearman(h_score, hall_normal["SSPG"]),
            "cgmacros_sensor_score_spearman": family_sensor["score_spearman"],
            "cgmacros_sensor_score_icc": family_sensor["score_icc_absolute_agreement"],
        }

    final_model = frozen_models[selected_name]
    final_lambda = final_model["lambda"]
    final_lambda_counts = frozen_family_results[selected_name]["lambda_selection_counts"]
    final_score_cgm = model_score(final_model, cgm)
    final_score_colas = model_score(final_model, colas)
    final_score_hall = model_score(final_model, hall)

    hall_normal_score = model_score(final_model, hall_normal)
    hall_night = hall_normal["night_mean"].to_numpy(float)
    hall_binary = {
        "n": len(hall_normal),
        "positive": int(hall_sspg_y.sum()),
        "omega_auc": bootstrap_auc(hall_sspg_y, hall_normal_score, hall_night),
        "omega_pr_auc": float(average_precision_score(hall_sspg_y, hall_normal_score)),
        "night_mean_auc": safe_auc(hall_sspg_y, hall_night),
        "auc_permutation": permutation_p(hall_sspg_y, hall_normal_score, safe_auc, "greater"),
    }
    hall_continuous = {
        "omega_vs_sspg": bootstrap_spearman(hall_normal_score, hall_normal["SSPG"]),
        "omega_vs_sspg_permutation": permutation_p(
            hall_normal["SSPG"].to_numpy(float), hall_normal_score,
            lambda y, score: safe_spearman(score, y), "two-sided", SEED + 1300,
        ),
        "night_mean_vs_sspg": safe_spearman(hall_night, hall_normal["SSPG"]),
    }

    hall_diag_y = hall["y"].to_numpy(int)
    hall_diagnosis = {
        "n": len(hall),
        "positive": int(hall_diag_y.sum()),
        "omega_auc": bootstrap_auc(hall_diag_y, final_score_hall, hall["night_mean"].to_numpy(float), SEED + 1400),
        "omega_pr_auc": float(average_precision_score(hall_diag_y, final_score_hall)),
        "night_mean_auc": safe_auc(hall_diag_y, hall["night_mean"]),
    }

    threshold = choose_threshold(colas["y"], final_score_colas)
    thresholds = {
        "calibration_note": "threshold selected on the full Colas discovery fit; exploratory transport test only",
        "colas": threshold_metrics(colas["y"], final_score_colas, threshold),
        "hall_diagnosis": threshold_metrics(hall_diag_y, final_score_hall, threshold),
    }

    sensor = sensor_reliability(frame, final_model)
    stability = window_stability(frame, final_model)
    direction = phase_direction_audit(cgm, colas, hall)

    auc_gate = hall_binary["omega_auc"]
    continuous_gate = hall_continuous["omega_vs_sspg_permutation"]
    key_stabilities = [
        stability.get(name, {}).get("spearman", np.nan)
        for name in ("cgmacros_libre", "cgmacros_dexcom", "colas", "hall")
    ]
    gates = {
        "cgmacros_sensor_spearman_gte_0_70": sensor["score_spearman"] >= 0.70,
        "cgmacros_sensor_icc_gte_0_70": sensor["score_icc_absolute_agreement"] >= 0.70,
        "hall_sspg_auc_gte_0_75": auc_gate["estimate"] >= 0.75,
        "hall_sspg_auc_ci_lower_gt_0_50": auc_gate["ci95"][0] > 0.50,
        "hall_delta_auc_vs_night_mean_gte_0_05": auc_gate["delta_estimate"] >= 0.05,
        "hall_delta_auc_ci_lower_gt_0": auc_gate["delta_ci95"][0] > 0,
        "hall_continuous_sspg_positive_p_lte_0_05": (
            hall_continuous["omega_vs_sspg"]["estimate"] > 0 and continuous_gate["p_value"] <= 0.05
        ),
        "all_primary_24_48_spearman_gte_0_70": bool(
            all(np.isfinite(value) and value >= 0.70 for value in key_stabilities)
        ),
    }
    deployment_eligible = bool(all(gates.values()))

    formula = {
        "name": "Omega_G composite glycemic abnormality residual",
        "semantics": "dimensionless fitted ordering coordinate; not a disease probability",
        "feature_transform": {
            feature: "log1p(max(x,0))" if feature in {
                "hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "volume"
            } else "identity"
            for feature in selected_features
        },
        "standardization": final_model["transformer"].export(),
        "weights": {
            feature: float(weight) for feature, weight in zip(selected_features, final_model["weights"])
        },
        "lambda": final_lambda,
        "lambda_selection_counts": final_lambda_counts,
        "equation": "Omega_G = sum_j weight_j * (transform(x_j)-center_j)/scale_j",
    }

    scored = frame.copy()
    scored["omega_g"] = model_score(final_model, scored)
    score_center = float(np.median(final_score_colas))
    score_scale = float((np.quantile(final_score_colas, 0.75) - np.quantile(final_score_colas, 0.25)) / 1.349)
    score_scale = score_scale if score_scale > 1e-8 else 1.0
    scored["omega_g_0_100_display"] = 100.0 * expit((scored["omega_g"] - score_center) / score_scale)
    scored.to_csv(FEATURE_PATH, index=False)

    result = {
        "protocol": {
            "frozen_date": "2026-08-10",
            "primary_window_hours": 48,
            "sensitivity_window_hours": 24,
            "discovery": ["CGMacros fasting insulin", "Colas T2DM"],
            "untouched_external_validation": ["Hall diagnosis-normal SSPG", "Hall diagnosis"],
            "stress_only": ["Iglu", "Dubosson", "Weinstock"],
            "model_families": {key: list(value) for key, value in MODEL_FAMILIES.items()},
            "outer_cv": f"{OUTER_REPEATS} repeats x {OUTER_FOLDS} folds; simultaneous subject splits",
            "inner_cv_folds": INNER_FOLDS,
            "lambda_grid": list(LAMBDAS),
            "family_selection": "maximum combined OOF; choose smallest family within 0.01",
        },
        "data": {
            "merged_rows": len(frame),
            "cgmacros_discovery_subjects": len(cgm),
            "colas_discovery_subjects": len(colas),
            "colas_positive": int(colas["y"].sum()),
            "hall_validation_subjects": len(hall),
            "hall_normal_sspg_complete": len(hall_normal),
        },
        "family_results": family_results,
        "frozen_family_fits": frozen_family_results,
        "selected_family": selected_name,
        "formula": formula,
        "discovery_fit": {
            "cgmacros_insulin_spearman": safe_spearman(cgm["insulin"], final_score_cgm),
            "colas_auc": safe_auc(colas["y"], final_score_colas),
            "colas_pr_auc": float(average_precision_score(colas["y"], final_score_colas)),
        },
        "hall_hidden_sspg_binary": hall_binary,
        "hall_hidden_sspg_continuous": hall_continuous,
        "hall_diagnosis": hall_diagnosis,
        "thresholds": thresholds,
        "cgmacros_sensor_reliability": sensor,
        "window_stability": stability,
        "phase_direction_audit": direction,
        "cohort_score_summary": score_summary(frame, final_model),
        "deployment_gates": gates,
        "deployment_eligible": deployment_eligible,
        "html_decision": "update index.html" if deployment_eligible else "retain existing index.html; experimental parameter did not pass every frozen gate",
        "limitations": [
            "Only 16 Colas positives remain after the common 48-hour quality gate.",
            "CGMacros fasting insulin, Hall SSPG, and diabetes diagnosis are related but non-identical endpoints.",
            "Treatment and diabetes type are not common input fields and therefore cannot enter Omega_G.",
            "The 24-hour versus 48-hour comparison tests short-window stability, not long-term repeatability.",
            "The 0-100 column is a monotone display mapping centered on the Colas discovery distribution; 50 is not a health threshold.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_ready({
        "selected_family": selected_name,
        "formula": formula,
        "hall_hidden_sspg_binary": hall_binary,
        "hall_hidden_sspg_continuous": hall_continuous,
        "sensor_reliability": sensor,
        "deployment_gates": gates,
        "deployment_eligible": deployment_eligible,
    }), indent=2, ensure_ascii=False))
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"wrote {FEATURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
