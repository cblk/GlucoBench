#!/usr/bin/env python3
"""Fit the nine-dimensional CGM space to a soft clinical continuum score.

CCAS-core (A1C + fasting plasma glucose) is the discovery target in CGMacros
and Colas. Hall is untouched until the full nine-feature model is frozen, then
CCAS-full adds 75 g OGTT and direct SSPG insulin-resistance information.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold

from validate_composite_abnormality import (
    ALL_FEATURES,
    BASE_FEATURES,
    MODEL_FAMILIES,
    OUTPUT,
    PHASE_FEATURES,
    ROOT,
    RobustTransformer,
    bootstrap_auc,
    cgmacros_sensor_average,
    choose_threshold,
    json_ready,
    load_features,
    model_score,
    safe_auc,
    safe_spearman,
    sensor_reliability,
    window_stability,
)


RESULT_PATH = OUTPUT / "clinical_continuum_results.json"
FEATURE_PATH = OUTPUT / "clinical_continuum_features.csv"

SEED = 20260810
OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 3
LAMBDAS = (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
ELASTIC_ALPHA = 0.50
BOOTSTRAPS = 5000
PERMUTATIONS = 5000

PRIMARY_SSPG_SCALE = 25.0
SSPG_SCALE_SENSITIVITY = (15.0, 25.0, 35.0)


def id_key(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def soft_component(value, prediabetes_threshold, diabetes_threshold):
    value = np.asarray(value, float)
    return expit(4.0 * (value - prediabetes_threshold) / (diabetes_threshold - prediabetes_threshold))


def max_mean(values: np.ndarray) -> np.ndarray:
    return 0.6 * np.max(values, axis=1) + 0.4 * np.mean(values, axis=1)


def ccas_core(a1c, fpg):
    components = np.column_stack([
        soft_component(a1c, 5.7, 6.5),
        soft_component(fpg, 100.0, 126.0),
    ])
    return 100.0 * max_mean(components)


def ccas_full(a1c, fpg, ogtt, sspg, sspg_scale=PRIMARY_SSPG_SCALE):
    glycemic = max_mean(np.column_stack([
        soft_component(a1c, 5.7, 6.5),
        soft_component(fpg, 100.0, 126.0),
        soft_component(ogtt, 140.0, 200.0),
    ]))
    insulin_resistance = expit((np.asarray(sspg, float) - 150.0) / sspg_scale)
    return 100.0 * max_mean(np.column_stack([glycemic, insulin_resistance]))


def leave_one_out_threshold_metrics(y, score, minimum_sensitivity=0.80):
    """Exploratory operating-point estimate without evaluating a row on its own label."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    predictions, thresholds = [], []
    for index in range(len(y)):
        train = np.arange(len(y)) != index
        threshold = choose_threshold(y[train], score[train], minimum_sensitivity)
        thresholds.append(threshold)
        predictions.append(score[index] >= threshold)
    predictions = np.asarray(predictions, bool)
    tp = int(np.sum(predictions & (y == 1)))
    fn = int(np.sum(~predictions & (y == 1)))
    tn = int(np.sum(~predictions & (y == 0)))
    fp = int(np.sum(predictions & (y == 0)))
    return {
        "threshold_selection": f"leave-one-out; training sensitivity >= {minimum_sensitivity:.2f}",
        "median_threshold": float(np.median(thresholds)),
        "threshold_range": [float(np.min(thresholds)), float(np.max(thresholds))],
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "balanced_accuracy": 0.5 * (
            tp / (tp + fn) + tn / (tn + fp)
        ) if tp + fn and tn + fp else None,
    }


def load_clinical_tables():
    with ZipFile(ROOT / "raw_data.zip") as archive:
        with archive.open("raw_data/colas.csv") as handle:
            colas = pd.read_csv(handle).groupby("id", as_index=False).first()
        with archive.open("raw_data/hall.csv") as handle:
            hall = pd.read_csv(handle).groupby("id", as_index=False).first()

    colas = colas.rename(columns={"glycaemia": "fpg", "HbA1c": "a1c"})
    colas["id"] = colas["id"].map(id_key)
    hall = hall.rename(columns={"A1C": "a1c", "FBG": "fpg", "ogtt.2hr": "ogtt"})
    hall["id"] = hall["id"].map(id_key)

    bio = pd.read_csv(OUTPUT / "cgmacros_subset" / "bio.csv")
    bio.columns = [str(column).strip() for column in bio.columns]
    bio = bio.rename(columns={
        "subject": "id",
        "A1c PDL (Lab)": "a1c",
        "Fasting GLU - PDL (Lab)": "fpg",
        "Insulin": "insulin",
    })
    bio["id"] = bio["id"].map(id_key)
    return colas, hall, bio


def prepare_data():
    feature_rows = load_features()
    previous = pd.read_csv(OUTPUT / "composite_abnormality_features.csv")[["cohort", "id", "omega_g"]]
    previous["id"] = previous["id"].map(id_key)
    feature_rows["id"] = feature_rows["id"].map(id_key)
    feature_rows = feature_rows.merge(previous, on=["cohort", "id"], how="left", validate="one_to_one")
    cgm_features = cgmacros_sensor_average(feature_rows, 48)
    cgm_features["id"] = cgm_features["id"].map(id_key)
    cgm_previous = (
        feature_rows[feature_rows["source_cohort"] == "cgmacros"]
        .query("window_hours == 48")
        .groupby("id", as_index=False)["omega_g"].mean()
    )
    cgm_features = cgm_features.merge(cgm_previous, on="id", how="left", validate="one_to_one")
    colas_features = feature_rows[feature_rows["cohort"] == "colas_w48"].copy()
    hall_features = feature_rows[feature_rows["cohort"] == "hall_w48"].copy()
    colas_features["id"] = colas_features["id"].map(id_key)
    hall_features["id"] = hall_features["id"].map(id_key)

    colas_clinical, hall_clinical, bio = load_clinical_tables()
    cgm = cgm_features.drop(columns=["a1c", "insulin"], errors="ignore").merge(
        bio[["id", "a1c", "fpg", "insulin"]], on="id", how="inner", validate="one_to_one"
    )
    colas = colas_features.drop(columns=["a1c"], errors="ignore").merge(
        colas_clinical[["id", "a1c", "fpg", "T2DM"]], on="id", how="inner", validate="one_to_one"
    )
    hall = hall_features.drop(columns=["a1c", "insulin", "SSPG", "diagnosis"], errors="ignore").merge(
        hall_clinical[["id", "a1c", "fpg", "ogtt", "insulin", "SSPG", "diagnosis"]],
        on="id", how="inner", validate="one_to_one"
    )

    for frame in (cgm, colas, hall):
        for column in ("a1c", "fpg", "ogtt", "insulin", "SSPG", "diagnosis"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame.loc[frame[column] < 0, column] = np.nan

    cgm = cgm[cgm[["a1c", "fpg"]].notna().all(axis=1)].reset_index(drop=True)
    colas = colas[colas[["a1c", "fpg"]].notna().all(axis=1)].reset_index(drop=True)
    hall = hall.reset_index(drop=True)
    cgm["ccas_core"] = ccas_core(cgm["a1c"], cgm["fpg"])
    colas["ccas_core"] = ccas_core(colas["a1c"], colas["fpg"])
    hall_core = hall[["a1c", "fpg"]].notna().all(axis=1)
    hall.loc[hall_core, "ccas_core"] = ccas_core(hall.loc[hall_core, "a1c"], hall.loc[hall_core, "fpg"])

    hall_full = hall[["a1c", "fpg", "ogtt", "SSPG"]].notna().all(axis=1)
    hall.loc[hall_full, "ccas_full"] = ccas_full(
        hall.loc[hall_full, "a1c"], hall.loc[hall_full, "fpg"],
        hall.loc[hall_full, "ogtt"], hall.loc[hall_full, "SSPG"],
    )
    cgm["homa_ir"] = cgm["fpg"] * cgm["insulin"] / 405.0
    hall_homa = hall[["fpg", "insulin"]].notna().all(axis=1)
    hall.loc[hall_homa, "homa_ir"] = hall.loc[hall_homa, "fpg"] * hall.loc[hall_homa, "insulin"] / 405.0
    return feature_rows, cgm, colas, hall


def fit_continuum(cgm, colas, features, lam):
    transformer = RobustTransformer(features).fit([cgm, colas])
    xc, xk = transformer.transform(cgm), transformer.transform(colas)
    x = np.vstack([xc, xk])
    y = np.r_[cgm["ccas_core"].to_numpy(float), colas["ccas_core"].to_numpy(float)] / 100.0
    sample_weight = np.r_[
        np.full(len(cgm), 0.5 / len(cgm)),
        np.full(len(colas), 0.5 / len(colas)),
    ]
    p = len(features)

    def objective(params):
        weights, intercept = params[:p], params[p]
        residual = intercept + x @ weights - y
        smooth_l1 = np.sqrt(weights ** 2 + 1e-8)
        loss = 0.5 * np.sum(sample_weight * residual ** 2)
        penalty = lam * (
            ELASTIC_ALPHA * smooth_l1.sum()
            + 0.5 * (1.0 - ELASTIC_ALPHA) * np.dot(weights, weights)
        )
        grad_w = x.T @ (sample_weight * residual)
        grad_w += lam * (
            ELASTIC_ALPHA * weights / smooth_l1
            + (1.0 - ELASTIC_ALPHA) * weights
        )
        return loss + penalty, np.r_[grad_w, np.sum(sample_weight * residual)]

    initial = np.zeros(p + 1)
    initial[p] = float(np.sum(sample_weight * y))
    bounds = [((0.0, None) if feature in BASE_FEATURES else (None, None)) for feature in features]
    bounds.append((None, None))
    fit = minimize(
        lambda params: objective(params), initial, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fit.success:
        raise RuntimeError(f"continuum optimization failed: {fit.message}")
    return {
        "features": tuple(features),
        "transformer": transformer,
        "weights": fit.x[:p],
        "intercept": float(fit.x[p]),
        "lambda": float(lam),
        "objective": float(fit.fun),
    }


def predict_ccas(model, frame):
    return 100.0 * (model["intercept"] + model_score(model, frame))


def r2_score(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    keep = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[keep], prediction[keep]
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator) if denominator > 1e-12 else float("nan")


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    keep = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[keep], prediction[keep]
    return {
        "n": len(y),
        "spearman": safe_spearman(y, prediction),
        "r2": r2_score(y, prediction),
        "mae": float(np.mean(np.abs(y - prediction))),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
    }


def simultaneous_splits(cgm, colas, folds, seed):
    c_split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    k_split = KFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    return zip(c_split.split(cgm), k_split.split(colas))


def select_lambda(cgm, colas, features, seed):
    errors = {lam: [] for lam in LAMBDAS}
    for (ct, cv), (kt, kv) in simultaneous_splits(cgm, colas, INNER_FOLDS, seed):
        for lam in LAMBDAS:
            model = fit_continuum(cgm.iloc[ct], colas.iloc[kt], features, lam)
            c_error = regression_metrics(cgm.iloc[cv]["ccas_core"], predict_ccas(model, cgm.iloc[cv]))["rmse"]
            k_error = regression_metrics(colas.iloc[kv]["ccas_core"], predict_ccas(model, colas.iloc[kv]))["rmse"]
            errors[lam].append(0.5 * (c_error + k_error))
    return min(LAMBDAS, key=lambda lam: (float(np.mean(errors[lam])), -lam))


def repeated_nested_cv(cgm, colas, features):
    c_predictions = {subject_id: [] for subject_id in cgm["id"]}
    k_predictions = {subject_id: [] for subject_id in colas["id"]}
    fold_rows, lambdas = [], []
    for repeat in range(OUTER_REPEATS):
        seed = SEED + repeat * 101
        for fold, ((ct, cv), (kt, kv)) in enumerate(simultaneous_splits(cgm, colas, OUTER_FOLDS, seed)):
            lam = select_lambda(cgm.iloc[ct], colas.iloc[kt], features, seed + fold + 31)
            model = fit_continuum(cgm.iloc[ct], colas.iloc[kt], features, lam)
            c_pred, k_pred = predict_ccas(model, cgm.iloc[cv]), predict_ccas(model, colas.iloc[kv])
            for subject_id, value in zip(cgm.iloc[cv]["id"], c_pred):
                c_predictions[subject_id].append(float(value))
            for subject_id, value in zip(colas.iloc[kv]["id"], k_pred):
                k_predictions[subject_id].append(float(value))
            cm = regression_metrics(cgm.iloc[cv]["ccas_core"], c_pred)
            km = regression_metrics(colas.iloc[kv]["ccas_core"], k_pred)
            fold_rows.append({"repeat": repeat, "fold": fold, "lambda": lam, "cgm": cm, "colas": km})
            lambdas.append(lam)

    c_oof = np.array([np.mean(c_predictions[subject_id]) for subject_id in cgm["id"]])
    k_oof = np.array([np.mean(k_predictions[subject_id]) for subject_id in colas["id"]])
    c_metrics = regression_metrics(cgm["ccas_core"], c_oof)
    k_metrics = regression_metrics(colas["ccas_core"], k_oof)
    return {
        "cgm_oof": c_metrics,
        "colas_oof": k_metrics,
        "equal_cohort_mean_spearman": 0.5 * (c_metrics["spearman"] + k_metrics["spearman"]),
        "equal_cohort_mean_rmse": 0.5 * (c_metrics["rmse"] + k_metrics["rmse"]),
        "lambda_counts": {str(key): int(value) for key, value in sorted(Counter(lambdas).items())},
        "cgm_predictions": c_oof,
        "colas_predictions": k_oof,
    }


def tune_full_lambda(cgm, colas, features):
    values = [select_lambda(cgm, colas, features, SEED + 900 + repeat) for repeat in range(OUTER_REPEATS)]
    counts = Counter(values)
    chosen = min(LAMBDAS, key=lambda lam: (-counts[lam], lam))
    return chosen, {str(key): int(value) for key, value in sorted(counts.items())}


def weighted_linear_baseline(cgm, colas, column):
    x = np.r_[cgm[column].to_numpy(float), colas[column].to_numpy(float)]
    y = np.r_[cgm["ccas_core"].to_numpy(float), colas["ccas_core"].to_numpy(float)]
    weights = np.r_[np.full(len(cgm), 0.5 / len(cgm)), np.full(len(colas), 0.5 / len(colas))]
    design = np.column_stack([np.ones(len(x)), x])
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y)
    coefficients = np.linalg.pinv(normal) @ rhs
    return {"intercept": float(coefficients[0]), "slope": float(coefficients[1])}


def baseline_predict(model, values):
    return model["intercept"] + model["slope"] * np.asarray(values, float)


def percentile_interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else [None, None]


def bootstrap_regression(y, prediction, baseline=None, seed=SEED + 2000):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    baseline = None if baseline is None else np.asarray(baseline, float)
    rng = np.random.default_rng(seed)
    rho, delta_rho, r2_values, mae_values = [], [], [], []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(y), len(y))
        current = safe_spearman(y[index], prediction[index])
        if np.isfinite(current):
            rho.append(current)
            if baseline is not None:
                delta_rho.append(current - safe_spearman(y[index], baseline[index]))
        r2_values.append(r2_score(y[index], prediction[index]))
        mae_values.append(float(np.mean(np.abs(y[index] - prediction[index]))))
    metrics = regression_metrics(y, prediction)
    metrics.update({
        "spearman_ci95": percentile_interval(rho),
        "r2_ci95": percentile_interval(r2_values),
        "mae_ci95": percentile_interval(mae_values),
        "bootstrap_replicates": BOOTSTRAPS,
    })
    if baseline is not None:
        metrics["delta_spearman_vs_baseline"] = metrics["spearman"] - safe_spearman(y, baseline)
        metrics["delta_spearman_ci95"] = percentile_interval(delta_rho)
    return metrics


def correlation_permutation(y, prediction, seed=SEED + 2100):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    observed = safe_spearman(y, prediction)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(PERMUTATIONS):
        if safe_spearman(rng.permutation(y), prediction) >= observed:
            count += 1
    return {"estimate": observed, "one_sided_p": (count + 1) / (PERMUTATIONS + 1), "permutations": PERMUTATIONS}


def formula_export(model, lambda_counts):
    return {
        "features": list(model["features"]),
        "transform": {
            feature: "log1p(max(x,0))" if feature in {
                "hyper_burden", "hypo_burden", "variation_load", "recovery_debt", "volume"
            } else "identity"
            for feature in model["features"]
        },
        "standardization": model["transformer"].export(),
        "weights": {feature: float(weight) for feature, weight in zip(model["features"], model["weights"])},
        "intercept": model["intercept"],
        "lambda": model["lambda"],
        "lambda_selection_counts": lambda_counts,
        "equation": "predicted_CCAS = 100*(intercept + sum_j weight_j*(transform(x_j)-center_j)/scale_j)",
    }


def main():
    feature_rows, cgm, colas, hall = prepare_data()

    cv_results, final_models, frozen_fits = {}, {}, {}
    for name, features in MODEL_FAMILIES.items():
        print(f"clinical continuum nested CV: {name}", flush=True)
        cv = repeated_nested_cv(cgm, colas, features)
        cv_results[name] = {key: value for key, value in cv.items() if not key.endswith("predictions")}
        lam, counts = tune_full_lambda(cgm, colas, features)
        model = fit_continuum(cgm, colas, features, lam)
        final_models[name] = model
        frozen_fits[name] = {
            "formula": formula_export(model, counts),
            "cgmacros_full_fit": regression_metrics(cgm["ccas_core"], predict_ccas(model, cgm)),
            "colas_full_fit": regression_metrics(colas["ccas_core"], predict_ccas(model, colas)),
        }

    primary = final_models["full9"]
    base = final_models["base5"]
    phase = final_models["phase4"]
    night_model = weighted_linear_baseline(cgm, colas, "night_mean")
    previous_model = weighted_linear_baseline(cgm, colas, "omega_g")

    hall_core = hall[hall["ccas_core"].notna()].copy()
    hall_full = hall[hall["ccas_full"].notna()].copy()
    full_prediction = predict_ccas(primary, hall_full)
    base_prediction = predict_ccas(base, hall_full)
    phase_prediction = predict_ccas(phase, hall_full)
    night_prediction = baseline_predict(night_model, hall_full["night_mean"])
    previous_prediction = baseline_predict(previous_model, hall_full["omega_g"])

    hall_full_metrics = {
        "full9": bootstrap_regression(hall_full["ccas_full"], full_prediction, night_prediction),
        "base5": bootstrap_regression(
            hall_full["ccas_full"], base_prediction, night_prediction, SEED + 2150
        ),
        "phase4": regression_metrics(hall_full["ccas_full"], phase_prediction),
        "night_mean": regression_metrics(hall_full["ccas_full"], night_prediction),
        "previous_omega_g": regression_metrics(hall_full["ccas_full"], previous_prediction),
        "full9_permutation": correlation_permutation(hall_full["ccas_full"], full_prediction),
    }

    core_prediction = predict_ccas(primary, hall_core)
    hall_core_metrics = bootstrap_regression(
        hall_core["ccas_core"], core_prediction,
        baseline_predict(night_model, hall_core["night_mean"]), SEED + 2200,
    )

    surface_normal = hall_full[(hall_full["a1c"] < 5.7) & (hall_full["fpg"] < 100)].copy()
    hidden_y = ((surface_normal["ogtt"] >= 140) | (surface_normal["SSPG"] >= 150)).astype(int).to_numpy()
    hidden_full9 = predict_ccas(primary, surface_normal)
    hidden_base = predict_ccas(base, surface_normal)
    hidden_night = surface_normal["night_mean"].to_numpy(float)
    hidden_previous = surface_normal["omega_g"].to_numpy(float)
    hidden_result = {
        "n_surface_normal": len(surface_normal),
        "hidden_positive": int(hidden_y.sum()),
        "fully_low_reference": int((hidden_y == 0).sum()),
        "full9": bootstrap_auc(hidden_y, hidden_full9, hidden_night, SEED + 2300),
        "base5": bootstrap_auc(hidden_y, hidden_base, hidden_night, SEED + 2350),
        "night_mean_auc": safe_auc(hidden_y, hidden_night),
        "previous_omega_g_auc": safe_auc(hidden_y, hidden_previous),
        "exploratory_loso_operating_points": {
            "full9": leave_one_out_threshold_metrics(hidden_y, hidden_full9),
            "base5": leave_one_out_threshold_metrics(hidden_y, hidden_base),
            "night_mean": leave_one_out_threshold_metrics(hidden_y, hidden_night),
        },
    }

    cgm_homa_prediction = predict_ccas(primary, cgm)
    hall_homa = hall[hall["homa_ir"].notna()].copy()
    homa_result = {
        "cgmacros": {
            "n": len(cgm),
            "prediction_vs_log1p_homa_spearman": safe_spearman(cgm_homa_prediction, np.log1p(cgm["homa_ir"])),
            "ccas_core_vs_log1p_homa_spearman": safe_spearman(cgm["ccas_core"], np.log1p(cgm["homa_ir"])),
        },
        "hall": {
            "n": len(hall_homa),
            "prediction_vs_log1p_homa_spearman": safe_spearman(
                predict_ccas(primary, hall_homa), np.log1p(hall_homa["homa_ir"])
            ),
            "ccas_core_vs_log1p_homa_spearman": safe_spearman(
                hall_homa["ccas_core"], np.log1p(hall_homa["homa_ir"])
            ),
        },
    }

    sensitivity = {}
    for scale in SSPG_SCALE_SENSITIVITY:
        target = ccas_full(hall_full["a1c"], hall_full["fpg"], hall_full["ogtt"], hall_full["SSPG"], scale)
        sensitivity[str(int(scale))] = regression_metrics(target, full_prediction)

    sensor = sensor_reliability(feature_rows, primary)
    stability = window_stability(feature_rows, primary)
    full9_delta_discovery = (
        cv_results["full9"]["equal_cohort_mean_spearman"]
        - cv_results["base5"]["equal_cohort_mean_spearman"]
    )

    gates = {
        "hall_full_spearman_gte_0_50": hall_full_metrics["full9"]["spearman"] >= 0.50,
        "hall_full_spearman_ci_lower_gt_0_20": hall_full_metrics["full9"]["spearman_ci95"][0] > 0.20,
        "hidden_abnormal_auc_gte_0_70": hidden_result["full9"]["estimate"] >= 0.70,
        "hidden_delta_auc_vs_night_mean_gte_0_05": hidden_result["full9"]["delta_estimate"] >= 0.05,
        "hidden_delta_auc_ci_lower_gt_0": hidden_result["full9"]["delta_ci95"][0] > 0,
        "full9_discovery_delta_spearman_vs_base5_gte_0_03": full9_delta_discovery >= 0.03,
        "cgmacros_sensor_spearman_gte_0_70": sensor["score_spearman"] >= 0.70,
        "cgmacros_sensor_icc_gte_0_70": sensor["score_icc_absolute_agreement"] >= 0.70,
        "colas_24_48_spearman_gte_0_70": stability.get("colas", {}).get("spearman", -np.inf) >= 0.70,
        "hall_24_48_spearman_gte_0_70": stability.get("hall", {}).get("spearman", -np.inf) >= 0.70,
        "sspg_scale_sensitivity_same_positive_direction": all(
            item["spearman"] > 0 for item in sensitivity.values()
        ),
    }
    deployment_eligible = bool(all(gates.values()))

    scored_rows = []
    for name, data in (("cgmacros", cgm), ("colas", colas), ("hall", hall)):
        copy = data.copy()
        copy["analysis_cohort"] = name
        copy["predicted_ccas_full9"] = predict_ccas(primary, copy)
        copy["predicted_ccas_base5"] = predict_ccas(base, copy)
        scored_rows.append(copy)
    pd.concat(scored_rows, ignore_index=True, sort=False).to_csv(FEATURE_PATH, index=False)

    result = {
        "protocol": {
            "frozen_date": "2026-08-10",
            "clinical_score_name": "CCAS clinical continuum abnormality score",
            "ccas_core": "100*(0.6*max(q_A1C,q_FPG)+0.4*mean(q_A1C,q_FPG))",
            "ccas_full": "100*(0.6*max(glycemic_core_with_OGTT,q_SSPG)+0.4*mean(...))",
            "component_mapping": "sigmoid(4*(x-prediabetes)/(diabetes-prediabetes))",
            "thresholds": {
                "A1C": {"prediabetes": 5.7, "diabetes": 6.5},
                "FPG_mg_dL": {"prediabetes": 100, "diabetes": 126},
                "OGTT_2h_mg_dL": {"prediabetes": 140, "diabetes": 200},
                "SSPG_mg_dL": {"center": 150, "primary_scale": PRIMARY_SSPG_SCALE},
            },
            "primary_model": "full9 fixed before Hall",
            "ablations": ["base5", "phase4"],
            "discovery": ["CGMacros CCAS-core", "Colas CCAS-core"],
            "external_validation": ["Hall CCAS-full", "Hall hidden abnormal subgroup"],
            "stress_only": ["Iglu", "Dubosson", "Weinstock"],
            "outer_cv": f"{OUTER_REPEATS} repeats x {OUTER_FOLDS} folds",
            "inner_cv_folds": INNER_FOLDS,
            "equal_cohort_loss_weight": True,
        },
        "data": {
            "cgmacros_discovery": len(cgm),
            "colas_discovery": len(colas),
            "hall_core_complete": len(hall_core),
            "hall_full_complete": len(hall_full),
            "hall_surface_normal_complete": len(surface_normal),
        },
        "discovery_nested_cv": cv_results,
        "frozen_fits": frozen_fits,
        "primary_formula": formula_export(primary, frozen_fits["full9"]["formula"]["lambda_selection_counts"]),
        "hall_ccas_full": hall_full_metrics,
        "hall_ccas_core": hall_core_metrics,
        "hall_hidden_abnormal": hidden_result,
        "homa_ir_audit": homa_result,
        "sspg_scale_sensitivity": sensitivity,
        "cgmacros_sensor_reliability": sensor,
        "window_stability": stability,
        "full9_discovery_delta_spearman_vs_base5": full9_delta_discovery,
        "deployment_gates": gates,
        "deployment_eligible": deployment_eligible,
        "html_decision": "update index.html" if deployment_eligible else "retain index.html; CCAS-fitted full9 model failed frozen deployment gates",
        "limitations": [
            "Colas glycaemia is treated as the dataset fasting glycaemia field; source metadata do not provide a separate fasting-duration audit.",
            "SSPG is a direct research measure of insulin-mediated glucose disposal, but 150 mg/dL is a study anchor rather than a universal diagnostic threshold.",
            "CCAS is an experimental validation continuum, not a clinical diagnostic score.",
            "Hall complete-case validation may be selected by availability of SSPG and other laboratory tests.",
            "Iglu, Dubosson, and Weinstock lack the required laboratory endpoints and cannot validate CCAS.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_ready({
        "data": result["data"],
        "discovery_nested_cv": cv_results,
        "primary_formula": result["primary_formula"],
        "hall_ccas_full": hall_full_metrics,
        "hall_hidden_abnormal": hidden_result,
        "deployment_gates": gates,
        "deployment_eligible": deployment_eligible,
    }), indent=2, ensure_ascii=False))
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"wrote {FEATURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
