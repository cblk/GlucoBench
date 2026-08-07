#!/usr/bin/env python3
"""Validate the preregistered within-person structural-reserve hypothesis.

Primary construct:
  physical arm = median robust-z of volume expansion, recovery debt, core shift
  dynamic arm  = median robust-z of Lyapunov loss, DET gain, ENTR gain
  consensus    = sqrt(max(physical, 0) * max(dynamic, 0))

All reference centers/scales used for outcome models are fitted inside the
training fold. Dimension and shape-ratio changes are explanatory geometry only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from validate_context_consensus import (
    average_precision,
    choose_threshold,
    fit_logistic,
    predict_logistic,
    rankdata,
    roc_auc,
    screening_metrics,
    spearman,
    stratified_folds,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output" / "structure_reserve_metrics.json"
OUTPUT = ROOT / "output" / "structure_reserve_results.json"
SEED = 20260807
N_REPEATS = 20
N_SPLITS = 5
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000

PHYSICAL = ["volumeExpansion", "recoveryDebt", "coreShift"]
DYNAMIC = ["lyapunovLoss", "detGain", "entrGain"]
GEOMETRY = ["dimensionShift", "shapeShift"]
RESERVE_FEATURES = PHYSICAL + DYNAMIC + GEOMETRY
CONVENTIONAL = ["nightMean", "meanRise", "logCVRatio", "outRise"]
MODEL_FEATURES = {
    "night_mean": ["nightMean"],
    "conventional": CONVENTIONAL,
    "conventional_plus_physical": CONVENTIONAL + ["physicalArm"],
    "conventional_plus_dynamic": CONVENTIONAL + ["dynamicArm"],
    "conventional_plus_consensus": CONVENTIONAL + ["reserveConsensus"],
    "conventional_plus_geometry": CONVENTIONAL + ["geometryShift"],
}


def finite(value):
    return np.asarray(value, float)


def safe_log_ratio(numerator, denominator):
    numerator, denominator = finite(numerator), finite(denominator)
    result = np.full(len(numerator), np.nan)
    keep = np.isfinite(numerator) & np.isfinite(denominator) & (numerator > 0) & (denominator > 0)
    result[keep] = np.log(numerator[keep] / denominator[keep])
    return result


def state_contrasts(frame, pseudo=False):
    if pseudo:
        reference, challenge, core = "pseudoNightEarly", "pseudoNightLate", "pseudoCoreShift"
    else:
        reference, challenge, core = "night", "daytime", "coreShift"
    result = pd.DataFrame(index=frame.index)
    result["volumeExpansion"] = safe_log_ratio(frame[f"{challenge}.volume"], frame[f"{reference}.volume"])
    result["recoveryDebt"] = safe_log_ratio(frame[f"{reference}.recovery"], frame[f"{challenge}.recovery"])
    result["coreShift"] = pd.to_numeric(frame[core], errors="coerce")
    result["lyapunovLoss"] = (
        pd.to_numeric(frame[f"{reference}.lyapunov"], errors="coerce")
        - pd.to_numeric(frame[f"{challenge}.lyapunov"], errors="coerce")
    )
    result["detGain"] = (
        pd.to_numeric(frame[f"{challenge}.det"], errors="coerce")
        - pd.to_numeric(frame[f"{reference}.det"], errors="coerce")
    )
    result["entrGain"] = (
        pd.to_numeric(frame[f"{challenge}.entr"], errors="coerce")
        - pd.to_numeric(frame[f"{reference}.entr"], errors="coerce")
    )
    result["dimensionShift"] = np.abs(
        pd.to_numeric(frame[f"{challenge}.dimension"], errors="coerce")
        - pd.to_numeric(frame[f"{reference}.dimension"], errors="coerce")
    )
    result["shapeShift"] = np.abs(safe_log_ratio(
        frame[f"{challenge}.shapeRatio"], frame[f"{reference}.shapeRatio"],
    ))
    return result.replace([np.inf, -np.inf], np.nan)


def conventional_features(frame):
    result = pd.DataFrame(index=frame.index)
    night_mean = pd.to_numeric(frame["night.conventional.mean"], errors="coerce")
    day_mean = pd.to_numeric(frame["daytime.conventional.mean"], errors="coerce")
    night_cv = pd.to_numeric(frame["night.conventional.cv"], errors="coerce")
    day_cv = pd.to_numeric(frame["daytime.conventional.cv"], errors="coerce")
    night_out = pd.to_numeric(frame["night.conventional.outOfRange"], errors="coerce")
    day_out = pd.to_numeric(frame["daytime.conventional.outOfRange"], errors="coerce")
    result["nightMean"] = night_mean
    result["meanRise"] = day_mean - night_mean
    result["logCVRatio"] = safe_log_ratio(day_cv, night_cv)
    result["outRise"] = day_out - night_out
    return result


def robust_scale(values):
    values = finite(values)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        raise ValueError("At least three finite values are required")
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if scale < 1e-8:
        scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
    if scale < 1e-8:
        scale = float(np.std(values))
    if scale < 1e-8:
        scale = 1.0
    return center, scale


def fit_reserve_reference(contrasts):
    return {feature: robust_scale(contrasts[feature]) for feature in RESERVE_FEATURES}


def row_median_at_least(values, minimum):
    array = np.asarray(values, float)
    count = np.sum(np.isfinite(array), axis=1)
    result = np.nanmedian(array, axis=1)
    result[count < minimum] = np.nan
    return result


def transform_reserve(contrasts, reference):
    z = pd.DataFrame(index=contrasts.index)
    for feature in RESERVE_FEATURES:
        center, scale = reference[feature]
        z[feature] = (contrasts[feature] - center) / scale
    physical = row_median_at_least(z[PHYSICAL], 2)
    dynamic = row_median_at_least(z[DYNAMIC], 2)
    geometry = row_median_at_least(np.abs(z[GEOMETRY]), 2)
    consensus = np.sqrt(np.maximum(physical, 0) * np.maximum(dynamic, 0))
    return pd.DataFrame({
        "physicalArm": physical,
        "dynamicArm": dynamic,
        "reserveConsensus": consensus,
        "geometryShift": geometry,
    }, index=contrasts.index)


def model_frame(frame, reference=None):
    conventional = conventional_features(frame)
    if reference is None:
        return conventional
    reserve = transform_reserve(state_contrasts(frame), reference)
    return pd.concat([conventional, reserve], axis=1)


def fit_nonnegative_ridge(X, y, l2=1.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-10] = 1.0
    Z = (X - mean) / scale
    y_mean = float(y.mean())
    centered_y = y - y_mean
    beta = np.zeros(Z.shape[1], float)
    residual = centered_y.copy()
    for _ in range(2000):
        old = beta.copy()
        for column in range(Z.shape[1]):
            residual += Z[:, column] * beta[column]
            denominator = float(Z[:, column] @ Z[:, column] + l2)
            beta[column] = max(0.0, float(Z[:, column] @ residual) / denominator)
            residual -= Z[:, column] * beta[column]
        if np.max(np.abs(beta - old)) < 1e-10:
            break
    return {"mean": mean, "scale": scale, "y_mean": y_mean, "beta": beta}


def predict_ridge(model, X):
    Z = (np.asarray(X, float) - model["mean"]) / model["scale"]
    return model["y_mean"] + Z @ model["beta"]


def regression_folds(y, n_splits, seed):
    y = finite(y)
    order = np.argsort(y, kind="mergesort")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), int)
    for start in range(0, len(order), n_splits):
        block = order[start:start + n_splits]
        labels = np.arange(len(block))
        rng.shuffle(labels)
        fold_of[block] = labels
    indices = np.arange(len(y))
    return [(indices[fold_of != fold], indices[fold_of == fold]) for fold in range(n_splits)]


def fit_regression_pipeline(frame, y, model_name):
    reference = None
    if model_name not in {"night_mean", "conventional"}:
        reference = fit_reserve_reference(state_contrasts(frame))
    transformed = model_frame(frame, reference)
    features = MODEL_FEATURES[model_name]
    ridge = fit_nonnegative_ridge(transformed[features], y)
    return {"reference": reference, "ridge": ridge, "model_name": model_name}


def predict_regression_pipeline(model, frame):
    transformed = model_frame(frame, model["reference"])
    features = MODEL_FEATURES[model["model_name"]]
    return predict_ridge(model["ridge"], transformed[features])


def cross_val_regression(frame, y, model_name):
    total = np.zeros(len(y), float)
    repeat_metrics = []
    candidate_coefficients = []
    for repeat in range(N_REPEATS):
        prediction = np.zeros(len(y), float)
        for train, test in regression_folds(y, N_SPLITS, SEED + repeat):
            model = fit_regression_pipeline(frame.iloc[train], y[train], model_name)
            prediction[test] = predict_regression_pipeline(model, frame.iloc[test])
            if model_name not in {"night_mean", "conventional"}:
                candidate_coefficients.append(float(model["ridge"]["beta"][-1]))
        total += prediction
        repeat_metrics.append({
            "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "spearman": spearman(prediction, y),
        })
    averaged = total / N_REPEATS
    return averaged, repeat_metrics, candidate_coefficients


def regression_summary(y, prediction, repeat_metrics, coefficients):
    residual = prediction - y
    result = {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)),
        "spearman": spearman(prediction, y),
        "repeatRmseMean": float(np.mean([row["rmse"] for row in repeat_metrics])),
        "repeatRmseSd": float(np.std([row["rmse"] for row in repeat_metrics])),
        "repeatSpearmanMean": float(np.mean([row["spearman"] for row in repeat_metrics])),
        "repeatSpearmanSd": float(np.std([row["spearman"] for row in repeat_metrics])),
    }
    if coefficients:
        result["candidateCoefficientPositiveFraction"] = float(np.mean(np.asarray(coefficients) > 1e-10))
        result["candidateCoefficientMedian"] = float(np.median(coefficients))
    return result


def fit_classification_pipeline(frame, y, model_name):
    reference = None
    if model_name not in {"night_mean", "conventional"}:
        reference = fit_reserve_reference(state_contrasts(frame))
    transformed = model_frame(frame, reference)
    features = MODEL_FEATURES[model_name]
    logistic = fit_logistic(transformed[features].to_numpy(float), y, monotonic=True)
    return {"reference": reference, "logistic": logistic, "model_name": model_name}


def predict_classification_pipeline(model, frame):
    transformed = model_frame(frame, model["reference"])
    features = MODEL_FEATURES[model["model_name"]]
    return predict_logistic(model["logistic"], transformed[features].to_numpy(float))


def nested_train_threshold(frame, y, model_name, seed):
    probability = np.zeros(len(y), float)
    folds = stratified_folds(y, min(3, int(min(np.sum(y == 0), np.sum(y == 1)))), seed)
    for train, test in folds:
        model = fit_classification_pipeline(frame.iloc[train], y[train], model_name)
        probability[test] = predict_classification_pipeline(model, frame.iloc[test])
    return choose_threshold(y, probability, min_sensitivity=0.75)


def cross_val_classification(frame, y, model_name):
    total = np.zeros(len(y), float)
    repeated_truth, repeated_flags = [], []
    repeat_auc, candidate_coefficients, thresholds = [], [], []
    for repeat in range(N_REPEATS):
        prediction = np.zeros(len(y), float)
        for fold_number, (train, test) in enumerate(stratified_folds(y, N_SPLITS, SEED + 100 + repeat)):
            threshold = nested_train_threshold(
                frame.iloc[train], y[train], model_name, SEED + 10000 + repeat * 10 + fold_number,
            )
            model = fit_classification_pipeline(frame.iloc[train], y[train], model_name)
            test_probability = predict_classification_pipeline(model, frame.iloc[test])
            prediction[test] = test_probability
            thresholds.append(threshold)
            repeated_truth.extend(y[test].tolist())
            repeated_flags.extend((test_probability >= threshold).tolist())
            if model_name not in {"night_mean", "conventional"}:
                candidate_coefficients.append(float(model["logistic"]["beta"][-1]))
        total += prediction
        repeat_auc.append(roc_auc(y, prediction))
    averaged = total / N_REPEATS
    repeated_truth = np.asarray(repeated_truth, int)
    repeated_flags = np.asarray(repeated_flags, bool)
    nested = screening_metrics(repeated_truth, repeated_flags.astype(float), 0.5)
    return averaged, repeat_auc, candidate_coefficients, nested, thresholds


def classification_summary(y, probability, repeat_auc, coefficients, nested, thresholds):
    result = {
        "rocAuc": roc_auc(y, probability),
        "prAuc": average_precision(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "repeatAucMean": float(np.mean(repeat_auc)),
        "repeatAucSd": float(np.std(repeat_auc)),
        "nestedThreshold": {
            **nested,
            "thresholdMedian": float(np.median(thresholds)),
            "thresholdIqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        },
    }
    if coefficients:
        result["candidateCoefficientPositiveFraction"] = float(np.mean(np.asarray(coefficients) > 1e-10))
        result["candidateCoefficientMedian"] = float(np.median(coefficients))
    return result


def bootstrap_spearman(x, y, seed, replicates=N_BOOTSTRAP):
    x, y = finite(x), finite(y)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(x), len(x))
        values[index] = spearman(x[sample], y[sample])
    values = values[np.isfinite(values)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_regression_bootstrap(y, baseline, candidate, seed):
    rng = np.random.default_rng(seed)
    delta_rmse, delta_rho = np.empty(N_BOOTSTRAP), np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(y), len(y))
        base_rmse = np.sqrt(np.mean((baseline[sample] - y[sample]) ** 2))
        candidate_rmse = np.sqrt(np.mean((candidate[sample] - y[sample]) ** 2))
        delta_rmse[index] = base_rmse - candidate_rmse
        delta_rho[index] = spearman(candidate[sample], y[sample]) - spearman(baseline[sample], y[sample])
    delta_rho = delta_rho[np.isfinite(delta_rho)]
    return {
        "rmseImprovement": {
            "estimate": float(np.sqrt(np.mean((baseline - y) ** 2)) - np.sqrt(np.mean((candidate - y) ** 2))),
            "ci95": [float(np.quantile(delta_rmse, 0.025)), float(np.quantile(delta_rmse, 0.975))],
            "probabilityGreaterThanZero": float(np.mean(delta_rmse > 0)),
        },
        "spearmanImprovement": {
            "estimate": float(spearman(candidate, y) - spearman(baseline, y)),
            "ci95": [float(np.quantile(delta_rho, 0.025)), float(np.quantile(delta_rho, 0.975))],
            "probabilityGreaterThanZero": float(np.mean(delta_rho > 0)),
        },
    }


def paired_auc_bootstrap(y, baseline, candidate, seed):
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    delta = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        delta[index] = roc_auc(y[sample], candidate[sample]) - roc_auc(y[sample], baseline[sample])
    return {
        "estimate": float(roc_auc(y, candidate) - roc_auc(y, baseline)),
        "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "probabilityGreaterThanZero": float(np.mean(delta > 0)),
    }


def fixed_prediction_permutation(y, baseline, candidate, seed):
    observed_rmse = float(np.sqrt(np.mean((baseline - y) ** 2)) - np.sqrt(np.mean((candidate - y) ** 2)))
    observed_rho = float(spearman(candidate, y) - spearman(baseline, y))
    rng = np.random.default_rng(seed)
    null_rmse, null_rho = np.empty(N_PERMUTATIONS), np.empty(N_PERMUTATIONS)
    for index in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        null_rmse[index] = (
            np.sqrt(np.mean((baseline - permuted) ** 2))
            - np.sqrt(np.mean((candidate - permuted) ** 2))
        )
        null_rho[index] = spearman(candidate, permuted) - spearman(baseline, permuted)
    return {
        "replicates": N_PERMUTATIONS,
        "rmseImprovementP": float((1 + np.sum(null_rmse >= observed_rmse)) / (N_PERMUTATIONS + 1)),
        "spearmanImprovementP": float((1 + np.sum(null_rho >= observed_rho)) / (N_PERMUTATIONS + 1)),
    }


def reliability_analysis(cohort_frame):
    splits = {
        name: cohort_frame[cohort_frame["split"] == name].sort_values("id").reset_index(drop=True)
        for name in ("full", "odd", "even")
    }
    if not (splits["full"]["id"].tolist() == splits["odd"]["id"].tolist() == splits["even"]["id"].tolist()):
        raise ValueError("Split IDs do not align")
    reference = fit_reserve_reference(state_contrasts(splits["full"]))
    scores = {
        name: transform_reserve(state_contrasts(frame), reference)
        for name, frame in splits.items()
    }
    raw = {name: state_contrasts(frame) for name, frame in splits.items()}
    result = {"n": len(splits["full"]), "features": {}}
    for feature in RESERVE_FEATURES:
        odd, even = raw["odd"][feature].to_numpy(float), raw["even"][feature].to_numpy(float)
        result["features"][feature] = {
            "calculability": float(np.mean(np.isfinite(raw["full"][feature]))),
            "oddEvenSpearman": spearman(odd, even),
            "ci95": bootstrap_spearman(odd, even, SEED + len(result["features"])),
        }
    result["scores"] = {}
    for feature in ("physicalArm", "dynamicArm", "reserveConsensus", "geometryShift"):
        odd = scores["odd"][feature].to_numpy(float)
        even = scores["even"][feature].to_numpy(float)
        result["scores"][feature] = {
            "calculability": float(np.mean(np.isfinite(scores["full"][feature]))),
            "activeFraction": float(np.mean(scores["full"][feature].to_numpy(float) > 0)),
            "oddEvenSpearman": spearman(odd, even),
            "ci95": bootstrap_spearman(odd, even, SEED + 100 + len(result["scores"])),
        }
    return result, splits, reference, scores


def partial_spearman(feature, outcome, covariates):
    feature, outcome, covariates = finite(feature), finite(outcome), np.asarray(covariates, float)
    keep = np.isfinite(feature) & np.isfinite(outcome) & np.all(np.isfinite(covariates), axis=1)
    feature, outcome, covariates = feature[keep], outcome[keep], covariates[keep]
    ranked_x = rankdata(feature)
    ranked_y = rankdata(outcome)
    ranked_covariates = np.column_stack([rankdata(covariates[:, index]) for index in range(covariates.shape[1])])
    design = np.column_stack([np.ones(len(feature)), ranked_covariates])
    residual_x = ranked_x - design @ (np.linalg.pinv(design) @ ranked_x)
    residual_y = ranked_y - design @ (np.linalg.pinv(design) @ ranked_y)
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def ogtt_associations(full, reference, scores):
    outcome = pd.to_numeric(full["clinical.ogtt.2hr"], errors="coerce").to_numpy(float)
    contrasts = state_contrasts(full)
    pseudo = state_contrasts(full, pseudo=True)
    pseudo_reference = fit_reserve_reference(pseudo)
    pseudo_scores = transform_reserve(pseudo, pseudo_reference)
    conventional = conventional_features(full).to_numpy(float)
    candidates = {
        **{feature: contrasts[feature].to_numpy(float) for feature in RESERVE_FEATURES},
        "physicalArm": scores["physicalArm"].to_numpy(float),
        "dynamicArm": scores["dynamicArm"].to_numpy(float),
        "reserveConsensus": scores["reserveConsensus"].to_numpy(float),
        "geometryShift": scores["geometryShift"].to_numpy(float),
    }
    associations = {}
    for index, (name, values) in enumerate(candidates.items()):
        associations[name] = {
            "spearman": spearman(values, outcome),
            "ci95": bootstrap_spearman(values, outcome, SEED + 200 + index),
            "partialSpearmanControllingConventional": partial_spearman(values, outcome, conventional),
        }

    observed = {name: abs(row["spearman"]) for name, row in associations.items()}
    rng = np.random.default_rng(SEED + 300)
    exceed = {name: 0 for name in candidates}
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(outcome)
        maximum = max(abs(spearman(values, permuted)) for values in candidates.values())
        for name in candidates:
            exceed[name] += maximum >= observed[name]
    for name in associations:
        associations[name]["maxTAdjustedP"] = float((1 + exceed[name]) / (N_PERMUTATIONS + 1))

    primary = candidates["reserveConsensus"]
    rng = np.random.default_rng(SEED + 301)
    observed_primary = abs(spearman(primary, outcome))
    primary_p = (1 + sum(abs(spearman(primary, rng.permutation(outcome))) >= observed_primary for _ in range(N_PERMUTATIONS))) / (N_PERMUTATIONS + 1)
    return {
        "n": int(np.sum(np.isfinite(outcome))),
        "associations": associations,
        "primaryConsensusPermutationP": float(primary_p),
        "negativeControl": {
            "definition": "00:00-03:00 versus 03:00-06:00 pseudo-state contrast",
            "pseudoConsensusSpearman": spearman(pseudo_scores["reserveConsensus"], outcome),
            "pseudoPhysicalSpearman": spearman(pseudo_scores["physicalArm"], outcome),
            "pseudoDynamicSpearman": spearman(pseudo_scores["dynamicArm"], outcome),
        },
    }


def exact_sign_test(positive, total):
    if total <= 0:
        return None
    tail = sum(math.comb(total, value) for value in range(positive + 1)) / (2 ** total)
    reverse = sum(math.comb(total, value) for value in range(positive, total + 1)) / (2 ** total)
    return float(min(1.0, 2 * min(tail, reverse)))


def dubosson_analysis(event_rows):
    if event_rows.empty:
        return {"events": 0, "subjects": 0}
    contrasts = state_contrasts(event_rows.rename(columns={
        **{f"pre.{metric}": f"night.{metric}" for metric in ("volume", "recovery", "lyapunov", "det", "entr", "dimension", "shapeRatio")},
        **{f"post.{metric}": f"daytime.{metric}" for metric in ("volume", "recovery", "lyapunov", "det", "entr", "dimension", "shapeRatio")},
    }))
    contrasts["coreShift"] = pd.to_numeric(event_rows["coreShift"], errors="coerce")
    subject = pd.concat([event_rows[["id"]].reset_index(drop=True), contrasts.reset_index(drop=True)], axis=1).groupby("id").median(numeric_only=True)
    feature_results = {}
    for feature in ["volumeExpansion", "recoveryDebt", "lyapunovLoss", "detGain", "entrGain"]:
        values = subject[feature].dropna().to_numpy(float)
        nonzero = values[np.abs(values) > 1e-12]
        positives = int(np.sum(nonzero > 0))
        feature_results[feature] = {
            "subjectMedian": float(np.median(values)) if len(values) else None,
            "subjectsPositive": positives,
            "subjectsFinite": int(len(values)),
            "subjectsNonzero": int(len(nonzero)),
            "exactTwoSidedSignP": exact_sign_test(positives, len(nonzero)),
        }
    envelope_area = pd.to_numeric(event_rows["envelope.standardizedEnvelopeArea"], errors="coerce")
    return_minutes = pd.to_numeric(event_rows["envelope.returnMinutes"], errors="coerce")
    return {
        "events": int(len(event_rows)),
        "subjects": int(event_rows["id"].nunique()),
        "eventsPerSubject": event_rows.groupby("id").size().astype(int).to_dict(),
        "eventTypeCounts": {
            "calorieInvolved": int(event_rows["eventTypes"].apply(lambda values: "calories" in values).sum()),
            "fastInsulinInvolved": int(event_rows["eventTypes"].apply(lambda values: "fast_insulin" in values).sum()),
        },
        "returnCensoredFraction": float(pd.to_numeric(event_rows["envelope.returnCensored"], errors="coerce").mean()),
        "envelopeAreaMedian": float(envelope_area.median()),
        "returnMinutesMedianAmongObserved": float(return_minutes.median()),
        "subjectDirection": feature_results,
        "limitations": "Short event windows yielded phase metrics for only a subset; n is mechanistic, not screening evidence.",
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    state = pd.json_normalize(payload["stateMetrics"])
    state_six_day = pd.json_normalize(payload.get("stateMetricsSixDay", []))
    events = pd.json_normalize(payload["dubossonEventMetrics"])

    reliability = {}
    split_cache, reference_cache, score_cache = {}, {}, {}
    for cohort in ("hall", "weinstock"):
        result, splits, reference, scores = reliability_analysis(state[state["cohort"] == cohort])
        reliability[cohort] = result
        split_cache[cohort], reference_cache[cohort], score_cache[cohort] = splits, reference, scores

    six_day_reliability = {}
    six_day_splits, six_day_references, six_day_scores = {}, {}, {}
    if not state_six_day.empty:
        for cohort in ("hall", "weinstock"):
            result, splits, reference, scores = reliability_analysis(
                state_six_day[state_six_day["cohort"] == cohort],
            )
            six_day_reliability[cohort] = result
            six_day_splits[cohort], six_day_references[cohort], six_day_scores[cohort] = splits, reference, scores

    hall_all = split_cache["hall"]["full"]
    ogtt_all = pd.to_numeric(hall_all["clinical.ogtt.2hr"], errors="coerce").to_numpy(float)
    # Hall encodes missing OGTT values as -1; never treat that sentinel as physiology.
    valid_ogtt = np.isfinite(ogtt_all) & (ogtt_all > 0)
    hall = hall_all.loc[valid_ogtt].reset_index(drop=True)
    ogtt = ogtt_all[valid_ogtt]
    regression = {"models": {}}
    regression_predictions = {}
    for model_name in MODEL_FEATURES:
        prediction, repeat_metrics, coefficients = cross_val_regression(hall, ogtt, model_name)
        regression_predictions[model_name] = prediction
        regression["models"][model_name] = regression_summary(ogtt, prediction, repeat_metrics, coefficients)
    regression["comparisonsVersusConventional"] = {}
    for comparison_index, model_name in enumerate((
        "conventional_plus_physical",
        "conventional_plus_dynamic",
        "conventional_plus_consensus",
        "conventional_plus_geometry",
    )):
        regression["comparisonsVersusConventional"][model_name] = {
            "pairedBootstrap": paired_regression_bootstrap(
                ogtt,
                regression_predictions["conventional"],
                regression_predictions[model_name],
                SEED + 400 + comparison_index,
            ),
            "fixedPredictionPermutation": fixed_prediction_permutation(
                ogtt,
                regression_predictions["conventional"],
                regression_predictions[model_name],
                SEED + 410 + comparison_index,
            ),
        }

    diagnosis = pd.to_numeric(hall_all["clinical.diagnosis"], errors="coerce").to_numpy(float)
    abnormal = (diagnosis >= 1).astype(int)
    classification = {
        "target": "Hall diagnosis code >=1 (prediabetes or diabetes; secondary internal screening endpoint)",
        "classCounts": {"reference": int(np.sum(abnormal == 0)), "abnormal": int(np.sum(abnormal == 1))},
        "models": {},
    }
    classification_predictions = {}
    for model_name in ("night_mean", "conventional", "conventional_plus_consensus"):
        probability, repeat_auc, coefficients, nested, thresholds = cross_val_classification(
            hall_all, abnormal, model_name,
        )
        classification_predictions[model_name] = probability
        classification["models"][model_name] = classification_summary(
            abnormal, probability, repeat_auc, coefficients, nested, thresholds,
        )
    classification["consensusVersusConventionalDeltaAuc"] = paired_auc_bootstrap(
        abnormal,
        classification_predictions["conventional"],
        classification_predictions["conventional_plus_consensus"],
        SEED + 500,
    )

    associations = ogtt_associations(
        hall,
        reference_cache["hall"],
        score_cache["hall"]["full"].loc[valid_ogtt].reset_index(drop=True),
    )

    six_day_outcome = None
    if six_day_splits:
        hall_six_all = six_day_splits["hall"]["full"]
        six_ogtt_all = pd.to_numeric(hall_six_all["clinical.ogtt.2hr"], errors="coerce").to_numpy(float)
        six_valid = np.isfinite(six_ogtt_all) & (six_ogtt_all > 0)
        hall_six = hall_six_all.loc[six_valid].reset_index(drop=True)
        six_ogtt = six_ogtt_all[six_valid]
        six_associations = ogtt_associations(
            hall_six,
            six_day_references["hall"],
            six_day_scores["hall"]["full"].loc[six_valid].reset_index(drop=True),
        )
        six_predictions = {}
        six_models = {}
        for model_name in ("conventional", "conventional_plus_consensus"):
            prediction, repeat_metrics, coefficients = cross_val_regression(
                hall_six, six_ogtt, model_name,
            )
            six_predictions[model_name] = prediction
            six_models[model_name] = regression_summary(
                six_ogtt, prediction, repeat_metrics, coefficients,
            )
        six_day_outcome = {
            "hallOgttAssociations": six_associations,
            "hallOgttRegression": {
                "models": six_models,
                "consensusVersusConventional": {
                    "pairedBootstrap": paired_regression_bootstrap(
                        six_ogtt,
                        six_predictions["conventional"],
                        six_predictions["conventional_plus_consensus"],
                        SEED + 550,
                    ),
                    "fixedPredictionPermutation": fixed_prediction_permutation(
                        six_ogtt,
                        six_predictions["conventional"],
                        six_predictions["conventional_plus_consensus"],
                        SEED + 551,
                    ),
                },
            },
        }
    dubosson = dubosson_analysis(events)

    reliability_pass = (
        reliability["hall"]["scores"]["reserveConsensus"]["calculability"] >= 0.95
        and reliability["hall"]["scores"]["reserveConsensus"]["oddEvenSpearman"] >= 0.70
        and reliability["weinstock"]["scores"]["reserveConsensus"]["calculability"] >= 0.95
        and reliability["weinstock"]["scores"]["reserveConsensus"]["oddEvenSpearman"] >= 0.70
    )
    consensus_comparison = regression["comparisonsVersusConventional"]["conventional_plus_consensus"]
    increment = consensus_comparison["pairedBootstrap"]["rmseImprovement"]
    increment_pass = (
        increment["ci95"][0] > 0
        and consensus_comparison["fixedPredictionPermutation"]["rmseImprovementP"] < 0.05
    )
    coefficient_pass = (
        regression["models"]["conventional_plus_consensus"].get("candidateCoefficientPositiveFraction", 0) >= 0.80
    )
    six_day_reliability_pass = bool(six_day_reliability) and all(
        six_day_reliability[cohort]["scores"]["reserveConsensus"]["calculability"] >= 0.95
        and six_day_reliability[cohort]["scores"]["reserveConsensus"]["oddEvenSpearman"] >= 0.70
        for cohort in ("hall", "weinstock")
    )
    result = {
        "metadata": {
            "analysis": "within-person structural reserve",
            "seed": SEED,
            "repeats": N_REPEATS,
            "folds": N_SPLITS,
            "bootstrapReplicates": N_BOOTSTRAP,
            "permutationReplicates": N_PERMUTATIONS,
            "primaryFormula": "sqrt(max(median(z(volume expansion),z(recovery debt),z(core shift)),0) * max(median(z(Lyapunov loss),z(DET gain),z(ENTR gain)),0))",
            "trainingFoldReferences": True,
        },
        "reliability": reliability,
        "sixDaySensitivityReliability": six_day_reliability,
        "sixDaySensitivityOutcome": six_day_outcome,
        "hallOgttAssociations": associations,
        "hallOgttRegression": regression,
        "hallDiagnosisClassification": classification,
        "dubossonMechanisticPilot": dubosson,
        "decision": {
            "reliabilityGate": {"pass": bool(reliability_pass), "threshold": "calculability >=0.95 and odd-even Spearman >=0.70 in Hall and Weinstock"},
            "incrementalOgttGate": {"pass": bool(increment_pass), "threshold": "paired bootstrap RMSE improvement CI entirely >0 and permutation p<0.05"},
            "coefficientStabilityGate": {"pass": bool(coefficient_pass), "threshold": "consensus coefficient positive in >=80% outer folds"},
            "sixDaySensitivityReliabilityGate": {"pass": six_day_reliability_pass, "threshold": "unchanged consensus; 3-day odd/even halves; Spearman >=0.70 in both cohorts"},
            "overallPass": bool(reliability_pass and increment_pass and coefficient_pass),
            "htmlAction": "update screening probability only if overallPass; otherwise preserve index.html",
        },
        "inputHashes": {
            "raw_data.zip": sha256(ROOT / "raw_data.zip"),
            "index.html": sha256(ROOT / "index.html"),
            "structure_reserve_windows.json": sha256(ROOT / "output" / "structure_reserve_windows.json"),
            "structure_reserve_metrics.json": sha256(INPUT),
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "hall_reliability": reliability["hall"]["scores"]["reserveConsensus"],
        "weinstock_reliability": reliability["weinstock"]["scores"]["reserveConsensus"],
        "six_day_reliability": {
            cohort: six_day_reliability[cohort]["scores"]["reserveConsensus"]
            for cohort in six_day_reliability
        },
        "ogtt_consensus": associations["associations"]["reserveConsensus"],
        "regression_conventional": regression["models"]["conventional"],
        "regression_plus_consensus": regression["models"]["conventional_plus_consensus"],
        "diagnosis_delta_auc": classification["consensusVersusConventionalDeltaAuc"],
        "decision": result["decision"],
        "dubosson": dubosson,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
