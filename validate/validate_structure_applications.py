#!/usr/bin/env python3
"""Validate treatment-agnostic structure consensus applications.

Frozen primary use case: predict high future glycemic fragility in elapsed days
3-5 from the first 3 days. Future fragility is the equal-weight percentile-rank
composite of CV and fraction outside 3.9-10 mmol/L; its upper quartile is the
internal binary target. This is a temporal research phenotype, not a clinical
diagnosis or guideline threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_context_consensus import (
    OUTPUT_DIR, ROOT, SEED, average_precision, choose_threshold, fit_logistic,
    newton_logistic, predict_logistic, rankdata, robust_scale, roc_auc,
    screening_metrics, spearman, stratified_folds,
)


N_REPEATS = 20
N_SPLITS = 5
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 1000
N_ASSOCIATION_PERMUTATIONS = 2000
STRUCTURE = ["logVolume", "avgRecovery", "coreDisplacement", "dimension", "logShapeRatio"]
MODEL_FEATURES = {
    "conventional": ["earlyCV", "earlyOut"],
    "structure_only": ["structureScore"],
    "conventional_plus_structure": ["earlyCV", "earlyOut", "structureScore"],
    "conventional_plus_expansion": ["earlyCV", "earlyOut", "expansionLoad"],
    "conventional_plus_contraction": ["earlyCV", "earlyOut", "contractionLoad"],
    "conventional_plus_geometry": ["earlyCV", "earlyOut", "geometryLoad"],
}


def load_window(cohort, k):
    path = OUTPUT_DIR / f"structure_metrics_{cohort}_k{k}.json"
    frame = pd.json_normalize(json.loads(path.read_text(encoding="utf-8")))
    frame["id"] = frame["id"].astype(str)
    frame["logVolume"] = np.log(np.maximum(frame["volume"].to_numpy(float), 1e-10))
    frame["logShapeRatio"] = np.log(np.maximum(frame["shapeRatio"].to_numpy(float), 1e-10))
    frame["earlyCV"] = frame["earlyConventional.cv"].astype(float)
    frame["earlyOut"] = frame["earlyConventional.outOfRangeFraction"].astype(float)
    frame["futureCV"] = frame["future.cv"].astype(float)
    frame["futureOut"] = frame["future.outOfRangeFraction"].astype(float)
    return frame


def add_future_target(frame):
    result = frame.copy()
    n = len(result)
    cv_percentile = (rankdata(result["futureCV"]) - 0.5) / n
    out_percentile = (rankdata(result["futureOut"]) - 0.5) / n
    result["futureFragility"] = 0.5 * (cv_percentile + out_percentile)
    cutoff = float(np.quantile(result["futureFragility"], 0.75))
    result["futureHighFragility"] = (result["futureFragility"] >= cutoff).astype(int)
    return result, cutoff


def fit_structure_reference(frame):
    reference = {}
    for feature in STRUCTURE:
        values = frame[feature].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            raise ValueError(f"Insufficient finite training references for {feature}")
        reference[feature] = robust_scale(values)
    return reference


def transform_structure(frame, reference):
    result = frame.copy()
    z = {}
    evidence = {}
    for feature in STRUCTURE:
        center, scale = reference[feature]
        z[feature] = (frame[feature].to_numpy(float) - center) / scale
        evidence[feature] = np.clip(np.abs(z[feature]) - 1.0, 0, 3)
    result["structureScore"] = np.median(
        np.column_stack([evidence[feature] for feature in STRUCTURE]), axis=1,
    )
    result["expansionLoad"] = np.median(np.column_stack([
        np.clip(z["logVolume"] - 1, 0, 3),
        np.clip(-z["avgRecovery"] - 1, 0, 3),
        np.clip(z["coreDisplacement"] - 1, 0, 3),
    ]), axis=1)
    result["contractionLoad"] = np.median(np.column_stack([
        np.clip(-z["logVolume"] - 1, 0, 3),
        np.clip(z["avgRecovery"] - 1, 0, 3),
        np.clip(-z["coreDisplacement"] - 1, 0, 3),
    ]), axis=1)
    result["geometryLoad"] = np.median(np.column_stack([
        evidence["dimension"], evidence["logShapeRatio"],
    ]), axis=1)
    return result


def fit_pipeline(frame, y, model_name):
    reference = None
    transformed = frame
    if model_name != "conventional":
        reference = fit_structure_reference(frame)
        transformed = transform_structure(frame, reference)
    features = MODEL_FEATURES[model_name]
    logistic = fit_logistic(transformed[features].to_numpy(float), y, monotonic=True)
    return {"reference": reference, "logistic": logistic, "model_name": model_name}


def predict_pipeline(model, frame):
    transformed = frame if model["reference"] is None else transform_structure(frame, model["reference"])
    return predict_logistic(model["logistic"], transformed[MODEL_FEATURES[model["model_name"]]].to_numpy(float))


def cross_val_predict(frame, y, model_name, repeats, seed):
    total = np.zeros(len(y))
    repeat_auc = []
    for repeat in range(repeats):
        fold_probability = np.zeros(len(y))
        for train, test in stratified_folds(y, N_SPLITS, seed + repeat):
            model = fit_pipeline(frame.iloc[train], y[train], model_name)
            fold_probability[test] = predict_pipeline(model, frame.iloc[test])
        total += fold_probability
        repeat_auc.append(roc_auc(y, fold_probability))
    return total / repeats, np.asarray(repeat_auc)


def calibration(y, probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))[:, None]
    beta = newton_logistic(logit, y, l2=0.0)
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def summarize_prediction(y, probability, future_fragility, repeat_auc):
    return {
        "roc_auc": roc_auc(y, probability),
        "pr_auc": average_precision(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "calibration": calibration(y, probability),
        "spearman_with_continuous_future_fragility": spearman(probability, future_fragility),
        "repeat_auc_mean": float(np.mean(repeat_auc)),
        "repeat_auc_sd": float(np.std(repeat_auc)),
    }


def paired_bootstrap(y, candidate, baseline, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    delta = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        delta[index] = roc_auc(y[sample], candidate[sample]) - roc_auc(y[sample], baseline[sample])
    return {
        "replicates": N_BOOTSTRAP,
        "mean_delta_auc": float(np.mean(delta)),
        "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "probability_delta_gt_zero": float(np.mean(delta > 0)),
    }


def fixed_candidate_permutation(frame, y, seed):
    baseline, _ = cross_val_predict(frame, y, "conventional", 1, seed)
    candidate, _ = cross_val_predict(frame, y, "conventional_plus_structure", 1, seed)
    observed = roc_auc(y, candidate) - roc_auc(y, baseline)
    rng = np.random.default_rng(seed + 1)
    null = np.empty(N_PERMUTATIONS)
    for index in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        base, _ = cross_val_predict(frame, permuted, "conventional", 1, seed + 10 + index)
        new, _ = cross_val_predict(frame, permuted, "conventional_plus_structure", 1, seed + 10 + index)
        null[index] = roc_auc(permuted, new) - roc_auc(permuted, base)
    return {
        "replicates": N_PERMUTATIONS,
        "observed_one_repeat_delta_auc": float(observed),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "one_sided_p": float((1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1)),
    }


def nested_threshold(frame, y, model_name, seed):
    rows, thresholds = [], []
    for repeat in range(10):
        for fold_no, (train, test) in enumerate(stratified_folds(y, N_SPLITS, seed + repeat)):
            inner, _ = cross_val_predict(frame.iloc[train], y[train], model_name, 3, seed + 500 + repeat * 17 + fold_no)
            threshold = choose_threshold(y[train], inner, 0.75)
            model = fit_pipeline(frame.iloc[train], y[train], model_name)
            probability = predict_pipeline(model, frame.iloc[test])
            row = screening_metrics(y[test], probability, threshold)
            row["n"] = len(test)
            rows.append(row); thresholds.append(threshold)
    weights = np.asarray([row["n"] for row in rows])
    return {
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "weighted_mean": {
            key: float(np.average([row[key] for row in rows], weights=weights))
            for key in ("sensitivity", "specificity", "balanced_accuracy", "flag_rate")
        },
    }


def coefficient_stability(frame, y, seed):
    beta = []
    for repeat in range(N_REPEATS):
        for train, _ in stratified_folds(y, N_SPLITS, seed + repeat):
            model = fit_pipeline(frame.iloc[train], y[train], "conventional_plus_structure")
            beta.append(float(model["logistic"]["beta"][3]))
    beta = np.asarray(beta)
    return {
        "outer_fits": len(beta),
        "structure_beta_positive_fraction": float(np.mean(beta > 1e-8)),
        "structure_beta_median": float(np.median(beta)),
        "structure_beta_iqr": [float(np.quantile(beta, 0.25)), float(np.quantile(beta, 0.75))],
    }


def stability(cohort):
    windows = {k: load_window(cohort, k) for k in (1, 2, 3)}
    common = set(windows[1]["id"]) & set(windows[2]["id"]) & set(windows[3]["id"])
    for k in windows:
        windows[k] = windows[k][windows[k]["id"].isin(common)].set_index("id").sort_index()
    valid = np.ones(len(windows[3]), dtype=bool)
    for k in windows:
        valid &= windows[k][STRUCTURE].notna().all(axis=1).to_numpy()
    ids = windows[3].index[valid]
    fixed_scores = {k: [] for k in windows}
    duration_scores = {k: [] for k in windows}
    for subject_id in ids:
        fixed_reference = fit_structure_reference(windows[3].drop(index=subject_id))
        for k in windows:
            row = windows[k].loc[[subject_id]]
            duration_reference = fit_structure_reference(windows[k].drop(index=subject_id))
            fixed_scores[k].append(float(transform_structure(row, fixed_reference)["structureScore"].iloc[0]))
            duration_scores[k].append(float(transform_structure(row, duration_reference)["structureScore"].iloc[0]))
    fixed_scores = {k: np.asarray(value) for k, value in fixed_scores.items()}
    duration_scores = {k: np.asarray(value) for k, value in duration_scores.items()}
    return {
        "common_subjects": len(common),
        "calculable_all_windows": len(ids),
        "calculability_fraction": float(len(ids) / len(common)),
        "one_vs_three_spearman": spearman(duration_scores[1], duration_scores[3]),
        "two_vs_three_spearman": spearman(duration_scores[2], duration_scores[3]),
        "one_vs_three_mae": float(np.mean(np.abs(duration_scores[1] - duration_scores[3]))),
        "two_vs_three_mae": float(np.mean(np.abs(duration_scores[2] - duration_scores[3]))),
        "zero_score_fraction": {str(k): float(np.mean(duration_scores[k] == 0)) for k in duration_scores},
        "fixed_three_day_reference_diagnostic": {
            "one_vs_three_spearman": spearman(fixed_scores[1], fixed_scores[3]),
            "two_vs_three_spearman": spearman(fixed_scores[2], fixed_scores[3]),
            "one_vs_three_mae": float(np.mean(np.abs(fixed_scores[1] - fixed_scores[3]))),
            "two_vs_three_mae": float(np.mean(np.abs(fixed_scores[2] - fixed_scores[3]))),
            "zero_score_fraction": {str(k): float(np.mean(fixed_scores[k] == 0)) for k in fixed_scores},
        },
    }


def profile_summary(frame):
    reference = fit_structure_reference(frame)
    transformed = transform_structure(frame, reference)
    axes = transformed[["expansionLoad", "contractionLoad", "geometryLoad"]].to_numpy(float)
    names = np.asarray(["expansion", "contraction", "geometry"])
    winner = names[np.argmax(axes, axis=1)]
    winner[np.max(axes, axis=1) <= 0] = "typical"
    rows = []
    for name in ("typical", "expansion", "contraction", "geometry"):
        keep = winner == name
        rows.append({
            "profile": name,
            "n": int(np.sum(keep)),
            "future_fragility_mean": float(transformed.loc[keep, "futureFragility"].mean()) if np.any(keep) else None,
            "future_high_fragility_rate": float(transformed.loc[keep, "futureHighFragility"].mean()) if np.any(keep) else None,
        })
    return rows, transformed


def axis_future_associations(frame):
    _, transformed = profile_summary(frame)
    y = transformed["futureHighFragility"].to_numpy(int)
    future = transformed["futureFragility"].to_numpy(float)
    return {
        feature: {
            "spearman_with_continuous_future_fragility": spearman(transformed[feature], future),
            "auc_for_future_high_fragility_descriptive": roc_auc(y, transformed[feature]),
            "zero_fraction": float(np.mean(transformed[feature].to_numpy(float) == 0)),
        }
        for feature in ("structureScore", "expansionLoad", "contractionLoad", "geometryLoad")
    }


def clinical_associations(frame, cohort):
    _, transformed = profile_summary(frame)
    score_features = ("structureScore", "expansionLoad", "contractionLoad", "geometryLoad")
    if cohort == "weinstock":
        complication_columns = [
            "clinical.Coronary artery disease", "clinical.Diabetic peripheral neuropathy",
            "clinical.Chronic kidney disease", "clinical.Proliferative diabetic retinopathy",
        ]
        complication_count = transformed[complication_columns].sum(axis=1).to_numpy(float)
        any_complication = (complication_count > 0).astype(int)
        pump = (transformed["clinical.InsDeliveryMethod"] == "Pump").astype(int)
        return {
            "by_structure_feature": {
                feature: {
                    "spearman_severe_hypoglycemia_history": spearman(transformed[feature], transformed["clinical.NumSHSinceT1DDiag"]),
                    "spearman_total_insulin_units": spearman(transformed[feature], transformed["clinical.UnitsInsTotal"]),
                    "spearman_complication_count": spearman(transformed[feature], complication_count),
                    "auc_any_complication_diagnostic_only": roc_auc(any_complication, transformed[feature]),
                    "auc_pump_vs_injections_diagnostic_only": roc_auc(pump, transformed[feature]),
                }
                for feature in score_features
            },
            "counts": {"any_complication": int(any_complication.sum()), "pump": int((transformed["clinical.InsDeliveryMethod"] == "Pump").sum())},
        }

    endpoints = ["A1C", "FBG", "ogtt.2hr", "insulin", "SSPG", "hs.CRP", "Trg", "HDL", "LDL", "mage", "modd", "coef_variation", "freq_severe"]
    correlations = {feature: {} for feature in score_features}
    sample_size = {}
    for endpoint in endpoints:
        values = transformed[f"clinical.{endpoint}"].to_numpy(float)
        keep = np.isfinite(values) & (values >= 0)
        for feature in score_features:
            correlations[feature][endpoint] = spearman(transformed[feature].to_numpy(float)[keep], values[keep])
        sample_size[endpoint] = int(np.sum(keep))
    return {"spearman_by_structure_feature": correlations, "n": sample_size}


def hall_clinical_maxT(frame, seed):
    """Selection-adjusted diagnostic for the pre-specified Hall clinical endpoints."""
    _, transformed = profile_summary(frame)
    score_features = ("structureScore", "expansionLoad", "contractionLoad", "geometryLoad")
    endpoints = ("A1C", "FBG", "ogtt.2hr", "insulin", "SSPG")
    scores = {feature: transformed[feature].to_numpy(float) for feature in score_features}
    clinical = {
        endpoint: transformed[f"clinical.{endpoint}"].to_numpy(float)
        for endpoint in endpoints
    }
    observed = {}
    for feature in score_features:
        observed[feature] = {}
        for endpoint in endpoints:
            values = clinical[endpoint]
            keep = np.isfinite(values) & (values >= 0)
            observed[feature][endpoint] = spearman(scores[feature][keep], values[keep])

    rng = np.random.default_rng(seed)
    null_max = np.empty(N_ASSOCIATION_PERMUTATIONS)
    for index in range(N_ASSOCIATION_PERMUTATIONS):
        order = rng.permutation(len(transformed))
        maximum = 0.0
        for feature in score_features:
            for endpoint in endpoints:
                values = clinical[endpoint][order]
                keep = np.isfinite(values) & (values >= 0)
                correlation = spearman(scores[feature][keep], values[keep])
                if np.isfinite(correlation):
                    maximum = max(maximum, abs(correlation))
        null_max[index] = maximum

    adjusted = {
        feature: {
            endpoint: float((1 + np.sum(null_max >= abs(observed[feature][endpoint]))) / (N_ASSOCIATION_PERMUTATIONS + 1))
            if np.isfinite(observed[feature][endpoint]) else None
            for endpoint in endpoints
        }
        for feature in score_features
    }
    return {
        "tests": len(score_features) * len(endpoints),
        "permutations": N_ASSOCIATION_PERMUTATIONS,
        "observed_spearman": observed,
        "maxT_adjusted_p": adjusted,
        "null_max_p95": float(np.quantile(null_max, 0.95)),
    }


def hall_expansion_ogtt_followup(frame, seed):
    """Post-selection influence analysis for the sole maxT-surviving association."""
    _, transformed = profile_summary(frame)
    expansion = transformed["expansionLoad"].to_numpy(float)
    ogtt = transformed["clinical.ogtt.2hr"].to_numpy(float)
    keep = np.isfinite(ogtt) & (ogtt >= 0)
    expansion, ogtt = expansion[keep], ogtt[keep]
    observed = spearman(expansion, ogtt)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.choice(len(ogtt), len(ogtt), True)
        bootstrap[index] = spearman(expansion[sample], ogtt[sample])
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    loo = []
    for omitted in range(len(ogtt)):
        retained = np.arange(len(ogtt)) != omitted
        value = spearman(expansion[retained], ogtt[retained])
        if np.isfinite(value): loo.append(value)
    active = expansion > 0
    return {
        "status": "post-selection exploratory follow-up; not confirmatory",
        "n": len(ogtt),
        "active_expansion_n": int(active.sum()),
        "spearman": observed,
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "leave_one_subject_out_spearman": {
            "min": float(np.min(loo)), "median": float(np.median(loo)), "max": float(np.max(loo)),
        },
        "ogtt_median_active": float(np.median(ogtt[active])) if np.any(active) else None,
        "ogtt_median_inactive": float(np.median(ogtt[~active])) if np.any(~active) else None,
    }


def evaluate(frame, cohort, seed):
    frame, cutoff = add_future_target(frame)
    y = frame["futureHighFragility"].to_numpy(int)
    future = frame["futureFragility"].to_numpy(float)
    predictions, models = {}, {}
    for model_name in MODEL_FEATURES:
        probability, repeat_auc = cross_val_predict(frame, y, model_name, N_REPEATS, seed)
        predictions[model_name] = probability
        models[model_name] = summarize_prediction(y, probability, future, repeat_auc)
    baseline = predictions["conventional"]
    candidate = predictions["conventional_plus_structure"]
    delta = models["conventional_plus_structure"]["roc_auc"] - models["conventional"]["roc_auc"]
    profiles, _ = profile_summary(frame)
    return {
        "n": len(frame), "positives": int(y.sum()), "future_fragility_cutoff": cutoff,
        "future_endpoint_summary": {
            "cv_median": float(frame["futureCV"].median()),
            "out_of_range_fraction_median": float(frame["futureOut"].median()),
        },
        "models": models,
        "delta_auc": float(delta),
        "paired_bootstrap": paired_bootstrap(y, candidate, baseline, seed + 100),
        "fixed_candidate_permutation": fixed_candidate_permutation(frame, y, seed + 200),
        "nested_thresholds": {
            name: nested_threshold(frame, y, name, seed + 300 + index * 100)
            for index, name in enumerate(("conventional", "conventional_plus_structure"))
        },
        "coefficient_stability": coefficient_stability(frame, y, seed + 500),
        "record_length_stability": stability(cohort),
        "profiles": profiles,
        "axis_future_associations_exploratory": axis_future_associations(frame),
        "clinical_associations_exploratory": clinical_associations(frame, cohort),
        "hall_clinical_maxT": hall_clinical_maxT(frame, seed + 900) if cohort == "hall" else None,
        "hall_expansion_ogtt_followup": hall_expansion_ogtt_followup(frame, seed + 1000) if cohort == "hall" else None,
    }


def clean(value):
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    return value


def main():
    weinstock = evaluate(load_window("weinstock", 3), "weinstock", SEED + 30000)
    hall = evaluate(load_window("hall", 3), "hall", SEED + 40000)
    primary = weinstock
    requirements = {
        "weinstock_delta_auc_gte_0_03": primary["delta_auc"] >= 0.03,
        "weinstock_bootstrap_ci_lower_gt_0": primary["paired_bootstrap"]["ci95"][0] > 0,
        "weinstock_permutation_p_lte_0_05": primary["fixed_candidate_permutation"]["one_sided_p"] <= 0.05,
        "weinstock_brier_improves": primary["models"]["conventional_plus_structure"]["brier"] < primary["models"]["conventional"]["brier"],
        "weinstock_two_vs_three_day_rho_gte_0_8": primary["record_length_stability"]["two_vs_three_spearman"] >= 0.8,
        "weinstock_calculability_gte_0_95": primary["record_length_stability"]["calculability_fraction"] >= 0.95,
        "structure_beta_positive_in_gte_0_9_fits": primary["coefficient_stability"]["structure_beta_positive_fraction"] >= 0.9,
        "hall_replication_delta_positive": hall["delta_auc"] > 0,
    }
    eligible = bool(all(requirements.values()))
    results = {
        "protocol": {
            "status": "strict internal temporal validation; not a clinical diagnosis",
            "frozen_structure_score": "median of five clip(abs(training-fold robust-z)-1,0,3) evidences",
            "predictor_window": "elapsed days 0-3",
            "future_window": "elapsed days 3-5",
            "future_fragility": "equal-weight percentile rank of future CV and fraction outside 3.9-10 mmol/L; upper quartile target",
            "outer_cv": f"{N_REPEATS}x{N_SPLITS} subject-level stratified CV",
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutation_replicates": N_PERMUTATIONS,
        },
        "weinstock_primary": weinstock,
        "hall_replication": hall,
        "deployment_gate": {
            "requirements": requirements,
            "eligible": eligible,
            "decision": "add structure resilience module" if eligible else "research only; do not change index.html",
        },
    }
    path = OUTPUT_DIR / "structure_applications_results.json"
    path.write_text(json.dumps(clean(results), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, row in (("Weinstock primary", weinstock), ("Hall replication", hall)):
        base, candidate = row["models"]["conventional"], row["models"]["conventional_plus_structure"]
        print(
            f"{name}: AUC {base['roc_auc']:.3f}->{candidate['roc_auc']:.3f} "
            f"delta={row['delta_auc']:+.3f}, CI={row['paired_bootstrap']['ci95']}, "
            f"perm p={row['fixed_candidate_permutation']['one_sided_p']:.4f}, "
            f"rho2v3={row['record_length_stability']['two_vs_three_spearman']:.3f}"
        )
    print(f"eligible={eligible}; decision={results['deployment_gate']['decision']}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
