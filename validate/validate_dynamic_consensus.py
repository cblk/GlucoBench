#!/usr/bin/env python3
"""Strict internal validation of the frozen nightMean + dynamic consensus.

The candidate was frozen before this run. All healthy references, scaling,
coefficients, and screening thresholds are fit inside training subjects only.
Hall (untreated) and Colas (treated) use opposite pre-specified directions and
are never pooled. This is strict internal validation, not external confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_context_consensus import (
    DYNAMIC, DIRECTIONS, OUTPUT_DIR, ROOT, SEED, average_precision,
    choose_threshold, cross_val_predict, fit_logistic, fit_pipeline,
    fit_reference, load_metrics, newton_logistic, predict_logistic,
    predict_pipeline, rankdata, roc_auc, screening_metrics, spearman,
    stratified_folds,
)


N_REPEATS = 20
N_SPLITS = 5
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 1000
SENSITIVITY_TARGET = 0.75
MODELS = ("night_only", "night_dynamic")


def basic_metrics(y, probability):
    return {
        "roc_auc": roc_auc(y, probability),
        "pr_auc": average_precision(y, probability),
        "brier": float(np.mean((np.asarray(probability) - y) ** 2)),
    }


def calibration(y, probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))[:, None]
    beta = newton_logistic(logit, y, l2=0.0)
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def paired_bootstrap(y, candidate, baseline, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        deltas[i] = roc_auc(y[idx], candidate[idx]) - roc_auc(y[idx], baseline[idx])
    return {
        "replicates": N_BOOTSTRAP,
        "mean_delta_auc": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def fixed_candidate_permutation(frame, y, regime, seed):
    """One-sided incremental test using the same fixed candidate each time."""
    observed = {}
    for name in MODELS:
        pred, _ = cross_val_predict(frame, y, regime, name, 1, seed)
        observed[name] = roc_auc(y, pred)
    observed_delta = observed["night_dynamic"] - observed["night_only"]
    rng = np.random.default_rng(seed + 1)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        base, _ = cross_val_predict(frame, permuted, regime, "night_only", 1, seed + 10 + i)
        candidate, _ = cross_val_predict(frame, permuted, regime, "night_dynamic", 1, seed + 10 + i)
        null[i] = roc_auc(permuted, candidate) - roc_auc(permuted, base)
    return {
        "replicates": N_PERMUTATIONS,
        "observed_one_repeat_auc": observed,
        "observed_delta_auc": float(observed_delta),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "one_sided_p": float((1 + np.sum(null >= observed_delta)) / (N_PERMUTATIONS + 1)),
    }


def nested_threshold(frame, y, regime, model_name, seed):
    fold_rows, thresholds = [], []
    for rep in range(N_REPEATS):
        for fold_no, (train, test) in enumerate(stratified_folds(y, N_SPLITS, seed + rep)):
            inner, _ = cross_val_predict(
                frame.iloc[train], y[train], regime, model_name, 3,
                seed + 1000 + rep * 37 + fold_no,
            )
            threshold = choose_threshold(y[train], inner, SENSITIVITY_TARGET)
            model = fit_pipeline(frame.iloc[train], y[train], regime, model_name)
            probability = predict_pipeline(model, frame.iloc[test])
            rows = screening_metrics(y[test], probability, threshold)
            rows.update({"n": int(len(test)), "threshold": float(threshold)})
            fold_rows.append(rows)
            thresholds.append(threshold)
    weights = np.asarray([row["n"] for row in fold_rows], float)
    return {
        "target_sensitivity": SENSITIVITY_TARGET,
        "outer_fits": len(fold_rows),
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "weighted_mean": {
            key: float(np.average([row[key] for row in fold_rows], weights=weights))
            for key in ("sensitivity", "specificity", "balanced_accuracy", "flag_rate")
        },
    }


def coefficient_stability(frame, y, regime, seed):
    rows = []
    for rep in range(N_REPEATS):
        for train, _ in stratified_folds(y, N_SPLITS, seed + rep):
            model = fit_pipeline(frame.iloc[train], y[train], regime, "night_dynamic")
            beta = model["logistic"]["beta"]
            rows.append((float(beta[1]), float(beta[2])))
    values = np.asarray(rows)
    return {
        "outer_fits": int(len(values)),
        "night_beta_positive_fraction": float(np.mean(values[:, 0] > 1e-8)),
        "dynamic_beta_positive_fraction": float(np.mean(values[:, 1] > 1e-8)),
        "dynamic_beta_median": float(np.median(values[:, 1])),
        "dynamic_beta_iqr": [float(np.quantile(values[:, 1], 0.25)), float(np.quantile(values[:, 1], 0.75))],
    }


def evidence_frame(frame, reference, regime, components):
    evidence = []
    for feature in components:
        center, scale = reference[feature]
        z = np.clip((frame[feature].to_numpy(float) - center) / scale, -3, 3)
        evidence.append(np.clip(DIRECTIONS[regime][feature] * z, 0, 3))
    result = frame.copy()
    result["ablationDynamic"] = np.median(np.column_stack(evidence), axis=1)
    return result


def component_oof(frame, y, regime, components, repeats, seed):
    total = np.zeros(len(y))
    for rep in range(repeats):
        fold_probability = np.zeros(len(y))
        for train, test in stratified_folds(y, N_SPLITS, seed + rep):
            reference = fit_reference(frame.iloc[train], y[train])
            train_frame = evidence_frame(frame.iloc[train], reference, regime, components)
            test_frame = evidence_frame(frame.iloc[test], reference, regime, components)
            features = ["nightMean", "ablationDynamic"]
            logistic = fit_logistic(train_frame[features].to_numpy(float), y[train], monotonic=True)
            fold_probability[test] = predict_logistic(logistic, test_frame[features].to_numpy(float))
        total += fold_probability
    return total / repeats


def ablations(frame, y, regime, seed):
    result = {}
    definitions = {
        "all_three": tuple(DYNAMIC),
        **{f"only_{feature}": (feature,) for feature in DYNAMIC},
        **{f"without_{feature}": tuple(name for name in DYNAMIC if name != feature) for feature in DYNAMIC},
    }
    for name, components in definitions.items():
        # Every ablation uses identical subject folds for a paired comparison.
        pred = component_oof(frame, y, regime, components, N_REPEATS, seed)
        result[name] = {"components": list(components), **basic_metrics(y, pred)}
    return result


def leave_one_positive_out(frame, y, regime, seed):
    rows = []
    for order, omitted in enumerate(np.flatnonzero(y == 1)):
        keep = np.arange(len(y)) != omitted
        deltas = {}
        for name in MODELS:
            pred, _ = cross_val_predict(frame.loc[keep].reset_index(drop=True), y[keep], regime, name, 5, seed + order * 20)
            deltas[name] = roc_auc(y[keep], pred)
        rows.append({
            "omitted_id": str(frame.iloc[omitted]["id"]),
            "night_only_auc": deltas["night_only"],
            "night_dynamic_auc": deltas["night_dynamic"],
            "delta_auc": deltas["night_dynamic"] - deltas["night_only"],
        })
    delta = np.asarray([row["delta_auc"] for row in rows])
    return {
        "runs": len(rows),
        "cv_per_run": "5x5 subject-level stratified CV",
        "positive_delta_fraction": float(np.mean(delta > 0)),
        "delta_median": float(np.median(delta)),
        "delta_range": [float(np.min(delta)), float(np.max(delta))],
        "details": rows,
    }


def load_prefix(name):
    path = OUTPUT_DIR / f"dynamic_prefix_metrics_{name}.json"
    if not path.exists():
        return None
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["logVolume"] = np.log(np.maximum(frame["volume"].to_numpy(float), 1e-8))
    return frame


def prefix_stability(full, y, regime, cohort, seed):
    full_model = fit_pipeline(full, y, regime, "night_dynamic")
    full_probability = predict_pipeline(full_model, full)
    full_by_id = dict(zip(full["id"].astype(str), full_probability))
    result = {}
    for k in (1, 2, 3, 5):
        prefix = load_prefix(f"{cohort}_k{k}")
        if prefix is None:
            result[str(k)] = {"available": False, "reason": "insufficient cohort record length"}
            continue
        required = ["nightMean", "lyapunov", "det", "entr", "rr"]
        valid = ~prefix[required].isna().any(axis=1)
        invalid_count = int((~valid).sum())
        prefix = prefix.loc[valid].reset_index(drop=True)
        prefix_y = prefix["y"].to_numpy(int)
        oof, _ = cross_val_predict(prefix, prefix_y, regime, "night_dynamic", 10, seed + k * 10)
        applied = predict_pipeline(full_model, prefix)
        matched_full = np.asarray([full_by_id[str(value)] for value in prefix["id"]])
        result[str(k)] = {
            "available": True,
            "n": int(len(prefix)),
            "positives": int(prefix_y.sum()),
            "excluded_uncalculable": invalid_count,
            "calculability_fraction": float(len(prefix) / (len(prefix) + invalid_count)),
            "oof_10x5": basic_metrics(prefix_y, oof),
            "full_model_score_stability": {
                "spearman_vs_full_record": spearman(applied, matched_full),
                "mean_absolute_probability_change": float(np.mean(np.abs(applied - matched_full))),
            },
        }
    return result


def evaluate(frame, regime, cohort, seed):
    y = frame["y"].to_numpy(int)
    predictions, models = {}, {}
    for name in MODELS:
        prediction, repeat_auc = cross_val_predict(frame, y, regime, name, N_REPEATS, seed)
        predictions[name] = prediction
        models[name] = {
            **basic_metrics(y, prediction),
            "repeat_auc_mean": float(np.mean(repeat_auc)),
            "repeat_auc_sd": float(np.std(repeat_auc)),
            "calibration": calibration(y, prediction),
        }
    delta = models["night_dynamic"]["roc_auc"] - models["night_only"]["roc_auc"]
    bootstrap = paired_bootstrap(y, predictions["night_dynamic"], predictions["night_only"], seed + 100)
    permutation = fixed_candidate_permutation(frame, y, regime, seed + 200)
    thresholds = {
        name: nested_threshold(frame, y, regime, name, seed + 300 + index * 100)
        for index, name in enumerate(MODELS)
    }
    coefficients = coefficient_stability(frame, y, regime, seed + 500)
    influence = leave_one_positive_out(frame, y, regime, seed + 600)
    requirements = {
        "delta_auc_gte_0_03": bool(delta >= 0.03),
        "bootstrap_ci_lower_gt_0": bool(bootstrap["ci95"][0] > 0),
        "fixed_candidate_permutation_p_lte_0_05": bool(permutation["one_sided_p"] <= 0.05),
        "brier_improves": bool(models["night_dynamic"]["brier"] < models["night_only"]["brier"]),
        "specificity_not_worse_at_sensitivity_target": bool(
            thresholds["night_dynamic"]["weighted_mean"]["specificity"]
            >= thresholds["night_only"]["weighted_mean"]["specificity"]
        ),
        "dynamic_beta_positive_in_gte_90pct_fits": bool(coefficients["dynamic_beta_positive_fraction"] >= 0.90),
        "most_lopo_runs_retain_positive_delta": bool(influence["positive_delta_fraction"] > 0.50),
    }
    return {
        "n": int(len(frame)), "positives": int(y.sum()), "models": models,
        "delta_auc": float(delta), "paired_bootstrap": bootstrap,
        "fixed_candidate_permutation": permutation, "nested_thresholds": thresholds,
        "coefficient_stability": coefficients, "leave_one_positive_out": influence,
        "ablations": ablations(frame, y, regime, seed),
        "record_length_stability": prefix_stability(frame, y, regime, cohort, seed + 800),
        "deployment_requirements": requirements, "eligible": bool(all(requirements.values())),
    }


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    hall, colas = load_metrics("hall"), load_metrics("colas")
    hall_result = evaluate(hall, "untreated", "hall", SEED + 10000)
    colas_result = evaluate(colas, "treated", "colas", SEED + 20000)
    dual_eligible = bool(hall_result["eligible"] and colas_result["eligible"])
    results = {
        "protocol": {
            "status": "strict internal validation; not independent external confirmation",
            "frozen_candidate": "logit(P)=b0+b1*nightMean+b2*median(directionally clipped robust-z Lyapunov, DET, ENTR); b1,b2>=0",
            "hall_direction": "low Lyapunov, high DET, high ENTR",
            "colas_direction": "high Lyapunov, low DET, low ENTR",
            "outer_cv": f"{N_REPEATS}x{N_SPLITS} subject-level stratified CV",
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutation_replicates": N_PERMUTATIONS,
            "seed": SEED,
        },
        "hall_untreated": hall_result,
        "colas_treated": colas_result,
        "unknown_treatment_fallback": {
            "formula": "sigmoid(1.064314*nightMean-6.746364)",
            "max_absolute_difference_from_current_v8_3": 0.0,
        },
        "deployment_gate": {
            "hall_primary_eligible": hall_result["eligible"],
            "colas_separate_eligible": colas_result["eligible"],
            "dual_context_eligible": dual_eligible,
            "decision": "deploy frozen dual-context candidate" if dual_eligible else "retain v8.3 nightMean-only risk",
        },
    }
    path = OUTPUT_DIR / "dynamic_consensus_results.json"
    path.write_text(json.dumps(clean(results), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, row in (("Hall untreated", hall_result), ("Colas treated", colas_result)):
        base, candidate = row["models"]["night_only"], row["models"]["night_dynamic"]
        print(
            f"{name}: AUC {base['roc_auc']:.3f}->{candidate['roc_auc']:.3f} "
            f"(delta {row['delta_auc']:+.3f}, CI {row['paired_bootstrap']['ci95']}, "
            f"perm p={row['fixed_candidate_permutation']['one_sided_p']:.4f}); eligible={row['eligible']}"
        )
        print(f"  requirements={row['deployment_requirements']}")
    print(f"Decision: {results['deployment_gate']['decision']}")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
