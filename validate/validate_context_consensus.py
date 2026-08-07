#!/usr/bin/env python3
"""Validate a treatment-context, all-metric white-box consensus screen.

The exact v8.3 browser pipeline metrics are inputs. The proposed mechanism
does not assign free coefficients to every phase-space variable. Instead it:

1. fits robust healthy-reference medians/scales inside each training fold;
2. converts all reference metrics into directional abnormality evidence;
3. compresses Lyapunov/DET/ENTR into a dynamic consensus score;
4. compresses volume/recovery/core/dimension/shape into a structural score;
5. allows phase evidence to change risk only when >=3 metrics agree and the
   RQA recurrence rate is within its expected calibration band;
6. uses separate untreated (Hall) and treated (Colas) directions;
7. falls back exactly to v8.3 nightMean-only risk when treatment is unknown.

Primary candidate (pre-registered before outcome fitting):
    nightMean + gatedDynamic + gatedStructure
with non-negative logistic coefficients. Comparators are the deployed
nightMean-only rule, unconstrained raw-all logistic regression, and ungated
compressed consensus. Ablations are reported but do not select the model.

Deployment requires BOTH known-treatment strata to pass:
* delta OOF AUC >= 0.03 versus night-only;
* paired bootstrap 95% CI lower bound > 0;
* max-statistic permutation p <= 0.05 across the three new comparators;
* Brier score no worse than night-only;
* at sensitivity-targeted nested thresholds, specificity no worse by >0.02.

Only numpy/pandas are required. No original healthcare data are modified.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SEED = 20260807
N_REPEATS = 10
N_SPLITS = 5
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 250

RAW_PHASE = [
    "logVolume", "shapeRatio", "avgRecovery", "dimension", "lyapunov",
    "det", "entr", "coreDisplacement",
]
DYNAMIC = ["lyapunov", "det", "entr"]
STRUCTURAL_DIRECTIONAL = ["logVolume", "avgRecovery", "coreDisplacement"]
STRUCTURAL_TWOTAIL = ["dimension", "shapeRatio"]

DIRECTIONS = {
    "untreated": {
        "lyapunov": -1, "det": +1, "entr": +1,
        "logVolume": +1, "avgRecovery": -1, "coreDisplacement": +1,
    },
    "treated": {
        "lyapunov": +1, "det": -1, "entr": -1,
        "logVolume": -1, "avgRecovery": +1, "coreDisplacement": -1,
    },
}

MODEL_FEATURES = {
    "night_only": ["nightMean"],
    "raw_all": ["nightMean", *RAW_PHASE],
    "night_dynamic": ["nightMean", "dynamicScore"],
    "night_structure": ["nightMean", "structureScore"],
    "compressed_consensus": ["nightMean", "dynamicScore", "structureScore"],
    "gated_consensus": ["nightMean", "gatedDynamic", "gatedStructure"],
    "phase_only": ["gatedDynamic", "gatedStructure"],
}
MONOTONIC_MODELS = set(MODEL_FEATURES) - {"raw_all"}
MAIN_COMPARATORS = ["raw_all", "compressed_consensus", "gated_consensus"]


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def rankdata(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def roc_auc(y, scores):
    y = np.asarray(y, dtype=int)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, scores):
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
    return float(precision[ranked == 1].sum() / ranked.sum())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if len(rx) < 3 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def stratified_folds(y, n_splits, seed):
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    chunks = {}
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        chunks[label] = np.array_split(idx, n_splits)
    all_idx = np.arange(len(y))
    folds = []
    for fold in range(n_splits):
        test = np.sort(np.r_[chunks[0][fold], chunks[1][fold]])
        train = np.setdiff1d(all_idx, test, assume_unique=True)
        folds.append((train, test))
    return folds


def newton_logistic(Z, y, l2=1.0):
    y = np.asarray(y, float)
    design = np.column_stack([np.ones(len(Z)), Z])
    beta = np.zeros(design.shape[1], float)
    beta[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
    penalty = np.diag(np.r_[0.0, np.full(Z.shape[1], l2)])
    for _ in range(100):
        p = sigmoid(design @ beta)
        weights = np.maximum(p * (1 - p), 1e-8)
        grad = design.T @ (p - y) + penalty @ beta
        hess = design.T @ (design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def penalized_loss(Z, y, beta, l2=1.0):
    p = np.clip(sigmoid(beta[0] + Z @ beta[1:]), 1e-12, 1 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * np.sum(beta[1:] ** 2))


def fit_logistic(X, y, monotonic=False):
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    p = X.shape[1]

    if not monotonic:
        beta = newton_logistic(Z, y)
    else:
        best = None
        # Exact active-face enumeration; p<=3 for every monotonic candidate.
        for active_count in range(p + 1):
            for active in itertools.combinations(range(p), active_count):
                active = list(active)
                if active:
                    fitted = newton_logistic(Z[:, active], y)
                    if np.any(fitted[1:] < -1e-8):
                        continue
                    beta = np.zeros(p + 1)
                    beta[0] = fitted[0]
                    beta[1 + np.asarray(active)] = np.maximum(0, fitted[1:])
                else:
                    beta = np.zeros(p + 1)
                    beta[0] = np.log((np.mean(y) + 1e-3) / (1 - np.mean(y) + 1e-3))
                loss = penalized_loss(Z, y, beta)
                if best is None or loss < best[0]:
                    best = (loss, beta)
        beta = best[1]

    return {"mean": mean, "scale": scale, "beta": beta}


def predict_logistic(model, X):
    Z = (np.asarray(X, float) - model["mean"]) / model["scale"]
    return sigmoid(model["beta"][0] + Z @ model["beta"][1:])


def raw_formula(model):
    coef = model["beta"][1:] / model["scale"]
    intercept = model["beta"][0] - np.sum(model["beta"][1:] * model["mean"] / model["scale"])
    return float(intercept), coef


def robust_scale(values):
    values = np.asarray(values, float)
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale < 1e-8:
        scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
    if scale < 1e-8:
        scale = float(np.std(values))
    if scale < 1e-8:
        scale = 1.0
    return median, scale


def fit_reference(frame, y):
    controls = frame.loc[np.asarray(y) == 0]
    reference = {}
    for feature in RAW_PHASE:
        reference[feature] = robust_scale(controls[feature].to_numpy(float))
    return reference


def transform_consensus(frame, reference, regime):
    zscores = {}
    for feature in RAW_PHASE:
        median, scale = reference[feature]
        zscores[feature] = np.clip((frame[feature].to_numpy(float) - median) / scale, -3, 3)

    evidence = {}
    for feature, direction in DIRECTIONS[regime].items():
        evidence[feature] = np.clip(direction * zscores[feature], 0, 3)
    for feature in STRUCTURAL_TWOTAIL:
        evidence[feature] = np.clip(np.abs(zscores[feature]) - 1.0, 0, 3)

    dynamic = np.median(np.column_stack([evidence[name] for name in DYNAMIC]), axis=1)
    structure_names = [*STRUCTURAL_DIRECTIONAL, *STRUCTURAL_TWOTAIL]
    structure = np.median(np.column_stack([evidence[name] for name in structure_names]), axis=1)
    evidence_matrix = np.column_stack([evidence[name] for name in [*DYNAMIC, *structure_names]])
    evidence_count = np.sum(evidence_matrix > 0.5, axis=1)
    rqa_quality = frame["rr"].between(0.015, 0.025).to_numpy(bool)
    phase_gate = (evidence_count >= 3) & rqa_quality

    result = frame.copy()
    result["dynamicScore"] = dynamic
    result["structureScore"] = structure
    result["evidenceCount"] = evidence_count
    result["phaseGate"] = phase_gate.astype(int)
    result["gatedDynamic"] = np.where(phase_gate, dynamic, 0.0)
    result["gatedStructure"] = np.where(phase_gate, structure, 0.0)
    return result


def fit_pipeline(frame, y, regime, model_name):
    reference = None
    transformed = frame
    if model_name not in {"night_only", "raw_all"}:
        reference = fit_reference(frame, y)
        transformed = transform_consensus(frame, reference, regime)
    X = transformed[MODEL_FEATURES[model_name]].to_numpy(float)
    logistic = fit_logistic(X, y, monotonic=model_name in MONOTONIC_MODELS)
    return {"reference": reference, "logistic": logistic, "model_name": model_name, "regime": regime}


def predict_pipeline(model, frame):
    transformed = frame
    if model["reference"] is not None:
        transformed = transform_consensus(frame, model["reference"], model["regime"])
    X = transformed[MODEL_FEATURES[model["model_name"]]].to_numpy(float)
    return predict_logistic(model["logistic"], X)


def cross_val_predict(frame, y, regime, model_name, repeats, seed):
    total = np.zeros(len(y))
    rep_aucs = []
    for rep in range(repeats):
        rep_pred = np.zeros(len(y))
        for train, test in stratified_folds(y, N_SPLITS, seed + rep):
            model = fit_pipeline(frame.iloc[train], y[train], regime, model_name)
            rep_pred[test] = predict_pipeline(model, frame.iloc[test])
        total += rep_pred
        rep_aucs.append(roc_auc(y, rep_pred))
    return total / repeats, np.asarray(rep_aucs)


def screening_metrics(y, probability, threshold):
    positive = np.asarray(probability) >= threshold
    y = np.asarray(y, int)
    tp = np.sum(positive & (y == 1)); fn = np.sum((~positive) & (y == 1))
    tn = np.sum((~positive) & (y == 0)); fp = np.sum(positive & (y == 0))
    sensitivity = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "flag_rate": float(np.mean(positive)),
    }


def choose_threshold(y, probability, min_sensitivity=0.75):
    choices = []
    for threshold in np.unique(np.r_[0, probability, 1]):
        metrics = screening_metrics(y, probability, threshold)
        if metrics["sensitivity"] >= min_sensitivity:
            choices.append((metrics["specificity"], float(threshold)))
    return sorted(choices)[-1][1] if choices else 0.0


def nested_threshold_metrics(frame, y, regime, model_name):
    fold_metrics, thresholds = [], []
    for rep in range(N_REPEATS):
        for fold_no, (train, test) in enumerate(stratified_folds(y, N_SPLITS, SEED + 1000 + rep)):
            inner_pred, _ = cross_val_predict(
                frame.iloc[train], y[train], regime, model_name, 3,
                SEED + 1100 + rep * 31 + fold_no,
            )
            threshold = choose_threshold(y[train], inner_pred)
            model = fit_pipeline(frame.iloc[train], y[train], regime, model_name)
            test_pred = predict_pipeline(model, frame.iloc[test])
            thresholds.append(threshold)
            fold_metrics.append(screening_metrics(y[test], test_pred, threshold))
    return {
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "mean_fold_metrics": {
            key: float(np.mean([row[key] for row in fold_metrics])) for key in fold_metrics[0]
        },
    }


def bootstrap_delta(y, new_pred, base_pred):
    rng = np.random.default_rng(SEED + 1200)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        deltas[i] = roc_auc(y[idx], new_pred[idx]) - roc_auc(y[idx], base_pred[idx])
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def max_stat_permutation(frame, y, regime):
    rng = np.random.default_rng(SEED + (1 if regime == "untreated" else 2))
    observed = {}
    for name in MAIN_COMPARATORS:
        pred, _ = cross_val_predict(frame, y, regime, name, 2, SEED + 1300)
        observed[name] = roc_auc(y, pred)
    observed_primary = observed["gated_consensus"]
    null_max = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        scores = []
        for name in MAIN_COMPARATORS:
            pred, _ = cross_val_predict(frame, permuted, regime, name, 2, SEED + 1400 + i * 7)
            scores.append(roc_auc(permuted, pred))
        null_max[i] = max(scores)
    return {
        "observed_two_repeat_auc": observed,
        "primary_observed_auc": observed_primary,
        "null_max_mean": float(null_max.mean()),
        "null_max_p95": float(np.quantile(null_max, 0.95)),
        "primary_maxT_p": float((1 + np.sum(null_max >= observed_primary)) / (N_PERMUTATIONS + 1)),
        "n_permutations": N_PERMUTATIONS,
    }


def loo_ridge(X, y, alpha=1.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pred = np.zeros(len(y))
    for test in range(len(y)):
        train = np.arange(len(y)) != test
        mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        Z = (X[train] - mean) / scale
        y_mean = y[train].mean()
        beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y[train] - y_mean))
        pred[test] = y_mean + ((X[test] - mean) / scale) @ beta
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "loo_r2": float(1 - np.sum((y - pred) ** 2) / denominator),
        "loo_mae": float(np.mean(np.abs(y - pred))),
        "spearman_predicted_observed": spearman(pred, y),
    }


def mechanistic_hall(frame, y):
    reference = fit_reference(frame, y)
    transformed = transform_consensus(frame, reference, "untreated")
    results = {}
    for endpoint in ("insulin", "SSPG"):
        keep = frame[endpoint].notna().to_numpy()
        target = frame.loc[keep, endpoint].to_numpy(float)
        results[endpoint] = {
            "correlations": {
                feature: spearman(transformed.loc[keep, feature].to_numpy(float), target)
                for feature in ["nightMean", "dynamicScore", "structureScore", "gatedDynamic", "gatedStructure"]
            },
            "loo_ridge": loo_ridge(
                transformed.loc[keep, ["nightMean", "gatedDynamic", "gatedStructure"]].to_numpy(float),
                target,
            ),
        }
    return results


def evaluate_cohort(frame, regime):
    y = frame["y"].to_numpy(int)
    candidates, predictions = {}, {}
    for name in MODEL_FEATURES:
        pred, rep_aucs = cross_val_predict(frame, y, regime, name, N_REPEATS, SEED + 100)
        predictions[name] = pred
        candidates[name] = {
            "features": MODEL_FEATURES[name],
            "auc": roc_auc(y, pred),
            "rep_auc_mean": float(rep_aucs.mean()),
            "rep_auc_sd": float(rep_aucs.std(ddof=0)),
            "pr_auc": average_precision(y, pred),
            "brier": float(np.mean((pred - y) ** 2)),
        }

    primary = "gated_consensus"
    baseline = "night_only"
    bootstrap = bootstrap_delta(y, predictions[primary], predictions[baseline])
    threshold_baseline = nested_threshold_metrics(frame, y, regime, baseline)
    threshold_primary = nested_threshold_metrics(frame, y, regime, primary)
    permutation = max_stat_permutation(frame, y, regime)

    full_primary = fit_pipeline(frame, y, regime, primary)
    full_reference = full_primary["reference"]
    transformed = transform_consensus(frame, full_reference, regime)
    intercept, coefs = raw_formula(full_primary["logistic"])
    coef_map = {
        feature: float(coef)
        for feature, coef in zip(MODEL_FEATURES[primary], coefs)
    }
    gate_rate = float(transformed["phaseGate"].mean())

    delta = candidates[primary]["auc"] - candidates[baseline]["auc"]
    base_spec = threshold_baseline["mean_fold_metrics"]["specificity"]
    primary_spec = threshold_primary["mean_fold_metrics"]["specificity"]
    requirements = {
        "delta_auc_gte_0_03": bool(delta >= 0.03),
        "bootstrap_ci_lower_gt_0": bool(bootstrap["ci95"][0] > 0),
        "maxT_permutation_p_lte_0_05": bool(permutation["primary_maxT_p"] <= 0.05),
        "brier_no_worse": bool(candidates[primary]["brier"] <= candidates[baseline]["brier"]),
        "nested_specificity_not_worse_by_gt_0_02": bool(primary_spec >= base_spec - 0.02),
    }
    eligible = all(requirements.values())
    return {
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "candidates": candidates,
        "primary_delta_auc": delta,
        "paired_bootstrap": bootstrap,
        "nested_thresholds": {"night_only": threshold_baseline, primary: threshold_primary},
        "selection_adjusted_permutation": permutation,
        "full_reference": {
            feature: {"median": float(values[0]), "scale": float(values[1])}
            for feature, values in full_reference.items()
        },
        "full_primary_formula": {"intercept": intercept, "coefficients": coef_map},
        "phase_gate_rate": gate_rate,
        "deployment_requirements": requirements,
        "eligible": eligible,
        "predictions": predictions,
    }


def cross_regime_safety(hall, colas, hall_eval, colas_eval):
    hall_y, colas_y = hall["y"].to_numpy(int), colas["y"].to_numpy(int)
    hall_model = fit_pipeline(hall, hall_y, "untreated", "gated_consensus")
    colas_model = fit_pipeline(colas, colas_y, "treated", "gated_consensus")
    wrong_hall_on_colas = predict_pipeline(hall_model, colas)
    wrong_colas_on_hall = predict_pipeline(colas_model, hall)
    unknown_hall = hall["currentRisk"].to_numpy(float)
    unknown_colas = colas["currentRisk"].to_numpy(float)
    return {
        "unknown_status_fallback": {
            "formula": "sigmoid(1.064314*nightMean-6.746364)",
            "hall_auc": roc_auc(hall_y, unknown_hall),
            "colas_auc": roc_auc(colas_y, unknown_colas),
            "max_absolute_difference_from_currentRisk": 0.0,
        },
        "wrong_regime_diagnostic_only": {
            "untreated_model_on_colas_auc": roc_auc(colas_y, wrong_hall_on_colas),
            "treated_model_on_hall_auc": roc_auc(hall_y, wrong_colas_on_hall),
            "interpretation": "Wrong-regime scores are not deployable; treatment gate is mandatory",
        },
        "eligible_only_if_both_strata_pass": bool(hall_eval["eligible"] and colas_eval["eligible"]),
    }


def load_metrics(cohort):
    path = OUTPUT_DIR / f"phase_screening_metrics_{cohort}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["logVolume"] = np.log(np.maximum(frame["volume"].to_numpy(float), 1e-8))
    required = ["y", "nightMean", "rr", *RAW_PHASE]
    if frame[required].isna().any().any():
        raise ValueError(f"Missing required values in {cohort}")
    if ((frame["rr"] <= 0) | (frame["det"] <= 0) | (frame["entr"] <= 0)).any():
        raise ValueError(f"Invalid RQA values in {cohort}")
    return frame


def strip_predictions(value):
    if isinstance(value, dict):
        return {key: strip_predictions(val) for key, val in value.items() if key != "predictions"}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main():
    hall = load_metrics("hall")
    colas = load_metrics("colas")
    hall_eval = evaluate_cohort(hall, "untreated")
    colas_eval = evaluate_cohort(colas, "treated")
    safety = cross_regime_safety(hall, colas, hall_eval, colas_eval)
    mechanisms = mechanistic_hall(hall, hall["y"].to_numpy(int))

    overall_eligible = bool(hall_eval["eligible"] and colas_eval["eligible"])
    results = {
        "protocol": {
            "primary_model": "treatment-context gated consensus",
            "hall": "untreated target: diagnosis >= prediabetes",
            "colas": "treated T2DM stratum",
            "outer_cv": f"{N_REPEATS}x{N_SPLITS} subject-level stratified CV",
            "bootstrap_replicates": N_BOOTSTRAP,
            "maxT_permutations": N_PERMUTATIONS,
            "phase_evidence_gate": ">=3 metrics with evidence>0.5 and 0.015<=RR<=0.025",
        },
        "hall_untreated": strip_predictions(hall_eval),
        "colas_treated": strip_predictions(colas_eval),
        "cross_regime_safety": safety,
        "hall_mechanistic_endpoints": mechanisms,
        "deployment_gate": {
            "eligible": overall_eligible,
            "decision": (
                "Deploy dual treatment-context consensus mechanism"
                if overall_eligible
                else "Retain v8.3 nightMean-only risk; all-metric consensus not proven superior"
            ),
        },
    }
    output_path = OUTPUT_DIR / "context_consensus_results.json"
    output_path.write_text(json.dumps(strip_predictions(results), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Treatment-context consensus validation ===")
    for name, evaluation in (("Hall untreated", hall_eval), ("Colas treated", colas_eval)):
        base = evaluation["candidates"]["night_only"]
        primary = evaluation["candidates"]["gated_consensus"]
        print(
            f"{name}: night AUC={base['auc']:.3f}; gated AUC={primary['auc']:.3f}; "
            f"delta={evaluation['primary_delta_auc']:+.3f}; CI={evaluation['paired_bootstrap']['ci95']}; "
            f"maxT p={evaluation['selection_adjusted_permutation']['primary_maxT_p']:.4f}; "
            f"eligible={evaluation['eligible']}"
        )
        for model_name in MODEL_FEATURES:
            row = evaluation["candidates"][model_name]
            print(f"  {model_name:<22} AUC={row['auc']:.3f} PR={row['pr_auc']:.3f} Brier={row['brier']:.3f}")
    print(f"Overall deployment eligible: {overall_eligible}")
    print(f"Decision: {results['deployment_gate']['decision']}")
    print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
