#!/usr/bin/env python3
"""Explore a bounded, stability-regularized five-feature CGM score.

Hall has already been inspected in a prior cycle. This script therefore treats
Hall as an exploratory cohort and uses leave-one-cohort-out evaluation rather
than claiming a new external validation. Iglu, Dubosson, and Weinstock provide
unlabelled 24/48-hour stability pairs only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold

from validate_composite_abnormality import (
    BASE_FEATURES,
    OUTPUT,
    ROOT,
    RobustTransformer,
    bootstrap_auc,
    cgmacros_sensor_average,
    icc_absolute_agreement,
    json_ready,
    load_features,
    safe_auc,
    safe_spearman,
)
from validate_clinical_continuum import (
    bootstrap_regression,
    leave_one_out_threshold_metrics,
    prepare_data,
    regression_metrics,
)


RESULT_PATH = OUTPUT / "stability_base5_results.json"
FEATURE_PATH = OUTPUT / "stability_base5_features.csv"

SEED = 20260810 + 500
INNER_FOLDS = 3
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03)
STABILITY_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0)
ELASTIC_ALPHA = 0.50
BOOTSTRAPS = 3000
COEFFICIENT_BOOTSTRAPS = 250
CLINICAL_COHORTS = ("cgmacros", "colas", "hall")
STRESS_COHORTS = ("iglu", "dubosson", "weinstock")

FAMILIES = {
    "night_mean": ("night_mean",),
    "anchor_only": ("anchor_level",),
    "base5_no_anchor": tuple(feature for feature in BASE_FEATURES if feature != "anchor_level"),
    "bounded_base5": tuple(BASE_FEATURES),
    "stability_base5": tuple(BASE_FEATURES),
}


def soft_component(value, lower, upper, slope=4.0):
    value = np.asarray(value, float)
    return expit(slope * (value - lower) / (upper - lower))


def weighted_max_mean(values, max_weight=0.60):
    values = np.asarray(values, float)
    return max_weight * np.max(values, axis=1) + (1.0 - max_weight) * np.mean(values, axis=1)


def ccas_core_variant(a1c, fpg, slope=4.0, max_weight=0.60):
    components = np.column_stack([
        soft_component(a1c, 5.7, 6.5, slope),
        soft_component(fpg, 100.0, 126.0, slope),
    ])
    return 100.0 * weighted_max_mean(components, max_weight)


def ccas_full_variant(a1c, fpg, ogtt, sspg=None, slope=4.0, max_weight=0.60, sspg_scale=25.0):
    glycemic = weighted_max_mean(np.column_stack([
        soft_component(a1c, 5.7, 6.5, slope),
        soft_component(fpg, 100.0, 126.0, slope),
        soft_component(ogtt, 140.0, 200.0, slope),
    ]), max_weight)
    if sspg is None:
        return 100.0 * glycemic
    q_sspg = expit((np.asarray(sspg, float) - 150.0) / sspg_scale)
    return 100.0 * weighted_max_mean(np.column_stack([glycemic, q_sspg]), max_weight)


def prepare_clinical_cohorts():
    feature_rows, cgm, colas, hall = prepare_data()
    cohorts = {
        "cgmacros": cgm.copy(),
        "colas": colas.copy(),
        "hall": hall[hall["ccas_core"].notna()].copy(),
    }
    for name, frame in cohorts.items():
        frame["id"] = frame["id"].astype(str)
        frame["clinical_cohort"] = name
        frame["target"] = frame["ccas_core"].to_numpy(float)
        cohorts[name] = frame.reset_index(drop=True)
    return feature_rows, cohorts, hall.copy()


def pair_frames(merged, suffix_a, suffix_b, features):
    left = pd.DataFrame({feature: merged[f"{feature}_{suffix_a}"] for feature in features})
    right = pd.DataFrame({feature: merged[f"{feature}_{suffix_b}"] for feature in features})
    left["id"] = merged["id"].astype(str).to_numpy()
    right["id"] = merged["id"].astype(str).to_numpy()
    return left, right


def build_pair_groups(feature_rows):
    features = tuple(dict.fromkeys((*BASE_FEATURES, "night_mean")))
    frame = feature_rows.copy()
    frame["id"] = frame["id"].astype(str)
    frame["sensor_key"] = frame["sensor"].fillna("").astype(str)
    groups = []

    for (source, sensor), subset in frame.groupby(["source_cohort", "sensor_key"], dropna=False):
        a = subset[subset["window_hours"] == 24][["id", *features]]
        b = subset[subset["window_hours"] == 48][["id", *features]]
        merged = a.merge(b, on="id", suffixes=("_24", "_48"), validate="one_to_one")
        if len(merged):
            left, right = pair_frames(merged, "24", "48", features)
            groups.append({
                "name": f"time_{source}_{sensor or 'default'}",
                "kind": "time",
                "source_cohort": source,
                "a": left,
                "b": right,
            })

    cgm = frame[frame["source_cohort"] == "cgmacros"]
    for hours in (24, 48):
        libre = cgm[(cgm["window_hours"] == hours) & (cgm["sensor_key"] == "libre")][["id", *features]]
        dexcom = cgm[(cgm["window_hours"] == hours) & (cgm["sensor_key"] == "dexcom")][["id", *features]]
        merged = libre.merge(dexcom, on="id", suffixes=("_libre", "_dexcom"), validate="one_to_one")
        left, right = pair_frames(merged, "libre", "dexcom", features)
        groups.append({
            "name": f"sensor_cgmacros_w{hours}",
            "kind": "sensor",
            "source_cohort": "cgmacros",
            "a": left,
            "b": right,
        })
    return groups


def training_pair_groups(pair_groups, training_frames):
    allowed = {name: set(frame["id"].astype(str)) for name, frame in training_frames.items()}
    selected = []
    for group in pair_groups:
        source = group["source_cohort"]
        if source in CLINICAL_COHORTS:
            if source not in allowed:
                continue
            keep = group["a"]["id"].astype(str).isin(allowed[source]).to_numpy()
            if not np.any(keep):
                continue
            current = dict(group)
            current["a"] = group["a"].loc[keep].reset_index(drop=True)
            current["b"] = group["b"].loc[keep].reset_index(drop=True)
            selected.append(current)
        else:
            selected.append(group)
    return selected


def soft_logloss(y, prediction):
    y = np.asarray(y, float) / 100.0
    prediction = np.clip(np.asarray(prediction, float) / 100.0, 1e-9, 1.0 - 1e-9)
    return float(np.mean(-y * np.log(prediction) - (1.0 - y) * np.log(1.0 - prediction)))


def fit_bounded_model(training_frames, features, lam, stability_weight, pair_groups):
    features = tuple(features)
    transformer = RobustTransformer(features).fit(list(training_frames.values()))
    x_parts, y_parts, weight_parts = [], [], []
    cohort_weight = 1.0 / len(training_frames)
    for frame in training_frames.values():
        x_parts.append(transformer.transform(frame))
        y_parts.append(frame["target"].to_numpy(float) / 100.0)
        weight_parts.append(np.full(len(frame), cohort_weight / len(frame)))
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    sample_weight = np.concatenate(weight_parts)
    transformed_pairs = [
        (transformer.transform(group["a"]), transformer.transform(group["b"]))
        for group in pair_groups
    ]
    p = len(features)

    def objective(params):
        weights, intercept = params[:p], params[p]
        eta = intercept + x @ weights
        probability = expit(eta)
        supervised = np.sum(sample_weight * (np.logaddexp(0.0, eta) - y * eta))
        gradient_w = x.T @ (sample_weight * (probability - y))
        gradient_b = float(np.sum(sample_weight * (probability - y)))

        stability_loss = 0.0
        if stability_weight > 0 and transformed_pairs:
            stability_gradient_w = np.zeros(p)
            stability_gradient_b = 0.0
            for xa, xb in transformed_pairs:
                pa = expit(intercept + xa @ weights)
                pb = expit(intercept + xb @ weights)
                difference = pa - pb
                dpa = pa * (1.0 - pa)
                dpb = pb * (1.0 - pb)
                stability_loss += 0.5 * float(np.mean(difference ** 2))
                stability_gradient_w += np.mean(
                    difference[:, None] * (dpa[:, None] * xa - dpb[:, None] * xb), axis=0
                )
                stability_gradient_b += float(np.mean(difference * (dpa - dpb)))
            divisor = len(transformed_pairs)
            stability_loss /= divisor
            gradient_w += stability_weight * stability_gradient_w / divisor
            gradient_b += stability_weight * stability_gradient_b / divisor

        smooth_l1 = np.sqrt(weights ** 2 + 1e-8)
        penalty = lam * (
            ELASTIC_ALPHA * smooth_l1.sum()
            + 0.5 * (1.0 - ELASTIC_ALPHA) * np.dot(weights, weights)
        )
        gradient_w += lam * (
            ELASTIC_ALPHA * weights / smooth_l1
            + (1.0 - ELASTIC_ALPHA) * weights
        )
        value = supervised + stability_weight * stability_loss + penalty
        return value, np.r_[gradient_w, gradient_b]

    weighted_mean = float(np.sum(sample_weight * y))
    weighted_mean = np.clip(weighted_mean, 1e-5, 1.0 - 1e-5)
    initial = np.zeros(p + 1)
    initial[p] = np.log(weighted_mean / (1.0 - weighted_mean))
    fit = minimize(
        lambda params: objective(params), initial, method="L-BFGS-B", jac=True,
        bounds=[(0.0, None)] * p + [(None, None)],
        options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fit.success:
        raise RuntimeError(f"bounded base5 optimization failed: {fit.message}")
    return {
        "features": features,
        "transformer": transformer,
        "weights": fit.x[:p],
        "intercept": float(fit.x[p]),
        "lambda": float(lam),
        "stability_weight": float(stability_weight),
        "objective": float(fit.fun),
    }


def predict(model, frame):
    transformed = model["transformer"].transform(frame)
    return 100.0 * expit(model["intercept"] + transformed @ model["weights"])


def formula_export(model):
    return {
        "features": list(model["features"]),
        "transform": {
            feature: "log1p(max(x,0))" if feature in {
                "hyper_burden", "hypo_burden", "variation_load", "recovery_debt"
            } else "identity"
            for feature in model["features"]
        },
        "standardization": model["transformer"].export(),
        "weights": {
            feature: float(value) for feature, value in zip(model["features"], model["weights"])
        },
        "intercept": model["intercept"],
        "lambda": model["lambda"],
        "stability_weight": model["stability_weight"],
        "equation": "score = 100*sigmoid(intercept + sum(weight_j*z_j)); all weights >= 0",
    }


def simultaneous_folds(frames, folds, seed):
    splits = {}
    for offset, (name, frame) in enumerate(frames.items()):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + 97 * offset)
        splits[name] = list(splitter.split(frame))
    for fold in range(folds):
        training = {name: frame.iloc[splits[name][fold][0]].reset_index(drop=True) for name, frame in frames.items()}
        validation = {name: frame.iloc[splits[name][fold][1]].reset_index(drop=True) for name, frame in frames.items()}
        yield training, validation


def validation_loss(validation_frames, model):
    return float(np.mean([
        soft_logloss(frame["target"], predict(model, frame))
        for frame in validation_frames.values()
    ]))


def select_parameters(training_frames, features, pair_groups, stability_grid, seed):
    rows = []
    folds = list(simultaneous_folds(training_frames, INNER_FOLDS, seed))
    for lam in LAMBDAS:
        for stability_weight in stability_grid:
            losses = []
            for fold_training, validation in folds:
                fold_pairs = training_pair_groups(pair_groups, fold_training)
                model = fit_bounded_model(
                    fold_training, features, lam, stability_weight, fold_pairs
                )
                losses.append(validation_loss(validation, model))
            rows.append({
                "lambda": lam,
                "stability_weight": stability_weight,
                "mean_validation_logloss": float(np.mean(losses)),
            })
    best = min(rows, key=lambda row: (
        row["mean_validation_logloss"], -row["lambda"], row["stability_weight"]
    ))
    return best, rows


def fit_loco_family(cohorts, pair_groups, features, stability_grid, seed_offset=0):
    predictions, fits = {}, {}
    for holdout_index, holdout in enumerate(CLINICAL_COHORTS):
        training = {name: frame for name, frame in cohorts.items() if name != holdout}
        best, grid = select_parameters(
            training, features, pair_groups, stability_grid,
            SEED + seed_offset + 1000 * holdout_index,
        )
        training_pairs = training_pair_groups(pair_groups, training)
        model = fit_bounded_model(
            training, features, best["lambda"], best["stability_weight"], training_pairs
        )
        predictions[holdout] = predict(model, cohorts[holdout])
        fits[holdout] = {
            "selected": best,
            "selection_grid": grid,
            "formula": formula_export(model),
        }
    return predictions, fits


def percentile_interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def macro_bootstrap(cohorts, prediction, baselines=None, seed=SEED + 7000):
    baselines = baselines or {}
    rng = np.random.default_rng(seed)
    rho_values = []
    delta_values = {name: [] for name in baselines}
    for _ in range(BOOTSTRAPS):
        cohort_rho = []
        cohort_delta = {name: [] for name in baselines}
        for cohort in CLINICAL_COHORTS:
            y = cohorts[cohort]["target"].to_numpy(float)
            index = rng.integers(0, len(y), len(y))
            current = safe_spearman(y[index], prediction[cohort][index])
            if not np.isfinite(current):
                continue
            cohort_rho.append(current)
            for name, baseline in baselines.items():
                cohort_delta[name].append(
                    current - safe_spearman(y[index], baseline[cohort][index])
                )
        if len(cohort_rho) == len(CLINICAL_COHORTS):
            rho_values.append(float(np.mean(cohort_rho)))
            for name in baselines:
                delta_values[name].append(float(np.mean(cohort_delta[name])))
    result = {
        "spearman_ci95": percentile_interval(rho_values),
        "bootstrap_replicates": len(rho_values),
    }
    for name, values in delta_values.items():
        result[f"delta_vs_{name}_ci95"] = percentile_interval(values)
    return result


def evaluate_loco(cohorts, predictions, baselines=None, seed=SEED + 8000):
    by_cohort = {
        name: bootstrap_regression(frame["target"], predictions[name], seed=seed + index * 31)
        for index, (name, frame) in enumerate(cohorts.items())
    }
    result = {
        "by_cohort": by_cohort,
        "macro_mean_spearman": float(np.mean([item["spearman"] for item in by_cohort.values()])),
        "macro_mean_rmse": float(np.mean([item["rmse"] for item in by_cohort.values()])),
    }
    if baselines:
        for baseline_name, baseline_predictions in baselines.items():
            baseline_rho = np.mean([
                safe_spearman(cohorts[name]["target"], baseline_predictions[name])
                for name in CLINICAL_COHORTS
            ])
            result[f"delta_macro_spearman_vs_{baseline_name}"] = (
                result["macro_mean_spearman"] - baseline_rho
            )
    result.update(macro_bootstrap(cohorts, predictions, baselines, seed + 400))
    return result


def refit_loco_fixed(cohorts, pair_groups, features, selected_by_holdout):
    predictions = {}
    for holdout in CLINICAL_COHORTS:
        training = {name: frame for name, frame in cohorts.items() if name != holdout}
        selected = selected_by_holdout[holdout]
        model = fit_bounded_model(
            training, features, selected["lambda"], selected["stability_weight"],
            training_pair_groups(pair_groups, training),
        )
        predictions[holdout] = predict(model, cohorts[holdout])
    return predictions


def target_sensitivity(original_cohorts, pair_groups, selected_by_holdout):
    results = {}
    for slope in (2.0, 4.0, 6.0):
        for max_weight in (0.50, 0.60, 0.75):
            cohorts = {}
            for name, frame in original_cohorts.items():
                copy = frame.copy()
                copy["target"] = ccas_core_variant(copy["a1c"], copy["fpg"], slope, max_weight)
                cohorts[name] = copy
            predictions = refit_loco_fixed(
                cohorts, pair_groups, BASE_FEATURES, selected_by_holdout
            )
            metrics = evaluate_loco(cohorts, predictions, seed=SEED + int(slope * 100 + max_weight * 10))
            results[f"slope_{int(slope)}_max_{max_weight:.2f}"] = {
                "slope": slope,
                "max_weight": max_weight,
                "macro_mean_spearman": metrics["macro_mean_spearman"],
                "by_cohort_spearman": {
                    name: item["spearman"] for name, item in metrics["by_cohort"].items()
                },
            }
    return results


def pair_score_metrics(model, group):
    a, b = predict(model, group["a"]), predict(model, group["b"])
    return {
        "n": len(a),
        "spearman": safe_spearman(a, b),
        "icc_absolute_agreement": icc_absolute_agreement(a, b),
        "median_absolute_score_difference": float(np.median(np.abs(a - b))),
    }


def final_stability(feature_rows, pair_groups, model):
    sensor = {
        group["name"]: pair_score_metrics(model, group)
        for group in pair_groups if group["kind"] == "sensor"
    }
    time = {
        group["name"]: pair_score_metrics(model, group)
        for group in pair_groups if group["kind"] == "time"
    }
    cgm24 = cgmacros_sensor_average(feature_rows, 24)
    cgm48 = cgmacros_sensor_average(feature_rows, 48)
    merged = cgm24[["id", *BASE_FEATURES]].merge(
        cgm48[["id", *BASE_FEATURES]], on="id", suffixes=("_24", "_48"), validate="one_to_one"
    )
    a, b = pair_frames(merged, "24", "48", BASE_FEATURES)
    time["time_cgmacros_sensor_average"] = pair_score_metrics(model, {
        "a": a, "b": b
    })
    return {"sensor": sensor, "time": time}


def contribution_audit(model, cohorts):
    frame = pd.concat(list(cohorts.values()), ignore_index=True)
    z = model["transformer"].transform(frame)
    absolute = np.mean(np.abs(z * model["weights"]), axis=0)
    total = float(np.sum(absolute))
    shares = absolute / total if total > 1e-12 else np.zeros_like(absolute)
    return {
        "mean_absolute_logit_contribution": {
            feature: float(value) for feature, value in zip(model["features"], absolute)
        },
        "contribution_share": {
            feature: float(value) for feature, value in zip(model["features"], shares)
        },
        "maximum_share": float(np.max(shares)) if len(shares) else None,
        "dominant_feature": model["features"][int(np.argmax(shares))] if len(shares) else None,
    }


def coefficient_bootstrap(cohorts, pair_groups, selected):
    rng = np.random.default_rng(SEED + 9000)
    weights = []
    for _ in range(COEFFICIENT_BOOTSTRAPS):
        sampled = {
            name: frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)
            for name, frame in cohorts.items()
        }
        model = fit_bounded_model(
            sampled, BASE_FEATURES, selected["lambda"], selected["stability_weight"], pair_groups
        )
        weights.append(model["weights"])
    weights = np.asarray(weights)
    return {
        "replicates": len(weights),
        "positive_frequency": {
            feature: float(np.mean(weights[:, index] > 1e-8))
            for index, feature in enumerate(BASE_FEATURES)
        },
        "nonzero_frequency_gt_0_001": {
            feature: float(np.mean(weights[:, index] > 0.001))
            for index, feature in enumerate(BASE_FEATURES)
        },
        "median_weight": {
            feature: float(np.median(weights[:, index]))
            for index, feature in enumerate(BASE_FEATURES)
        },
        "weight_ci95": {
            feature: percentile_interval(weights[:, index])
            for index, feature in enumerate(BASE_FEATURES)
        },
    }


def hall_exploratory(cohorts, hall_raw, predictions):
    hall = cohorts["hall"].copy()
    hall["prediction"] = predictions["stability_base5"]["hall"]
    hall["night_prediction"] = predictions["night_mean"]["hall"]
    hall["anchor_prediction"] = predictions["anchor_only"]["hall"]

    full = hall[hall[["a1c", "fpg", "ogtt", "SSPG"]].notna().all(axis=1)].copy()
    full["ccas_full"] = ccas_full_variant(full["a1c"], full["fpg"], full["ogtt"], full["SSPG"])
    full_metrics = {
        "stability_base5": bootstrap_regression(
            full["ccas_full"], full["prediction"], full["night_prediction"], SEED + 10000
        ),
        "night_mean": regression_metrics(full["ccas_full"], full["night_prediction"]),
        "anchor_only": regression_metrics(full["ccas_full"], full["anchor_prediction"]),
    }

    surface = full[(full["a1c"] < 5.7) & (full["fpg"] < 100)].copy()
    hidden_y = ((surface["ogtt"] >= 140) | (surface["SSPG"] >= 150)).astype(int).to_numpy()
    hidden = {
        "n": len(surface),
        "positive": int(hidden_y.sum()),
        "stability_base5": bootstrap_auc(
            hidden_y, surface["prediction"], surface["night_prediction"], SEED + 10100
        ),
        "night_mean_auc": safe_auc(hidden_y, surface["night_prediction"]),
        "anchor_only_auc": safe_auc(hidden_y, surface["anchor_prediction"]),
        "exploratory_loso_operating_point": leave_one_out_threshold_metrics(
            hidden_y, surface["prediction"]
        ),
    }

    homa = hall[hall["homa_ir"].notna()].copy()
    cgm = cohorts["cgmacros"].copy()
    cgm["prediction"] = predictions["stability_base5"]["cgmacros"]
    homa_audit = {
        "cgmacros": {
            "n": len(cgm),
            "prediction_vs_log1p_homa_spearman": safe_spearman(
                cgm["prediction"], np.log1p(cgm["homa_ir"])
            ),
        },
        "hall": {
            "n": len(homa),
            "prediction_vs_log1p_homa_spearman": safe_spearman(
                homa["prediction"], np.log1p(homa["homa_ir"])
            ),
        },
    }

    full_sensitivity = {"glycemic_only": regression_metrics(
        ccas_full_variant(full["a1c"], full["fpg"], full["ogtt"], None), full["prediction"]
    )}
    for scale in (15.0, 25.0, 35.0):
        target = ccas_full_variant(
            full["a1c"], full["fpg"], full["ogtt"], full["SSPG"], sspg_scale=scale
        )
        full_sensitivity[f"sspg_scale_{int(scale)}"] = regression_metrics(target, full["prediction"])
    return {
        "ccas_full": full_metrics,
        "hidden_abnormal": hidden,
        "homa_ir": homa_audit,
        "full_target_sensitivity": full_sensitivity,
    }


def main():
    feature_rows, cohorts, hall_raw = prepare_clinical_cohorts()
    pair_groups = build_pair_groups(feature_rows)

    all_predictions, fits, evaluations = {}, {}, {}
    for index, (family, features) in enumerate(FAMILIES.items()):
        print(f"five-dimensional LOCO: {family}", flush=True)
        stability_grid = STABILITY_WEIGHTS if family == "stability_base5" else (0.0,)
        prediction, family_fits = fit_loco_family(
            cohorts, pair_groups, features, stability_grid, seed_offset=index * 10000
        )
        all_predictions[family] = prediction
        fits[family] = family_fits

    baseline_predictions = {
        "night_mean": all_predictions["night_mean"],
        "anchor_only": all_predictions["anchor_only"],
    }
    for index, family in enumerate(FAMILIES):
        baselines = baseline_predictions if family not in baseline_predictions else None
        evaluations[family] = evaluate_loco(
            cohorts, all_predictions[family], baselines, SEED + 20000 + index * 500
        )

    selected_by_holdout = {
        holdout: fits["stability_base5"][holdout]["selected"] for holdout in CLINICAL_COHORTS
    }
    deletion_results = {}
    for feature in BASE_FEATURES:
        subset = tuple(current for current in BASE_FEATURES if current != feature)
        deletion_prediction = refit_loco_fixed(cohorts, pair_groups, subset, selected_by_holdout)
        deletion_results[f"drop_{feature}"] = evaluate_loco(
            cohorts, deletion_prediction, seed=SEED + 30000 + len(deletion_results) * 100
        )

    sensitivity = target_sensitivity(cohorts, pair_groups, selected_by_holdout)

    final_selected, final_grid = select_parameters(
        cohorts, BASE_FEATURES, pair_groups, STABILITY_WEIGHTS, SEED + 40000
    )
    final_pairs = training_pair_groups(pair_groups, cohorts)
    final_model = fit_bounded_model(
        cohorts, BASE_FEATURES, final_selected["lambda"],
        final_selected["stability_weight"], final_pairs,
    )
    formula = formula_export(final_model)
    stability = final_stability(feature_rows, pair_groups, final_model)
    contribution = contribution_audit(final_model, cohorts)
    coefficient_stability = coefficient_bootstrap(cohorts, final_pairs, final_selected)
    exploratory = hall_exploratory(cohorts, hall_raw, all_predictions)

    active_features = [
        feature for feature, weight in formula["weights"].items() if weight > 0.001
    ]
    coefficient_gate = all(
        coefficient_stability["positive_frequency"][feature] >= 0.80
        for feature in active_features
    )
    main_time = {
        "cgmacros": stability["time"]["time_cgmacros_sensor_average"],
        "colas": stability["time"]["time_colas_default"],
        "hall": stability["time"]["time_hall_default"],
    }
    sensor48 = stability["sensor"]["sensor_cgmacros_w48"]
    primary_eval = evaluations["stability_base5"]
    gates = {
        "macro_delta_vs_night_mean_gte_0_05": primary_eval["delta_macro_spearman_vs_night_mean"] >= 0.05,
        "macro_delta_vs_night_mean_ci_lower_gt_0": primary_eval["delta_vs_night_mean_ci95"][0] > 0,
        "macro_delta_vs_anchor_only_gte_0_05": primary_eval["delta_macro_spearman_vs_anchor_only"] >= 0.05,
        "macro_delta_vs_anchor_only_ci_lower_gt_0": primary_eval["delta_vs_anchor_only_ci95"][0] > 0,
        "every_clinical_cohort_spearman_gte_0_20": all(
            item["spearman"] >= 0.20 for item in primary_eval["by_cohort"].values()
        ),
        "cgmacros_sensor48_spearman_gte_0_70": sensor48["spearman"] >= 0.70,
        "cgmacros_sensor48_icc_gte_0_70": sensor48["icc_absolute_agreement"] >= 0.70,
        "main_cohort_24_48_spearman_all_gte_0_70": all(
            item["spearman"] >= 0.70 for item in main_time.values()
        ),
        "maximum_feature_contribution_lte_0_70": contribution["maximum_share"] <= 0.70,
        "active_coefficient_positive_frequency_gte_0_80": coefficient_gate,
    }
    candidate_freeze_eligible = bool(all(gates.values()))

    scored = []
    for name, frame in cohorts.items():
        copy = frame.copy()
        copy["analysis_cohort"] = name
        copy["stability_base5_full_fit"] = predict(final_model, copy)
        for family in FAMILIES:
            copy[f"loco_{family}"] = all_predictions[family][name]
        scored.append(copy)
    pd.concat(scored, ignore_index=True, sort=False).to_csv(FEATURE_PATH, index=False)

    result = {
        "protocol": {
            "date": "2026-08-10",
            "primary_model": "stability_base5",
            "features": list(BASE_FEATURES),
            "equation_family": "100*sigmoid(nonnegative sparse linear predictor)",
            "clinical_target": "CCAS-core for three-cohort LOCO",
            "hall_status": "previously inspected exploratory cohort; not independent external validation",
            "unlabelled_stability_only": list(STRESS_COHORTS),
            "clinical_loss": "equal cohort soft-label logistic cross-entropy",
            "stability_loss": "equal pair-group mean squared prediction disagreement",
            "lambda_grid": list(LAMBDAS),
            "stability_weight_grid": list(STABILITY_WEIGHTS),
            "inner_folds": INNER_FOLDS,
            "comparators": list(FAMILIES),
        },
        "data": {
            "clinical": {name: len(frame) for name, frame in cohorts.items()},
            "pair_groups": {
                group["name"]: len(group["a"]) for group in pair_groups
            },
        },
        "loco_evaluation": evaluations,
        "loco_fits": fits,
        "feature_deletion": deletion_results,
        "ccas_core_sensitivity": sensitivity,
        "final_selection": {"selected": final_selected, "grid": final_grid},
        "final_formula": formula,
        "coefficient_stability": coefficient_stability,
        "contribution_audit": contribution,
        "device_and_window_stability": stability,
        "hall_exploratory": exploratory,
        "candidate_freeze_gates": gates,
        "candidate_freeze_eligible": candidate_freeze_eligible,
        "deployment_eligible": False,
        "html_decision": "retain index.html; no unseen clinical cohort remains for independent confirmation",
        "limitations": [
            "Hall influenced the research question in a prior cycle and cannot be called independent validation.",
            "Colas glycaemia is used as provided; fasting duration cannot be independently audited from source metadata.",
            "Stability-only T1D cohorts improve measurement regularization but provide no CCAS clinical labels.",
            "CCAS and the bounded CGM score are experimental research constructs, not diagnostic scales.",
            "Treatment, meals, sleep, activity, and device model are not uniformly available across cohorts.",
        ],
    }
    RESULT_PATH.write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_ready({
        "data": result["data"],
        "loco_evaluation": evaluations,
        "final_formula": formula,
        "contribution_audit": contribution,
        "device_and_window_stability": stability,
        "hall_exploratory": exploratory,
        "candidate_freeze_gates": gates,
        "candidate_freeze_eligible": candidate_freeze_eligible,
    }), indent=2, ensure_ascii=False))
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"wrote {FEATURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
