#!/usr/bin/env python3
"""Validate frozen five- and nine-dimensional scores on new public cohorts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PRIMITIVE_PATH = OUTPUT / "external_base5_primitives.csv"
V87_PATH = OUTPUT / "external_v87_metrics.json"
STANFORD_PATH = OUTPUT / "external_stanford_intervention_features.csv"
BASE5_MODEL_PATH = OUTPUT / "stability_base5_results.json"
FULL9_MODEL_PATH = OUTPUT / "clinical_continuum_results.json"
RESULT_PATH = OUTPUT / "external_base5_validation_results.json"
SCORE_PATH = OUTPUT / "external_base5_validation_scores.csv"

SEED = 20260811
BOOTSTRAPS = 3000
PERMUTATIONS = 3000


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def rankdata(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def percentile_interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def correlation_test(frame, endpoint, score, baseline=None, seed=SEED):
    selected = frame[[endpoint, score] + ([baseline] if baseline else [])].replace([np.inf, -np.inf], np.nan).dropna()
    x = selected[score].to_numpy(float)
    y = selected[endpoint].to_numpy(float)
    reference = selected[baseline].to_numpy(float) if baseline else None
    observed = spearman(x, y)
    observed_reference = spearman(reference, y) if baseline else np.nan
    rng = np.random.default_rng(seed)
    boot = []
    deltas = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(selected), len(selected))
        rho = spearman(x[indices], y[indices])
        boot.append(rho)
        if baseline:
            deltas.append(rho - spearman(reference[indices], y[indices]))
    extreme = 0
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    for _ in range(PERMUTATIONS):
        permuted_ranks = rng.permutation(ranked_y)
        permuted_rho = float(np.corrcoef(ranked_x, permuted_ranks)[0, 1])
        if permuted_rho >= observed - 1e-15:
            extreme += 1
    result = {
        "n": int(len(selected)),
        "spearman": observed,
        "spearman_ci95": percentile_interval(boot),
        "one_sided_permutation_p": (extreme + 1) / (PERMUTATIONS + 1),
    }
    if baseline:
        result.update({
            "baseline": baseline,
            "baseline_spearman": observed_reference,
            "delta_spearman_vs_baseline": observed - observed_reference,
            "delta_spearman_ci95": percentile_interval(deltas),
        })
    return result


def transform_column(values, transform):
    values = np.asarray(values, float)
    if transform.startswith("log1p"):
        return np.log1p(np.maximum(values, 0.0))
    return values


def apply_formula(frame, formula, bounded):
    z = np.full(len(frame), float(formula["intercept"]), float)
    contributions = {}
    for feature in formula["features"]:
        values = transform_column(frame[feature], formula["transform"][feature])
        reference = formula["standardization"][feature]
        standardized = (values - float(reference["center"])) / float(reference["scale"])
        contribution = float(formula["weights"][feature]) * standardized
        contributions[feature] = contribution
        z += contribution
    if bounded:
        z = np.clip(z, -40.0, 40.0)
        prediction = 100.0 / (1.0 + np.exp(-z))
    else:
        prediction = 100.0 * z
    return prediction, contributions


def contribution_audit(contributions):
    average = {key: float(np.nanmean(np.abs(value))) for key, value in contributions.items()}
    total = sum(average.values())
    shares = {key: value / total if total > 0 else np.nan for key, value in average.items()}
    return {
        "mean_absolute_logit_contribution": average,
        "contribution_share": shares,
        "maximum_share": max(shares.values()) if shares else np.nan,
        "dominant_feature": max(shares, key=shares.get) if shares else None,
    }


def feature_rank_correlations(frame, endpoint, features):
    output = {}
    for feature in features:
        selected = frame[[endpoint, feature]].replace([np.inf, -np.inf], np.nan).dropna()
        output[feature] = {"n": int(len(selected)), "spearman": spearman(selected[feature], selected[endpoint])}
    return output


def load_frames():
    frame = pd.read_csv(PRIMITIVE_PATH)
    frame["id"] = frame["id"].astype(str)
    frame = frame[frame["eligible"].astype(bool)].copy()
    v87 = pd.DataFrame(json.loads(V87_PATH.read_text(encoding="utf-8")))
    v87["id"] = v87["id"].astype(str)
    v87 = v87.rename(columns={"nightMean": "v87NightMean"})
    frame = frame.merge(
        v87[["cohort", "id", "v87Risk", "v87Mode", "v87UsedDynamic", "v87NightMean", "workIntegral", "ascendFriction", "nightFriction"]],
        on=["cohort", "id"], how="left",
    )
    dynamic_frames = []
    for cohort in ("kobe_shift0_w48", "shanghai_t2dm_w48", "big_ideas_w48"):
        path = OUTPUT / f"external_full9_{cohort}.json"
        dynamic = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        dynamic["id"] = dynamic["id"].astype(str)
        keep = dynamic[["id", "volume", "lyapunov", "det", "entr"]].copy()
        keep["cohort"] = cohort
        dynamic_frames.append(keep)
    frame = frame.merge(pd.concat(dynamic_frames, ignore_index=True), on=["cohort", "id"], how="left")
    return frame


def primary_shanghai(frame):
    shanghai = frame[frame["cohort"].eq("shanghai_t2dm_w48")].copy()
    shanghai["visit_index"] = pd.to_numeric(shanghai["visit_index"], errors="coerce").fillna(0)
    return shanghai.sort_values(["base_subject_id", "visit_index", "id"]).drop_duplicates("base_subject_id", keep="first")


def repeat_visit_stability(frame, score):
    shanghai = frame[frame["cohort"].eq("shanghai_t2dm_w48")].copy()
    shanghai["visit_index"] = pd.to_numeric(shanghai["visit_index"], errors="coerce").fillna(0)
    shanghai = shanghai.sort_values(["base_subject_id", "visit_index", "id"])
    pairs = []
    for subject_id, group in shanghai.groupby("base_subject_id"):
        if len(group) >= 2:
            pairs.append({
                "base_subject_id": subject_id,
                "first": float(group.iloc[0][score]),
                "second": float(group.iloc[1][score]),
            })
    pairs = pd.DataFrame(pairs)
    if pairs.empty:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(len(pairs)),
        "spearman": spearman(pairs["first"], pairs["second"]),
        "median_absolute_difference": float(np.median(np.abs(pairs["first"] - pairs["second"]))),
    }


def kobe_phase_stability(frame, score):
    subset = frame[frame["cohort"].str.startswith("kobe_shift")].copy()
    pivot = subset.pivot(index="base_subject_id", columns="phase_shift_hours", values=score)
    columns = sorted(pivot.columns)
    rho_vs_shift0 = {str(int(column)): spearman(pivot[0], pivot[column]) for column in columns}
    within_range = pivot.max(axis=1) - pivot.min(axis=1)
    return {
        "n": int(len(pivot)),
        "rho_vs_shift0": rho_vs_shift0,
        "median_within_subject_range": float(np.median(within_range)),
        "p95_within_subject_range": float(np.percentile(within_range, 95)),
    }


def stanford_analysis():
    frame = pd.read_csv(STANFORD_PATH)
    frame["negative_recovery_fraction"] = -pd.to_numeric(frame["recovery_fraction"], errors="coerce")
    candidates = [
        "peak_delta_mgdl", "time_to_peak_min", "iauc_mgdl_min",
        "residual_180_mgdl", "negative_recovery_fraction",
    ]
    glycemic = {
        feature: correlation_test(frame, "ccas_full", feature, seed=SEED + 500 + index)
        for index, feature in enumerate(candidates)
    }
    direct = frame[frame["SSPG"].notna() & frame["DI"].notna()].copy()
    direct["negative_di"] = -pd.to_numeric(direct["DI"], errors="coerce")
    mechanism = {}
    for feature in candidates:
        mechanism[feature] = {
            "n": int(len(direct)),
            "spearman_vs_sspg": spearman(direct[feature], direct["SSPG"]),
            "spearman_vs_negative_di": spearman(direct[feature], direct["negative_di"]),
        }
    return {
        "glycemic_transfer_29_subjects": glycemic,
        "direct_mechanism_overlap_5_subjects_descriptive_only": mechanism,
        "warning": "CCAS-full includes OGTT_2h, so these short-curve correlations are not independent mechanism validation.",
    }


def main():
    frame = load_frames()
    base5_formula = json.loads(BASE5_MODEL_PATH.read_text(encoding="utf-8"))["final_formula"]
    full9_formula = json.loads(FULL9_MODEL_PATH.read_text(encoding="utf-8"))["primary_formula"]

    frame["frozen_base5"], base5_contributions = apply_formula(frame, base5_formula, bounded=True)
    for feature, values in base5_contributions.items():
        frame[f"base5_contribution_{feature}"] = values
    frame["frozen_full9"] = np.nan
    full9_ready = frame[list(full9_formula["features"])].notna().all(axis=1)
    full9_prediction, _ = apply_formula(frame.loc[full9_ready], full9_formula, bounded=False)
    frame.loc[full9_ready, "frozen_full9"] = full9_prediction

    kobe_results = {}
    for index, shift in enumerate((0, 6, 12, 18)):
        cohort = f"kobe_shift{shift}_w48"
        subset = frame[frame["cohort"].eq(cohort)].copy()
        subset["clamp_di_abnormality"] = -pd.to_numeric(subset["clamp_di"], errors="coerce")
        subset["oral_di_abnormality"] = -pd.to_numeric(subset["oral_di"], errors="coerce")
        metrics = {
            "frozen_base5_vs_clamp_di_abnormality": correlation_test(
                subset, "clamp_di_abnormality", "frozen_base5", "night_mean", SEED + index * 20
            ),
            "v87_vs_clamp_di_abnormality": correlation_test(
                subset, "clamp_di_abnormality", "v87Risk", "v87NightMean", SEED + index * 20 + 1
            ),
            "frozen_base5_vs_oral_di_abnormality": correlation_test(
                subset, "oral_di_abnormality", "frozen_base5", "night_mean", SEED + index * 20 + 2
            ),
        }
        if shift == 0:
            metrics["frozen_full9_vs_clamp_di_abnormality"] = correlation_test(
                subset, "clamp_di_abnormality", "frozen_full9", "night_mean", SEED + 3
            )
        kobe_results[str(shift)] = metrics

    shanghai = primary_shanghai(frame)
    shanghai_results = {
        "primary_earliest_record": {
            "n_subjects": int(shanghai["base_subject_id"].nunique()),
            "frozen_base5_vs_ccas_core": correlation_test(
                shanghai, "ccas_core", "frozen_base5", "night_mean", SEED + 100
            ),
            "v87_treated_vs_ccas_core": correlation_test(
                shanghai, "ccas_core", "v87Risk", "v87NightMean", SEED + 101
            ),
            "frozen_full9_vs_ccas_core": correlation_test(
                shanghai, "ccas_core", "frozen_full9", "night_mean", SEED + 102
            ),
        },
        "repeat_visit_stability": {
            "frozen_base5": repeat_visit_stability(frame, "frozen_base5"),
            "v87_treated": repeat_visit_stability(frame, "v87Risk"),
        },
    }
    no_insulin = shanghai[(~shanghai["uses_insulin_agent"].astype(bool)) & shanghai["homa_ir_exploratory"].notna()].copy()
    if len(no_insulin) >= 3:
        shanghai_results["non_insulin_agent_homa_ir_exploratory"] = {
            "frozen_base5": correlation_test(no_insulin, "homa_ir_exploratory", "frozen_base5", "night_mean", SEED + 103),
            "v87_treated": correlation_test(no_insulin, "homa_ir_exploratory", "v87Risk", "v87NightMean", SEED + 104),
            "unit_note": "fasting insulin pmol/L divided by 6 before HOMA-IR; assay-dependent exploratory conversion",
        }

    big = frame[frame["cohort"].eq("big_ideas_w48")].copy()
    big_results = {
        "n": int(len(big)),
        "frozen_base5_vs_a1c_component": correlation_test(
            big, "a1c_component", "frozen_base5", "night_mean", SEED + 200
        ),
        "v87_unknown_fallback_vs_a1c_component": correlation_test(
            big, "a1c_component", "v87Risk", "v87NightMean", SEED + 201
        ),
        "frozen_full9_vs_a1c_component": correlation_test(
            big, "a1c_component", "frozen_full9", "night_mean", SEED + 202
        ),
    }

    kobe_primary = kobe_results["0"]["frozen_base5_vs_clamp_di_abnormality"]
    phase_rhos = [
        kobe_results[str(shift)]["frozen_base5_vs_clamp_di_abnormality"]["spearman"]
        for shift in (0, 6, 12, 18)
    ]
    kobe_shift0 = frame[frame["cohort"].eq("kobe_shift0_w48")]
    kobe_contrib = contribution_audit({
        feature: values[kobe_shift0.index.to_numpy()]
        for feature, values in base5_contributions.items()
    })
    kobe_shift0_features = kobe_shift0.copy()
    kobe_shift0_features["clamp_di_abnormality"] = -pd.to_numeric(kobe_shift0_features["clamp_di"], errors="coerce")
    external_feature_audit = {
        "kobe_shift0_vs_clamp_di_abnormality": feature_rank_correlations(
            kobe_shift0_features, "clamp_di_abnormality", base5_formula["features"]
        ),
        "shanghai_earliest_vs_ccas_core": feature_rank_correlations(
            shanghai, "ccas_core", base5_formula["features"]
        ),
        "big_ideas_vs_a1c_component": feature_rank_correlations(
            big, "a1c_component", base5_formula["features"]
        ),
    }
    contribution_by_cohort = {}
    for name, subset in {
        "kobe_shift0_w48": kobe_shift0,
        "shanghai_primary_earliest": shanghai,
        "big_ideas_w48": big,
    }.items():
        contribution_by_cohort[name] = contribution_audit({
            feature: subset[f"base5_contribution_{feature}"].to_numpy(float)
            for feature in base5_formula["features"]
        })
    gates = {
        "kobe_clamp_spearman_gte_0_35": bool(kobe_primary["spearman"] >= 0.35),
        "kobe_clamp_ci_lower_gt_0": bool(kobe_primary["spearman_ci95"][0] > 0),
        "kobe_delta_vs_night_mean_gte_0_05": bool(kobe_primary["delta_spearman_vs_baseline"] >= 0.05),
        "kobe_delta_ci_lower_gt_0": bool(kobe_primary["delta_spearman_ci95"][0] > 0),
        "kobe_all_phase_directions_positive": bool(min(phase_rhos) > 0),
        "kobe_worst_phase_spearman_gte_0_20": bool(min(phase_rhos) >= 0.20),
        "external_max_feature_contribution_lte_0_70": bool(kobe_contrib["maximum_share"] <= 0.70),
        "explicit_participant_key_available": False,
        "absolute_clock_origin_documented": False,
    }
    quantitative = [value for key, value in gates.items() if key not in {"explicit_participant_key_available", "absolute_clock_origin_documented"}]
    candidate_freeze_eligible = bool(all(quantitative) and gates["explicit_participant_key_available"] and gates["absolute_clock_origin_documented"])

    result = {
        "protocol": {
            "date": "2026-08-11",
            "primary_candidate": "previously fitted frozen bounded_base5",
            "primary_external_endpoint": "Kobe negative clamp disposition index",
            "comparators": ["night_mean", "current v8.7 treatment path", "historical frozen full9"],
            "window_hours": 48,
            "bootstrap_replicates": BOOTSTRAPS,
            "permutation_replicates": PERMUTATIONS,
            "no_refitting_on_external_cohorts": True,
        },
        "kobe_direct_mechanism": kobe_results,
        "kobe_clock_phase_stability": {
            "frozen_base5": kobe_phase_stability(frame, "frozen_base5"),
            "v87": kobe_phase_stability(frame, "v87Risk"),
            "night_mean": kobe_phase_stability(frame, "night_mean"),
        },
        "kobe_frozen_base5_contribution_audit": kobe_contrib,
        "external_feature_rank_audit": external_feature_audit,
        "frozen_base5_contribution_by_cohort": contribution_by_cohort,
        "shanghai_treated_stress": shanghai_results,
        "big_ideas_a1c_only_transfer": big_results,
        "stanford_intervention_channel": stanford_analysis(),
        "deployment_gates": gates,
        "candidate_freeze_eligible": candidate_freeze_eligible,
        "deployment_eligible": False,
        "html_decision": "Keep index.html unchanged; external evidence did not clear linkage, clock, and quantitative gates.",
        "limitations": [
            "Kobe Fig.1d linkage is strongly auditable by retained row order and two paper examples but lacks an explicit participant-ID column.",
            "Kobe minute zero has no documented wall-clock timestamp; four clock phases are sensitivity analyses.",
            "Shanghai is diagnosed, treated T2DM with repeat visits and laboratory timing/treatment confounding.",
            "BIG IDEAs has HbA1c only and one of 16 subjects failed the frozen 48-hour qualification gate.",
            "Stanford home CGM consists of short OGTT windows; only five participants overlap direct SSPG/DI phenotyping.",
            "CCAS and both CGM coordinates are research constructs, not diagnostic probabilities.",
        ],
    }
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCORE_PATH, index=False)
    RESULT_PATH.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    print(f"wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"wrote {SCORE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
