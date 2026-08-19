#!/usr/bin/env python3
"""Analysis for the BIG IDEAs Food_Log meal-perturbation deep-dive.

Tests whether w_carb (specific_work) / strain_per_carb / tau_relax -- already
graduation-criterion-1-satisfied candidates in candidate_tensor_staging_matrix.md
-- still show a directionally consistent relationship with HbA1c inside this
cohort's unusually narrow, all-non-diabetic HbA1c band (5.3-6.4%). This is a
sensitivity-floor probe, not a graduation attempt (criterion 1 already met by
CGMacros + Stanford OGTT-CGM, two protocol-independent sources).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

SUBJECT_METRICS = [
    "mean_strain_per_carb", "median_strain_per_carb",
    "mean_specific_work", "median_specific_work",
    "mean_tau_relax", "median_tau_relax",
    "mean_ascend_slope", "median_ascend_slope",
    "mean_delta_g", "median_delta_g",
]


def latest_result_file():
    files = sorted(glob.glob("reports/wind_tunnel_big_ideas_meal_dynamics_*.json"))
    if not files:
        raise FileNotFoundError("No BIG IDEAs meal-dynamics result file found.")
    return files[-1]


def main():
    result_file = latest_result_file()
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ok = [r for r in data["results"] if "error" not in r]
    print(f"Loaded {len(ok)} subjects with valid meal challenges from {result_file}.")
    for r in ok:
        print(f"  {r['id']}: n_meals={r['n_meals']}, HbA1c={r.get('a1c')}, "
              f"median_specific_work={r['median_specific_work']:.3f}, "
              f"median_strain_per_carb={r['median_strain_per_carb']:.4f}")

    hba1c = np.array([r.get("a1c") for r in ok], dtype=float)
    valid_hba1c = ~np.isnan(hba1c)
    n_missing = (~valid_hba1c).sum()
    if n_missing:
        print(f"WARNING: {n_missing} subjects missing HbA1c, excluded from correlation.")

    median_h = float(np.median(hba1c[valid_hba1c]))
    print(f"\nHbA1c median split at {median_h}% (n valid={valid_hba1c.sum()})")

    struct_out = {
        "result_file": result_file,
        "n_subjects_with_valid_meals": len(ok),
        "n_hba1c_valid": int(valid_hba1c.sum()),
        "hba1c_median_pct": median_h,
        "n_meals_total": int(sum(r["n_meals"] for r in ok)),
        "metrics": {},
    }

    high_mask = np.zeros(len(ok), dtype=bool)
    low_mask = np.zeros(len(ok), dtype=bool)
    high_mask[valid_hba1c] = hba1c[valid_hba1c] > median_h
    low_mask[valid_hba1c] = hba1c[valid_hba1c] <= median_h

    print(f"\n{'metric':<28}{'median_high':>14}{'median_low':>14}{'AUC':>8}{'MW_p':>10}{'rho':>10}{'sp_p':>10}")
    for m in SUBJECT_METRICS:
        vals = np.array([r.get(m) for r in ok], dtype=float)
        valid = ~np.isnan(vals) & valid_hba1c
        v_high = vals[high_mask & valid]
        v_low = vals[low_mask & valid]
        if len(v_high) < 3 or len(v_low) < 3:
            print(f"  {m}: insufficient valid n (high={len(v_high)}, low={len(v_low)}), skip.")
            continue

        u_stat, mw_p = stats.mannwhitneyu(v_high, v_low, alternative="two-sided")
        rho, sp_p = stats.spearmanr(vals[valid], hba1c[valid])
        auc = u_stat / (len(v_high) * len(v_low))

        print(f"  {m:<26}{np.median(v_high):>14.4f}{np.median(v_low):>14.4f}{auc:>8.4f}{mw_p:>10.4f}{rho:>10.4f}{sp_p:>10.4f}")

        struct_out["metrics"][m] = {
            "median_high": float(np.median(v_high)),
            "median_low": float(np.median(v_low)),
            "rank_sep_auc": float(auc),
            "mannwhitney_p": float(mw_p),
            "spearman_rho": float(rho),
            "spearman_p": float(sp_p),
            "n_high": int(len(v_high)),
            "n_low": int(len(v_low)),
        }

    out_path = Path("reports/analysis_big_ideas_meal_dynamics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct_out, f, indent=2)
    print(f"\nWrote structured analysis to {out_path}")


if __name__ == "__main__":
    main()
