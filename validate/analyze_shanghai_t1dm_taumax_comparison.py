#!/usr/bin/env python3
"""Compare ShanghaiT1DM's production (max_lag=60) vs supplementary
(max_lag=120, corrected) full-pipeline results, per subject and per the same
within-cohort HbA1c grouping used in the original baseline report. Answers:
does correcting the tau-truncation change the HbA1c-group-separation
conclusion (the "reversed direction, Fail-Closed" finding)?
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

METRICS = ["workIntegral", "dim", "det", "entr", "rr", "tau"]


def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f), files[-1]


def main():
    prod, prod_file = load_latest("reports/wind_tunnel_shanghai_t1dm_night_taumax60_*.json")
    supp, supp_file = load_latest("reports/wind_tunnel_shanghai_t1dm_night_taumax120_SUPPLEMENTARY_*.json")
    print(f"Production (max_lag=60): {prod_file}")
    print(f"Supplementary (max_lag=120, corrected): {supp_file}")

    prod_by_id = {r["id"]: r for r in prod["results"] if "error" not in r}
    supp_by_id = {r["id"]: r for r in supp["results"] if "error" not in r}
    common_ids = sorted(set(prod_by_id) & set(supp_by_id))
    print(f"\nCommon successful subjects: {len(common_ids)}")

    print(f"\n{'id':<20}{'tau_60':>8}{'tau_120':>9}{'WI_60':>10}{'WI_120':>10}{'dim_60':>8}{'dim_120':>9}")
    for sid in common_ids:
        p, s = prod_by_id[sid], supp_by_id[sid]
        print(f"{sid:<20}{p['tau']:>8}{s['tau']:>9}{p['workIntegral']:>10.2f}{s['workIntegral']:>10.2f}{p['dim']:>8}{s['dim']:>9}")

    # Per-metric paired shift (all 16 subjects, all first-visit or repeat -- same subject, two tau windows)
    print("\n--- Per-metric mean shift (production -> corrected), all common subjects ---")
    for m in METRICS:
        p_vals = np.array([prod_by_id[sid].get(m) for sid in common_ids], dtype=float)
        s_vals = np.array([supp_by_id[sid].get(m) for sid in common_ids], dtype=float)
        valid = ~np.isnan(p_vals) & ~np.isnan(s_vals)
        if valid.sum() == 0:
            continue
        print(f"  {m:<14}: prod_mean={np.nanmean(p_vals[valid]):.4f}, corrected_mean={np.nanmean(s_vals[valid]):.4f}, "
              f"mean_delta={np.nanmean(s_vals[valid]-p_vals[valid]):+.4f}")

    # Re-run the SAME first-visit HbA1c median-split analysis as the original baseline report,
    # but on the corrected (max_lag=120) results.
    first_visit = [supp_by_id[sid] for sid in common_ids if supp_by_id[sid].get("visit_index") == 0]
    print(f"\n--- Corrected (max_lag=120) HbA1c median-split re-analysis, first-visit only, n={len(first_visit)} ---")
    hba1c = np.array([r["hba1c_mmol_mol"] for r in first_visit], dtype=float)
    valid_hba1c = ~np.isnan(hba1c)
    median_h = float(np.median(hba1c[valid_hba1c]))
    print(f"HbA1c median split at {median_h} mmol/mol (n valid={valid_hba1c.sum()})")

    high_mask = np.zeros(len(first_visit), dtype=bool)
    low_mask = np.zeros(len(first_visit), dtype=bool)
    high_mask[valid_hba1c] = hba1c[valid_hba1c] > median_h
    low_mask[valid_hba1c] = hba1c[valid_hba1c] <= median_h

    struct_out = {
        "production_file": prod_file, "supplementary_file": supp_file,
        "n_first_visit": len(first_visit), "hba1c_median_mmol_mol": median_h,
        "metrics_corrected": {},
    }

    print(f"\n{'metric':<14}{'median_high':>14}{'median_low':>14}{'AUC':>8}{'MW_p':>10}{'rho':>10}{'sp_p':>10}")
    for m in METRICS:
        vals = np.array([r.get(m) for r in first_visit], dtype=float)
        valid = ~np.isnan(vals) & valid_hba1c
        v_high = vals[high_mask & valid]
        v_low = vals[low_mask & valid]
        if len(v_high) < 3 or len(v_low) < 3:
            print(f"  {m}: insufficient valid n (high={len(v_high)}, low={len(v_low)}), skip.")
            struct_out["metrics_corrected"][m] = {"skipped": True}
            continue
        u_stat, mw_p = stats.mannwhitneyu(v_high, v_low, alternative="two-sided")
        rho, sp_p = stats.spearmanr(vals[valid], hba1c[valid])
        auc = u_stat / (len(v_high) * len(v_low))
        print(f"  {m:<14}{np.median(v_high):>14.4f}{np.median(v_low):>14.4f}{auc:>8.4f}{mw_p:>10.4f}{rho:>10.4f}{sp_p:>10.4f}")
        struct_out["metrics_corrected"][m] = {
            "median_high": float(np.median(v_high)), "median_low": float(np.median(v_low)),
            "rank_sep_auc": float(auc), "mannwhitney_p": float(mw_p),
            "spearman_rho": float(rho), "spearman_p": float(sp_p),
            "n_high": int(len(v_high)), "n_low": int(len(v_low)),
        }

    out_path = Path("reports/analysis_shanghai_t1dm_taumax_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct_out, f, indent=2)
    print(f"\nWrote structured comparison to {out_path}")


if __name__ == "__main__":
    main()
