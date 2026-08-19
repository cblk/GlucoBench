#!/usr/bin/env python3
"""Baseline analysis for the BIG IDEAs wind-tunnel run ("跑一遍数据集" pass).

n=16, HbA1c band 5.3-6.4% (narrow, no participant in the diabetic range).
Since there is no natural binary clinical split (no diabetes diagnosis in
this cohort), the "prism" used here is a simple median split on HbA1c --
purely for cross-sectional group-separation description, never fed back
into any operator (Section 9.1.2 Labels as Prisms, Not Targets). Reported
alongside continuous Spearman correlation for the same reason mcPHASES/
Shanghai reports do: a rank-based view is more honest than a p-value when
n=16 and the label variance is this compressed.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

METRICS = ["workIntegral", "dim", "det", "entr", "rr", "tau"]


def latest_result_file():
    files = sorted(glob.glob("reports/wind_tunnel_big_ideas_night_taumax60_*.json"))
    if not files:
        raise FileNotFoundError("No BIG IDEAs wind-tunnel result file found.")
    return files[-1]


def main():
    result_file = latest_result_file()
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ok = [r for r in data["results"] if "error" not in r]
    print(f"Loaded {len(ok)} successful BIG IDEAs subjects from {result_file}.")

    hba1c = np.array([r["hba1c_pct"] for r in ok], dtype=float)
    median_h = float(np.median(hba1c))
    high_mask = hba1c > median_h
    low_mask = hba1c <= median_h
    print(f"HbA1c median split at {median_h}: n_high={high_mask.sum()}, n_low={low_mask.sum()}")
    print(f"HbA1c range: [{hba1c.min()}, {hba1c.max()}]")

    struct_out = {"result_file": result_file, "n": len(ok), "hba1c_median": median_h, "metrics": {}}

    for m in METRICS:
        vals = np.array([r.get(m) for r in ok], dtype=float)
        valid = ~np.isnan(vals)
        v_high = vals[high_mask & valid]
        v_low = vals[low_mask & valid]
        if len(v_high) < 3 or len(v_low) < 3:
            print(f"  {m}: insufficient valid n (high={len(v_high)}, low={len(v_low)}), skip.")
            continue

        u_stat, mw_p = stats.mannwhitneyu(v_high, v_low, alternative="two-sided")
        rho, sp_p = stats.spearmanr(vals[valid], hba1c[valid])

        # AUC-style rank separation (P(high > low), matching prior reports' convention)
        auc = u_stat / (len(v_high) * len(v_low))

        print(f"  {m}: median_high={np.median(v_high):.4f}, median_low={np.median(v_low):.4f}, "
              f"rank_sep(AUC)={auc:.4f}, MW_p={mw_p:.4f}, spearman_rho={rho:.4f}, spearman_p={sp_p:.4f}")

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

    out_path = Path("reports/analysis_big_ideas_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct_out, f, indent=2)
    print(f"\nWrote structured analysis to {out_path}")


if __name__ == "__main__":
    main()
