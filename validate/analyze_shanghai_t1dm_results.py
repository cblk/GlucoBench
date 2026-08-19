#!/usr/bin/env python3
"""Within-cohort baseline analysis for the Shanghai_T1DM wind-tunnel run.

STRICT ISOLATION (AGENTS.md T1D strong-warning doctrine): this script NEVER
loads or compares against shanghai_t2dm_subjects.json / its wind-tunnel
results. All grouping and comparison below happens strictly WITHIN the
T1DM cohort (HbA1c severity split among T1DM patients themselves).

Two analyses:
  1. Primary (n=12, first-visit-only, cross-sectional): HbA1c median split
     + continuous Spearman correlation, mirroring the BIG IDEAs baseline
     scan design.
  2. Supplementary (descriptive only, n=2 patients): the two multi-visit
     patients' (1002, 1006) metric trajectories across their 3 admissions
     each, reported as raw numbers with NO statistical test (n=2 is far
     below any meaningful test's minimum), per Section 8.1 No Inference &
     No Fabrication -- honest description, not a disguised statistical claim.
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
    files = sorted(glob.glob("reports/wind_tunnel_shanghai_t1dm_night_taumax60_*.json"))
    if not files:
        raise FileNotFoundError("No Shanghai_T1DM wind-tunnel result file found.")
    return files[-1]


def main():
    result_file = latest_result_file()
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ok = [r for r in data["results"] if "error" not in r]
    print(f"Loaded {len(all_ok)} successful Shanghai_T1DM recordings from {result_file}.")

    # --- Primary: first-visit-only, n=12 ---
    first_visit = [r for r in all_ok if r.get("visit_index") == 0]
    print(f"Primary cross-sectional set (first visit only): n={len(first_visit)}")

    hba1c = np.array([r["hba1c_mmol_mol"] for r in first_visit], dtype=float)
    valid_hba1c = ~np.isnan(hba1c)
    n_missing = (~valid_hba1c).sum()
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(first_visit)} patients missing HbA1c -- excluded from grouping (honest drop, not imputed).")

    median_h = float(np.median(hba1c[valid_hba1c]))
    print(f"HbA1c median split at {median_h} mmol/mol (n valid={valid_hba1c.sum()})")

    struct_out = {
        "result_file": result_file,
        "isolation_note": "STRICT T1D-only cohort; never compared against shanghai_t2dm.",
        "n_first_visit": len(first_visit),
        "n_hba1c_valid": int(valid_hba1c.sum()),
        "hba1c_median_mmol_mol": median_h,
        "metrics": {},
        "multi_visit_descriptive": {},
    }

    fv_idx = np.arange(len(first_visit))
    high_mask = np.zeros(len(first_visit), dtype=bool)
    low_mask = np.zeros(len(first_visit), dtype=bool)
    high_mask[valid_hba1c] = hba1c[valid_hba1c] > median_h
    low_mask[valid_hba1c] = hba1c[valid_hba1c] <= median_h

    for m in METRICS:
        vals = np.array([r.get(m) for r in first_visit], dtype=float)
        valid = ~np.isnan(vals) & valid_hba1c
        v_high = vals[high_mask & valid]
        v_low = vals[low_mask & valid]
        if len(v_high) < 3 or len(v_low) < 3:
            print(f"  {m}: insufficient valid n (high={len(v_high)}, low={len(v_low)}), skip formal test.")
            struct_out["metrics"][m] = {"skipped": True, "n_high": int(len(v_high)), "n_low": int(len(v_low))}
            continue

        u_stat, mw_p = stats.mannwhitneyu(v_high, v_low, alternative="two-sided")
        rho, sp_p = stats.spearmanr(vals[valid], hba1c[valid])
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

    # --- Supplementary: multi-visit descriptive (n=2 patients, NO stats) ---
    print("\n--- Supplementary descriptive: multi-visit patients (n=2, no statistical test) ---")
    by_patient = {}
    for r in all_ok:
        by_patient.setdefault(r["patient_base_id"], []).append(r)

    for pid, recs in sorted(by_patient.items()):
        if len(recs) < 2:
            continue
        recs = sorted(recs, key=lambda r: r["visit_index"])
        traj = {m: [r.get(m) for r in recs] for m in METRICS}
        traj["hba1c_mmol_mol"] = [r.get("hba1c_mmol_mol") for r in recs]
        traj["admission_date"] = [r.get("admission_date") for r in recs]
        traj["hypoglycemic_agents"] = [r.get("hypoglycemic_agents") for r in recs]
        struct_out["multi_visit_descriptive"][pid] = traj
        print(f"  Patient {pid} ({len(recs)} visits): "
              f"workIntegral={traj['workIntegral']}, dim={traj['dim']}, "
              f"HbA1c={traj['hba1c_mmol_mol']}, dates={traj['admission_date']}")

    out_path = Path("reports/analysis_shanghai_t1dm_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct_out, f, indent=2)
    print(f"\nWrote structured analysis to {out_path}")


if __name__ == "__main__":
    main()
