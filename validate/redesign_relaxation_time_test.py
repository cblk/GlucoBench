"""
Wind-tunnel collision test (Section 4.B step 3) for the interpolated-crossing redesign of
候选 #5 `relaxationTime` (see _relaxation_time_v2.py for the Anomaly Targeting / High-
Dimensional Hypothesis writeup). Re-runs the SAME preprocessing (resample -> tau -> dim ->
filter_chunks) as run_subject_legacy() in _wind_tunnel_common.py, but swaps in
compute_excursion_kinetics_interp() instead of the original point-snapped version, on BOTH
already-tested cohorts (Stanford SSPG, Shanghai T2DM), using the EXACT same grouping
methodology as the original analyses for apples-to-apples comparison.

Section 9.1.2 Labels as Prisms: sspg_class / hba1c_mmol_mol used only for post-hoc group
rank-separation, never as fit targets. Section 9.5 Product Isolation: prints comparison to
stdout + writes one JSON to reports/; does not touch index_v4.html/_legacy_metrics_v4.py/
_wind_tunnel_common.py.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
from _relaxation_time_v2 import compute_excursion_kinetics_interp


def rank_sep(a_list, b_list):
    if not a_list or not b_list:
        return None
    wins = 0.0
    for a in a_list:
        for b in b_list:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(a_list) * len(b_list))


def permutation_p(high_vals, low_vals, observed_sep, n_perm=20000, seed=20260819):
    rng = random.Random(seed)
    pooled = high_vals + low_vals
    n_high = len(high_vals)
    observed_dev = abs(observed_sep - 0.5)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_high = pooled[:n_high]
        perm_low = pooled[n_high:]
        perm_sep = rank_sep(perm_high, perm_low)
        if abs(perm_sep - 0.5) >= observed_dev:
            extreme += 1
    return extreme / n_perm


def recompute_subject(subject, tau_max=120):
    sid = subject["id"]
    try:
        ts, raw_vs = wt.resample_raw(subject["timestamps"], subject["values"])
        valid_n = sum(1 for v in raw_vs if v is not None)
        if valid_n < 60:
            return None
        raw_json = json.dumps(raw_vs)
        tau_res = json.loads(wt.eng.engine.extract_tau(raw_json, max_lag=tau_max))
        tau = tau_res.get("result")
        if tau is None:
            return None
        filt_res = json.loads(wt.eng.engine.filter_chunks(raw_json, 2, 0.08))
        smooth_vs = filt_res.get("result")
        if smooth_vs is None:
            return None
        kin = compute_excursion_kinetics_interp(ts, smooth_vs)
        kin["id"] = sid
        return kin
    except Exception as e:
        print(f"  [WARN] {sid} recompute failed: {e}")
        return None


def run_stanford():
    with open("output/stanford_sspg_subjects.json", "r", encoding="utf-8") as f:
        subjects = json.load(f)["subjects"]
    with open("reports/wind_tunnel_stanford_sspg_legacymetrics_night_taumax120_20260819_1730.json",
              "r", encoding="utf-8") as f:
        orig = {r["id"]: r for r in json.load(f)["results"] if "error" not in r}

    rows = []
    for s in subjects:
        kin = recompute_subject(s)
        if kin is None:
            continue
        orig_r = orig.get(s["id"], {})
        rows.append({
            "id": s["id"], "sspg_class": s["sspg_class"],
            "relaxationTime_v2": kin["relaxationTime"], "relaxationTime_raw_v2": kin["relaxationTime_raw"],
            "relaxationTime_v3": kin["relaxationTime_decayed_only"],
            "n_qualifying": kin["n_qualifying_excursions"], "n_undecayed_fallback": kin["n_undecayed_fallback"],
            "relaxationTime_v1": orig_r.get("relaxationTime"),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== Stanford SSPG: recomputed {len(df)}/{len(subjects)} subjects ===")
    print(f"Undecayed fallback fired for {df['n_undecayed_fallback'].sum()} total excursions "
          f"across {(df['n_undecayed_fallback']>0).sum()} subjects.")

    is_v1 = df[df.sspg_class == "IS"]["relaxationTime_v1"].dropna().tolist()
    ir_v1 = df[df.sspg_class == "IR"]["relaxationTime_v1"].dropna().tolist()
    is_v2 = df[df.sspg_class == "IS"]["relaxationTime_v2"].dropna().tolist()
    ir_v2 = df[df.sspg_class == "IR"]["relaxationTime_v2"].dropna().tolist()
    is_v3 = df[df.sspg_class == "IS"]["relaxationTime_v3"].dropna().tolist()
    ir_v3 = df[df.sspg_class == "IR"]["relaxationTime_v3"].dropna().tolist()

    sep_v1 = rank_sep(ir_v1, is_v1)
    sep_v2 = rank_sep(ir_v2, is_v2)
    sep_v3 = rank_sep(ir_v3, is_v3)
    p_v1 = permutation_p(ir_v1, is_v1, sep_v1)
    p_v2 = permutation_p(ir_v2, is_v2, sep_v2)
    p_v3 = permutation_p(ir_v3, is_v3, sep_v3)
    print(f"  v1 (point-snap):         P(IR>IS)={sep_v1:.4f}, p={p_v1:.4f}")
    print(f"  v2 (interpolated):       P(IR>IS)={sep_v2:.4f}, p={p_v2:.4f}")
    print(f"  v3 (interp+decay-only):  P(IR>IS)={sep_v3:.4f}, p={p_v3:.4f}  "
          f"(n_IS={len(is_v3)}, n_IR={len(ir_v3)})")
    return df, {"sep_v1": sep_v1, "p_v1": p_v1, "sep_v2": sep_v2, "p_v2": p_v2, "sep_v3": sep_v3, "p_v3": p_v3}


def run_shanghai():
    with open("output/shanghai_t2dm_subjects.json", "r", encoding="utf-8") as f:
        subjects = json.load(f)["subjects"]
    with open("reports/wind_tunnel_shanghai_t2dm_legacymetrics_night_taumax120_20260819_1947.json",
              "r", encoding="utf-8") as f:
        orig_raw = json.load(f)["results"]
    orig = {r["id"]: r for r in orig_raw if "error" not in r}

    rows = []
    for s in subjects:
        kin = recompute_subject(s)
        orig_r = orig.get(s["id"])
        if kin is None or orig_r is None:
            continue
        rows.append({
            "id": s["id"], "patient_base_id": orig_r["patient_base_id"],
            "admission_date": orig_r["admission_date"], "hba1c_mmol_mol": orig_r.get("hba1c_mmol_mol"),
            "relaxationTime_v2": kin["relaxationTime"], "relaxationTime_raw_v2": kin["relaxationTime_raw"],
            "relaxationTime_v3": kin["relaxationTime_decayed_only"],
            "n_qualifying": kin["n_qualifying_excursions"], "n_undecayed_fallback": kin["n_undecayed_fallback"],
            "relaxationTime_v1": orig_r.get("relaxationTime"),
        })
    df_all = pd.DataFrame(rows).sort_values("admission_date")
    df = df_all.groupby("patient_base_id", as_index=False).first()
    print(f"\n=== Shanghai T2DM: recomputed {len(df_all)} recordings -> {len(df)} independent patients ===")
    print(f"Undecayed fallback fired for {df['n_undecayed_fallback'].sum()} total excursions "
          f"across {(df['n_undecayed_fallback']>0).sum()} subjects.")

    hba1c_valid = df.dropna(subset=["hba1c_mmol_mol"])
    median_hba1c = hba1c_valid["hba1c_mmol_mol"].median()
    df["hba1c_group"] = np.where(
        df["hba1c_mmol_mol"].isna(), None,
        np.where(df["hba1c_mmol_mol"] > median_hba1c, "high", "low"),
    )
    high_v1 = df[df.hba1c_group == "high"]["relaxationTime_v1"].dropna().tolist()
    low_v1 = df[df.hba1c_group == "low"]["relaxationTime_v1"].dropna().tolist()
    high_v2 = df[df.hba1c_group == "high"]["relaxationTime_v2"].dropna().tolist()
    low_v2 = df[df.hba1c_group == "low"]["relaxationTime_v2"].dropna().tolist()
    high_v3 = df[df.hba1c_group == "high"]["relaxationTime_v3"].dropna().tolist()
    low_v3 = df[df.hba1c_group == "low"]["relaxationTime_v3"].dropna().tolist()

    sep_v1 = rank_sep(high_v1, low_v1)
    sep_v2 = rank_sep(high_v2, low_v2)
    sep_v3 = rank_sep(high_v3, low_v3)
    p_v1 = permutation_p(high_v1, low_v1, sep_v1)
    p_v2 = permutation_p(high_v2, low_v2, sep_v2)
    p_v3 = permutation_p(high_v3, low_v3, sep_v3)
    print(f"  v1 (point-snap):         P(high>low)={sep_v1:.4f}, p={p_v1:.4f}  (n_high={len(high_v1)}, n_low={len(low_v1)})")
    print(f"  v2 (interpolated):       P(high>low)={sep_v2:.4f}, p={p_v2:.4f}  (n_high={len(high_v2)}, n_low={len(low_v2)})")
    print(f"  v3 (interp+decay-only):  P(high>low)={sep_v3:.4f}, p={p_v3:.4f}  (n_high={len(high_v3)}, n_low={len(low_v3)})")
    return df, {"sep_v1": sep_v1, "p_v1": p_v1, "sep_v2": sep_v2, "p_v2": p_v2, "sep_v3": sep_v3, "p_v3": p_v3}


if __name__ == "__main__":
    _, stanford_stats = run_stanford()
    _, shanghai_stats = run_shanghai()

    out_path = Path("reports/redesign_relaxation_time_interp_comparison_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stanford_sspg": stanford_stats, "shanghai_t2dm": shanghai_stats}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote comparison summary to {out_path}")
