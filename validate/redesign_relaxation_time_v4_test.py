"""
Wind-tunnel collision test for the exponential-decay-curve-fit redesign
(_relaxation_time_v4_expfit.py) of 候选 #5 `relaxationTime`. Tests TWO pre-declared honest
variants (no post-hoc tuning to maximize either cohort's score):
  - v4a: all valid-slope fits kept, no R^2 quality gate.
  - v4b: R^2 >= 0.5 quality gate applied (drops poorly-fit exponential events).
Compares both against the already-registered v1 (point-snap) baseline on the SAME two
cohorts/grouping methodology as all prior candidate #5 reports.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
from _relaxation_time_v4_expfit import compute_excursion_kinetics_expfit


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


def recompute_subject(subject, tau_max=120, r2_gate=None):
    sid = subject["id"]
    try:
        ts, raw_vs = wt.resample_raw(subject["timestamps"], subject["values"])
        valid_n = sum(1 for v in raw_vs if v is not None)
        if valid_n < 60:
            return None
        raw_json = json.dumps(raw_vs)
        tau_res = json.loads(wt.eng.engine.extract_tau(raw_json, max_lag=tau_max))
        if tau_res.get("result") is None:
            return None
        filt_res = json.loads(wt.eng.engine.filter_chunks(raw_json, 2, 0.08))
        smooth_vs = filt_res.get("result")
        if smooth_vs is None:
            return None
        kin = compute_excursion_kinetics_expfit(ts, smooth_vs, r2_gate=r2_gate)
        kin["id"] = sid
        return kin
    except Exception as e:
        print(f"  [WARN] {sid} recompute failed: {e}")
        return None


def evaluate(label, high_v, low_v):
    sep = rank_sep(high_v, low_v)
    if sep is None:
        print(f"  {label}: insufficient data")
        return None, None
    p = permutation_p(high_v, low_v, sep)
    print(f"  {label}: P={sep:.4f}, p={p:.4f}  (n_high={len(high_v)}, n_low={len(low_v)})")
    return sep, p


def run_stanford():
    with open("output/stanford_sspg_subjects.json", "r", encoding="utf-8") as f:
        subjects = json.load(f)["subjects"]

    rows_a, rows_b = [], []
    for s in subjects:
        ka = recompute_subject(s, r2_gate=None)
        kb = recompute_subject(s, r2_gate=0.5)
        if ka is None or kb is None:
            continue
        rows_a.append({"id": s["id"], "sspg_class": s["sspg_class"], "tau": ka["relaxationTime"],
                        "median_r2": ka["median_r2"], "n_events": ka["n_qualifying_excursions"]})
        rows_b.append({"id": s["id"], "sspg_class": s["sspg_class"], "tau": kb["relaxationTime"]})

    dfa, dfb = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    print(f"\n=== Stanford SSPG: n={len(dfa)}, median per-subject event-median R^2={dfa['median_r2'].median():.3f} ===")
    print("v4a (no R^2 gate):")
    evaluate("tau", dfa[dfa.sspg_class == "IR"]["tau"].dropna().tolist(), dfa[dfa.sspg_class == "IS"]["tau"].dropna().tolist())
    print("v4b (R^2>=0.5 gate):")
    evaluate("tau", dfb[dfb.sspg_class == "IR"]["tau"].dropna().tolist(), dfb[dfb.sspg_class == "IS"]["tau"].dropna().tolist())
    return dfa, dfb


def run_shanghai():
    with open("output/shanghai_t2dm_subjects.json", "r", encoding="utf-8") as f:
        subjects = json.load(f)["subjects"]
    with open("reports/wind_tunnel_shanghai_t2dm_legacymetrics_night_taumax120_20260819_1947.json",
              "r", encoding="utf-8") as f:
        orig = {r["id"]: r for r in json.load(f)["results"] if "error" not in r}

    rows_a, rows_b = [], []
    for s in subjects:
        orig_r = orig.get(s["id"])
        if orig_r is None:
            continue
        ka = recompute_subject(s, r2_gate=None)
        kb = recompute_subject(s, r2_gate=0.5)
        if ka is None or kb is None:
            continue
        rows_a.append({"id": s["id"], "patient_base_id": orig_r["patient_base_id"],
                        "admission_date": orig_r["admission_date"], "hba1c_mmol_mol": orig_r.get("hba1c_mmol_mol"),
                        "tau": ka["relaxationTime"], "median_r2": ka["median_r2"], "n_events": ka["n_qualifying_excursions"]})
        rows_b.append({"id": s["id"], "patient_base_id": orig_r["patient_base_id"],
                        "admission_date": orig_r["admission_date"], "hba1c_mmol_mol": orig_r.get("hba1c_mmol_mol"),
                        "tau": kb["relaxationTime"]})

    def dedupe_and_group(rows):
        df_all = pd.DataFrame(rows).sort_values("admission_date")
        df = df_all.groupby("patient_base_id", as_index=False).first()
        hba1c_valid = df.dropna(subset=["hba1c_mmol_mol"])
        median_hba1c = hba1c_valid["hba1c_mmol_mol"].median()
        df["hba1c_group"] = np.where(
            df["hba1c_mmol_mol"].isna(), None,
            np.where(df["hba1c_mmol_mol"] > median_hba1c, "high", "low"),
        )
        return df

    dfa, dfb = dedupe_and_group(rows_a), dedupe_and_group(rows_b)
    print(f"\n=== Shanghai T2DM: n={len(dfa)} independent patients, "
          f"median per-subject event-median R^2={dfa['median_r2'].median():.3f} ===")
    print("v4a (no R^2 gate):")
    evaluate("tau", dfa[dfa.hba1c_group == "high"]["tau"].dropna().tolist(), dfa[dfa.hba1c_group == "low"]["tau"].dropna().tolist())
    print("v4b (R^2>=0.5 gate):")
    evaluate("tau", dfb[dfb.hba1c_group == "high"]["tau"].dropna().tolist(), dfb[dfb.hba1c_group == "low"]["tau"].dropna().tolist())
    return dfa, dfb


if __name__ == "__main__":
    run_stanford()
    run_shanghai()
