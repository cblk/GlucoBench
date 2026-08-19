"""
Apples-to-apples re-run of analyze_shanghai_results.py's EXACT duration-
stratified rank-separation methodology, at BOTH tau_max=60 (production
default) and tau_max=120 (candidate), to check whether the original
"short-cycle collapse" headline finding (short(<7d) P(high>low)=0.7899 vs
long(>=10d) P(high>low)=0.6716, delta=-0.1183, i.e. direction OPPOSITE to
the collapse hypothesis) survives at the wider tau window.

WHY THIS SCRIPT EXISTS (correcting a methodological artifact in the
2026-08-19 15:20 fleet comparison): compare_taumax60_vs_120_fleet.py used a
DIFFERENT, simpler grouping (median-split of duration_days itself, comparing
mean Work Integral across the resulting two duration groups) than what the
original duration-stratified report actually tested (HbA1c high/low
rank-separation P(high>low), computed SEPARATELY within short/mid/long
duration buckets). Worse, the fleet script's median split was computed
INDEPENDENTLY on each tau setting's surviving subject subset (n=109 at
tau=60 vs n=104 at tau=120), so part of its reported "86% gap shrinkage"
was an artifact of the two tau settings literally comparing different sets
of patients to different sets of patients, not the same comparison at two
tau values. This script fixes both problems: it reproduces the ORIGINAL
metric (HbA1c rank-separation within duration buckets, one FIXED HbA1c
median-split threshold per tau setting, first-visit dedup identical to the
original script) and reports both tau settings side by side.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_dedupe(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data["results"]
    df_all = pd.DataFrame(raw)
    df_all = df_all.sort_values("admission_date")
    df = df_all.groupby("patient_base_id", as_index=False).first()
    return df, len(df_all) - len(df)


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


def run_one(path, label):
    df, n_excluded = load_and_dedupe(path)
    print(f"\n[{label}] file={path}")
    print(f"[{label}] {len(df)} independent patients ({n_excluded} repeat-visit recordings excluded).")

    df["duration_bucket"] = pd.cut(
        df["duration_days"], bins=[0, 7, 10, 100],
        labels=["short(<7d)", "mid(7-10d)", "long(>=10d)"],
    )
    hba1c_valid = df.dropna(subset=["hba1c_mmol_mol"])
    median_hba1c = hba1c_valid["hba1c_mmol_mol"].median()
    print(f"[{label}] Fixed HbA1c median-split threshold = {median_hba1c:.2f} mmol/mol "
          f"(n={len(hba1c_valid)} with HbA1c available).")
    df["hba1c_group"] = np.where(
        df["hba1c_mmol_mol"].isna(), None,
        np.where(df["hba1c_mmol_mol"] > median_hba1c, "high", "low"),
    )

    rows = {}
    for bucket in ("short(<7d)", "long(>=10d)"):
        sub = df[(df["duration_bucket"] == bucket) & df["hba1c_group"].notna()]
        high = sub[sub["hba1c_group"] == "high"]["workIntegral"].dropna().tolist()
        low = sub[sub["hba1c_group"] == "low"]["workIntegral"].dropna().tolist()
        sep = rank_sep(high, low)
        rows[bucket] = {"n_high": len(high), "n_low": len(low), "rank_sep": sep}
        print(f"[{label}] {bucket:14s} n_high={len(high):3d} n_low={len(low):3d}  P(high>low)={sep}")

    delta = None
    if rows["short(<7d)"]["rank_sep"] is not None and rows["long(>=10d)"]["rank_sep"] is not None:
        delta = rows["long(>=10d)"]["rank_sep"] - rows["short(<7d)"]["rank_sep"]
        print(f"[{label}] headline delta (long - short) = {delta:+.4f} "
              f"({'still opposite to collapse hypothesis' if delta < 0 else 'now consistent with collapse hypothesis'})")

    return {"median_hba1c": float(median_hba1c), "buckets": rows, "delta_long_minus_short": delta,
            "n_independent_patients": len(df), "n_excluded_repeat_visits": n_excluded}


def main():
    f60 = sorted(Path("reports").glob("wind_tunnel_shanghai_t2dm_night_taumax60_*.json"))[-1]
    f120 = sorted(Path("reports").glob("wind_tunnel_shanghai_t2dm_night_taumax120_*.json"))[-1]

    res60 = run_one(f60, "tau_max=60")
    res120 = run_one(f120, "tau_max=120")

    print("\n" + "=" * 78)
    print("=== Side-by-side headline comparison ===")
    print("=" * 78)
    print(f"  short(<7d)  P(high>low): 60={res60['buckets']['short(<7d)']['rank_sep']}  "
          f"120={res120['buckets']['short(<7d)']['rank_sep']}")
    print(f"  long(>=10d) P(high>low): 60={res60['buckets']['long(>=10d)']['rank_sep']}  "
          f"120={res120['buckets']['long(>=10d)']['rank_sep']}")
    print(f"  delta(long-short):        60={res60['delta_long_minus_short']}  "
          f"120={res120['delta_long_minus_short']}")

    out_path = Path("reports/shanghai_duration_stratified_taumax_comparison_20260819_1545.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tau_max_60": res60, "tau_max_120": res120}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote structured comparison to {out_path}")


if __name__ == "__main__":
    main()
