"""
Same-body paired analysis (mirrors analyze_t1d_uom_paired.py's design for candidate #4
`dim`) for candidate #5 `relaxationTime` THIRD independent-cohort test on T1D-UOM, plus the
5 other Group-2 legacy metrics swept unconditionally (No Frankenstein Scores discipline).

Primary target: relaxationTime_v4a (exponential-decay-fit redesign, see
_relaxation_time_v4_expfit.py) -- recomputed directly here (NOT via run_subject_legacy(),
which still uses the original v1 point-snap _legacy_metrics_v4.compute_excursion_kinetics).
Secondary/context: the 6 original v1 legacy metrics from
wind_tunnel_t1d_uom_activity_legacymetrics_*.json, for comparison and honesty about whether
v1's relaxationTime shows the same paired signal as v4a.

Design: per subject, split THAT SUBJECT's own weekly runs at their own median
weekly_step_count_total (identical to analyze_t1d_uom_paired.py). Paired Wilcoxon
signed-rank + sign test on subject-level (high_mean - low_mean). Holm-Bonferroni across the
7-metric family (6 legacy v1 metrics + relaxationTime_v4a), applied separately to Wilcoxon
and sign-test p-values. Confound disclosure: identical paired procedure on
weekly_basal_dose_total itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
from _relaxation_time_v4_expfit import compute_excursion_kinetics_expfit

METRICS = ["earlyDelay", "relaxationTime", "ar1", "angularVelocity", "ascendFriction", "nightFriction",
           "relaxationTime_v4a"]
CONFOUND_METRICS = ["weekly_active_kcal_total", "weekly_basal_dose_total"]

files = sorted(Path("reports").glob("wind_tunnel_t1d_uom_activity_legacymetrics_night_taumax120_*.json"))
latest = files[-1]
print("File:", latest)

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

raw = data["results"]
n_total = len(raw)
ok_raw = [r for r in raw if "error" not in r]
print(f"Total weekly runs: {n_total} | successful (v1 legacy metrics): {len(ok_raw)}")

with open("output/t1d_uom_subjects.json", "r", encoding="utf-8") as f:
    subj_data = json.load(f)
subjects_by_id = {s["id"]: s for s in subj_data["subjects"]}


def weeks_to_pseudo_subjects(subject, min_points=30):
    out = []
    for w in subject["weeks"]:
        if len(w["values"]) < min_points:
            continue
        out.append({
            "id": f"{subject['id']}_wk_{w['week_start'][:10]}",
            "timestamps": w["timestamps"], "values": w["values"],
        })
    return out


pseudo_lookup = {}
for s in subj_data["subjects"]:
    for pw in weeks_to_pseudo_subjects(s):
        pseudo_lookup[pw["id"]] = pw

print(f"\nRecomputing relaxationTime_v4a (exponential-decay fit) for {len(ok_raw)} weekly runs...")
v4a_by_id = {}
n_recompute_fail = 0
for r in ok_raw:
    sid = r["id"]
    pw = pseudo_lookup.get(sid)
    if pw is None:
        n_recompute_fail += 1
        continue
    try:
        ts, raw_vs = wt.resample_raw(pw["timestamps"], pw["values"])
        if sum(1 for v in raw_vs if v is not None) < 60:
            n_recompute_fail += 1
            continue
        raw_json = json.dumps(raw_vs)
        tau_res = json.loads(wt.eng.engine.extract_tau(raw_json, max_lag=120))
        if tau_res.get("result") is None:
            n_recompute_fail += 1
            continue
        filt_res = json.loads(wt.eng.engine.filter_chunks(raw_json, 2, 0.08))
        smooth_vs = filt_res.get("result")
        if smooth_vs is None:
            n_recompute_fail += 1
            continue
        kin = compute_excursion_kinetics_expfit(ts, smooth_vs)
        v4a_by_id[sid] = kin["relaxationTime"]
    except Exception as e:
        print(f"  [WARN] {sid} v4a recompute failed: {e}")
        n_recompute_fail += 1

print(f"v4a recompute: {len(v4a_by_id)} succeeded, {n_recompute_fail} failed/skipped.")

df = pd.DataFrame(ok_raw)
df["relaxationTime_v4a"] = df["id"].map(v4a_by_id)
df = df.dropna(subset=["weekly_step_count_total"])
print(f"Weekly runs with a valid activity total: {len(df)}")


def _assign_bucket(g):
    med = g["weekly_step_count_total"].median()
    g = g.copy()
    g["activity_bucket"] = np.where(g["weekly_step_count_total"] > med, "high",
                                     np.where(g["weekly_step_count_total"] < med, "low", "median_tie"))
    return g


df = df.groupby("original_id", group_keys=False)[df.columns].apply(_assign_bucket)
n_subjects_total = df["original_id"].nunique()
print("\nActivity bucket sizes (weekly runs, all subjects pooled):")
print(df["activity_bucket"].value_counts().to_string())

struct_out = {"source_file": str(latest), "n_total_weekly_runs": n_total,
              "n_success_weekly_runs": len(ok_raw), "n_subjects_total": int(n_subjects_total),
              "metrics": {}, "confound_checks": {}}


def paired_report(metric_col, label):
    cell = df.groupby(["original_id", "activity_bucket"])[metric_col].mean().reset_index()
    pivot = cell.pivot(index="original_id", columns="activity_bucket", values=metric_col)
    for col in ("high", "low"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    sub = pivot[["high", "low"]].dropna()
    n_pairs = len(sub)
    print(f"\n--- {label} (n_pairs={n_pairs}/{n_subjects_total} subjects with both buckets populated) ---")
    if n_pairs < 3:
        print("  too few paired subjects, skipped")
        return {"n_pairs": n_pairs, "note": "insufficient paired subjects"}

    delta = sub["high"] - sub["low"]
    n_pos = int((delta > 0).sum())
    n_neg = int((delta < 0).sum())
    n_tie = n_pairs - n_pos - n_neg
    if (delta != 0).any():
        wstat, wp = stats.wilcoxon(sub["high"], sub["low"])
    else:
        wstat, wp = float("nan"), float("nan")
    sign_p = stats.binomtest(n_pos, n_pos + n_neg, p=0.5).pvalue if (n_pos + n_neg) > 0 else float("nan")
    mean_delta = float(delta.mean())
    sd_delta = float(delta.std(ddof=1)) if n_pairs > 1 else float("nan")
    cohens_d = mean_delta / sd_delta if sd_delta and sd_delta > 0 else float("nan")

    print(f"  high_mean={sub['high'].mean():.4f}  low_mean={sub['low'].mean():.4f}")
    print(f"  mean_delta(high-low)={mean_delta:+.4f}  cohen_d={cohens_d:+.3f}  "
          f"n_pos={n_pos} n_neg={n_neg} n_tie={n_tie}  Wilcoxon_p={wp:.4f}  sign_p={sign_p:.4f}")

    return {
        "n_pairs": n_pairs, "n_pos_high_gt_low": n_pos, "n_neg_high_lt_low": n_neg, "n_tie": n_tie,
        "high_mean": float(sub["high"].mean()), "low_mean": float(sub["low"].mean()),
        "mean_delta_high_minus_low": mean_delta, "cohens_d_paired": cohens_d,
        "wilcoxon_stat": float(wstat), "wilcoxon_p": float(wp), "sign_test_p": float(sign_p),
    }


print("\n" + "=" * 78)
print("=== Primary+context sweep: 6 v1 legacy metrics + relaxationTime_v4a, high vs low activity week ===")
print("=" * 78)
for m in METRICS:
    struct_out["metrics"][m] = paired_report(m, m)


def holm_bonferroni(pvals_by_metric):
    items = sorted(pvals_by_metric.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    for rank, (metric, p) in enumerate(items, start=1):
        threshold = 0.05 / (m - rank + 1)
        results[metric] = {"p_raw": p, "holm_threshold": threshold, "survives_holm": bool(p < threshold)}
    return results


wilcoxon_ps = {m: struct_out["metrics"][m]["wilcoxon_p"] for m in METRICS if "wilcoxon_p" in struct_out["metrics"][m]}
sign_ps = {m: struct_out["metrics"][m]["sign_test_p"] for m in METRICS if "sign_test_p" in struct_out["metrics"][m]}
struct_out["holm_bonferroni_wilcoxon"] = holm_bonferroni(wilcoxon_ps)
struct_out["holm_bonferroni_sign_test"] = holm_bonferroni(sign_ps)

print("\n" + "=" * 78)
print(f"=== Holm-Bonferroni correction across the {len(wilcoxon_ps)}-metric family ===")
print("=" * 78)
print("Wilcoxon (magnitude-weighted):")
for metric, r in sorted(struct_out["holm_bonferroni_wilcoxon"].items(), key=lambda kv: kv[1]["p_raw"]):
    print(f"  {metric:20s}: p={r['p_raw']:.4f}  Holm_threshold={r['holm_threshold']:.4f}  "
          f"survives={'YES' if r['survives_holm'] else 'no'}")
print("Sign test (direction-only):")
for metric, r in sorted(struct_out["holm_bonferroni_sign_test"].items(), key=lambda kv: kv[1]["p_raw"]):
    print(f"  {metric:20s}: p={r['p_raw']:.4f}  Holm_threshold={r['holm_threshold']:.4f}  "
          f"survives={'YES' if r['survives_holm'] else 'no'}")

print("\n" + "=" * 78)
print("=== Confound disclosure: does 'high activity' itself co-vary with insulin dosing? ===")
print("=" * 78)
for m in CONFOUND_METRICS:
    n_valid = df[m].notna().sum()
    print(f"\n[{m}] valid weekly values: {n_valid}/{len(df)}")
    if n_valid < 10:
        print("  insufficient data, skipped")
        struct_out["confound_checks"][m] = {"note": "insufficient data"}
        continue
    struct_out["confound_checks"][m] = paired_report(m, m)

out_path = Path("reports") / f"t1d_uom_legacymetrics_paired_analysis_{latest.stem.split('_')[-2]}_{latest.stem.split('_')[-1]}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(struct_out, f, indent=2, ensure_ascii=False, default=str)
print(f"\nStructured output written to: {out_path}")
