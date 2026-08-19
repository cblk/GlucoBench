"""
Paired (same-body, cross-week) analysis for the T1D-UOM cohort: does the
SAME subject's own high-activity weeks systematically differ from their own
low-activity weeks, on Work Integral / tau / dim / det / entr?

Why paired, not pooled-cross-subject (mirrors analyze_mcphases_paired.py's
rationale, adapted for a behavioral rather than hormonal prism):
  T1D-UOM is under a strong-warning (dataset_fleet_registry.md Section 4):
  exogenous insulin dominates T1D glucose dynamics, so cross-SUBJECT pooling
  against T2D/non-diabetic cohorts is forbidden. But a same-body, own-median
  activity split sidesteps that entirely -- every comparison is a person
  against their own baseline, controlling for inter-subject insulin-regimen
  and disease-severity variance by construction (AGENTS.md Section 7).

Design:
  1. Per subject, rank ALL of that subject's own successful weekly runs by
     `weekly_step_count_total` and split at THAT SUBJECT'S OWN median (not a
     population-wide threshold -- baseline activity varies hugely between
     individuals, so an absolute step count would be meaningless).
  2. Aggregate each subject's "high" weeks and "low" weeks to a single mean
     per subject per bucket (collapsing multiple weeks in the same bucket
     into one cell, exactly like analyze_mcphases_paired.py's guard against
     pseudo-replication).
  3. Paired contrast: delta = high - low, per subject. Wilcoxon signed-rank +
     sign test, for ALL FIVE metrics swept identically and unconditionally
     (workIntegral is the user-facing target; tau/dim/det/entr are reported
     the same way, per No Frankenstein Scores discipline -- never combined).
  4. Confound disclosure (Wind-Tunnel Doctrine v1.1 Thermodynamic Bill):
     also run the IDENTICAL paired-delta procedure on `weekly_basal_dose_total`
     and `weekly_active_kcal_total` themselves, to check whether the "high
     activity" designation itself co-varies with a change in basal insulin
     dosing -- if it does, that is a genuine physiological confound this
     script must surface, not hide.

Doctrine compliance:
  - Section 9.1.2 Labels as Prisms, Not Targets: weekly_step_count_total only
    determines which two per-subject cells get subtracted; never a fit target.
  - Section 8.1 No Inference & No Fabrication: this script computes neutral
    statistics only; it does not attribute a causal mechanism (that step
    belongs to the Homomorphic Anchor Forge report, written afterward).
  - Section 9.4 Bit-for-Bit Truth Across Tracks: consumes the wind-tunnel JSON
    verbatim; recomputes no tensor operator.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

METRICS = ["workIntegral", "tau", "dim", "det", "entr"]
CONFOUND_METRICS = ["weekly_active_kcal_total", "weekly_basal_dose_total"]

files = sorted(Path("reports").glob("wind_tunnel_t1d_uom_activity_night_taumax60_*.json"))
if not files:
    raise FileNotFoundError("No wind_tunnel_t1d_uom_activity_night_taumax60_*.json found under reports/.")
latest = files[-1]
print("File:", latest)

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

raw = data["results"]
n_total = len(raw)
ok = [r for r in raw if r.get("workIntegral") is not None]
failed = [r for r in raw if r.get("workIntegral") is None]
print(f"Total weekly runs: {n_total} | successful: {len(ok)} | failed: {len(failed)}")

df = pd.DataFrame(ok)
df = df.dropna(subset=["weekly_step_count_total"])
print(f"Weekly runs with a valid activity total: {len(df)}")

# Step 1: per-subject median split on the subject's OWN step-count distribution.
def _assign_bucket(g):
    med = g["weekly_step_count_total"].median()
    g = g.copy()
    g["activity_bucket"] = np.where(g["weekly_step_count_total"] > med, "high",
                                     np.where(g["weekly_step_count_total"] < med, "low", "median_tie"))
    return g

df = df.groupby("original_id", group_keys=False)[df.columns].apply(_assign_bucket)
print("\nActivity bucket sizes (weekly runs, all subjects pooled):")
print(df["activity_bucket"].value_counts().to_string())

n_subjects_total = df["original_id"].nunique()
struct_out = {
    "source_file": str(latest),
    "n_total_weekly_runs": n_total,
    "n_success_weekly_runs": len(ok),
    "n_subjects_total": int(n_subjects_total),
    "metrics": {},
    "confound_checks": {},
}


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
print("=== Primary sweep: Work Integral + tau/dim/det/entr, high vs low activity week ===")
print("=" * 78)
for m in METRICS:
    struct_out["metrics"][m] = paired_report(m, m)


def holm_bonferroni(pvals_by_metric, m_family):
    """Holm-Bonferroni step-down correction across the METRICS family (m=5),
    applied SEPARATELY to Wilcoxon p-values and sign-test p-values (they test
    different null hypotheses -- magnitude-weighted vs direction-only -- so
    must not be pooled into one correction family). Per the mcPHASES paired
    report's explicit action item: multiple-comparison correction is
    mandatory before any metric in this unconditional 5-metric sweep can be
    called a lead, not just eyeballed against the uncorrected 0.05 line."""
    items = sorted(pvals_by_metric.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    for rank, (metric, p) in enumerate(items, start=1):
        threshold = 0.05 / (m - rank + 1)
        results[metric] = {"p_raw": p, "holm_threshold": threshold, "survives_holm": bool(p < threshold)}
    return results


wilcoxon_ps = {m: struct_out["metrics"][m]["wilcoxon_p"] for m in METRICS if "wilcoxon_p" in struct_out["metrics"][m]}
sign_ps = {m: struct_out["metrics"][m]["sign_test_p"] for m in METRICS if "sign_test_p" in struct_out["metrics"][m]}
struct_out["holm_bonferroni_wilcoxon"] = holm_bonferroni(wilcoxon_ps, METRICS)
struct_out["holm_bonferroni_sign_test"] = holm_bonferroni(sign_ps, METRICS)

print("\n" + "=" * 78)
print("=== Holm-Bonferroni correction across the 5-metric family (mandatory per mcPHASES precedent) ===")
print("=" * 78)
print("Wilcoxon (magnitude-weighted):")
for metric, r in sorted(struct_out["holm_bonferroni_wilcoxon"].items(), key=lambda kv: kv[1]["p_raw"]):
    print(f"  {metric:14s}: p={r['p_raw']:.4f}  Holm_threshold={r['holm_threshold']:.4f}  "
          f"survives={'YES' if r['survives_holm'] else 'no'}")
print("Sign test (direction-only):")
for metric, r in sorted(struct_out["holm_bonferroni_sign_test"].items(), key=lambda kv: kv[1]["p_raw"]):
    print(f"  {metric:14s}: p={r['p_raw']:.4f}  Holm_threshold={r['holm_threshold']:.4f}  "
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

out_path = Path("reports") / f"t1d_uom_activity_paired_analysis_{latest.stem.split('_')[-2]}_{latest.stem.split('_')[-1]}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(struct_out, f, indent=2, ensure_ascii=False, default=str)
print(f"\nStructured output written to: {out_path}")
