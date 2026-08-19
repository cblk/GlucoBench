"""
Analysis: does the "short-cycle collapse" seen in Colas (n=208, ~2-day
recordings, Work Integral rank-sep 0.699 -> 0.563 vs Hall) replicate WITHIN
a single homogeneous cohort (same hospital, same disease type, same
extraction protocol) when we deliberately split it by recording duration?

Design (mirrors dataset_fleet_registry.md Section 5's pre-registered question):
  1. Deduplicate to ONE record per unique patient (earliest admission_date),
     to preserve cross-sectional independence -- Shanghai_T2DM has 8 patients
     with 2-3 separate hospital admissions, and pooling all visits as if they
     were independent i.i.d. samples would violate the Wind-Tunnel Doctrine
     v1.1 Causal Chirality constraint (a later admission is not a symmetric
     repeat of an earlier one).
  2. Split the resulting n=100 independent patients into duration buckets:
     short (<7 days, Colas-like), mid (7-10 days), long (>=10 days,
     Hall/Stanford-like).
  3. Use ONE fixed HbA1c median-split threshold computed across the FULL
     n=100 population (not recomputed per bucket) as the prism, so duration
     is the only thing that varies between the sub-analyses.
  4. Compare Work Integral's rank-separation power (P(high HbA1c > low HbA1c))
     across duration buckets. Also report DET/Dim/Tau for the same buckets as
     an unconditional sweep (Section 9.1.3 No Frankenstein Scores: reported
     independently, not combined).

Doctrine compliance:
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c only used to split
    groups for a rank-separation comparison, never as a regression target.
  - Section 8.1 No Inference & No Fabrication: patients with missing HbA1c
    (the "/" sentinel, already converted to None at extraction time) are
    dropped from the HbA1c-stratified comparison, not imputed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

files = sorted(Path("reports").glob("wind_tunnel_shanghai_t2dm_night_taumax60_*.json"))
if not files:
    raise FileNotFoundError("No wind_tunnel_shanghai_t2dm_night_taumax60_*.json found under reports/.")
latest = files[-1]
print("File:", latest)

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

raw = data["results"]
print(f"Total recordings: {len(raw)} | with workIntegral: {sum(1 for r in raw if r.get('workIntegral') is not None)}")

df_all = pd.DataFrame(raw)

# Step 1: dedupe to first visit per patient (earliest admission_date), preserving
# cross-sectional independence per the Causal Chirality constraint.
df_all = df_all.sort_values("admission_date")
df = df_all.groupby("patient_base_id", as_index=False).first()
print(f"After first-visit dedup: {len(df)} independent patients "
      f"({len(df_all) - len(df)} repeat-visit recordings excluded from this cross-sectional analysis).")

# Step 2: duration buckets.
df["duration_bucket"] = pd.cut(
    df["duration_days"], bins=[0, 7, 10, 100],
    labels=["short(<7d)", "mid(7-10d)", "long(>=10d)"]
)
print("\nDuration bucket sizes:")
print(df["duration_bucket"].value_counts().sort_index().to_string())

# Step 3: ONE fixed HbA1c median split across the full n=100 population.
hba1c_valid = df.dropna(subset=["hba1c_mmol_mol"])
median_hba1c = hba1c_valid["hba1c_mmol_mol"].median()
print(f"\nHbA1c available for {len(hba1c_valid)}/{len(df)} patients. "
      f"Fixed global median split threshold = {median_hba1c:.2f} mmol/mol.")
df["hba1c_group"] = np.where(
    df["hba1c_mmol_mol"].isna(), None,
    np.where(df["hba1c_mmol_mol"] > median_hba1c, "high", "low")
)


def rank_sep(a_list, b_list):
    """P(a > b), 0.5-tie-weighted. a='high' group, b='low' group."""
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


metrics = ["workIntegral", "tau", "dim", "det", "entr"]

print("\n" + "=" * 78)
print("=== Work Integral (and sweep metrics) rank-separation by duration bucket ===")
print("=== Fixed prism: HbA1c high vs low, split at global median (all buckets) ===")
print("=" * 78)

bucket_order = ["short(<7d)", "mid(7-10d)", "long(>=10d)"]
summary_rows = []
for bucket in bucket_order:
    sub = df[(df["duration_bucket"] == bucket) & df["hba1c_group"].notna()]
    high = sub[sub["hba1c_group"] == "high"]
    low = sub[sub["hba1c_group"] == "low"]
    print(f"\n--- Bucket: {bucket} (n_high={len(high)}, n_low={len(low)}) ---")
    for m in metrics:
        h = high[m].dropna().tolist()
        l = low[m].dropna().tolist()
        sep = rank_sep(h, l)
        if sep is None:
            print(f"  {m:14s}: insufficient data")
            continue
        h_med = np.median(h) if h else float("nan")
        l_med = np.median(l) if l else float("nan")
        print(f"  {m:14s}: high_median={h_med:8.4f} (n={len(h)}) | low_median={l_med:8.4f} (n={len(l)}) | P(high>low)={sep:.4f}")
        summary_rows.append({
            "bucket": bucket, "metric": m, "n_high": len(h), "n_low": len(l),
            "high_median": h_med, "low_median": l_med, "rank_sep": sep,
        })

print("\n" + "=" * 78)
print("=== Pooled (all durations together) reference, for comparison to Hall/Colas history ===")
print("=" * 78)
sub_all = df[df["hba1c_group"].notna()]
high_all = sub_all[sub_all["hba1c_group"] == "high"]
low_all = sub_all[sub_all["hba1c_group"] == "low"]
for m in metrics:
    h = high_all[m].dropna().tolist()
    l = low_all[m].dropna().tolist()
    sep = rank_sep(h, l)
    if sep is None:
        continue
    print(f"  {m:14s}: P(high>low)={sep:.4f}  (n_high={len(h)}, n_low={len(l)})")
    summary_rows.append({
        "bucket": "pooled(all)", "metric": m, "n_high": len(h), "n_low": len(l),
        "high_median": float(np.median(h)) if h else None,
        "low_median": float(np.median(l)) if l else None,
        "rank_sep": sep,
    })

# Step 4: the headline short-cycle-collapse test -- explicit delta.
wi_short = next((r for r in summary_rows if r["bucket"] == "short(<7d)" and r["metric"] == "workIntegral"), None)
wi_long = next((r for r in summary_rows if r["bucket"] == "long(>=10d)" and r["metric"] == "workIntegral"), None)
print("\n" + "=" * 78)
print("=== Headline short-cycle-collapse test (Work Integral only) ===")
print("=" * 78)
if wi_short and wi_long:
    delta = wi_long["rank_sep"] - wi_short["rank_sep"]
    print(f"  short(<7d)  P(high>low) = {wi_short['rank_sep']:.4f}  (n_high={wi_short['n_high']}, n_low={wi_short['n_low']})")
    print(f"  long(>=10d) P(high>low) = {wi_long['rank_sep']:.4f}  (n_high={wi_long['n_high']}, n_low={wi_long['n_low']})")
    print(f"  delta (long - short) = {delta:+.4f}")

out_path = Path("reports") / f"shanghai_duration_stratified_analysis_{latest.stem.split('_')[-2]}_{latest.stem.split('_')[-1]}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "source_file": str(latest),
        "median_hba1c_mmol_mol": float(median_hba1c),
        "n_independent_patients": int(len(df)),
        "n_excluded_repeat_visits": int(len(df_all) - len(df)),
        "duration_bucket_sizes": df["duration_bucket"].value_counts().to_dict(),
        "rows": summary_rows,
    }, f, indent=2, ensure_ascii=False, default=str)
print(f"\nStructured output written to: {out_path}")
