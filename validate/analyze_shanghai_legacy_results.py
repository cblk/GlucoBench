"""
Analysis companion for wind_tunnel_v4_shanghai_legacy.py -- SECOND independent-cohort
replication attempt for candidate_tensor_staging_matrix.md 候选 #5 (`relaxationTime`),
using the EXACT dedup/grouping methodology of analyze_shanghai_results.py (first-visit
dedup for cross-sectional independence, ONE fixed global HbA1c median-split threshold,
rank-separation P(high>low) + permutation p-value) for direct comparability with the
already-reported Work Integral/DET/ENTR/Dim findings on this cohort.

Section 9.1.2: hba1c_mmol_mol is used ONLY to partition subjects into two groups for a
post-hoc distribution comparison, never as a fit/regression target.
Section 8.1 No Fabrication: patients with missing HbA1c dropped, not imputed.

Secondary check: duration_days ranges 2.6-13.9 days in this cohort (unlike Stanford's
uniformly long records). Since earlyDelay/relaxationTime require enough forced-rise
excursion EVENTS (not just enough raw points) to compute a per-subject median, short
recordings are a plausible confound specific to these two metrics (not shared by
AR1/friction/angular-velocity which use continuous-window statistics). Reported as an
explicit secondary stratification, not folded into the primary pooled result.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

files = sorted(Path('reports').glob('wind_tunnel_shanghai_t2dm_legacymetrics_night_taumax120_*.json'))
latest = files[-1]
print('Loading:', latest)

with open(latest, 'r', encoding='utf-8') as f:
    data = json.load(f)

raw = data['results']
print(f"Total recordings: {len(raw)} | succeeded: {sum(1 for r in raw if 'error' not in r)}")

df_all = pd.DataFrame([r for r in raw if 'error' not in r])
df_all = df_all.sort_values('admission_date')
df = df_all.groupby('patient_base_id', as_index=False).first()
n_excluded = len(df_all) - len(df)
print(f"After first-visit dedup: {len(df)} independent patients ({n_excluded} repeat-visit recordings excluded).")

hba1c_valid = df.dropna(subset=['hba1c_mmol_mol'])
median_hba1c = hba1c_valid['hba1c_mmol_mol'].median()
print(f"HbA1c available for {len(hba1c_valid)}/{len(df)} patients. Fixed global median split = {median_hba1c:.2f} mmol/mol.")
df['hba1c_group'] = np.where(
    df['hba1c_mmol_mol'].isna(), None,
    np.where(df['hba1c_mmol_mol'] > median_hba1c, 'high', 'low'),
)

metrics = ['earlyDelay', 'relaxationTime', 'ar1', 'angularVelocity', 'ascendFriction', 'nightFriction']


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


sub_all = df[df['hba1c_group'].notna()]
high_all = sub_all[sub_all['hba1c_group'] == 'high']
low_all = sub_all[sub_all['hba1c_group'] == 'low']

print("\n" + "=" * 78)
print("=== PRIMARY: Pooled (all durations) HbA1c high vs low, Legacy Metrics ===")
print(f"=== n_high={len(high_all)}, n_low={len(low_all)} ===")
print("=" * 78)
pvals = {}
seps = {}
for m in metrics:
    h = high_all[m].dropna().tolist()
    l = low_all[m].dropna().tolist()
    sep = rank_sep(h, l)
    if sep is None:
        print(f"  {m:16s}: insufficient data")
        continue
    p = permutation_p(h, l, sep)
    seps[m] = sep
    pvals[m] = p
    h_med = np.median(h) if h else float('nan')
    l_med = np.median(l) if l else float('nan')
    print(f"  {m:16s}: high_median={h_med:9.4f} (n={len(h):3d}) | low_median={l_med:9.4f} (n={len(l):3d}) | "
          f"P(high>low)={sep:.4f} | permutation p={p:.4f}")

print("\n=== Holm-Bonferroni correction (6-metric family, alpha=0.05) ===")
order = sorted(pvals.items(), key=lambda kv: kv[1])
m_count = len(order)
for i, (m, p) in enumerate(order, start=1):
    thresh = 0.05 / (m_count - i + 1)
    status = "SURVIVES" if p <= thresh else "fails"
    print(f"  rank {i}: {m:16s} p={p:.4f}  threshold={thresh:.5f}  -> {status}")

print("\n" + "=" * 78)
print("=== SECONDARY: duration-bucket breakdown for earlyDelay/relaxationTime (confound check) ===")
print("=" * 78)
df['duration_bucket'] = pd.cut(
    df['duration_days'], bins=[0, 7, 10, 100],
    labels=['short(<7d)', 'mid(7-10d)', 'long(>=10d)'],
)
print(df['duration_bucket'].value_counts().sort_index().to_string())
for bucket in ['short(<7d)', 'mid(7-10d)', 'long(>=10d)']:
    sub = df[(df['duration_bucket'] == bucket) & df['hba1c_group'].notna()]
    high = sub[sub['hba1c_group'] == 'high']
    low = sub[sub['hba1c_group'] == 'low']
    print(f"\n-- bucket {bucket} (n_high={len(high)}, n_low={len(low)}) --")
    for m in ['earlyDelay', 'relaxationTime']:
        h = high[m].dropna().tolist()
        l = low[m].dropna().tolist()
        sep = rank_sep(h, l)
        if sep is None:
            print(f"   {m}: insufficient data")
            continue
        print(f"   {m}: n_avail_high={len(h)}, n_avail_low={len(l)}, P(high>low)={sep:.4f}")

out_path = Path('reports/shanghai_t2dm_legacymetrics_analysis_20260819_1947.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'median_hba1c_mmol_mol': float(median_hba1c),
        'n_independent_patients': int(len(df)),
        'n_excluded_repeat_visits': int(n_excluded),
        'primary_pooled': {m: {'rank_sep': seps.get(m), 'permutation_p': pvals.get(m)} for m in metrics},
    }, f, indent=2, ensure_ascii=False, default=str)
print(f"\nWrote structured output to {out_path}")
