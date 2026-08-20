"""
Phase C rank-separation test for Core Dist (`coreDistAll`) -- the ONE Group-1 neutral metric
that survived the Phase B redundancy audit (see reports/group1_neutral_metrics_redundancy_
audit_20260819_2100.md: |rho|<=0.29 against all 4 graduated metrics in both cohorts, the
weakest coupling of the six). User explicitly approved testing Core Dist ONLY (not the other
5, which are Fail-Closed as redundant per that audit) -- AGENTS.md Section 9.3 Topological
Victory standard: P(high>low) > 0.80 replicated in >=2 independent heterogeneous cohorts.

Cohort 1: Stanford SSPG (sspg_class IR vs IS), same grouping as analyze_stanford_sspg_legacy_
results.py for direct comparability.
Cohort 2: Shanghai T2DM (HbA1c fixed global median split, first-visit dedup), same grouping as
analyze_shanghai_legacy_results.py.

Section 9.1.2 Labels as Prisms: sspg_class / hba1c_mmol_mol used ONLY to partition into groups
for a post-hoc distribution comparison, never as fit/regression targets.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


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


def permutation_p(a_vals, b_vals, observed_sep, n_perm=20000, seed=20260819):
    rng = random.Random(seed)
    pooled = a_vals + b_vals
    n_a = len(a_vals)
    observed_dev = abs(observed_sep - 0.5)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        perm_sep = rank_sep(perm_a, perm_b)
        if abs(perm_sep - 0.5) >= observed_dev:
            extreme += 1
    return extreme / n_perm


def analyze_stanford():
    files = sorted(Path('reports').glob('wind_tunnel_stanford_sspg_legacymetrics_group1_*.json'))
    latest = files[-1]
    print(f'\n=== Stanford SSPG (Core Dist), loading {latest} ===')
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = [r for r in data['results'] if 'error' not in r]

    is_vals = [r['coreDistAll'] for r in results if r.get('sspg_class') == 'IS' and r.get('coreDistAll') is not None]
    ir_vals = [r['coreDistAll'] for r in results if r.get('sspg_class') == 'IR' and r.get('coreDistAll') is not None]
    print(f'  n_IS={len(is_vals)}, n_IR={len(ir_vals)}')
    print(f'  IS median={np.median(is_vals):.4f}, IR median={np.median(ir_vals):.4f}')

    sep = rank_sep(ir_vals, is_vals)
    p = permutation_p(ir_vals, is_vals, sep)
    print(f'  P(IR > IS) = {sep:.4f}, permutation p = {p:.4f}')
    return {'cohort': 'stanford_sspg', 'sep': sep, 'p': p, 'n_a': len(ir_vals), 'n_b': len(is_vals)}


def analyze_shanghai():
    files = sorted(Path('reports').glob('wind_tunnel_shanghai_t2dm_legacymetrics_group1_night_taumax120_*.json'))
    latest = files[-1]
    print(f'\n=== Shanghai T2DM (Core Dist), loading {latest} ===')
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = data['results']
    df_all = pd.DataFrame([r for r in raw if 'error' not in r])
    df_all = df_all.sort_values('admission_date')
    df = df_all.groupby('patient_base_id', as_index=False).first()
    n_excluded = len(df_all) - len(df)
    print(f'  After first-visit dedup: {len(df)} independent patients ({n_excluded} repeat-visit recordings excluded).')

    hba1c_valid = df.dropna(subset=['hba1c_mmol_mol'])
    median_hba1c = hba1c_valid['hba1c_mmol_mol'].median()
    print(f'  HbA1c available for {len(hba1c_valid)}/{len(df)} patients. Fixed global median split = {median_hba1c:.2f} mmol/mol.')
    df['hba1c_group'] = np.where(
        df['hba1c_mmol_mol'].isna(), None,
        np.where(df['hba1c_mmol_mol'] > median_hba1c, 'high', 'low'),
    )

    sub = df[df['hba1c_group'].notna()]
    high = sub[sub['hba1c_group'] == 'high']['coreDistAll'].dropna().tolist()
    low = sub[sub['hba1c_group'] == 'low']['coreDistAll'].dropna().tolist()
    print(f'  n_high={len(high)}, n_low={len(low)}')
    print(f'  high median={np.median(high):.4f}, low median={np.median(low):.4f}')

    sep = rank_sep(high, low)
    p = permutation_p(high, low, sep)
    print(f'  P(high > low) = {sep:.4f}, permutation p = {p:.4f}')
    return {'cohort': 'shanghai_t2dm', 'sep': sep, 'p': p, 'n_a': len(high), 'n_b': len(low),
            'median_hba1c_mmol_mol': float(median_hba1c)}


def main():
    r1 = analyze_stanford()
    r2 = analyze_shanghai()

    print('\n' + '=' * 78)
    print('=== Section 9.3 Topological Victory check (P > 0.80 in >=2 independent cohorts) ===')
    print('=' * 78)
    for r in (r1, r2):
        sep = r['sep']
        verdict = 'PASS' if sep is not None and (sep > 0.80 or sep < 0.20) else 'fail'
        print(f"  {r['cohort']:16s}: sep={sep:.4f}  p={r['p']:.4f}  -> {verdict}")

    out_path = Path('reports/group1_coredist_topological_test_20260819_2130.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'stanford_sspg': r1, 'shanghai_t2dm': r2}, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nWrote structured output to {out_path}')


if __name__ == '__main__':
    main()
