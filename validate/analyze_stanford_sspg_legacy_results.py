"""
Analysis companion for wind_tunnel_v4_stanford_sspg_legacy.py -- reuses the EXACT same
methodology as analyze_stanford_results.py (rank-separation P(IR > IS) via Mann-Whitney-style
probability of superiority, grouped by the pre-existing sspg_class label) for direct
comparability with the already-reported Dim/Work Integral/DET/ENTR findings on this cohort.
Section 9.1.2: sspg_class is used ONLY to partition subjects into two groups for a post-hoc
distribution comparison, never as a fit/regression target.
"""
import json
import random
from pathlib import Path

files = list(Path('reports').glob('wind_tunnel_stanford_sspg_legacymetrics_*.json'))
latest = sorted(files)[-1]
print('Loading:', latest)

with open(latest, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def pct(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return vals_sorted[idx]

    return {
        'n': n, 'mean': round(sum(vals_sorted) / n, 4),
        'p25': round(pct(0.25), 4), 'median': round(pct(0.5), 4), 'p75': round(pct(0.75), 4),
        'min': round(vals_sorted[0], 4), 'max': round(vals_sorted[-1], 4)
    }


metrics = ['earlyDelay', 'relaxationTime', 'ar1', 'angularVelocity', 'ascendFriction', 'nightFriction']

print('\n=== Stanford SSPG Class (IS vs IR), period=night, tau_max=120, Legacy Metrics ===')
for grp_val, grp_name in [('IS', 'Insulin Sensitive (IS)'), ('IR', 'Insulin Resistant (IR)')]:
    grp = [r for r in results if r.get('sspg_class') == grp_val]
    print(f'-- group {grp_name} (n={len(grp)}) --')
    for m in metrics:
        s = stats([r.get(m) for r in grp])
        print(f'   {m}: {s}')

def rank_sep(ir_vals, is_vals):
    wins = 0.0
    for a in ir_vals:
        for b in is_vals:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(ir_vals) * len(is_vals)) if (is_vals and ir_vals) else None


def permutation_p(ir_vals, is_vals, observed_sep, n_perm=20000, seed=20260819):
    """Two-sided permutation test: null = no true group effect, i.e. group labels are
    exchangeable. Reports P(|null_sep - 0.5| >= |observed_sep - 0.5|) -- honest significance
    context for a small cohort (n=29) where a raw point-estimate separation alone risks
    overstating confidence (AGENTS.md Section 8 Honest Fail-Closed)."""
    rng = random.Random(seed)
    pooled = ir_vals + is_vals
    n_ir = len(ir_vals)
    observed_dev = abs(observed_sep - 0.5)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_ir = pooled[:n_ir]
        perm_is = pooled[n_ir:]
        perm_sep = rank_sep(perm_ir, perm_is)
        if abs(perm_sep - 0.5) >= observed_dev:
            extreme += 1
    return extreme / n_perm


print('\n=== Single-Metric Rank Separation P(IR > IS) + Permutation p-value (n=20000, two-sided) ===')
for m in metrics:
    is_vals = [r[m] for r in results if r.get('sspg_class') == 'IS' and r.get(m) is not None]
    ir_vals = [r[m] for r in results if r.get('sspg_class') == 'IR' and r.get(m) is not None]
    sep = rank_sep(ir_vals, is_vals)
    if sep is None:
        print(f'   {m}: separation unavailable (empty group).')
        continue
    p = permutation_p(ir_vals, is_vals, sep)
    print(f'   {m}: separation P(IR > IS) = {round(sep, 4)}, permutation p = {round(p, 4)}  (n_IR={len(ir_vals)}, n_IS={len(is_vals)})')

print('\n=== Per-Subject Raw Comparison ===')
for r in sorted(results, key=lambda x: x.get('sspg', 0)):
    print(f"{r['id']}: SSPG={r.get('sspg')}, Class={r.get('sspg_class')}, "
          f"AR1={r.get('ar1')}, AscendFric={r.get('ascendFriction')}, NightFric={r.get('nightFriction')}, "
          f"AngVel={r.get('angularVelocity')}, EarlyDelay={r.get('earlyDelay')}, RelaxTime={r.get('relaxationTime')}")
