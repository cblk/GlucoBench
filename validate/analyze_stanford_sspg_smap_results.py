"""
Analysis companion for wind_tunnel_v4_stanford_sspg_smap.py -- reuses the EXACT same
rank-separation P(IR > IS) + permutation-p methodology as
analyze_stanford_sspg_legacy_results.py for direct comparability with the already-reported
legacy-metric findings on this cohort.

Working hypothesis (roadmap Section 4.3 / Section 3 Phase C stage 1, provisional target
population): deep decompensation failure (IR, insulin-resistant) is hypothesized to show
LOWER delta_rho (nonlinear-predictability gain) than IS (insulin-sensitive) -- i.e. the
system's dynamics have degraded toward linear/white-noise. This script reports BOTH
P(IR > IS) (for direct comparability with other metrics' printed convention) AND explicitly
states which direction the working hypothesis actually predicts for delta_rho specifically
(IS > IR, i.e. P(IR > IS) < 0.5) so the sign is not misread.

Section 9.1.2: sspg_class is used ONLY to partition subjects into two groups for a post-hoc
distribution comparison, never as a fit/regression target.
"""
import json
import random
from pathlib import Path

files = list(Path('reports').glob('wind_tunnel_stanford_sspg_smap_tp*.json'))
latest = sorted(files)[-1]
print('Loading:', latest)

with open(latest, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
print(f"tp={data.get('tp')}, e_max={data.get('e_max')}, min_lib={data.get('min_lib')}")


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


metrics = ['delta_rho', 'theta_best', 'e_best', 'n_nights_used']

print('\n=== Stanford SSPG Class (IS vs IR), night RAW track, tp=10 (30min), S-Map Delta-rho ===')
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


def permutation_p(ir_vals, is_vals, observed_sep, n_perm=20000, seed=20260820):
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
print('(Working hypothesis predicts delta_rho: IS > IR, i.e. P(IR > IS) < 0.5 -- see module docstring)')
for m in ['delta_rho', 'theta_best', 'e_best']:
    is_vals = [r[m] for r in results if r.get('sspg_class') == 'IS' and r.get(m) is not None]
    ir_vals = [r[m] for r in results if r.get('sspg_class') == 'IR' and r.get(m) is not None]
    sep = rank_sep(ir_vals, is_vals)
    if sep is None:
        print(f'   {m}: separation unavailable (empty group).')
        continue
    p = permutation_p(ir_vals, is_vals, sep)
    print(f'   {m}: separation P(IR > IS) = {round(sep, 4)}, permutation p = {round(p, 4)}  (n_IR={len(ir_vals)}, n_IS={len(is_vals)})')

print('\n=== Continuous SSPG / DI correlation (Spearman-style rank correlation via Pearson-on-ranks) ===')


def rank_of(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def spearman(xs, ys):
    n = len(xs)
    rx = rank_of(xs)
    ry = rank_of(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = (sum((r - mx) ** 2 for r in rx)) ** 0.5
    deny = (sum((r - my) ** 2 for r in ry)) ** 0.5
    return num / (denx * deny) if denx > 0 and deny > 0 else None


for label_field in ['sspg', 'di']:
    pairs = [(r['delta_rho'], r.get(label_field)) for r in results if r.get('delta_rho') is not None and r.get(label_field) is not None]
    if len(pairs) < 5:
        print(f'   delta_rho vs {label_field}: insufficient paired data (n={len(pairs)}).')
        continue
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = spearman(xs, ys)
    print(f'   delta_rho vs {label_field}: n={len(pairs)}, spearman_rho={round(rho, 4) if rho is not None else None}')

print('\n=== Per-Subject Raw Comparison (sorted by SSPG) ===')
for r in sorted(results, key=lambda x: x.get('sspg', 0) or 0):
    print(f"{r['id']}: SSPG={r.get('sspg')}, DI={r.get('di')}, Class={r.get('sspg_class')}, "
          f"delta_rho={r.get('delta_rho')}, theta_best={r.get('theta_best')}, e_best={r.get('e_best')}, "
          f"n_nights_used={r.get('n_nights_used')}")
