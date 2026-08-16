import json
from pathlib import Path

files = list(Path('reports').glob('wind_tunnel_stanford_sspg_*.json'))
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
        idx = min(n-1, int(round(p*(n-1))))
        return vals_sorted[idx]
    return {
        'n': n, 'mean': round(sum(vals_sorted)/n, 4),
        'p25': round(pct(0.25), 4), 'median': round(pct(0.5), 4), 'p75': round(pct(0.75), 4),
        'min': round(vals_sorted[0], 4), 'max': round(vals_sorted[-1], 4)
    }

metrics = ['workIntegral', 'det', 'entr', 'tau', 'dim']

print('\n=== Stanford: SSPG Class (IS n=16, IR n=13), period=night, tau_max=60 ===')
for grp_val, grp_name in [('IS', 'Insulin Sensitive (IS)'), ('IR', 'Insulin Resistant (IR)')]:
    grp = [r for r in results if r.get('sspg_class') == grp_val]
    print(f'-- group {grp_name} (n={len(grp)}) --')
    for m in metrics:
        s = stats([r.get(m) for r in grp])
        print(f'   {m}: {s}')

# Rank separation P(IR > IS) for each metric
print('\n=== Single-Metric Rank Separation P(IR > IS) ===')
for m in metrics:
    is_vals = [r[m] for r in results if r.get('sspg_class') == 'IS' and r.get(m) is not None]
    ir_vals = [r[m] for r in results if r.get('sspg_class') == 'IR' and r.get(m) is not None]
    wins = 0.0
    for a in ir_vals:
        for b in is_vals:
            if a > b: wins += 1.0
            elif a == b: wins += 0.5
    sep = wins / (len(ir_vals) * len(is_vals)) if (is_vals and ir_vals) else None
    print(f'   {m}: separation P(IR > IS) = {round(sep, 4)}')

print('\n=== Per-Subject Raw Comparison ===')
for r in sorted(results, key=lambda x: x.get('sspg', 0)):
    print(f"{r['id']}: SSPG={r.get('sspg')}, Class={r.get('sspg_class')}, WorkInt={r.get('workIntegral')}, Tau={r.get('tau')}, Dim={r.get('dim')}, DET={r.get('det')}")
