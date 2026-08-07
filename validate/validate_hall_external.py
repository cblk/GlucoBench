#!/usr/bin/env python3
"""
External validation of the DEPLOYED Colas-calibrated screening model on the
Hall cohort (PLOS Biol pbio.2005143, n=57: 38 non-diabetic / 14 pre-diabetic /
5 diabetic). Features and risk come from output/js_hall_metrics.json, produced
by output/js_hall_validation.mjs running the ACTUAL index.html pipeline
(vm sandbox, identical to js_cohort.mjs).

Questions:
1. AUC of deployed risk P for non-diabetic vs pre-diabetic+diabetic
   (and non-diabetic vs diabetic, directional only).
2. Sens/Spec of deployed thresholds 0.30/0.50/0.70 on Hall.
3. MRI recalibrated threshold hit rates by group (transfer check).
4. Risk correlation with A1C/FBG/OGTT-2h/insulin (Spearman).
5. Clinical ceilings: single-feature AUC of A1C/FBG/OGTT for the same split.
6. DET/ENTR desaturation probe (RR=0.5% vs deployed 2%): dynamic range + AUC.
"""
import json
import numpy as np
from scipy import stats

m = json.load(open('output/js_hall_metrics.json'))
print(f"n={len(m)}  diagnosis: "
      f"{sum(1 for x in m if x['diagnosis']==0)} non / "
      f"{sum(1 for x in m if x['diagnosis']==1)} pre / "
      f"{sum(1 for x in m if x['diagnosis']==2)} diab")

def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    if len(np.unique(y)) < 2: return float('nan')
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    n1, n0 = len(pos), len(neg)
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5: return float('nan'), 1.0
    return stats.spearmanr(x[ok], y[ok])

risk = np.array([x['risk'] for x in m])
diag = np.array([x['diagnosis'] for x in m])
y_pre = (diag >= 1).astype(int)   # pre-diabetic + diabetic
y_diab = (diag >= 2).astype(int)  # diabetic only

print("\n=== 1. Deployed model discrimination (external) ===")
for name, y in [('non vs pre+diab', y_pre), ('non vs diab', y_diab)]:
    a = auc(y, risk)
    print(f"  {name:<16} AUC={a:.3f}  (n_pos={y.sum()})")

print("\n=== 2. Deployed thresholds on Hall ===")
for th in (0.30, 0.50, 0.70):
    pred = risk >= th
    tp = (pred & (y_pre == 1)).sum(); fn = ((~pred) & (y_pre == 1)).sum()
    fp = (pred & (y_pre == 0)).sum(); tn = ((~pred) & (y_pre == 0)).sum()
    se = tp / (tp + fn) if tp + fn else float('nan')
    sp = tn / (tn + fp) if tn + fp else float('nan')
    print(f"  th={th:.2f}: Sens={se:.3f} Spec={sp:.3f} flagged={(pred.mean()*100):.1f}% "
          f"(Colas OOF: 0.30→0.882/0.209, 0.50→0.765/0.618, 0.70→0.059/0.942)")

print("\n=== 3. MRI recalibrated thresholds — hit rate by group (transfer check) ===")
def pct(cond, mask): 
    s = sum(1 for x, mm in zip(m, mask) if cond(x) and mm); return s / sum(mask)
def gmask(v): return [x['diagnosis'] == v for x in m]
for name, cond in [
    ('volume<-1(bad-lo)',  lambda x: np.log(max(x['volume'], 1e-6)) < -1.0),
    ('volume<0(warn-lo)',  lambda x: np.log(max(x['volume'], 1e-6)) < 0.0),
    ('volume>3(bad-hi)',   lambda x: np.log(max(x['volume'], 1e-6)) > 3.0),
    ('det<0.990(bad)',     lambda x: x['det'] < 0.990),
    ('det<0.994(warn)',    lambda x: x['det'] < 0.994),
    ('entr<2.6(bad)',      lambda x: x['entr'] < 2.6),
    ('entr<2.9(warn)',     lambda x: x['entr'] < 2.9),
    ('lyap>0.15(warn)',    lambda x: x['lyapunov'] > 0.15),
    ('lyap>0.20(bad)',     lambda x: x['lyapunov'] > 0.20),
]:
    print(f"  {name:<20} non={pct(cond, gmask(0)):.2f} pre={pct(cond, gmask(1)):.2f} "
          f"diab={pct(cond, gmask(2)):.2f}")

print("\n=== 4. Risk vs clinical labels (Spearman, all 57) ===")
for k in ('a1c', 'fbg', 'ogtt', 'insulin', 'bmi'):
    r, p = spearman(risk, [x[k] for x in m])
    print(f"  risk vs {k:<8} rho={r:+.3f} p={p:.3f}")

print("\n=== 5. Clinical ceilings (single-feature AUC, non vs pre+diab) ===")
for k in ('a1c', 'fbg', 'ogtt'):
    a = auc(y_pre, [x[k] for x in m])
    print(f"  {k:<8} AUC={a:.3f}")
a = auc(y_pre, risk); print(f"  {'deployed-risk':<8} AUC={a:.3f}")

print("\n=== 6. DET/ENTR desaturation probe (RR target 0.5% vs deployed 2%) ===")
r05 = {x['id']: x for x in json.load(open('output/js_hall_rr05.json'))}
ids = [x['id'] for x in m]
det02 = np.array([x['det'] for x in m]); entr02 = np.array([x['entr'] for x in m])
det05 = np.array([r05[i]['det'] for i in ids]); entr05 = np.array([r05[i]['entr'] for i in ids])
print(f"  det@2%:  range [{det02.min():.4f},{det02.max():.4f}] sd={det02.std():.4f}  AUC={auc(y_pre, det02):.3f}")
print(f"  det@0.5%:range [{det05.min():.4f},{det05.max():.4f}] sd={det05.std():.4f}  AUC={auc(y_pre, det05):.3f}")
print(f"  entr@2%:  range [{entr02.min():.3f},{entr02.max():.3f}] sd={entr02.std():.3f}  AUC={auc(y_pre, entr02):.3f}")
print(f"  entr@0.5%:range [{entr05.min():.3f},{entr05.max():.3f}] sd={entr05.std():.3f}  AUC={auc(y_pre, entr05):.3f}")
# correlation of the two calibrations (stability)
r, p = spearman(det02, det05); print(f"  det@2% vs det@0.5% rho={r:.3f}")
r, p = spearman(entr02, entr05); print(f"  entr@2% vs entr@0.5% rho={r:.3f}")

print("\n=== feature means by group (direction check) ===")
for k in ('lyapunov', 'entr', 'nightMean', 'volume', 'dimension', 'avgRecovery', 'det'):
    vals = np.array([x[k] for x in m])
    print(f"  {k:<10} non={vals[diag==0].mean():.4f} pre={vals[diag==1].mean():.4f} "
          f"diab={vals[diag==2].mean():.4f}")
