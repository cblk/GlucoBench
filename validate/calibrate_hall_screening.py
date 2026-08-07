#!/usr/bin/env python3
"""
Hall-cohort screening model calibration (untreated population — the actual
screening target). Features from output/js_hall_metrics.json (deployed JS
pipeline, v8.3 with sampled-RQA bugfix).

Protocol (mirrors validate_takens_v3.py):
  - 10x repeated 5-fold stratified CV, LR with standardization fit INSIDE folds
  - pre-registered candidate feature sets (no selection on the full data)
  - OOF AUC + permutation test (200 shuffles) on the chosen model
  - threshold profile on OOF probabilities
Also: Colas cross-check (train Hall → test Colas treated-T2DM) to quantify
population transfer, and vice versa.
"""
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

RNG = np.random.default_rng(42)

m = json.load(open('output/js_hall_metrics.json'))
Xraw = {k: np.array([x[k] for x in m], float) for k in
        ('lyapunov', 'entr', 'nightMean', 'volume', 'avgRecovery', 'det', 'dimension')}
diag = np.array([x['diagnosis'] for x in m])
y = (diag >= 1).astype(int)   # pre-diabetic + diabetic = positive (screening target)
print(f"Hall n={len(y)} positives={y.sum()}")

CANDIDATES = {
    'nightMean':            ['nightMean'],
    'nightMean+volume':     ['nightMean', 'volume'],
    'nightMean+volume+entr':['nightMean', 'volume', 'entr'],
    'nightMean+lyap+entr':  ['nightMean', 'lyapunov', 'entr'],   # deployed structure
    'entr+lyap+nightMean':  ['entr', 'lyapunov', 'nightMean'],   # deployed, ordered
    'volume+entr+nightMean+lyap': ['volume', 'entr', 'nightMean', 'lyapunov'],
    'all7':                 ['lyapunov', 'entr', 'nightMean', 'volume', 'avgRecovery', 'det', 'dimension'],
}

def oof_auc(feats, yv, cv):
    X = np.column_stack([Xraw[k] for k in feats])
    preds = np.zeros(len(yv))
    for tr, te in cv.split(X, yv):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, yv[tr])
        preds[te] = lr.predict_proba(Xte)[:, 1]
    pos, neg = preds[yv == 1], preds[yv == 0]
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), preds

print("\n=== honest OOF AUC (10x5 stratified CV, Hall) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, feats in CANDIDATES.items():
    aucs = []
    for rep in range(10):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
        a, _ = oof_auc(feats, y, cv)
        aucs.append(a)
    results[name] = (np.mean(aucs), np.std(aucs))
    print(f"  {name:<26} OOF AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# permutation test on the best candidate (and deployed structure for reference)
def perm_test(feats, n_perm=200):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    a, _ = oof_auc(feats, y, cv)
    nulls = []
    for _ in range(n_perm):
        yp = RNG.permutation(y)
        aa, _ = oof_auc(feats, yp, cv)
        nulls.append(aa)
    return a, np.mean(nulls), np.percentile(nulls, 95), (1 + sum(n >= a for n in nulls)) / (n_perm + 1)

print("\n=== permutation tests (200 shuffles) ===")
for name in ('nightMean', 'nightMean+volume', 'nightMean+volume+entr', 'entr+lyap+nightMean'):
    a, nm, p95, p = perm_test(CANDIDATES[name])
    print(f"  {name:<26} OOF AUC={a:.3f} null_mean={nm:.3f} null_p95={p95:.3f} p={p:.3f}")

# ---- cross-population transfer ----
print("\n=== cross-population transfer ===")
m_colas = json.load(open('output/js_cohort_metrics.json'))
cX = {k: np.array([x[k] for x in m_colas], float) for k in Xraw}
cy = np.array([x['y'] for x in m_colas])
# Colas-trained (deployed) model on Hall: use coefficients from Colas fit
def fit_predict(Xf, feats, yf, Xt):
    X = np.column_stack([Xf[k] for k in feats])
    Xt = np.column_stack([Xt[k] for k in feats])
    sc = StandardScaler().fit(X); lr = LogisticRegression(C=1.0).fit(sc.transform(X), yf)
    return lr.predict_proba(sc.transform(Xt))[:, 1]
for name, feats in (('entr+lyap+nightMean', ['entr', 'lyapunov', 'nightMean']),
                    ('nightMean+volume', ['nightMean', 'volume']),
                    ('nightMean', ['nightMean'])):
    # Hall-trained -> Colas
    ph = fit_predict(Xraw, feats, y, cX)
    pos, neg = ph[cy == 1], ph[cy == 0]
    r = stats.rankdata(np.concatenate([pos, neg]))
    auc_h2c = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    # Colas-trained -> Hall
    pc = fit_predict(cX, feats, cy, Xraw)
    pos, neg = pc[y == 1], pc[y == 0]
    r = stats.rankdata(np.concatenate([pos, neg]))
    auc_c2h = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f"  {name:<26} Hall→Colas AUC={auc_h2c:.3f} | Colas→Hall AUC={auc_c2h:.3f}")

# ---- deployed-model inversion on Hall (document) ----
deployed = np.array([x['risk'] for x in m])
pos, neg = deployed[y == 1], deployed[y == 0]
r = stats.rankdata(np.concatenate([pos, neg]))
auc_dep = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print(f"\n  deployed Colas LR on Hall: AUC={auc_dep:.3f} (below 0.5 = inverted)")
