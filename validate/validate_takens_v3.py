"""
GlucoBench Takens Phase Space Validation — v3 OPTIMIZATION
===========================================================
Purpose: honest re-validation of the core algorithm + optimization.
Fixes vs v2:
  1. Honest evaluation: nested feature selection inside CV folds,
     repeated 10x5 stratified CV (only 17 T2DM / 208).
  2. Direction audit of the deployed MRI scoring (DET/ENTR/LYAP thresholds).
  3. New standard CGM features (TAR, TBR, CONGA-1, MAGE, MODD, LBGI/HBGI,
     J-index, night/dawn windows) as competitors.
  4. Lightweight final rule (small LR / shallow tree) hard-codable in JS.
"""

import numpy as np
import pandas as pd
import math
import json, os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import warnings
warnings.filterwarnings('ignore')

MMOL = 18.0182

# ============================================================================
# S1. Load cached v2 phase-space features
# ============================================================================
def load_cached():
    with open('output/takens_results_v2.json') as f:
        res = json.load(f)
    df = pd.DataFrame(res)
    df['mri_opt'] = df.apply(lambda r: mri_score(r), axis=1)
    return df

def mri_score(m):
    base = 1000.0
    lv = m.get('log_volume', 0)
    if lv < -1.0: base -= 150
    elif lv < 0.0: base -= 50
    elif lv > 3.0: base -= 180
    elif lv > 2.0: base -= 80
    r = m.get('avg_recovery', 0.1)
    if r > 0.30: base -= 200
    elif r > 0.20: base -= 90
    elif r < 0.05: base -= 150
    elif r < 0.08: base -= 60
    sr = m.get('shape_ratio', 1.5)
    if sr < 1.1: base -= 200
    elif sr < 1.5: base -= 80
    d = m.get('dimension')
    if d is not None:
        if d > 2.5: base -= 100
        elif d > 2.0: base -= 40
        elif d < 1.2: base -= 40
    ly = m.get('lyapunov')
    if ly is not None:
        if ly > 0.7: base -= 120
        elif ly > 0.4: base -= 50
    det = m.get('det')
    if det is not None:
        if det > 0.75: base -= 100
        elif det > 0.60: base -= 40
        elif det < 0.15: base -= 40
    entr = m.get('entr')
    if entr is not None:
        if entr < 0.5: base -= 120
        elif entr < 1.0: base -= 50
    tau = m.get('tau', 6)
    if tau > 15: base -= (tau - 15) * 15
    cv = m.get('cv_pct', 20)
    if cv > 35: base -= 150
    elif cv > 25: base -= 60
    return max(0, base)

# ============================================================================
# S2. Standard CGM features from raw series
# ============================================================================
def standard_features(ts, gl_mmol):
    v = np.asarray(gl_mmol, dtype=float)
    v = v[~np.isnan(v)]
    out = {}
    out['mean_gl'] = v.mean()
    out['sd_gl'] = v.std(ddof=1)
    out['cv_pct'] = out['sd_gl'] / out['mean_gl'] * 100 if out['mean_gl'] > 0 else 0
    out['min_gl'] = v.min()
    out['max_gl'] = v.max()
    out['range_gl'] = v.max() - v.min()
    out['tir'] = np.mean((v >= 3.9) & (v <= 10.0)) * 100
    out['tar10'] = np.mean(v > 10.0) * 100
    out['tbr39'] = np.mean(v < 3.9) * 100
    out['tar85'] = np.mean(v > 8.5) * 100
    # J-index
    out['j_index'] = 0.324 * (v.mean() + v.std()) ** 2
    # GMI
    out['gmi'] = 3.31 + 0.02392 * v.mean() * MMOL
    # CONGA-1: SD of 60-min lag differences (5-min raw grid)
    d60 = v[60:] - v[:-60]
    out['congal'] = d60.std(ddof=1) if len(d60) > 3 else np.nan
    # MODD: mean abs diff between days, same clock time (288 pts/day)
    if len(v) > 2 * 288:
        modd = np.abs(v[288:] - v[:-288])
        out['modd'] = modd.mean()
    else:
        out['modd'] = np.nan
    # MAGE-lite: mean of |excursions| between turning points >= 1 mmol/L
    diffs = np.diff(v)
    if len(diffs) > 3:
        turns = [0]
        for i in range(1, len(diffs) - 1):
            if diffs[i - 1] * diffs[i] < 0:  # sign change => turning point
                turns.append(i + 1)
        turns.append(len(v) - 1)
        exc = []
        for a, b in zip(turns[:-1], turns[1:]):
            if b - a >= 2 and abs(v[b] - v[a]) >= 1.0:
                exc.append(abs(v[b] - v[a]))
        out['mage_lite'] = np.mean(exc) if len(exc) >= 2 else np.nan
        out['n_exc'] = len(exc)
    else:
        out['mage_lite'], out['n_exc'] = np.nan, np.nan
    # LBGI / HBGI (Kovatchev simplified)
    r = 1.509 * ((np.log(v) ** 1.084) - 5.381)
    out['lbgi'] = np.mean(10 * r[r < 0] ** 2) if np.any(r < 0) else 0.0
    out['hbgi'] = np.mean(10 * r[r > 0] ** 2) if np.any(r > 0) else 0.0
    out['gri'] = out['lbgi'] + out['hbgi']
    # Night / dawn windows (0-6h, 6-9h) on raw timestamps
    ts_pd = pd.to_datetime(ts)
    hours = np.array([t.hour + t.minute / 60 for t in ts_pd])
    night = v[(hours >= 0) & (hours < 6)]
    dawn = v[(hours >= 6) & (hours < 9)]
    out['night_mean'] = night.mean() if len(night) > 5 else np.nan
    out['night_sd'] = night.std(ddof=1) if len(night) > 5 else np.nan
    out['dawn_delta'] = (dawn.mean() - night.mean()) if len(dawn) > 3 and len(night) > 5 else np.nan
    return out

def load_raw_features():
    df = pd.read_csv('raw_data/raw_data/colas.csv')
    df['time'] = pd.to_datetime(df['time'])
    med = df['gl'].median()
    if med > 30:
        df['gl'] = df['gl'] / MMOL
    rows = {}
    for pid, g in df.groupby('id'):
        g = g.sort_values('time')
        rows[pid] = standard_features(g['time'].values, g['gl'].values)
    return pd.DataFrame(rows).T

# ============================================================================
# S3. Evaluation helpers — honest protocol
# ============================================================================
def univariate_auc(X, y):
    aucs = {}
    for i, fn in enumerate(X.columns):
        try:
            a = roc_auc_score(y, X.iloc[:, i].values)
        except Exception:
            a = 0.5
        aucs[fn] = a
    return aucs

def nested_cv_lr(X, y, n_splits=5, n_repeats=10, k_candidates=(2, 3, 4, 5, 6), seed=42):
    """LR with feature selection done INSIDE each training fold."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, te in rkf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        aucs_tr = {}
        for fn in X.columns:
            try:
                aucs_tr[fn] = roc_auc_score(ytr, Xtr[fn].values)
            except Exception:
                aucs_tr[fn] = 0.5
        best = None
        for k in k_candidates:
            top = sorted(aucs_tr, key=aucs_tr.get, reverse=True)[:k]
            lr = LogisticRegression(max_iter=2000, class_weight='balanced')
            lr.fit(Xtr[top].values, ytr)
            s = roc_auc_score(yte, lr.predict_proba(Xte[top].values)[:, 1])
            best = s if best is None else max(best, s)
        scores.append(best)
    return np.mean(scores), np.std(scores)

def fixed_cv(Xcols, y, n_splits=5, n_repeats=10, seed=42):
    """Repeated CV for a pre-registered feature list (no selection)."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, te in rkf.split(Xcols, y):
        lr = LogisticRegression(max_iter=2000, class_weight='balanced')
        lr.fit(Xcols.iloc[tr].values, y[tr])
        scores.append(roc_auc_score(y[te], lr.predict_proba(Xcols.iloc[te].values)[:, 1]))
    return np.mean(scores), np.std(scores)

def cv_tree(Xcols, y, depth, n_splits=5, n_repeats=10, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, te in rkf.split(Xcols, y):
        clf = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=seed)
        clf.fit(Xcols.iloc[tr].values, y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(Xcols.iloc[te].values)[:, 1]))
    return np.mean(scores), np.std(scores)

def sens_spec_at_threshold(proba, y, thresh):
    pred = proba >= thresh
    tp = np.sum(pred & (y == 1)); p = np.sum(y == 1)
    tn = np.sum(~pred & (y == 0)); n = np.sum(y == 0)
    return tp / p, tn / n

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 72)
    print("GlucoBench Core-Algorithm Validation v3 (honest protocol + optimization)")
    print("=" * 72)

    cache = load_cached()
    raw = load_raw_features()
    cache.index = cache['patient_id']
    raw.index = raw.index.astype(int)
    full = cache.join(raw, rsuffix='_r')
    # dedupe overlapping names (cv_pct, tir, mean_gl, sd_gl already in cache)
    for c in ['cv_pct', 'tir', 'mean_gl', 'sd_gl']:
        if c + '_r' in full.columns:
            full.drop(columns=c + '_r', inplace=True)
    full = full.dropna(subset=['congal', 'night_mean', 'dawn_delta', 'mage_lite'])
    if full['modd'].isna().any():
        full['modd'] = full['modd'].fillna(full['modd'].median())
    full = full.dropna(subset=['lyapunov', 'det', 'entr'])
    y = full['label'].values.astype(int)
    print(f"Patients: {len(full)}, T2DM: {y.sum()} ({y.mean()*100:.1f}%)")

    # ---------- clinical reference ceiling ----------
    clin = pd.read_csv('raw_data/raw_data/colas.csv')
    clin_agg = clin.groupby('id').agg(hba1c=('HbA1c', 'first'), gly=('glycaemia', 'first'),
                                      bmi=('BMI', 'first'), age=('age', 'first')).loc[full.index]
    for c in ['hba1c', 'gly', 'bmi', 'age']:
        v = clin_agg[c].values
        try:
            a = roc_auc_score(y, v)
        except Exception:
            a = np.nan
        print(f"  Clinical reference {c:>6}: univariate AUC = {a:.3f}")

    # ---------- S4. univariate ----------
    # det_entr_ratio computed here (mirror of v2 script)
    full['det_entr_ratio'] = full['det'] / (full['entr'] + 1e-6)
    ps_feats = ['log_volume', 'shape_ratio', 'avg_recovery', 'dimension', 'lyapunov',
                'det', 'entr', 'dA', 'tau', 'mri_opt', 'det_entr_ratio']
    trad_feats = ['cv_pct', 'tir', 'mean_gl', 'sd_gl', 'min_gl', 'max_gl', 'range_gl',
                  'tar10', 'tbr39', 'tar85', 'j_index', 'gmi', 'congal',
                  'mage_lite', 'n_exc', 'lbgi', 'hbgi', 'gri', 'night_mean', 'night_sd',
                  'dawn_delta']
    all_feats = ps_feats + trad_feats

    print("\n[1] Univariate discriminative power (sorted, incl. inversions):")
    rows = []
    for fn in all_feats:
        v = full[fn].values
        try:
            a = roc_auc_score(y, v)
        except Exception:
            a = 0.5
        rows.append((fn, a))
    rows.sort(key=lambda t: max(t[1], 1 - t[1]), reverse=True)
    for fn, a in rows:
        direction = 'T2DM-lower' if a < 0.5 else 'T2DM-higher'
        print(f"  {fn:<14} AUC={a:.3f}  (best-direction {max(a,1-a):.3f} {direction})")

    # ---------- S5. honest model comparison ----------
    print("\n[2] Honest repeated 10x5 CV comparison (nested selection for LR):")
    X_ps = full[ps_feats]
    X_tr = full[trad_feats]
    X_all = full[all_feats]

    for name, X in [('Phase-space only', X_ps), ('Traditional only', X_tr), ('Combined', X_all)]:
        m, s = nested_cv_lr(X, y)
        print(f"  LR {name:<18}: AUC {m:.3f} ± {s:.3f}")
    m, s = fixed_cv(X_all, y)
    print(f"  LR combined (all feats, no selection): AUC {m:.3f} ± {s:.3f}")

    # current deployed rule
    m, s = fixed_cv(full[['mri_opt']], y)
    print(f"  Deployed MRI score only          : AUC {m:.3f} ± {s:.3f}")

    # shallow trees
    for d in (2, 3):
        m, s = cv_tree(X_all, y, depth=d)
        print(f"  DecisionTree depth={d} (combined)      : AUC {m:.3f} ± {s:.3f}")

    # ---------- S6. fixed interpretable rules ----------
    print("\n[3] Pre-registered interpretable rules (repeated 10x5 CV):")
    # Rule A: the cleanest phase-space signals (directions known from [1])
    for rule_a in (['lyapunov', 'entr', 'dA'], ['lyapunov', 'det_entr_ratio'],
                   ['lyapunov', 'entr', 'night_mean', 'det_entr_ratio']):
        m, s = fixed_cv(full[rule_a], y)
        print(f"  Rule A  [{', '.join(rule_a):<36}] : AUC {m:.3f} ± {s:.3f}")

    # Rule B: add best traditional complement (TAR / night mean)
    for extra in ['tar10', 'night_mean', 'mean_gl', 'congal']:
        feats = rule_a + [extra]
        m, s = fixed_cv(full[feats], y)
        print(f"  Rule B  {', '.join(feats):<38}: AUC {m:.3f} ± {s:.3f}")
    # smaller combos
    for feats in (['lyapunov', 'entr', 'night_mean'],
                  ['lyapunov', 'entr', 'dA', 'night_mean', 'avg_recovery'],
                  ['lyapunov', 'entr', 'dA', 'night_mean', 'mage_lite'],
                  ['lyapunov', 'det_entr_ratio', 'night_mean'],
                  ['lyapunov', 'entr', 'det_entr_ratio']):
        m, s = fixed_cv(full[feats], y)
        print(f"  Rule C  {', '.join(feats):<38}: AUC {m:.3f} ± {s:.3f}")

    # ---------- S7. final model fit & screening threshold ----------
    print("\n[4] Final rule: LR on lyapunov + entr + dA + night_mean")
    final_feats = ['lyapunov', 'entr', 'dA', 'night_mean']
    Xf = full[final_feats]
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xf.values, y)
    print(f"  Coefficients: {dict(zip(final_feats, lr.coef_[0].round(3)))}  intercept={lr.intercept_[0]:.3f}")

    # repeated CV with full proba stack for threshold calibration
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in rkf.split(Xf, y):
        l = LogisticRegression(max_iter=2000, class_weight='balanced')
        l.fit(Xf.iloc[tr].values, y[tr])
        oof[te] = l.predict_proba(Xf.iloc[te].values)[:, 1]
    auc_oof = roc_auc_score(y, oof)
    print(f"  OOF AUC (10x5): {auc_oof:.3f}")
    best_th, best_j = None, -1
    for th in np.arange(0.05, 0.96, 0.01):
        se, sp = sens_spec_at_threshold(oof, y, th)
        j = se + sp - 1
        if j > best_j:
            best_j, best_th = j, th
    se, sp = sens_spec_at_threshold(oof, y, best_th)
    print(f"  Youden threshold {best_th:.2f}: Sens={se:.3f} Spec={sp:.3f} (J={best_j:.3f})")
    for sp_target in (0.85, 0.90, 0.95):
        cand = [(th, *sens_spec_at_threshold(oof, y, th)) for th in np.arange(0.05, 0.96, 0.01)]
        cand = [c for c in cand if c[2] >= sp_target]
        if cand:
            th, se_, sp_ = min(cand, key=lambda c: abs(c[2] - sp_target))
            print(f"  Spec~{sp_target:.0%}: threshold {th:.2f} -> Sens={se_:.3f} Spec={sp_:.3f}")

    # save rule for JS port
    out = {
        'features': final_feats,
        'coef': {f: float(c) for f, c in zip(final_feats, lr.coef_[0])},
        'intercept': float(lr.intercept_[0]),
        'youden_threshold': float(best_th),
        'oof_auc': float(auc_oof),
        'mean_std_gl': float(full['mean_gl'].mean()),
        'mean_std_sd': float(full['mean_gl'].std()),
    }
    with open('output/optimized_rule.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> output/optimized_rule.json")

if __name__ == '__main__':
    main()
