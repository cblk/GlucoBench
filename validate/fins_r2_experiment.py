#!/usr/bin/env python3
"""
FINS R² 提升实验 — 设计 v1
==========================
问题: 现行管线 5 吸引子特征对空腹胰岛素 LOO R² = -0.09 (纯噪声)。
实验假设:
  H1 聚合毁信号 — per-tau 展开 (20 特征) 保留尺度特异性
  H2 特征族太窄 — 引入标准 CGM 指标 (paper 衍生表) 与合并族
  H3 模型欠调 — alpha 网格内层选择 + 折内标准化 + log 目标变换 + Huber 稳健性
协议 (诚实, 沿用 validate_takens_v3 哲学):
  - LOO 为主指标 (n=53); 标准化/alpha 选择全部在训练折内完成
  - 每配置 500 次标签置换检验 (alpha=1 固定版, 保守)
  - 临床族 (Age/BMI/A1C/FBG) 为可移植性上限参照, 不移植
成功标准: LOO R² > 0.05 且置换 p < 0.05 → 管线一致重算特征并移植 index.html
"""
import json, sqlite3, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(42)
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# ---------------- 1. data assembly ----------------
feats = json.load(open('output/js_fins_features.json'))
con = sqlite3.connect('data1/pbio.2005143.s014.db')
clin = pd.read_sql('SELECT * FROM clinical', con)

rows = []
for f in feats:
    row = {'id': f['id'], **f['features']}
    for tau_s, vals in f['perTau'].items():
        for k, v in vals.items():
            row[f'{k}@{tau_s}'] = v
    rows.append(row)
df = pd.DataFrame(rows).merge(clin, left_on='id', right_on='userID')
df = df[df['insulin'].notna()].reset_index(drop=True)
y = df['insulin'].values.astype(float)
print(f"n={len(df)}  FINS: mean {y.mean():.2f} sd {y.std(ddof=1):.2f} "
      f"min {y.min():.1f} max {y.max():.1f} skew {pd.Series(y).skew():.2f}")

ATTR = ['volume', 'shapeRatio', 'avgRecovery', 'dimension', 'lyapunov']
TAU = [c for c in df.columns if '@' in c]
STD = ['mean_glucose', 'sd_glucose', 'range_glucose', 'min_glucose', 'max_glucose',
       'quartile.25_glucose', 'median_glucose', 'quartile.75_glucose', 'mean_slope', 'max_slope',
       'number_Random140', 'number_Random200', 'percent_below.80', 'percent_above.130',
       'se_glucose_mean', 'numGE', 'mage', 'j_index', 'IQR', 'modd', 'distance_traveled',
       'coef_variation', 'number_Random140_normByDays', 'number_Random200_normByDays',
       'numGE_normByDays', 'distance_traveled_normByDays', 'freq_low', 'freq_moderate', 'freq_severe']
CLIN = ['Age', 'BMI', 'A1C', 'FBG']

families = {
    'attr(5)': ATTR,
    'attr-per-tau(20)': TAU,
    'std-cgm': STD,
    'attr-tau+std': TAU + STD,
    'clinical-ceiling': CLIN,
}
Xmat = {}
for name, cols in families.items():
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    keep = [c for c in cols if X[c].notna().all() and X[c].std() > 0]
    Xmat[name] = X[keep].values.astype(float)
    print(f"  family {name:<18} features: {len(keep)}")

# ---------------- 2. univariate screen ----------------
print("\n=== univariate |r| vs FINS (top 15 of all candidates) ===")
all_cols = ATTR + TAU + STD
uni = []
for c in all_cols:
    v = pd.to_numeric(df[c], errors='coerce').values
    if np.isnan(v).any() or v.std() == 0:
        continue
    r = np.corrcoef(v, y)[0, 1]
    uni.append((abs(r), c, r))
for a, c, r in sorted(uni, reverse=True)[:15]:
    print(f"  |r|={a:.3f}  r={r:+.3f}  {c}")

# ---------------- 3. honest LOO evaluation ----------------
def fit_predict_loo(X, yy, alpha, scale=True, logy=False, select_k=None, robust=False):
    """LOO: scaler/selector fit on training fold only."""
    preds = np.zeros(len(yy))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        Xtr, ytr = X[tr], yy[tr]
        cols = np.arange(X.shape[1])
        if select_k is not None:
            rs = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                if Xtr[:, j].std() > 0:
                    rs[j] = abs(np.corrcoef(Xtr[:, j], ytr)[0, 1])
            rs[np.isnan(rs)] = 0
            cols = np.argsort(-rs)[:select_k]
        Xtr, Xte = Xtr[:, cols], X[te][:, cols]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = HuberRegressor(alpha=alpha, max_iter=2000) if robust else Ridge(alpha=alpha)
        m.fit(Xtr, ytr)
        preds[te] = m.predict(Xte)[0]
    if logy:
        preds = np.exp(preds)
    return preds

def loo_r2(X, yy, alpha, scale=True, logy=False, select_k=None, robust=False):
    return r2_score(y, fit_predict_loo(X, yy, alpha, scale, logy, select_k, robust))

def loo_r2_alpha_tuned(X, yy, scale=True, logy=False, select_k=None):
    """alpha chosen by inner LOO on the training fold (fully nested)."""
    preds = np.zeros(len(yy))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        Xtr, ytr = X[tr], yy[tr]
        cols = np.arange(X.shape[1])
        if select_k is not None:
            rs = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                if Xtr[:, j].std() > 0:
                    rs[j] = abs(np.corrcoef(Xtr[:, j], ytr)[0, 1])
            rs[np.isnan(rs)] = 0
            cols = np.argsort(-rs)[:select_k]
        Xtr, Xte = Xtr[:, cols], X[te][:, cols]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        best_a, best_mse = None, np.inf
        for a in ALPHAS:
            s = 0.0
            for tr2, te2 in LeaveOneOut().split(Xtr):
                s += (ytr[te2] - Ridge(alpha=a).fit(Xtr[tr2], ytr[tr2]).predict(Xtr[te2])[0]) ** 2
            if s < best_mse:
                best_mse, best_a = s, a
        preds[te] = Ridge(alpha=best_a).fit(Xtr, ytr).predict(Xte)[0]
    if logy:
        preds = np.exp(preds)
    return r2_score(y, preds)

print("\n=== LOO R² (nested, honest) ===")
print(f"{'family':<20} {'y':<6} {'alpha=1':>8} {'alpha-tuned':>11} {'robust':>8}")
results = {}
for name, X in Xmat.items():
    for logy in (False, True):
        tag = 'log(FINS)' if logy else 'FINS'
        yy = np.log(y) if logy else y
        r1 = loo_r2(X, yy, alpha=1.0, scale=True, logy=logy)
        rt = loo_r2_alpha_tuned(X, yy, scale=True, logy=logy)
        rr = loo_r2(X, yy, alpha=1.0, scale=True, logy=logy, robust=True)
        print(f"{name:<20} {tag:<6} {r1:>8.3f} {rt:>11.3f} {rr:>8.3f}")
        results[(name, logy)] = r1

# nested feature-selection variant for wide families
print("\n=== wide families with nested top-k selection (alpha=1, standardized) ===")
for name in ('attr-per-tau(20)', 'std-cgm', 'attr-tau+std'):
    X = Xmat[name]
    for k in (5, 10):
        r = loo_r2(X, y, alpha=1.0, scale=True, logy=False, select_k=k)
        print(f"  {name:<18} k={k:<3} LOO R² = {r:+.3f}")

# ---------------- 4. permutation tests (alpha=1, standardized, 500 perms) ----------------
print("\n=== permutation test (500 label shuffles, LOO, alpha=1, standardized) ===")
for (name, logy), obs in results.items():
    yy = np.log(y) if logy else y
    X = Xmat[name]
    rng = np.random.default_rng(42)
    null = []
    for _ in range(500):
        yp = rng.permutation(yy)
        null.append(loo_r2(X, yp, alpha=1.0, scale=True, logy=logy))
    p = float(np.mean(np.array(null) >= obs))
    print(f"  {name:<20} {'log' if logy else 'raw':<4} obs={obs:+.3f} "
          f"null mean={np.mean(null):+.3f} p95={np.percentile(null,95):+.3f} p={p:.3f}")
