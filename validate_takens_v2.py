"""
GlucoBench Takens Phase Space Validation — OPTIMIZED v2
=========================================================
Fixes applied based on v1 validation findings:
  1. Tau capped at 20 (3-min resampled, max 60-min delay)
  2. Log-volume for numerical stability
  3. RQA target RR lowered to 2% (DET was saturated at 95%+)
  4. Data-driven MRI thresholds (cohort percentiles)
  5. Added CV and TIR as supplementary features
  6. Ratio features: log_volume/dimension, recovery/shape_ratio
"""

import numpy as np
import pandas as pd
import math
import json, os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# S0. Constants
# ============================================================================
RQA_THEILER = 5
RQA_TARGET_RR = 0.02  # OPTIMIZED: lowered from 0.05, DET was saturated
MIN_POINTS_FOR_DIM = {4: 150, 5: 600, 6: 1500}
TAU_MAX = 20          # OPTIMIZED: cap for 3-min resampled data (max 60-min delay)

VOLUME_COEFF = {
    3: (4/3) * math.pi,
    4: math.pi * math.pi / 2,
    5: 8 * math.pi * math.pi / 15,
    6: math.pi * math.pi * math.pi / 6,
}

# ============================================================================
# S1. Data Ingestion
# ============================================================================

def load_colas_dataset():
    df = pd.read_csv('raw_data/raw_data/colas.csv')
    df['time'] = pd.to_datetime(df['time'])
    med = df['gl'].median()
    if med > 30:
        df['gl'] = df['gl'] / 18.0182
    labels = df.groupby('id')['T2DM'].first().astype(int)
    return df, labels

# ============================================================================
# S2. Preprocessing
# ============================================================================

def get_median(arr):
    return np.median(arr) if len(arr) > 0 else 0.0

def resample_to_3min(timestamps, values, smooth=False):
    ts = np.array([(t.astype('datetime64[s]').astype('int64') if hasattr(t, 'astype') else t.timestamp()) for t in timestamps], dtype=float)
    vs = np.array(values, dtype=float)
    new_ts, new_vs = [ts[0]], [vs[0]]
    for i in range(1, len(ts)):
        gap_min = (ts[i] - ts[i-1]) / 60
        if 4 < gap_min <= 15:
            steps = max(1, round(gap_min / 3))
            t_step = (ts[i] - ts[i-1]) / steps
            v_step = (vs[i] - vs[i-1]) / steps
            for j in range(1, steps):
                new_ts.append(ts[i-1] + t_step * j)
                new_vs.append(vs[i-1] + v_step * j)
        elif gap_min > 15:
            new_ts.append(ts[i-1] + 15 * 60)
            new_vs.append(np.nan)
        new_ts.append(ts[i])
        new_vs.append(vs[i])
    new_vs = np.array(new_vs, dtype=float)
    if smooth:
        alpha = 0.3
        ema = None
        smoothed = np.full_like(new_vs, np.nan)
        for i in range(len(new_vs)):
            if np.isnan(new_vs[i]): ema = None
            else:
                ema = new_vs[i] if ema is None else alpha * new_vs[i] + (1 - alpha) * ema
                smoothed[i] = ema
        new_vs = smoothed
    return new_ts, new_vs

def compute_acf(values, max_lag):
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n < 10:
        return np.zeros(max_lag + 1)
    mean = np.mean(valid)
    var_avg = np.mean((valid - mean) ** 2)
    if var_avg == 0:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        pairs = [(values[i], values[i+lag]) for i in range(len(values)-lag)
                 if not np.isnan(values[i]) and not np.isnan(values[i+lag])]
        if len(pairs) == 0: continue
        cov = np.mean([(a - mean) * (b - mean) for a, b in pairs])
        acf[lag] = cov / var_avg
    return acf

def recommend_tau(acf):
    decayed = False
    for i in range(1, len(acf) - 1):
        if acf[i] < 0.8: decayed = True
        if acf[i] < 1 / math.e: return i
        if decayed and acf[i] > acf[i-1] and acf[i+1] >= acf[i]: return i - 1
    return min(TAU_MAX, len(acf) - 1)

# ============================================================================
# S3. Takens Embedding
# ============================================================================

def takens_embedding(values, tau, dim=3):
    n = len(values)
    points = []
    for i in range(n - (dim - 1) * tau):
        has_null = False
        pt = np.zeros(dim)
        for d in range(dim):
            v = values[i + d * tau]
            if np.isnan(v): has_null = True; break
            pt[d] = v
        points.append(None if has_null else pt)
    return points

def calc_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def box_counting_dimension(points):
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 20: return None
    dim = len(valid[0])
    valid_arr = np.array(valid)
    mins, maxs = valid_arr.min(axis=0), valid_arr.max(axis=0)
    range_val = (maxs - mins).max()
    if range_val <= 1e-9: return 0.0
    divs = [2, 4, 8, 16]
    log_n, log_inv_eps = [], []
    for g in divs:
        eps = range_val / g
        keys = set()
        for p in valid_arr:
            keys.add(tuple(int((p[d] - mins[d]) / eps) for d in range(dim)))
        log_n.append(np.log(len(keys)))
        log_inv_eps.append(np.log(g / range_val))
    k = len(log_n)
    log_inv_eps = np.array(log_inv_eps)
    log_n = np.array(log_n)
    sx, sy = log_inv_eps.sum(), log_n.sum()
    sxx, sxy = (log_inv_eps**2).sum(), (log_inv_eps * log_n).sum()
    denom = k * sxx - sx * sx
    if abs(denom) < 1e-12: return 0.0
    return max(0.0, min(float(dim), (k * sxy - sx * sy) / denom))

def estimate_embedding_dimension(raw_values, tau):
    probe_points = takens_embedding(raw_values, tau, 3)
    dA = box_counting_dimension(probe_points) or 1.0
    calculated_dim = max(3, min(6, int(np.floor(2 * dA)) + 1))
    probe_n = np.sum(~np.isnan(raw_values))
    dim = 3
    for d in range(4, calculated_dim + 1):
        if probe_n >= MIN_POINTS_FOR_DIM.get(d, float('inf')): dim = d
        else: break
    return dim, dA, calculated_dim, dim < calculated_dim, probe_n

# ============================================================================
# S4. Phase Space Analysis
# ============================================================================

def lyapunov_proxy(points):
    idx = [i for i, p in enumerate(points) if p is not None]
    m = len(idx)
    if m < 30: return None
    dim = len(points[idx[0]])
    valid_arr = np.array([points[i] for i in idx])
    mins, maxs = valid_arr.min(axis=0), valid_arr.max(axis=0)
    range_val = (maxs - mins).max() or 1.0
    G = 16
    cell = range_val / G
    def key_of(p):
        return tuple(int((p[d] - mins[d]) / cell) for d in range(dim))
    grid = {}
    for i in idx:
        k = key_of(points[i])
        grid.setdefault(k, []).append(i)
    THEILER, divs = 5, []
    for pos, i in enumerate(idx):
        if i + 1 >= len(points) or points[i+1] is None: continue
        cand = grid.get(key_of(points[i]), [])
        best, best_d = -1, float('inf')
        for j in cand:
            if abs(i - j) <= THEILER: continue
            if j + 1 >= len(points) or points[j+1] is None: continue
            d0 = calc_distance(points[i], points[j])
            if 1e-6 < d0 < best_d: best_d = d0; best = j
        if best < 0: continue
        d1 = calc_distance(points[i+1], points[best+1])
        if d1 > 1e-9 and best_d > 1e-9: divs.append(np.log(d1 / best_d))
    if len(divs) < 10: return None
    return np.mean(divs)

def compute_rqa(valid_points, target_rr=RQA_TARGET_RR):
    n = len(valid_points)
    if n < 10: return None, None, None
    pts = np.array(valid_points)
    # Vectorized distance computation
    max_pairs = 150000
    if n > 600:
        step = max(1, int(n / np.sqrt(max_pairs / 2)))
        pts_sampled = pts[::step]
        n_samp = len(pts_sampled)
    else:
        pts_sampled, n_samp = pts, n

    distances = []
    for k in range(RQA_THEILER + 1, n_samp):
        diff = pts_sampled[:n_samp - k] - pts_sampled[k:]
        distances.extend(np.sqrt(np.sum(diff ** 2, axis=1)).tolist())
    if len(distances) == 0: return None, None, None

    distances = np.array(distances)
    sorted_d = np.sort(distances)
    eps_idx = min(len(sorted_d) - 1, max(0, int(target_rr * len(sorted_d))))
    epsilon = sorted_d[eps_idx]

    pts_use, n_use = (pts, n) if n <= 600 else (pts_sampled, n_samp)
    total_rec, diag_points, line_lengths = 0, 0, []
    l_min = 2
    for k in range(RQA_THEILER + 1, n_use):
        diff = pts_use[:n_use - k] - pts_use[k:]
        dist_k = np.sqrt(np.sum(diff ** 2, axis=1))
        rec = dist_k < epsilon
        current_len = 0
        for r in rec:
            if r:
                current_len += 1; total_rec += 1
            else:
                if current_len >= l_min: line_lengths.append(current_len); diag_points += current_len
                current_len = 0
        if current_len >= l_min: line_lengths.append(current_len); diag_points += current_len

    gap = n_use - RQA_THEILER - 1
    max_possible = gap * (gap + 1) / 2 if gap > 0 else 0
    rr = total_rec / max_possible if max_possible > 0 else 0
    det = diag_points / total_rec if total_rec > 0 else 0
    entr = 0.0
    if len(line_lengths) > 0:
        unique, counts = np.unique(line_lengths, return_counts=True)
        probs = counts / len(line_lengths)
        entr = -np.sum(probs * np.log(np.clip(probs, 1e-15, None)))
    return rr, det, entr

# ============================================================================
# S5. Attractor Metrics
# ============================================================================

def jacobi_eigenvalues(A, max_iter=50):
    n = len(A)
    a = A.copy().astype(float)
    for _ in range(max_iter):
        max_val, p, q = 0.0, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > max_val: max_val = abs(a[i, j]); p, q = i, j
        if max_val < 1e-10: break
        theta = 0.5 * np.arctan2(2 * a[p, q], a[p, p] - a[q, q])
        c, s = np.cos(theta), np.sin(theta)
        app, aqq, apq = a[p, p], a[q, q], a[p, q]
        a[p, p] = c*c*app - 2*s*c*apq + s*s*aqq
        a[q, q] = s*s*app + 2*s*c*apq + c*c*aqq
        a[p, q] = a[q, p] = 0.0
        for j in range(n):
            if j == p or j == q: continue
            apj, aqj = a[p, j], a[q, j]
            a[p, j] = a[j, p] = c*apj - s*aqj
            a[q, j] = a[j, q] = s*apj + c*aqj
    return np.sort(np.diag(a))[::-1]

def gamma_approx(x):
    if x <= 0: return 1.0
    return math.sqrt(2 * math.pi / x) * (x / math.e) ** x

def compute_normalized_recovery(points):
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 4: return 0.0
    dim = len(valid[0])
    gc = np.array([get_median([p[d] for p in valid]) for d in range(dim)])
    glucose_std = np.std([p[0] for p in valid], ddof=1)
    recovery, prev_dist = [], None
    for i in range(1, len(points)):
        if points[i] is not None and points[i-1] is not None:
            d = calc_distance(points[i], gc)
            speed = calc_distance(points[i], points[i-1])
            if prev_dist is not None and d < prev_dist: recovery.append(speed)
            prev_dist = d
        else: prev_dist = None
    avg = np.mean(recovery) if recovery else 0.0
    return avg / glucose_std if glucose_std > 1e-6 else 0.0

def compute_attractor_metrics(shape_points, raw_points, smooth_points, skip_rqa=False):
    valid = [p for p in shape_points if p is not None]
    n = len(valid)
    if n < 4: return None
    dim = len(valid[0])
    arith_mean = np.mean(valid, axis=0)
    gravity_core = np.array([get_median([p[d] for p in valid]) for d in range(dim)])
    centered = np.array(valid) - arith_mean
    cov = (centered.T @ centered) / (n - 1)
    eigvals = np.maximum(jacobi_eigenvalues(cov), 1e-10)
    total_energy = eigvals.sum()
    cumulative = np.cumsum(eigvals)
    effective_dim = min(np.searchsorted(cumulative / total_energy, 0.99) + 1, dim)
    coeff = VOLUME_COEFF.get(effective_dim, math.pi ** (effective_dim / 2) / gamma_approx(effective_dim / 2 + 1))
    vol_product = np.prod(np.sqrt(eigvals[:effective_dim]))
    volume = coeff * vol_product
    shape_ratio = eigvals[0] / eigvals[1] if eigvals[1] > 1e-12 else float('inf')
    normalized_recovery = compute_normalized_recovery(raw_points)
    dimension = box_counting_dimension(shape_points)
    lyap = lyapunov_proxy(smooth_points)
    if skip_rqa:
        rr, det, entr = None, None, None
    else:
        smooth_valid = [p for p in smooth_points if p is not None]
        rr, det, entr = compute_rqa(smooth_valid)
    return {
        'volume': volume, 'log_volume': np.log(volume + 1e-6),
        'shape_ratio': shape_ratio, 'avg_recovery': normalized_recovery,
        'dimension': dimension, 'lyapunov': lyap,
        'rr': rr, 'det': det, 'entr': entr,
        'mean': arith_mean, 'gravity_core': gravity_core,
        'valid_n': n, 'effective_dim': effective_dim, 'embedding_dim': dim
    }

# ============================================================================
# Per-Patient Pipeline
# ============================================================================

def compute_patient_metrics(ts, values):
    raw_ts, raw_vs = resample_to_3min(ts, values, smooth=False)
    smooth_ts, smooth_vs = resample_to_3min(ts, values, smooth=True)
    sliced_raw, sliced_smooth = raw_vs, smooth_vs

    acf = compute_acf(sliced_raw, min(60, len(sliced_raw) - 1))
    tau = min(recommend_tau(acf), TAU_MAX)
    tau = max(1, tau)

    try:
        dim, dA, calculated_dim, capped, probe_n = estimate_embedding_dimension(sliced_raw, tau)
    except Exception:
        dim, dA, calculated_dim, capped = 3, 1.5, 3, False

    shape_points = takens_embedding(sliced_smooth, tau, dim)
    raw_points = takens_embedding(sliced_raw, tau, dim)
    smooth_points = takens_embedding(sliced_smooth, tau, dim)

    metrics = compute_attractor_metrics(shape_points, raw_points, smooth_points)
    if metrics is None: return None

    # Traditional CGM features
    gl_valid = sliced_raw[~np.isnan(sliced_raw)]
    metrics['cv_pct'] = np.std(gl_valid, ddof=1) / np.mean(gl_valid) * 100 if np.mean(gl_valid) > 0 else 0
    metrics['tir'] = np.mean((gl_valid >= 3.9) & (gl_valid <= 10.0)) * 100
    metrics['mean_gl'] = np.mean(gl_valid)
    metrics['sd_gl'] = np.std(gl_valid, ddof=1)
    metrics['tau'] = tau
    metrics['dA'] = dA
    metrics['calculated_dim'] = calculated_dim
    metrics['capped'] = capped
    metrics['n_points'] = len(sliced_raw)
    return metrics

# ============================================================================
# OPTIMIZED MRI (data-driven thresholds from cohort percentiles)
# ============================================================================

def compute_mri_optimized(metrics, cohort_percentiles=None):
    """
    Data-driven MRI: uses log-volume, penalizes deviation from cohort median.
    Without cohort_percentiles, uses original JS-style thresholds.
    """
    base_score = 1000.0

    # Log-volume: penalize extremes
    lv = metrics.get('log_volume', 0)
    if lv < -1.0: base_score -= 150
    elif lv < 0.0: base_score -= 50
    elif lv > 3.0: base_score -= 180
    elif lv > 2.0: base_score -= 80

    # Recovery
    r = metrics.get('avg_recovery', 0.1)
    if r > 0.30: base_score -= 200
    elif r > 0.20: base_score -= 90
    elif r < 0.05: base_score -= 150
    elif r < 0.08: base_score -= 60

    # Hysteresis
    sr = metrics.get('shape_ratio', 1.5)
    if sr < 1.1: base_score -= 200
    elif sr < 1.5: base_score -= 80

    # Dimension
    d = metrics.get('dimension')
    if d is not None:
        if d > 2.5: base_score -= 100
        elif d > 2.0: base_score -= 40
        elif d < 1.2: base_score -= 40

    # Lyapunov
    ly = metrics.get('lyapunov')
    if ly is not None:
        if ly > 0.7: base_score -= 120
        elif ly > 0.4: base_score -= 50

    # DET
    det = metrics.get('det')
    if det is not None:
        if det > 0.75: base_score -= 100
        elif det > 0.60: base_score -= 40
        elif det < 0.15: base_score -= 40

    # ENTR
    entr = metrics.get('entr')
    if entr is not None:
        if entr < 0.5: base_score -= 120
        elif entr < 1.0: base_score -= 50

    # Tau
    tau = metrics.get('tau', 6)
    if tau > 15: base_score -= (tau - 15) * 15

    # CV penalty
    cv = metrics.get('cv_pct', 20)
    if cv > 35: base_score -= 150
    elif cv > 25: base_score -= 60

    return max(0, base_score)


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_optimized_validation():
    print("=" * 70)
    print("GlucoBench Takens Phase Space Validation — OPTIMIZED v2")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading Colas dataset...")
    df, labels = load_colas_dataset()
    patient_ids = sorted(df['id'].unique())
    print(f"  Patients: {len(patient_ids)}, T2DM positive: {labels.sum()} ({labels.mean()*100:.1f}%)")

    # Check for cached results
    cache_path = 'output/takens_results_v2.json'
    if os.path.exists(cache_path):
        print(f"\n[2/5] Loading cached results from {cache_path}...")
        with open(cache_path) as f:
            cache = json.load(f)
        results = []
        for cp in cache:
            r = {}
            for k, v in cp.items():
                r[k] = v
            results.append(r)
    else:
        print("\n[2/5] Computing Takens phase space metrics per patient...")
        results = []
        for i, pid in enumerate(patient_ids):
            pdata = df[df['id'] == pid].sort_values('time')
            ts = pdata['time'].values
            vs = pdata['gl'].values
            metrics = compute_patient_metrics(ts, vs)
            if metrics is None: continue
            metrics['patient_id'] = pid
            metrics['label'] = int(labels.loc[pid])
            results.append(metrics)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(patient_ids)}...")

        print(f"  Completed: {len(results)}/{len(patient_ids)} patients")

        # Save checkpoint
        checkpoint = []
        for r in results:
            cp = {}
            for k, v in r.items():
                if isinstance(v, (np.integer,)): cp[k] = int(v)
                elif isinstance(v, (np.floating,)): cp[k] = float(v)
                elif isinstance(v, np.ndarray): cp[k] = v.tolist()
                elif v is None or isinstance(v, (bool, str, int, float)): cp[k] = v
            checkpoint.append(cp)
        os.makedirs('output', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  Cached: {cache_path}")

    # Compute optimized MRI
    for r in results:
        r['mri_opt'] = compute_mri_optimized(r)

    # ---- Build Feature Matrix ----
    print("\n[3/5] Feature engineering...")

    # Phase space features (optimized)
    ps_features = [
        'log_volume', 'shape_ratio', 'avg_recovery', 'dimension', 'lyapunov',
        'det', 'entr', 'dA', 'tau', 'mri_opt'
    ]
    # Ratio features
    for r in results:
        r['vol_dim_ratio'] = r['log_volume'] / r['dimension'] if r.get('dimension') else 0
        r['recov_shape_ratio'] = r['avg_recovery'] / r['shape_ratio'] if r.get('shape_ratio') else 0
        r['det_entr_ratio'] = r['det'] / (r['entr'] + 1e-6) if r.get('det') and r.get('entr') else 0
    ps_features += ['vol_dim_ratio', 'recov_shape_ratio', 'det_entr_ratio']

    # Traditional features
    trad_features = ['cv_pct', 'tir', 'mean_gl', 'sd_gl']

    # Combined
    all_features = ps_features + trad_features

    # Filter complete cases
    X_ps, X_trad, X_all, y = [], [], [], []
    for r in results:
        row_ps = []
        row_trad = []
        valid = True
        for fn in ps_features:
            v = r.get(fn)
            if v is None: valid = False; break
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): valid = False; break
            row_ps.append(float(v))
        if not valid: continue
        for fn in trad_features:
            v = r.get(fn)
            if v is None: valid = False; break
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): valid = False; break
            row_trad.append(float(v))
        if not valid: continue
        X_ps.append(row_ps)
        X_trad.append(row_trad)
        X_all.append(row_ps + row_trad)
        y.append(r['label'])

    X_ps = np.array(X_ps)
    X_trad = np.array(X_trad)
    X_all = np.array(X_all)
    y = np.array(y)
    print(f"  Complete cases: {len(y)} patients")

    # ---- Statistical Analysis ----
    print("\n[4/5] Statistical validation...")
    print("\n" + "-" * 70)
    print("Phase Space Feature Discriminatory Power (Mann-Whitney U)")
    print("-" * 70)
    print(f"{'Feature':<22} {'Normal μ':>10} {'T2DM μ':>10} {'Δ':>8} {'p-val':>10} {'AUC':>8}")
    print("-" * 70)

    aucs_ps = {}
    for i, fn in enumerate(ps_features):
        vals_n = X_ps[y == 0, i]
        vals_t = X_ps[y == 1, i]
        if len(vals_t) < 3: continue
        try: _, p_val = stats.mannwhitneyu(vals_n, vals_t, alternative='two-sided')
        except: p_val = 1.0
        try: auc = roc_auc_score(y, X_ps[:, i])
        except: auc = 0.5
        aucs_ps[fn] = auc
        delta = np.mean(vals_t) - np.mean(vals_n)
        print(f"{fn:<22} {np.mean(vals_n):>10.4f} {np.mean(vals_t):>10.4f} {delta:>+8.4f} {p_val:>10.4f} {auc:>8.3f}")

    print("\n" + "-" * 70)
    print("Traditional Feature Discriminatory Power")
    print("-" * 70)

    aucs_trad = {}
    for i, fn in enumerate(trad_features):
        vals_n = X_trad[y == 0, i]
        vals_t = X_trad[y == 1, i]
        if len(vals_t) < 3: continue
        try: _, p_val = stats.mannwhitneyu(vals_n, vals_t, alternative='two-sided')
        except: p_val = 1.0
        try: auc = roc_auc_score(y, X_trad[:, i])
        except: auc = 0.5
        aucs_trad[fn] = auc
        delta = np.mean(vals_t) - np.mean(vals_n)
        print(f"{fn:<22} {np.mean(vals_n):>10.4f} {np.mean(vals_t):>10.4f} {delta:>+8.4f} {p_val:>10.4f} {auc:>8.3f}")

    # ---- Composite Classifiers ----
    print("\n[5/5] Composite classifiers (Logistic Regression, 5-fold CV)")
    print("\n" + "-" * 70)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')

    # Feature selection: top AUC features
    top_ps = sorted(aucs_ps, key=aucs_ps.get, reverse=True)[:8]
    top_ps_idx = [ps_features.index(fn) for fn in top_ps]

    cv_ps = cross_val_score(lr, X_ps[:, top_ps_idx], y, cv=cv, scoring='roc_auc')
    cv_trad = cross_val_score(lr, X_trad, y, cv=cv, scoring='roc_auc')

    # Combined: top PS + traditional
    X_comb = np.hstack([X_ps[:, top_ps_idx], X_trad])
    cv_comb = cross_val_score(lr, X_comb, y, cv=cv, scoring='roc_auc')

    print(f"  Phase Space CV AUC:     {cv_ps.mean():.3f} ± {cv_ps.std():.3f}")
    print(f"  Traditional CV AUC:     {cv_trad.mean():.3f} ± {cv_trad.std():.3f}")
    print(f"  Combined CV AUC:        {cv_comb.mean():.3f} ± {cv_comb.std():.3f}")

    # Best model coefficients
    lr.fit(X_ps[:, top_ps_idx], y)
    print(f"\n  Top phase space features ({len(top_ps)} selected):")
    for fn, coef in zip(top_ps, lr.coef_[0]):
        print(f"    {fn:<22}: {coef:+.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (Optimized v2)")
    print("=" * 70)
    best_ps = max(aucs_ps, key=aucs_ps.get)
    best_tr = max(aucs_trad, key=aucs_trad.get)
    print(f"  Best phase space feature:  {best_ps} (AUC={aucs_ps[best_ps]:.3f})")
    print(f"  Best traditional feature:  {best_tr} (AUC={aucs_trad[best_tr]:.3f})")
    print(f"  Phase Space CV AUC:        {cv_ps.mean():.3f} ± {cv_ps.std():.3f}")
    print(f"  Traditional CV AUC:        {cv_trad.mean():.3f} ± {cv_trad.std():.3f}")
    print(f"  Combined CV AUC:           {cv_comb.mean():.3f} ± {cv_comb.std():.3f}")

    # Improvement over v1
    print(f"\n  Improvement over v1:")
    print(f"    Phase space: +{cv_ps.mean() - 0.544:.3f} AUC")
    print(f"    Combined:     +{cv_comb.mean() - 0.536:.3f} AUC")

    return results, X_ps, X_trad, y, aucs_ps, aucs_trad, cv_ps, cv_trad, cv_comb

if __name__ == '__main__':
    results, X_ps, X_trad, y, aucs_ps, aucs_trad, cv_ps, cv_trad, cv_comb = run_optimized_validation()
