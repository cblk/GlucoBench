"""
GlucoBench Takens Phase Space Validation Pipeline
===================================================
Mirrors the index.html JS algorithm in Python, computes phase space
metrics for every patient, and validates against clinical ground truth:
- Colas dataset: 208 patients, T2DM label
- Hall dataset: 57 patients, diagnosis + glucotype labels

Pipeline mirrors index.html v7.0 exactly:
  S1. Data ingestion (from CSV)
  S2. Preprocessing (resample 3min, EMA, ACF → tau)
  S3. Takens embedding (delay coordinates, SYC dimension estimation)
  S4. Phase analysis (box-counting dim, Lyapunov, RQA)
  S5. Attractor metrics (volume, hysteresis, recovery, core displacement)
  S6. Clinical score (MRI composite)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# S0. Constants (mirrors JS)
# ============================================================================
RQA_THEILER = 5
RQA_TARGET_RR = 0.05
MIN_POINTS_FOR_DIM = {4: 150, 5: 600, 6: 1500}

# Volume coefficients: V_d = π^(d/2) / Γ(d/2 + 1)
import math
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
    """Load Colas dataset, convert to mmol/L, return per-patient time series."""
    df = pd.read_csv('raw_data/raw_data/colas.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Convert mg/dL to mmol/L
    med = df['gl'].median()
    if med > 30:
        df['gl'] = df['gl'] / 18.0182

    # Get labels
    labels = df.groupby('id')['T2DM'].first().astype(int)
    return df, labels

def load_hall_dataset():
    """Load Hall dataset, convert to mmol/L, return per-patient time series."""
    df = pd.read_csv('raw_data/raw_data/hall.csv')
    df['time'] = pd.to_datetime(df['time'])

    med = df['gl'].median()
    if med > 30:
        df['gl'] = df['gl'] / 18.0182

    # diagnosis: 0=normal, 1=prediabetes, 2=diabetes
    # Binary: 0=normal, 1=abnormal (>=1)
    labels = df.groupby('id')['diagnosis'].first()
    labels_binary = (labels >= 1).astype(int)
    return df, labels_binary, labels

# ============================================================================
# S2. Preprocessing
# ============================================================================

def get_median(arr):
    if len(arr) == 0:
        return 0
    return np.median(arr)

def resample_to_3min(timestamps, values, smooth=False):
    """Resample to 3-minute grid. Mirrors JS resampleData()."""
    ts = np.array([(t.astype('datetime64[s]').astype('int64') if hasattr(t, 'astype') else t.timestamp()) for t in timestamps], dtype=float)
    vs = np.array(values, dtype=float)

    new_ts = [ts[0]]
    new_vs = [vs[0]]

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
        # EMA filter (α=0.3)
        alpha = 0.3
        ema = None
        smoothed = np.full_like(new_vs, np.nan)
        for i in range(len(new_vs)):
            if np.isnan(new_vs[i]):
                ema = None
            else:
                ema = new_vs[i] if ema is None else alpha * new_vs[i] + (1 - alpha) * ema
                smoothed[i] = ema
        new_vs = smoothed

    return new_ts, new_vs

def compute_acf(values, max_lag):
    """Auto-correlation function. Mirrors JS computeACF()."""
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
        if len(pairs) == 0:
            continue
        cov = np.mean([(a - mean) * (b - mean) for a, b in pairs])
        acf[lag] = cov / var_avg
    return acf

def recommend_tau(acf):
    """Tau selection: first local min after ACF < 0.8, or 1/e crossing."""
    decayed = False
    for i in range(1, len(acf) - 1):
        if acf[i] < 0.8:
            decayed = True
        if acf[i] < 1 / math.e:
            return i
        if decayed and acf[i] > acf[i-1] and acf[i+1] >= acf[i]:
            return i - 1
    return min(40, len(acf) - 1)

# ============================================================================
# S3. Takens Embedding
# ============================================================================

def takens_embedding(values, tau, dim=3):
    """Delay-coordinate embedding. Mirrors JS takensEmbedding()."""
    n = len(values)
    points = []
    for i in range(n - (dim - 1) * tau):
        has_null = False
        pt = np.zeros(dim)
        for d in range(dim):
            v = values[i + d * tau]
            if np.isnan(v):
                has_null = True
                break
            pt[d] = v
        if has_null:
            points.append(None)
        else:
            points.append(pt)
    return points

def calc_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def box_counting_dimension(points):
    """N-dimensional box-counting fractal dimension."""
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 20:
        return None
    dim = len(valid[0])

    valid_arr = np.array(valid)
    mins = valid_arr.min(axis=0)
    maxs = valid_arr.max(axis=0)
    range_val = (maxs - mins).max()
    if range_val <= 1e-9:
        return 0.0

    divs = [2, 4, 8, 16]
    log_n, log_inv_eps = [], []
    for g in divs:
        eps = range_val / g
        keys = set()
        for p in valid_arr:
            parts = tuple(int((p[d] - mins[d]) / eps) for d in range(dim))
            keys.add(parts)
        log_n.append(np.log(len(keys)))
        log_inv_eps.append(np.log(g / range_val))

    k = len(log_n)
    log_inv_eps = np.array(log_inv_eps)
    log_n = np.array(log_n)
    sx, sy = log_inv_eps.sum(), log_n.sum()
    sxx, sxy = (log_inv_eps**2).sum(), (log_inv_eps * log_n).sum()
    denom = k * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    slope = (k * sxy - sx * sy) / denom
    return max(0.0, min(float(dim), slope))

def estimate_embedding_dimension(raw_values, tau):
    """SYC theorem + sampling density cap. Mirrors JS estimateEmbeddingDimension()."""
    probe_points = takens_embedding(raw_values, tau, 3)
    dA = box_counting_dimension(probe_points) or 1.0

    # SYC: m > 2*dA → floor(2*dA) + 1
    calculated_dim = max(3, min(6, int(np.floor(2 * dA)) + 1))

    # Sampling density cap
    probe_n = np.sum(~np.isnan(raw_values))
    dim = 3
    for d in range(4, calculated_dim + 1):
        if probe_n >= MIN_POINTS_FOR_DIM.get(d, float('inf')):
            dim = d
        else:
            break
    capped = dim < calculated_dim
    return dim, dA, calculated_dim, capped, probe_n

# ============================================================================
# S4. Phase Space Analysis
# ============================================================================

def lyapunov_proxy(points):
    """Rosenstein-lite Lyapunov exponent proxy."""
    idx = [i for i, p in enumerate(points) if p is not None]
    m = len(idx)
    if m < 30:
        return None
    dim = len(points[idx[0]])

    valid_arr = np.array([points[i] for i in idx])
    mins = valid_arr.min(axis=0)
    maxs = valid_arr.max(axis=0)
    range_val = (maxs - mins).max() or 1.0
    G = 16
    cell = range_val / G

    def key_of(p):
        return tuple(int((p[d] - mins[d]) / cell) for d in range(dim))

    grid = {}
    for i in idx:
        k = key_of(points[i])
        grid.setdefault(k, []).append(i)

    THEILER = 5
    divs = []
    for pos, i in enumerate(idx):
        if i + 1 >= len(points) or points[i+1] is None:
            continue
        cand = grid.get(key_of(points[i]), [])
        best, best_d = -1, float('inf')
        for j in cand:
            if abs(i - j) <= THEILER:
                continue
            if j + 1 >= len(points) or points[j+1] is None:
                continue
            d0 = calc_distance(points[i], points[j])
            if 1e-6 < d0 < best_d:
                best_d = d0
                best = j
        if best < 0:
            continue
        d1 = calc_distance(points[i+1], points[best+1])
        if d1 > 1e-9 and best_d > 1e-9:
            divs.append(np.log(d1 / best_d))

    if len(divs) < 10:
        return None
    return np.mean(divs)

def compute_rqa(valid_points, target_rr=RQA_TARGET_RR, skip_rp=False):
    """Recurrence Quantification Analysis. Vectorized with numpy."""
    n = len(valid_points)
    if n < 10:
        return None, None, None

    pts = np.array(valid_points)

    # Compute Theiler-excluded pair distances using vectorized approach
    # For n <= 600, full O(n²) is fine; above that, sample
    max_pairs = 150000  # Cap total pairs to keep runtime ~1s
    if n > 600:
        # Sample to cap pair count
        step = max(1, int(n / np.sqrt(max_pairs / 2)))
        pts_sampled = pts[::step]
        n_samp = len(pts_sampled)
    else:
        pts_sampled = pts
        n_samp = n

    distances = []
    for k in range(RQA_THEILER + 1, n_samp):
        diff = pts_sampled[:n_samp - k] - pts_sampled[k:]
        dist_k = np.sqrt(np.sum(diff ** 2, axis=1))
        distances.extend(dist_k.tolist())

    if len(distances) == 0:
        return None, None, None

    distances = np.array(distances)

    # Auto-calibrate epsilon
    sorted_d = np.sort(distances)
    eps_idx = min(len(sorted_d) - 1, max(0, int(target_rr * len(sorted_d))))
    epsilon = sorted_d[eps_idx]

    # Re-scan with original resolution for final DET/ENTR
    if n <= 600:
        # Use full resolution
        pts_use = pts
        n_use = n
    else:
        pts_use = pts_sampled
        n_use = n_samp

    total_rec = 0
    diag_points = 0
    line_lengths = []
    l_min = 2

    for k in range(RQA_THEILER + 1, n_use):
        diff = pts_use[:n_use - k] - pts_use[k:]
        dist_k = np.sqrt(np.sum(diff ** 2, axis=1))
        rec = dist_k < epsilon

        # Find contiguous segments of True
        current_len = 0
        for r in rec:
            if r:
                current_len += 1
                total_rec += 1
            else:
                if current_len >= l_min:
                    line_lengths.append(current_len)
                    diag_points += current_len
                current_len = 0
        if current_len >= l_min:
            line_lengths.append(current_len)
            diag_points += current_len

    # RR
    gap = n_use - RQA_THEILER - 1
    max_possible = gap * (gap + 1) / 2 if gap > 0 else 0
    rr = total_rec / max_possible if max_possible > 0 else 0

    # DET
    det = diag_points / total_rec if total_rec > 0 else 0

    # ENTR
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
    """Jacobi eigenvalue algorithm for symmetric matrices."""
    n = len(A)
    a = A.copy().astype(float)
    for _ in range(max_iter):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > max_val:
                    max_val = abs(a[i, j])
                    p, q = i, j
        if max_val < 1e-10:
            break
        theta = 0.5 * np.arctan2(2 * a[p, q], a[p, p] - a[q, q])
        c, s = np.cos(theta), np.sin(theta)
        app, aqq, apq = a[p, p], a[q, q], a[p, q]
        a[p, p] = c*c*app - 2*s*c*apq + s*s*aqq
        a[q, q] = s*s*app + 2*s*c*apq + c*c*aqq
        a[p, q] = a[q, p] = 0.0
        for j in range(n):
            if j == p or j == q:
                continue
            apj, aqj = a[p, j], a[q, j]
            a[p, j] = a[j, p] = c*apj - s*aqj
            a[q, j] = a[j, q] = s*apj + c*aqj
    ev = np.sort(np.diag(a))[::-1]
    return ev

def gamma_approx(x):
    if x <= 0:
        return 1.0
    return math.sqrt(2 * math.pi / x) * (x / math.e) ** x

def compute_normalized_recovery(points):
    """Recovery velocity toward gravity core."""
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 4:
        return 0.0
    dim = len(valid[0])
    gc = np.array([get_median([p[d] for p in valid]) for d in range(dim)])

    glucose_std = np.std([p[0] for p in valid], ddof=1)

    recovery = []
    prev_dist = None
    for i in range(1, len(points)):
        if points[i] is not None and points[i-1] is not None:
            d = calc_distance(points[i], gc)
            speed = calc_distance(points[i], points[i-1])
            if prev_dist is not None and d < prev_dist:
                recovery.append(speed)
            prev_dist = d
        else:
            prev_dist = None

    avg = np.mean(recovery) if recovery else 0.0
    return avg / glucose_std if glucose_std > 1e-6 else 0.0

def compute_attractor_metrics(shape_points, raw_points, smooth_points, skip_rqa=False):
    """Full attractor metrics. Mirrors JS computeAttractorMetrics()."""
    valid = [p for p in shape_points if p is not None]
    n = len(valid)
    if n < 4:
        return None
    dim = len(valid[0])

    # Arithmetic mean
    arith_mean = np.mean(valid, axis=0)

    # Gravity core (median)
    gravity_core = np.array([get_median([p[d] for p in valid]) for d in range(dim)])

    # Covariance matrix
    centered = np.array(valid) - arith_mean
    cov = (centered.T @ centered) / (n - 1)

    eigvals = jacobi_eigenvalues(cov)
    eigvals = np.maximum(eigvals, 1e-10)

    # 99% variance razor
    total_energy = eigvals.sum()
    cumulative = np.cumsum(eigvals)
    effective_dim = np.searchsorted(cumulative / total_energy, 0.99) + 1
    effective_dim = min(effective_dim, dim)

    # Hyper-ellipsoid volume
    coeff = VOLUME_COEFF.get(effective_dim,
                              math.pi ** (effective_dim / 2) / gamma_approx(effective_dim / 2 + 1))
    vol_product = np.prod(np.sqrt(eigvals[:effective_dim]))
    volume = coeff * vol_product

    # Hysteresis ratio: λ₁/λ₂
    shape_ratio = eigvals[0] / eigvals[1] if eigvals[1] > 1e-12 else float('inf')

    # Recovery (raw data)
    normalized_recovery = compute_normalized_recovery(raw_points)

    # Fractal dimension
    dimension = box_counting_dimension(shape_points)

    # Lyapunov (smoothed data)
    lyap = lyapunov_proxy(smooth_points)

    # RQA
    if skip_rqa:
        rr, det, entr = None, None, None
    else:
        smooth_valid = [p for p in smooth_points if p is not None]
        rr, det, entr = compute_rqa(smooth_valid)

    return {
        'volume': volume,
        'shape_ratio': shape_ratio,
        'avg_recovery': normalized_recovery,
        'dimension': dimension,
        'lyapunov': lyap,
        'rr': rr,
        'det': det,
        'entr': entr,
        'mean': arith_mean,
        'gravity_core': gravity_core,
        'valid_n': n,
        'effective_dim': effective_dim,
        'embedding_dim': dim
    }

# ============================================================================
# S6. Per-Patient Pipeline
# ============================================================================

def compute_patient_metrics(ts, values, period='all'):
    """Run full Takens pipeline for one patient. Returns dict of metrics."""
    # S2: Preprocessing
    raw_ts, raw_vs = resample_to_3min(ts, values, smooth=False)
    smooth_ts, smooth_vs = resample_to_3min(ts, values, smooth=True)

    # Slice by period (always 'all' for full-day analysis)
    # For simplicity, use all data
    sliced_raw = raw_vs
    sliced_smooth = smooth_vs
    sliced_ts = raw_ts

    # Tau estimation from raw data
    acf = compute_acf(sliced_raw, min(60, len(sliced_raw) - 1))
    tau = recommend_tau(acf)
    tau = max(1, tau)

    # S3: Embedding dimension estimation
    try:
        dim, dA, calculated_dim, capped, probe_n = estimate_embedding_dimension(sliced_raw, tau)
    except Exception:
        dim = 3
        dA = 1.5
        calculated_dim = 3
        capped = False

    # S3: Takens embedding
    shape_points = takens_embedding(sliced_smooth, tau, dim)
    raw_points = takens_embedding(sliced_raw, tau, dim)
    smooth_points = takens_embedding(sliced_smooth, tau, dim)

    # S4-S5: Attractor metrics
    metrics = compute_attractor_metrics(shape_points, raw_points, smooth_points)

    if metrics is None:
        return None

    metrics['tau'] = tau
    metrics['dA'] = dA
    metrics['calculated_dim'] = calculated_dim
    metrics['capped'] = capped
    metrics['n_points'] = len(sliced_raw)

    return metrics

# ============================================================================
# S7. MRI Score (mirrors JS)
# ============================================================================

def compute_mri(metrics):
    """Compute Metabolic Resilience Index. Mirrors JS renderMetrics scoring."""
    base_score = 1000.0

    # Volume
    v = metrics['volume']
    if v < 0.5: base_score -= 150
    elif v < 1.0: base_score -= 50
    elif v > 4.0: base_score -= 180
    elif v > 3.0: base_score -= 80

    # Recovery
    r = metrics['avg_recovery']
    if r > 0.35: base_score -= 200
    elif r > 0.25: base_score -= 90
    elif r < 0.06: base_score -= 150
    elif r < 0.10: base_score -= 60

    # Hysteresis
    sr = metrics['shape_ratio']
    if sr < 1.2: base_score -= 200
    elif sr < 1.8: base_score -= 80

    # Dimension
    d = metrics['dimension']
    if d is not None:
        if d > 2.7: base_score -= 100
        elif d > 2.3: base_score -= 40
        elif d < 1.2: base_score -= 40

    # Lyapunov
    ly = metrics['lyapunov']
    if ly is not None:
        if ly > 0.95: base_score -= 120
        elif ly > 0.70: base_score -= 50

    # DET
    det = metrics['det']
    if det is not None:
        if det > 0.85: base_score -= 100
        elif det > 0.70: base_score -= 40
        elif det < 0.20: base_score -= 40

    # ENTR
    entr = metrics['entr']
    if entr is not None:
        if entr < 0.5: base_score -= 120
        elif entr < 1.0: base_score -= 50

    # Tau penalty
    tau = metrics.get('tau', 6)
    if tau > 12:
        base_score -= (tau - 12) * 20

    return max(0, base_score)


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_validation():
    print("=" * 70)
    print("GlucoBench Takens Phase Space Validation")
    print("=" * 70)

    # ---- Colas Dataset (primary) ----
    print("\n[1/4] Loading Colas dataset (208 patients, T2DM labels)...")
    df, labels = load_colas_dataset()
    patient_ids = sorted(df['id'].unique())
    print(f"  Patients: {len(patient_ids)}, T2DM positive: {labels.sum()} ({labels.mean()*100:.1f}%)")

    # ---- Compute metrics for each patient ----
    print("\n[2/4] Computing Takens phase space metrics per patient...")
    results = []
    for i, pid in enumerate(patient_ids):
        pdata = df[df['id'] == pid].sort_values('time')
        ts = pdata['time'].values
        vs = pdata['gl'].values

        metrics = compute_patient_metrics(ts, vs)
        if metrics is None:
            continue

        mri = compute_mri(metrics)
        metrics['patient_id'] = pid
        metrics['label'] = labels.loc[pid]
        metrics['mri'] = mri
        results.append(metrics)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(patient_ids)}...")

    print(f"  Completed: {len(results)}/{len(patient_ids)} patients with valid metrics")

    # Checkpoint: save intermediate results
    import json as _json, os as _os
    _checkpoint = []
    for _r in results:
        _cp = {}
        for _k, _v in _r.items():
            if isinstance(_v, (np.integer,)): _cp[_k] = int(_v)
            elif isinstance(_v, (np.floating,)): _cp[_k] = float(_v)
            elif isinstance(_v, np.ndarray): _cp[_k] = _v.tolist()
            elif _v is None or isinstance(_v, (bool, str, int, float)): _cp[_k] = _v
        _checkpoint.append(_cp)
    _os.makedirs('output', exist_ok=True)
    with open('output/takens_results.json', 'w') as _f:
        _json.dump(_checkpoint, _f)
    print(f"  Checkpoint saved: output/takens_results.json")

    feature_names = ['volume', 'shape_ratio', 'avg_recovery', 'dimension', 'lyapunov',
                     'det', 'entr', 'tau', 'dA', 'mri']

    X = []
    y = []
    for r in results:
        row = []
        valid = True
        for fn in feature_names:
            v = r.get(fn)
            if v is None or np.isnan(v) or np.isinf(v):
                valid = False
                break
            row.append(v)
        if valid:
            X.append(row)
            y.append(r['label'])

    X = np.array(X)
    y = np.array(y)
    print(f"  Complete cases: {len(y)} patients")

    # ---- Per-feature analysis ----
    print("\n" + "-" * 70)
    print("Individual Feature Discriminatory Power (Mann-Whitney U)")
    print("-" * 70)
    print(f"{'Feature':<20} {'Normal μ':>10} {'T2DM μ':>10} {'Δ':>8} {'p-value':>10} {'AUC':>8}")
    print("-" * 70)

    aucs = {}
    for i, fn in enumerate(feature_names):
        vals_normal = X[y == 0, i]
        vals_t2dm = X[y == 1, i]

        if len(vals_t2dm) < 3:
            continue

        try:
            u_stat, p_val = stats.mannwhitneyu(vals_normal, vals_t2dm, alternative='two-sided')
        except Exception:
            p_val = 1.0

        try:
            auc = roc_auc_score(y, X[:, i])
        except Exception:
            auc = 0.5

        aucs[fn] = auc
        delta = np.mean(vals_t2dm) - np.mean(vals_normal)
        print(f"{fn:<20} {np.mean(vals_normal):>10.4f} {np.mean(vals_t2dm):>10.4f} {delta:>+8.4f} {p_val:>10.4f} {auc:>8.3f}")

    # ---- Composite classifier ----
    print("\n" + "-" * 70)
    print("Composite Classifier (Logistic Regression on Phase Space Features)")
    print("-" * 70)

    # Use features with AUC > 0.55 or p < 0.2
    selected_features = [fn for fn in feature_names if aucs.get(fn, 0.5) > 0.52]
    if len(selected_features) < 2:
        selected_features = feature_names[:5]  # fallback

    selected_idx = [feature_names.index(fn) for fn in selected_features]
    X_sel = X[:, selected_idx]
    print(f"  Selected features: {selected_features}")

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    cv_scores = cross_val_score(lr, X_sel, y, cv=cv, scoring='roc_auc')
    print(f"  CV AUC (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Full fit for coefficients
    lr.fit(X_sel, y)
    print(f"\n  Feature coefficients:")
    for fn, coef in zip(selected_features, lr.coef_[0]):
        print(f"    {fn:<20}: {coef:+.4f}")

    # ---- Compare with traditional CGM metrics ----
    print("\n" + "-" * 70)
    print("Comparison: Traditional CGM Metrics vs Phase Space Metrics")
    print("-" * 70)

    # Compute traditional metrics for each patient
    traditional_results = []
    for pid in patient_ids:
        pdata = df[df['id'] == pid].sort_values('time')
        vs = pdata['gl'].values

        if len(vs) < 10:
            continue

        tir_70_180 = np.mean((vs >= 3.9) & (vs <= 10.0)) * 100
        cv_pct = np.std(vs, ddof=1) / np.mean(vs) * 100 if np.mean(vs) > 0 else 0
        mean_gl = np.mean(vs)
        sd_gl = np.std(vs, ddof=1)
        below_70 = np.mean(vs < 3.9) * 100
        above_180 = np.mean(vs > 10.0) * 100

        traditional_results.append({
            'patient_id': pid,
            'label': labels.loc[pid],
            'tir': tir_70_180,
            'cv': cv_pct,
            'mean_gl': mean_gl,
            'sd_gl': sd_gl,
            'below_70': below_70,
            'above_180': above_180,
        })

    trad = pd.DataFrame(traditional_results)
    trad_features = ['tir', 'cv', 'mean_gl', 'sd_gl', 'below_70', 'above_180']
    X_trad = trad[trad_features].values
    y_trad = trad['label'].values

    print(f"{'Metric':<20} {'AUC':>8}")
    print("-" * 30)
    trad_aucs = {}
    for i, fn in enumerate(trad_features):
        try:
            auc = roc_auc_score(y_trad, X_trad[:, i])
        except Exception:
            auc = 0.5
        trad_aucs[fn] = auc
        print(f"{fn:<20} {auc:>8.3f}")

    # Traditional composite
    cv_trad = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_trad, y_trad, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    print(f"\n  Traditional CV AUC: {cv_trad.mean():.3f} ± {cv_trad.std():.3f}")

    # Combined model
    X_combined = np.hstack([X_sel, X_trad])
    cv_combined = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_combined, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    print(f"  Phase Space CV AUC:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Combined CV AUC:     {cv_combined.mean():.3f} ± {cv_combined.std():.3f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    best_phase_feature = max(aucs, key=aucs.get)
    best_trad_feature = max(trad_aucs, key=trad_aucs.get)
    print(f"  Best phase space feature: {best_phase_feature} (AUC={aucs[best_phase_feature]:.3f})")
    print(f"  Best traditional feature: {best_trad_feature} (AUC={trad_aucs[best_trad_feature]:.3f})")
    print(f"  Phase space composite AUC: {cv_scores.mean():.3f}")
    print(f"  Traditional composite AUC: {cv_trad.mean():.3f}")
    print(f"  Combined AUC:              {cv_combined.mean():.3f}")

    return results, X, y, aucs, trad_aucs, cv_scores, cv_trad

if __name__ == '__main__':
    results, X, y, aucs, trad_aucs, cv_scores, cv_trad = run_validation()
