"""
Faithful Python ports of the six pre-doctrinal, PURELY-NEUTRAL (no warn/bad color coding)
legacy JS attractor-geometry metrics from index_v4.html, for Wind-Tunnel testing purposes ONLY
(AGENTS.md Section 9). Ported for the 2026-08-19 "第一组8张(实为6张)中性卡审计" changelog entry.

Companion to _legacy_metrics_v4.py (which covers the 6 Group-2 color-coded cards). This module
covers the 6 Group-1 cards that have never carried a warn/bad judgment at all:
  相空间展开体积 (Volume), 归一化向心步长 (Recovery), 主轴各向异性 (λ1/λ2 Shape Ratio),
  当前-夜间核心距离 (Core Dist), 盒计数几何维度 (Box-Counting Dimension), 一步近邻发散代理
  (Lyapunov).

Why this file exists instead of reusing _extracted_tensor_engine_v4.py: these six cards have
NO Python/Pyodide implementation in production -- they are pure JavaScript
(computeAttractorMetrics's covariance/eigen block, boxCountingDimension, lyapunovProxy,
computeNormalizedRecovery, calcDistance). This module is the translation, hand-ported
line-for-line from the JS source (exact line ranges cited per function below, as of the
2026-08-19 21:00 HEAD used for this port) and cross-checked against the actual JS via Node.js
(see _js_legacy_metrics_group1_crosscheck.mjs) rather than trusted on manual transcription
alone.

Section 9.5 Product Isolation applies in full: NOTHING here is wired into index_v4.html. The
JS remains the sole production source of truth; this module is a read-only research proxy.
"""
import math
from typing import List, Optional, Sequence, Tuple


# Port of VOLUME_COEFF, index_v4.html:1590-1597.
VOLUME_COEFF = {
    1: 2.0,
    2: math.pi,
    3: (4.0 / 3.0) * math.pi,
    4: math.pi * math.pi / 2.0,
    5: 8.0 * math.pi * math.pi / 15.0,
    6: math.pi * math.pi * math.pi / 6.0,
}


def gamma_approx(x: float) -> float:
    """Port of gammaApprox(x), index_v4.html:2448-2455."""
    if x <= 0:
        return 1.0
    if abs(x - 2.5) < 0.001:
        return 1.329340388179137
    if abs(x - 3.0) < 0.001:
        return 2.0
    if abs(x - 3.5) < 0.001:
        return 3.3233509704478426
    if abs(x - 4.0) < 0.001:
        return 6.0
    return math.sqrt(2 * math.pi / x) * (x / math.e) ** x


def calc_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Port of calcDistance(p1, p2), index_v4.html:2171-2175."""
    s = 0.0
    for d in range(len(p1)):
        diff = p1[d] - p2[d]
        s += diff * diff
    return math.sqrt(s)


def box_counting_dimension(points: List[Optional[List[float]]]) -> Optional[float]:
    """Port of boxCountingDimension(points), index_v4.html:2178-2224. Feeds "盒计数几何维度"."""
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 20:
        return None
    dim = len(valid[0])

    mn = [math.inf] * dim
    mx = [-math.inf] * dim
    for p in valid:
        for d in range(dim):
            if p[d] < mn[d]:
                mn[d] = p[d]
            if p[d] > mx[d]:
                mx[d] = p[d]
    rng = 0.0
    for d in range(dim):
        rng = max(rng, mx[d] - mn[d])
    if rng <= 1e-9:
        return 0.0

    divs = [2, 4, 8, 16]
    log_n: List[float] = []
    log_inv_eps: List[float] = []
    for g in divs:
        eps = rng / g
        occ = set()
        for p in valid:
            key = 0
            for d in range(dim):
                key = key * (g + 1) + int(math.floor((p[d] - mn[d]) / eps))
            occ.add(key)
        log_n.append(math.log(len(occ)))
        log_inv_eps.append(math.log(g / rng))

    k = len(log_n)
    sx = sy = sxx = sxy = 0.0
    for i in range(k):
        sx += log_inv_eps[i]
        sy += log_n[i]
        sxx += log_inv_eps[i] * log_inv_eps[i]
        sxy += log_inv_eps[i] * log_n[i]
    denom = k * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    val = (k * sxy - sx * sy) / denom
    return max(0.0, min(float(dim), val))


def lyapunov_proxy(points: List[Optional[List[float]]]) -> Optional[float]:
    """Port of lyapunovProxy(points), index_v4.html:2227-2281. Feeds "一步近邻发散代理"."""
    idx = [i for i, p in enumerate(points) if p is not None]
    m = len(idx)
    if m < 30:
        return None
    dim = len(points[idx[0]])

    mn = [math.inf] * dim
    mx = [-math.inf] * dim
    for i in idx:
        for d in range(dim):
            v = points[i][d]
            if v < mn[d]:
                mn[d] = v
            if v > mx[d]:
                mx[d] = v
    rng = 0.0
    for d in range(dim):
        rng = max(rng, mx[d] - mn[d])
    rng = rng or 1.0
    G = 16
    cell = rng / G

    def key_of(p: Sequence[float]) -> int:
        key = 0
        for d in range(dim):
            key = key * (G + 1) + int(math.floor((p[d] - mn[d]) / cell))
        return key

    grid: dict = {}
    for i in idx:
        kk = key_of(points[i])
        grid.setdefault(kk, []).append(i)

    THEILER = 5
    divs: List[float] = []
    idx_set_next_valid = {i: (i + 1 < len(points) and points[i + 1] is not None) for i in idx}
    for i in idx:
        if not idx_set_next_valid.get(i):
            continue
        cand = grid.get(key_of(points[i]), [])
        best = -1
        best_d = math.inf
        for j in cand:
            if abs(i - j) <= THEILER:
                continue
            if j + 1 >= len(points) or points[j + 1] is None:
                continue
            d0 = calc_distance(points[i], points[j])
            if d0 > 1e-6 and d0 < best_d:
                best_d = d0
                best = j
        if best < 0:
            continue
        d1 = calc_distance(points[i + 1], points[best + 1])
        if d1 > 1e-9 and best_d > 1e-9:
            divs.append(math.log(d1 / best_d))

    if len(divs) < 10:
        return None
    return sum(divs) / len(divs)


def _median(arr: Sequence[float]) -> float:
    """Local getMedian equivalent (index_v4.html global getMedian), returns 0.0 for empty."""
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0


def jacobi_eigenvalues(A: List[List[float]], max_iter: int = 50) -> List[float]:
    """Port of jacobiEigenvalues(A, maxIter=50), index_v4.html:2413-2441. Cyclic-Jacobi
    eigenvalue solver for symmetric matrices, ported iteration-for-iteration (NOT replaced
    with numpy.linalg.eigvalsh) to stay bit-for-bit faithful to the JS algorithm's specific
    convergence path, per Section 9.4 Bit-for-Bit Truth."""
    n = len(A)
    a = [row[:] for row in A]
    for _ in range(max_iter):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j
        if max_val < 1e-10:
            break
        theta = 0.5 * math.atan2(2 * a[p][q], a[p][p] - a[q][q])
        c, s = math.cos(theta), math.sin(theta)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for j in range(n):
            if j == p or j == q:
                continue
            apj, aqj = a[p][j], a[q][j]
            a[p][j] = a[j][p] = c * apj - s * aqj
            a[q][j] = a[j][q] = s * apj + c * aqj
    ev = [a[i][i] for i in range(n)]
    ev.sort(reverse=True)
    return ev


def compute_volume_shape(points: List[Optional[List[float]]]) -> Optional[dict]:
    """Port of the covariance/eigen/volume/shape block inside computeAttractorMetrics(),
    index_v4.html:2845-2897. Feeds "相空间展开体积 (Volume)" and "主轴各向异性 (λ1/λ2)", and
    also returns gravityCore (median-per-column) needed downstream for Core Dist.

    `points` here is `shapePoints` at the JS call site (index_v4.html:3669's first argument),
    i.e. the period-sliced SMOOTH-track Takens embedding under the production default
    smooth=true UI toggle.
    """
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 4:
        return None
    dim = len(valid[0])

    arith_mean = [0.0] * dim
    for p in valid:
        for d in range(dim):
            arith_mean[d] += p[d]
    for d in range(dim):
        arith_mean[d] /= n

    cols = [[valid[i][d] for i in range(n)] for d in range(dim)]
    gravity_core = [_median(cols[d]) for d in range(dim)]

    cov = [[0.0] * dim for _ in range(dim)]
    for p in valid:
        centered = [p[d] - arith_mean[d] for d in range(dim)]
        for i in range(dim):
            ci = centered[i]
            for j in range(i, dim):
                cov[i][j] += ci * centered[j]
    for i in range(dim):
        for j in range(i + 1, dim):
            cov[j][i] = cov[i][j]
        for j in range(dim):
            cov[i][j] /= (n - 1)

    eigvals = [max(v, 1e-10) for v in jacobi_eigenvalues(cov)]

    total_energy = sum(eigvals)
    accumulated_energy = 0.0
    effective_dim = 0
    for i in range(dim):
        accumulated_energy += eigvals[i]
        effective_dim += 1
        if accumulated_energy / total_energy > 0.99:
            break

    coeff = VOLUME_COEFF.get(effective_dim)
    if coeff is None:
        coeff = (math.pi ** (effective_dim / 2.0)) / gamma_approx(effective_dim / 2.0 + 1.0)
    vol_product = 1.0
    for i in range(effective_dim):
        vol_product *= math.sqrt(eigvals[i])
    volume = coeff * vol_product

    shape_ratio = (eigvals[0] / eigvals[1]) if eigvals[1] > 1e-12 else math.inf

    return {
        "volume": volume,
        "shapeRatio": shape_ratio,
        "gravityCore": gravity_core,
        "effectiveDim": effective_dim,
        "validPoints": valid,
    }


def compute_normalized_recovery(points: List[Optional[List[float]]]) -> float:
    """Port of computeNormalizedRecovery(points), index_v4.html:2598-2623. Feeds "归一化向心
    步长 (Recovery)". `points` here is `rawPoints` at the JS call site (index_v4.html:2902) --
    the period-sliced RAW-track Takens embedding, always unsmoothed regardless of the UI
    toggle (Blueprint v3.3 L0 2.2 Dual-Track Law)."""
    valid = [p for p in points if p is not None]
    n = len(valid)
    if n < 4:
        return 0.0
    dim = len(valid[0])

    cols = [[valid[i][d] for i in range(n)] for d in range(dim)]
    gc = [_median(cols[d]) for d in range(dim)]

    mean0 = sum(p[0] for p in valid) / n
    var0 = sum((p[0] - mean0) ** 2 for p in valid)
    glucose_std = math.sqrt(var0 / (n - 1))

    recovery: List[float] = []
    prev_dist: Optional[float] = None
    for i in range(1, len(points)):
        if points[i] is not None and points[i - 1] is not None:
            d = calc_distance(points[i], gc)
            speed = calc_distance(points[i], points[i - 1])
            if prev_dist is not None and d < prev_dist:
                recovery.append(speed)
            prev_dist = d
        else:
            prev_dist = None

    avg = (sum(recovery) / len(recovery)) if recovery else 0.0
    return (avg / glucose_std) if glucose_std > 1e-6 else 0.0


def compute_group1_attractor_metrics(
    shape_points: List[Optional[List[float]]],
    raw_points: List[Optional[List[float]]],
    smooth_points: List[Optional[List[float]]],
) -> dict:
    """Convenience wrapper bundling all 6 Group-1 operators from a single (shapePoints,
    rawPoints, smoothPoints) triple, mirroring computeAttractorMetrics()'s call signature
    (index_v4.html:2845) minus the RQA/workIntegral fields (those are Section 9.1.1
    Calculation-Firewall-compliant Python engine outputs already covered by run_subject(),
    not re-derived here).
    """
    vs = compute_volume_shape(shape_points)
    if vs is None:
        return {
            "volume": None, "shapeRatio": None, "gravityCore": None,
            "avgRecovery": None, "dimension": None, "lyapunov": None,
        }
    avg_recovery = compute_normalized_recovery(raw_points)
    dimension = box_counting_dimension(shape_points)
    lyapunov = lyapunov_proxy(smooth_points)
    return {
        "volume": vs["volume"],
        "shapeRatio": vs["shapeRatio"] if math.isfinite(vs["shapeRatio"]) else None,
        "gravityCore": vs["gravityCore"],
        "avgRecovery": avg_recovery,
        "dimension": dimension,
        "lyapunov": lyapunov,
    }
