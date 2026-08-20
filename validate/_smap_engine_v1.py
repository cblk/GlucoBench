"""
S-Map / Simplex Projection Engine v1 (Sugihara & May 1990; Sugihara 1994).

Candidate operator #6 in `reports/candidate_tensor_staging_matrix.md`: Delta-rho
Nonlinear-Predictability Gain. This module is the AGENTS.md Section 4.B stage 2
("高维猜想" High-Dimensional Hypothesis) -> stage 3 ("风洞对撞" Collision) FIRST
engineering step for Phase C of `reports/wind_tunnel_v4_1_to_v4_2_refactoring_
roadmap_20260820.md` (approved 2026-08-20 15:58, Section 4.3). Target: replace
AR1 (critical slowing down) and the Lyapunov proxy -- both audited Fail-Closed/
redundant on 2026-08-19 -- with a single non-linear-vs-linear predictability
gain metric that measures the SAME underlying physical quantity ("has the
system's deterministic nonlinear structure degraded into a near-linear/white-
noise machine") that AR1/Lyapunov were both trying and failing to measure.

DOCTRINE NOTE ON TAU (read before touching this file):
Blueprint v3.3 Section 3.1 locks ONE autocorrelation-derived tau per epoch for
the WHOLE production pipeline (RQA / Work Integral / dimension estimation),
because those operators need a delay large enough to avoid geometrically
redundant, highly autocorrelated embedding axes. S-Map/Simplex is a DIFFERENT
mathematical tool from a different literature (Empirical Dynamic Modeling,
Sugihara school) with its OWN native convention: embed at unit lag (tau=1,
i.e. every consecutive 3-minute grid sample) and let the embedding DIMENSION E
(chosen by leave-one-out Simplex projection, not FNN) carry the memory depth.
Using tau=1 here is a deliberate, literature-grounded choice -- NOT a silent
reinvention of the Blueprint's tau, and NOT a Section 9.4 Bit-for-Bit-Truth
violation (that section binds re-implementations of EXISTING production
operators; S-Map is a brand-new candidate with no production counterpart yet).
It is also the ONLY choice that makes S-Map computationally feasible at all
inside a single ~6h night window: the production tau can lock as high as 120
steps (360 minutes) under Blueprint v3.6, which would consume an entire night
in a single embedding step and leave zero library points for forecasting --
this is precisely the "何种规模才可行" feasibility risk flagged in the roadmap
Section 3 Phase C risk declaration.

DOCTRINE NOTE ON FORECAST HORIZON `tp` [v1.1, post-Hall-smoke-test finding]:
The FIRST Hall smoke test run (`wind_tunnel_v4_hall_smap_smoke.py`, 2026-08-20)
used the literature-default tp=1 (predict exactly the NEXT 3-minute sample) on
the SMOOTH night track, and produced delta_rho == 0.0000 for all 57/57
subjects with theta_best pinned at the sweep's ceiling (8) yet rho(0)==rho(8)
==~1.0 -- a **ceiling-saturation artifact**, not evidence of "no nonlinear
structure": at a 3-minute unit lag, physiological glucose literally cannot
change enough between consecutive samples for ANY reasonable model (linear or
not) to fail, so theta comparison has zero headroom to discriminate anything.
Diagnostic sweep (see `reports/` smoke-test report) confirmed: (a) switching
to the RAW (unsmoothed) track alone does not fix it at tp=1 (rho~0.97-0.99
ceiling persists, still short-term persistence-dominated); (b) increasing the
forecast horizon `tp` (predicting `tp` unit-lag steps ahead, still embedding
at unit lag) on the RAW track breaks the ceiling cleanly -- e.g. one Hall
subject: tp=1 rho(0)=0.97 flat across theta; tp=6 rho(0)=0.36 -> rho(8)=0.57
(delta_rho=+0.21); tp=10 rho(0)=0.07 -> rho(8)=0.52 (delta_rho=+0.45). This
matches Blueprint's OWN existing Dual-Track assignment logic (AR1 -- the
metric S-Map is meant to replace -- already lives on the RAW track per L0 2.2)
and is physiologically sensible: linear extrapolation is a free, correct
lunch at horizons far shorter than the system's own characteristic relaxation
time, and only degrades enough to reveal nonlinear headroom once the horizon
approaches a physiologically meaningful timescale. `tp` is therefore now an
explicit, REQUIRED, non-magic-constant parameter on every function in this
module (no silent default of 1) -- see the Hall tp-sweep survey report for the
current best-available (not yet cross-cohort-validated) recommended value.

DOCTRINE NOTE ON NIGHT SEGMENTATION (No-Cross-Night / No-Cross-Gap Pooling):
Unlike `_legacy_metrics_v4.py::compute_critical_slowing_down` (AR1), which
concatenates all valid readings within a calendar night regardless of internal
gaps (an established legacy behavior this module does NOT need to replicate --
S-Map is greenfield), this module requires MAXIMAL CONTIGUOUS RUNS of valid
3-minute-grid samples (see `extract_contiguous_runs`). A delay-embedding step
that silently bridges a data dropout would fabricate a physically nonexistent
state transition -- exactly the "无摩擦的人造直线弦" pathology Blueprint v3.3
Section 2.1 legislates against. Runs are additionally never allowed to span a
calendar-night boundary because the source array positions outside the night
window (00:00-06:00) are already masked to None by `slice_by_period`, so a
day-time gap always breaks the run naturally.

Zero-Magic-Constant compliance (AGENTS.md Section 8.3): `min_lib` is an
explicit, documented feasibility floor (not a silently baked-in fallback) --
callers see it in every returned dict and it is logged when a run/night is
excluded for falling below it.
"""
import numpy as np


def _pearson(a, b):
    """Returns None (never 0.0 or NaN) when correlation is undefined -- honest
    Fail-Closed per AGENTS.md Section 8.2, not a fabricated neutral value."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r):
        return None
    return float(r)


def _embed(y, E, tp):
    """Unit-lag (tau=1) delay embedding with an explicit forecast horizon `tp`.
    X[i] = [y[j-1], y[j-2], ..., y[j-E]] (most-recent-first, state observed up
    to and including time j-1), predicting y[j-1+tp] (tp steps into the
    future). tp=1 reduces to the literature-default "predict the very next
    sample" (see module docstring's [v1.1] note for why that default is
    UNSAFE on this data at the smooth/raw tracks tried so far). Returns
    (X, target_idx) where target_idx[i] is the index into `y` of the value
    X[i] is trying to predict.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    last_i = n - tp  # j ranges [E, n - tp] inclusive so that j-1+tp <= n-1
    if last_i < E:
        return np.empty((0, E)), np.empty((0,), dtype=int)
    n_lib = last_i - E + 1
    X = np.zeros((n_lib, E))
    target_idx = np.zeros(n_lib, dtype=int)
    for k, j in enumerate(range(E, last_i + 1)):
        X[k] = y[j - E:j][::-1]
        target_idx[k] = j - 1 + tp
    return X, target_idx


def simplex_projection(y, tp, e_max=8, min_lib=20, min_lib_ratio=4):
    """Leave-one-out Simplex projection (Sugihara & May 1990) to select the
    embedding dimension E that maximizes one-step-ahead forecast skill.

    For each candidate E in [1, e_max], embeds the (contiguous, gap-free) series
    `y` at unit lag, and for every target point uses its E+1 nearest OTHER
    neighbors in state space (exponential-distance-weighted, normalized to the
    nearest neighbor's distance per the original Simplex formulation) to
    forecast the next value. rho(E) = Pearson correlation of predicted vs
    actual over all leave-one-out targets. Returns the E with the highest rho.

    min_lib is an explicit floor on the number of leave-one-out target points
    available AFTER embedding (n - E) -- below this, statistics are considered
    too noisy to trust and that E is skipped entirely (not penalized to 0).

    [v1.2, post-Stanford-SSPG-Fail-Closed finding] `min_lib_ratio` is a SECOND,
    E-SCALED floor: n_lib must be >= min_lib_ratio * (E + 1), not just the flat
    `min_lib`.     Rationale: the 2026-08-20 Stanford SSPG run found e_best/theta_best
    pinned at the grid ceiling (E_max=8) for 27/29 subjects with e_max=8 and a
    flat min_lib=20 -- i.e. E=8 (9-dimensional Simplex neighbor search / 9-column
    S-Map regression) was being accepted on libraries as thin as ~20-30 points,
    which is exactly the curse-of-dimensionality overfitting regime the module's
    [v1.1] docstring note (Homomorphic Loss prediction (b)) flagged as a risk
    BEFORE that run happened. Requiring 4x(E+1) points per E means E=8 now needs
    n_lib>=36 and E=15 needs n_lib>=64.

    [RESULT, post-grid-widening check, see `wind_tunnel_v4_hall_smap_grid_check.py`
    and `reports/experiment_20260820_1630_smap_phase_c_kickoff.md` Section 6]:
    widening e_max from 8 to 16 (with this min_lib_ratio floor active) did NOT
    resolve the ceiling-pinning -- 46/57 Hall subjects still pinned at E=16, and
    a per-subject rho(E) curve traced out to E=20 kept climbing rather than
    saturating (0.068 @E=1 -> 0.799 @E=12 -> dips to 0.790 @E=14 -> climbs again
    to 0.822 @E=20). This is diagnosed as `The_Cybernetic_Wind_Tunnel_Doctrine_
    v1.1.md` Section 2's "operator divergence" (敬畏算子发散), NOT a fixable grid-
    size problem: raw leave-one-out rho-maximization for E selection, with no
    explicit model-complexity penalty (e.g. AICc), will keep rewarding larger E
    indefinitely on a modest-sized library because higher dimensions always let
    the leave-one-out neighbor search find a "coincidentally" closer point. DO
    NOT "fix" this by raising e_max further -- that is exactly the forbidden
    "resuscitation" move (Section 2's "拒绝抢救失效指标"). A real fix requires a
    complexity-penalized E-selection criterion, which is a materially deeper
    redesign than a parameter widening, and per the 2026-08-20 report requires
    fresh human sign-off before further cohort spend.
    """
    y = np.asarray(y, dtype=float)
    rho_curve = {}
    for E in range(1, e_max + 1):
        X, target_idx = _embed(y, E, tp)
        n_lib = len(target_idx)
        floor = max(min_lib, min_lib_ratio * (E + 1))
        if n_lib < floor:
            continue
        k = E + 1
        if n_lib < k + 1:
            continue
        preds = np.full(n_lib, np.nan)
        for i in range(n_lib):
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:k]
            nn_dists = dists[nn_idx]
            d1 = nn_dists[0]
            if d1 <= 1e-12:
                w = (nn_dists <= 1e-12).astype(float)
                if w.sum() == 0:
                    w = np.ones(k)
            else:
                w = np.exp(-nn_dists / d1)
            w = w / w.sum()
            preds[i] = float(np.dot(w, y[target_idx[nn_idx]]))
        rho = _pearson(preds, y[target_idx])
        if rho is not None:
            rho_curve[E] = rho
    if not rho_curve:
        return {"e_best": None, "rho_best": None, "rho_curve": {},
                "error": f"No E in [1,{e_max}] reached min_lib={min_lib} valid leave-one-out targets."}
    e_best = max(rho_curve, key=lambda e: rho_curve[e])
    return {"e_best": e_best, "rho_best": rho_curve[e_best], "rho_curve": rho_curve}


def smap_theta_sweep(y, E, tp, thetas=(0, 0.5, 1, 2, 3, 4, 6, 8), min_lib=20, min_lib_ratio=4):
    """Leave-one-out S-Map (Sugihara 1994) at a FIXED embedding dimension E
    and forecast horizon `tp`.

    theta=0 degenerates to a single global linear (OLS) autoregressive model
    (uniform weights over the whole library) -- this is the "system behaves
    linearly" null. theta>0 progressively localizes the weighted linear
    regression to state-space neighbors (weight = exp(-theta * d_j / mean(d)))
    -- higher rho at higher theta means local (nonlinear/state-dependent)
    structure genuinely improves forecast skill beyond what a single global
    linear model can capture.

    `min_lib_ratio` mirrors `simplex_projection`'s [v1.2] E-scaled floor (see
    that function's docstring) -- an E-dimensional S-Map regression (E+1 free
    coefficients including intercept) fit on a library barely larger than
    `min_lib` is exactly the overfitting regime the 2026-08-20 Stanford SSPG
    run's ceiling-pinning artifact came from.
    """
    y = np.asarray(y, dtype=float)
    X, target_idx = _embed(y, E, tp)
    n_lib = len(target_idx)
    floor = max(min_lib, min_lib_ratio * (E + 1))
    if n_lib < floor:
        return {"theta_curve": {}, "error": f"Insufficient library ({n_lib} < floor={floor}) for E={E}."}
    actual = y[target_idx]
    A = np.hstack([np.ones((n_lib, 1)), X])
    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    theta_curve = {}
    for theta in thetas:
        preds = np.full(n_lib, np.nan)
        for i in range(n_lib):
            d = dmat[i].copy()
            d[i] = np.inf
            finite = np.isfinite(d)
            d_mean = np.mean(d[finite]) if np.any(finite) else 1e-12
            if d_mean <= 1e-12:
                d_mean = 1e-12
            if theta == 0:
                w = np.ones_like(d)
            else:
                w = np.exp(-theta * d / d_mean)
            w[i] = 0.0
            sw = np.sqrt(w)
            keep = sw > 1e-8
            if keep.sum() < E + 2:
                continue
            Aw = A[keep] * sw[keep, None]
            bw = actual[keep] * sw[keep]
            coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            preds[i] = float(np.dot(A[i], coef))
        valid = ~np.isnan(preds)
        rho = _pearson(preds[valid], actual[valid]) if valid.sum() >= 3 else None
        theta_curve[theta] = rho
    return {"theta_curve": theta_curve}


def compute_smap_delta_rho(y, tp, e_max=8, thetas=(0, 0.5, 1, 2, 3, 4, 6, 8), min_lib=20, min_lib_ratio=4):
    """Orchestrates Simplex projection (E selection) -> S-Map theta sweep at
    E_best -> Delta-rho = rho(theta_best) - rho(theta=0). `tp` (forecast
    horizon, in unit-lag steps) is a REQUIRED, explicit, non-magic-constant
    argument -- see module docstring [v1.1] note; there is no silent default.

    Positive delta_rho: nonlinear/state-dependent local models forecast
    meaningfully better than the single global linear model -- the system
    still has live deterministic nonlinear structure to exploit ("stiffness"
    intact). delta_rho collapsing toward 0 (or negative, meaning the "best"
    theta found is noise-driven overfitting of an otherwise flat curve): the
    system's dynamics have degraded toward linear/white-noise -- the working
    hypothesis (roadmap Section 3 Phase C) is that this is a marker of
    critical slowing down / stiffened decompensation, replacing what AR1 was
    trying (and, per the 2026-08-19 fleet audit, failing) to measure.
    """
    events = []
    simplex_res = simplex_projection(y, tp, e_max=e_max, min_lib=min_lib, min_lib_ratio=min_lib_ratio)
    if simplex_res.get("e_best") is None:
        events.append(f"[ERROR] [SMap] simplex_projection failed: {simplex_res.get('error')}")
        return {"delta_rho": None, "theta_best": None, "e_best": None,
                "rho_theta0": None, "rho_theta_best": None,
                "rho_curve_E": {}, "theta_curve": {}, "events": events}
    e_best = simplex_res["e_best"]
    events.append(
        f"[INFO] [SMap] Simplex projection locked E={e_best} (rho={simplex_res['rho_best']:.4f}); "
        f"rho_curve_E={ {k: round(v, 4) for k, v in simplex_res['rho_curve'].items()} }"
    )
    theta_res = smap_theta_sweep(y, e_best, tp, thetas=thetas, min_lib=min_lib, min_lib_ratio=min_lib_ratio)
    theta_curve = theta_res.get("theta_curve", {})
    if theta_res.get("error"):
        events.append(f"[ERROR] [SMap] smap_theta_sweep failed: {theta_res['error']}")
        return {"delta_rho": None, "theta_best": None, "e_best": e_best,
                "rho_theta0": None, "rho_theta_best": None,
                "rho_curve_E": simplex_res["rho_curve"], "theta_curve": {}, "events": events}
    valid_thetas = {t: r for t, r in theta_curve.items() if r is not None}
    if 0 not in valid_thetas:
        events.append("[ERROR] [SMap] theta=0 linear baseline rho unavailable; cannot compute delta_rho.")
        return {"delta_rho": None, "theta_best": None, "e_best": e_best,
                "rho_theta0": None, "rho_theta_best": None,
                "rho_curve_E": simplex_res["rho_curve"], "theta_curve": theta_curve, "events": events}
    if not valid_thetas:
        events.append("[ERROR] [SMap] All theta rho values unavailable.")
        return {"delta_rho": None, "theta_best": None, "e_best": e_best,
                "rho_theta0": None, "rho_theta_best": None,
                "rho_curve_E": simplex_res["rho_curve"], "theta_curve": theta_curve, "events": events}
    theta_best = max(valid_thetas, key=lambda t: valid_thetas[t])
    rho_theta0 = valid_thetas[0]
    rho_best = valid_thetas[theta_best]
    delta_rho = rho_best - rho_theta0
    events.append(
        f"[INFO] [SMap] Theta sweep theta_curve={ {k: (round(v, 4) if v is not None else None) for k, v in theta_curve.items()} }; "
        f"theta_best={theta_best} (rho={rho_best:.4f}) vs theta=0 baseline (rho={rho_theta0:.4f}); delta_rho={delta_rho:.4f}"
    )
    return {
        "delta_rho": delta_rho,
        "theta_best": theta_best,
        "e_best": e_best,
        "rho_theta0": rho_theta0,
        "rho_theta_best": rho_best,
        "rho_curve_E": simplex_res["rho_curve"],
        "theta_curve": theta_curve,
        "events": events,
    }


def extract_contiguous_runs(values, min_len=1):
    """Splits a full-length array (with None at masked/missing positions --
    e.g. the output of `_wind_tunnel_common.slice_by_period`) into maximal
    contiguous runs of non-None values. A run never bridges a None, so it
    never bridges a calendar-night boundary (already masked to None by
    slice_by_period) NOR a genuine sensor dropout within a night. Returns a
    list of (start_index_in_original_array, [values...]) tuples with
    len(values) >= min_len.
    """
    runs = []
    cur_start = None
    cur_vals = []
    for i, v in enumerate(values):
        if v is not None:
            if cur_start is None:
                cur_start = i
            cur_vals.append(v)
        else:
            if cur_start is not None and len(cur_vals) >= min_len:
                runs.append((cur_start, cur_vals))
            cur_start = None
            cur_vals = []
    if cur_start is not None and len(cur_vals) >= min_len:
        runs.append((cur_start, cur_vals))
    return runs


def _median(arr):
    if not arr:
        return None
    s = sorted(arr)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0


def compute_subject_smap(night_values, tp, e_max=8, thetas=(0, 0.5, 1, 2, 3, 4, 6, 8), min_lib=20, min_lib_ratio=4):
    """Subject-level aggregation: runs compute_smap_delta_rho() independently on
    EVERY contiguous night run (see module docstring's No-Cross-Night-Pooling
    note), then takes the MEDIAN of delta_rho/theta_best/e_best across nights
    that produced a valid result -- mirroring the existing per-night-then-
    median aggregation pattern already established for Night AR1
    (`_legacy_metrics_v4.py::compute_critical_slowing_down`), for consistency
    of aggregation philosophy across all "per-night" L2/candidate operators.

    `night_values` must already be period='night'-sliced (see
    `_wind_tunnel_common.slice_by_period`) -- this module does not re-derive
    period-slicing itself, consistent with the Calculation Firewall boundary
    (this module receives fully-prepared numeric arrays only, never raw
    subject records). Per the [v1.1] Hall smoke-test finding, callers should
    pass the RAW (not smooth) track -- the smooth track saturates to a
    ceiling rho~1.0 regardless of theta/tp and carries no usable signal.
    """
    min_run_len = tp + max(min_lib, min_lib_ratio * (e_max + 1)) + 2
    runs = extract_contiguous_runs(night_values, min_len=min_run_len)
    events = [f"[INFO] [SMap] {len(runs)} contiguous night run(s) >= {min_run_len} points found."]
    if not runs:
        events.append(f"[ERROR] [SMap] No contiguous run reached the min_run_len={min_run_len} floor; subject excluded.")
        return {"delta_rho": None, "theta_best": None, "e_best": None, "n_nights_used": 0, "per_night": [], "events": events}

    per_night = []
    for start_idx, vals in runs:
        res = compute_smap_delta_rho(vals, tp, e_max=e_max, thetas=thetas, min_lib=min_lib, min_lib_ratio=min_lib_ratio)
        res["start_idx"] = start_idx
        res["n_points"] = len(vals)
        per_night.append(res)
        events.extend(res["events"])

    valid = [r for r in per_night if r["delta_rho"] is not None]
    if not valid:
        events.append("[ERROR] [SMap] All contiguous runs failed to produce a valid delta_rho; subject excluded.")
        return {"delta_rho": None, "theta_best": None, "e_best": None,
                "n_nights_used": 0, "per_night": per_night, "events": events}

    delta_rho_med = _median([r["delta_rho"] for r in valid])
    theta_best_med = _median([r["theta_best"] for r in valid])
    e_best_med = _median([r["e_best"] for r in valid])
    events.append(
        f"[INFO] [SMap] Subject-level median across {len(valid)}/{len(runs)} valid night runs: "
        f"delta_rho={delta_rho_med:.4f}, theta_best_median={theta_best_med}, e_best_median={e_best_med}"
    )
    return {
        "delta_rho": delta_rho_med,
        "theta_best": theta_best_med,
        "e_best": e_best_med,
        "n_nights_used": len(valid),
        "n_nights_total": len(runs),
        "per_night": per_night,
        "events": events,
    }
