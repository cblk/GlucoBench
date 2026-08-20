"""
One-shot cross-check driver (2026-08-19): builds a single synthetic fixture, runs it through
BOTH the real JS (via Node.js + _js_legacy_metrics_crosscheck.mjs, a verbatim extraction of
the four legacy functions from index_v4.html) and the new Python port
(_legacy_metrics_v4.py), then diffs every field. This is infrastructure-only -- it does not
touch any real cohort, does not draw any clinical conclusion, and is not wired into any
production or wind-tunnel driver path (Section 9.5 Product Isolation). Its sole purpose is to
raise confidence that _legacy_metrics_v4.py is a faithful (not hand-transcription-error-prone)
port before it is ever pointed at real cohort data.

Run: python validate/crosscheck_legacy_metrics.py
Requires Node.js on PATH (checked at the top of this session via `node --version` -> v26.3.0).
"""
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _legacy_metrics_v4 as legacy

HERE = Path(__file__).parent
FIXTURE_PATH = HERE / "_crosscheck_fixture.json"
JS_OUT_PATH = HERE / "_crosscheck_js_output.json"


def build_fixture():
    """3 synthetic nights (30+ points each, mild AR(1) structure + noise) stitched to 3
    synthetic days containing two forced-excursion cycles each (rise > 1.5, then decay
    with a clean half-recovery point), sampled on a 3-minute grid, with a handful of None
    gaps sprinkled in to exercise every null-guard branch in all four ported functions.
    """
    import random

    random.seed(20260819)
    start = dt.datetime(2026, 1, 1, 0, 0, 0)
    times = []
    values = []

    t = start
    day_baseline = 5.5
    ar1_state = day_baseline
    for day in range(3):
        # Night block: 00:00-06:00, 3-min grid -> 120 points, mild AR(1) + noise so ar1 != 0.
        for _ in range(120):
            ar1_state = day_baseline + 0.7 * (ar1_state - day_baseline) + random.gauss(0, 0.15)
            times.append(t)
            values.append(round(ar1_state, 4))
            t += dt.timedelta(minutes=3)

        # Day block: 06:00-24:00, 3-min grid -> 360 points, two forced excursions (rise then
        # exponential-ish decay back toward baseline, each amplitude > 1.5 so the >1.5
        # excursion-threshold guard in computeExcursionKinetics fires).
        n_day_pts = 360
        for i in range(n_day_pts):
            phase = (i % 180) / 180.0
            if phase < 0.15:
                v = day_baseline + (phase / 0.15) * 4.0  # rise to +4.0 over 45min
            else:
                decay_frac = (phase - 0.15) / 0.85
                v = day_baseline + 4.0 * math.exp(-3.0 * decay_frac)
            v += random.gauss(0, 0.05)
            times.append(t)
            values.append(round(v, 4))
            t += dt.timedelta(minutes=3)

    # Sprinkle a handful of None gaps (every 97th point, avoiding first/last 2 samples so
    # every function's boundary-index arithmetic still has real neighbors to fall back on).
    for i in range(5, len(values) - 5, 97):
        values[i] = None

    # Independent synthetic 3D phase-space trajectory for compute_asymmetric_friction: a
    # slowly precessing spiral around core=[0,0,0] so both v_g>0 (ascending) and v_g<0
    # (descending) segments occur, plus every 40th point set to None to exercise the gap
    # guard. dim=3 matches the production Takens embedding's typical m=3-6.
    n_pts = 400
    core = [0.0, 0.0, 0.0]
    points = []
    for i in range(n_pts):
        theta = i * 0.15
        r = 2.0 + 0.5 * math.sin(i * 0.05)
        points.append([r * math.cos(theta), r * math.sin(theta), 0.3 * math.sin(i * 0.08)])
    for i in range(4, n_pts, 40):
        points[i] = None

    # Independent "night" phase-space trajectory (smaller, denser spiral) + its own core,
    # for the second compute_asymmetric_friction call site (Night Friction).
    n_night_pts = 150
    night_core = [0.1, -0.1, 0.0]
    night_points = []
    for i in range(n_night_pts):
        theta = i * 0.22
        r = 1.0 + 0.3 * math.sin(i * 0.09)
        night_points.append([r * math.cos(theta) + 0.1, r * math.sin(theta) - 0.1, 0.15 * math.cos(i * 0.11)])
    for i in range(6, n_night_pts, 35):
        night_points[i] = None

    night_mean = day_baseline  # matches how nightMean is derived from raw night values in JS

    return {
        "timestamps": [tt.isoformat() for tt in times],
        "values": values,
        "points": points,
        "core": core,
        "nightPoints": night_points,
        "nightCore": night_core,
        "nightMean": night_mean,
    }


def run_js(fixture_path: Path, out_path: Path):
    subprocess.run(
        ["node", str(HERE / "_js_legacy_metrics_crosscheck.mjs"), str(fixture_path), str(out_path)],
        check=True,
        cwd=str(HERE),
    )
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_python(fixture: dict):
    times = [dt.datetime.fromisoformat(t) for t in fixture["timestamps"]]
    vals = fixture["values"]
    return {
        "criticalSlowingDown": legacy.compute_critical_slowing_down(times, vals),
        "ascendFriction": legacy.compute_asymmetric_friction(fixture["points"], fixture["core"]),
        "nightFriction": legacy.compute_asymmetric_friction(fixture["nightPoints"], fixture["nightCore"]),
        "excursionKinetics": legacy.compute_excursion_kinetics(times, vals),
        "keplerKinematics": legacy.compute_kepler_kinematics(times, vals, fixture["nightMean"]),
    }


def diff_values(path: str, js_v, py_v, tol=1e-9, mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(js_v, dict) and isinstance(py_v, dict):
        for k in js_v.keys() | py_v.keys():
            diff_values(f"{path}.{k}", js_v.get(k), py_v.get(k), tol, mismatches)
    elif js_v is None or py_v is None:
        if js_v != py_v:
            mismatches.append((path, js_v, py_v))
    elif isinstance(js_v, (int, float)) and isinstance(py_v, (int, float)):
        if not math.isclose(js_v, py_v, rel_tol=tol, abs_tol=tol):
            mismatches.append((path, js_v, py_v))
    else:
        if js_v != py_v:
            mismatches.append((path, js_v, py_v))
    return mismatches


def main():
    fixture = build_fixture()
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture ({len(fixture['values'])} raw samples) to {FIXTURE_PATH}")

    js_result = run_js(FIXTURE_PATH, JS_OUT_PATH)
    py_result = run_python(fixture)

    print("\n--- JS  result ---")
    print(json.dumps(js_result, indent=2))
    print("\n--- PY  result ---")
    print(json.dumps(py_result, indent=2))

    mismatches = diff_values("root", js_result, py_result)
    print("\n--- DIFF ---")
    if not mismatches:
        print("PASS: zero mismatches between JS and Python across all 4 ported functions.")
    else:
        print(f"FAIL: {len(mismatches)} mismatch(es):")
        for path, js_v, py_v in mismatches:
            print(f"  {path}: JS={js_v!r}  PY={py_v!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
