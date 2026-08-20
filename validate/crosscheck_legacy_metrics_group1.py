"""
One-shot cross-check driver (2026-08-19) for the 6 Group-1 (neutral, no color-coding) legacy
JS attractor-geometry metrics: builds a synthetic phase-space fixture, runs it through BOTH
the real JS (via Node.js + _js_legacy_metrics_group1_crosscheck.mjs, a verbatim extraction from
index_v4.html) and the new Python port (_legacy_metrics_group1_v4.py), then diffs every field.
Infrastructure-only (Section 9.5 Product Isolation): does not touch any real cohort.

Run: python validate/crosscheck_legacy_metrics_group1.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _legacy_metrics_group1_v4 as g1

HERE = Path(__file__).parent
FIXTURE_PATH = HERE / "_crosscheck_group1_fixture.json"
JS_OUT_PATH = HERE / "_crosscheck_group1_js_output.json"


def build_fixture():
    """Synthetic 3D phase-space trajectories exercising all 6 operators:
      - shapePoints: a precessing spiral with slowly drifting radius (dim=3), enough variance
        spread across axes 0/1/2 to give a non-degenerate covariance matrix (exercises
        jacobiEigenvalues/Volume/Shape/box-counting), with periodic None gaps.
      - rawPoints: a noisier version of the same spiral (Recovery uses the RAW, unsmoothed
        track per the Dual-Track Law) with its own gaps.
      - smoothPoints: a cleaner, denser spiral (Lyapunov needs >=30 valid points with a
        next-step neighbor) with sparser gaps.
      - nightCore: an independent reference point for the Core-Dist crosscheck.
    """
    import random
    random.seed(20260819)

    def make_spiral(n, r0, dr_amp, z_amp, noise, gap_every, gap_offset):
        pts = []
        for i in range(n):
            theta = i * 0.13
            r = r0 + dr_amp * math.sin(i * 0.045)
            x = r * math.cos(theta) + random.gauss(0, noise)
            y = r * math.sin(theta) + random.gauss(0, noise)
            z = z_amp * math.sin(i * 0.07) + random.gauss(0, noise * 0.5)
            pts.append([x, y, z])
        for i in range(gap_offset, n, gap_every):
            pts[i] = None
        return pts

    shape_points = make_spiral(500, 3.0, 0.8, 1.2, 0.05, 41, 3)
    raw_points = make_spiral(500, 3.0, 0.8, 1.2, 0.15, 37, 5)
    smooth_points = make_spiral(500, 3.0, 0.8, 1.2, 0.02, 53, 7)
    night_core = [0.2, -0.3, 0.1]

    return {
        "shapePoints": shape_points,
        "rawPoints": raw_points,
        "smoothPoints": smooth_points,
        "nightCore": night_core,
    }


def run_js(fixture_path: Path, out_path: Path):
    subprocess.run(
        ["node", str(HERE / "_js_legacy_metrics_group1_crosscheck.mjs"), str(fixture_path), str(out_path)],
        check=True,
        cwd=str(HERE),
    )
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_python(fixture: dict):
    shape_points = fixture["shapePoints"]
    raw_points = fixture["rawPoints"]
    smooth_points = fixture["smoothPoints"]
    night_core = fixture["nightCore"]

    vs = g1.compute_volume_shape(shape_points)
    vs_out = None
    core_dist = None
    if vs is not None:
        vs_out = {
            "volume": vs["volume"],
            "shapeRatio": vs["shapeRatio"] if math.isfinite(vs["shapeRatio"]) else None,
            "gravityCore": vs["gravityCore"],
            "effectiveDim": vs["effectiveDim"],
        }
        core_dist = g1.calc_distance(vs["gravityCore"], night_core)

    return {
        "volumeShape": vs_out,
        "boxCountingDimension": g1.box_counting_dimension(shape_points),
        "lyapunovProxy": g1.lyapunov_proxy(smooth_points),
        "normalizedRecovery": g1.compute_normalized_recovery(raw_points),
        "coreDist": core_dist,
    }


def diff_values(path: str, js_v, py_v, tol=1e-9, mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(js_v, dict) and isinstance(py_v, dict):
        for k in js_v.keys() | py_v.keys():
            diff_values(f"{path}.{k}", js_v.get(k), py_v.get(k), tol, mismatches)
    elif isinstance(js_v, list) and isinstance(py_v, list):
        if len(js_v) != len(py_v):
            mismatches.append((path, js_v, py_v))
        else:
            for i in range(len(js_v)):
                diff_values(f"{path}[{i}]", js_v[i], py_v[i], tol, mismatches)
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
    print(f"Wrote fixture to {FIXTURE_PATH}")

    js_result = run_js(FIXTURE_PATH, JS_OUT_PATH)
    py_result = run_python(fixture)

    print("\n--- JS  result ---")
    print(json.dumps(js_result, indent=2))
    print("\n--- PY  result ---")
    print(json.dumps(py_result, indent=2))

    mismatches = diff_values("root", js_result, py_result)
    print("\n--- DIFF ---")
    if not mismatches:
        print("PASS: zero mismatches between JS and Python across all 6 Group-1 ported operators.")
    else:
        print(f"FAIL: {len(mismatches)} mismatch(es):")
        for path, js_v, py_v in mismatches:
            print(f"  {path}: JS={js_v!r}  PY={py_v!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
