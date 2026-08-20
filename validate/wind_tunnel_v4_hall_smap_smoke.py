"""
Wind-Tunnel Smoke Test v2: S-Map Delta-rho `tp`-sweep on Hall (57 subjects).

AGENTS.md Section 4.B stage 3 ("风洞对撞") FIRST step per
`reports/wind_tunnel_v4_1_to_v4_2_refactoring_roadmap_20260820.md` Section 4.3 /
Section 3 Phase C: "先在 Hall 做工程实现 + 烟雾测试（非独立样本，只验证代码可跑）"。

THIS IS NOT A VALIDATION RUN. Hall has already been used to tune/audit multiple
other legacy metrics in this project's history, so any group-separation signal
observed here would NOT count toward Section 9.3's Topological Victory
criterion (independent, previously-untouched cohort required). The purpose of
THIS v2 script is:
  1. Confirm `_smap_engine_v1` still runs end-to-end without crashing on the
     RAW (not smooth) night track, which the v1 run below discovered is the
     required track (smooth saturates to a ceiling rho~1.0 regardless of
     theta -- see `_smap_engine_v1` module docstring [v1.1] note).
  2. Sweep the forecast horizon `tp` across a candidate grid and report the
     FLEET-WIDE distribution of delta_rho / n_nights_used at each `tp`, to
     give the human architect an evidence-based (not guessed) basis for
     choosing which `tp` to carry into the first real cross-cohort test
     (Phase C stage 3 second half). This is exploration on a non-independent
     cohort -- cheap and safe -- specifically to avoid burning a fresh,
     previously-untouched cohort's "one shot" on an unvalidated `tp` guess.

[v1 residual, superseded]: The FIRST version of this script (now overwritten,
git history retains it) ran tp=1 on the SMOOTH track and found delta_rho ==
0.0000 for all 57/57 Hall subjects -- a ceiling-saturation artifact, not "no
candidates have signal". See `reports/wind_tunnel_hall_smap_smoke_test_
20260820_1609.json` for the raw (superseded) v1 output, kept per Section 8.2
Honest Fail-Closed / v1.4 No-History-Truncation (the finding that motivated
the tp-sweep redesign must stay traceable, not be silently overwritten).

Reuses the SAME shared preprocessing harness (`_wind_tunnel_common`) as every
other wind-tunnel driver in this repo (Section 9.4 Bit-for-Bit Truth Across
Tracks) for resample/tau/smooth -- this script does NOT reimplement those
steps. `tau` is still extracted for record-keeping (comparability with the
production gauge) but is NOT fed into S-Map's own unit-lag embedding (see
module docstring for why the two tau conventions are deliberately distinct).
"""
import json
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
import _extracted_tensor_engine_v4 as eng
import _smap_engine_v1 as sm

TAU_MAX = 120  # production gauge, computed for record-keeping only (not fed to S-Map)
E_MAX = 8
THETAS = (0, 0.5, 1, 2, 3, 4, 6, 8)
MIN_LIB = 20
TP_GRID = (1, 3, 6, 10, 15, 20)  # 3/9/18/30/45/60 minutes at the 3-minute grid


def run_subject_smap(subject, tp, tau_max=TAU_MAX, e_max=E_MAX, thetas=THETAS, min_lib=MIN_LIB):
    sid = subject["id"]
    log = []
    try:
        ts, raw_vs = wt.resample_raw(subject["timestamps"], subject["values"])
    except Exception as e:
        return {"id": sid, "error": f"[HARNESS ERROR] resample_raw crashed: {e}"}

    valid_n = sum(1 for v in raw_vs if v is not None)
    if valid_n < 60:
        return {"id": sid, "error": f"[L0 ERROR] Insufficient valid points after resample ({valid_n})."}

    raw_json = json.dumps(raw_vs)
    tau_res = json.loads(eng.engine.extract_tau(raw_json, max_lag=tau_max))
    log += tau_res.get("events", [])
    tau_locked = tau_res.get("result")  # record-keeping only; not used by S-Map (see module docstring)

    night_raw_vs = wt.slice_by_period(ts, raw_vs, "night")

    try:
        smap_res = sm.compute_subject_smap(night_raw_vs, tp, e_max=e_max, thetas=thetas, min_lib=min_lib)
    except Exception as e:
        return {"id": sid, "error": f"[SMAP HARNESS ERROR] compute_subject_smap crashed: {type(e).__name__}: {e}", "events": log}
    log += smap_res.get("events", [])

    record = {
        "id": sid,
        "tp": tp,
        "tau_production_gauge": tau_locked,
        "delta_rho": smap_res.get("delta_rho"),
        "theta_best": smap_res.get("theta_best"),
        "e_best": smap_res.get("e_best"),
        "n_nights_used": smap_res.get("n_nights_used"),
        "n_nights_total": smap_res.get("n_nights_total"),
        "events": log,
    }
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def summarize(tp, results):
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    with_delta = [r for r in ok if r["delta_rho"] is not None]
    print(f"[tp={tp}] {len(ok)}/{len(results)} succeeded, {len(with_delta)}/{len(ok)} produced non-None delta_rho.")
    if with_delta:
        deltas = sorted(r["delta_rho"] for r in with_delta)
        n = len(deltas)
        nights_used = sorted(r["n_nights_used"] for r in with_delta)
        thetas_best = [r["theta_best"] for r in with_delta]
        print(f"        delta_rho: min={deltas[0]:.4f} p25={deltas[n // 4]:.4f} median={deltas[n // 2]:.4f} "
              f"p75={deltas[3 * n // 4]:.4f} max={deltas[-1]:.4f}")
        print(f"        n_nights_used: min={nights_used[0]} median={nights_used[len(nights_used) // 2]} max={nights_used[-1]}")
        print(f"        theta_best=0 (i.e. delta_rho==0, linear already optimal) count: {thetas_best.count(0)}/{len(with_delta)}")
    return ok, failed, with_delta


def main():
    with open("output/phase_screening_subjects.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    hall = data["hall"]

    all_results = {}
    for tp in TP_GRID:
        results = [run_subject_smap(s, tp) for s in hall]
        ok, failed, with_delta = summarize(tp, results)
        for r in failed:
            print(f"  FAILED tp={tp} {r['id']}: {r['error']}")
        all_results[tp] = results

    out_path = Path("reports")
    out_path.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_path / f"wind_tunnel_hall_smap_tp_sweep_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "hall", "purpose": "tp_sweep_smoke_test_non_independent_sample",
            "tp_grid": list(TP_GRID), "e_max": E_MAX, "thetas": list(THETAS), "min_lib": MIN_LIB,
            "n_total": len(hall),
            "results_by_tp": {str(tp): all_results[tp] for tp in TP_GRID},
        }, f, indent=2)
    print(f"Wrote raw per-subject results to {out_file}")
    return out_file


if __name__ == "__main__":
    main()
