"""
Wind-Tunnel Grid-Widening Check: S-Map E_max/theta_max ceiling-pinning diagnostic
on Hall (57 subjects, non-independent sample).

Path A (2026-08-20 user decision, see `reports/experiment_20260820_1630_smap_
phase_c_kickoff.md` Section 4): before spending a second independent cohort on
S-Map, first confirm on Hall (non-independent, cheap to re-run) that widening
the search grid (E_max, theta_max) actually stops e_best/theta_best from
pinning at the ceiling -- the artifact discovered in the Stanford SSPG run
(27/29 subjects pinned at E_max=theta_max=8 under the OLD flat min_lib=20).

This script does NOT decide graduation. It only answers: "at tp=10 (the
value already locked from the prior Hall tp-sweep), does the [v1.2] E-scaled
min_lib_ratio floor + a wider grid produce an INTERIOR optimum (not pinned at
the new ceiling) for the large majority of subjects?" If yes, the resulting
FIXED (E_max, theta grid, min_lib_ratio) configuration is what gets carried,
untouched, into the next independent cohort (Shanghai T2DM) -- widening
further AFTER seeing that cohort's result would be the forbidden "resuscitation"
move (`The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md` Section 2 "拒绝抢救失效指标").
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
import _extracted_tensor_engine_v4 as eng
import _smap_engine_v1 as sm

TAU_MAX = 120
TP = 10  # carried over unchanged from the prior Hall tp-sweep decision
E_MAX = 16
THETAS = (0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16)
MIN_LIB = 20
MIN_LIB_RATIO = 4


def run_subject_smap(subject):
    sid = subject["id"]
    try:
        ts, raw_vs = wt.resample_raw(subject["timestamps"], subject["values"])
    except Exception as e:
        return {"id": sid, "error": f"[HARNESS ERROR] resample_raw crashed: {e}"}

    valid_n = sum(1 for v in raw_vs if v is not None)
    if valid_n < 60:
        return {"id": sid, "error": f"[L0 ERROR] Insufficient valid points after resample ({valid_n})."}

    night_raw_vs = wt.slice_by_period(ts, raw_vs, "night")

    try:
        smap_res = sm.compute_subject_smap(
            night_raw_vs, TP, e_max=E_MAX, thetas=THETAS, min_lib=MIN_LIB, min_lib_ratio=MIN_LIB_RATIO
        )
    except Exception as e:
        return {"id": sid, "error": f"[SMAP HARNESS ERROR] compute_subject_smap crashed: {type(e).__name__}: {e}"}

    return {
        "id": sid,
        "delta_rho": smap_res.get("delta_rho"),
        "theta_best": smap_res.get("theta_best"),
        "e_best": smap_res.get("e_best"),
        "n_nights_used": smap_res.get("n_nights_used"),
        "n_nights_total": smap_res.get("n_nights_total"),
    }


def main():
    with open("output/phase_screening_subjects.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    hall = data["hall"]

    results = [run_subject_smap(s) for s in hall]
    ok = [r for r in results if "error" not in r and r.get("delta_rho") is not None]
    failed = [r for r in results if r not in ok]
    print(f"[Grid Check tp={TP}, E_max={E_MAX}, thetas={THETAS}] {len(ok)}/{len(hall)} produced delta_rho.")

    e_bests = [r["e_best"] for r in ok]
    theta_bests = [r["theta_best"] for r in ok]
    pinned_e = sum(1 for e in e_bests if e is not None and e >= E_MAX - 0.001)
    pinned_theta = sum(1 for t in theta_bests if t is not None and t >= THETAS[-1] - 0.001)
    print(f"  e_best pinned at ceiling ({E_MAX}): {pinned_e}/{len(ok)}")
    print(f"  theta_best pinned at ceiling ({THETAS[-1]}): {pinned_theta}/{len(ok)}")
    print(f"  e_best distribution: { {e: e_bests.count(e) for e in sorted(set(e_bests))} }")
    print(f"  theta_best distribution: { {t: theta_bests.count(t) for t in sorted(set(theta_bests))} }")
    deltas = sorted(r["delta_rho"] for r in ok)
    n = len(deltas)
    print(f"  delta_rho: min={deltas[0]:.4f} median={deltas[n//2]:.4f} max={deltas[-1]:.4f}")
    for r in failed:
        print(f"  FAILED/EXCLUDED {r['id']}: {r.get('error', 'no valid delta_rho')}")


if __name__ == "__main__":
    main()
