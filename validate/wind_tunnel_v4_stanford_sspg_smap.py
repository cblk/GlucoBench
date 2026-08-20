"""
Wind-Tunnel Experiment Driver -- S-Map Delta-rho FIRST genuine out-of-sample test
(AGENTS.md Section 3.3/9, Section 4.B stage 3 "风洞对撞" second half).
Cohort: Stanford Metabolic Subphenotype (29 subjects with Home CGM + SSPG gold standard).

Per `reports/wind_tunnel_v4_1_to_v4_2_refactoring_roadmap_20260820.md` Section 4.3 /
Section 3 Phase C: "先在 Hall 做工程实现 + 烟雾测试...再选一个从未被此算子测过的队列做真正的
样本外分组检验". Stanford SSPG has never been touched by S-Map before this run, so a
signal here is non-circular. tp=10 (30 minutes) is carried over from the Hall fleet-wide
tp-sweep survey (`reports/wind_tunnel_hall_smap_tp_sweep_*.json`) -- see that report's
reasoning for why tp=10 was picked as the first candidate to spend an independent cohort on
(NOT re-derived/refit on this cohort, which would be circular).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: `_smap_engine_v1` never reads sspg/sspg_class/di.
  - Section 9.1.2 Labels as Prisms, Not Targets: sspg / sspg_class / di / hba1c / fpg / bmi are
    attached purely as metadata for the POST-HOC analysis script, never fed into S-Map itself.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only, index_v4.html
    untouched (candidate #6 has not graduated -- see candidate_tensor_staging_matrix.md).
"""
import json
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt
import _extracted_tensor_engine_v4 as eng
import _smap_engine_v1 as sm

TAU_MAX = 120
TP = 10  # 30 minutes; carried over from the Hall tp-sweep survey, see module docstring
E_MAX = 8
THETAS = (0, 0.5, 1, 2, 3, 4, 6, 8)
MIN_LIB = 20


def run_subject_smap(subject, tp=TP, tau_max=TAU_MAX, e_max=E_MAX, thetas=THETAS, min_lib=MIN_LIB):
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
    tau_locked = tau_res.get("result")

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
    # Labels: prism-only (Section 9.1.2), never fed into the computation above.
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def main():
    data_path = Path("output/stanford_sspg_subjects.json")
    if not data_path.exists():
        import export_stanford_sspg_subjects
        export_stanford_sspg_subjects.export_stanford_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]
    print(f"Loaded {len(subjects)} Stanford subjects from {data_path}.")

    results = [run_subject_smap(s) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    with_delta = [r for r in ok if r["delta_rho"] is not None]
    print(f"Processed {len(subjects)} stanford_sspg subjects: {len(ok)} succeeded, {len(failed)} failed, "
          f"{len(with_delta)} produced non-None delta_rho.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")

    out_path = Path("reports")
    out_path.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_path / f"wind_tunnel_stanford_sspg_smap_tp{TP}_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "stanford_sspg", "tp": TP, "e_max": E_MAX, "thetas": list(THETAS), "min_lib": MIN_LIB,
            "n_total": len(subjects), "n_success": len(ok), "n_failed": len(failed),
            "n_with_delta_rho": len(with_delta),
            "results": results,
        }, f, indent=2)
    print(f"Wind tunnel run complete. Results saved to {out_file}")
    return out_file


if __name__ == "__main__":
    main()
