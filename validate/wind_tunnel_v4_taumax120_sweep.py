"""
Supplementary (non-production) sweep: re-run ALL remaining tau-dependent
wind-tunnel cohorts with a corrected tau-extraction window (max_lag=120),
per Option 2 of wind_tunnel_shanghai_t1dm_20260819_1200_tau_max_boundary_calibration.md,
now formally selected by the human architect for a full-fleet comparison
(2026-08-19 15:xx decision, AGENTS.md Section 3.3 / Section B.5).

SCOPE -- which cohorts this script covers, and which it does NOT:
  Of the 9 already-tested cohorts referenced in the calibration report
  (Hall/Colas/Stanford SSPG/CGMacros/Shanghai T2DM/Shanghai T1DM/BIG IDEAs/
  T1D-UOM/mcPHASES), only 8 are re-run here. Shanghai T1DM is EXCLUDED
  because it already has its own dedicated supplementary re-run
  (wind_tunnel_v4_shanghai_t1dm_taumax120.py ->
  reports/wind_tunnel_shanghai_t1dm_night_taumax120_SUPPLEMENTARY_20260819_1209.json).
  Re-running it again here would duplicate that work and risk a second,
  possibly-diverging result for the same cohort (Section 9.4 Bit-for-Bit
  discipline: one result per cohort per parameter set, not two).

  Stanford OGTT / CGMacros meals / BIG IDEAs meals are NOT tau-dependent at
  all (they compute meal-perturbation dynamics -- delta_g / strain_per_carb /
  work_meal -- directly from timestamped peak/threshold/integral arithmetic,
  never calling extract_tau). They are correctly excluded from this sweep;
  changing tau_max has zero effect on their math.

WHY THIS IS STILL NOT A PRODUCTION CHANGE:
  This script calls the SAME shared harness (_wind_tunnel_common.run_cohort)
  and the SAME engine method (_extracted_tensor_engine_v4.extract_tau) that
  every production-mirroring driver (wind_tunnel_v4_hall.py, etc.) uses --
  it only supplies tau_max=120 instead of the default 60. Both files' default
  behavior (tau_max=60, called with no override) was regression-tested
  byte-for-byte against the historical Hall/Colas archives before this sweep
  was written; see the regression note in candidate_tensor_staging_matrix.md /
  dataset_fleet_registry.md changelog dated 2026-08-19. index_v4.html's
  mirrored PYTHON_ENGINE_CODE has NOT been touched.

  Every output file below is written ALONGSIDE (never replacing) the
  existing production-default (taumax60) archive for that cohort, tagged
  with taumax120 in the filename by the shared write_results() helper --
  both are kept, per Section 8.2 Honest Fail-Closed / v1.4 No History
  Truncation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt  # noqa: E402

TAU_MAX_CORRECTED = 120
PERIOD = "night"


def _load(path_str):
    path = Path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sweep_hall_colas():
    data = _load("output/phase_screening_subjects.json")
    out = {}
    for name in ("hall", "colas"):
        subjects = data[name]
        print(f"[{name}] loaded {len(subjects)} subjects.")
        results, ok, failed = wt.run_cohort(name, subjects, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
        out_file = wt.write_results(name, subjects, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
        print(f"[{name}] {len(ok)}/{len(subjects)} succeeded -> {out_file}")
        out[name] = (results, ok, failed, out_file)
    return out


def sweep_stanford_sspg():
    data = _load("output/stanford_sspg_subjects.json")
    subjects = data["subjects"]
    print(f"[stanford_sspg] loaded {len(subjects)} subjects.")
    results, ok, failed = wt.run_cohort("stanford_sspg", subjects, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("stanford_sspg", subjects, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[stanford_sspg] {len(ok)}/{len(subjects)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def sweep_cgmacros_night():
    data = _load("output/cgmacros_subjects.json")
    subjects = data["subjects"]
    print(f"[cgmacros_night] loaded {len(subjects)} subjects.")
    results, ok, failed = wt.run_cohort("cgmacros_night", subjects, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("cgmacros_night", subjects, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[cgmacros_night] {len(ok)}/{len(subjects)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def sweep_shanghai_t2dm():
    data = _load("output/shanghai_t2dm_subjects.json")
    subjects = data["subjects"]
    print(f"[shanghai_t2dm] loaded {len(subjects)} recordings.")
    results, ok, failed = wt.run_cohort("shanghai_t2dm", subjects, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("shanghai_t2dm", subjects, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[shanghai_t2dm] {len(ok)}/{len(subjects)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def sweep_big_ideas():
    data = _load("output/big_ideas_subjects.json")
    subjects = data["subjects"]
    print(f"[big_ideas] loaded {len(subjects)} subjects.")
    results, ok, failed = wt.run_cohort("big_ideas", subjects, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("big_ideas", subjects, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[big_ideas] {len(ok)}/{len(subjects)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def sweep_t1d_uom_activity():
    # Reuse the EXACT weekly-segmentation helper from the production driver
    # (Section 9.4: the segmentation step is structural, not mathematical,
    # but it must still not be re-typed/duplicated -- import, don't copy).
    import wind_tunnel_v4_t1d_uom_activity as drv

    data = _load("output/t1d_uom_subjects.json")
    subjects = data["subjects"]
    all_weeks = []
    for s in subjects:
        all_weeks.extend(drv.weeks_to_pseudo_subjects(s))
    print(f"[t1d_uom_activity] loaded {len(subjects)} subjects -> {len(all_weeks)} weekly runs.")
    results, ok, failed = wt.run_cohort("t1d_uom_activity", all_weeks, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("t1d_uom_activity", all_weeks, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[t1d_uom_activity] {len(ok)}/{len(all_weeks)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def sweep_mcphases_phase():
    import wind_tunnel_v4_mcphases_phase as drv

    data = _load("output/mcphases_subjects.json")
    subjects = data["subjects"]
    all_segments = []
    for s in subjects:
        all_segments.extend(drv.segment_by_phase(s))
    print(f"[mcphases_phase] loaded {len(subjects)} subjects -> {len(all_segments)} phase segments.")
    results, ok, failed = wt.run_cohort("mcphases_phase", all_segments, period=PERIOD, tau_max=TAU_MAX_CORRECTED)
    out_file = wt.write_results("mcphases_phase", all_segments, results, ok, failed, PERIOD, TAU_MAX_CORRECTED)
    print(f"[mcphases_phase] {len(ok)}/{len(all_segments)} succeeded -> {out_file}")
    return results, ok, failed, out_file


def main():
    print(f"=== tau_max={TAU_MAX_CORRECTED} full-fleet sweep (Option 2 comparison run) ===")
    print("Shanghai T1DM excluded (already has its own dedicated taumax120 supplementary run).")
    print("Stanford OGTT / CGMacros meals / BIG IDEAs meals excluded (not tau-dependent).\n")

    sweep_hall_colas()
    sweep_stanford_sspg()
    sweep_cgmacros_night()
    sweep_shanghai_t2dm()
    sweep_big_ideas()
    sweep_t1d_uom_activity()
    sweep_mcphases_phase()

    print("\n=== Sweep complete. All outputs written to reports/ with taumax120 tag, "
          "alongside (not replacing) the existing taumax60 archives. ===")


if __name__ == "__main__":
    main()
