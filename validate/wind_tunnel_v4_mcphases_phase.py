"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: mcPHASES (42 subjects, PhysioNet Restricted Health Data, Interval 1 only)
Source: https://physionet.org/content/mcphases/1.0.0/

Design note -- WHY this driver looks different from every other wind_tunnel_v4_*.py:
  Every prior cohort (Hall/Colas/Stanford/CGMacros/...) carries STATIC per-subject
  labels (SSPG, A1C, diagnosis) that never change within a subject's recording
  window, so run_subject() is called exactly once per subject and results are
  grouped ACROSS subjects.

  mcPHASES's `phase` (menstrual cycle: Follicular/Fertility/Luteal/Menstrual) is
  TIME-VARYING -- it cycles roughly every ~28 days within a single subject's
  ~88-day recording window. Grouping by phase across subjects would throw away
  the far more powerful comparison this dataset actually affords: the SAME BODY
  measured under different endocrine states (AGENTS.md Section 7 "同一肉身跨周期
  对比" -- natural Epoch0/Epoch1 pairs with zero inter-subject baseline noise).

  So this driver SEGMENTS each subject's series into contiguous same-phase runs
  BEFORE calling run_subject(), and calls run_subject() once per segment, not
  once per subject. Everything downstream of that (the actual math) is
  byte-for-byte identical Section 9.4 harness code -- only the segmentation
  step is new, and it is pure index/label arithmetic, not a math operator.

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: `phase` is NEVER passed into
    extract_tau / estimate_dimension / compute_rqa / compute_work_integral.
    It only determines where a segment BOUNDARY falls (structural, not
    mathematical), then rides along as passthrough metadata on the result
    record (identical mechanism _wind_tunnel_common.run_subject() already
    uses for SSPG/A1C/diagnosis on other cohorts).
  - Section 9.1.2 Labels as Prisms, Not Targets: phase is a grouping prism,
    never a fit target.
  - Section 8.3 Zero Magic-Constant: segment admission uses NO invented
    threshold. Every segment, however short, is submitted to run_subject();
    the harness's own pre-existing, already-justified checks
    (valid_n<60 / probe_valid_n<30 / embedding-length<10) decide pass/fail
    and report a per-segment [L0/L1/L2 ERROR] reason. The only prefilter here
    (MIN_RAW_POINTS=10) exists to skip zero/near-zero-length degenerate runs
    before wasting a harness call, not to pre-judge scientific validity.
  - Section 8.1 No Inference & No Fabrication: a day with phase=None (missing
    self-report) ends a segment rather than being bridged/interpolated across.
    We do not guess what phase a day was.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: uses shared _wind_tunnel_common.py.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 60
MIN_RAW_POINTS = 10  # skip only degenerate near-empty runs; see docstring


def segment_by_phase(subject):
    """Split one subject's (timestamps, values, phase) into contiguous
    same-non-None-phase runs. Returns a list of pseudo-subject dicts shaped
    exactly like what run_subject() expects, with all original subject
    metadata (minus timestamps/values/id/phase) plus segment-specific fields
    carried through as passthrough prism metadata.
    """
    ts = subject["timestamps"]
    vs = subject["values"]
    phases = subject["phase"]
    n = len(ts)

    segments = []
    seg_start = None
    for i in range(n + 1):
        cur_phase = phases[i] if i < n else None
        boundary = (i == n) or (cur_phase != (phases[seg_start] if seg_start is not None else None))
        if seg_start is None:
            if i < n and cur_phase is not None:
                seg_start = i
            continue
        if boundary:
            seg_end = i  # exclusive
            if seg_end - seg_start >= MIN_RAW_POINTS:
                segments.append((seg_start, seg_end, phases[seg_start]))
            seg_start = i if (i < n and cur_phase is not None) else None

    pseudo_subjects = []
    for k, (start, end, phase) in enumerate(segments):
        seg_ts = ts[start:end]
        seg_vs = vs[start:end]
        n_days = round(
            (
                __import__("datetime").datetime.fromisoformat(seg_ts[-1])
                - __import__("datetime").datetime.fromisoformat(seg_ts[0])
            ).total_seconds()
            / 86400.0,
            2,
        )
        pseudo = {k2: v for k2, v in subject.items() if k2 not in ("timestamps", "values", "id", "phase")}
        pseudo["timestamps"] = seg_ts
        pseudo["values"] = seg_vs
        pseudo["id"] = f"{subject['id']}_seg{k:02d}_{phase}"
        pseudo["original_id"] = subject["id"]
        pseudo["phase"] = phase
        pseudo["segment_index"] = k
        pseudo["segment_n_points"] = end - start
        pseudo["segment_n_days"] = n_days
        pseudo_subjects.append(pseudo)
    return pseudo_subjects


def main():
    data_path = Path("output/mcphases_subjects.json")
    if not data_path.exists():
        import export_mcphases_subjects
        export_mcphases_subjects.export_mcphases_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]
    print(f"Loaded {len(subjects)} mcPHASES subjects from {data_path}.")

    all_segments = []
    for s in subjects:
        all_segments.extend(segment_by_phase(s))
    print(f"Segmented into {len(all_segments)} contiguous same-phase runs (>= {MIN_RAW_POINTS} raw points each).")

    results, ok, failed = wt.run_cohort("mcphases_phase", all_segments, period=PERIOD)
    out_file = wt.write_results("mcphases_phase", all_segments, results, ok, failed, PERIOD, TAU_MAX)

    # Informational summary only -- NOT a doctrine conclusion. The Homomorphic
    # Anchor Forge report (written by the LLM Navigator afterward from this
    # JSON) is where residuals get interpreted, per Section B.5.
    by_phase = defaultdict(list)
    for r in ok:
        by_phase[r.get("phase")].append(r)
    print("\nPer-phase segment yield (successful runs only):")
    for phase, rs in sorted(by_phase.items(), key=lambda kv: -len(kv[1])):
        wis = [r["workIntegral"] for r in rs if r.get("workIntegral") is not None]
        if wis:
            print(f"  {phase:12s} n_segments={len(rs):3d}  Work Integral mean={sum(wis)/len(wis):.2f}")
        else:
            print(f"  {phase:12s} n_segments={len(rs):3d}  (no Work Integral values)")

    by_subject = defaultdict(set)
    for r in ok:
        by_subject[r.get("original_id")].add(r.get("phase"))
    n_multi_phase = sum(1 for phases in by_subject.values() if len(phases) >= 2)
    print(
        f"\nSubjects with >=2 distinct successful phases (eligible for same-body "
        f"paired contrast): {n_multi_phase} / {len(by_subject)}"
    )

    print(f"\nWind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
