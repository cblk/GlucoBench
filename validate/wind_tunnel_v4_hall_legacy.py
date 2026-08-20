"""
Wind-Tunnel Experiment Driver -- Group 2 legacy-metrics test plan (AGENTS.md Section 3.3/9,
dataset_fleet_registry.md changelog "开始进行第二组6张判色卡的风洞测试基础设施搭建").
Cohort: Hall (57 subjects, output/phase_screening_subjects.json)

Runs the 6 legacy-JS-ported candidate metrics (Early Phase Delay / Relaxation Time / AR1 /
Angular Velocity / Ascend Friction / Night Friction, via _legacy_metrics_v4.py +
_wind_tunnel_common.run_subject_legacy) against Hall -- the SAME cohort these thresholds were
partly calibrated on historically (v8.3/v8.4 era comments in index_v4.html reference
"Hall 校准" explicitly). Per Section 9.1.2, this first pass on Hall is diagnostic infrastructure
smoke-testing ONLY (does the port run end-to-end on real data without crashing, what do the
raw distributions look like) -- it must NOT be read as an independent validation, since a
"positive" result here would be circular (testing a metric on the same cohort it was
originally threshold-fit on). Independent validation requires an OUT-of-sample cohort
(Stanford/CGMacros/Shanghai/T1D-UOM/BIG IDEAs), run separately.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120


def main():
    with open("output/phase_screening_subjects.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    hall = data["hall"]

    results, ok, failed = wt.run_cohort_legacy("hall", hall, period=PERIOD)
    wt.write_results("hall_legacymetrics", hall, results, ok, failed, PERIOD, TAU_MAX)


if __name__ == "__main__":
    main()
