"""
Wind-Tunnel Experiment Driver -- Group 1 legacy-metrics smoke test (AGENTS.md Section 3.3/9,
dataset_fleet_registry.md changelog "第一组6张中性卡的冗余审计基础设施搭建").
Cohort: Hall (57 subjects, output/phase_screening_subjects.json)

Runs the 6 purely-neutral legacy-JS-ported candidate metrics (Volume / Recovery / Shape Ratio
λ1/λ2 / Box-Counting Dimension / Lyapunov / Core Dist, via _legacy_metrics_group1_v4.py +
_wind_tunnel_common.run_subject_legacy_group1) against Hall for infrastructure smoke-testing
ONLY (does the port run end-to-end on real data without crashing, what do the raw
distributions look like). No color-coded thresholds exist for these 6 cards to begin with, so
there is no circularity concern here (unlike Group 2's Hall re-run) -- this is purely an
engineering sanity check before the redundancy audit on Stanford SSPG / Shanghai T2DM.
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

    results, ok, failed = wt.run_cohort_legacy_group1("hall", hall, period=PERIOD)
    wt.write_results("hall_legacymetrics_group1", hall, results, ok, failed, PERIOD, TAU_MAX)


if __name__ == "__main__":
    main()
