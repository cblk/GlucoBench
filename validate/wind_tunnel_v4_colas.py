"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: Colas (208 subjects, output/phase_screening_subjects.json)

Thin per-cohort driver, same shared harness as wind_tunnel_v4_hall.py (see
_wind_tunnel_common.py for doctrine compliance notes). Colas subjects only
carry the `y` label (diagnosis/insulin/SSPG are None for this cohort per
output/phase_screening_subjects.json's own schema) -- that absence is
preserved as None, never fabricated (Section 8.1).

Purpose of this run: replicate the Hall-cohort Work Integral protocol
(period='night', tau_max=60) on a larger, independent cohort to test
whether the directionally-consistent-but-unresolved Work Integral signal
from reports/wind_tunnel_hall_20260815_2212_night_taumax60.md reproduces
outside the Hall sample, per Section 9.3 Topological Victory standard.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120  # 2026-08-19/Blueprint v3.6: tracks production index_v4.html's max_lag default


def main():
    with open("output/phase_screening_subjects.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    colas = data["colas"]

    results, ok, failed = wt.run_cohort("colas", colas, period=PERIOD)
    wt.write_results("colas", colas, results, ok, failed, PERIOD, TAU_MAX)


if __name__ == "__main__":
    main()
