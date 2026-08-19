"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: T1D-UOM (17 subjects, 29-177 day longitudinal multimodal T1D recordings)
Source: University of Manchester, Zenodo DOI 10.5281/zenodo.15169263

Wind-Tunnel Use-Case Forge (The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md Section 4,
六段式), filled BEFORE writing the computation below:

  1. [物理目标] 追查 mcPHASES 配对报告（2026-08-16）遗留的开放行动项：`dim`
     （嵌入维度）在 Stanford SSPG 静息相比较中独立露出了与 DI/HbA1c 的显著相关，
     是否是"代偿负荷"的一个可复现指纹？本次用 T1D-UOM 的 90+ 天纵向数据，检验
     同一肉身的机械拉扯负荷（周活动量）高低是否会引起 Work Integral 或嵌入维度
     的可辨识位移——这是 Vector 3（机械拉扯）干预矢量在真实自由生活数据里的
     自然对照，而非人为施加的干预实验。
  2. [动力学限制] 夜间相（period='night', tau_max=60），与既往队列同源。**但
     必须显式声明**：T1D 受试者夜间仍有持续的基础胰岛素输注（泵或长效制剂），
     不同于 T2D/非糖尿病队列"夜间=无外源扰动的静息态"这一假设——本次夜间相
     测的是"外源胰岛素输注下的相对稳态"，不是纯内源代谢反馈。
  3. [因果手性限制] 高/低活动周的比较是**同一受试者内部的相对排序**（按该
     受试者自己的周活动总量做中位数分割），不是跨受试者比较，天然避免了 T1D
     队列被禁止跨池比较的红线。但活动量本身可能随季节/训练适应性存在单向漂移
     （如队列覆盖 2023年10月-2024年8月，冬季活动量本来就会系统性偏低，这是
     环境驱动而非"意愿"），本报告不会把"高活动周"简单等同于"主动运动干预"。
  4. [预判的失效点] 若机械拉扯确实拉低了迟滞做功或改变了嵌入维度，预期高活动周
     的 Work Integral/Dim 应与低活动周出现方向一致的位移；若被基础胰岛素的
     代偿性调整（T1D 患者常在高活动日主动下调基础剂量以防低血糖）完全抹平，
     则会看到零信号或方向不稳定的信号——且需要用周基础胰岛素总量的配对差异
     去检验"胰岛素调整"这个竞争性解释是否成立。
  5. [第一性原理裁决] 见 analyze_t1d_uom_paired.py 生成的报告。
  6. [热力学与残差账单] 周尺度聚合抹平了日内层面进食/大剂量胰岛素与运动时段的
     精确时序对齐；仅 15/17 受试者有 Nutrition 数据，本次未纳入；3/17 受试者
     无 Basal 数据，其高/低活动周的胰岛素混杂检验将诚实记录为 None，不做填补。

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: weekly step_count/active_Kcal/
    basal_dose_total are attached purely as metadata, never fed into the
    computation, never used as fit targets.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: uses shared _wind_tunnel_common.py.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 60
MIN_WEEK_POINTS = 30  # already enforced at export time; re-checked here for clarity, no new threshold invented


def weeks_to_pseudo_subjects(subject):
    """Each subject's pre-segmented weekly records (from export_t1d_uom_subjects.py)
    become independent run_subject() calls, exactly mirroring the mcPHASES phase
    driver's segment-then-run pattern (Section 9.4: reuse the same harness, only
    the segmentation step differs per cohort)."""
    pseudo_subjects = []
    for w in subject["weeks"]:
        if len(w["values"]) < MIN_WEEK_POINTS:
            continue
        pseudo = {
            "cohort": "t1d_uom",
            "id": f"{subject['id']}_wk_{w['week_start'][:10]}",
            "original_id": subject["id"],
            "subject_num": subject["subject_num"],
            "week_start": w["week_start"],
            "week_end": w["week_end"],
            "weekly_step_count_total": w["weekly_step_count_total"],
            "weekly_active_kcal_total": w["weekly_active_kcal_total"],
            "weekly_basal_dose_total": w["weekly_basal_dose_total"],
            "n_activity_records": w["n_activity_records"],
            "timestamps": w["timestamps"],
            "values": w["values"],
        }
        pseudo_subjects.append(pseudo)
    return pseudo_subjects


def main():
    data_path = Path("output/t1d_uom_subjects.json")
    if not data_path.exists():
        import export_t1d_uom_subjects
        export_t1d_uom_subjects.export_t1d_uom_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]
    print(f"Loaded {len(subjects)} T1D-UOM subjects from {data_path}.")

    all_weeks = []
    for s in subjects:
        all_weeks.extend(weeks_to_pseudo_subjects(s))
    print(f"Segmented into {len(all_weeks)} weekly runs (>= {MIN_WEEK_POINTS} raw points each).")

    results, ok, failed = wt.run_cohort("t1d_uom_activity", all_weeks, period=PERIOD)
    out_file = wt.write_results("t1d_uom_activity", all_weeks, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
