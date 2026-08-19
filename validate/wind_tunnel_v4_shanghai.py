"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: Shanghai_T2DM (100 unique patients, 109 recordings, 2.6-13.9 day CGM windows)
Source: Wang et al. 2023 (Shanghai_T2DM/T1DM public archive)

Wind-Tunnel Use-Case Forge (The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md Section 4,
六段式), filled BEFORE writing the computation below, per doctrine mandate:

  1. [物理目标] 这次不是找“T2DM 预测率”，而是找 Work Integral 这把尺子本身的一个
     可疑的方法学伪影是否真实存在：dataset_fleet_registry.md 第五节记录的
     "短周期坍缩"猜想——即样本量越大、记录周期越短，Work Integral 的组间判别力
     是否会系统性坍缩（Colas: n=208, ~2天, AUC 0.699→0.563 vs Hall）。
  2. [动力学限制] 全部取夜间静息相（period='night', tau_max=60），与既往所有
     队列保持同源比较基准；不引入进食扰动（Shanghai 的 Dietary intake 是自由文本，
     无结构化碳水克数，不具备做 CGMacros 式急性冲击分析的条件）。
  3. [因果手性限制] Shanghai_T2DM 含 8 名患者的 2-3 次独立住院复诊记录（如
     "2001_0"/"2001_1"，相隔数月，治疗方案可能已调整）。这些复诊记录之间的时间箭头
     不可逆（后一次住院不是前一次的对称重复，可能反映病情演化或用药调整后的新状态）。
     本驱动脚本导出全部访次（不做筛选——机械提取原则），但下游横截面分组对比
     （analyze_shanghai_results.py）必须只取每位患者的首次访次，避免同一肉身的
     非独立重复样本污染秩分离度统计；多访次数据保留作为未来同一肉身纵向对比的
     候选（不在本轮报告范围内展开）。
  4. [预判的失效点] 若"短周期坍缩"是 Work Integral 的真实方法学伪影（而非
     Colas 特有的队列噪声），则本队列中 <7 天的短记录子集应复现出比 Hall/Stanford
     式长记录更弱的 HbA1c 分组撕裂度；若 >=10 天子集的撕裂度显著优于 <7 天子集，
     则证实了坍缩假说；若两者无差异，则说明 Colas 的坍缩另有他因（如该队列的
     T2DM 二值标签本身信息量不足，或存在未知的采集协议差异）。
  5. [第一性原理裁决] 无论哪种结果，都不会得出"该算子无效"的笼统结论，而是
     进一步收窄 Work Integral 的合法适用周期长度边界（这是本项目对该算子已经
     做过的第 N 次边界标定，前几次分别标定了频率选择性、代偿黑盒边界）。
  6. [热力学与残差账单] 本次测试完全不涉及碳水/胰岛素给药的精确时间戳（Shanghai
     的用药记录是自由文本+泛化剂量，无法重建精确的药代动力学窗口），因此任何
     "做功异常是否由外源胰岛素给药引起"的因果归因在本报告中都是盲区，必须在结论
     中显性声明，不得暗示因果。

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c / diabetes duration /
    complications / duration_days are attached purely as metadata and NEVER
    used in fits or regressions.
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


def main():
    data_path = Path("output/shanghai_t2dm_subjects.json")
    if not data_path.exists():
        import export_shanghai_subjects
        export_shanghai_subjects.export_shanghai_t2dm_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Shanghai_T2DM recordings "
          f"({data['n_unique_patients']} unique patients) from {data_path}.")
    results, ok, failed = wt.run_cohort("shanghai_t2dm", subjects, period=PERIOD)
    out_file = wt.write_results("shanghai_t2dm", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
