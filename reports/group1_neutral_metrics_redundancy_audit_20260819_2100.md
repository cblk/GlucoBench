# 第一组6张中性卡：工程移植 + 冗余审计报告

**日期**：2026-08-19 21:00
**范围**：AGENTS.md 第 3.3/9 节风洞实验协议，用户批准顺序"先做 Phase A(工程移植+交叉验证) + Phase B(冗余审计)，再决定 Phase C 测哪些"
**涉及卡片**：相空间展开体积 (Volume)、归一化向心步长 (Recovery)、主轴各向异性 (λ1/λ2 Shape Ratio)、盒计数几何维度 (Box-Counting Dimension)、一步近邻发散代理 (Lyapunov)、当前–夜间核心距离 (Core Dist)

---

## Phase A：工程移植与交叉验证

### 拓扑骨架提取
这6张卡此前均为纯 JavaScript 实现（`computeAttractorMetrics` 的协方差/特征分解块、`boxCountingDimension`、`lyapunovProxy`、`computeNormalizedRecovery`、`calcDistance`），`_extracted_tensor_engine_v4.py` 无对应 Python 镜像——第 9.4 节同源纪律在此无"生产 Python"可比对，只能逐行手工移植。

移植产物：
- `validate/_legacy_metrics_group1_v4.py`：6 个算子的 Python 端口（`jacobi_eigenvalues`/`gamma_approx`+`VOLUME_COEFF`/`compute_volume_shape`/`box_counting_dimension`/`lyapunov_proxy`/`compute_normalized_recovery`/`calc_distance`）。
- `validate/_js_legacy_metrics_group1_crosscheck.mjs`：从 `index_v4.html` 逐字节抽取的 JS 函数原文（非重写）。
- `validate/crosscheck_legacy_metrics_group1.py`：合成 500 点三维螺旋相空间轨迹（含周期性缺口），同时喂给真实 JS 与 Python 端口，逐字段 diff。

**交叉验证结果：0 处不匹配**（Volume/ShapeRatio/GravityCore/EffectiveDim/BoxCountingDim/Lyapunov/NormalizedRecovery/CoreDist 全部在 1e-9 容差内完全一致）。

### 关键设计发现：Core Dist 的期相依赖陷阱
生产 JS 在 `period==='night'` 时把位移**硬编码为 0**（同夜间对比自身，定义上必为零），这意味着若沿用其余 5 张卡的 `period="night"` 默认约定，Core Dist 在本次审计中会永远测出 0，毫无意义。因此新增的 `run_subject_legacy_group1()` 把 Core Dist **无条件**改用 `period="all"` 的引力核对比夜间 RAW 核心（与生产 UI 实际展示时的用法一致），与其余 5 张卡的 `period` 参数解耦。

### 烟雾测试（Hall，57人）
`validate/wind_tunnel_v4_hall_legacy_group1.py`：**57/57 全部成功，零 null**，六项指标数值范围均在物理合理区间（ShapeRatio≥1.0 符合定义、BoxCountingDim∈[1.05,1.77]、Lyapunov 有正有负符合"局部稳定/发散混合"的预期、CoreDist 均为正值）。

---

## Phase B：冗余审计（Stanford SSPG n=29 + Shanghai T2DM n=104）

### 为什么必须先做这一步
这6个算子全部共享同一套 Takens 嵌入点集（`period-sliced smooth/raw track`），其中 Volume/ShapeRatio/CoreDist 三者甚至来自**同一个协方差矩阵特征分解**。若不先审计就直接对6个指标逐一做拓扑对撞检验，统计显著即被计入"新发现"，会把"1个信号被算6遍"误记为"6份独立证据"——这正是第 9.1.3 节《拒绝人工复合分数》要防的镜像陷阱。

### 与已毕业指标(workIntegral/DET/ENTR/Dim)的秩相关（Spearman ρ，队列内分别计算）

| Group-1 指标 | vs workIntegral | vs DET | vs ENTR | vs Dim(FNN) | 判定 |
|---|---|---|---|---|---|
| **Volume** | Stanford +0.545 / Shanghai **+0.725** | -0.09 / -0.27 | +0.10 / +0.40 | -0.16 / -0.48 | ⚠️ **两队列方向一致且随样本量增大转强，判定与 Work Integral 存在实质性冗余** |
| **Lyapunov** | +0.32 / +0.06 | -0.25 / +0.06 | -0.46 / -0.15 | Stanford **-0.753** / Shanghai **-0.557** | ⚠️ **两队列均强耦合，判定与嵌入维度 Dim 存在实质性冗余**（`dim` 直接决定 Lyapunov 计算所用相空间的坐标数，机制上本就不独立） |
| Shape Ratio (λ1/λ2) | -0.04 / -0.03 | +0.13 / -0.08 | +0.15 / +0.19 | -0.45 / -0.47 | 中等耦合（未过0.70红线，但两队列方向/幅度高度一致，需带着"部分与 Dim 共线"的认识去解释任何未来结果） |
| Box-Counting Dim | -0.02 / -0.14 | +0.08 / +0.37 | -0.05 / +0.10 | +0.27 / **+0.57** | 队列间不一致（Stanford弱、Shanghai中等），中等风险 |
| Recovery (avgRecovery) | +0.08 / -0.25 | -0.05 / +0.42 | -0.41 / -0.34 | +0.30 / **+0.67** | 队列间不一致，中等风险；且概念上与已耗尽大量精力仍未毕业的 `relaxationTime`（候选#5）高度重叠（都是"扰动后向核心恢复速率"，只是一个用几何步长一个用时间轴测） |
| **Core Dist** | +0.23 / +0.23 | -0.01 / -0.13 | -0.01 / +0.25 | +0.09 / -0.19 | ✅ **两队列中与全部4个已毕业指标的相关性均最弱（\|ρ\|≤0.29），是6个指标中最独立的一个** |

### Group-1 内部交叉相关（额外发现的共祖风险）
`avgRecovery` 与 `Volume` 内部强相关（Stanford -0.54, Shanghai **-0.74**），说明即便不考虑与已毕业指标的关系，Recovery 本身也很大程度上是 Volume 的镜像——而 Volume 又已被判定为 Work Integral 的冗余项，等于 Recovery 间接地也是 Work Integral 信号的三级复述。

### 非对称咬合（残差判定）
- **Volume、Lyapunov**：判定为**与已毕业指标存在实质性冗余**。即便未来在其上跑拓扑对撞测出"显著"分离，也不能计入新证据——那只是 Work Integral / Dim 信号的重复计数，违反第 9.1.3 节精神。
- **Recovery**：判定为**双重冗余**（内部与 Volume 强相关 + 概念上与已耗尽投入的 relaxationTime 重叠），性价比最低，不建议投入。
- **Shape Ratio、Box-Counting Dim**：判定为**中等耦合，非纯净但也非纯冗余**——若测试，须在报告中明确注明"结果部分可能来自与 Dim 共线的贡献，不能单独归因"。
- **Core Dist**：判定为**审计通过，六者中最独立**，是 Phase C 拓扑对撞检验的最优先候选。

---

## 残差与熵增清算表

| 项目 | 状态 |
|---|---|
| JS→Python 移植保真度 | ✅ 0 处不匹配（crosscheck） |
| Hall 烟雾测试 | ✅ 57/57 成功，零异常值 |
| Stanford SSPG 冗余审计样本 | ✅ 29/29 成功 |
| Shanghai T2DM 冗余审计样本 | ⚠️ 104/109 成功，5 例 `estimate_dimension` 失败（与既有 Group2/非-legacy 跑法同一批次失败，非本次移植引入的新问题） |
| Volume/Lyapunov 独立性 | ❌ Fail-Closed（判定为冗余，不建议单独测试） |
| Recovery 独立性 | ❌ Fail-Closed（判定为双重冗余） |
| Shape Ratio/Box-Counting Dim 独立性 | ⚠️ 部分冗余，可测但需谨慎解释 |
| Core Dist 独立性 | ✅ 通过审计，最优先 Phase C 候选 |

---

## Phase C：Core Dist 拓扑对撞检验（用户批准"仅测最独立的 Core Dist"）

用户裁决：Volume/Lyapunov/Recovery 因判定冗余不投入；Shape Ratio/Box-Counting Dim 因中等耦合暂不投入；仅对审计通过的 **Core Dist** 执行第 9.3 节拓扑对撞检验。

沿用与 Group2 完全相同的分组方法学（第 9.1.2 节标签仅作分光镜）：
- **Stanford SSPG**（`sspg_class` IR vs IS 分组）：n_IS=16, n_IR=13。IS 中位数 0.3906，IR 中位数 0.3328。**P(IR>IS) = 0.5192，置换检验 p=0.8598**。
- **Shanghai T2DM**（HbA1c 固定全局中位数分组，首诊去重后 95 独立患者）：中位数分组阈值 68.31 mmol/mol，n_high=43, n_low=44。high 中位数 1.2917，low 中位数 1.5044（**方向与预期相反**：血糖控制更差组的核心距离反而更小）。**P(high>low) = 0.4704，置换检验 p=0.6454**。

### 非对称咬合（最终判定）
两队列秩分离度均落在 0.47-0.52，**与抛硬币（0.5）在统计上无法区分**（p=0.65-0.86，远未接近任何合理显著性阈值），且两队列的方向甚至互不一致。**Core Dist 判定 Fail-Closed**——尽管它在 Phase B 审计中是6个指标里与已毕业指标耦合最弱（最"干净"）的一个，但"独立"不等于"有信号"：它只是恰好没有从其他算子那里继承冗余信号，本身也没有携带任何与本次测试的临床分组相关的新信息。诚实记录，不升格，不与其余5个冗余/中等耦合的指标混为一谈重新讨论。

## 最终结论
第一组6张纯中性卡的审计+检验全部完成，**无一项通过第 9.3 节拓扑胜利判定**：
- Volume / Lyapunov / Recovery：Phase B 判定与已毕业指标存在实质性冗余，未投入检验。
- Shape Ratio / Box-Counting Dim：Phase B 判定中等耦合，用户裁决暂不投入检验。
- Core Dist：Phase B 审计通过（最独立），但 Phase C 检验 Fail-Closed（两队列均无区分度）。

这6张卡维持现状：纯中性显示，无判色逻辑（本就没有），无星标（未通过第 9.3 节标准）。`index_v4.html`/Blueprint/Contract 均未改动。
