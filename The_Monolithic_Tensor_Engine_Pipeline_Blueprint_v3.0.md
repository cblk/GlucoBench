# 📜 The Monolithic Tensor Engine: Pipeline Blueprint (v3.0)
**—— 融合高维分形嵌入学与 RQA 的单周期物理测度仪**

> **版本**: v3.0 (Single-Epoch Monolithic Blueprint)
> **状态**: Draft / Proposed Architectural Standard
> **核心定位**: 第一层「硬计算测度仪 (HUD)」的底层数学与物理管线规范

---

## 1. 核心公理 (The Core Axioms)

1. **单周期绝对基线 (Single-Epoch Absolute Baseline)**：
   本管线仅处理单一连续时间序列（14-16 天）。不进行任何跨周期的 $\Delta$ 计算，彻底根除因双周期规范漂移（Gauge Drift）带来的虚假导数伪影。
2. **物理真实高于数学平滑 (Physical Reality > Smooth Interp)**：
   数据断链代表系统测量或传感器的真实物理断层。绝对禁止使用长时间平滑或高阶插值掩盖物理数据缺失。
3. **拓扑不变量优先 (Topological Invariants First)**：
   抛弃所有基于 1D 时间序列的微观拟合与脆弱极值点测量（如寻找 peak/trough 计算时间差），全面拥抱在 $m$ 维相空间中计算的全局拓扑不变量（如磁滞环面积、RQA 确定性 DET、对角线熵 ENTR）。
4. **硬计算与高维解析隔离 (Dual-Clutch Boundary)**：
   本管线仅输出纯粹、无偏的物理测度与极值熔毁预警。绝对禁止在管线内部嵌入任何临床诊断分类、疾病风险概率，或针对 Vector 1-7 的具体物理干预指令。

---

## 2. L0: 数据摄入与时空重构层 (Ingestion & Spacetime)

**目标：将异构的原始数据转化为标准的、带有物理断层的离散相空间网格。**

### 2.1 3 分钟固定网格重采样
* **标准采样间隔**: $\Delta t = 180\text{s}$ (3 分钟)。
* **断链法则 (The 15-Min Rule)**:
  * 相邻数据点间隙 $\le 15$ 分钟：允许线性插值补齐网格。
  * 相邻数据点间隙 $> 15$ 分钟：**强制填入 `null`**。在相空间重构时，任何包含 `null` 的相点都将被直接丢弃，防止生成“无摩擦的人造直线弦”。

### 2.2 双轨数据流 (Dual-Track Architecture)
* **原始轨 (Raw Track)**: 仅经过重采样与断链填 `null`，保留所有高频微观摩擦与真实噪点。
  * **应用算子**: `Normalized Recovery`（向心力）, `Night Friction`（夜间阻力）, `Night AR1`（临界慢化）。
* **平滑轨 (Smooth Track)**: 对原始轨应用轻量级 EMA（指数移动平均，$\alpha = 0.3$），重置于断链处。用于过滤硬件高频底噪，提取宏观拓扑结构。
  * **应用算子**: `Work Integral`（磁滞做功）, `DET`（确定性）, `ENTR`（对角线熵）。

---

## 3. L1: 拓扑规范确立层 (Topological Gauge)

**目标：为当前周期确立唯一的、自适应的相空间坐标系 $(\tau, m_{eff})$。**

### 3.1 时间延迟 ($\tau$)
* **输入**: 全量平滑轨数据。
* **算法**: 计算自相关函数 (ACF)。寻找 ACF 首次衰减至 $1/e$ ($\approx 0.368$) 或出现首个确证的局部极小值（需连续两步回升确证，且 ACF $< 0.8$）的滞后步数。
* **物理意义**: 系统消除初始状态记忆所需的特征松弛时间（时间债）。
* **边界**: 上限强制锁定为 20 步（60 分钟）。

### 3.2 嵌入维度 ($m$) 与有效自由度截断 (% Razor)
* **输入**: 全量平滑轨数据。
* **算法**:
  1. 基于盒计数法 (Box-Counting) 估算吸引子分形维度 $dA$。
  2. 应用 Sauer-Yorke-Casdagli (SYC) 扩展定理，计算理论无损重构维度 $m_{syc} = \lceil 2 \cdot dA \rceil + 1$。
  3. 对 $m_{syc}$ 维相空间点的协方差矩阵进行 **Jacobi 特征值分解**，提取所有特征值（能量分量）。
  4. **% Razor 信噪比剃刀**: 提取累计方差占比 $> 99\%$ 的主特征值个数，定义为 **有效自由度 ($m_{eff}$)**。
* **物理意义**: 切断高维几何噪声爆炸，自适应锁定包含 99% 主能量的正交维度。

---

## 4. L2: 纯粹物理算子层 (Pure Tensor Operators)

**目标：在确立的 $(\tau, m_{eff})$ 规范下，提取 7 个正交的物理摩擦力与拓扑指标。**

### 轴 I：能量与做功 (Energy & Work)
1. **Work Integral (相空间磁滞做功)**
   * **算子算法**: 使用平滑轨数据，在 $d=0$ 与 $d=1$ 构成的 2D 延迟相图上，应用 Shoelace 公式计算相图轨迹围成的封闭闭环面积。
   * **物理意义**: 系统受迫升糖与向心弛豫全过程中，对抗高熵环境（进食/压力冲击）所付出的总热力学废热。
2. **Topological Leverage (拓扑杠杆率)**
   * **算子算法**: $\text{Leverage} = \frac{\text{Work Integral}}{\text{Normalized Recovery}}$。
   * **物理意义**: 为了获得单位向心收敛速度，系统付出的总做功代价。极高值（$>5.0$）代表极低效的代偿游荡。

### 轴 II：拓扑摩擦与阻力 (Topological Friction)
3. **Normalized Recovery (归一化向心力)**
   * **算子算法**: 使用原始轨数据。计算每个相点 $P_i$ 朝向全天引力核（各维度中位数）的向心移动速度 $\Delta d = \text{dist}(P_{i-1}, Core) - \text{dist}(P_i, Core)$。取所有 $\Delta d > 0$ 的平均值，并除以全局血糖标准差进行归一化。
   * **物理意义**: 衡量系统在受迫达峰后向中心收敛的绝对力量与速度（胰岛素响应阻力）。
4. **Night Friction (夜间相变阻力)**
   * **算子算法**: 使用原始轨数据。提取夜间（0:00-6:00）相点，计算其在 $m_{eff}$ 维相空间中偏离夜间核心吸引子的欧氏距离的加权平均值（弥散度）。
   * **物理意义**: 肝脏夜间重置与内源性稳态恢复的物理阻力。

### 轴 III：系统演化与韧性 (Evolution & Resilience)
5. **Night AR1 (夜间临界慢化 - 逐夜中位数)**
   * **算子算法**: 使用原始轨数据。**绝对禁止跨夜池化 (Pooling)**。按自然日切割，逐夜提取 0:00-6:00 序列，计算其 Lag-1 自相关系数 $AR1_{night}$。取所有有效夜晚的**中位数**。
   * **物理意义**: 探测复杂系统在崩溃前夕的临界慢化 (Critical Slowing Down) 现象，反映内生纠错机制的疲劳度。
6. **DET (代谢僵化度 - RQA 确定性)**
   * **算子算法**: 使用平滑轨数据。构建递归图 (Recurrence Plot)，**强制应用 Theiler 窗口隔离 ($T=5$)**（排除时间连续性伪影），直接从相空间距离分布中按 **目标递归率 (RR=5%) 反推自适应 Epsilon 阈值**。计算组成平行于主对角线线段（长度 $l \ge 2$）的递归点占总递归点的比例。
   * **物理意义**: 测量系统轨迹被因果律锁死的程度。极高（$>85\%$）代表代谢僵死，极低（$<20\%$）代表白噪音化失序。
7. **ENTR (适应性资本 - RQA 对角线熵)**
   * **算子算法**: 使用平滑轨数据。在上述 RQA 递归矩阵中，提取所有有效对角线长度分布 $P(l)$，计算其香农熵 $\text{ENTR} = -\sum P(l) \ln P(l)$。
   * **物理意义**: 测量系统应对外源冲击时相空间轨迹的“手段丰富度”与“计算不可约性”。熵越高，代谢灵活性与适应性资本越雄厚。

---

## 5. L3: 遥测序列化与熔毁判定层 (Telemetry & SSVS)

**目标：输出标准化 SSVS JSON，向高维解析层（LLM Navigator）提供绝对客观的残差凹坑。**

### 5.1 极值熔毁预警 (Critical Meltdown Warning)
* **触发条件**（基于单周期 P90/P99 风洞分布）：
  * `Night AR1 > 0.990`（内生纠错机制濒临崩溃）
  * `Work Integral > 80.0`（不可逆病理废热严重淤积）
  * `Topological Leverage > 5.0`（极低效的代偿游荡）
* **警告输出**: `🚨 [CRITICAL MELTDOWN WARNING] 物理摩擦力残差已越过熔毁阈值。干预路由由高维解析层裁定。`

### 5.2 SSVS JSON 序列化规范
```json
{
  "subject_id": "Somatic-Node-Single",
  "analysis_mode": "Cybernetic_Baseline_Single",
  "pipeline_version": "v3.0_Monolithic",
  "topological_gauge": {
    "tau_steps": 6,
    "tau_minutes": 18,
    "fractal_dimension_dA": 1.52,
    "effective_dimension_m": 4
  },
  "tensor_metrics": {
    "work_integral": {
      "metric_name": "相空间磁滞做功 (Hysteresis Work)",
      "value": 32.18,
      "semantic_tag": "Energy_Compensation"
    },
    "topological_leverage": {
      "metric_name": "拓扑杠杆率 (Topological Leverage)",
      "value": 3.42,
      "semantic_tag": "Compensation_Efficiency"
    },
    "normalized_recovery": {
      "metric_name": "归一化向心力 (Normalized Recovery)",
      "value": 0.325,
      "semantic_tag": "Insulin_Response_Force"
    },
    "night_friction": {
      "metric_name": "夜间相变阻力 (Night Friction)",
      "value": 16.12,
      "semantic_tag": "Internal_Reset_Resistance"
    },
    "night_ar1_median": {
      "metric_name": "夜间临界慢化中位数 (Night AR1 Median)",
      "value": 0.9790,
      "semantic_tag": "System_Meltdown_Risk"
    },
    "rqa_determinism_det": {
      "metric_name": "代谢僵化度 (RQA Determinism)",
      "value": 0.742,
      "semantic_tag": "Causal_Rigidity"
    },
    "rqa_entropy_entr": {
      "metric_name": "适应性资本 (RQA Diagonal Entropy)",
      "value": 1.48,
      "semantic_tag": "Adaptive_Capital"
    }
  }
}
```

---

## 6. 与旧版本 (v8.8) 的清理映射表 (Refactoring Checklist)

在按照本蓝图重构 `index.html` 时，必须彻底删除以下旧代码模块：

| 旧版模块 (v8.8) | 处理动作 | 替换/重构方案 |
|---|---|---|
| 双周期 (Epoch 0 / Epoch 1) 输入 UI 及对比逻辑 | 彻底删除 | 退回单周期极简输入 UI |
| `computeExcursionKinetics` (早相迟滞/弛豫时间) | 彻底删除 | 由 L2 轴 III 的 `DET` 和 `ENTR` 代替 |
| `computeKeplerKinematics` (角速度/扫过速率) | 彻底删除 | 算子物理意义重叠，直接剥离 |
| 空腹胰岛素预测器及 Ridge 权重 (`predictInsulinBaseline`) | 彻底删除 | 已于之前废弃，彻底清空遗留代码 |
| 几何体积 (`Volume`) / 拟合形状比 (`Shape Ratio`) 等 | 彻底删除 | 恢复为 v6.3 纯粹物理轴 |
| AR1 全夜拼接池化逻辑 | 重构 | 改为按自然日切割的逐夜 AR1 并取中位数 |
| 带有 `warn/bad` 的色彩与健康分类评级 | 移除评级逻辑 | HUD 仅做展示与熔毁预警，评级交由第二层 |
