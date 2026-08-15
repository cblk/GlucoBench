# 📜 The Monolithic Tensor Engine: Pipeline Blueprint (v3.3)
**—— 融合高维分形嵌入学、RQA、响应式共效应与 Kairos 握手协议的单周期物理测度仪**

> **版本**: v3.3 (Single-Epoch Monolithic Blueprint - Cordis Genesis v1.2 Integration)
> **状态**: Active Architectural Standard
> **核心定位**: 第一层「硬计算测度仪 (HUD)」的底层数学与物理管线规范

---

## 1. 核心公理 (The Core Axioms)

1. **单周期绝对基线 (Single-Epoch Absolute Baseline)**：
   本管线仅处理单一连续时间序列（14-16 天）。不进行任何跨周期的 $\Delta$ 计算，彻底根除因双周期规范漂移（Gauge Drift）带来的虚假导数伪影。
2. **物理真实高于数学平滑 (Physical Reality > Smooth Interp)**：
   数据断链代表系统测量或传感器的真实物理断层。绝对禁止使用长时间平滑或高阶插值掩盖物理数据缺失。
3. **拓扑不变量优先 (Topological Invariants First)**：
   抛弃所有基于 1D 时间序列的微观拟合与脆弱极值点测量（如寻找 peak/trough 计算时间差），全面拥抱在 $m$ 维相空间中计算的全局拓扑不变量（如相图曲线积分、RQA 确定性 DET、对角线熵 ENTR）。
4. **硬计算与高维解析隔离 (Dual-Clutch Boundary / The Oracle Handshake)**：
   本管线仅输出纯粹、无偏的物理测度。绝对禁止在管线内部嵌入任何临床诊断分类、疾病风险概率，或干预指令。所有的“病情裁决”与“奇点捕获”，必须通过 `kairos_routing_flags` 向上抛出，全权移交给第四层架构（`The Kairos Oracle v1.0`）进行拓扑裁决。

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
* **平滑轨 (Smooth Track)**: 对原始轨应用**零相移滤波器 (Zero-phase digital filter, 如 Forward-Backward Butterworth 滤波)**，重置于断链处。彻底消灭传统 EMA 带来的时间相位滞后 (Phase Shift) 残差，在过滤硬件高频底噪的同时，绝对保持峰谷的物理时间戳不变。
  * **应用算子**: `Work Integral`（磁滞做功）, `DET`（确定性）, `ENTR`（对角线熵）。

---

## 3. L1: 拓扑规范确立层 (Topological Gauge)

**目标：为当前周期确立唯一的、自适应的相空间坐标系 $(\tau, m_{eff})$。**

### 3.1 时间延迟 ($\tau$)
* **输入**: 全量平滑轨数据。
* **算法**: 计算自相关函数 (ACF)。寻找 ACF 首次衰减至 $1/e$ ($\approx 0.368$) 或出现首个确证的局部极小值（需连续两步回升确证，且 ACF $< 0.8$）的滞后步数。
* **边界约束**: 为防止 `null` 断链在 ACF 中传染畸变，仅在有效的连续数据块 (Valid Chunks) 中计算自相关并融合。上限强制锁定为 20 步（60 分钟）。
* **物理意义**: 系统消除初始状态记忆所需的特征松弛时间（时间债）。

### 3.2 嵌入维度 ($m$) 与有效自由度截断 (% Razor)
* **输入**: 全量平滑轨数据。
* **算法**:
  1. 应用 **False Nearest Neighbors (FNN，虚假最近邻法)** 算法，基于空间距离展开，寻找消除轨迹虚假交叉的最小理论重构维度 $m_{fnn}$。彻底抛弃易引发维数灾难的 Box-Counting。
  2. 建立 **Null-Masking Matrix (掩码矩阵)**：切分出无断链的连续有效数据块，严禁 `null` 点在延迟重构矩阵中跨时空连乘传染。
  3. 在有效块内，对 $m_{fnn}$ 维相空间点的协方差矩阵进行 **Jacobi 特征值分解**，提取所有特征值（能量分量）。
  4. **% Razor 信噪比剃刀**: 提取累计方差占比 $> 99\%$ 的主特征值个数，定义为 **有效自由度 ($m_{eff}$)**。
* **物理意义**: 切断高维几何噪声爆炸，自适应锁定包含 99% 主能量的正交维度。

---

## 4. L2: 纯粹物理算子层 (Pure Tensor Operators)

**目标：在确立的 $(\tau, m_{eff})$ 规范下，提取 7 个正交的物理摩擦力与拓扑指标，为 Kairos 供弹。**

**【响应式共效应声明 (Reactive Coeffects)】**
*   所有 L2 算子绝不是孤立的函数，它们必须显式依赖 L0 或 L1 的输出状态。若前置依赖失败，后续算子必须优雅熔断并追加事件流，严禁强行计算。

### 轴 I：能量与做功 (Energy & Work)
1. **Work Integral (相空间磁滞做功)**
   * **共效应依赖 (Requires)**: `[smooth_track_chunked, tau_locked]`
   * **算子算法**: 使用平滑轨数据，必须在连续的有效数据块 (Chunks) 内独立计算，绝对禁止跨越 `null` 断链强制连接。在 $d=0$ 与 $d=\tau$ 构成的 2D 延迟相图上，计算每个 Chunk 内部轨迹的**绝对曲线积分 (Absolute Path Integral, $\int |p \cdot dq|$)** 或累积绝对位移。
   * **时间归一化法则 (The Time-Normalization Mandate)**: 由于绝对曲线积分是一个广延量（Extensive Property，时间越长数值越大），必须将所有 Chunk 的积分总和，**归一化到 24 小时的物理时间基准**（即除以 $\frac{valid\_points}{480.0}$，假设 3 分钟一帧）。
   * **物理意义**: 衡量对抗高熵环境所付出的废热。归一化确保了在不同数据丢失率下，热力学账单的可比性。为 Kairos Oracle 的“热力学残差清算”提供账单。
2. **Topological Leverage (拓扑杠杆率)**
   * **共效应依赖 (Requires)**: `[Work_Integral_valid, Normalized_Recovery_valid]`
   * **算子算法**: $\text{Leverage} = \frac{\text{Work Integral}}{\text{Normalized Recovery}}$。
   * **物理意义**: 为了获得单位向心收敛速度，系统付出的总做功代价。极高值（$>5.0$）代表极低效的代偿游荡。

### 轴 II：拓扑摩擦与阻力 (Topological Friction)
3. **Normalized Recovery (归一化向心力)**
   * **共效应依赖 (Requires)**: `[raw_track_chunked, m_eff_locked]`
   * **算子算法**: 使用原始轨数据。计算每个相点 $P_i$ 朝向全天引力核（各维度中位数）的向心移动速度，取均值并归一化。
4. **Night Friction (夜间相变阻力)**
   * **共效应依赖 (Requires)**: `[raw_track_chunked, m_eff_locked]`
   * **算子算法**: 使用原始轨数据。提取夜间（0:00-6:00）相点，计算其在 $m_{eff}$ 维相空间中偏离夜间核心吸引子的欧氏距离的加权平均值。

### 轴 III：系统演化与韧性 (Evolution & Resilience)
5. **Night AR1 (夜间临界慢化 - 逐夜中位数)**
   * **共效应依赖 (Requires)**: `[raw_track_chunked]`
   * **算子算法**: 使用原始轨数据。按自然日切割，逐夜提取 0:00-6:00 序列，计算 Lag-1 自相关系数 $AR1_{night}$，取中位数。
   * **物理意义**: 探测系统崩溃前夕的临界慢化。为 Kairos Oracle 的“分岔雷达与奇点距离”提供直接弹药。
6. **DET (代谢僵化度 - RQA 确定性)**
   * **共效应依赖 (Requires)**: `[smooth_track_chunked, tau_locked]`
   * **算子算法**: 使用平滑轨数据。构建递归图，**强制绑定自适应 Theiler 窗口 $T = \max(5, \tau)$**，按 **RR=5% 反推 Epsilon**。计算平行于主对角线线段（$l \ge 2$）的比例。
   * **物理意义**: 极高代表代谢僵死。为 Kairos Oracle 判断“系统是否陷入深谷吸引子”提供依据。
7. **ENTR (适应性资本 - RQA 对角线熵)**
   * **共效应依赖 (Requires)**: `[DET_matrix_valid]`
   * **算子算法**: 计算有效对角线长度分布 $P(l)$ 的香农熵 $\text{ENTR} = -\sum P(l) \ln P(l)$。

---

## 5. L3: 遥测序列化与熔毁判定层 (Telemetry & SSVS)

**目标：输出标准化 SSVS JSON，向高维解析层（Kairos Oracle）提供绝对客观的残差凹坑与路由信标。**

### 5.1 极值熔毁预警与 Kairos 映射
* **触发条件与路由映射**（基于单周期 P90/P99 风洞分布）：
  * `Night AR1 > 0.990` $\rightarrow$ **映射:** 临界慢化爆发，系统逼近分界线（`is_approaching_saddle = true`）。
  * `DET > 0.85` $\rightarrow$ **映射:** 系统陷入强拓扑刚性（`is_in_deep_attractor = true`）。
  * `Work Integral > 80.0` 或 `Topological Leverage > 5.0` $\rightarrow$ **映射:** 不可逆病理废热严重淤积（`thermodynamic_bill_heavy = true`）。
* **警告输出**: `🚨 [CRITICAL MELTDOWN WARNING] 物理摩擦力残差已越过熔毁阈值。底盘计算终止，全权移交 The Kairos Oracle 进行拓扑裁决。`

### 5.2 SSVS JSON 序列化规范与事件流 (The Oracle Handshake API)
```json
{
  "subject_id": "Somatic-Node-Single",
  "analysis_mode": "Cybernetic_Baseline_Single",
  "pipeline_version": "v3.3_Monolithic",
  "kairos_routing_flags": {
    "is_approaching_saddle": false,
    "is_in_deep_attractor": true,
    "thermodynamic_bill_heavy": false
  },
  "topological_gauge": {
    "tau_steps": 6,
    "tau_minutes": 18,
    "fnn_dimension_m_syc": 6,
    "effective_dimension_m": 4
  },
  "tensor_metrics": {
    "work_integral": {
      "metric_name": "相空间磁滞做功 (Hysteresis Work)",
      "value": 32.18,
      "semantic_tag": "Energy_Compensation"
    },
    // ... 其他指标
  },
  "event_sequence": [
    "[INFO] [L0] Data Ingested, length: 6720",
    "[INFO] [L0] Chunking Engine completed: 4 Valid Chunks",
    "[INFO] [L1] Autocorrelation Tau locked at: 6",
    "[INFO] [L1] FNN/Jacobi Razor M_eff locked at: 4",
    "[INFO] [L2] RQA Matrix built successfully, Epsilon: 0.15"
  ]
}
```

---

## 6. L0-L3 重构核心检查单 (v3.2 Refactoring Checklist)

在进入代码实现阶段时，必须核对以下修正项：

| 拓扑病理 / 旧版缺陷 | v3.2 的绝对修正协议 | The Kairos Oracle 协同意义 |
|---|---|---|
| Shoelace 公式算混沌面积会互相湮灭 | 改为 **绝对曲线积分 (Absolute Path Integral)** | 提供绝对真实的“废热账单”，供 Oracle 清算 |
| Theiler 窗口被硬编码 ($T=5$) | 动态解绑，强制 **$T = \max(5, \tau)$** | 确保 `DET` 对深谷吸引子的判断不会出现时间伪影 |
| Box-Counting 维数灾难 | 斩断算力黑洞，改用 **False Nearest Neighbors (FNN)** | 确保引擎在几秒内吐出相空间维度，而不是卡死崩溃 |
| EMA 指数移动平均引发相位漂移 | 拔除 EMA，改用 **零相移滤波器 (Zero-phase Filter)** | 保证 L0 传导给 L2 的峰谷物理时间点（Kairos点）绝不位移 |
| `null` 在张量窗口中无限传染 | 引入 **Null-Masking Matrix (掩码矩阵)** | 即使传感器大面积断链，也能保证截取的拓扑特征不被污染 |
| 试图在管线内评判健康分类 | 彻底剥夺诊断权限，增设 **kairos_routing_flags** | 完美实现“计算与决策”的 **双离合边界** 架构 |
| **【v3.3 新增】** 算子孤岛与追溯断链 | 引入 **响应式共效应 (Reactive Coeffects) 与 Event Log** | 灾难发生时，为人类架构师提供时间箭头上绝对的崩溃回溯点 |
