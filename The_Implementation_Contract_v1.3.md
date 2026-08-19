# 📜 The Implementation Contract (v1.3)
**—— 针对 Pyodide/WASM 架构的防漂移底层执行契约与自治校验法则**

> **目标:** 封死 LLM 在代码生成中的微观参数漂移、库幻觉与边界条件崩溃，并赋予其自我诊断与资源回收（LIFO）的绝对律法。
> **前置依赖:** 必须通过 Pyodide 加载 `numpy (>=1.20)`, `scipy (>=1.7)`。
> **语言环境:** 核心算法**必须且只能**指向 Python 3.9+ 科学计算栈。前端 JavaScript 仅负责 UI 渲染与数据流转，绝对禁止在 JS 层进行任何数学算子（如滤波、相空间重构、特征分解）的计算仿写。

---

## 0. 架构级总则 (The Prime Directives)

### Law One: Crash Accountability (The Mirror of Truth)
- 任何异常崩溃 (Crash) 必须硬性归咎于代码或数据质量，禁止含糊其辞。
- **强制指令:** 如果在处理批次数据时，前端抛出任何未捕获异常（尤其是 Python/Pyodide 层传出的异常），必须在 `console.error` 中**完整打印最后积攒的 `Event Log Sequence`**。人类需要知道它死在了相空间的哪一步。

### Law Two: Mathematical Purity (Pyodide Execution)
- 严禁使用 Vanilla JavaScript 实现复杂数学算法（尤其是信号处理和非线性动力学相关）。
- 涉及如：零相移滤波 (`filtfilt`)、相空间延迟重构、最近邻计算 (FNN/Theiler)、递归量化分析 (RQA)，**必须通过 Pyodide 调用标准的 `scipy` 和 `numpy` 接口。**
- JS 代码仅允许处理简单的 I/O、UI 渲染、以及基础数组组装。

### Law Three: State Immutability
- 全局状态禁止被直接覆盖。必须基于不可变的更新（如返回新的对象而不是在原处修改）。

### Law Four: LIFO Memory Destructor Mandate
- 由于 Pyodide 与宿主环境之间存在内存边界，任何在 Python 侧实例化的对象（如有）必须拥有明确的生命周期或交由 Pyodide 的 `destroy()` 处理。
- JS 侧的图表渲染（如 `Plotly.newPlot`），必须在前置使用 `Plotly.purge` 释放旧的 DOM 实例。

---

## 🔒 1. 数据结构与全局掩码约束 (Data & Masking Rules)

### 1.1 核心数据结构 (Data Types)
所有时间序列处理过程，严禁混用 Python `list`，必须强制统一为单精度/双精度浮点数数组。
*   **Time Array**: `np.ndarray (dtype=np.int64)` - 强制为 Unix 时间戳 (秒)。
*   **Value Arrays**: `np.ndarray (dtype=np.float64)` - 代表原始血糖/滤波血糖。`null` 断链处必须严格编码为 `np.nan`，绝不能用 `0` 或 `-999` 替代。

### 1.2 掩码与切块引擎 (Null-Masking & Chunking Engine)
为了防止 `np.nan` 污染相空间重构，所有 L1-L2 的拓扑算子**绝对禁止**直接对全长序列进行操作，必须经过 `Chunking Engine` 切分。
*   **Chunk 切分规则:** 沿时间轴，按连续的非 `NaN` 数据片段切分为 `list[np.ndarray]`。
*   **最小可用长度截断:** 定义全局变量 `MIN_CHUNK_LEN = 30` (相当于 90 分钟)。任何 `len(chunk) < MIN_CHUNK_LEN` 的碎片强制抛弃，不参与重构。

### 1.3 JS 边界重采样阶段的空值防污染 (Resample-Boundary Null Immunity)
*   **[v1.5 新增，2026-08-16 残差修复]** `resampleDataImpl`（JS 侧、Pyodide 之外的 3 分钟网格重采样函数）在执行 4–15 分钟间隙的线性插值前，**必须**先校验插值窗口两端的原始读数本身是否有效（非 `null`/`undefined`/`NaN`）。
*   **禁止的失效模式:** JavaScript 的隐式类型转换会把 `null` 在算术运算中强制转换为 `0`（即 `vs[i] - vs[i-1]` 在 `vs[i-1]` 为 `null` 时不报错，而是静默算出错误结果）。这会把一次真实的传感器单点丢失，伪造成一段"血糖趋近于 0"的虚假插值轨迹，且不产生任何 `Event Log` 痕迹——直接违反第 0 节 Law One（崩溃可归因性）与 Blueprint v3.3 第 2.1 节"绝对禁止使用插值掩盖物理数据缺失"。
*   **强制处理:** 若插值窗口任一端点为空值，**禁止**进入线性插值分支，让该端点的空值原样传递到重采样输出序列中（与 `gap > 15` 分支产生的 `null` 断链享有同等的下游处理路径——被 1.2 节的 Chunking Engine 自然切除，绝不参与 L1-L2 计算）。
*   **发现溯源:** 由 mcPHASES 队列（`validate/wind_tunnel_v4_mcphases_phase.py`）风洞测试首次触发——该队列存在"时间间隙正常但读数本身缺失"的真实传感器丢点模式，是 Hall/Colas/Stanford/CGMacros 均未覆盖到的边界条件。

---

## 🔒 2. L0: 零相移滤波器参数焊死 (Zero-Phase Filter Config)

**严禁 LLM 自由发挥滤波器参数，必须严格复制以下规范：**

*   **算子调用:** `scipy.signal.filtfilt` (绝对禁止使用 `lfilter` 导致相位滞后)。
*   **滤波器设计:** `scipy.signal.butter`。
*   **硬编码参数:**
    *   `order = 2` (极低阶，防止对物理波峰产生过度过冲振荡 Artifacts)。
    *   `Wn = 0.08` (归一化截止频率，$\approx 0.08 \times \text{Nyquist}$)。
*   **执行边界:** 滤波操作必须在 `Chunking Engine` 切分后的**每一个有效 Chunk 内**独立执行。绝不可在包含 `np.nan` 的全量数组上直接滤波。

---

## 🔒 3. L1: 拓扑尺确立的数学契约 (Topological Gauge Config)

### 3.1 延迟时间 $\tau$ 提取 (Autocorrelation)
*   **算子逻辑:** 对每个 Chunk 独立计算 ACF，然后对各 Chunk 的 ACF 结果求加权平均（按 Chunk 长度加权）。
*   **极小值搜寻法则 (The $1/e$ Rule):** 
    *   在加权平均后的 ACF 曲线上，从 index=1 开始向右扫描。
    *   **触发条件 1:** ACF 首次下降穿过 $1/e \approx 0.3678$。
    *   **触发条件 2:** ACF 首次遇到局部极小值（当前点小于前一点和后一点），且此时 ACF $< 0.7$。
*   **边界锁死:** $\tau$ 必须为整数。`tau_min = 1`，`tau_max = 120` (最大 360 分钟)。若超过上限仍未触发，强行锁定 `tau = 120`。
*   **[v1.4 修订] 上限来源:** 该上限由 2026-08-15 Hall 队列风洞实测（`reports/wind_tunnel_hall_20260815_2149.md`）驱动上调，此前的 `tau_max = 20` 曾导致 42/57 人的 $\tau$ 被顶在天花板上（测量天花板伪影），详见 Pipeline Blueprint v3.3 第 3.1 节修订记录。
*   **[v1.5 修订] 二次上限来源（60→120）:** 2026-08-19 ShanghaiT1DM 边界标定测出 68.8% 受试者顶在 60 步天花板，随后对全部 9 个已测队列用 `max_lag=120` 全量重跑对比（`reports/wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md`）：触顶率全线降至 0%-1.0%，`Dim`/Work Integral 跨窗口秩相关 $\ge 0.95$，全部队列既有方向性结论零翻转，唯一代价是 Shanghai T2DM 新增 5/109（4.6%）例诚实失败。详见 Pipeline Blueprint v3.3 第 3.1 节 [v3.6 修订] 记录。

### 3.2 最小重构维度 $m_{fnn}$ (False Nearest Neighbors)
*   **严禁双重 for 循环:** 距离计算必须使用 `scipy.spatial.distance.pdist` 或基于 `KDTree` 的 $O(N \log N)$ 邻域搜索。
*   **硬编码超参数:**
    *   距离阈值比 $R_{tol} = 15.0$
    *   吸引子大小容差 $A_{tol} = 2.0$
*   **搜索边界:** 尝试 $m = 1$ 到 $m = 10$。当虚假最近邻比例（FNN Ratio） $< 0.05$ (5%) 或出现拐点不降反升时，停止搜索并返回该 $m$ 值。如果搜到 10 还不满足，强行截断 `m_fnn = 10`。

### 3.3 有效自由度 $m_{eff}$ (Jacobi / PCA Razor)
*   **计算域:** 取所有长度 $\ge \tau \times m_{fnn}$ 的有效 Chunk，构造延迟矩阵，将所有矩阵拼接后计算全局协方差。
*   **特征值分解:** `numpy.linalg.eigh`。
*   **% Razor 阈值:** 降序排列特征值，计算累计方差贡献率，找到首次 $\ge 0.99$ (99%) 的维度数 $m_{eff}$。
*   **边界锁死:** $m_{eff}$ 最小值为 2，最大值为 $m_{fnn}$。

---

## 🔒 4. L2: 纯粹物理算子的防溢出处理 (Tensor Operators Config)

### 4.1 绝对曲线积分 (Absolute Path Integral)
*   **计算边界锁死:** 必须在切分好的连续片段 (`chunks`) 内部独立执行计算，然后将各 chunk 的距离求和。**严禁**直接将包含 `null` 断链剔除后压扁的数组放入 `np.diff()` 计算，这会制造跨越时空的虚假热力学跳跃。
*   **执行公式:** 对 2D 相图 (延迟 $0$ 和 $\tau$ 两维) 中的有效轨迹段，计算相邻点的欧氏距离，并求和。
    *   `Raw_Work = sum_over_chunks( sum( sqrt( (x[i] - x[i-1])^2 + (y[i] - y[i-1])^2 ) ) )`
*   **24小时归一化法则 (The Time-Normalization Rule):** 
    *   `Final_Work_Integral = Raw_Work / (total_valid_points / 480.0)`
    *   如果最终结果因有效点数为 0 而触发除零，必须返回安全默认值 `None` 或 `NaN`。
*   **禁止操作:** 绝对禁止调用任何 Shoelace、Convex Hull 等围成面积的算法。

### 4.2 向心力归一化与除零防御 (Zero-Division Guard)
*   **计算核心:** 在 $m_{eff}$ 维空间中，取所有有效点的各维度中位数作为“全天引力核 (Core)”。
*   **防爆指令:** 
    *   计算 `Topological Leverage = Work Integral / Normalized Recovery` 时，必须加入保护机制。
    *   `if Normalized_Recovery < 1e-6: Normalized_Recovery = 1e-6`
    *   计算全局血糖标准差归一化时，`if std < 1e-3: std = 1e-3`。

### 4.3 夜间切分与池化法则 (Night Isolation)
*   **时间切片绝对法则:** 提取“0:00-6:00”数据时，必须按本地自然日循环 `(day 1, day 2, ...)` 独立切片。
*   **禁止跨夜:** `Night AR1` 必须计算“每一夜”独立的 Lag-1 自相关系数，最后取 `np.nanmedian()`。绝不能把所有的夜晚拼成一个长数组算 AR1，那会导致时间断裂处的巨大虚假跳跃。

### 4.4 递归定量分析 (RQA) 引擎配置
*   **Theiler 窗口时空对齐法则 (Spatiotemporal Alignment of Theiler Window):** 
    *   $T = \max(5, \tau)$。在构建递归矩阵 $R_{i,j}$ 时，若 $|i - j| \le T$，强制令该点不算作有效递归。
    *   **防扭曲指令:** 构建距离矩阵（Distance Matrix）与 Theiler 掩码时，**绝对禁止**通过剔除 `null` 的方式把数组压扁（这会导致索引 $i, j$ 丧失真实的物理时间意义）。如果存在断链，要么在有效 Chunks 内部计算，要么在全局矩阵中保留 `null` 并将相关行/列做掩盖处理，确保 $T$ 的宽度代表真实的物理时间跨度。
*   **自适应 $\epsilon$ 阈值二分法 (Bisection Method):**
    *   目标 RR (Recurrence Rate) = 0.05 (5%)。
    *   初始搜索区间: $\epsilon_{min} = 0$, $\epsilon_{max} = $ 距离矩阵的最大值。
    *   循环条件: 最大迭代 20 次，或 $|RR_{current} - 0.05| < 0.005$。
*   **禁止内存爆炸:** 若输入点数 $> 4000$，禁止生成完整的密集 $O(N^2)$ 距离矩阵。必须使用稀疏矩阵（Sparse Matrix）构建或强制采样计算对角线分布。

---

## 🔒 5. 接口与返回值强类型 (API Signature)

Agent 生成的主执行函数必须严格符合以下签名：

```python
def run_tensor_engine_v3(
    time_array: np.ndarray, 
    value_array: np.ndarray,
    debug_plots: bool = False
) -> dict:
    """
    输入必须是对齐的 1D numpy array。
    输出必须严格符合 Blueprint v3.2 中的 SSVS JSON 字典结构，包含 kairos_routing_flags。
    """
    # 强制执行此函数，内部错误一律以安全默认值/NaN 掩码抛出，禁止程序 Crash。
```

---

## ⚖️ 6. 自治校验与残差回收法则 (Autonomous Diagnostics & Event Recycling)

**【绝对指令：当 Agent (你) 编写的代码在执行时报错或生成荒谬数值，你必须强制调用本章节法则进行自我验伪，并吐出完整事件流日志。禁止掩盖错误！】**

### 6.1 法则一：Crash 归因于代码，且必须交出 Event Log (Crash vs. Distortion)
*   **[代码违约 (Code Illusion)]:** 如果运行中抛出 `IndexError`, `ValueError`, `ZeroDivisionError`, `MemoryError` 等运行时异常，**100% 判定为 Agent 代码编写违背了 1-4 章的防爆指令。**
    *   *自裁动作:* Agent 必须捕获异常，并且**必须在终端或 JS 控制台中完整打印出 Blueprint v3.3 中定义的 `event_sequence` 数组**，向人类架构师展示系统在崩溃前走到了哪一步物理状态。随后立即在原函数内修复防爆逻辑并重试。
    *   *[v1.4 新增]* **禁止历史截断 (No History Truncation):** 严禁 Agent 在修复代码后“删去崩溃相关的历史对话”。所有的 Traceback 和报错日志必须作为残差永久保留在上下文中，供后续复盘。 (DSH W12: 崩溃恢复靠“关闭”而非“截断”：历史保持诚实)
    *   *[v1.4 新增]* **Fail-Closed 沙箱不协商法则:** 如果在前端 WASM 执行中遇到跨域或浏览器安全策略阻断，必须抛出明确的异常并终结分析。**绝对禁止**将“无权限”状态返回给大模型让其自行尝试绕过沙箱。 (DSH W14: 沙箱只报告拒绝，绝不协商权限)
*   **[图纸缺陷 (Architectural Flaw)]:** 如果代码平稳运行未报错，但返回物理上不可能的数值（如 `AR1 = 50.0`, `DET = 0.0`），则属于算法设计未考虑极端边界行为。
    *   *自裁动作:* Agent 必须向用户抛出异常数据表现及对应的 `Event Log`，提请架构师修改《执行契约》的数学定义，严禁 Agent 私自修改数学公式。

### 6.2 法则二：中间产物的可视化验伪探针 (Visual Falsification Probes)
Agent 必须在主执行函数中预留 `debug_plots=True` 开关，并埋入以下 4 个 Matplotlib 探针接口。排错时必须强制画图比对：
1.  **L0 探针 (零相移检验):** 画出一段 24 小时的原始轨与滤波轨。若滤波轨波峰波谷在时间轴上有哪怕 1 步的滞后，Agent 必须立刻自查是否违规使用了带有相移的滤波器（如 `lfilter`）。
2.  **L1 探针 (ACF 截断检验):** 画出 ACF 曲线。若曲线未跌破 $1/e$ 就戛然而止，Agent 必须排查 $\tau$ 搜索算法是否写出死循环或边界溢出。
3.  **L2 探针 A (2D 延迟相图):** 画出 $t$ 与 $t-\tau$ 的二维散点图。若图中充满“长距离跳跃飞线”，Agent 必须核查 `Chunking Engine` 是否失效，导致了断链处的强行相连。
4.  **L2 探针 B (RQA 矩阵遮罩):** 画出 RQA 矩阵图像。若主对角线及 $T$ 范围内存在黑色递归点，Agent 必须立刻检查 Theiler 窗口遮罩代码是否失效。

### 6.3 法则三：平庸数据绝对单元测试 (The Trivial Data Unit Test)
在触碰真实的 14 天 CGMS 数据之前，Agent **必须首先生成并在以下三组假数据上跑通代码**：
1.  **纯正弦波 ($y = \sin(t)$):** `DET` 必须接近 1.0 (95%+)。
2.  **纯高斯白噪 ($y = \text{np.random.normal}()$):** `DET` 必须接近 0.0，`AR1` 必须接近 0.0。
3.  **绝对水平直线 ($y = 5.5$):** 必须能安静退出或返回安全默认值 `0.0/NaN`，**绝对禁止抛出任何 Crash 或 Divide by Zero**。

> **终极训诫:** 
> “LLM，你不是造物主，你是一台精密机床。用第 6 章的法则测试你的每一行代码，用代码去印证图纸。如有违约，自我裁决。”