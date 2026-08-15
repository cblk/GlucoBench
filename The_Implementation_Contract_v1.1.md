# 📜 The Implementation Contract (v1.2)
**—— 针对 Monolithic Tensor Engine v3.3 的防漂移底层执行契约与自治校验法则**

> **目标:** 封死 LLM 在代码生成中的微观参数漂移、库幻觉与边界条件崩溃，并赋予其自我诊断与资源回收（LIFO）的绝对律法。
> **前置依赖:** `numpy (>=1.20)`, `scipy (>=1.7)`。
> **语言环境:** 默认指向 Python 3.9+ 科学计算栈。若迁移至 TypeScript 等其他语言，必须严格复现此契约中的向量化操作。

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
*   **边界锁死:** $\tau$ 必须为整数。`tau_min = 1`，`tau_max = 20` (最大 60 分钟)。若超过上限仍未触发，强行锁定 `tau = 20`。

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
*   **执行公式:** 对 2D 相图 (延迟 $0$ 和 $\tau$ 两维) 中的有效轨迹段，计算相邻点的欧氏距离，并求和。
    *   `Work_Integral = sum( sqrt( (x[i] - x[i-1])^2 + (y[i] - y[i-1])^2 ) )`
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
*   **Theiler 窗口锁定:** $T = \max(5, \tau)$。在构建递归矩阵 $R_{i,j}$ 时，若 $|i - j| \le T$，强制令该点不算作有效递归（相当于主对角线附近带宽为 $T$ 的区域被抹去）。
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
    *   *自裁动作:* Agent 必须捕获异常，并且**必须在终端或日志中完整打印出 Blueprint v3.3 中定义的 `event_sequence` 数组**，向人类架构师展示系统在崩溃前走到了哪一步物理状态。随后立即在原函数内修复防爆逻辑并重试。
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

### 6.4 法则四：LIFO 内存析构强制令 (LIFO Cleanup Contract)
**绝对禁止内存与句柄泄漏。** 任何在引擎内部申请宿主资源的操作，必须严格执行“后进先出”清理。
*   特别是当 `debug_plots=True` 时，如果 Agent 生成了 Matplotlib 探针图像，**必须使用 `try...finally:` 块结构**。
*   在 `finally:` 块中，必须显式调用 `plt.close(fig)` 和相应的垃圾回收逻辑，确保即使探针函数发生内部崩溃，底层图像句柄也能被释放，保证连续批处理（Epoch-over-Epoch）不会导致服务器 OOM。

> **终极训诫:** 
> “LLM，你不是造物主，你是一台精密机床。用第 6 章的法则测试你的每一行代码，用代码去印证图纸；当你失败时，交出你的 Event Log。如有违约，自我裁决。”