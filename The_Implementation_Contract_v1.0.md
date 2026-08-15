# 📜 The Implementation Contract (v1.0)
**—— 针对 Monolithic Tensor Engine v3.2 的防漂移底层执行契约**

> **目标:** 封死 LLM 在代码生成中的微观参数漂移、库幻觉与边界条件崩溃。
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
    value_array: np.ndarray
) -> dict:
    """
    输入必须是对齐的 1D numpy array。
    输出必须严格符合 Blueprint v3.2 中的 SSVS JSON 字典结构，包含 kairos_routing_flags。
    """
    # 强制执行此函数，内部错误一律以安全默认值/NaN 掩码抛出，禁止程序 Crash。
```

> **最终训诫:** 
> “LLM，放弃你对数学算法的创造力。你是一台执行这 5 个章节死命令的打字机。任何参数的偏离，都将导致物理引擎的崩溃。”