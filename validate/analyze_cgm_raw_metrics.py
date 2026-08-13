#!/usr/bin/env python3
"""Label-blind, subject-level CGM metric research pipeline.

The loader deliberately copies only cohort/id/timestamps/values plus technical
window/device fields. Clinical labels present in some historical export files
are never read into the analysis frame or written to the research outputs.

Primary numerical convention:
  * glucose unit: mmol/L
  * common grid: 5 minutes
  * interpolate only across observed gaps <= 15 minutes
  * Takens embedding: m=3, tau selected from ACF and capped at 60 minutes
  * RQA recurrence threshold: subject/window-specific 2% distance quantile

This script is research-only. It does not edit index.html, models, thresholds,
or raw source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
RESEARCH_OUTPUT = OUTPUT / "cgm_metric_research"
SEED = 20260812
GRID_MINUTES = 5
MAX_INTERPOLATION_GAP_MINUTES = 15
PRIMARY_48_KEYS = (
    "colas_w48",
    "hall_w48",
    "iglu_w48",
    "dubosson_w48",
    "weinstock_w48",
    "cgmacros_libre_w48",
)
EXTERNAL_48_KEYS = ("kobe_shift0_w48", "shanghai_t2dm_w48", "big_ideas_w48")
LENGTH_24_KEYS = (
    "colas_w24",
    "hall_w24",
    "iglu_w24",
    "dubosson_w24",
    "weinstock_w24",
    "cgmacros_libre_w24",
)
SIMPLE_BASES = ("mean_glucose", "cv_pct", "tir_70_180")


@dataclass
class SeriesRecord:
    role: str
    cohort: str
    subject_id: str
    timestamps: List[str]
    values: List[float]
    window: str
    split: str = ""
    device: str = ""


METRIC_DEFINITIONS: List[dict] = [
    {"id": "mean_glucose", "label": "全日平均血糖", "group": "standard", "unit": "mmol/L", "boundary": "暴露水平描述；不定义健康或诊断"},
    {"id": "night_mean", "label": "夜间平均血糖", "group": "standard", "unit": "mmol/L", "boundary": "00:00-06:00窗口描述；受睡眠和晚餐影响"},
    {"id": "sd_glucose", "label": "血糖标准差", "group": "standard", "unit": "mmol/L", "boundary": "波动幅度描述"},
    {"id": "cv_pct", "label": "变异系数 CV", "group": "standard", "unit": "%", "boundary": "相对波动描述；依赖均糖"},
    {"id": "tir_70_180", "label": "TIR 70-180", "group": "standard", "unit": "%", "boundary": "范围暴露描述；非诊断"},
    {"id": "tar_180", "label": "TAR >180", "group": "standard", "unit": "%", "boundary": "高糖范围暴露"},
    {"id": "tar_250", "label": "TAR >250", "group": "standard", "unit": "%", "boundary": "较高糖范围暴露"},
    {"id": "tbr_70", "label": "TBR <70", "group": "standard", "unit": "%", "boundary": "低糖范围暴露"},
    {"id": "tbr_54", "label": "TBR <54", "group": "standard", "unit": "%", "boundary": "较低糖范围暴露"},
    {"id": "gmi", "label": "GMI", "group": "standard", "unit": "%", "boundary": "仅在>=14天且覆盖>=70%时输出；不是实验室A1C"},
    {"id": "mage", "label": "MAGE", "group": "standard", "unit": "mmol/L", "boundary": "实现依赖明显；报告算法版本"},
    {"id": "modd", "label": "MODD", "group": "standard", "unit": "mmol/L", "boundary": "相邻日同一时刻差异"},
    {"id": "linear_slope_day", "label": "全窗线性趋势", "group": "linear", "unit": "mmol/L/day", "boundary": "窗口内趋势，不外推"},
    {"id": "median_daily_slope", "label": "逐日趋势中位数", "group": "linear", "unit": "mmol/L/day", "boundary": "合格自然日内趋势"},
    {"id": "median_night_slope", "label": "逐夜趋势中位数", "group": "linear", "unit": "mmol/L/hour", "boundary": "合格夜间内趋势"},
    {"id": "rate_mean_abs", "label": "平均绝对变化速率", "group": "linear", "unit": "mmol/L/hour", "boundary": "连续有效点的一阶变化"},
    {"id": "rate_p95_up", "label": "P95上升速率", "group": "linear", "unit": "mmol/L/hour", "boundary": "上升尾部速率"},
    {"id": "rate_p95_down", "label": "P95下降速率", "group": "linear", "unit": "mmol/L/hour", "boundary": "下降尾部绝对速率"},
    {"id": "event_amplitude", "label": "信号扰动幅度", "group": "event", "unit": "mmol/L", "boundary": "盲法自由生活扰动，不等同餐后反应"},
    {"id": "event_iauc_0_180", "label": "iAUC 0-180", "group": "event", "unit": "mmol*min/L", "boundary": "相对事件前基线的正面积"},
    {"id": "event_rb_0_180", "label": "RB 0-180", "group": "event", "unit": "mmol*min/L", "boundary": "本协议下与iAUC同公式，用于恢复负担命名"},
    {"id": "time_to_decel", "label": "Time-to-Deceleration", "group": "event", "unit": "min", "boundary": "事件上升段最大斜率出现时间"},
    {"id": "time_to_peak", "label": "Time-to-Peak", "group": "event", "unit": "min", "boundary": "信号事件起点到峰值"},
    {"id": "recovery_t50", "label": "半恢复时间 T50", "group": "event", "unit": "min", "boundary": "峰后回落50%振幅耗时"},
    {"id": "recovery_baseline_time", "label": "回归基线时间", "group": "event", "unit": "min", "boundary": "峰后回落至基线+10%振幅"},
    {"id": "unrecovered_event_pct", "label": "未恢复事件比例", "group": "event", "unit": "%", "boundary": "180分钟内未回归基线+10%"},
    {"id": "log_volume", "label": "相空间展开体积 logV", "group": "takens", "unit": "log(mmol/L)^m", "boundary": "几何尺度；无统一健康方向"},
    {"id": "recovery", "label": "归一化向心步长", "group": "takens", "unit": "ratio", "boundary": "几何恢复代理；无统一健康方向"},
    {"id": "shape_ratio", "label": "主轴各向异性", "group": "takens", "unit": "ratio", "boundary": "几何形态；无统一健康方向"},
    {"id": "core_dist", "label": "当前-夜间核心距离", "group": "takens", "unit": "mmol/L", "boundary": "状态差异描述"},
    {"id": "box_dimension", "label": "盒计数维度", "group": "takens", "unit": "dimensionless", "boundary": "有限样本充盈度；非生理自由度"},
    {"id": "lyapunov_rosenstein", "label": "Rosenstein最大Lyapunov代理", "group": "nonlinear", "unit": "1/hour", "boundary": "有限窗发散率代理，不是严格连续系统指数"},
    {"id": "lyapunov_one_step", "label": "一步近邻发散代理", "group": "nonlinear", "unit": "log ratio/step", "boundary": "与页面轻量指标同类，强依赖采样"},
    {"id": "rqa_det", "label": "RQA DET", "group": "nonlinear", "unit": "%", "boundary": "条件化重复结构；不等同生理僵化"},
    {"id": "rqa_entr", "label": "RQA ENTR", "group": "nonlinear", "unit": "nat", "boundary": "对角线长度多样性"},
    {"id": "rqa_lam", "label": "RQA LAM", "group": "nonlinear", "unit": "%", "boundary": "垂直递归结构比例"},
    {"id": "rqa_tt", "label": "RQA Trapping Time", "group": "nonlinear", "unit": "steps", "boundary": "垂直线平均长度"},
    {"id": "sample_entropy", "label": "Sample Entropy", "group": "nonlinear", "unit": "nat", "boundary": "m=2,r=0.2SD；短窗和噪声敏感"},
    {"id": "permutation_entropy", "label": "Permutation Entropy", "group": "nonlinear", "unit": "normalized", "boundary": "阶数3、delay=tau；序模式复杂度"},
    {"id": "night_ar1_detrended", "label": "去趋势夜间AR1", "group": "nonlinear", "unit": "coefficient", "boundary": "相关记忆代理；不是临床熔断线"},
    {"id": "angular_velocity", "label": "相平面角速度", "group": "aether", "unit": "rad/hour", "boundary": "工程代理；健康方向待验证"},
    {"id": "work_integral", "label": "相空间磁滞做功积分", "group": "aether", "unit": "arbitrary", "boundary": "工程面积代理；不是热量或生理功"},
    {"id": "ascend_friction", "label": "上升相阻力", "group": "aether", "unit": "ratio", "boundary": "距离/步长工程比值；不是物理摩擦系数"},
    {"id": "night_friction", "label": "夜间相变阻力", "group": "aether", "unit": "ratio", "boundary": "夜间距离/步长工程比值"},
]
METRIC_IDS = [item["id"] for item in METRIC_DEFINITIONS]
METRIC_LOOKUP = {item["id"]: item for item in METRIC_DEFINITIONS}

OPERATIONAL_DEFINITIONS = {
    "mean_glucose": "有效规则网格值的算术平均",
    "night_mean": "本地时钟00:00-06:00有效值的算术平均",
    "sd_glucose": "有效规则网格值的样本标准差(ddof=1)",
    "cv_pct": "100*SD/mean",
    "tir_70_180": "3.9-10.0 mmol/L闭区间内时间百分比",
    "tar_180": ">10.0 mmol/L时间百分比",
    "tar_250": ">13.9 mmol/L时间百分比",
    "tbr_70": "<3.9 mmol/L时间百分比",
    "tbr_54": "<3.0 mmol/L时间百分比",
    "gmi": "3.31+0.02392*mean(mg/dL)，仅在跨度>=14天且覆盖>=70%时输出",
    "mage": "相邻方向相反且振幅>=1个全窗SD的有效极值差之均值",
    "modd": "相隔24小时的同一网格点绝对差之均值",
    "linear_slope_day": "全窗 glucose~elapsed_hours OLS斜率*24",
    "median_daily_slope": "覆盖>=70%的自然日 OLS 日斜率中位数",
    "median_night_slope": "覆盖>=70%的00:00-06:00 OLS 小时斜率中位数",
    "rate_mean_abs": "相邻有效网格一阶差分绝对值/小时的均值",
    "rate_p95_up": "正一阶变化速率的95百分位",
    "rate_p95_down": "负一阶变化速率绝对值的95百分位",
    "event_amplitude": "盲法检测的局部扰动峰值减事件前30分钟中位基线",
    "event_iauc_0_180": "事件后0-180分钟 max(glucose-baseline,0) 梯形积分",
    "event_rb_0_180": "与本分析iAUC相同的正偏离积分，仅作为恢复负担命名",
    "time_to_decel": "事件起点至上升段最大正斜率时刻",
    "time_to_peak": "事件起点至事件峰值时刻",
    "recovery_t50": "峰后首次回落至 baseline+0.5*amplitude 的时间",
    "recovery_baseline_time": "峰后首次回落至 baseline+0.1*amplitude 的时间",
    "unrecovered_event_pct": "180分钟内未回到 baseline+0.1*amplitude 的事件比例",
    "log_volume": "m=3延迟嵌入的99%有效维椭球体积之log",
    "recovery": "向全窗嵌入稳健核心靠近时的相邻步长均值/原信号SD",
    "shape_ratio": "嵌入协方差第一特征值/第二特征值",
    "core_dist": "全窗嵌入稳健中心与夜间嵌入稳健中心的欧氏距离",
    "box_dimension": "四个尺度下占据盒数与1/尺度的log-log OLS斜率",
    "lyapunov_rosenstein": "带Theiler窗的最近邻平均log发散曲线早段斜率(每小时)",
    "lyapunov_one_step": "最近邻距离的一步log比值中位数",
    "rqa_det": "递归点中属于长度>=2对角线的比例",
    "rqa_entr": "长度>=2对角线长度分布的Shannon熵",
    "rqa_lam": "递归点中属于长度>=2垂直线的比例",
    "rqa_tt": "长度>=2垂直线的平均长度",
    "sample_entropy": "SampEn(m=2,r=0.2*SD)",
    "permutation_entropy": "归一化排列熵(order=3,delay=tau)",
    "night_ar1_detrended": "逐夜去线性趋势后残差的一阶自相关系数中位数",
    "angular_velocity": "以夜间血糖中位数为中心，在(G-core,dG/dt)平面按离散角动量/半径平方汇总",
    "work_integral": "Takens嵌入前两坐标围绕全窗稳健核心的有向面积绝对值/2",
    "ascend_friction": "上升相中到全窗嵌入稳健核心距离/相邻嵌入步长的均值",
    "night_friction": "夜间下降相中到夜间嵌入稳健核心距离/相邻嵌入步长的均值",
}


def safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_time_glucose(row: dict) -> Tuple[List[str], List[float]]:
    """Whitelist copy: deliberately never access any clinical endpoint key."""
    timestamps = list(row.get("timestamps", []))
    values = []
    for value in row.get("values", []):
        number = safe_float(value)
        values.append(number if number is not None else float("nan"))
    return timestamps, values


def load_records() -> Tuple[List[SeriesRecord], List[SeriesRecord], List[SeriesRecord], List[SeriesRecord]]:
    composite = load_json(OUTPUT / "composite_abnormality_subjects.json")
    external = load_json(OUTPUT / "external_base5_subjects.json")
    reserve = load_json(OUTPUT / "structure_reserve_windows.json")
    full = load_json(OUTPUT / "phase_screening_subjects.json")

    primary: List[SeriesRecord] = []
    for key in PRIMARY_48_KEYS:
        for row in composite.get(key, []):
            timestamps, values = copy_time_glucose(row)
            device = "libre" if key.startswith("cgmacros_libre") else ""
            primary.append(SeriesRecord("primary48", key.removesuffix("_w48"), str(row.get("id")), timestamps, values, "48h", device=device))
    for key in EXTERNAL_48_KEYS:
        for row in external.get(key, []):
            timestamps, values = copy_time_glucose(row)
            primary.append(SeriesRecord("primary48", key.removesuffix("_w48"), str(row.get("id")), timestamps, values, "48h"))

    length24: List[SeriesRecord] = []
    for key in LENGTH_24_KEYS:
        for row in composite.get(key, []):
            timestamps, values = copy_time_glucose(row)
            device = "libre" if key.startswith("cgmacros_libre") else ""
            length24.append(SeriesRecord("length24", key.removesuffix("_w24"), str(row.get("id")), timestamps, values, "24h", device=device))

    reliability: List[SeriesRecord] = []
    for row in reserve.get("stateRecords", []):
        split = str(row.get("split", ""))
        if split not in {"odd", "even"}:
            continue
        timestamps, values = copy_time_glucose(row)
        reliability.append(SeriesRecord("reliability", str(row.get("cohort")), str(row.get("id")), timestamps, values, "2day_alternate", split=split))

    device: List[SeriesRecord] = []
    for key, device_name in (("cgmacros_libre_w48", "libre"), ("cgmacros_dexcom_w48", "dexcom")):
        for row in composite.get(key, []):
            timestamps, values = copy_time_glucose(row)
            device.append(SeriesRecord("device", "cgmacros", str(row.get("id")), timestamps, values, "48h", device=device_name))

    # Full records are used only for GMI qualification audit. The same whitelist
    # is applied; very sparse calendar-spanning records are never regularized.
    long_records: List[SeriesRecord] = []
    for cohort in ("hall", "colas"):
        for row in full.get(cohort, []):
            timestamps, values = copy_time_glucose(row)
            long_records.append(SeriesRecord("gmi_audit", cohort, str(row.get("id")), timestamps, values, "full"))
    return primary, length24, reliability, device, long_records


def sanitize_series(record: SeriesRecord) -> pd.DataFrame:
    frame = pd.DataFrame({"time": pd.to_datetime(record.timestamps, errors="coerce"), "glucose": pd.to_numeric(pd.Series(record.values), errors="coerce")})
    frame = frame.dropna(subset=["time", "glucose"]).copy()
    frame = frame[(frame["glucose"] >= 2.2) & (frame["glucose"] <= 33.3)]
    frame = frame.sort_values("time").groupby("time", as_index=False)["glucose"].mean()
    return frame


def regularize(frame: pd.DataFrame, grid_minutes: int = GRID_MINUTES, ema_alpha: Optional[float] = None) -> pd.DataFrame:
    if len(frame) < 2:
        return pd.DataFrame(columns=["time", "glucose"])
    start = frame["time"].iloc[0].floor(f"{grid_minutes}min")
    end = frame["time"].iloc[-1].ceil(f"{grid_minutes}min")
    if end <= start:
        return pd.DataFrame(columns=["time", "glucose"])
    grid = pd.date_range(start, end, freq=f"{grid_minutes}min")
    output = np.full(len(grid), np.nan, dtype=float)
    observed_time = frame["time"].astype("int64").to_numpy(dtype=np.int64) / 60_000_000_000.0
    observed_value = frame["glucose"].to_numpy(dtype=float)
    grid_time = grid.astype("int64").to_numpy(dtype=np.int64) / 60_000_000_000.0
    split_points = np.flatnonzero(np.diff(observed_time) > MAX_INTERPOLATION_GAP_MINUTES) + 1
    for indices in np.split(np.arange(len(observed_time)), split_points):
        if len(indices) == 0:
            continue
        lo, hi = observed_time[indices[0]], observed_time[indices[-1]]
        mask = (grid_time >= lo) & (grid_time <= hi)
        if len(indices) == 1:
            nearest = int(np.argmin(np.abs(grid_time - lo)))
            if abs(grid_time[nearest] - lo) <= grid_minutes / 2:
                output[nearest] = observed_value[indices[0]]
        else:
            output[mask] = np.interp(grid_time[mask], observed_time[indices], observed_value[indices])
    if ema_alpha is not None:
        smoothed = output.copy()
        state = np.nan
        for i, value in enumerate(output):
            if not np.isfinite(value):
                state = np.nan
                continue
            state = value if not np.isfinite(state) else ema_alpha * value + (1.0 - ema_alpha) * state
            smoothed[i] = state
        output = smoothed
    return pd.DataFrame({"time": grid, "glucose": output})


def longest_missing_gap_minutes(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return float("nan")
    gaps = frame["time"].diff().dt.total_seconds().div(60).dropna()
    return float(gaps.max()) if len(gaps) else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def ols_slope(times_hours: np.ndarray, values: np.ndarray, unit_hours: float = 24.0) -> float:
    mask = np.isfinite(times_hours) & np.isfinite(values)
    if mask.sum() < 3:
        return float("nan")
    x = times_hours[mask]
    y = values[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y - y.mean()) / denom * unit_hours)


def compute_mage(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < 24:
        return float("nan")
    series = pd.Series(values).rolling(3, center=True, min_periods=1).mean().to_numpy()
    sd = float(np.nanstd(series, ddof=1))
    if not math.isfinite(sd) or sd <= 1e-12:
        return 0.0
    finite_idx = np.flatnonzero(np.isfinite(series))
    extrema: List[int] = []
    for pos in range(1, len(finite_idx) - 1):
        i0, i1, i2 = finite_idx[pos - 1], finite_idx[pos], finite_idx[pos + 1]
        if i1 != i0 + 1 or i2 != i1 + 1:
            continue
        a, b, c = series[i0], series[i1], series[i2]
        if (b >= a and b > c) or (b <= a and b < c):
            extrema.append(i1)
    if len(extrema) < 2:
        return float("nan")
    excursions = [abs(series[b] - series[a]) for a, b in zip(extrema[:-1], extrema[1:]) if abs(series[b] - series[a]) >= sd]
    return float(np.mean(excursions)) if excursions else float("nan")


def compute_modd(values: np.ndarray, points_per_day: int) -> float:
    if len(values) <= points_per_day:
        return float("nan")
    first, second = values[:-points_per_day], values[points_per_day:]
    mask = np.isfinite(first) & np.isfinite(second)
    return float(np.mean(np.abs(second[mask] - first[mask]))) if mask.sum() >= 12 else float("nan")


def event_metrics(times: pd.Series, values: np.ndarray, grid_minutes: int) -> dict:
    n = len(values)
    if n < int(6 * 60 / grid_minutes):
        return {key: float("nan") for key in (
            "event_amplitude", "event_iauc_0_180", "event_rb_0_180", "time_to_decel",
            "time_to_peak", "recovery_t50", "recovery_baseline_time", "unrecovered_event_pct"
        )} | {"event_count": 0}
    back = int(60 / grid_minutes)
    forward = int(180 / grid_minutes)
    min_separation = int(180 / grid_minutes)
    candidates = []
    for i in range(back, n - 1):
        if not np.isfinite(values[i]):
            continue
        local = values[max(0, i - 2):min(n, i + 3)]
        if not np.isfinite(local).all() or values[i] < np.max(local):
            continue
        prior = values[i - back:i]
        if np.isfinite(prior).sum() < max(6, back // 2):
            continue
        baseline = float(np.nanmedian(prior))
        amplitude = values[i] - baseline
        if amplitude >= 1.5:
            candidates.append((float(amplitude), i, baseline))
    selected = []
    for amplitude, peak, baseline in sorted(candidates, reverse=True):
        if all(abs(peak - other[1]) >= min_separation for other in selected):
            selected.append((amplitude, peak, baseline))
    selected.sort(key=lambda item: item[1])
    rows = []
    for amplitude, peak, baseline_hint in selected:
        search_start = max(0, peak - int(90 / grid_minutes))
        prior = values[search_start:peak + 1]
        if np.isfinite(prior).sum() < 6:
            continue
        onset = search_start + int(np.nanargmin(prior))
        baseline_window = values[max(0, onset - int(30 / grid_minutes)):onset + 1]
        baseline = float(np.nanmedian(baseline_window)) if np.isfinite(baseline_window).sum() >= 3 else baseline_hint
        amplitude = float(values[peak] - baseline)
        if amplitude < 1.5 or peak <= onset:
            continue
        rising = values[onset:peak + 1]
        diffs = np.diff(rising)
        finite_diff = np.flatnonzero(np.isfinite(diffs))
        decel = float("nan")
        if len(finite_diff):
            max_idx = finite_diff[int(np.argmax(diffs[finite_diff]))]
            decel = max_idx * grid_minutes
        end = min(n - 1, onset + forward)
        segment = values[onset:end + 1]
        if np.isfinite(segment).sum() < int(120 / grid_minutes):
            continue
        excess = np.maximum(segment - baseline, 0.0)
        excess[~np.isfinite(segment)] = np.nan
        pair_mask = np.isfinite(excess[:-1]) & np.isfinite(excess[1:])
        iauc = float(np.sum((excess[:-1][pair_mask] + excess[1:][pair_mask]) * 0.5 * grid_minutes))
        after_peak = values[peak:end + 1]
        t50_idx = np.flatnonzero(np.isfinite(after_peak) & (after_peak <= baseline + 0.5 * amplitude))
        base_idx = np.flatnonzero(np.isfinite(after_peak) & (after_peak <= baseline + 0.1 * amplitude))
        rows.append({
            "event_amplitude": amplitude,
            "event_iauc_0_180": iauc,
            "event_rb_0_180": iauc,
            "time_to_decel": decel,
            "time_to_peak": (peak - onset) * grid_minutes,
            "recovery_t50": float(t50_idx[0] * grid_minutes) if len(t50_idx) else float("nan"),
            "recovery_baseline_time": float(base_idx[0] * grid_minutes) if len(base_idx) else float("nan"),
            "unrecovered": 0.0 if len(base_idx) else 1.0,
        })
    if not rows:
        return {key: float("nan") for key in (
            "event_amplitude", "event_iauc_0_180", "event_rb_0_180", "time_to_decel",
            "time_to_peak", "recovery_t50", "recovery_baseline_time", "unrecovered_event_pct"
        )} | {"event_count": 0}
    result = {}
    for key in (
        "event_amplitude", "event_iauc_0_180", "event_rb_0_180", "time_to_decel",
        "time_to_peak", "recovery_t50", "recovery_baseline_time"
    ):
        candidates = np.asarray([row[key] for row in rows], dtype=float)
        candidates = candidates[np.isfinite(candidates)]
        result[key] = float(np.median(candidates)) if len(candidates) else float("nan")
    result["unrecovered_event_pct"] = float(100 * np.mean([row["unrecovered"] for row in rows]))
    result["event_count"] = len(rows)
    return result


def acf_tau(values: np.ndarray, grid_minutes: int) -> Tuple[int, np.ndarray]:
    max_lag = max(2, int(60 / grid_minutes))
    valid = values[np.isfinite(values)]
    if len(valid) < 20 or np.nanvar(valid) <= 1e-12:
        return 1, np.zeros(max_lag + 1)
    mean = float(np.mean(valid))
    variance = float(np.mean((valid - mean) ** 2))
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        a, b = values[:-lag], values[lag:]
        mask = np.isfinite(a) & np.isfinite(b)
        acf[lag] = float(np.mean((a[mask] - mean) * (b[mask] - mean)) / variance) if mask.sum() >= 10 else 0.0
    decayed = False
    for lag in range(1, max_lag):
        if acf[lag] < 0.8:
            decayed = True
        if acf[lag] < 1 / math.e:
            return lag, acf
        if decayed and acf[lag] > acf[lag - 1] and acf[lag + 1] >= acf[lag]:
            return max(1, lag - 1), acf
    return max_lag, acf


def takens_embedding(values: np.ndarray, tau: int, dim: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    count = len(values) - (dim - 1) * tau
    if count <= 0:
        return np.empty((0, dim)), np.empty(0, dtype=int)
    rows, indices = [], []
    for i in range(count):
        point = values[i:i + dim * tau:tau]
        if len(point) == dim and np.isfinite(point).all():
            rows.append(point)
            indices.append(i)
    return (np.asarray(rows, dtype=float), np.asarray(indices, dtype=int)) if rows else (np.empty((0, dim)), np.empty(0, dtype=int))


def runs_of_true(values: np.ndarray, minimum: int = 2) -> List[int]:
    if len(values) == 0:
        return []
    padded = np.concatenate(([False], values.astype(bool), [False])).astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [int(end - start) for start, end in zip(starts, ends) if end - start >= minimum]


def box_dimension(points: np.ndarray) -> float:
    if len(points) < 20:
        return float("nan")
    ranges = np.ptp(points, axis=0)
    max_range = float(np.max(ranges))
    if max_range <= 1e-12:
        return 0.0
    xs, ys = [], []
    minima = np.min(points, axis=0)
    for divisions in (2, 4, 8, 16):
        epsilon = max_range / divisions
        cells = np.floor((points - minima) / epsilon).astype(int)
        occupied = len({tuple(row) for row in cells})
        xs.append(math.log(divisions / max_range))
        ys.append(math.log(max(occupied, 1)))
    slope = np.polyfit(xs, ys, 1)[0]
    return float(np.clip(slope, 0.0, points.shape[1]))


def phase_metrics(values: np.ndarray, times: pd.Series, grid_minutes: int) -> dict:
    tau, _ = acf_tau(values, grid_minutes)
    smooth = pd.Series(values).ewm(alpha=0.3, adjust=False, ignore_na=False).mean().to_numpy(copy=True)
    smooth[~np.isfinite(values)] = np.nan
    raw_points, raw_indices = takens_embedding(values, tau, 3)
    points, point_indices = takens_embedding(smooth, tau, 3)
    result = {
        "tau_steps": tau,
        "tau_minutes": tau * grid_minutes,
        "embedding_dim": 3,
        "valid_phase_points": len(points),
    }
    if len(points) < 30 or len(raw_points) < 30:
        for key in ("log_volume", "recovery", "shape_ratio", "core_dist", "box_dimension", "lyapunov_rosenstein", "lyapunov_one_step", "rqa_det", "rqa_entr", "rqa_lam", "rqa_tt", "sample_entropy", "permutation_entropy", "angular_velocity", "work_integral", "ascend_friction", "night_friction"):
            result[key] = float("nan")
        return result

    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 1e-10)
    ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    effective_dim = int(np.searchsorted(ratio, 0.99) + 1)
    coefficient = math.pi ** (effective_dim / 2) / math.gamma(effective_dim / 2 + 1)
    volume = coefficient * float(np.prod(np.sqrt(eigenvalues[:effective_dim])))
    result["log_volume"] = float(math.log(max(volume, 1e-12)))
    result["shape_ratio"] = float(eigenvalues[0] / eigenvalues[1])
    result["box_dimension"] = box_dimension(points)
    result["effective_dim"] = effective_dim

    raw_core = np.median(raw_points, axis=0)
    raw_sd = float(np.std(raw_points[:, 0], ddof=1))
    recovery_steps = []
    work = 0.0
    ascend = []
    descend = []
    prior_distance = None
    for i in range(1, len(raw_points)):
        if raw_indices[i] != raw_indices[i - 1] + 1:
            prior_distance = None
            continue
        current = raw_points[i]
        previous = raw_points[i - 1]
        distance = float(np.linalg.norm(current - raw_core))
        step = float(np.linalg.norm(current - previous))
        if prior_distance is not None and distance < prior_distance:
            recovery_steps.append(step)
        prior_distance = distance
        work += (previous[0] - raw_core[0]) * (current[1] - previous[1]) - (previous[1] - raw_core[1]) * (current[0] - previous[0])
        delta_g = current[0] - previous[0]
        if step > 1e-9 and delta_g > 0.01:
            ascend.append(distance / step)
        elif step > 1e-9 and delta_g < -0.01:
            descend.append(distance / step)
    result["recovery"] = float(np.mean(recovery_steps) / raw_sd) if recovery_steps and raw_sd > 1e-9 else float("nan")
    result["work_integral"] = abs(float(work)) / 2.0
    result["ascend_friction"] = float(np.mean(ascend)) if ascend else float("nan")

    hours = pd.to_datetime(times).dt.hour.to_numpy()
    night_values = values[(hours >= 0) & (hours < 6)]
    night_points, night_indices = takens_embedding(night_values, tau, 3)
    if len(night_points) >= 10:
        night_core = np.median(night_points, axis=0)
        result["core_dist"] = float(np.linalg.norm(raw_core - night_core))
        night_friction_values = []
        for i in range(1, len(night_points)):
            if night_indices[i] != night_indices[i - 1] + 1:
                continue
            step = float(np.linalg.norm(night_points[i] - night_points[i - 1]))
            delta_g = night_points[i, 0] - night_points[i - 1, 0]
            if step > 1e-9 and delta_g < -0.01:
                night_friction_values.append(float(np.linalg.norm(night_points[i] - night_core)) / step)
        result["night_friction"] = float(np.mean(night_friction_values)) if night_friction_values else float("nan")
    else:
        result["core_dist"] = float("nan")
        result["night_friction"] = float("nan")

    # Pairwise distances support both nearest-neighbour divergence and RQA.
    differences = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    compact_n = len(points)
    temporal = np.abs(point_indices[:, None] - point_indices[None, :])
    neighbour_dist = distances.copy()
    neighbour_dist[temporal <= 5] = np.inf
    nearest = np.argmin(neighbour_dist, axis=1)
    nearest_distance = neighbour_dist[np.arange(compact_n), nearest]
    valid_neighbour = np.isfinite(nearest_distance) & (nearest_distance > 1e-9)
    one_step = []
    divergence_curves = []
    max_horizon = max(3, int(60 / grid_minutes))
    for i in np.flatnonzero(valid_neighbour):
        j = int(nearest[i])
        if i + 1 < compact_n and j + 1 < compact_n and point_indices[i + 1] == point_indices[i] + 1 and point_indices[j + 1] == point_indices[j] + 1:
            d1 = distances[i + 1, j + 1]
            if d1 > 1e-9:
                one_step.append(math.log(d1 / nearest_distance[i]))
        curve = []
        for horizon in range(max_horizon + 1):
            if i + horizon >= compact_n or j + horizon >= compact_n:
                break
            if point_indices[i + horizon] != point_indices[i] + horizon or point_indices[j + horizon] != point_indices[j] + horizon:
                break
            distance = distances[i + horizon, j + horizon]
            curve.append(math.log(max(distance, 1e-9)))
        if len(curve) >= 4:
            divergence_curves.append(curve)
    result["lyapunov_one_step"] = float(np.mean(one_step)) if len(one_step) >= 10 else float("nan")
    horizon_means, horizon_x = [], []
    for horizon in range(max_horizon + 1):
        samples = [curve[horizon] for curve in divergence_curves if len(curve) > horizon]
        if len(samples) >= 20:
            horizon_x.append(horizon * grid_minutes / 60.0)
            horizon_means.append(float(np.mean(samples)))
    fit_count = min(len(horizon_x), max(4, int(30 / grid_minutes) + 1))
    result["lyapunov_rosenstein"] = float(np.polyfit(horizon_x[:fit_count], horizon_means[:fit_count], 1)[0]) if fit_count >= 4 else float("nan")

    upper_mask = np.triu(temporal > 5, 1)
    candidate_distances = distances[upper_mask]
    candidate_distances = candidate_distances[np.isfinite(candidate_distances)]
    if len(candidate_distances) >= 50:
        epsilon = float(np.quantile(candidate_distances, 0.02))
        recurrence = (distances <= epsilon) & (temporal > 5)
        np.fill_diagonal(recurrence, False)
        recurrence_count = int(recurrence.sum())
        diagonal_lengths = []
        for offset in range(-(compact_n - 1), compact_n):
            diagonal_lengths.extend(runs_of_true(np.diagonal(recurrence, offset=offset), 2))
        vertical_lengths = []
        for column in range(compact_n):
            vertical_lengths.extend(runs_of_true(recurrence[:, column], 2))
        det_points = sum(diagonal_lengths)
        lam_points = sum(vertical_lengths)
        result["rqa_det"] = 100.0 * det_points / recurrence_count if recurrence_count else float("nan")
        result["rqa_lam"] = 100.0 * lam_points / recurrence_count if recurrence_count else float("nan")
        result["rqa_tt"] = float(np.mean(vertical_lengths)) if vertical_lengths else float("nan")
        if diagonal_lengths:
            counts = np.asarray(list(Counter(diagonal_lengths).values()), dtype=float)
            probabilities = counts / counts.sum()
            result["rqa_entr"] = float(-np.sum(probabilities * np.log(probabilities)))
        else:
            result["rqa_entr"] = float("nan")
        result["rqa_rr"] = 100.0 * recurrence_count / max(int((temporal > 5).sum()), 1)
        result["rqa_epsilon"] = epsilon
    else:
        result.update({"rqa_det": float("nan"), "rqa_entr": float("nan"), "rqa_lam": float("nan"), "rqa_tt": float("nan"), "rqa_rr": float("nan"), "rqa_epsilon": float("nan")})

    result["sample_entropy"] = sample_entropy(values, 2, 0.2)
    result["permutation_entropy"] = permutation_entropy(values, 3, tau)
    result["angular_velocity"] = angular_velocity(values, times, result.get("night_core_scalar", float(np.nanmedian(night_values)) if np.isfinite(night_values).any() else float("nan")))
    return result


def sample_entropy(values: np.ndarray, order: int = 2, r_fraction: float = 0.2) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < 40:
        return float("nan")
    sd = float(np.std(valid, ddof=1))
    tolerance = r_fraction * sd
    if tolerance <= 1e-12:
        return 0.0
    counts = []
    for dimension in (order, order + 1):
        patterns = []
        for i in range(len(values) - dimension + 1):
            pattern = values[i:i + dimension]
            if np.isfinite(pattern).all():
                patterns.append(pattern)
        if len(patterns) < 20:
            return float("nan")
        matrix = np.asarray(patterns)
        distances = np.max(np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2)
        matches = int(np.sum(np.triu(distances <= tolerance, 1)))
        counts.append(matches)
    if counts[0] <= 0 or counts[1] <= 0:
        return float("nan")
    return float(-math.log(counts[1] / counts[0]))


def permutation_entropy(values: np.ndarray, order: int = 3, delay: int = 1) -> float:
    patterns = []
    width = (order - 1) * delay + 1
    for i in range(len(values) - width + 1):
        pattern = values[i:i + width:delay]
        if np.isfinite(pattern).all():
            patterns.append(tuple(np.argsort(pattern, kind="stable")))
    if len(patterns) < 20:
        return float("nan")
    counts = np.asarray(list(Counter(patterns).values()), dtype=float)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(math.factorial(order))


def angular_velocity(values: np.ndarray, times: pd.Series, core: float) -> float:
    if not math.isfinite(core) or len(values) < 10:
        return float("nan")
    parsed_times = pd.to_datetime(times)
    hours = (parsed_times - parsed_times.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    x = values - core
    y = np.full(len(values), np.nan)
    for i in range(1, len(values) - 1):
        if np.isfinite(x[i - 1:i + 2]).all() and hours[i + 1] > hours[i - 1]:
            y[i] = (x[i + 1] - x[i - 1]) / (hours[i + 1] - hours[i - 1])
    dy = np.full(len(values), np.nan)
    for i in range(1, len(values) - 1):
        if np.isfinite(y[i - 1:i + 2]).all() and hours[i + 1] > hours[i - 1]:
            dy[i] = (y[i + 1] - y[i - 1]) / (hours[i + 1] - hours[i - 1])
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy) & ((x * x + y * y) >= 0.05)
    if mask.sum() < 5:
        return float("nan")
    sweep = 0.5 * np.abs(x[mask] * dy[mask] - y[mask] * y[mask])
    radius_sq = x[mask] * x[mask] + y[mask] * y[mask]
    return float(2.0 * np.sum(sweep) / np.sum(radius_sq)) if np.sum(radius_sq) > 0 else float("nan")


def detrended_night_ar1(times: pd.Series, values: np.ndarray) -> float:
    frame = pd.DataFrame({"time": pd.to_datetime(times), "glucose": values})
    frame = frame[np.isfinite(frame["glucose"])].copy()
    if frame.empty:
        return float("nan")
    frame["date"] = frame["time"].dt.date
    residual_pairs = []
    for _, group in frame[(frame["time"].dt.hour >= 0) & (frame["time"].dt.hour < 6)].groupby("date"):
        if len(group) < 24:
            continue
        x = (group["time"] - group["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        y = group["glucose"].to_numpy(dtype=float)
        fit = np.polyfit(x, y, 1)
        residual = y - np.polyval(fit, x)
        residual_pairs.extend(zip(residual[:-1], residual[1:]))
    if len(residual_pairs) < 20:
        return float("nan")
    pairs = np.asarray(residual_pairs, dtype=float)
    denominator = float(np.dot(pairs[:, 0], pairs[:, 0]))
    return float(np.dot(pairs[:, 0], pairs[:, 1]) / denominator) if denominator > 1e-12 else float("nan")


def compute_metric_row(record: SeriesRecord, grid_minutes: int = GRID_MINUTES, ema_alpha: Optional[float] = None) -> dict:
    source = sanitize_series(record)
    regular = regularize(source, grid_minutes, ema_alpha)
    values = regular["glucose"].to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    span_hours = float((source["time"].iloc[-1] - source["time"].iloc[0]).total_seconds() / 3600.0) if len(source) >= 2 else 0.0
    coverage = float(len(valid) / len(values)) if len(values) else 0.0
    row = {
        "role": record.role,
        "cohort": record.cohort,
        "subject_id": record.subject_id,
        "window": record.window,
        "split": record.split,
        "device": record.device,
        "grid_minutes": grid_minutes,
        "ema_alpha": ema_alpha if ema_alpha is not None else "",
        "source_points": len(source),
        "valid_points": len(valid),
        "span_hours": span_hours,
        "coverage": coverage,
        "source_max_gap_min": longest_missing_gap_minutes(source),
        "eligible_qc": bool(span_hours >= (18 if record.window == "24h" else 30) and coverage >= 0.60 and len(valid) >= 100),
    }
    for metric in METRIC_IDS:
        row[metric] = float("nan")
    if len(valid) < 20:
        row["event_count"] = 0
        return row

    mean = float(np.mean(valid))
    sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    row.update({
        "mean_glucose": mean,
        "night_mean": float(np.nanmean(values[(regular["time"].dt.hour.to_numpy() >= 0) & (regular["time"].dt.hour.to_numpy() < 6)])) if np.isfinite(values[(regular["time"].dt.hour.to_numpy() >= 0) & (regular["time"].dt.hour.to_numpy() < 6)]).any() else float("nan"),
        "sd_glucose": sd,
        "cv_pct": 100.0 * sd / mean if mean > 1e-12 else float("nan"),
        "tir_70_180": 100.0 * float(np.mean((valid >= 70 / 18.0) & (valid <= 180 / 18.0))),
        "tar_180": 100.0 * float(np.mean(valid > 180 / 18.0)),
        "tar_250": 100.0 * float(np.mean(valid > 250 / 18.0)),
        "tbr_70": 100.0 * float(np.mean(valid < 70 / 18.0)),
        "tbr_54": 100.0 * float(np.mean(valid < 54 / 18.0)),
        "mage": compute_mage(values),
        "modd": compute_modd(values, int(24 * 60 / grid_minutes)),
    })
    gmi_formula = 3.31 + 0.02392 * mean * 18.0
    gmi_eligible = span_hours >= 14 * 24 and coverage >= 0.70
    row["gmi"] = gmi_formula if gmi_eligible else float("nan")
    row["gmi_formula_unqualified"] = gmi_formula
    row["gmi_eligible"] = gmi_eligible

    time_hours = (regular["time"] - regular["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    row["linear_slope_day"] = ols_slope(time_hours, values, 24.0)
    daily_slopes, night_slopes = [], []
    local = regular.copy()
    local["date"] = local["time"].dt.date
    for _, group in local.groupby("date"):
        group_values = group["glucose"].to_numpy(dtype=float)
        if np.isfinite(group_values).mean() >= 0.70 and len(group) >= int(18 * 60 / grid_minutes):
            group_hours = (group["time"] - group["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            daily_slopes.append(ols_slope(group_hours, group_values, 24.0))
        night = group[(group["time"].dt.hour >= 0) & (group["time"].dt.hour < 6)]
        if len(night) and np.isfinite(night["glucose"]).mean() >= 0.70 and len(night) >= int(4 * 60 / grid_minutes):
            night_hours = (night["time"] - night["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            night_slopes.append(ols_slope(night_hours, night["glucose"].to_numpy(dtype=float), 1.0))
    row["median_daily_slope"] = float(np.nanmedian(daily_slopes)) if daily_slopes else float("nan")
    row["median_night_slope"] = float(np.nanmedian(night_slopes)) if night_slopes else float("nan")

    delta = np.diff(values)
    consecutive = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    rates = delta[consecutive] / (grid_minutes / 60.0)
    row["rate_mean_abs"] = float(np.mean(np.abs(rates))) if len(rates) else float("nan")
    row["rate_p95_up"] = percentile(rates[rates > 0], 0.95)
    row["rate_p95_down"] = percentile(-rates[rates < 0], 0.95)
    row.update(event_metrics(regular["time"], values, grid_minutes))
    row.update(phase_metrics(values, regular["time"], grid_minutes))
    row["night_ar1_detrended"] = detrended_night_ar1(regular["time"], values)
    return row


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"), "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def median_relative_difference(primary_values: Sequence[float], comparison_values: Sequence[float]) -> float:
    primary = pd.to_numeric(pd.Series(primary_values), errors="coerce").to_numpy(dtype=float)
    comparison = pd.to_numeric(pd.Series(comparison_values), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(primary) & np.isfinite(comparison)
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs(comparison[mask] - primary[mask]) / np.maximum(np.abs(primary[mask]), 1e-6)))


def icc_a1(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values).all(axis=1)
    values = values[mask]
    n, k = values.shape if values.ndim == 2 else (0, 0)
    if n < 3 or k < 2:
        return float("nan"), float("nan")
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ssr = k * float(np.sum((row_means - grand) ** 2))
    ssc = n * float(np.sum((col_means - grand) ** 2))
    residual = values - row_means[:, None] - col_means[None, :] + grand
    sse = float(np.sum(residual ** 2))
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    icc = (msr - mse) / denominator if abs(denominator) > 1e-12 else float("nan")
    return float(icc), float(mse)


def bootstrap_icc(values: np.ndarray, repeats: int = 300, seed: int = SEED) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = values[rng.integers(0, len(values), len(values))]
        estimate, _ = icc_a1(sample)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return (float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))) if len(estimates) >= repeats // 2 else (float("nan"), float("nan"))


def agreement_table(left: pd.DataFrame, right: pd.DataFrame, key_columns: List[str], context: str) -> pd.DataFrame:
    merged = left.merge(right, on=key_columns, suffixes=("_a", "_b"))
    rows = []
    for index, metric in enumerate(METRIC_IDS):
        a = pd.to_numeric(merged.get(f"{metric}_a"), errors="coerce")
        b = pd.to_numeric(merged.get(f"{metric}_b"), errors="coerce")
        mask = a.notna() & b.notna()
        pairs = np.column_stack([a[mask].to_numpy(float), b[mask].to_numpy(float)]) if mask.sum() else np.empty((0, 2))
        icc, mse = icc_a1(pairs)
        low, high = bootstrap_icc(pairs, seed=SEED + index)
        pooled_scale = float(np.nanmedian(np.abs(pairs))) if len(pairs) else float("nan")
        rows.append({
            "context": context,
            "metric_id": metric,
            "n_pairs": int(len(pairs)),
            "icc_a1": icc,
            "icc_ci_low": low,
            "icc_ci_high": high,
            "spearman": spearman(pairs[:, 0], pairs[:, 1]) if len(pairs) else float("nan"),
            "median_abs_difference": float(np.median(np.abs(pairs[:, 1] - pairs[:, 0]))) if len(pairs) else float("nan"),
            "median_relative_abs_difference": float(np.median(np.abs(pairs[:, 1] - pairs[:, 0]) / np.maximum(np.abs(pairs[:, 0]), 1e-6))) if len(pairs) else float("nan"),
            "mdc95": 1.96 * math.sqrt(2.0) * math.sqrt(max(mse, 0.0)) if math.isfinite(mse) else float("nan"),
            "mdc95_relative_to_pooled_median_abs": (1.96 * math.sqrt(2.0) * math.sqrt(max(mse, 0.0)) / pooled_scale) if math.isfinite(mse) and pooled_scale > 1e-9 else float("nan"),
        })
    return pd.DataFrame(rows)


def oof_linear_r2(frame: pd.DataFrame, metric: str) -> float:
    columns = list(SIMPLE_BASES) + [metric, "subject_id", "cohort"]
    local = frame[columns].apply(lambda col: pd.to_numeric(col, errors="coerce") if col.name not in {"subject_id", "cohort"} else col).dropna()
    if len(local) < 50 or local[metric].nunique() < 3:
        return float("nan")
    fold = local["subject_id"].astype(str).map(lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 5).to_numpy()
    prediction = np.full(len(local), np.nan)
    x_all = local[list(SIMPLE_BASES)].to_numpy(float)
    y_all = local[metric].to_numpy(float)
    for split in range(5):
        train, test = fold != split, fold == split
        if train.sum() < 20 or test.sum() == 0:
            continue
        center = np.median(x_all[train], axis=0)
        scale = np.quantile(x_all[train], 0.75, axis=0) - np.quantile(x_all[train], 0.25, axis=0)
        scale[scale < 1e-9] = 1.0
        x_train = np.column_stack([np.ones(train.sum()), (x_all[train] - center) / scale])
        x_test = np.column_stack([np.ones(test.sum()), (x_all[test] - center) / scale])
        beta = np.linalg.lstsq(x_train, y_all[train], rcond=None)[0]
        prediction[test] = x_test @ beta
    mask = np.isfinite(prediction)
    denominator = float(np.sum((y_all[mask] - np.mean(y_all[mask])) ** 2))
    return float(1.0 - np.sum((y_all[mask] - prediction[mask]) ** 2) / denominator) if mask.sum() >= 20 and denominator > 1e-12 else float("nan")


def redundancy_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_IDS:
        correlations = {base: spearman(primary[metric], primary[base]) for base in SIMPLE_BASES if base != metric}
        finite = [abs(value) for value in correlations.values() if math.isfinite(value)]
        rows.append({
            "metric_id": metric,
            **{f"spearman_vs_{base}": correlations.get(base, float("nan")) for base in SIMPLE_BASES},
            "max_abs_spearman_simple": max(finite) if finite else float("nan"),
            "oof_r2_from_mean_cv_tir": 1.0 if metric in SIMPLE_BASES else oof_linear_r2(primary, metric),
        })
    return pd.DataFrame(rows)


def synthetic_checks() -> Tuple[pd.DataFrame, dict]:
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=576, freq="5min")
    t_days = np.arange(576) * 5 / 1440.0
    rng = np.random.default_rng(SEED)
    signals = {
        "constant": np.full(576, 6.0),
        "linear": 5.0 + t_days,
        "sine": 6.0 + np.sin(2 * np.pi * np.arange(576) / 72.0),
        "step": np.where(np.arange(576) < 288, 5.5, 8.5),
        "white_noise": 6.0 + rng.normal(0, 0.4, 576),
    }
    ar = np.zeros(576)
    for i in range(1, 576):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 0.15)
    signals["ar09"] = 6.0 + ar
    excursion = np.full(576, 5.5)
    onset, peak = 120, 132
    excursion[onset:peak + 1] = np.linspace(5.5, 9.5, peak - onset + 1)
    for i in range(peak + 1, peak + 73):
        excursion[i] = 5.5 + 4.0 * math.exp(-(i - peak) * 5 / 45.0)
    signals["excursion"] = excursion
    rows = []
    for name, values in signals.items():
        record = SeriesRecord("synthetic", name, name, [value.isoformat() for value in times], values.tolist(), "48h")
        rows.append(compute_metric_row(record))
    frame = pd.DataFrame(rows)
    lookup = frame.set_index("cohort")
    checks = {
        "constant_mean": abs(lookup.loc["constant", "mean_glucose"] - 6.0) < 1e-9,
        "constant_sd": abs(lookup.loc["constant", "sd_glucose"]) < 1e-9,
        "constant_tir": abs(lookup.loc["constant", "tir_70_180"] - 100.0) < 1e-9,
        "linear_slope": abs(lookup.loc["linear", "linear_slope_day"] - 1.0) < 0.01,
        "linear_rate": abs(lookup.loc["linear", "rate_mean_abs"] - 1.0 / 24.0) < 0.002,
        "excursion_detected": lookup.loc["excursion", "event_count"] >= 1,
        "excursion_amplitude": abs(lookup.loc["excursion", "event_amplitude"] - 4.0) < 0.5,
        "ar1_ordering": lookup.loc["ar09", "night_ar1_detrended"] > lookup.loc["white_noise", "night_ar1_detrended"],
        "entropy_ordering": lookup.loc["white_noise", "permutation_entropy"] > lookup.loc["sine", "permutation_entropy"],
        "constant_phase_volume": (not math.isfinite(lookup.loc["constant", "log_volume"])) or lookup.loc["constant", "log_volume"] < -10,
    }
    return frame, checks


def phase_randomize(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    transform = np.fft.rfft(centered)
    phases = rng.uniform(0, 2 * math.pi, len(transform))
    phases[0] = 0.0
    if len(values) % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(np.abs(transform) * np.exp(1j * phases), n=len(values))
    return surrogate + np.mean(values)


def surrogate_analysis(primary_records: List[SeriesRecord], maximum_subjects: int = 60, repeats: int = 19) -> pd.DataFrame:
    selected = []
    per_cohort = defaultdict(int)
    for record in primary_records:
        if per_cohort[record.cohort] >= 8:
            continue
        selected.append(record)
        per_cohort[record.cohort] += 1
        if len(selected) >= maximum_subjects:
            break
    metric_names = ["lyapunov_rosenstein", "lyapunov_one_step", "rqa_det", "rqa_entr", "rqa_lam", "rqa_tt", "sample_entropy", "permutation_entropy"]
    differences = defaultdict(list)
    rng = np.random.default_rng(SEED + 9000)
    used = 0
    for record in selected:
        regular = regularize(sanitize_series(record), 5)
        values = regular["glucose"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        # Longest continuous segment, capped at 288 points, makes the surrogate
        # audit comparable and preserves the original power spectrum.
        runs = runs_of_true(finite, minimum=144)
        if not runs:
            continue
        padded = np.concatenate(([False], finite, [False])).astype(int)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
        choice = int(np.argmax(lengths))
        segment = values[starts[choice]:min(ends[choice], starts[choice] + 288)]
        segment_times = pd.Series(pd.date_range("2026-01-01", periods=len(segment), freq="5min"))
        actual = phase_metrics(segment, segment_times, 5)
        surrogate_metrics = defaultdict(list)
        for _ in range(repeats):
            randomized = phase_randomize(segment, rng)
            metrics = phase_metrics(randomized, segment_times, 5)
            for metric in metric_names:
                if math.isfinite(metrics.get(metric, float("nan"))):
                    surrogate_metrics[metric].append(metrics[metric])
        for metric in metric_names:
            if math.isfinite(actual.get(metric, float("nan"))) and surrogate_metrics[metric]:
                differences[metric].append(actual[metric] - float(np.median(surrogate_metrics[metric])))
        used += 1
    rows = []
    for metric in metric_names:
        values = np.asarray(differences[metric], dtype=float)
        if len(values) == 0:
            continue
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        majority = max(positive, negative)
        n_nonzero = positive + negative
        p_value = 2.0 * sum(math.comb(n_nonzero, k) for k in range(majority, n_nonzero + 1)) / (2 ** n_nonzero) if n_nonzero else float("nan")
        rows.append({
            "metric_id": metric,
            "n_subjects": len(values),
            "median_actual_minus_phase_surrogate": float(np.median(values)),
            "direction_consistency": majority / n_nonzero if n_nonzero else float("nan"),
            "two_sided_sign_test_p": min(1.0, p_value) if math.isfinite(p_value) else float("nan"),
            "phase_spectrum_preserved": True,
        })
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, float_format="%.10g")


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def summarize_metric_evidence(primary: pd.DataFrame, agreement: pd.DataFrame, sensitivity: pd.DataFrame, redundancy: pd.DataFrame, surrogate: pd.DataFrame, synthetic_pass: bool) -> pd.DataFrame:
    agreement_index = agreement.set_index(["context", "metric_id"])
    redundancy_index = redundancy.set_index("metric_id")
    surrogate_index = surrogate.set_index("metric_id") if len(surrogate) else pd.DataFrame()
    rows = []
    for metric in METRIC_IDS:
        compute_rate = float(pd.to_numeric(primary[metric], errors="coerce").notna().mean())
        rel = agreement_index.loc[("odd_even_days", metric)] if ("odd_even_days", metric) in agreement_index.index else None
        rel_hall = agreement_index.loc[("odd_even_days_hall", metric)] if ("odd_even_days_hall", metric) in agreement_index.index else None
        rel_weinstock = agreement_index.loc[("odd_even_days_weinstock", metric)] if ("odd_even_days_weinstock", metric) in agreement_index.index else None
        length = agreement_index.loc[("24h_vs_48h", metric)] if ("24h_vs_48h", metric) in agreement_index.index else None
        device = agreement_index.loc[("libre_vs_dexcom", metric)] if ("libre_vs_dexcom", metric) in agreement_index.index else None
        sens = sensitivity[sensitivity["metric_id"] == metric]
        sens_min = float(sens["spearman"].min()) if len(sens) and sens["spearman"].notna().any() else float("nan")
        rel_icc = float(rel["icc_a1"]) if rel is not None else float("nan")
        rel_low = float(rel["icc_ci_low"]) if rel is not None else float("nan")
        within_cohort_iccs = [float(item["icc_a1"]) for item in (rel_hall, rel_weinstock) if item is not None and math.isfinite(float(item["icc_a1"]))]
        within_cohort_lows = [float(item["icc_ci_low"]) for item in (rel_hall, rel_weinstock) if item is not None and math.isfinite(float(item["icc_ci_low"]))]
        minimum_within_cohort_icc = min(within_cohort_iccs) if len(within_cohort_iccs) == 2 else float("nan")
        minimum_within_cohort_ci_low = min(within_cohort_lows) if len(within_cohort_lows) == 2 else float("nan")
        device_icc = float(device["icc_a1"]) if device is not None else float("nan")
        quantitative_primary = compute_rate >= 0.90 and synthetic_pass
        robust = quantitative_primary and minimum_within_cohort_icc >= 0.75 and minimum_within_cohort_ci_low >= 0.60 and sens_min >= 0.75 and device_icc >= 0.60
        if robust:
            grade = "Q1_quantitative_robust"
        elif quantitative_primary and math.isfinite(minimum_within_cohort_icc) and minimum_within_cohort_icc >= 0.50 and math.isfinite(sens_min) and sens_min >= 0.70:
            grade = "Q2_quantitative_context_sensitive"
        elif compute_rate >= 0.50:
            grade = "D_descriptive_only"
        else:
            grade = "L_literature_or_insufficient_data"
        redundancy_row = redundancy_index.loc[metric]
        surrogate_row = surrogate_index.loc[metric] if len(surrogate_index) and metric in surrogate_index.index else None
        rows.append({
            "metric_id": metric,
            "label": METRIC_LOOKUP[metric]["label"],
            "group": METRIC_LOOKUP[metric]["group"],
            "unit": METRIC_LOOKUP[metric]["unit"],
            "primary_compute_rate": compute_rate,
            "odd_even_n": int(rel["n_pairs"]) if rel is not None else 0,
            "odd_even_icc_a1": rel_icc,
            "odd_even_icc_ci_low": rel_low,
            "odd_even_spearman": float(rel["spearman"]) if rel is not None else float("nan"),
            "odd_even_mdc95": float(rel["mdc95"]) if rel is not None else float("nan"),
            "hall_odd_even_icc_a1": float(rel_hall["icc_a1"]) if rel_hall is not None else float("nan"),
            "hall_odd_even_icc_ci_low": float(rel_hall["icc_ci_low"]) if rel_hall is not None else float("nan"),
            "weinstock_odd_even_icc_a1": float(rel_weinstock["icc_a1"]) if rel_weinstock is not None else float("nan"),
            "weinstock_odd_even_icc_ci_low": float(rel_weinstock["icc_ci_low"]) if rel_weinstock is not None else float("nan"),
            "minimum_within_cohort_icc_a1": minimum_within_cohort_icc,
            "minimum_within_cohort_icc_ci_low": minimum_within_cohort_ci_low,
            "length_n": int(length["n_pairs"]) if length is not None else 0,
            "length_icc_a1": float(length["icc_a1"]) if length is not None else float("nan"),
            "length_spearman": float(length["spearman"]) if length is not None else float("nan"),
            "device_n": int(device["n_pairs"]) if device is not None else 0,
            "device_icc_a1": device_icc,
            "device_spearman": float(device["spearman"]) if device is not None else float("nan"),
            "minimum_sensitivity_spearman": sens_min,
            "max_abs_spearman_simple": float(redundancy_row["max_abs_spearman_simple"]),
            "oof_r2_from_mean_cv_tir": float(redundancy_row["oof_r2_from_mean_cv_tir"]),
            "phase_surrogate_direction_consistency": float(surrogate_row["direction_consistency"]) if surrogate_row is not None else float("nan"),
            "phase_surrogate_sign_p": float(surrogate_row["two_sided_sign_test_p"]) if surrogate_row is not None else float("nan"),
            "data_grade": grade,
            "interpretation_boundary": METRIC_LOOKUP[metric]["boundary"],
        })
    return pd.DataFrame(rows)


def write_metric_definitions() -> None:
    with (RESEARCH_OUTPUT / "metric_definitions.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "schema": "glucobench.cgm-metric-research.v1",
            "analysis_date": "2026-08-12",
            "label_blinding": {"allowed_fields": ["cohort", "id", "timestamps", "values", "window", "split", "device"], "clinical_fields_loaded": False},
            "primary_grid_minutes": GRID_MINUTES,
            "maximum_interpolation_gap_minutes": MAX_INTERPOLATION_GAP_MINUTES,
            "computational_contract": {
                "units": "mmol/L unless metric unit states otherwise",
                "resampling": "5-minute grid",
                "interpolation": "linear interpolation only across gaps <=15 minutes",
                "takens_embedding": "m=3; first ACF below 1/e capped at 60 minutes",
                "rqa": "Euclidean recurrence threshold fixed to 2% recurrence rate",
                "event_semantics": "signal-defined free-living excursions; no meal labels",
            },
            "metrics": [dict(item, operational_definition=OPERATIONAL_DEFINITIONS[item["id"]]) for item in METRIC_DEFINITIONS],
        }, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller sensitivity/surrogate samples for a smoke test")
    parser.add_argument("--definitions-only", action="store_true", help="Refresh metric_definitions.json without reading analysis data")
    args = parser.parse_args()
    RESEARCH_OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.definitions_only:
        write_metric_definitions()
        print("wrote", RESEARCH_OUTPUT / "metric_definitions.json", flush=True)
        return
    primary_records, length24_records, reliability_records, device_records, long_records = load_records()

    if args.quick:
        primary_records = primary_records[:20]
        length24_records = length24_records[:20]
        reliability_records = [row for row in reliability_records if row.cohort == "hall"][:20]
        paired_ids = {row.subject_id for row in device_records if row.device == "libre"}
        paired_ids = set(sorted(paired_ids)[:8])
        device_records = [row for row in device_records if row.subject_id in paired_ids]

    print(f"primary={len(primary_records)} length24={len(length24_records)} reliability={len(reliability_records)} device={len(device_records)}")
    primary_rows = []
    for index, record in enumerate(primary_records, 1):
        primary_rows.append(compute_metric_row(record))
        if index % 50 == 0 or index == len(primary_records):
            print(f"primary metrics {index}/{len(primary_records)}", flush=True)
    primary = pd.DataFrame(primary_rows)

    length_rows = []
    for index, record in enumerate(length24_records, 1):
        length_rows.append(compute_metric_row(record))
        if index % 75 == 0 or index == len(length24_records):
            print(f"24h metrics {index}/{len(length24_records)}", flush=True)
    length24 = pd.DataFrame(length_rows)

    reliability_rows = []
    for index, record in enumerate(reliability_records, 1):
        reliability_rows.append(compute_metric_row(record))
        if index % 75 == 0 or index == len(reliability_records):
            print(f"reliability metrics {index}/{len(reliability_records)}", flush=True)
    reliability = pd.DataFrame(reliability_rows)

    device_rows = []
    # Libre rows already exist in primary; recompute both devices to keep this
    # table self-contained and preserve exact paired extraction.
    for index, record in enumerate(device_records, 1):
        device_rows.append(compute_metric_row(record))
        if index % 44 == 0 or index == len(device_records):
            print(f"device metrics {index}/{len(device_records)}", flush=True)
    device = pd.DataFrame(device_rows)

    sensitivity_records = primary_records[:40] if args.quick else []
    if not args.quick:
        counts = defaultdict(int)
        for record in primary_records:
            if counts[record.cohort] < 24:
                sensitivity_records.append(record)
                counts[record.cohort] += 1
    sensitivity_profiles = []
    for index, record in enumerate(sensitivity_records, 1):
        base = compute_metric_row(record, 5)
        for profile_name, grid, alpha in (("grid10", 10, None), ("grid15", 15, None), ("ema03", 5, 0.3)):
            comparison = compute_metric_row(record, grid, alpha)
            for metric in METRIC_IDS:
                sensitivity_profiles.append({
                    "cohort": record.cohort,
                    "subject_id": record.subject_id,
                    "profile": profile_name,
                    "metric_id": metric,
                    "primary_value": base.get(metric),
                    "comparison_value": comparison.get(metric),
                })
        if index % 25 == 0 or index == len(sensitivity_records):
            print(f"sensitivity profiles {index}/{len(sensitivity_records)}", flush=True)
    sensitivity_long = pd.DataFrame(sensitivity_profiles)
    sensitivity_summary = sensitivity_long.groupby(["profile", "metric_id"], as_index=False).apply(
        lambda group: pd.Series({
            "n_pairs": int((pd.to_numeric(group["primary_value"], errors="coerce").notna() & pd.to_numeric(group["comparison_value"], errors="coerce").notna()).sum()),
            "spearman": spearman(group["primary_value"], group["comparison_value"]),
            "median_relative_abs_difference": median_relative_difference(group["primary_value"], group["comparison_value"]),
        }), include_groups=False
    ).reset_index(drop=True)

    # Agreement contexts.
    rel_odd = reliability[reliability["split"] == "odd"]
    rel_even = reliability[reliability["split"] == "even"]
    agreement_rel = agreement_table(rel_odd, rel_even, ["cohort", "subject_id"], "odd_even_days")
    agreement_rel_hall = agreement_table(rel_odd[rel_odd["cohort"] == "hall"], rel_even[rel_even["cohort"] == "hall"], ["cohort", "subject_id"], "odd_even_days_hall")
    agreement_rel_weinstock = agreement_table(rel_odd[rel_odd["cohort"] == "weinstock"], rel_even[rel_even["cohort"] == "weinstock"], ["cohort", "subject_id"], "odd_even_days_weinstock")
    primary_for_length = primary[primary["cohort"].isin([key.removesuffix("_w24") for key in LENGTH_24_KEYS])]
    agreement_length = agreement_table(length24, primary_for_length, ["cohort", "subject_id"], "24h_vs_48h")
    agreement_device = agreement_table(device[device["device"] == "libre"], device[device["device"] == "dexcom"], ["cohort", "subject_id"], "libre_vs_dexcom")
    agreement = pd.concat([agreement_rel, agreement_rel_hall, agreement_rel_weinstock, agreement_length, agreement_device], ignore_index=True)

    redundancy = redundancy_table(primary)
    synthetic_frame, checks = synthetic_checks()
    print("synthetic checks", checks, flush=True)
    surrogate = surrogate_analysis(primary_records, maximum_subjects=12 if args.quick else 60, repeats=5 if args.quick else 19)
    evidence = summarize_metric_evidence(primary, agreement, sensitivity_summary, redundancy, surrogate, all(checks.values()))

    # GMI qualification audit uses raw point density to avoid allocating very
    # large grids for records with calendar-scale gaps.
    gmi_rows = []
    for record in long_records:
        source = sanitize_series(record)
        if len(source) < 2:
            continue
        span_days = (source["time"].iloc[-1] - source["time"].iloc[0]).total_seconds() / 86400.0
        expected = span_days * 288.0
        approximate_coverage = len(source) / expected if expected > 0 else 0.0
        gmi_rows.append({"cohort": record.cohort, "subject_id": record.subject_id, "span_days": span_days, "raw_points": len(source), "approximate_5min_coverage": approximate_coverage, "gmi_eligible_14d_70pct": bool(span_days >= 14 and approximate_coverage >= 0.70)})
    gmi_audit = pd.DataFrame(gmi_rows)

    metrics_long = pd.concat([primary, length24, reliability, device], ignore_index=True)
    quality_summary = metrics_long.groupby(["role", "cohort"], as_index=False).agg(
        records=("subject_id", "count"),
        eligible_records=("eligible_qc", "sum"),
        median_span_hours=("span_hours", "median"),
        median_coverage=("coverage", "median"),
        median_source_points=("source_points", "median"),
    )
    metric_distribution_rows = []
    for cohort, group in primary.groupby("cohort"):
        for metric in METRIC_IDS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            metric_distribution_rows.append({
                "cohort": cohort,
                "metric_id": metric,
                "n": len(values),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "q25": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
                "q75": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
                "minimum": float(np.min(values)) if len(values) else float("nan"),
                "maximum": float(np.max(values)) if len(values) else float("nan"),
            })
    distributions = pd.DataFrame(metric_distribution_rows)

    write_metric_definitions()
    write_csv(metrics_long, RESEARCH_OUTPUT / "metrics_long.csv")
    write_csv(quality_summary, RESEARCH_OUTPUT / "quality_summary.csv")
    write_csv(distributions, RESEARCH_OUTPUT / "metric_distributions.csv")
    write_csv(agreement, RESEARCH_OUTPUT / "agreement_summary.csv")
    write_csv(sensitivity_summary, RESEARCH_OUTPUT / "sensitivity_summary.csv")
    write_csv(redundancy, RESEARCH_OUTPUT / "redundancy_summary.csv")
    write_csv(surrogate, RESEARCH_OUTPUT / "surrogate_summary.csv")
    write_csv(evidence, RESEARCH_OUTPUT / "metric_evidence_summary.csv")
    write_csv(gmi_audit, RESEARCH_OUTPUT / "gmi_qualification_audit.csv")
    write_csv(synthetic_frame, RESEARCH_OUTPUT / "synthetic_metrics.csv")
    summary = {
        "schema": "glucobench.cgm-metric-research.summary.v1",
        "analysis_date": "2026-08-12",
        "label_blind": True,
        "index_html_modified": False,
        "raw_data_modified": False,
        "record_counts": {
            "primary48": len(primary),
            "length24": len(length24),
            "reliability_windows": len(reliability),
            "device_windows": len(device),
            "sensitivity_subjects": len(sensitivity_records),
        },
        "quality": quality_summary.to_dict(orient="records"),
        "synthetic_checks": checks,
        "data_grade_counts": evidence["data_grade"].value_counts().to_dict(),
        "gmi_eligible_records": int(gmi_audit["gmi_eligible_14d_70pct"].sum()) if len(gmi_audit) else 0,
    }
    with (RESEARCH_OUTPUT / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(summary["record_counts"], ensure_ascii=False), flush=True)
    print("wrote", RESEARCH_OUTPUT, flush=True)


if __name__ == "__main__":
    main()
