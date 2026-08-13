#!/usr/bin/env python3
"""Render dependency-free SVG diagnostics for the label-blind CGM metric audit."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "cgm_metric_research"


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def save_svg(name: str, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        '<style>text{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;fill:#17212b}'
        '.title{font-size:20px;font-weight:700}.sub{font-size:12px;fill:#52606d}'
        '.label{font-size:11px}.axis{stroke:#9aa5b1;stroke-width:1}.grid{stroke:#d9e2ec;stroke-width:1}'
        '.note{font-size:11px;fill:#52606d}</style>'
        f'{body}</svg>'
    )
    (DATA_DIR / name).write_text(svg, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> None:
    tracked = sorted(path for path in DATA_DIR.iterdir() if path.is_file() and path.name != "analysis_manifest.json")
    code_files = [
        ROOT / "validate" / "analyze_cgm_raw_metrics.py",
        ROOT / "validate" / "render_cgm_metric_research_charts.py",
    ]
    manifest = {
        "schema": "glucobench.cgm-metric-research.manifest.v1",
        "analysis_date": "2026-08-12",
        "label_blind": True,
        "index_html_modified_by_research": False,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in tracked
        },
        "code": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in code_files
        },
        "index_html_sha256": sha256(ROOT / "index.html"),
    }
    (DATA_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def reliability_chart(evidence: list[dict[str, str]]) -> None:
    rows = sorted(
        evidence,
        key=lambda row: number(row["minimum_within_cohort_icc_a1"]) or -2.0,
        reverse=True,
    )[:18]
    width, left, top, row_h = 960, 265, 86, 25
    plot_w = 620
    height = top + row_h * len(rows) + 70
    pieces = [
        '<text x="30" y="35" class="title">队列内独立日窗重复性：前 18 项</text>',
        '<text x="30" y="58" class="sub">ICC(A,1)；分开计算 Hall 与 Weinstock，虚线为预冻结 Q1 阈值 0.75</text>',
    ]
    for tick in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + (tick + 0.25) / 1.25 * plot_w
        pieces.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-45}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height-27}" text-anchor="middle" class="label">{tick:.2f}</text>')
    threshold_x = left + (0.75 + 0.25) / 1.25 * plot_w
    pieces.append(f'<line x1="{threshold_x:.1f}" y1="{top-10}" x2="{threshold_x:.1f}" y2="{height-45}" stroke="#c2410c" stroke-dasharray="5 4"/>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        label = html.escape(row["label"])
        hall = number(row["hall_odd_even_icc_a1"])
        wein = number(row["weinstock_odd_even_icc_a1"])
        pieces.append(f'<text x="250" y="{y+8}" text-anchor="end" class="label">{label}</text>')
        for value, color, offset in ((hall, "#2563eb", -4), (wein, "#f59e0b", 4)):
            if value is None:
                continue
            x0 = left + 0.25 / 1.25 * plot_w
            x1 = left + (value + 0.25) / 1.25 * plot_w
            pieces.append(f'<line x1="{x0:.1f}" y1="{y+offset:.1f}" x2="{x1:.1f}" y2="{y+offset:.1f}" stroke="{color}" stroke-width="5"/>')
            pieces.append(f'<circle cx="{x1:.1f}" cy="{y+offset:.1f}" r="3" fill="{color}"/>')
    pieces.extend([
        f'<rect x="{left}" y="{height-18}" width="11" height="6" fill="#2563eb"/><text x="{left+17}" y="{height-11}" class="note">Hall</text>',
        f'<rect x="{left+78}" y="{height-18}" width="11" height="6" fill="#f59e0b"/><text x="{left+95}" y="{height-11}" class="note">Weinstock</text>',
    ])
    save_svg("reliability_icc.svg", width, height, "".join(pieces))


def context_chart(evidence: list[dict[str, str]]) -> None:
    wanted = [
        "mean_glucose", "cv_pct", "rate_mean_abs", "event_iauc_0_180",
        "log_volume", "lyapunov_rosenstein", "permutation_entropy",
        "night_ar1_detrended", "work_integral", "night_friction",
    ]
    lookup = {row["metric_id"]: row for row in evidence}
    rows = [lookup[key] for key in wanted]
    width, left, top, row_h, plot_w = 980, 260, 84, 30, 650
    height = top + row_h * len(rows) + 70
    colors = ("#0f766e", "#7c3aed", "#dc2626")
    pieces = [
        '<text x="30" y="35" class="title">重复性具有强烈的观察语境依赖</text>',
        '<text x="30" y="58" class="sub">同队列较弱 ICC、24h/48h 一致性、双设备一致性；负值从零向左显示</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_w
        pieces.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{height-45}" class="grid"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height-27}" text-anchor="middle" class="label">{tick:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        pieces.append(f'<text x="245" y="{y+7}" text-anchor="end" class="label">{html.escape(row["label"])}</text>')
        values = (
            number(row["minimum_within_cohort_icc_a1"]),
            number(row["length_icc_a1"]),
            number(row["device_icc_a1"]),
        )
        for j, (value, color) in enumerate(zip(values, colors)):
            if value is None:
                continue
            clipped = max(0.0, min(1.0, value))
            pieces.append(f'<rect x="{left}" y="{y-8+j*7}" width="{clipped*plot_w:.1f}" height="5" fill="{color}"/>')
    legend = (("队列内较弱 ICC", colors[0]), ("24h/48h ICC", colors[1]), ("双设备 ICC", colors[2]))
    for index, (label, color) in enumerate(legend):
        x = left + index * 155
        pieces.append(f'<rect x="{x}" y="{height-17}" width="11" height="6" fill="{color}"/><text x="{x+17}" y="{height-10}" class="note">{label}</text>')
    save_svg("context_agreement.svg", width, height, "".join(pieces))


def surrogate_chart(rows: list[dict[str, str]], evidence: list[dict[str, str]]) -> None:
    width, left, top, row_h, plot_w = 900, 250, 84, 34, 550
    labels = {row["metric_id"]: row["label"] for row in evidence}
    rows = sorted(rows, key=lambda row: abs(number(row["median_actual_minus_phase_surrogate"]) or 0), reverse=True)
    values = [number(row["median_actual_minus_phase_surrogate"]) or 0 for row in rows]
    extent = max(abs(min(values)), abs(max(values)), 0.01)
    height = top + row_h * len(rows) + 65
    center = left + plot_w / 2
    pieces = [
        '<text x="30" y="35" class="title">相位随机替代序列检验</text>',
        '<text x="30" y="58" class="sub">中位数：真实指标 − 保持功率谱的替代序列；非零说明存在超出线性频谱的时间结构</text>',
        f'<line x1="{center:.1f}" y1="{top-12}" x2="{center:.1f}" y2="{height-40}" class="axis"/>',
    ]
    for index, (row, value) in enumerate(zip(rows, values)):
        y = top + index * row_h
        x = center + value / (2 * extent) * plot_w
        color = "#047857" if value >= 0 else "#b91c1c"
        x0, bar_w = min(center, x), max(1.0, abs(x - center))
        label = labels.get(row["metric_id"], row["metric_id"])
        pieces.append(f'<text x="235" y="{y+5}" text-anchor="end" class="label">{html.escape(label)}</text>')
        pieces.append(f'<rect x="{x0:.1f}" y="{y-8}" width="{bar_w:.1f}" height="14" fill="{color}" opacity="0.82"/>')
        pieces.append(f'<text x="{x + (6 if value >= 0 else -6):.1f}" y="{y+3}" text-anchor="{("start" if value >= 0 else "end")}" class="label">{value:.3f}</text>')
    pieces.append('<text x="30" y="{}" class="note">结构差异不等于健康方向，也不自动产生个体级可靠指标。</text>'.format(height-15))
    save_svg("surrogate_residual.svg", width, height, "".join(pieces))


def main() -> None:
    evidence = load_csv("metric_evidence_summary.csv")
    reliability_chart(evidence)
    context_chart(evidence)
    surrogate_chart(load_csv("surrogate_summary.csv"), evidence)
    write_manifest()
    print("wrote SVG diagnostics and analysis_manifest.json")


if __name__ == "__main__":
    main()
