#!/usr/bin/env node
/** Regression-test v8.4.2 auditable text/JSON output without a browser. */

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await fs.readFile(path.join(root, "index.html"), "utf8");
const inline = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(match => match[1]).filter(text => text.trim()).at(-1);
assert.ok(inline, "inline application script must exist");

function makeClassList(initial = []) {
  const values = new Set(initial);
  return {
    add(value) { values.add(value); },
    remove(value) { values.delete(value); },
    contains(value) { return values.has(value); },
  };
}

function makeMetricCard(id) {
  const value = { textContent: "", style: {} };
  const description = { textContent: "" };
  return {
    id,
    className: "metric-card",
    innerHTML: "",
    querySelector(selector) {
      if (selector === ".metric-value") return value;
      if (selector === ".metric-desc") return description;
      return null;
    },
  };
}

const cardIds = [
  "metric-volume", "metric-velocity", "metric-shape", "metric-core",
  "metric-dimension", "metric-lyapunov", "metric-det", "metric-entr",
  "metric-insulin-pred",
];
const elements = Object.fromEntries(cardIds.map(id => [id, makeMetricCard(id)]));
elements["sel-treatment"] = { value: "treated" };
elements["health-rating"] = { className: "", innerHTML: "", title: "" };
elements["btn-copy-oracle"] = { style: {} };
elements["report-modal"] = {
  classList: makeClassList(), attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
};
elements["report-preview"] = { textContent: "" };
elements["report-status"] = { textContent: "", className: "" };
elements["btn-close-report"] = { focusCalled: false, focus() { this.focusCalled = true; } };

let copiedText = null;
let downloaded = null;
let revokedHref = null;
const returnFocus = { focusCalled: false, focus() { this.focusCalled = true; } };
const bodyClassList = makeClassList();

const documentMock = {
  addEventListener() {},
  getElementById(id) { return elements[id] || null; },
  activeElement: returnFocus,
  body: {
    classList: bodyClassList,
    appendChild(node) { return node; },
  },
  createElement(tag) {
    assert.equal(tag, "a", "clipboard API should avoid textarea fallback in this test");
    return {
      href: "", download: "", removed: false,
      click() { downloaded = { href: this.href, filename: this.download }; },
      remove() { this.removed = true; },
    };
  },
};
class BlobMock {
  constructor(parts, options) { this.parts = parts; this.type = options.type; BlobMock.last = this; }
}
const context = vm.createContext({
  console: { log() {}, warn() {}, error: console.error },
  document: documentMock,
  navigator: { clipboard: { async writeText(value) { copiedText = value; } } },
  Blob: BlobMock,
  URL: {
    createObjectURL() { return "blob:audit-test"; },
    revokeObjectURL(value) { revokedHref = value; },
  },
  window: {}, setTimeout, clearTimeout,
});
vm.runInContext(`${inline}\nglobalThis.__logTest = { UI };`, context, { filename: "index-inline.js" });
const UI = context.__logTest.UI;
UI.rawData = {
  collapsed: true,
  timestamps: [
    new Date("2026-07-01T00:00:00+08:00"),
    new Date("2026-07-03T23:57:00+08:00"),
  ],
  values: Array.from({ length: 960 }, (_, index) => 4.8 + (index % 50) / 100),
};
UI._screenDynamics = { lyapunov: 0.19, det: 0.991, entr: 2.88 };

const metric = {
  volume: 12,
  avgRecovery: 0.12,
  shapeRatio: 1.5,
  dimension: 1.91,
  lyapunov: 0.19,
  det: 0.991,
  entr: 2.88,
  rr: 0.02,
  effectiveDim: 2,
  validPoints: Array.from({ length: 150 }, () => [5, 5, 5]),
};
const insulin = { predicted: 9.4, confidence: "high", nScales: 3 };
UI.renderMetrics(
  metric, 0.8, 6, [3.7, 4.9, 5.3, 8.9, 11.4], true, insulin,
  1.5, 4, false, 4, 5.8,
  { period: "daytime", probeN: 240, nightPointCount: 320 },
);

const treatedJson = structuredClone(UI.lastReportJSON);
const treatedText = UI.lastReport;
assert.equal(treatedJson.schema_version, "glucobench.audit-log.v1");
assert.equal(treatedJson.app_version, "8.4.2");
assert.equal(treatedJson.screening_version, "8.4");
assert.equal(treatedJson.screening.treatment_context, "正在使用降糖药");
assert.notEqual(treatedJson.screening.inputs.dynamic_score, null);
assert.ok(treatedJson.screening.model_validation);
assert.match(treatedJson.screening.score_semantics, /不是患病概率/);
assert.equal(treatedJson.data_quality.metric_view, "日间 06:00–18:00");
assert.equal(treatedJson.data_quality.raw_records, 960);
assert.equal(treatedJson.data_quality.night_valid_values, 320);
assert.match(treatedJson.data_quality.observed_start, /^2026-07-0[12]/);
assert.match(treatedJson.data_quality.observed_end, /^2026-07-0[34]/);
assert.equal(treatedJson.data_quality.calendar_days, 2);
assert.match(treatedJson.data_quality.screening_scope, /不改变顶部筛查分值/);
assert.equal(treatedJson.metrics.length, 9);
assert.equal(treatedJson.metrics.at(-1).status, "low_validity");
assert.match(treatedJson.metrics.at(-1).interpretation, /R²≈0\.01/);
assert.equal(treatedJson.privacy.raw_glucose_series_included, false);
assert.equal(treatedJson.privacy.local_file_path_included, false);
assert.equal(treatedJson.privacy.local_file_name_included, false);
assert.ok(treatedJson.evidence_limits.some(item => /0\.763→0\.763/.test(item)));
assert.ok(treatedJson.evidence_limits.some(item => /Hall ρ=0\.298/.test(item)));
assert.equal(treatedJson.references.length, 2);
assert.ok(treatedJson.range_summary.flags.length >= 2);

for (const section of [
  "## 1. 筛查输出", "## 2. 数据质量与分析语境", "## 3. 解释性结构指标",
  "## 5. 证据边界", "## 6. 后续核实", "## 7. 可复现性与隐私",
]) {
  assert.match(treatedText, new RegExp(section.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
}
assert.match(treatedText, /模型分值不是患病概率/);
assert.match(treatedText, /不含原始血糖序列，不含本地文件路径/);
assert.doesNotMatch(treatedText, /Janus|宿主|吞噬|高维情报|暴力平账|安全结界|糖毒性极值/);

UI.openReportModal();
assert.equal(elements["report-modal"].attributes["aria-hidden"], "false");
assert.equal(elements["report-modal"].classList.contains("open"), true);
assert.equal(bodyClassList.contains("report-open"), true);
assert.equal(elements["report-preview"].textContent, treatedText);
assert.equal(elements["btn-close-report"].focusCalled, true);
await UI.copyReportText();
assert.equal(copiedText, treatedText);
assert.equal(elements["report-status"].textContent, "文本已复制");
UI.downloadReportJSON();
assert.match(downloaded.filename, /^glucobench_audit_.*\.json$/);
assert.equal(downloaded.href, "blob:audit-test");
assert.equal(revokedHref, "blob:audit-test");
assert.equal(JSON.parse(BlobMock.last.parts[0]).schema_version, "glucobench.audit-log.v1");
UI.closeReportModal();
assert.equal(elements["report-modal"].attributes["aria-hidden"], "true");
assert.equal(elements["report-modal"].classList.contains("open"), false);
assert.equal(bodyClassList.contains("report-open"), false);
assert.equal(returnFocus.focusCalled, true);

elements["sel-treatment"].value = "unknown";
UI.renderMetrics(
  metric, 0, 6, [4.8, 5.0, 5.3], true, insulin, 1.5, 4, false, 4, 5.8,
  { period: "night", probeN: 220, nightPointCount: 320 },
);
const fallbackJson = UI.lastReportJSON;
assert.equal(fallbackJson.screening.inputs.dynamic_score, null);
assert.equal(fallbackJson.screening.model_validation, null);
assert.match(fallbackJson.screening.model_path, /夜间均糖回退路径/);
assert.equal(fallbackJson.data_quality.metric_view, "夜间 00:00–06:00");

console.log(JSON.stringify({
  status: "PASS",
  schema: treatedJson.schema_version,
  treated_model: treatedJson.screening.model_path,
  fallback_model: fallbackJson.screening.model_path,
  metric_count: treatedJson.metrics.length,
  privacy: treatedJson.privacy,
}, null, 2));
