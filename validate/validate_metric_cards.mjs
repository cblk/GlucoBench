#!/usr/bin/env node
/** Regression-test v8.4.3 metric-card wording, value bands, and stratified meaning. */

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
    value,
    description,
  };
}

const cardIds = [
  "metric-volume", "metric-velocity", "metric-shape", "metric-core",
  "metric-dimension", "metric-lyapunov", "metric-det", "metric-entr",
  "metric-insulin-pred",
];
const elements = Object.fromEntries(cardIds.map(id => [id, makeMetricCard(id)]));
elements["sel-treatment"] = { value: "unknown" };
elements["health-rating"] = { className: "", innerHTML: "", title: "" };
elements["btn-copy-oracle"] = { style: {} };

const documentMock = {
  addEventListener() {},
  getElementById(id) { return elements[id] || null; },
};
const context = vm.createContext({
  console: { log() {}, warn() {}, error: console.error },
  document: documentMock,
  window: {}, setTimeout, clearTimeout,
});
vm.runInContext(`${inline}\nglobalThis.__cardTest = { UI };`, context, { filename: "index-inline.js" });
const UI = context.__cardTest.UI;
UI.rawData = { collapsed: false };
UI._screenDynamics = null;

function resetCards() {
  for (const id of cardIds) {
    elements[id] = makeMetricCard(id);
  }
}

function renderCase(name, values, treatment = "unknown", displacement = values.displacement) {
  resetCards();
  elements["sel-treatment"].value = treatment;
  const metric = {
    volume: values.volume,
    avgRecovery: values.recovery,
    shapeRatio: values.shape,
    dimension: values.dimension,
    lyapunov: values.lyapunov,
    det: values.det,
    entr: values.entr,
    rr: 0.02,
    effectiveDim: 2,
    validPoints: Array.from({ length: 120 }, () => [5, 5, 5]),
  };
  const insulin = { predicted: 9.4, confidence: "high", nScales: 3 };
  UI.renderMetrics(metric, displacement, 8, [4.8, 5.1, 6.2, 8.4], true, insulin, 1.5, 4, false, 4, 5.6);
  const cards = {};
  for (const id of cardIds.slice(0, -1)) {
    cards[id] = {
      className: elements[id].className,
      value: elements[id].value.textContent,
      description: elements[id].description.textContent,
    };
  }
  cards["metric-insulin-pred"] = {
    className: elements["metric-insulin-pred"].className,
    html: elements["metric-insulin-pred"].innerHTML,
  };
  return { name, cards };
}

const reference = renderCase("reference", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: 0.3,
  dimension: 1.90, lyapunov: 0.17, det: 0.98, entr: 2.5,
});
const shifted = renderCase("shifted", {
  volume: 10, recovery: 0.12, shape: 1.5, displacement: 1.0,
  dimension: 1.805, lyapunov: 0.13, det: 0.991, entr: 2.9,
});
const extreme = renderCase("extreme", {
  volume: 25, recovery: 0.08, shape: 1.1, displacement: 2.0,
  dimension: 2.03, lyapunov: 0.08, det: 0.995, entr: 3.2,
});
const unavailableCore = renderCase("unavailable-core", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: null,
  dimension: 1.90, lyapunov: 0.17, det: 0.98, entr: 2.5,
}, "unknown", null);
const treated = renderCase("treated-context", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: 0.3,
  dimension: 1.90, lyapunov: 0.12, det: 0.995, entr: 2.9,
}, "treated");
const untreated = renderCase("untreated-context", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: 0.3,
  dimension: 1.90, lyapunov: 0.17, det: 0.98, entr: 2.5,
}, "untreated");
const treatedRiskDirection = renderCase("treated-risk-direction", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: 0.3,
  dimension: 1.90, lyapunov: 0.25, det: 0.988, entr: 2.5,
}, "treated");
const untreatedRiskDirection = renderCase("untreated-risk-direction", {
  volume: 5, recovery: 0.16, shape: 2.1, displacement: 0.3,
  dimension: 1.90, lyapunov: 0.10, det: 0.995, entr: 3.0,
}, "untreated");

const bandCards = ["metric-volume", "metric-velocity", "metric-core", "metric-dimension", "metric-lyapunov", "metric-det", "metric-entr"];
for (const id of bandCards) {
  assert.match(reference.cards[id].className, /reference/, `${id} reference band`);
  assert.match(shifted.cards[id].className, /shifted/, `${id} shifted band`);
  assert.match(extreme.cards[id].className, /extreme/, `${id} extreme band`);
}
assert.match(reference.cards["metric-shape"].description, /明显拉伸（≥1\.8）/);
assert.match(shifted.cards["metric-shape"].description, /轻度拉伸（1\.2–1\.8）/);
assert.match(extreme.cards["metric-shape"].description, /近各向同性（<1\.2）/);
assert.match(unavailableCore.cards["metric-core"].className, /unavailable/);
assert.equal(unavailableCore.cards["metric-core"].value, "N/A");
assert.match(reference.cards["metric-core"].value, /mmol\/L$/);
assert.match(reference.cards["metric-insulin-pred"].className, /unavailable/);
assert.match(reference.cards["metric-insulin-pred"].html, /R²≈0\.01/);
assert.match(reference.cards["metric-insulin-pred"].html, /几乎无个体预测力/);

for (const id of ["metric-lyapunov", "metric-det", "metric-entr"]) {
  assert.match(treated.cards[id].description, /治疗中内部健康参考/);
  assert.match(treated.cards[id].description, /风险关联方向/);
  assert.match(untreated.cards[id].description, /未治疗内部健康参考/);
  assert.match(untreated.cards[id].description, /风险关联方向/);
  assert.match(treatedRiskDirection.cards[id].className, /shifted|extreme/);
  assert.match(treatedRiskDirection.cards[id].description, /与异常标签同向/);
  assert.match(untreatedRiskDirection.cards[id].className, /shifted|extreme/);
  assert.match(untreatedRiskDirection.cards[id].description, /与异常标签同向/);
  assert.match(reference.cards[id].description, /治疗未知，不作健康方向判断|治疗未知时不判断健康方向/);
  assert.match(reference.cards[id].description, /不进入.*共识/);
}

const prohibitedClaims = /系统发散|屏障脆弱|重度抵抗|因果律健康|策略耗散|代谢从容|系统性透支/;
for (const result of [reference, shifted, extreme, unavailableCore, treated, untreated, treatedRiskDirection, untreatedRiskDirection]) {
  for (const card of Object.values(result.cards)) {
    const text = card.description || card.html || "";
    assert.doesNotMatch(text, prohibitedClaims, `${result.name} must not contain an overclaim`);
  }
}

console.log(JSON.stringify({
  status: "PASS",
  cases: [reference, shifted, extreme, unavailableCore, treated, untreated, treatedRiskDirection, untreatedRiskDirection].map(result => ({
    name: result.name,
    bands: Object.fromEntries(Object.entries(result.cards).map(([id, card]) => [id, card.className])),
  })),
}, null, 2));
