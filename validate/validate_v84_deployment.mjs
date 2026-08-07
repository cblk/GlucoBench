#!/usr/bin/env node
/** Verify the v8.4 browser deployment rule directly from index.html. */

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await fs.readFile(path.join(root, "index.html"), "utf8");
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(match => match[1]).filter(text => text.trim());
assert.ok(scripts.length, "inline application script not found");

const context = vm.createContext({
  console,
  document: { addEventListener() {} },
  window: {},
  setTimeout,
  clearTimeout,
});
vm.runInContext(`${scripts.at(-1)}\nglobalThis.__v84 = { computeScreeningRiskV84, directionalEvidenceV84, SCREENING_MODELS_V84 };`, context);
const { computeScreeningRiskV84, directionalEvidenceV84, SCREENING_MODELS_V84 } = context.__v84;

function auc(y, score) {
  const order = score.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const ranks = new Array(score.length);
  for (let start = 0; start < order.length;) {
    let end = start + 1;
    while (end < order.length && order[end].value === order[start].value) end++;
    const averageRank = (start + 1 + end) / 2;
    for (let i = start; i < end; i++) ranks[order[i].index] = averageRank;
    start = end;
  }
  const positives = y.reduce((sum, value) => sum + value, 0);
  const negatives = y.length - positives;
  const positiveRanks = ranks.reduce((sum, rank, index) => sum + (y[index] ? rank : 0), 0);
  return (positiveRanks - positives * (positives + 1) / 2) / (positives * negatives);
}

const cohorts = [
  { name: "hall", mode: "untreated" },
  { name: "colas", mode: "treated" },
];
const expected = JSON.parse(await fs.readFile(path.join(root, "output", "v84_expected.json"), "utf8"));
const summary = {};
for (const cohort of cohorts) {
  const rows = JSON.parse(await fs.readFile(path.join(root, "output", `phase_screening_metrics_${cohort.name}.json`), "utf8"));
  const knownRisk = [];
  for (const row of rows) {
    const dynamics = { lyapunov: row.lyapunov, det: row.det, entr: row.entr };
    const unknown = computeScreeningRiskV84(row.nightMean, dynamics, "unknown");
    assert.ok(Math.abs(unknown.risk - row.currentRisk) < 1e-15, `${cohort.name}/${row.id}: unknown fallback drift`);
    const missing = computeScreeningRiskV84(row.nightMean, { ...dynamics, lyapunov: null }, cohort.mode);
    assert.ok(Math.abs(missing.risk - row.currentRisk) < 1e-15, `${cohort.name}/${row.id}: missing fallback drift`);
    const known = computeScreeningRiskV84(row.nightMean, dynamics, cohort.mode);
    assert.equal(known.usedConsensus, true);
    assert.ok(Number.isFinite(known.risk) && known.risk > 0 && known.risk < 1);
    assert.ok(known.dynamicScore >= 0 && known.dynamicScore <= 3);
    assert.ok(
      Math.abs(known.risk - expected[cohort.name].probability_by_id[String(row.id)]) < 1e-12,
      `${cohort.name}/${row.id}: JS/Python deployment probability mismatch`,
    );
    knownRisk.push(known.risk);
  }
  const browserModel = SCREENING_MODELS_V84[cohort.mode];
  const pythonModel = expected[cohort.name].parameters;
  for (const key of ["intercept", "nightMeanCoef", "dynamicCoef"]) {
    assert.ok(Math.abs(browserModel[key] - pythonModel[key]) < 1e-12, `${cohort.name}: ${key} mismatch`);
  }
  for (const key of ["lyapunov", "det", "entr"]) {
    assert.ok(Math.abs(browserModel.reference[key].median - pythonModel.reference[key].median) < 1e-12);
    assert.ok(Math.abs(browserModel.reference[key].scale - pythonModel.reference[key].scale) < 1e-12);
  }
  summary[cohort.name] = {
    n: rows.length,
    positives: rows.reduce((sum, row) => sum + row.y, 0),
    full_fit_auc_diagnostic_only: auc(rows.map(row => row.y), knownRisk),
    unknown_fallback_max_absolute_difference: Math.max(...rows.map((row, index) => {
      const dynamics = { lyapunov: row.lyapunov, det: row.det, entr: row.entr };
      return Math.abs(computeScreeningRiskV84(row.nightMean, dynamics, "unknown").risk - row.currentRisk);
    })),
  };
}

// Consensus semantics: one abnormal metric cannot move the median, but every
// abnormal pair must increase risk; moving all metrics in the healthy direction cannot.
for (const mode of ["untreated", "treated"]) {
  const model = SCREENING_MODELS_V84[mode];
  const baselineDynamics = Object.fromEntries(
    Object.entries(model.reference).map(([key, ref]) => [key, ref.median]),
  );
  const baseline = computeScreeningRiskV84(6, baselineDynamics, mode);
  for (const key of ["lyapunov", "det", "entr"]) {
    const ref = model.reference[key];
    assert.equal(directionalEvidenceV84(ref.median, key, mode), 0);
    const single = { ...baselineDynamics, [key]: ref.median + ref.direction * 2 * ref.scale };
    assert.ok(Math.abs(computeScreeningRiskV84(6, single, mode).risk - baseline.risk) < 1e-15);
  }
  const keys = ["lyapunov", "det", "entr"];
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const pair = { ...baselineDynamics };
      for (const key of [keys[i], keys[j]]) {
        const ref = model.reference[key];
        pair[key] = ref.median + ref.direction * 2 * ref.scale;
      }
      assert.ok(computeScreeningRiskV84(6, pair, mode).risk > baseline.risk);
    }
  }
}

const outputPath = path.join(root, "output", "v84_deployment_check.json");
await fs.writeFile(outputPath, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify(summary, null, 2));
console.log(`wrote ${path.relative(root, outputPath)}`);
