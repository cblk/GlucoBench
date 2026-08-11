#!/usr/bin/env node
/** Verify the v8.5 browser deployment rule directly from index.html. */

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
vm.runInContext(`${scripts.at(-1)}\nglobalThis.__v85 = { computeScreeningRiskV85, SCREENING_MODELS_V85 };`, context);
const { computeScreeningRiskV85, SCREENING_MODELS_V85 } = context.__v85;

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

async function main() {
  const summary = {};
  for (const cohort of cohorts) {
    const rows = JSON.parse(await fs.readFile(path.join(root, "output", `phase_screening_metrics_${cohort.name}.json`), "utf8"));
    const knownRisk = [];
    const knownY = [];
    
    for (const row of rows) {
      if (row.asymFriction === null || row.asymFriction === undefined) {
          continue; // Skip invalid
      }
      
      const friction = row.asymFriction;
      
      const unknown = computeScreeningRiskV85(row.nightMean, friction, "unknown");
      assert.ok(Math.abs(unknown.risk - row.currentRisk) < 1e-15, `${cohort.name}/${row.id}: unknown fallback drift`);
      
      const missing = computeScreeningRiskV85(row.nightMean, null, cohort.mode);
      assert.ok(Math.abs(missing.risk - row.currentRisk) < 1e-15, `${cohort.name}/${row.id}: missing fallback drift`);
      
      const known = computeScreeningRiskV85(row.nightMean, friction, cohort.mode);
      assert.equal(known.usedFriction, true);
      assert.ok(Number.isFinite(known.risk) && known.risk > 0 && known.risk < 1);
      
      knownRisk.push(known.risk);
      knownY.push(row.y);
    }
    
    summary[cohort.name] = {
      n: knownY.length,
      positives: knownY.reduce((sum, val) => sum + val, 0),
      full_fit_auc_diagnostic_only: auc(knownY, knownRisk),
    };
  }

  const outputPath = path.join(root, "output", "v85_deployment_check.json");
  await fs.writeFile(outputPath, JSON.stringify(summary, null, 2), "utf8");
  console.log(JSON.stringify(summary, null, 2));
  console.log(`wrote ${path.relative(root, outputPath)}`);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
