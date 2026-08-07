#!/usr/bin/env node
/** Extract structure-only metrics by directly executing index.html's v8.4 JS. */

import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await fs.readFile(path.join(root, "index.html"), "utf8");
const inline = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(match => match[1]).filter(text => text.trim()).at(-1);
if (!inline) throw new Error("Unable to locate the inline application script");

const context = vm.createContext({
  console: { log() {}, warn: console.warn, error: console.error },
  document: { addEventListener() {} },
  window: {}, setTimeout, clearTimeout,
});
vm.runInContext(`${inline}\nglobalThis.__structure = {
  resampleData, computeACF, recommendTau, estimateEmbeddingDimension,
  takensEmbedding, computeAttractorMetrics, sliceByPeriod, getMedian, calcDistance
};`, context, { filename: "index-inline.js" });
const pipeline = context.__structure;

const sourcePath = process.argv[2]
  ? path.resolve(root, process.argv[2])
  : path.join(root, "output", "structure_windows.json");
const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const requested = process.argv[3]
  ? new Set(process.argv[3].split(",").map(value => value.trim()).filter(Boolean))
  : null;
const cohorts = Object.keys(source).filter(key => Array.isArray(source[key]) && (!requested || requested.has(key)));

function analyze(subject) {
  const rawData = {
    timestamps: subject.timestamps.map(value => new Date(value)),
    values: subject.values,
  };
  const raw = pipeline.resampleData(rawData, false);
  const smooth = pipeline.resampleData(rawData, true);
  const tau = pipeline.recommendTau(pipeline.computeACF(raw.values, 60));
  const dimInfo = pipeline.estimateEmbeddingDimension(raw.values, tau);
  const smoothPoints = pipeline.takensEmbedding(smooth.values, tau, dimInfo.dim);
  const rawPoints = pipeline.takensEmbedding(raw.values, tau, dimInfo.dim);
  const metrics = pipeline.computeAttractorMetrics(smoothPoints, rawPoints, smoothPoints, true);
  if (!metrics) throw new Error(`Insufficient phase points for ${subject.cohort}/${subject.id}/k${subject.windowDays}`);

  const night = pipeline.sliceByPeriod(raw.timestamps, raw.values, "night");
  const nightPoints = pipeline.takensEmbedding(night.values, tau, dimInfo.dim).filter(point => point !== null);
  let coreDisplacement = null;
  if (nightPoints.length) {
    const nightCore = Array.from(
      { length: dimInfo.dim },
      (_, dimension) => pipeline.getMedian(nightPoints.map(point => point[dimension])),
    );
    coreDisplacement = pipeline.calcDistance(metrics.gravityCore, nightCore);
  }
  return {
    cohort: subject.cohort,
    id: subject.id,
    windowDays: subject.windowDays,
    nRaw: subject.values.length,
    nResampled: raw.values.filter(value => value !== null).length,
    tau,
    embeddingDim: dimInfo.dim,
    volume: metrics.volume,
    avgRecovery: metrics.avgRecovery,
    coreDisplacement,
    dimension: metrics.dimension,
    shapeRatio: metrics.shapeRatio,
    earlyConventional: subject.earlyConventional,
    future: subject.future,
    clinical: subject.clinical,
  };
}

for (const cohort of cohorts) {
  const rows = [];
  for (let index = 0; index < source[cohort].length; index++) {
    rows.push(analyze(source[cohort][index]));
    if ((index + 1) % 25 === 0 || index + 1 === source[cohort].length) {
      console.log(`${cohort}: ${index + 1}/${source[cohort].length}`);
    }
  }
  const outputPath = path.join(root, "output", `structure_metrics_${cohort}.json`);
  await fs.writeFile(outputPath, JSON.stringify(rows, null, 2), "utf8");
  console.log(`wrote ${path.relative(root, outputPath)}`);
}
