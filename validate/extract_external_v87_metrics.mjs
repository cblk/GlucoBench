#!/usr/bin/env node
/** Run the frozen v8.7 production functions on external 48-hour subjects. */

import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await fs.readFile(path.join(root, "index.html"), "utf8");
const inline = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(match => match[1]).filter(text => text.trim()).at(-1);
if (!inline) throw new Error("index.html inline pipeline not found");

const context = vm.createContext({
  console: { log() {}, warn() {}, error: console.error },
  document: { addEventListener() {} },
  window: {},
  setTimeout,
  clearTimeout,
});
vm.runInContext(`${inline}\nglobalThis.__external = {
  resampleData, computeACF, recommendTau, estimateEmbeddingDimension,
  takensEmbedding, computeAttractorMetrics, sliceByPeriod, getMedian,
  computeAsymmetricFriction, computeScreeningRiskV87
};`, context, { filename: "index-inline.js" });
const pipeline = context.__external;

const source = JSON.parse(await fs.readFile(path.join(root, "output", "external_base5_subjects.json"), "utf8"));
const output = [];

function treatmentFor(cohort) {
  if (cohort.startsWith("kobe_")) return "untreated";
  if (cohort.startsWith("shanghai_")) return "treated";
  return "unknown";
}

function analyze(subject) {
  const rawData = {
    timestamps: subject.timestamps.map(value => new Date(value)),
    values: subject.values,
  };
  const raw = pipeline.resampleData(rawData, false);
  const smooth = pipeline.resampleData(rawData, true);
  const tau = pipeline.recommendTau(pipeline.computeACF(raw.values, 60));
  const dimInfo = pipeline.estimateEmbeddingDimension(raw.values, tau);
  const pointsShape = pipeline.takensEmbedding(smooth.values, tau, dimInfo.dim);
  const pointsRaw = pipeline.takensEmbedding(raw.values, tau, dimInfo.dim);
  const pointsSmooth = pipeline.takensEmbedding(smooth.values, tau, dimInfo.dim);
  const metrics = pipeline.computeAttractorMetrics(pointsShape, pointsRaw, pointsSmooth, true);
  if (!metrics) throw new Error(`insufficient phase points: ${subject.cohort}/${subject.id}`);

  const night = pipeline.sliceByPeriod(raw.timestamps, raw.values, "night");
  const nightValues = night.values.filter(value => value !== null && Number.isFinite(value));
  const nightMean = nightValues.length >= 6
    ? nightValues.reduce((sum, value) => sum + value, 0) / nightValues.length
    : null;
  const nightPointsAll = pipeline.takensEmbedding(night.values, tau, dimInfo.dim);
  const nightPoints = nightPointsAll.filter(point => point !== null);
  let nightFriction = null;
  if (nightPoints.length) {
    const nightCore = Array.from(
      { length: dimInfo.dim },
      (_, dimension) => pipeline.getMedian(nightPoints.map(point => point[dimension])),
    );
    const friction = pipeline.computeAsymmetricFriction(nightPointsAll, nightCore);
    nightFriction = friction ? friction.asymFriction : null;
  }

  const treatmentStatus = treatmentFor(subject.cohort);
  const screening = pipeline.computeScreeningRiskV87(
    nightMean,
    metrics.workIntegral,
    nightFriction,
    metrics.ascendFriction,
    treatmentStatus,
  );
  return {
    cohort: subject.cohort,
    id: String(subject.id),
    treatmentStatus,
    nResampled: raw.values.filter(value => value !== null).length,
    tau,
    embeddingDim: dimInfo.dim,
    nightMean,
    workIntegral: metrics.workIntegral,
    ascendFriction: metrics.ascendFriction,
    nightFriction,
    v87Risk: screening.risk,
    v87Mode: screening.mode,
    v87UsedDynamic: screening.usedDynamic,
  };
}

for (const [cohort, subjects] of Object.entries(source)) {
  if (!Array.isArray(subjects)) continue;
  for (let index = 0; index < subjects.length; index += 1) {
    output.push(analyze(subjects[index]));
    if ((index + 1) % 20 === 0 || index + 1 === subjects.length) {
      console.log(`${cohort}: ${index + 1}/${subjects.length}`);
    }
  }
}

const outputPath = path.join(root, "output", "external_v87_metrics.json");
await fs.writeFile(outputPath, JSON.stringify(output, null, 2), "utf8");
console.log(`wrote ${path.relative(root, outputPath)}`);
