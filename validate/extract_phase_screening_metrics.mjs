#!/usr/bin/env node
/**
 * Run index.html's current v8.3 numerical pipeline on exported subjects.
 *
 * No algorithm is copied: the inline script is evaluated in a VM and the
 * production functions are called directly. RQA line statistics remain full
 * resolution; only the recurrence-plot coordinates are skipped.
 */

import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";


const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const html = await fs.readFile(path.join(root, "index.html"), "utf8");
const scriptMatches = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
const inline = scriptMatches.map(match => match[1]).filter(text => text.trim()).at(-1);
if (!inline) throw new Error("Unable to locate index.html inline pipeline script");

const pipelineConsole = process.env.GLUCOBENCH_PIPELINE_SILENT === "1"
  ? { log() {}, warn() {}, error: console.error }
  : console;
const context = vm.createContext({
  console: pipelineConsole,
  document: { addEventListener() {} },
  window: {},
  setTimeout,
  clearTimeout,
});
const expose = `
globalThis.__pipeline = {
  resampleData, computeACF, recommendTau, sliceByPeriod, takensEmbedding,
  estimateEmbeddingDimension, computeAttractorMetrics, computeRQA,
  calcDistance, getMedian, RQA_TARGET_RR
};`;
vm.runInContext(`${inline}\n${expose}`, context, { filename: "index-inline.js" });
const pipeline = context.__pipeline;

const sourcePath = process.argv[2]
  ? path.resolve(root, process.argv[2])
  : path.join(root, "output", "phase_screening_subjects.json");
const outputPrefix = process.argv[3] || "phase_screening_metrics";
const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));

function analyzeSubject(subject) {
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
  if (!metrics) throw new Error(`Insufficient phase points for ${subject.cohort}/${subject.id}`);

  const validSmooth = pointsSmooth.filter(point => point !== null);
  const rqa = pipeline.computeRQA(validSmooth, pipeline.RQA_TARGET_RR, true);
  const night = pipeline.sliceByPeriod(raw.timestamps, raw.values, "night");
  const nightValues = night.values.filter(value => value !== null);
  const nightMean = nightValues.length >= 6
    ? nightValues.reduce((sum, value) => sum + value, 0) / nightValues.length
    : null;
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
    y: subject.y,
    prefixNights: subject.prefixNights ?? null,
    diagnosis: subject.diagnosis,
    insulin: subject.insulin,
    SSPG: subject.SSPG,
    nRaw: subject.values.length,
    nResampled: raw.values.filter(value => value !== null).length,
    tau,
    embeddingDim: dimInfo.dim,
    dA: dimInfo.dA,
    calculatedDim: dimInfo.calculatedDim,
    dimCapped: dimInfo.capped,
    nightMean,
    volume: metrics.volume,
    shapeRatio: metrics.shapeRatio,
    avgRecovery: metrics.avgRecovery,
    dimension: metrics.dimension,
    lyapunov: metrics.lyapunov,
    det: rqa.det,
    entr: rqa.entr,
    rr: rqa.rr,
    coreDisplacement,
    currentRisk: nightMean === null ? null : 1 / (1 + Math.exp(-(1.064314 * nightMean - 6.746364))),
  };
}

const requestedCohorts = process.argv[4]
  ? new Set(process.argv[4].split(",").map(value => value.trim()).filter(Boolean))
  : null;
const cohorts = Object.keys(source).filter(
  key => Array.isArray(source[key]) && (!requestedCohorts || requestedCohorts.has(key)),
);
if (!cohorts.length) throw new Error(`No subject arrays found in ${sourcePath}`);

for (const cohort of cohorts) {
  const output = [];
  const subjects = source[cohort];
  for (let index = 0; index < subjects.length; index++) {
    output.push(analyzeSubject(subjects[index]));
    if ((index + 1) % 10 === 0 || index + 1 === subjects.length) {
      console.log(`${cohort}: ${index + 1}/${subjects.length}`);
    }
  }
  const outputPath = path.join(root, "output", `${outputPrefix}_${cohort}.json`);
  await fs.writeFile(outputPath, JSON.stringify(output, null, 2), "utf8");
  console.log(`wrote ${path.relative(root, outputPath)}`);
}
