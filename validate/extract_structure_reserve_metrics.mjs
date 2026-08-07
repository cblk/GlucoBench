#!/usr/bin/env node
/** Extract structure-reserve metrics by executing index.html's native v8.4 JS. */

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
vm.runInContext(`${inline}\nglobalThis.__reserve = {
  resampleData, computeACF, recommendTau, estimateEmbeddingDimension,
  takensEmbedding, computeAttractorMetrics, computeRQA, sliceByPeriod,
  getMedian, calcDistance
};`, context, { filename: "index-inline.js" });
const pipeline = context.__reserve;

const sourcePath = path.join(root, "output", "structure_reserve_windows.json");
const outputPath = path.join(root, "output", "structure_reserve_metrics.json");
const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const eventsOnly = process.argv.includes("--events-only");


function sliceHours(ts, vs, startHour, endHour) {
  const timestamps = [], values = [];
  for (let i = 0; i < ts.length; i++) {
    if (vs[i] === null) {
      timestamps.push(ts[i]); values.push(null); continue;
    }
    const hour = ts[i].getHours() + ts[i].getMinutes() / 60;
    const inRange = hour >= startHour && hour < endHour;
    if (inRange) {
      timestamps.push(ts[i]); values.push(vs[i]);
    } else if (values.length && values.at(-1) !== null) {
      timestamps.push(ts[i]); values.push(null);
    }
  }
  return { timestamps, values };
}


function conventional(values) {
  const valid = values.filter(value => value !== null && Number.isFinite(value));
  if (!valid.length) return null;
  const mean = valid.reduce((sum, value) => sum + value, 0) / valid.length;
  const variance = valid.reduce((sum, value) => sum + (value - mean) ** 2, 0) / valid.length;
  const sd = Math.sqrt(variance);
  return {
    n: valid.length,
    mean,
    sd,
    cv: mean > 1e-8 ? sd / mean : null,
    outOfRange: valid.filter(value => value < 3.9 || value > 10).length / valid.length,
    high: valid.filter(value => value > 10).length / valid.length,
    low: valid.filter(value => value < 3.9).length / valid.length,
  };
}


function stateMetrics(raw, smooth, tau, dim) {
  const rawPoints = pipeline.takensEmbedding(raw.values, tau, dim);
  const smoothPoints = pipeline.takensEmbedding(smooth.values, tau, dim);
  const metrics = pipeline.computeAttractorMetrics(smoothPoints, rawPoints, smoothPoints, true);
  if (!metrics) return null;
  const validSmooth = smoothPoints.filter(point => point !== null);
  const rqa = pipeline.computeRQA(validSmooth, 0.02, true);
  return {
    volume: metrics.volume,
    recovery: metrics.avgRecovery,
    lyapunov: metrics.lyapunov,
    det: rqa.det,
    entr: rqa.entr,
    rr: rqa.rr,
    dimension: metrics.dimension,
    shapeRatio: metrics.shapeRatio,
    gravityCore: metrics.gravityCore,
    radius: pipeline.getMedian(validSmooth.map(point => pipeline.calcDistance(point, metrics.gravityCore))),
    validPhasePoints: validSmooth.length,
    conventional: conventional(raw.values),
  };
}


function coreShift(reference, challenge) {
  if (!reference || !challenge || !Number.isFinite(reference.radius) || reference.radius <= 1e-8) return null;
  return pipeline.calcDistance(reference.gravityCore, challenge.gravityCore) / reference.radius;
}


function analyzeStateRecord(subject) {
  const data = {
    timestamps: subject.timestamps.map(value => new Date(value)),
    values: subject.values,
  };
  const raw = pipeline.resampleData(data, false);
  const smooth = pipeline.resampleData(data, true);
  const tau = pipeline.recommendTau(pipeline.computeACF(raw.values, 60));
  const dimInfo = pipeline.estimateEmbeddingDimension(raw.values, tau);

  const nightRaw = pipeline.sliceByPeriod(raw.timestamps, raw.values, "night");
  const nightSmooth = pipeline.sliceByPeriod(smooth.timestamps, smooth.values, "night");
  const dayRaw = pipeline.sliceByPeriod(raw.timestamps, raw.values, "daytime");
  const daySmooth = pipeline.sliceByPeriod(smooth.timestamps, smooth.values, "daytime");
  const earlyRaw = sliceHours(raw.timestamps, raw.values, 0, 3);
  const earlySmooth = sliceHours(smooth.timestamps, smooth.values, 0, 3);
  const lateRaw = sliceHours(raw.timestamps, raw.values, 3, 6);
  const lateSmooth = sliceHours(smooth.timestamps, smooth.values, 3, 6);

  const night = stateMetrics(nightRaw, nightSmooth, tau, dimInfo.dim);
  const daytime = stateMetrics(dayRaw, daySmooth, tau, dimInfo.dim);
  const nightEarly = stateMetrics(earlyRaw, earlySmooth, tau, dimInfo.dim);
  const nightLate = stateMetrics(lateRaw, lateSmooth, tau, dimInfo.dim);
  if (!night || !daytime) throw new Error(`Insufficient state metrics: ${subject.cohort}/${subject.id}/${subject.split}`);

  return {
    cohort: subject.cohort,
    id: subject.id,
    split: subject.split,
    pairedDates: subject.pairedDates,
    tau,
    embeddingDim: dimInfo.dim,
    calculatedEmbeddingDim: dimInfo.calculatedDim,
    embeddingCapped: dimInfo.capped,
    night,
    daytime,
    coreShift: coreShift(night, daytime),
    pseudoNightEarly: nightEarly,
    pseudoNightLate: nightLate,
    pseudoCoreShift: coreShift(nightEarly, nightLate),
    clinical: subject.clinical,
  };
}


function quantile(values, probability) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * probability;
  const low = Math.floor(position), high = Math.ceil(position);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}


function envelopeResponse(preRaw, postRaw) {
  const pre = preRaw.values.filter(value => value !== null && Number.isFinite(value));
  const post = postRaw.values.map((value, index) => ({ value, time: postRaw.timestamps[index] }))
    .filter(item => item.value !== null && Number.isFinite(item.value));
  if (!pre.length || !post.length) return null;
  const center = pipeline.getMedian(pre);
  const mad = pipeline.getMedian(pre.map(value => Math.abs(value - center)));
  const robustScale = Math.max(0.3, 1.4826 * mad);
  const limit = 1.5 * robustScale;
  const excursion = post.map(item => Math.abs(item.value - center) / robustScale);
  let peakIndex = 0;
  for (let i = 1; i < excursion.length; i++) if (excursion[i] > excursion[peakIndex]) peakIndex = i;
  let returnMinutes = null;
  for (let i = peakIndex; i <= post.length - 3; i++) {
    if (post.slice(i, i + 3).every(item => Math.abs(item.value - center) <= limit)) {
      returnMinutes = (post[i].time.getTime() - post[0].time.getTime()) / 60000;
      break;
    }
  }
  const minutes = post.map((item, index) => index === 0 ? 0 : (item.time - post[index - 1].time) / 60000);
  const envelopeArea = excursion.reduce((sum, value, index) => sum + Math.max(0, value - 1.5) * Math.max(0, minutes[index]), 0);
  return {
    baselineCenter: center,
    baselineRobustScale: robustScale,
    peakStandardizedExcursion: excursion[peakIndex],
    returnMinutes,
    returnCensored: returnMinutes === null,
    standardizedEnvelopeArea: envelopeArea,
    postMedian: quantile(post.map(item => item.value), 0.5),
  };
}


function analyzeDubossonEvent(event) {
  const preData = {
    timestamps: event.preTimestamps.map(value => new Date(value)),
    values: event.preValues,
  };
  const postData = {
    timestamps: event.postTimestamps.map(value => new Date(value)),
    values: event.postValues,
  };
  // Match the structural observation durations exactly: pre is [-120,-15)
  // (105 minutes), so post structure uses [+15,+120] (also 105 minutes).
  // The full +180-minute post series remains available only to the envelope
  // return calculation below and cannot inflate Volume/DET/ENTR.
  const eventTime = new Date(event.eventTime);
  const structuralEnd = new Date(eventTime.getTime() + 120 * 60000);
  const postStructureData = {
    timestamps: [],
    values: [],
  };
  for (let index = 0; index < postData.timestamps.length; index++) {
    if (postData.timestamps[index] <= structuralEnd) {
      postStructureData.timestamps.push(postData.timestamps[index]);
      postStructureData.values.push(postData.values[index]);
    }
  }
  const combined = {
    timestamps: [...preData.timestamps, ...postStructureData.timestamps],
    values: [...preData.values, ...postStructureData.values],
  };
  const combinedRaw = pipeline.resampleData(combined, false);
  const tau = pipeline.recommendTau(pipeline.computeACF(combinedRaw.values, 60));
  const dimInfo = pipeline.estimateEmbeddingDimension(combinedRaw.values, tau);
  const preRaw = pipeline.resampleData(preData, false);
  const preSmooth = pipeline.resampleData(preData, true);
  const postRaw = pipeline.resampleData(postStructureData, false);
  const postSmooth = pipeline.resampleData(postStructureData, true);
  const fullPostRaw = pipeline.resampleData(postData, false);
  const pre = stateMetrics(preRaw, preSmooth, tau, dimInfo.dim);
  const post = stateMetrics(postRaw, postSmooth, tau, dimInfo.dim);
  if (!pre || !post) return null;
  return {
    id: event.id,
    eventId: event.eventId,
    eventTime: event.eventTime,
    eventTypes: event.eventTypes,
    fastInsulinPeak30m: event.fastInsulinPeak30m,
    caloriesPeak30m: event.caloriesPeak30m,
    tau,
    embeddingDim: dimInfo.dim,
    pre,
    post,
    coreShift: coreShift(pre, post),
    envelope: envelopeResponse(preRaw, fullPostRaw),
  };
}


let stateMetricsRows = [];
let stateMetricsSixDayRows = [];
if (eventsOnly) {
  const previous = JSON.parse(await fs.readFile(outputPath, "utf8"));
  stateMetricsRows = previous.stateMetrics;
  stateMetricsSixDayRows = previous.stateMetricsSixDay || [];
  console.log(`state records preserved: ${stateMetricsRows.length}`);
  console.log(`six-day state records preserved: ${stateMetricsSixDayRows.length}`);
} else {
  for (let index = 0; index < source.stateRecords.length; index++) {
    stateMetricsRows.push(analyzeStateRecord(source.stateRecords[index]));
    if ((index + 1) % 50 === 0 || index + 1 === source.stateRecords.length) {
      console.log(`state records: ${index + 1}/${source.stateRecords.length}`);
    }
  }
  for (let index = 0; index < source.stateRecordsSixDay.length; index++) {
    stateMetricsSixDayRows.push(analyzeStateRecord(source.stateRecordsSixDay[index]));
    if ((index + 1) % 50 === 0 || index + 1 === source.stateRecordsSixDay.length) {
      console.log(`six-day state records: ${index + 1}/${source.stateRecordsSixDay.length}`);
    }
  }
}

const eventMetricsRows = [];
for (let index = 0; index < source.dubossonEvents.length; index++) {
  const metrics = analyzeDubossonEvent(source.dubossonEvents[index]);
  if (metrics) eventMetricsRows.push(metrics);
}

await fs.writeFile(outputPath, JSON.stringify({
  metadata: {
    source: path.relative(root, sourcePath),
    implementation: "index.html v8.4 inline JavaScript",
    rqaTargetRecurrenceRate: 0.02,
    coreShiftNormalization: "distance(night core, daytime core) / median night attractor radius",
    pseudoState: "00:00-03:00 versus 03:00-06:00",
    dubossonStructuralWindows: "matched 105 minutes: pre [-120,-15), post [+15,+120]",
    dubossonEnvelopeWindow: "post [+15,+180]",
  },
  stateMetrics: stateMetricsRows,
  stateMetricsSixDay: stateMetricsSixDayRows,
  dubossonEventMetrics: eventMetricsRows,
}, null, 2), "utf8");
console.log(`dubosson events: ${eventMetricsRows.length}/${source.dubossonEvents.length}`);
console.log(`wrote ${path.relative(root, outputPath)}`);
