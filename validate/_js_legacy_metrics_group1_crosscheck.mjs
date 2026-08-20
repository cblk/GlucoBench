// Cross-check harness (2026-08-19): verbatim copies of the 6 legacy Group-1 (neutral, no
// warn/bad color coding) JS functions from index_v4.html (calcDistance, boxCountingDimension,
// lyapunovProxy, jacobiEigenvalues, gammaApprox/VOLUME_COEFF, the covariance/volume/shape
// block from computeAttractorMetrics, computeNormalizedRecovery), run against a synthetic
// fixture and dumped as JSON so validate/_legacy_metrics_group1_v4.py's Python port can be
// diffed against the REAL JS output. NOT wired into any production path; this file exists
// solely to produce crosscheck evidence. Do not import this from any other script.
//
// Provenance of the verbatim blocks below: index_v4.html lines 1590-1597 (VOLUME_COEFF),
// 2171-2175 (calcDistance), 2178-2224 (boxCountingDimension), 2227-2281 (lyapunovProxy),
// 2413-2441 (jacobiEigenvalues), 2448-2455 (gammaApprox), 2845-2897 (covariance/volume/shape
// block inside computeAttractorMetrics), 2598-2623 (computeNormalizedRecovery), as of the
// 2026-08-19 21:00 HEAD used for this port.

import { readFileSync, writeFileSync } from 'fs';

function getMedian(arr) {
  if (!arr || arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const VOLUME_COEFF = {
  1: 2,
  2: Math.PI,
  3: (4/3) * Math.PI,
  4: Math.PI * Math.PI / 2,
  5: 8 * Math.PI * Math.PI / 15,
  6: Math.PI * Math.PI * Math.PI / 6,
};

function gammaApprox(x) {
  if (x <= 0) return 1;
  if (Math.abs(x - 2.5) < 0.001) return 1.329340388179137;
  if (Math.abs(x - 3.0) < 0.001) return 2;
  if (Math.abs(x - 3.5) < 0.001) return 3.3233509704478426;
  if (Math.abs(x - 4.0) < 0.001) return 6;
  return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
}

function calcDistance(p1, p2) {
  let sum = 0;
  for (let d = 0; d < p1.length; d++) sum += (p1[d] - p2[d]) ** 2;
  return Math.sqrt(sum);
}

function boxCountingDimension(points) {
  const valid = points.filter(p => p !== null);
  const n = valid.length;
  if (n < 20) return null;
  const dim = valid[0].length;

  const min = new Array(dim).fill(Infinity);
  const max = new Array(dim).fill(-Infinity);
  for (const p of valid) {
    for (let d = 0; d < dim; d++) {
      if (p[d] < min[d]) min[d] = p[d];
      if (p[d] > max[d]) max[d] = p[d];
    }
  }
  let range = 0;
  for (let d = 0; d < dim; d++) range = Math.max(range, max[d] - min[d]);
  if (range <= 1e-9) return 0;

  const divs = [2, 4, 8, 16];
  const logN = [], logInvEps = [];
  for (const g of divs) {
    const eps = range / g;
    const occ = new Set();
    for (const p of valid) {
      let key = 0;
      for (let d = 0; d < dim; d++) key = key * (g + 1) + Math.floor((p[d] - min[d]) / eps);
      occ.add(key);
    }
    logN.push(Math.log(occ.size));
    logInvEps.push(Math.log(g / range));
  }

  const k = logN.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < k; i++) {
    sx += logInvEps[i]; sy += logN[i];
    sxx += logInvEps[i] * logInvEps[i]; sxy += logInvEps[i] * logN[i];
  }
  const denom = k * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return 0;
  return Math.max(0, Math.min(dim, (k * sxy - sx * sy) / denom));
}

function lyapunovProxy(points) {
  const idx = [];
  for (let i = 0; i < points.length; i++) if (points[i] !== null) idx.push(i);
  const m = idx.length;
  if (m < 30) return null;
  const dim = points[idx[0]].length;

  const min = new Array(dim).fill(Infinity);
  const max = new Array(dim).fill(-Infinity);
  for (const i of idx) {
    for (let d = 0; d < dim; d++) {
      const v = points[i][d];
      if (v < min[d]) min[d] = v;
      if (v > max[d]) max[d] = v;
    }
  }
  let range = 0;
  for (let d = 0; d < dim; d++) range = Math.max(range, max[d] - min[d]);
  range = range || 1;
  const G = 16, cell = range / G;

  const keyOf = (p) => {
    let key = 0;
    for (let d = 0; d < dim; d++) key = key * (G + 1) + Math.floor((p[d] - min[d]) / cell);
    return key;
  };

  const grid = new Map();
  for (const i of idx) {
    const kk = keyOf(points[i]);
    if (!grid.has(kk)) grid.set(kk, []);
    grid.get(kk).push(i);
  }

  const THEILER = 5;
  const divs = [];
  for (const i of idx) {
    if (points[i + 1] == null) continue;
    const cand = grid.get(keyOf(points[i])) || [];
    let best = -1, bestD = Infinity;
    for (const j of cand) {
      if (Math.abs(i - j) <= THEILER) continue;
      if (points[j + 1] == null) continue;
      const d0 = calcDistance(points[i], points[j]);
      if (d0 > 1e-6 && d0 < bestD) { bestD = d0; best = j; }
    }
    if (best < 0) continue;
    const d1 = calcDistance(points[i + 1], points[best + 1]);
    if (d1 > 1e-9 && bestD > 1e-9) divs.push(Math.log(d1 / bestD));
  }
  if (divs.length < 10) return null;
  return divs.reduce((a, b) => a + b, 0) / divs.length;
}

function jacobiEigenvalues(A, maxIter = 50) {
  const n = A.length;
  const a = A.map(row => [...row]);
  for (let iter = 0; iter < maxIter; iter++) {
    let maxVal = 0, p = 0, q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(a[i][j]) > maxVal) { maxVal = Math.abs(a[i][j]); p = i; q = j; }
      }
    }
    if (maxVal < 1e-10) break;
    const theta = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q]);
    const c = Math.cos(theta), s = Math.sin(theta);
    const app = a[p][p], aqq = a[q][q], apq = a[p][q];
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (let j = 0; j < n; j++) {
      if (j === p || j === q) continue;
      const apj = a[p][j], aqj = a[q][j];
      a[p][j] = a[j][p] = c * apj - s * aqj;
      a[q][j] = a[j][q] = s * apj + c * aqj;
    }
  }
  const ev = [];
  for (let i = 0; i < n; i++) ev.push(a[i][i]);
  ev.sort((x, y) => y - x);
  return ev;
}

// Verbatim covariance/volume/shape block from computeAttractorMetrics(), index_v4.html:2845-2897.
function computeVolumeShape(points) {
  const valid = points.filter(p => p !== null);
  const n = valid.length;
  if (n < 4) return null;
  const dim = valid[0].length;

  const arithMean = new Array(dim).fill(0);
  for (const p of valid) for (let d = 0; d < dim; d++) arithMean[d] += p[d];
  for (let d = 0; d < dim; d++) arithMean[d] /= n;

  const cols = Array.from({ length: dim }, () => new Array(n));
  for (let i = 0; i < n; i++) { const p = valid[i]; for (let d = 0; d < dim; d++) cols[d][i] = p[d]; }
  const gravityCore = new Array(dim);
  for (let d = 0; d < dim; d++) gravityCore[d] = getMedian(cols[d]);

  const cov = Array.from({ length: dim }, () => new Float64Array(dim));
  const centered = new Array(dim);
  for (const p of valid) {
    for (let d = 0; d < dim; d++) centered[d] = p[d] - arithMean[d];
    for (let i = 0; i < dim; i++) {
      const ci = centered[i];
      for (let j = i; j < dim; j++) cov[i][j] += ci * centered[j];
    }
  }
  for (let i = 0; i < dim; i++) {
    for (let j = i + 1; j < dim; j++) cov[j][i] = cov[i][j];
    for (let j = 0; j < dim; j++) cov[i][j] /= (n - 1);
  }

  const eigvals = jacobiEigenvalues(cov).map(v => Math.max(v, 1e-10));

  const totalEnergy = eigvals.reduce((a, b) => a + b, 0);
  let accumulatedEnergy = 0;
  let effectiveDim = 0;
  for (let i = 0; i < dim; i++) {
    accumulatedEnergy += eigvals[i];
    effectiveDim++;
    if (accumulatedEnergy / totalEnergy > 0.99) break;
  }

  const coeff = VOLUME_COEFF[effectiveDim] || (Math.pow(Math.PI, effectiveDim / 2) / gammaApprox(effectiveDim / 2 + 1));
  let volProduct = 1;
  for (let i = 0; i < effectiveDim; i++) volProduct *= Math.sqrt(eigvals[i]);
  const volume = coeff * volProduct;

  const shapeRatio = eigvals[1] > 1e-12 ? eigvals[0] / eigvals[1] : Infinity;

  return { volume, shapeRatio: Number.isFinite(shapeRatio) ? shapeRatio : null, gravityCore, effectiveDim };
}

function computeNormalizedRecovery(points) {
  const valid = points.filter(p => p !== null);
  const n = valid.length; if (n < 4) return 0;
  const dim = valid[0].length;
  const cols = Array.from({ length: dim }, () => new Array(n));
  for (let i = 0; i < n; i++) { const p = valid[i]; for (let d = 0; d < dim; d++) cols[d][i] = p[d]; }
  const gc = new Array(dim);
  for (let d = 0; d < dim; d++) gc[d] = getMedian(cols[d]);

  const mean0 = valid.reduce((s, p) => s + p[0], 0) / n;
  let var0 = 0; for (const p of valid) var0 += (p[0] - mean0) * (p[0] - mean0);
  const glucoseStd = Math.sqrt(var0 / (n - 1));

  const recovery = []; let prevDist = null;
  for (let i = 1; i < points.length; i++) {
    if (points[i] && points[i - 1]) {
      const d = calcDistance(points[i], gc);
      const speed = calcDistance(points[i], points[i - 1]);
      if (prevDist !== null && d < prevDist) recovery.push(speed);
      prevDist = d;
    } else prevDist = null;
  }
  const avg = recovery.length > 0 ? recovery.reduce((a, b) => a + b, 0) / recovery.length : 0;
  return glucoseStd > 1e-6 ? avg / glucoseStd : 0;
}

// --- Harness driver ---
const fixturePath = process.argv[2];
const outPath = process.argv[3];
const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));

const shapePoints = fixture.shapePoints.map(p => p === null ? null : p);
const rawPoints = fixture.rawPoints.map(p => p === null ? null : p);
const smoothPoints = fixture.smoothPoints.map(p => p === null ? null : p);

const vs = computeVolumeShape(shapePoints);

const result = {
  volumeShape: vs,
  boxCountingDimension: boxCountingDimension(shapePoints),
  lyapunovProxy: lyapunovProxy(smoothPoints),
  normalizedRecovery: computeNormalizedRecovery(rawPoints),
  coreDist: vs ? calcDistance(vs.gravityCore, fixture.nightCore) : null,
};

writeFileSync(outPath, JSON.stringify(result, null, 2));
console.log(`Wrote JS Group-1 cross-check output to ${outPath}`);
