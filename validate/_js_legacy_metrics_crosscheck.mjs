// Cross-check harness (2026-08-19): verbatim copies of the 4 legacy JS functions from
// index_v4.html (computeCriticalSlowingDown, computeAsymmetricFriction,
// computeExcursionKinetics, computeKeplerKinematics, + the getMedian helper they depend
// on), run against a synthetic fixture and dumped as JSON so validate/_legacy_metrics_v4.py's
// Python port can be diffed against the REAL JS output rather than trusting hand-transcription
// alone. NOT wired into any production path; this file exists solely to produce
// reports/legacy_metrics_js_vs_python_crosscheck_*.json evidence. Do not import this from any
// other script.
//
// Provenance of the verbatim blocks below: index_v4.html lines 1692-1697 (getMedian),
// 2485-2592 (computeAsymmetricFriction), 2630-2695 (computeKeplerKinematics),
// 2698-2785 (computeExcursionKinetics), 2791-2841 (computeCriticalSlowingDown), as of the
// 2026-08-19 16:10 HEAD used for this port (commit bf8113a).

import { readFileSync, writeFileSync } from 'fs';

function getMedian(arr) {
  if (!arr || arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function computeAsymmetricFriction(points, core) {
  let frictionSum = 0;
  let count = 0;
  let workIntegral = 0;

  let ascFrictionSum = 0;
  let ascCount = 0;
  let descDistances = [];
  let descFrictions = [];

  if (!points || points.length < 3 || !core) return { asymFriction: null, workIntegral: null, ascendFriction: null, frictionGradient: null };

  let dim = 3;
  for (const p of points) {
    if (p !== null) {
      dim = p.length;
      break;
    }
  }

  for (let i = 1; i < points.length; i++) {
    if (points[i] !== null && points[i-1] !== null) {
      const v_g = points[i][0] - points[i-1][0];

      let vSq = 0;
      let distSq = 0;
      for(let d=0; d<dim; d++){
          const v = points[i][d] - points[i-1][d];
          vSq += v * v;
          const dist = points[i][d] - core[d];
          distSq += dist * dist;
      }
      const lenV = Math.sqrt(vSq);
      const lenDist = Math.sqrt(distSq);

      if (dim >= 2) {
        const dx = points[i][0] - points[i-1][0];
        const dy = points[i][1] - points[i-1][1];
        workIntegral += Math.sqrt(dx * dx + dy * dy);
      }

      if (lenV > 1e-6) {
        const currentFric = lenDist / lenV;
        if (v_g < -0.01) {
          frictionSum += currentFric;
          count++;
          descDistances.push(lenDist);
          descFrictions.push(currentFric);
        } else if (v_g > 0.01) {
          ascFrictionSum += currentFric;
          ascCount++;
        }
      }
    }
  }

  let asymFriction = count > 0 ? frictionSum / count : null;
  let ascendFriction = ascCount > 0 ? ascFrictionSum / ascCount : null;

  let validPhasePoints = 0;
  for (let i = 0; i < points.length; i++) {
    if (points[i] !== null) validPhasePoints++;
  }
  if (validPhasePoints > 0) {
    workIntegral = workIntegral / (validPhasePoints / 480.0);
  }

  let frictionGradient = null;
  if (count >= 4) {
    const sortedDists = [...descDistances].sort((a, b) => a - b);
    const mid = Math.floor(sortedDists.length / 2);
    const medianDist = sortedDists.length % 2 !== 0 ? sortedDists[mid] : (sortedDists[mid - 1] + sortedDists[mid]) / 2.0;

    let innerSum = 0, innerCount = 0;
    let outerSum = 0, outerCount = 0;

    for (let i = 0; i < count; i++) {
      if (descDistances[i] < medianDist) {
        innerSum += descFrictions[i];
        innerCount++;
      } else {
        outerSum += descFrictions[i];
        outerCount++;
      }
    }

    if (innerCount > 0 && outerCount > 0) {
      const innerFric = innerSum / innerCount;
      const outerFric = outerSum / outerCount;
      if (innerFric > 1e-6) {
        frictionGradient = outerFric / innerFric;
      }
    }
  }

  return {
    asymFriction,
    workIntegral,
    ascendFriction,
    frictionGradient
  };
}

function computeKeplerKinematics(times, vals, core) {
  if (!vals || !times || vals.length < 3 || core === null) return { angularVelocity: null, sweepRate: null };

  const X = new Array(vals.length);
  const Y = new Array(vals.length);
  const T_hours = new Float64Array(vals.length);

  for (let i = 0; i < vals.length; i++) {
    T_hours[i] = times[i].getTime() / (3600 * 1000);
    X[i] = vals[i] !== null ? vals[i] - core : null;
  }

  for (let i = 0; i < vals.length; i++) {
    if (X[i] === null) { Y[i] = null; continue; }
    if (i > 0 && i < vals.length - 1 && X[i-1] !== null && X[i+1] !== null) {
      Y[i] = (X[i+1] - X[i-1]) / (T_hours[i+1] - T_hours[i-1]);
    } else if (i === 0 && X[i+1] !== null) {
      Y[i] = (X[i+1] - X[i]) / (T_hours[i+1] - T_hours[i]);
    } else if (i === vals.length - 1 && X[i-1] !== null) {
      Y[i] = (X[i] - X[i-1]) / (T_hours[i] - T_hours[i-1]);
    } else {
      Y[i] = null;
    }
  }

  const dY = new Array(vals.length);
  for (let i = 0; i < vals.length; i++) {
    if (Y[i] === null) { dY[i] = null; continue; }
    if (i > 0 && i < vals.length - 1 && Y[i-1] !== null && Y[i+1] !== null) {
      dY[i] = (Y[i+1] - Y[i-1]) / (T_hours[i+1] - T_hours[i-1]);
    } else {
       dY[i] = null;
    }
  }

  let sumSweep = 0;
  let sumR2 = 0;
  let validCount = 0;

  for (let i = 1; i < vals.length - 1; i++) {
    if (X[i] === null || Y[i] === null || dY[i] === null) continue;

    const r2 = X[i]*X[i] + Y[i]*Y[i];
    if (r2 < 0.05) continue;

    const sweep = 0.5 * Math.abs(X[i] * dY[i] - Y[i] * Y[i]);

    sumSweep += sweep;
    sumR2 += r2;
    validCount++;
  }

  if (validCount < 5 || sumR2 === 0) return { angularVelocity: null, sweepRate: null };

  const avgSweepRate = sumSweep / validCount;
  const weightedAngularVel = 2.0 * sumSweep / sumR2;

  return { angularVelocity: weightedAngularVel, sweepRate: avgSweepRate };
}

function computeExcursionKinetics(times, vals) {
  if (!vals || !times || vals.length < 10) return { earlyDelay: null, relaxationTime: null };

  let peaks = [];
  let valleys = [];
  let dir = 0;

  for (let i = 1; i < vals.length; i++) {
    if (vals[i] === null || vals[i-1] === null) continue;
    let diff = vals[i] - vals[i-1];
    if (Math.abs(diff) < 0.01) continue;

    let newDir = diff > 0 ? 1 : -1;
    if (dir !== 0 && newDir !== dir) {
      if (newDir === 1) valleys.push({ idx: i-1, val: vals[i-1] });
      else peaks.push({ idx: i-1, val: vals[i-1] });
    }
    dir = newDir;
  }

  let earlyDelays = [];
  let relaxTimes = [];

  for (let p of peaks) {
    let prevValley = null;
    for (let j = valleys.length - 1; j >= 0; j--) {
      if (valleys[j].idx < p.idx) { prevValley = valleys[j]; break; }
    }
    if (!prevValley && p.idx > 0) prevValley = { idx: 0, val: vals[0] };

    if (!prevValley || p.val - prevValley.val < 1.5) continue;

    let maxDv = -Infinity;
    let maxDvIdx = prevValley.idx;
    for (let i = prevValley.idx; i < p.idx; i++) {
      if (vals[i] !== null && vals[i+1] !== null) {
        let dv = vals[i+1] - vals[i];
        if (dv > maxDv) { maxDv = dv; maxDvIdx = i; }
      }
    }
    earlyDelays.push((times[maxDvIdx].getTime() - times[prevValley.idx].getTime()) / 60000);

    let targetVal = p.val - 0.5 * (p.val - prevValley.val);
    let relaxEndIdx = p.idx;
    let found = false;

    let nextValley = null;
    for (let j = 0; j < valleys.length; j++) {
      if (valleys[j].idx > p.idx) { nextValley = valleys[j]; break; }
    }
    let searchEnd = nextValley ? nextValley.idx : vals.length - 1;

    for (let i = p.idx; i <= searchEnd; i++) {
      if (vals[i] !== null && vals[i] <= targetVal) {
        relaxEndIdx = i;
        found = true;
        break;
      }
    }

    if (found) {
      relaxTimes.push((times[relaxEndIdx].getTime() - times[p.idx].getTime()) / 60000);
    } else {
      let duration = (times[searchEnd].getTime() - times[p.idx].getTime()) / 60000;
      relaxTimes.push(duration * 1.5);
    }
  }

  const getMedianLocal = (arr) => {
    if (!arr || arr.length === 0) return null;
    arr.sort((a,b) => a - b);
    return arr.length % 2 !== 0
      ? arr[Math.floor(arr.length / 2)]
      : (arr[arr.length / 2 - 1] + arr[arr.length / 2]) / 2;
  };

  return {
    earlyDelay: getMedianLocal(earlyDelays),
    relaxationTime: getMedianLocal(relaxTimes)
  };
}

function computeCriticalSlowingDown(times, vals) {
  const nightsByDay = new Map();
  for (let i = 0; i < times.length; i++) {
    if (!times[i] || vals[i] === null) continue;
    const h = times[i].getHours();
    if (h >= 0 && h < 6) {
      const dayKey = `${times[i].getFullYear()}-${times[i].getMonth()}-${times[i].getDate()}`;
      if (!nightsByDay.has(dayKey)) nightsByDay.set(dayKey, []);
      nightsByDay.get(dayKey).push(vals[i]);
    }
  }

  const perNightAr1 = [];
  const perNightVariance = [];
  const perNightSkewness = [];

  for (const nightVals of nightsByDay.values()) {
    if (nightVals.length < 30) continue;

    const mean = nightVals.reduce((a, b) => a + b, 0) / nightVals.length;
    const variance = nightVals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (nightVals.length - 1);
    const m3 = nightVals.reduce((a, b) => a + Math.pow(b - mean, 3), 0) / nightVals.length;
    const skewness = variance > 0 ? m3 / Math.pow(variance, 1.5) : 0;

    let num = 0, den = 0;
    for (let i = 1; i < nightVals.length; i++) {
      num += (nightVals[i] - mean) * (nightVals[i-1] - mean);
    }
    for (let i = 0; i < nightVals.length; i++) {
      den += Math.pow(nightVals[i] - mean, 2);
    }
    const ar1 = den > 1e-8 ? num / den : 0;

    perNightAr1.push(ar1);
    perNightVariance.push(variance);
    perNightSkewness.push(skewness);
  }

  if (perNightAr1.length === 0) return { ar1: null, variance: null, skewness: null };

  return {
    ar1: getMedian(perNightAr1),
    variance: getMedian(perNightVariance),
    skewness: getMedian(perNightSkewness)
  };
}

// --- Harness driver ---
const fixturePath = process.argv[2];
const outPath = process.argv[3];
const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));

const times = fixture.timestamps.map(t => new Date(t));
const vals = fixture.values.map(v => v === null ? null : v);
const points = fixture.points.map(p => p === null ? null : p);
const core = fixture.core;
const nightPoints = fixture.nightPoints.map(p => p === null ? null : p);
const nightCore = fixture.nightCore;
const nightMean = fixture.nightMean;

const result = {
  criticalSlowingDown: computeCriticalSlowingDown(times, vals),
  ascendFriction: computeAsymmetricFriction(points, core),
  nightFriction: computeAsymmetricFriction(nightPoints, nightCore),
  excursionKinetics: computeExcursionKinetics(times, vals),
  keplerKinematics: computeKeplerKinematics(times, vals, nightMean),
};

writeFileSync(outPath, JSON.stringify(result, null, 2));
console.log(`Wrote JS cross-check output to ${outPath}`);
