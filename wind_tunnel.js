const fs = require('fs');

// 1. Read and extract JS from index.html
const html = fs.readFileSync('index.html', 'utf8');
const scriptMatch = html.match(/<script>(.*?)<\/script>/s);
if (!scriptMatch) {
    console.error("Could not find script block in index.html");
    process.exit(1);
}

// 2. Mock browser globals needed by the script
const mockEnv = `
    const window = { location: { search: '' }, addEventListener: () => {} };
    const document = { 
        addEventListener: () => {},
        getElementById: () => ({ addEventListener: () => {}, style: {}, classList: { add: ()=>{}, remove: ()=>{} }, innerHTML: '', value: '' }),
        createElement: () => ({ style: {} }),
        querySelector: () => ({ addEventListener: () => {} }),
        querySelectorAll: () => []
    };
    const Plotly = { newPlot: () => {}, react: () => {} };
    const Papa = { parse: () => {} };
    const XLSX = { read: () => {} };
`;

// 3. Eval the script
const scriptContent = scriptMatch[1];
eval(mockEnv + scriptContent);

// 4. Load dataset
console.log("Loading dataset...");
const data = JSON.parse(fs.readFileSync('output/phase_screening_subjects.json', 'utf8'));

// 5. Run wind tunnel
console.log("Starting Wind Tunnel Calibration...");

const cohorts = ['hall', 'colas'];
const results = [];

for (const cohort of cohorts) {
    if (!data[cohort]) continue;
    
    for (const subject of data[cohort]) {
        try {
            // Label Blinding: We ONLY use timestamps and values.
            const timestamps = subject.timestamps;
            const values = subject.values;
            
            if (!timestamps || !values || timestamps.length < 100) {
                console.log(`Skipping subject ${subject.id}, length: ${timestamps ? timestamps.length : 0}`);
                continue;
            }
            
            // Reconstruct the pipeline logic from index.html
            // We need to mock the Analyzer class or just use the functions directly
            
            // 1. Resample data
            const rawData = { 
                timestamps: timestamps.map(t => new Date(t)), 
                values: values 
            };
            const resampledRaw = resampleData(rawData, false);
            const resampledSmooth = resampleData(rawData, true);
            const rawValues = resampledRaw.values;
            const smoothValues = resampledSmooth.values;
            const resampledTimestamps = resampledRaw.timestamps;
            
            // 2. Slice by period
            const slicedRaw = sliceByPeriod(resampledTimestamps, rawValues, 'all');
            const slicedSmooth = sliceByPeriod(resampledTimestamps, smoothValues, 'all');
            
            // 3. Estimate embedding dimension
            const tau = 4; // default tau
            const { dim } = estimateEmbeddingDimension(slicedRaw.values, tau);
            
            // 4. Takens embedding
            const points = takensEmbedding(slicedSmooth.values, tau, dim);
            const pointsRaw = takensEmbedding(slicedRaw.values, tau, dim);
            const pointsSmooth = takensEmbedding(slicedSmooth.values, tau, dim);
            
            // 5. Night core
            const nightS = sliceByPeriod(resampledTimestamps, rawValues, 'night');
            const nightPointsAll = takensEmbedding(nightS.values, tau, dim);
            const nightPoints = nightPointsAll.filter(p => p !== null);
            const nightCore = nightPoints.length > 0
              ? new Array(dim).fill(0).map((_, d) => getMedian(nightPoints.map(p => p[d])))
              : null;
              
            let nightFriction = null;
            if (nightCore) {
              const nightFricObj = computeAsymmetricFriction(nightPointsAll, nightCore);
              nightFriction = nightFricObj ? nightFricObj.asymFriction : null;
            }
            
            // 6. Attractor metrics (Work Integral, Ascend Friction)
            const metrics = computeAttractorMetrics(points, pointsRaw, pointsSmooth, true);
            
            // 7. CSD (AR1)
            const csd = computeCriticalSlowingDown(resampledTimestamps, rawValues);
            
            if (metrics) {
                results.push({
                    cohort,
                    id: subject.id,
                    workIntegral: metrics.workIntegral,
                    ascendFriction: metrics.ascendFriction,
                    nightFriction: nightFriction,
                    ar1: csd.ar1
                });
            }
        } catch (e) {
            console.log(`Error on subject ${subject.id}: ${e.stack}`);
        }
    }
}

// 6. Analyze and print results
console.log(`\nProcessed ${results.length} subjects successfully.`);

const metrics = ['workIntegral', 'ascendFriction', 'nightFriction', 'ar1'];

for (const metric of metrics) {
    const validVals = results.map(r => r[metric]).filter(v => v !== null && isFinite(v));
    if (validVals.length === 0) continue;
    
    validVals.sort((a, b) => a - b);
    const min = validVals[0];
    const max = validVals[validVals.length - 1];
    const mean = validVals.reduce((a, b) => a + b, 0) / validVals.length;
    const p50 = validVals[Math.floor(validVals.length * 0.5)];
    const p90 = validVals[Math.floor(validVals.length * 0.9)];
    const p95 = validVals[Math.floor(validVals.length * 0.95)];
    const p99 = validVals[Math.floor(validVals.length * 0.99)];
    
    console.log(`\n--- Metric: ${metric} ---`);
    console.log(`Count: ${validVals.length}`);
    console.log(`Mean:  ${mean.toFixed(4)}`);
    console.log(`Min:   ${min.toFixed(4)}`);
    console.log(`P50:   ${p50.toFixed(4)}`);
    console.log(`P90:   ${p90.toFixed(4)}`);
    console.log(`P95:   ${p95.toFixed(4)}`);
    console.log(`P99:   ${p99.toFixed(4)}`);
    console.log(`Max:   ${max.toFixed(4)}`);
}

console.log("\nWind Tunnel Calibration Complete.");
