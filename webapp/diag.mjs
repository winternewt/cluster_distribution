// Diagnostic: reproduce JS DBSCAN + R~ computation, print detailed statistics
// Run: node diag.mjs

import { createRequire } from 'module';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dir = dirname(fileURLToPath(import.meta.url));

// Minimal Math.random seed for reproducibility (LCG)
let seed = 42;
function rand() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}

const N = 10000, RADIUS = 100, MINS = 10, MINC = 10;
const LAM0 = N / (Math.PI * RADIUS * RADIUS);
const EPS = 1.40;

function genField() {
  const xy = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) {
    const r = RADIUS * Math.sqrt(rand()), t = 2 * Math.PI * rand();
    xy[2*i] = r * Math.cos(t); xy[2*i+1] = r * Math.sin(t);
  }
  return xy;
}

function dbscan(xy, eps) {
  const cs = eps, side = Math.ceil(2 * RADIUS / cs) + 2, off = RADIUS + cs;
  const cell = i => {
    const cx = Math.floor((xy[2*i]+off)/cs), cy = Math.floor((xy[2*i+1]+off)/cs);
    return cx * side + cy;
  };
  const grid = new Map();
  for (let i = 0; i < N; i++) {
    const k = cell(i); (grid.get(k) || grid.set(k,[]).get(k)).push(i);
  }
  const e2 = eps * eps;
  function region(i) {
    const cx = Math.floor((xy[2*i]+off)/cs), cy = Math.floor((xy[2*i+1]+off)/cs), out = [];
    const xi = xy[2*i], yi = xy[2*i+1];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const arr = grid.get((cx+dx)*side+(cy+dy)); if (!arr) continue;
      for (const j of arr) {
        const ddx = xy[2*j]-xi, ddy = xy[2*j+1]-yi;
        if (ddx*ddx+ddy*ddy <= e2) out.push(j);
      }
    }
    return out;
  }
  const lab = new Int32Array(N).fill(-2); let cid = 0;
  for (let i = 0; i < N; i++) {
    if (lab[i] !== -2) continue;
    const Ni = region(i);
    if (Ni.length < MINS) { lab[i] = -1; continue; }
    lab[i] = cid; const q = []; for (const j of Ni) if (j !== i) q.push(j);
    while (q.length) {
      const j = q.pop();
      if (lab[j] === -1) lab[j] = cid;
      if (lab[j] !== -2) continue;
      lab[j] = cid; const Nj = region(j); if (Nj.length >= MINS) for (const k of Nj) q.push(k);
    }
    cid++;
  }
  return { lab, cid };
}

function hullArea(pts) {
  if (pts.length < 3) return { area: 0, hull: [] };
  const p = pts.slice().sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  const cross = (o,a,b) => (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lo = []; for (const q of p) { while (lo.length>=2 && cross(lo[lo.length-2],lo[lo.length-1],q)<=0) lo.pop(); lo.push(q); }
  const up = []; for (let i = p.length-1; i >= 0; i--) { const q = p[i]; while (up.length>=2 && cross(up[up.length-2],up[up.length-1],q)<=0) up.pop(); up.push(q); }
  const h = lo.slice(0,-1).concat(up.slice(0,-1));
  let a = 0; for (let i = 0; i < h.length; i++) { const j = (i+1)%h.length; a += h[i][0]*h[j][1]-h[j][0]*h[i][1]; }
  return { area: Math.abs(a)/2, hull: h };
}

// Run N_FIELDS fields and collect Rt, N', S' statistics
const N_FIELDS = 200;
const Rt_all = [], Np_all = [], Sp_all = [];

console.log(`Running ${N_FIELDS} fields at eps=${EPS}, N=${N}, LAM0=${LAM0.toFixed(6)}`);

let totalClusters = 0;
for (let f = 0; f < N_FIELDS; f++) {
  const xy = genField();
  const { lab, cid } = dbscan(xy, EPS);
  const groups = Array.from({ length: cid }, () => []);
  for (let i = 0; i < N; i++) if (lab[i] >= 0) groups[lab[i]].push(i);
  for (const g of groups) {
    if (g.length < MINC) continue;
    const pts = g.map(i => [xy[2*i], xy[2*i+1]]);
    const { area } = hullArea(pts);
    if (area <= 0) continue;
    const R = (g.length / area) / LAM0;
    const alpha = 2.031525 + 0.258273 * Math.log(EPS);  // alpha(eps)=c0+c1*ln(eps)
    const Rt = R * Math.pow(EPS, alpha);
    Rt_all.push(Rt);
    Np_all.push(g.length);
    Sp_all.push(area);
    totalClusters++;
  }
}

function stats(arr, name) {
  const n = arr.length;
  const sorted = [...arr].sort((a,b) => a-b);
  const mean = arr.reduce((s,x) => s+x, 0) / n;
  const variance = arr.reduce((s,x) => s+(x-mean)**2, 0) / n;
  const sd = Math.sqrt(variance);
  const q = p => sorted[Math.floor(p*n)];
  console.log(`${name}: n=${n} mean=${mean.toFixed(3)} sd=${sd.toFixed(3)} q05=${q(.05).toFixed(3)} q25=${q(.25).toFixed(3)} median=${q(.5).toFixed(3)} q75=${q(.75).toFixed(3)} q95=${q(.95).toFixed(3)} max=${sorted[n-1].toFixed(3)}`);
  // histogram in bins of width 1 from 0 to 40
  const bins = new Array(45).fill(0);
  for (const x of arr) { const b = Math.floor(x); if (b >= 0 && b < bins.length) bins[b]++; }
  const bmax = Math.max(...bins);
  console.log(`  Histogram (bin=1.0, scale=50):`);
  for (let i = 10; i <= 40; i++) {
    const bar = '#'.repeat(Math.round(bins[i]/bmax*50));
    console.log(`  ${String(i).padStart(3)}: ${bar} (${bins[i]})`);
  }
}

stats(Rt_all, 'R̃');
stats(Np_all, "N'");
stats(Sp_all, "S'");

// Also check what fraction falls in each region vs master SF
// Log-logistic median = ll_scale = 24.057; SF(22) ≈ 0.65, SF(24) ≈ 0.50
const below22 = Rt_all.filter(x => x < 22).length / Rt_all.length;
const below24 = Rt_all.filter(x => x < 24).length / Rt_all.length;
console.log(`\nFraction R̃<22: ${below22.toFixed(3)} (master expects ~0.35)`);
console.log(`Fraction R̃<24: ${below24.toFixed(3)} (master expects ~0.50)`);
console.log(`Clusters/field: ${(totalClusters/N_FIELDS).toFixed(2)}`);
