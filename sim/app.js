"use strict";
/* 응급실 시뮬레이터
   sim_data.js(실제 데이터에서 구한 확률)로 가짜 타임라인을 만들고, 실제 타임라인과 같은 방식으로 비교한다.
   타임라인 값: 1 꽉 참, 0 자리 있음, -1 모름(수집 실패). 가짜에도 실제와 같은 자리를 -1로 지워서 공정하게 비교. */

const DATA = window.SIM_DATA;
const HOSP = DATA.hospitals, NH = HOSP.length, NT = DATA.n, STEP = DATA.step;
const SLOT_NAMES = ["새벽", "오전", "오후", "저녁"];
const DOW = ["일", "월", "화", "수", "목", "금", "토"];
const RED = "#b42318", SIM = "#475467", BAND = "#d0d5dd", INK = "#1d2939", GREEN = "#067647", AVAIL = "#c3c9d2";
const $ = (id) => document.getElementById(id);

// ---- 시각 ------------------------------------------------------------------------
const [dPart, tPart] = DATA.start.split("T");
const [Y0, M0, D0] = dPart.split("-").map(Number);
const [h0, m0] = tPart.split(":").map(Number);
const BASE_MS = Date.UTC(Y0, M0 - 1, D0);
const START_DOW = new Date(BASE_MS).getUTCDay();
const DAY = new Int16Array(NT), HOUR = new Uint8Array(NT), SLOT = new Uint8Array(NT), WKEND = new Uint8Array(NT);
for (let t = 0; t < NT; t++) {
  const min = h0 * 60 + m0 + t * STEP, day = Math.floor(min / 1440), hour = Math.floor((min % 1440) / 60);
  DAY[t] = day; HOUR[t] = hour;
  SLOT[t] = hour < 6 ? 0 : hour < 12 ? 1 : hour < 18 ? 2 : 3;
  const dow = (START_DOW + day) % 7;
  WKEND[t] = dow === 0 || dow === 6 ? 1 : 0;
}
const NDAY = DAY[NT - 1] + 1;
function timeLabel(t) {
  const d = new Date(BASE_MS + (h0 * 60 + m0 + t * STEP) * 60000);
  const p2 = (v) => String(v).padStart(2, "0");
  return `${d.getUTCMonth() + 1}/${d.getUTCDate()}(${DOW[d.getUTCDay()]}) ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
}
const dayLabel = (day) => {
  const d = new Date(BASE_MS + day * 86400000);
  return `${d.getUTCMonth() + 1}/${d.getUTCDate()}(${DOW[d.getUTCDay()]})`;
};

// ---- 실제 타임라인 ----------------------------------------------------------------
const REAL = HOSP.map((h) => {
  const a = new Int8Array(NT);
  for (let t = 0; t < NT; t++) a[t] = h.timeline[t] === "." ? -1 : h.timeline[t] === "1" ? 1 : 0;
  return a;
});
const meanOf = (arrs) => {
  let n = 0, f = 0;
  for (const a of arrs) for (const v of a) if (v >= 0) { n++; f += v; }
  return f / n;
};
const P_REAL = REAL.map((a) => meanOf([a]));
const P_ALL = meanOf(REAL);
const CAND = [...Array(NH).keys()].filter((h) => P_REAL[h] > 0.01);
const SHOW = [...CAND].sort((a, b) => P_REAL[b] - P_REAL[a]);
const completeRows = (hs) => {
  const rows = [];
  for (let t = 0; t < NT; t++) if (hs.every((h) => REAL[h][t] >= 0)) rows.push(t);
  return rows;
};
const ROWS_ALL = completeRows([...Array(NH).keys()]);
const ROWS_CAND = completeRows(CAND);
// 예시로 보여 줄 두 병원 (1차 발표와 같은 쌍)
const findH = (id, word) => Math.max(0, HOSP.findIndex((h) => h.id === id || h.name.includes(word)));
const PAIR = [findH("A1100015", "강남세브란스"), findH("A1100043", "강동경희")];

// ---- 꽉 찬 구간 --------------------------------------------------------------------
function runsOf(S) {
  const len = [], start = [];
  for (const a of S) {
    let k = 1;
    while (k < NT) {
      if (a[k] === 1 && a[k - 1] === 0) {
        const st = k;
        while (k < NT && a[k] === 1) k++;
        if (k < NT && a[k] === 0) { len.push((k - st) * STEP); start.push(st); }
      }
      k++;
    }
  }
  return { len, start };
}
const DSLOT = (() => {     // 시간대별 평균 길이 / 전체 평균
  const r = runsOf(REAL), sum = [0, 0, 0, 0], n = [0, 0, 0, 0];
  r.len.forEach((v, i) => { sum[SLOT[r.start[i]]] += v; n[SLOT[r.start[i]]]++; });
  const all = r.len.reduce((a, b) => a + b, 0) / r.len.length;
  return sum.map((s, k) => (n[k] ? s / n[k] / all : 1));
})();

// ---- 난수 --------------------------------------------------------------------------
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  const u = 1 - r(), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function poissonInv(u, lam) {   // 균등난수 u 하나로 Poisson(lam) 뽑기 (CDF 역함수)
  if (lam <= 0) return 0;
  let k = 0, p = Math.exp(-lam), F = p;
  while (u > F && k < 60) { k++; p *= lam / k; F += p; }
  return k;
}

// ---- 공통 엔진 ---------------------------------------------------------------------
const cfg = { level: 3, sigma: 0.3, reps: 20, durHosp: true, durSlot: false, mix: false, pi: 0.6, ms: 10 };

function pTable(h, level) {   // 길이 8: [평일 새벽..저녁, 주말 새벽..저녁]
  if (level === 1) return Array(8).fill(P_ALL);
  if (level === 2) return Array(8).fill(P_REAL[h]);
  return [...HOSP[h].P[0], ...HOSP[h].P[1]].map((v) => (v == null || isNaN(v) ? P_REAL[h] : v));
}

function simulate(c, seed) {
  const r = rng(seed);
  const L = new Float64Array(NDAY);
  for (let d = 0; d < NDAY; d++) L[d] = Math.max(0.05, 1 + c.sigma * gauss(r));
  const out = [];
  for (let h = 0; h < NH; h++) {
    const P = pTable(h, c.level);
    const Dm = c.durHosp ? HOSP[h].D : DATA.D_all;
    const ms = Math.min(c.ms, Dm);
    const ml = Math.max(ms, (Dm - c.pi * ms) / (1 - c.pi));
    const s = new Int8Array(NT), real = REAL[h];
    let x = r() < P[WKEND[0] * 4 + SLOT[0]] ? 1 : 0, short = 0;
    for (let t = 0; t < NT; t++) {
      const k = SLOT[t], f = c.durSlot ? DSLOT[k] : 1;
      if (x) {
        const mean = c.mix ? (short ? ms : ml) : Dm;
        if (r() < Math.min(1, STEP / (mean * f))) x = 0;
      } else {
        const p = Math.min(0.95, P[WKEND[t] * 4 + k] * L[DAY[t]]);
        if (r() < Math.min(1, (STEP / (Dm * f)) * p / (1 - p))) {
          x = 1;
          short = c.mix && r() < c.pi ? 1 : 0;
        }
      }
      s[t] = real[t] < 0 ? -1 : x;
    }
    out.push(s);
  }
  return out;
}

// ---- 진짜·가짜에 똑같이 쓰는 통계 ---------------------------------------------------
const D_GRID = Array.from({ length: 73 }, (_, i) => i * 5);        // 0..360분
const WAIT_S = [0, 30, 60, 120];
const NW = Math.ceil(ROWS_CAND.length / 32);
function popcnt(x) {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24);
}
function esum(p, k) {
  const e = new Float64Array(k + 1); e[0] = 1;
  for (const v of p) for (let j = k; j >= 1; j--) e[j] += v * e[j - 1];
  return e[k];
}
function groupRatios(S) {   // k곳 묶음 전부: 실제 동시에 꽉 참 / 따로 논다고 칠 때
  const NR = ROWS_CAND.length;
  const B = CAND.map((h) => {
    const b = new Uint32Array(NW), a = S[h];
    ROWS_CAND.forEach((t, i) => { if (a[t] === 1) b[i >> 5] |= 1 << (i & 31); });
    return b;
  });
  const p = B.map((b) => b.reduce((s, w) => s + popcnt(w), 0) / NR);
  const t2 = new Uint32Array(NW), t3 = new Uint32Array(NW);
  let J2 = 0, J3 = 0, J4 = 0;
  const m = B.length;
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
    let c2 = 0;
    for (let w = 0; w < NW; w++) { t2[w] = B[i][w] & B[j][w]; c2 += popcnt(t2[w]); }
    if (!c2) continue;
    J2 += c2;
    for (let l = j + 1; l < m; l++) {
      let c3 = 0;
      for (let w = 0; w < NW; w++) { t3[w] = t2[w] & B[l][w]; c3 += popcnt(t3[w]); }
      if (!c3) continue;
      J3 += c3;
      for (let q = l + 1; q < m; q++) {
        let c4 = 0;
        for (let w = 0; w < NW; w++) c4 += popcnt(t3[w] & B[q][w]);
        J4 += c4;
      }
    }
  }
  return [J2, J3, J4].map((J, i) => J / NR / esum(p, i + 2));
}

function durMetrics(S) {
  const M = {};
  const D = runsOf(S).len.sort((a, b) => a - b), nd = D.length;
  const mean = D.reduce((a, b) => a + b, 0) / nd;
  M.dMean = mean;
  M.dMedian = nd % 2 ? D[(nd - 1) / 2] : (D[nd / 2 - 1] + D[nd / 2]) / 2;
  M.dSd = Math.sqrt(D.reduce((s, v) => s + (v - mean) ** 2, 0) / (nd - 1));
  const gt = (x) => { let lo = 0, hi = nd; while (lo < hi) { const md = (lo + hi) >> 1; if (D[md] > x) hi = md; else lo = md + 1; } return nd - lo; };
  M.more30 = gt(30) / nd;
  M.more60at30 = gt(60) / gt(30);
  M.more150at120 = gt(150) / Math.max(1, gt(120));
  M.surv = D_GRID.map((d) => gt(d) / nd);
  M.wait = WAIT_S.map((s) => {   // 이미 s분째 꽉 차 있을 때 앞으로 더 갈 시간의 평균
    let c = 0, sum = 0;
    for (let i = nd - 1; i >= 0 && D[i] > s; i--) { c++; sum += D[i] - s; }
    return c >= 5 ? sum / c : NaN;
  });
  const lag = 15 / STEP;
  let rn = 0, rk = 0, fn = 0, fk = 0;
  for (const a of S) for (let t = 0; t + lag < NT; t++) {
    const x = a[t], y = a[t + lag];
    if (x < 0 || y < 0) continue;
    if (x === 1) { rn++; if (y === 0) rk++; } else { fn++; if (y === 1) fk++; }
  }
  M.recover15 = rk / rn; M.fill15 = fk / fn;
  return M;
}

function metrics(S) {
  const M = durMetrics(S);
  let n = 0, f = 0;
  const sn = [0, 0, 0, 0], sf = [0, 0, 0, 0], wn = [0, 0], wf = [0, 0];
  const dn = new Float64Array(NDAY), df = new Float64Array(NDAY);
  for (const a of S) for (let t = 0; t < NT; t++) {
    const v = a[t];
    if (v < 0) continue;
    n++; f += v; sn[SLOT[t]]++; sf[SLOT[t]] += v; wn[WKEND[t]]++; wf[WKEND[t]] += v; dn[DAY[t]]++; df[DAY[t]] += v;
  }
  M.full = f / n;
  for (let k = 0; k < 4; k++) M["slot" + k] = sf[k] / sn[k];
  M.weekday = wf[0] / wn[0]; M.weekend = wf[1] / wn[1];
  const dr = [];
  for (let d = 0; d < NDAY; d++) if (dn[d] > NH * 24) dr.push(df[d] / dn[d]);
  const dm = dr.reduce((a, b) => a + b, 0) / dr.length;
  M.daySd = Math.sqrt(dr.reduce((s, v) => s + (v - dm) ** 2, 0) / (dr.length - 1));

  const hist = new Float64Array(NH + 1);
  let cs = 0, cs2 = 0;
  for (const t of ROWS_ALL) {
    let c = 0;
    for (let h = 0; h < NH; h++) c += S[h][t];
    hist[c]++; cs += c; cs2 += c * c;
  }
  const nr = ROWS_ALL.length;
  M.cHist = Array.from(hist, (v) => v / nr);
  M.cMean = cs / nr; M.cVar = cs2 / nr - M.cMean ** 2;
  [M.ratio2, M.ratio3, M.ratio4] = groupRatios(S);

  const [a, b] = PAIR;
  let pn = 0, pk = 0;
  for (let t = 0; t < NT; t++) if (S[a][t] >= 0 && S[b][t] >= 0) { pn++; if (S[a][t] === 1 && S[b][t] === 1) pk++; }
  M.pair = pk / pn;

  M.hourly = [];   // 1시간마다 꽉 찬 병원 수 평균 (그래프용)
  for (let t0 = 0; t0 < NT; t0 += 12) {
    let c = 0, k = 0;
    for (let t = t0; t < Math.min(NT, t0 + 12); t++) {
      let cnt = 0, obs = 0;
      for (let h = 0; h < NH; h++) { const v = S[h][t]; if (v >= 0) { obs++; cnt += v; } }
      if (obs > NH * 0.8) { c += (cnt * NH) / obs; k++; }
    }
    M.hourly.push(k ? c / k : NaN);
  }
  return M;
}

function band(vals) {
  vals = vals.filter((v) => v != null && !isNaN(v));
  const n = vals.length, m = vals.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / (n - 1)) : 0;
  return { mean: m, sd, lo: m - 2 * sd, hi: m + 2 * sd };
}
const verdict = (real, b) => Math.abs(real - b.mean) <= Math.max(2 * b.sd, Math.abs(b.mean) * 1e-6);
const simVals = (key) => state.sims.map((s) => s.m[key]);
const simBand = (key) => band(simVals(key));
const okKey = (key) => verdict(REAL_M[key], simBand(key));

// ---- 상태 / 실행 -------------------------------------------------------------------
const state = { seed: 1, sims: [], token: 0, tab: "engine", week: 1, q2cache: null, t: 0, playing: false };
const REAL_M = metrics(REAL);

function runEngine() {
  const token = ++state.token, sims = [];
  const c = { ...cfg }, n = cfg.reps, seed = state.seed;
  $("reroll").disabled = true;
  const step = (i) => {
    if (token !== state.token) return;
    if (i === n) {
      state.sims = sims; state.q2cache = null;
      $("progress").textContent = `가짜 서울 ${n}개 완성`;
      $("reroll").disabled = false;
      render();
      return;
    }
    $("progress").textContent = `가짜 서울 만드는 중… ${i + 1} / ${n}`;
    const S = simulate(c, seed + i * 7919);
    sims.push({ S, m: metrics(S) });
    setTimeout(() => step(i + 1), 0);
  };
  setTimeout(() => step(0), 10);
}
let debounce = 0;
const rerun = () => { clearTimeout(debounce); debounce = setTimeout(runEngine, 250); };

// ---- 그리기 도구 --------------------------------------------------------------------
const pct = (v, d = 1) => (v == null || isNaN(v) ? "–" : (v * 100).toFixed(d) + "%");
const fixed = (d, unit = "") => (v) => (v == null || isNaN(v) ? "–" : v.toFixed(d) + unit);
const minText = (v) => (v == null || isNaN(v) ? "–" : v >= 90 ? `${Math.floor(v / 60)}시간 ${Math.round(v % 60)}분` : `${Math.round(v)}분`);
const badge = (good) => `<span class="verdict ${good ? "ok" : "bad"}">${good ? "비슷함" : "다름"}</span>`;
const LEGEND_STD = [{ label: "진짜", color: RED }, { label: "가짜 (평균)", color: SIM }, { label: "가짜가 흔들리는 범위 (± 2σ)", color: BAND, box: true }];

function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}

/** 막대 목록 (HTML). rows: {group} 또는 {label, value, color, text, mark} */
function bars(rows, max) {
  let s = `<div class="bars">`;
  for (const r of rows) {
    if (r.group) { s += `<div class="grp">${r.group}</div>`; continue; }
    const w = Math.max(0, Math.min(100, (r.value / max) * 100));
    const mk = r.mark != null ? `<span class="mark" style="left:${Math.min(100, (r.mark / max) * 100)}%"></span>` : "";
    s += `<div class="row"><span class="lab">${r.label}</span><span class="track"><span class="fill" style="display:block;width:${isNaN(w) ? 0 : w}%;background:${r.color}"></span>${mk}</span><span class="val" style="color:${r.color === BAND ? SIM : r.color}">${r.text}</span></div>`;
  }
  return s + `</div>`;
}

function chart(el, o) {
  const W = Math.max(340, el.clientWidth || 640), H = o.h || 260, m = { l: 56, r: 16, t: 12, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const log = o.y.log, ty = (v) => (log ? Math.log10(Math.max(v, o.y.min)) : v);
  const y0 = ty(o.y.min), y1 = ty(o.y.max);
  const X = (v) => m.l + ((v - o.x.min) / (o.x.max - o.x.min)) * iw;
  const Y = (v) => m.t + ih - ((Math.min(Math.max(ty(v), y0), y1) - y0) / (y1 - y0)) * ih;
  const fx = o.x.fmt || ((v) => v), fy = o.y.fmt || ((v) => v);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
  s += `<g class="grid">`;
  for (const v of o.y.ticks) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
  s += `</g>`;
  for (const v of o.y.ticks) s += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${fy(v)}</text>`;
  for (const v of o.x.ticks) s += `<text x="${X(v)}" y="${H - m.b + 18}" text-anchor="middle">${fx(v)}</text>`;
  s += `<g class="axis"><line x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/></g>`;
  if (o.x.label) s += `<text class="lbl" x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${o.x.label}</text>`;
  if (o.y.label) s += `<text class="lbl" transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${o.y.label}</text>`;
  const ok = (v) => v != null && !isNaN(v);
  for (const se of o.series) {
    if (se.type === "band") {
      const idx = se.xs.map((_, i) => i).filter((i) => ok(se.lo[i]) && ok(se.hi[i]));
      if (!idx.length) continue;
      const pts = idx.map((i) => `${X(se.xs[i])},${Y(se.hi[i])}`).concat(idx.reverse().map((i) => `${X(se.xs[i])},${Y(se.lo[i])}`));
      s += `<polygon points="${pts.join(" ")}" fill="${se.color}" opacity="${se.opacity ?? 0.55}"/>`;
    } else if (se.type === "line") {
      let d = "", pen = false;
      se.xs.forEach((x, i) => {
        if (!ok(se.ys[i])) { pen = false; return; }
        d += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(se.ys[i]).toFixed(1)}`; pen = true;
      });
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round"/>`;
    } else if (se.type === "points") {
      se.xs.forEach((x, i) => { if (ok(se.ys[i])) s += `<circle cx="${X(x)}" cy="${Y(se.ys[i])}" r="${se.r || 4}" fill="${se.color}"><title>${se.tip ? se.tip(i) : ""}</title></circle>`; });
    } else if (se.type === "bars") {
      const bw = (se.width / (o.x.max - o.x.min)) * iw;
      se.xs.forEach((x, i) => {
        if (!ok(se.ys[i])) return;
        const cx = X(x + (se.offset || 0)), top = Y(se.ys[i]), base = Y(log ? o.y.min : Math.max(0, o.y.min));
        s += `<rect x="${cx - bw / 2}" y="${Math.min(top, base)}" width="${bw}" height="${Math.abs(base - top)}" fill="${se.color}" rx="1.5"><title>${se.tip ? se.tip(i) : ""}</title></rect>`;
      });
    } else if (se.type === "err") {
      se.xs.forEach((x, i) => {
        if (!ok(se.lo[i])) return;
        const cx = X(x + (se.offset || 0));
        s += `<line x1="${cx}" x2="${cx}" y1="${Y(se.lo[i])}" y2="${Y(se.hi[i])}" stroke="${se.color}" stroke-width="1.4"/>`;
      });
    } else if (se.type === "hline") {
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(se.y)}" y2="${Y(se.y)}" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""}/>`;
      if (se.label) s += `<text x="${W - m.r}" y="${Y(se.y) - 6}" text-anchor="end" style="fill:${se.color};font-weight:600">${se.label}</text>`;
    }
  }
  s += `</svg>`;
  if (o.legend) {
    s += `<div class="legend">${o.legend.map((l) => `<span><i class="${l.box ? "box" : ""}" style="background:${l.color}"></i>${l.label}</span>`).join("")}</div>`;
  }
  el.innerHTML = s;
}

function compareTable(rows) {
  let s = `<table class="tbl"><tr><th>무엇을</th><th class="num">진짜</th><th class="num">가짜 (평균 ± 2σ)</th><th class="num">판정</th></tr>`;
  for (const r of rows) {
    if (r.group) { s += `<tr class="group"><td colspan="4">${r.group}</td></tr>`; continue; }
    const b = band(r.sim);
    s += `<tr><td>${r.label}</td><td class="num realv">${r.fmt(r.real)}</td><td class="num">${r.fmt(b.mean)} <span style="color:var(--muted)">± ${r.fmt(2 * b.sd).replace(/^-/, "")}</span></td>` +
      `<td class="num">${badge(verdict(r.real, b))}</td></tr>`;
  }
  return s + `</table>`;
}

// ---- 전체 재현 탭: 재생 지도 ---------------------------------------------------------
const MAP = (() => {
  const W = 520, H = 430, pad = 22, k = Math.cos((37.55 * Math.PI) / 180);
  const la = HOSP.map((h) => h.lat), lo = HOSP.map((h) => h.lon);
  const la0 = Math.min(...la), la1 = Math.max(...la), lo0 = Math.min(...lo), lo1 = Math.max(...lo);
  const sc = Math.min((W - 2 * pad) / ((lo1 - lo0) * k), (H - 2 * pad) / (la1 - la0));
  const ox = (W - (lo1 - lo0) * k * sc) / 2, oy = (H - (la1 - la0) * sc) / 2;
  return { W, H, x: (lon) => ox + (lon - lo0) * k * sc, y: (lat) => H - oy - (lat - la0) * sc };
})();
const GU_SHOW = ["종로구", "마포구", "강남구", "송파구", "노원구", "강서구", "관악구", "은평구", "성동구", "구로구"];
const mapDots = {};
function buildMap(id) {
  let s = `<svg viewBox="0 0 ${MAP.W} ${MAP.H}"><rect width="${MAP.W}" height="${MAP.H}" rx="8" fill="#fafafa"/>`;
  for (const o of DATA.origins) if (GU_SHOW.includes(o.name)) {
    s += `<text x="${MAP.x(o.lon)}" y="${MAP.y(o.lat)}" text-anchor="middle" font-size="11" fill="#b8bec9">${o.name}</text>`;
  }
  HOSP.forEach((h, i) => { s += `<circle data-i="${i}" cx="${MAP.x(h.lon)}" cy="${MAP.y(h.lat)}" r="5.5"><title>${h.name}</title></circle>`; });
  $(id).innerHTML = s + `</svg>`;
  mapDots[id] = [...$(id).querySelectorAll("circle")];
}
function paintMap(id, S, t) {
  let cnt = 0;
  mapDots[id].forEach((c, i) => {
    const v = S ? S[i][t] : -1;
    if (v === 1) cnt++;
    c.setAttribute("fill", v === 1 ? RED : v === 0 ? AVAIL : "#fff");
    c.setAttribute("stroke", v < 0 ? AVAIL : "none");
    c.setAttribute("r", v === 1 ? 7 : 5);
  });
  return cnt;
}
function showTime(t) {
  state.t = Math.max(0, Math.min(NT - 1, Math.round(t)));
  const tt = state.t;
  $("clock").textContent = timeLabel(tt);
  const missing = REAL[0][tt] < 0 && REAL[1][tt] < 0;
  const cr = paintMap("mapReal", REAL, tt);
  $("cntReal").textContent = missing ? "기록 없음" : `꽉 찬 곳 ${cr}곳`;
  if (state.sims.length) {
    const cs = paintMap("mapSim", state.sims[0].S, tt);
    $("cntSim").textContent = missing ? "기록 없음" : `꽉 찬 곳 ${cs}곳`;
  }
  $("scrub").value = tt;
  const ph = $("playhead");
  if (ph) { const x = state.sparkX(tt); ph.setAttribute("x1", x); ph.setAttribute("x2", x); }
}
function renderSpark() {
  const el = $("spark"), W = Math.max(340, el.clientWidth || 800), H = 96, l = 30, r = 8, top = 6, bot = 20;
  const real = REAL_M.hourly, sim = state.sims.length ? state.sims[0].m.hourly : [];
  const n = real.length, ymax = Math.max(...real.filter((v) => !isNaN(v)), ...sim.filter((v) => !isNaN(v))) * 1.1;
  const X = (i) => l + (i / (n - 1)) * (W - l - r), Y = (v) => top + (1 - v / ymax) * (H - top - bot);
  state.sparkX = (t) => X(t / 12);
  const path = (arr) => { let d = "", pen = false; arr.forEach((v, i) => { if (isNaN(v)) { pen = false; return; } d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; }); return d; };
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block">`;
  for (const v of niceTicks(0, ymax, 2)) s += `<line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#f2f4f7"/><text x="${l - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="10" fill="#667085">${v}</text>`;
  for (let d = 0; d < NDAY; d++) {
    const t = Math.max(0, d * 288 - (h0 * 12 + m0 / 5));
    if (t >= NT) break;
    const dow = (START_DOW + d) % 7;
    if (dow === 0 || dow === 6) s += `<rect x="${X(t / 12)}" y="${top}" width="${X((t + 288) / 12) - X(t / 12)}" height="${H - top - bot}" fill="#f6f7f9"/>`;
    if (d % 2 === 0) s += `<text x="${X(t / 12) + 2}" y="${H - 5}" font-size="10" fill="#667085">${dayLabel(d)}</text>`;
  }
  s += `<path d="${path(sim)}" fill="none" stroke="${SIM}" stroke-width="1.2" opacity=".8"/>`;
  s += `<path d="${path(real)}" fill="none" stroke="${RED}" stroke-width="1.4"/>`;
  s += `<line id="playhead" y1="${top}" y2="${H - bot}" stroke="${INK}" stroke-width="1.5"/>`;
  el.innerHTML = s + `</svg><div class="legend"><span><i style="background:${RED}"></i>진짜</span><span><i style="background:${SIM}"></i>가짜 (1번째)</span><span><i class="box" style="background:#f0f1f4"></i>주말</span></div>`;
  el.querySelector("svg").onclick = (e) => {
    const rc = e.currentTarget.getBoundingClientRect();
    const i = (((e.clientX - rc.left) / rc.width) * W - l) / (W - l - r) * (n - 1);
    showTime(i * 12);
  };
}
let lastFrame = 0;
function tick(ts) {
  if (!state.playing) return;
  const dt = lastFrame ? (ts - lastFrame) / 1000 : 0;
  lastFrame = ts;
  state.tf = (state.tf ?? state.t) + dt * +$("speed").value;
  if (state.tf >= NT - 1) state.tf = 0;
  showTime(state.tf);
  requestAnimationFrame(tick);
}
function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚ 멈춤" : "▶ 재생";
  state.tf = state.t; lastFrame = 0;
  if (state.playing) requestAnimationFrame(tick);
}

// ---- 전체 재현 탭: 답 · 체크 -------------------------------------------------------
function renderEngine() {
  renderSpark();
  showTime(state.t);
  const [a, b] = PAIR, ind = P_REAL[a] * P_REAL[b];
  const pr = REAL_M.pair, ps = simBand("pair"), r4 = simBand("ratio4");
  const pairOk = verdict(pr, ps), groupOk = ["ratio2", "ratio3", "ratio4", "cVar"].every(okKey);
  let verdictText;
  if (pairOk && groupOk) verdictText = `지금 설정으로 만든 가짜 서울도 진짜처럼 같이 붐빕니다. 이 설정이 현실을 잘 설명한다는 뜻이에요.`;
  else if (r4.mean < REAL_M.ratio4) verdictText = `지금 설정의 가짜 서울은 진짜보다 덜 같이 붐빕니다. 왼쪽 '날마다 붐비는 정도 차이'를 키워 보세요.`;
  else verdictText = `지금 설정의 가짜 서울은 진짜보다 더 같이 붐빕니다. 왼쪽 '날마다 붐비는 정도 차이'를 줄여 보세요.`;
  const nm = (h) => HOSP[h].name.replace(/^연세대 /, "");
  const mx = Math.max(pr, ps.hi, ind) * 1.15;
  $("engineAnswer").innerHTML =
    `<p class="q">질문: 서울 응급실들은 따로따로 붐빌까, 같이 붐빌까?</p>` +
    `<p class="a">같이 붐빈다. ${verdictText}</p>` +
    `<div class="bars">${bars([
      { group: `${nm(a)}과 ${nm(b)}이 동시에 꽉 찰 확률` },
      { label: "진짜", value: pr, color: RED, text: pct(pr) },
      { label: "가짜 (지금 설정)", value: ps.mean, color: SIM, text: pct(ps.mean) },
      { label: "서로 따로 논다고 치면", value: ind, color: BAND, text: pct(ind) },
    ], mx)}</div>` +
    `<p class="why">아무 병원 4곳을 골라도, '동시에 꽉 찰 확률'이 따로 논다고 칠 때보다 진짜는 평균 <b>${REAL_M.ratio4.toFixed(1)}배</b>, 가짜는 <b>${r4.mean.toFixed(1)}배</b> 높습니다.</p>`;

  const pp = (k) => pct(REAL_M[k], 0), sp = (k) => pct(simBand(k).mean, 0);
  const cards = [
    { title: "하루 리듬 (오후에 가장 붐빔)", ok: [0, 1, 2, 3].every((k) => okKey("slot" + k)),
      text: `꽉 찰 확률 새벽 → 오후: 진짜 ${pp("slot0")} → ${pp("slot2")}, 가짜 ${sp("slot0")} → ${sp("slot2")}` },
    { title: "평일이 주말보다 붐빔", ok: okKey("weekday") && okKey("weekend"),
      text: `평일 / 주말: 진짜 ${pp("weekday")} / ${pp("weekend")}, 가짜 ${sp("weekday")} / ${sp("weekend")}` },
    { title: "여러 병원이 같이 꽉 참", ok: groupOk,
      text: `동시에 꽉 찬 병원 수의 흩어짐(분산): 진짜 ${REAL_M.cVar.toFixed(1)}, 가짜 ${simBand("cVar").mean.toFixed(1)}` },
    { title: "꽉 찬 상태가 풀리는 속도", ok: okKey("dMedian") && okKey("recover15"),
      text: `절반이 풀리는 시간: 진짜 ${Math.round(REAL_M.dMedian)}분, 가짜 ${Math.round(simBand("dMedian").mean)}분. ${okKey("dMedian") ? "" : "→ Q3 탭에서 고쳐 보기"}` },
  ];
  $("engineChecks").innerHTML = cards.map((c) => `<div class="ck"><b>${c.title}</b>${badge(c.ok)}<p>${c.text}</p></div>`).join("");
  renderDetails();
  renderSweep();
}

// 자세히 보기
const ENGINE_ROWS = [
  { group: "꽉 찰 확률" },
  { key: "full", label: "전체", fmt: pct },
  { key: "weekday", label: "평일", fmt: pct },
  { key: "weekend", label: "주말", fmt: pct },
  ...[0, 1, 2, 3].map((k) => ({ key: "slot" + k, label: SLOT_NAMES[k], fmt: pct })),
  { group: "여러 병원이 같이 꽉 차는 정도" },
  { key: "pair", label: "강남세브란스 · 강동경희대 동시에", fmt: pct },
  { key: "ratio2", label: "아무 2곳 동시에 (따로 논다고 칠 때의 몇 배)", fmt: fixed(2, "배") },
  { key: "ratio3", label: "아무 3곳 동시에 (〃)", fmt: fixed(2, "배") },
  { key: "ratio4", label: "아무 4곳 동시에 (〃)", fmt: fixed(2, "배") },
  { key: "cVar", label: "동시에 꽉 찬 병원 수의 분산", fmt: fixed(2) },
  { key: "daySd", label: "하루 꽉 찰 확률의 표준편차", fmt: (v) => pct(v, 2) },
  { group: "시간이 흐르면 (Q1 · Q3와 연결)" },
  { key: "recover15", label: "꽉 참 → 15분 뒤 자리 생김", fmt: pct },
  { key: "fill15", label: "자리 있음 → 15분 뒤 꽉 참", fmt: pct },
  { key: "dMedian", label: "꽉 찬 상태 길이 (중앙값)", fmt: fixed(0, "분") },
  { key: "dMean", label: "꽉 찬 상태 길이 (평균)", fmt: fixed(0, "분") },
];
function renderDetails() {
  if (!document.querySelector("#v-engine details:last-of-type").open) return;
  $("engineTable").innerHTML = compareTable(ENGINE_ROWS.map((r) => (r.group ? r : { ...r, real: REAL_M[r.key], sim: simVals(r.key) })));
  const xs = [...Array(21).keys()];
  const sh = xs.map((c) => band(state.sims.map((s) => s.m.cHist[c] || 0)));
  const ymax = Math.max(...xs.map((c) => Math.max(REAL_M.cHist[c], sh[c].hi))) * 1.1;
  chart($("chartC"), {
    h: 230, x: { min: -0.7, max: 20.7, ticks: [0, 5, 10, 15, 20], fmt: (v) => v + "곳", label: "동시에 꽉 찬 병원 수" },
    y: { min: 0, max: ymax, ticks: niceTicks(0, ymax, 4), fmt: (v) => pct(v, 0) },
    series: [
      { type: "bars", xs, ys: xs.map((c) => REAL_M.cHist[c]), color: RED, width: 0.4, offset: -0.21 },
      { type: "bars", xs, ys: sh.map((b) => b.mean), color: SIM, width: 0.4, offset: 0.21 },
      { type: "err", xs, lo: sh.map((b) => Math.max(0, b.lo)), hi: sh.map((b) => b.hi), color: INK, offset: 0.21 },
    ],
    legend: [{ label: "진짜", color: RED, box: true }, { label: "가짜 (평균 ± 2σ)", color: SIM, box: true }],
  });
  renderTimeline();
}
const WEEKS = (() => { const w = []; for (let d = 0; d < NDAY; d += 7) w.push(d); return w; })();
function weekRange(i) {
  const d0 = WEEKS[i], d1 = d0 + 7;
  let a = NT, b = 0;
  for (let t = 0; t < NT; t++) if (DAY[t] >= d0 && DAY[t] < d1) { a = Math.min(a, t); b = t; }
  return [a, b + 1];
}
function drawTL(canvas, S, a, b) {
  const w = b - a, h = SHOW.length;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h);
  SHOW.forEach((hh, row) => {
    for (let t = a; t < b; t++) {
      const v = S[hh][t], i = (row * w + (t - a)) * 4;
      const c = v === 1 ? [180, 35, 24] : v === 0 ? [233, 236, 241] : [255, 255, 255];
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
    }
  });
  ctx.putImageData(img, 0, 0);
  canvas._tl = { S, a, b };
}
function renderTimeline() {
  const [a, b] = weekRange(state.week);
  drawTL($("tlReal"), REAL, a, b);
  if (state.sims.length) drawTL($("tlSim"), state.sims[0].S, a, b);
  let ax = "";
  for (let t = a; t < b; t++) if (t === a || DAY[t] !== DAY[t - 1]) ax += `<span style="left:${((t - a) / (b - a)) * 100}%">${dayLabel(DAY[t])}</span>`;
  $("tlAxis").innerHTML = ax;
}
function hookTip(canvas, who) {
  canvas.addEventListener("mousemove", (e) => {
    const tl = canvas._tl;
    if (!tl) return;
    const r = canvas.getBoundingClientRect();
    const t = tl.a + Math.floor(((e.clientX - r.left) / r.width) * (tl.b - tl.a));
    const row = Math.floor(((e.clientY - r.top) / r.height) * SHOW.length);
    const h = SHOW[Math.min(row, SHOW.length - 1)], v = tl.S[h][t];
    const tip = $("tip");
    tip.style.display = "block";
    tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
    tip.textContent = `${who} · ${HOSP[h].name} · ${timeLabel(t)} · ${v === 1 ? "꽉 참" : v === 0 ? "자리 있음" : "기록 없음"}`;
  });
  canvas.addEventListener("mouseleave", () => ($("tip").style.display = "none"));
}

// 날마다 붐비는 정도 차이 훑어보기
const SWEEP_S = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
let sweep = null;
function runSweep() {
  const c0 = { ...cfg }, res = [], N = 10, keys = ["ratio2", "ratio3", "ratio4", "cVar", "daySd"];
  $("sweepBtn").disabled = true;
  let i = 0, vals = [];
  const step = () => {
    if (i === SWEEP_S.length * N) {
      const score = res.map((r) => keys.reduce((s, k) => s + ((REAL_M[k] - r[k].mean) / (r[k].sd || 1e-9)) ** 2, 0));
      sweep = { res, best: SWEEP_S[score.indexOf(Math.min(...score))] };
      $("sweepBtn").disabled = false; $("sweepBtn").textContent = "다시 찾기";
      renderSweep();
      return;
    }
    const si = Math.floor(i / N), S = simulate({ ...c0, sigma: SWEEP_S[si] }, 100 + i);
    const g = groupRatios(S), m = metrics(S);
    vals.push({ ratio2: g[0], ratio3: g[1], ratio4: g[2], cVar: m.cVar, daySd: m.daySd });
    if (vals.length === N) {
      const o = {};
      for (const k of keys) o[k] = band(vals.map((v) => v[k]));
      res.push(o); vals = [];
    }
    $("sweepBtn").textContent = `찾는 중… ${SWEEP_S[si].toFixed(2)}`;
    i++;
    setTimeout(step, 0);
  };
  step();
}
function renderSweep() {
  const el = $("chartSweep");
  if (!sweep) { el.innerHTML = ""; return; }
  const ys = sweep.res.map((r) => r.ratio4);
  const ymax = Math.max(REAL_M.ratio4, ...ys.map((b) => b.hi)) * 1.08, ymin = Math.min(1, ...ys.map((b) => b.lo));
  chart(el, {
    h: 240, x: { min: 0, max: 0.6, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label: "날마다 붐비는 정도 차이 (σ)" },
    y: { min: ymin, max: ymax, ticks: niceTicks(ymin, ymax, 4), fmt: (v) => v.toFixed(1) + "배", label: "아무 4곳 동시에 꽉 참" },
    series: [
      { type: "band", xs: SWEEP_S, lo: ys.map((b) => b.lo), hi: ys.map((b) => b.hi), color: BAND },
      { type: "line", xs: SWEEP_S, ys: ys.map((b) => b.mean), color: SIM },
      { type: "points", xs: SWEEP_S, ys: ys.map((b) => b.mean), color: SIM, r: 3 },
      { type: "hline", y: REAL_M.ratio4, color: RED, label: `진짜 ${REAL_M.ratio4.toFixed(2)}배` },
    ],
    legend: LEGEND_STD,
  });
  el.insertAdjacentHTML("beforeend",
    `<p class="hint" style="margin-top:10px">0이면 '매일 똑같은 날'이라 같이 붐비는 정도가 부족하고, 너무 크면 지나칩니다. 가장 비슷한 값 ≈ <b style="color:var(--red)">${sweep.best.toFixed(2)}</b>
     <button class="btn ghost narrow" style="padding:3px 10px;margin-left:6px" id="useBest">이 값으로 설정</button></p>`);
  $("useBest").onclick = () => { $("sigma").value = sweep.best; onSigma(); };
}

// ---- Q1 ---------------------------------------------------------------------------
const q1 = { a: 0.34, b: 0, s: 0 };
const q1Real = () => { const v = $("q1slot").value; return (v === "all" ? DATA.q1.all : DATA.q1.slot[+v]).map((r) => r.p); };
const poisTail = (a, b) => { let s = 0, t = Math.exp(-a); for (let k = 0; k < b; k++) { s += t; t *= a / (k + 1); } return 1 - s; };
function q1Run(a, be, s, seed, N) {   // 15분 동안: 도착 I ~ Poisson(a·G), 퇴원 O ~ Poisson(be), G ~ N(1, s²)
  const r = rng(seed), cnt = [0, 0, 0, 0, 0];
  for (let i = 0; i < N; i++) {
    const G = Math.max(0, 1 + s * gauss(r));
    const net = poissonInv(r(), a * G) - poissonInv(r(), be);
    for (let b = 1; b <= 5; b++) if (net >= b) cnt[b - 1]++;
  }
  return cnt.map((c) => c / N);
}
function q1Reset() {
  q1.a = -Math.log(1 - q1Real()[0]); q1.b = 0; q1.s = 0;
  $("q1a").value = q1.a.toFixed(2); $("q1b").value = 0; $("q1s").value = 0;
}
function q1Auto() {
  $("q1auto").disabled = true; $("q1auto").textContent = "찾는 중…";
  setTimeout(() => {
    const real = q1Real(), N = 4000;
    let best = null;
    for (let be = 0; be <= 1.2001; be += 0.2) for (let s = 0; s <= 2.0001; s += 0.1) for (let a = 0.1; a <= 1.5; a += 0.02) {
      const p = q1Run(a, be, s, 77, N);
      let sc = 0;
      for (let i = 0; i < 4; i++) sc += (Math.log(p[i] + 2e-4) - Math.log(real[i] + 2e-4)) ** 2;
      if (!best || sc < best.sc) best = { sc, a, be, s };
    }
    $("q1a").value = best.a.toFixed(2); $("q1b").value = best.be.toFixed(2); $("q1s").value = best.s.toFixed(2);
    $("q1auto").disabled = false; $("q1auto").textContent = "실제와 가장 비슷한 값 자동으로 찾기";
    renderQ1();
  }, 30);
}
function renderQ1() {
  q1.a = +$("q1a").value; q1.b = +$("q1b").value; q1.s = +$("q1s").value;
  $("q1aV").textContent = q1.a.toFixed(2) + "명";
  $("q1bV").textContent = q1.b.toFixed(2) + "명";
  $("q1sV").textContent = q1.s.toFixed(2);
  $("q1sHelp").textContent = q1.s === 0 ? "σ₁. 0 = 15분마다 고르게 옴 (보통의 Poisson)"
    : `σ₁. 어떤 15분은 평소의 ${(1 + q1.s).toFixed(1)}배, 어떤 15분은 ${Math.max(0, 1 - q1.s).toFixed(1)}배 정도로 몰리거나 뜸함`;
  const real = q1Real(), bs = [1, 2, 3, 4, 5], reps = Math.min(cfg.reps, 20);
  const runs = [];
  for (let rep = 0; rep < reps; rep++) runs.push(q1Run(q1.a, q1.b, q1.s, 500 + rep, 20000));
  const bd = bs.map((_, i) => band(runs.map((r) => r[i])));
  const pois = bs.map((b) => poisTail(-Math.log(1 - real[0]), b));
  const oks = bs.map((_, i) => verdict(real[i], bd[i]));
  const nOk = oks.slice(0, 4).filter(Boolean).length, plain = q1.s === 0 && q1.b === 0;
  const slotTxt = $("q1slot").value === "all" ? "" : `(${$("q1slot").selectedOptions[0].text}) `;
  $("q1Answer").innerHTML =
    `<p class="q">질문: 지금 병상이 남아 있다고 나와도, 15분 뒤 도착하면 있을까?</p>` +
    `<p class="a">${slotTxt}병상이 1개 남았으면 15분 뒤 ${pct(real[0], 0)} 확률로 이미 꽉 차 있다. 2개면 ${pct(real[1], 0)}, 3개면 ${pct(real[2], 1)}.</p>` +
    `<p class="why">${plain
      ? `환자가 고르게 온다(Poisson)고만 보면 2개일 때 ${pct(pois[1], 1)}로, 진짜의 절반 정도밖에 안 나옵니다. 환자가 몰려서 온다는 뜻이에요. '환자가 몰려오는 정도'를 올려 보거나 자동으로 찾아 보세요.`
      : `지금 설정의 가짜 응급실은 병상 1~4개 중 <b>${nOk}개</b>에서 진짜와 비슷합니다.`}</p>`;
  const mx = Math.max(...real, ...bd.map((b) => b.mean)) * 1.1;
  const rows = [];
  bs.forEach((b, i) => {
    rows.push({ group: `남은 병상 ${b}개 ${oks[i] ? "" : "<span style='color:var(--red);font-weight:500'>· 다름</span>"}` });
    rows.push({ label: "진짜", value: real[i], color: RED, text: pct(real[i], b > 2 ? 2 : 1) });
    rows.push({ label: "가짜 (지금 설정)", value: bd[i].mean, color: SIM, text: pct(bd[i].mean, b > 2 ? 2 : 1) });
    if (!plain) rows.push({ label: "고르게 온다고 치면", value: pois[i], color: BAND, text: pct(pois[i], b > 2 ? 2 : 1) });
  });
  $("q1Bars").innerHTML = bars(rows, mx);
  const fl = 1e-4;
  chart($("chartQ1"), {
    h: 260, x: { min: 0.7, max: 5.3, ticks: bs, fmt: (v) => `${v}개`, label: "지금 남은 병상 수 b" },
    y: { min: fl, max: 1, log: true, ticks: [1e-4, 1e-3, 1e-2, 1e-1, 1], fmt: (v) => (v >= 0.01 ? pct(v, 0) : pct(v, v >= 0.001 ? 1 : 2)) },
    series: [
      { type: "band", xs: bs, lo: bd.map((b) => Math.max(fl, b.lo)), hi: bd.map((b) => Math.max(fl, b.hi)), color: BAND },
      { type: "line", xs: bs, ys: pois, color: SIM, dash: "4 4", width: 1.5 },
      { type: "line", xs: bs, ys: bd.map((b) => Math.max(fl, b.mean)), color: SIM },
      { type: "line", xs: bs, ys: real, color: RED, width: 1.5 },
      { type: "points", xs: bs, ys: real, color: RED, r: 5 },
    ],
    legend: [...LEGEND_STD, { label: "고르게 온다고 칠 때 (점선)", color: SIM }],
  });
  $("q1Table").innerHTML = compareTable(bs.map((b, i) => ({ label: `남은 병상 ${b}개`, real: real[i], sim: runs.map((r) => r[i]), fmt: (v) => pct(v, 2) })));
}

// ---- Q2 ---------------------------------------------------------------------------
const q2 = { origin: -1, strat: "near", H: DATA.handling, speed: DATA.speed, roll: 0 };
function hav(a1, o1, a2, o2) {
  const r = Math.PI / 180, dl = (a2 - a1) * r, dn = (o2 - o1) * r;
  const x = Math.sin(dl / 2) ** 2 + Math.cos(a1 * r) * Math.cos(a2 * r) * Math.sin(dn / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(x));
}
const KM_OH = DATA.origins.map((o) => HOSP.map((h) => hav(o.lat, o.lon, h.lat, h.lon)));
const KM_HH = HOSP.map((a) => HOSP.map((b) => hav(a.lat, a.lon, b.lat, b.lon)));
const NEAR = KM_OH.map((row) => [...row.keys()].sort((a, b) => row[a] - row[b]));
function perms(a) {
  if (a.length <= 1) return [a];
  return a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [x, ...p]));
}
function q2Setup() {
  const f = (DATA.road / q2.speed) * 60;
  const OM = KM_OH.map((r) => r.map((k) => k * f)), HM = KM_HH.map((r) => r.map((k) => k * f));
  const EXP = NEAR.map((near, o) => {   // 가까운 5곳 순서 중 E[T]가 가장 작은 것 (요일·시간대별)
    const out = [];
    for (let wk = 0; wk < 8; wk++) {
      const pf = (h) => { const v = HOSP[h].P[wk >> 2][wk & 3]; return v == null ? P_REAL[h] : v; };
      let best = null, bt = Infinity;
      for (const p of perms(near.slice(0, 5))) {
        let t = 0, surv = 1, prev = -1;
        for (const h of p) { t += surv * (prev < 0 ? OM[o][h] : q2.H + HM[prev][h]); surv *= pf(h); prev = h; }
        if (t < bt) { bt = t; best = p; }
      }
      out.push([...best, ...near.slice(5)]);
    }
    return out;
  });
  return { OM, HM, EXP };
}
function trip(S, o, t, strat, env, keep) {
  const ord = strat === "near" ? NEAR[o] : env.EXP[o][WKEND[t] * 4 + SLOT[t]];
  let clock = 0, prev = -1;
  const path = [];
  for (let v = 0; v < 10; v++) {
    const h = ord[v];
    clock += prev < 0 ? env.OM[o][h] : q2.H + env.HM[prev][h];
    const step = t + Math.round(clock / STEP);
    if (step >= NT) return null;
    const s = S[h][step];
    if (s < 0) return null;
    if (keep) path.push({ h, clock, full: s === 1 });
    if (s === 0) return { V: v + 1, T: clock, path };
    prev = h;
  }
  return { V: 11, T: clock, path };
}
function replay(S, origins, strat, env) {
  const dist = [0, 0, 0, 0, 0];
  let n = 0, sumT = 0;
  for (const o of origins) for (let t = 0; t < NT; t += 6) {
    const r = trip(S, o, t, strat, env);
    if (!r) continue;
    n++; sumT += r.T; dist[Math.min(r.V, 5) - 1]++;
  }
  return { dist: dist.map((d) => d / n), meanT: sumT / n, first: dist[0] / n, ge3: 1 - (dist[0] + dist[1]) / n, n };
}
function renderQ2() {
  if (!state.sims.length) return;
  $("q2hV").textContent = q2.H + "분";
  $("q2vV").textContent = q2.speed + " km/h";
  const key = [q2.origin, q2.H, q2.speed, state.token].join("|");
  if (!state.q2cache || state.q2cache.key !== key) {
    const env = q2Setup(), origins = q2.origin < 0 ? [...DATA.origins.keys()] : [q2.origin];
    const res = {};
    for (const st of ["near", "exp"]) res[st] = { real: replay(REAL, origins, st, env), sims: state.sims.map((s) => replay(s.S, origins, st, env)) };
    state.q2cache = { key, env, res };
  }
  const { res } = state.q2cache, R = res[q2.strat];
  const where = q2.origin < 0 ? "서울 어디서 출발하든" : `${DATA.origins[q2.origin].name}에서 출발하면`;
  const gain = res.near.real.meanT - res.exp.real.meanT, gainSim = band(res.near.sims.map((s) => s.meanT)).mean - band(res.exp.sims.map((s) => s.meanT)).mean;
  const allOk = ["first", "ge3", "meanT"].every((k) => verdict(R.real[k], band(R.sims.map((s) => s[k]))));
  $("q2Answer").innerHTML =
    `<p class="q">질문: 첫 병원이 꽉 찼으면, 몇 곳을 더 돌아야 할까?</p>` +
    `<p class="a">${where} 첫 병원에서 받아 줄 확률이 ${pct(res.near.real.first, 0)}. 두 곳 연속 거절당할 확률은 ${pct(res.near.real.ge3, 1)}이다.</p>` +
    `<p class="why">가까운 순서 대신 '꽉 찰 확률까지 따진 순서'로 가면 평균 <b>${gain.toFixed(1)}분</b> 빨라집니다 (가짜 서울에서는 ${gainSim.toFixed(1)}분). ` +
    `${allOk ? "가짜 서울도 진짜와 같은 결과를 내서, 우리 확률 모델로 Q2를 설명할 수 있어요." : "가짜 서울은 일부 지표에서 진짜와 다릅니다. 아래 표를 확인하세요."}</p>`;
  const xs = [1, 2, 3, 4, 5], sb = xs.map((_, i) => band(R.sims.map((s) => s.dist[i])));
  const ymax = Math.max(...R.real.dist, ...sb.map((b) => b.hi)) * 1.1;
  chart($("chartQ2"), {
    h: 230, x: { min: 0.4, max: 5.6, ticks: xs, fmt: (v) => (v === 5 ? "5번째 이후" : `${v}번째`) },
    y: { min: 0, max: ymax, ticks: niceTicks(0, ymax, 4), fmt: (v) => pct(v, 0) },
    series: [
      { type: "bars", xs, ys: R.real.dist, color: RED, width: 0.36, offset: -0.19, tip: (i) => `진짜: ${pct(R.real.dist[i], 2)}` },
      { type: "bars", xs, ys: sb.map((b) => b.mean), color: SIM, width: 0.36, offset: 0.19, tip: (i) => `가짜: ${pct(sb[i].mean, 2)}` },
      { type: "err", xs, lo: sb.map((b) => Math.max(0, b.lo)), hi: sb.map((b) => b.hi), color: INK, offset: 0.19 },
    ],
    legend: [{ label: "진짜", color: RED, box: true }, { label: "가짜 (평균 ± 2σ)", color: SIM, box: true }],
  });
  const row = (st, k, label, fmt) => ({ label, real: res[st].real[k], sim: res[st].sims.map((s) => s[k]), fmt });
  $("q2Table").innerHTML = compareTable([
    { group: "가까운 순서로 갈 때" },
    row("near", "first", "첫 병원에서 받아 줌", (v) => pct(v, 1)),
    row("near", "ge3", "두 곳 연속 거절", (v) => pct(v, 2)),
    row("near", "meanT", "병상까지 걸린 시간 (평균)", fixed(1, "분")),
    { group: "꽉 찰 확률까지 따진 순서로 갈 때" },
    row("exp", "first", "첫 병원에서 받아 줌", (v) => pct(v, 1)),
    row("exp", "meanT", "병상까지 걸린 시간 (평균)", fixed(1, "분")),
  ]);
  renderQ2Map();
}
function renderQ2Map() {
  const env = state.q2cache.env, S = state.sims[0].S;
  const r = rng(9000 + q2.roll);
  let o = q2.origin < 0 ? Math.floor(r() * DATA.origins.length) : q2.origin, res = null, t = 0;
  for (let k = 0; k < 2000 && !res; k++) {
    t = Math.floor(r() * (NT - 40));
    res = trip(S, o, t, q2.strat, env, true);
    if (res && k < 1500 && res.V === 1 && r() < 0.85) res = null;   // 대부분 1곳이라, 여러 곳 도는 장면이 주로 보이도록
  }
  if (!res) return;
  // 경로 주변만 확대 (최소 가로세로 약 3km)
  const W = 640, H = 470, pad = 40, k = Math.cos((37.55 * Math.PI) / 180);
  const O = DATA.origins[o];
  const lats = [O.lat, ...res.path.map((p) => HOSP[p.h].lat)], lons = [O.lon, ...res.path.map((p) => HOSP[p.h].lon)];
  const cla = (Math.min(...lats) + Math.max(...lats)) / 2, clo = (Math.min(...lons) + Math.max(...lons)) / 2;
  const span = Math.max(0.03, (Math.max(...lats) - Math.min(...lats)) * 1.5, ((Math.max(...lons) - Math.min(...lons)) * k * 1.5 * H) / W);
  const sc = (H - 2 * pad) / span;
  const px = (lon) => W / 2 + (lon - clo) * k * sc, py = (lat) => H / 2 - (lat - cla) * sc;
  let s = `<svg viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="#fafafa" rx="8"/>`;
  const kmPx = sc / 111;
  s += `<line x1="${W - 24 - kmPx}" x2="${W - 24}" y1="${H - 20}" y2="${H - 20}" stroke="${INK}" stroke-width="2"/><text x="${W - 24 - kmPx / 2}" y="${H - 28}" text-anchor="middle" font-size="11" fill="#667085">1 km</text>`;
  HOSP.forEach((h, i) => {
    const v = S[i][t], x = px(h.lon), y = py(h.lat);
    if (x < 0 || x > W || y < 0 || y > H) return;
    s += `<circle cx="${x}" cy="${y}" r="${v === 1 ? 5 : 4}" fill="${v === 1 ? RED : v === 0 ? AVAIL : "#fff"}" opacity=".7"><title>${h.name}</title></circle>`;
  });
  let d = `M${px(O.lon)},${py(O.lat)}`;
  res.path.forEach((p) => { d += `L${px(HOSP[p.h].lon)},${py(HOSP[p.h].lat)}`; });
  s += `<path d="${d}" fill="none" stroke="${INK}" stroke-width="2" stroke-dasharray="5 4"/>`;
  s += `<rect x="${px(O.lon) - 7}" y="${py(O.lat) - 7}" width="14" height="14" fill="${INK}" rx="2"/><text x="${px(O.lon) - 12}" y="${py(O.lat) - 10}" text-anchor="end" font-size="13" font-weight="700" fill="${INK}">${O.name} 출발</text>`;
  res.path.forEach((p, i) => {
    s += `<circle cx="${px(HOSP[p.h].lon)}" cy="${py(HOSP[p.h].lat)}" r="11" fill="${p.full ? RED : GREEN}"/><text x="${px(HOSP[p.h].lon)}" y="${py(HOSP[p.h].lat) + 4}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#fff">${i + 1}</text>`;
  });
  res.path.forEach((p, i) => {
    s += `<text x="20" y="${30 + i * 20}" font-size="12.5" fill="${INK}"><tspan font-weight="700" fill="${p.full ? RED : GREEN}">${i + 1}</tspan>  ${HOSP[p.h].name} · ${p.full ? "꽉 참" : "자리 있음"}</text>`;
  });
  $("q2map").innerHTML = s + `</svg>`;
  const steps = res.path.map((p, i) => `${i + 1}. ${HOSP[p.h].name}에 ${Math.round(p.clock)}분 만에 도착 → ${p.full ? `<b style="color:var(--red)">꽉 참</b>, 거절당하는 데 ${q2.H}분` : `<b style="color:var(--ok)">자리 있음</b>`}`);
  $("q2story").innerHTML = `<b>${timeLabel(t)}</b>, ${O.name}에서 구급차 출발<br>${steps.join("<br>")}<br><b>총 ${Math.round(res.T)}분</b>, ${res.V > 10 ? "10곳 모두 꽉 참" : res.V + "곳 방문"}`;
}

// ---- Q3 ---------------------------------------------------------------------------
const MOVE0 = (() => {
  const f = (DATA.road / DATA.speed) * 60;
  const mins = KM_HH.map((row, i) => Math.min(...row.filter((_, j) => j !== i)) * f).sort((a, b) => a - b);
  return Math.round(DATA.handling + mins[Math.floor(mins.length / 2)]);
})();
const Q3_KEYS = [
  ["dMedian", "꽉 찬 상태 길이 (중앙값)", fixed(0, "분")],
  ["dMean", "꽉 찬 상태 길이 (평균)", fixed(0, "분")],
  ["more30", "30분 넘게 감", pct],
  ["more60at30", "이미 30분째 → 30분 더 감", pct],
  ["more150at120", "이미 2시간째 → 30분 더 감", pct],
  ["recover15", "꽉 참 → 15분 뒤 자리 생김", pct],
];
function renderQ3() {
  $("q3piV").textContent = Math.round(+$("q3pi").value * 100) + "%";
  $("q3msV").textContent = $("q3ms").value + "분";
  $("q3moveV").textContent = $("q3move").value + "분";
  $("q3mixBox").classList.toggle("off", !$("q3mix").checked);
  if (!state.sims.length) return;
  const move = +$("q3move").value;
  const real = REAL_M.wait, sim = WAIT_S.map((_, i) => band(state.sims.map((s) => s.m.wait[i])));
  const nOk = Q3_KEYS.filter(([k]) => okKey(k)).length;
  $("q3Answer").innerHTML =
    `<p class="q">질문: 꽉 찬 병원에 도착했다면, 기다릴까 옮길까?</p>` +
    `<p class="a">${real[0] > move ? "옮기는 게 낫다." : "잠깐은 기다려 볼 만하다."} 막 꽉 찬 병원은 평균 ${minText(real[0])}, 이미 1시간째 꽉 찬 병원은 평균 ${minText(real[2])} 더 기다려야 하는데, 다른 병원까지는 ${move}분이다.</p>` +
    `<p class="why">오래 꽉 차 있던 병원일수록 앞으로도 더 오래 갑니다 (지수분포의 무기억성이 성립하지 않음). ` +
    (cfg.mix ? `지금 '섞기' 모델의 가짜 응급실은 6개 지표 중 <b>${nOk}개</b>에서 진짜와 비슷합니다.`
      : `지금 '한 종류(지수분포)' 모델로는 6개 지표 중 <b>${nOk}개</b>만 진짜와 비슷해요. '섞기'를 켜거나 자동으로 찾아 보세요.`) + `</p>`;
  const labels = ["막 꽉 찼을 때", "이미 30분째", "이미 1시간째", "이미 2시간째"];
  const mx = Math.max(...real.filter((v) => !isNaN(v)), ...sim.map((b) => b.mean).filter((v) => !isNaN(v)), move) * 1.1;
  const rows = [];
  labels.forEach((lb, i) => {
    rows.push({ group: lb });
    rows.push({ label: "진짜", value: real[i], color: RED, text: minText(real[i]), mark: move });
    rows.push({ label: "가짜 (지금 설정)", value: sim[i].mean, color: SIM, text: minText(sim[i].mean), mark: move });
  });
  $("q3Bars").innerHTML = bars(rows, mx) + `<p class="hint" style="margin-top:12px">검은 세로선 = 다른 병원으로 옮기는 데 드는 시간 (${move}분). 막대가 선보다 길면 옮기는 게 낫습니다.</p>`;
  $("q3Table").innerHTML = compareTable(Q3_KEYS.map(([k, label, fmt]) => ({ label, real: REAL_M[k], sim: simVals(k), fmt })));
  const sv = D_GRID.map((_, i) => band(state.sims.map((s) => s.m.surv[i])));
  const fl = 1e-3;
  chart($("chartQ3a"), {
    h: 240, x: { min: 0, max: 360, ticks: [0, 60, 120, 180, 240, 300, 360], fmt: (v) => v + "분" },
    y: { min: fl, max: 1, log: true, ticks: [0.001, 0.01, 0.1, 1], fmt: (v) => pct(v, v < 0.01 ? 1 : 0) },
    series: [
      { type: "band", xs: D_GRID, lo: sv.map((b) => Math.max(fl, b.lo)), hi: sv.map((b) => Math.max(fl, b.hi)), color: BAND },
      { type: "line", xs: D_GRID, ys: sv.map((b) => Math.max(fl, b.mean)), color: SIM },
      { type: "line", xs: D_GRID, ys: REAL_M.surv.map((v) => (v > 0 ? v : NaN)), color: RED },
    ],
    legend: LEGEND_STD,
  });
}
function q3Auto() {   // 섞기 모델의 두 값을 조금씩 바꿔 가며 진짜와 가장 비슷한 조합 찾기
  const grid = [];
  for (let pi = 0.3; pi <= 0.901; pi += 0.1) for (let ms = 5; ms <= 30; ms += 5) grid.push([+pi.toFixed(1), ms]);
  const c0 = { ...cfg, mix: true }, keys = Q3_KEYS.map(([k]) => k);
  let i = 0, best = null;
  $("q3auto").disabled = true;
  const step = () => {
    if (i === grid.length) {
      $("q3mix").checked = true; $("q3pi").value = best.pi; $("q3ms").value = best.ms;
      $("q3auto").disabled = false;
      $("q3prog").textContent = `가장 비슷한 조합: 금방 풀리는 만실 ${Math.round(best.pi * 100)}%, 평균 ${best.ms}분 (${grid.length}가지 비교)`;
      durChange();
      return;
    }
    const [pi, ms] = grid[i];
    let sc = 0;
    for (let rep = 0; rep < 2; rep++) {
      const m = durMetrics(simulate({ ...c0, pi, ms }, 300 + rep));
      for (const k of keys) sc += ((m[k] - REAL_M[k]) / REAL_M[k]) ** 2;
    }
    if (!best || sc < best.sc) best = { sc, pi, ms };
    $("q3prog").textContent = `조금씩 바꿔 보는 중… ${i + 1} / ${grid.length}`;
    i++;
    setTimeout(step, 0);
  };
  step();
}
function durChange() {
  cfg.durHosp = $("q3hosp").checked; cfg.durSlot = $("q3slot").checked; cfg.mix = $("q3mix").checked;
  cfg.pi = +$("q3pi").value; cfg.ms = +$("q3ms").value;
  renderQ3(); durSummary(); rerun();
}
function durSummary() {
  const parts = [cfg.mix ? `금방 풀리는 것 ${Math.round(cfg.pi * 100)}% (평균 ${cfg.ms}분) + 오래가는 것` : "한 종류 (지수분포)"];
  if (cfg.durHosp) parts.push("병원마다 다르게");
  if (cfg.durSlot) parts.push("시간대마다 다르게");
  $("durSummary").textContent = parts.join(" · ");
}

// ---- 화면 연결 --------------------------------------------------------------------
function render() {
  durSummary();
  if (state.tab === "engine") renderEngine();
  if (state.tab === "q1") renderQ1();
  if (state.tab === "q2") renderQ2();
  if (state.tab === "q3") renderQ3();
}
function sigmaLabel() {
  cfg.sigma = +$("sigma").value;
  $("sigmaV").textContent = cfg.sigma.toFixed(2);
  $("sigmaHelp").textContent = cfg.sigma === 0 ? "0 = 매일 똑같이 붐빔"
    : `바쁜 날은 평소의 약 ${(1 + cfg.sigma).toFixed(2)}배, 한가한 날은 ${(1 - cfg.sigma).toFixed(2)}배 (열흘 중 7일 정도가 이 사이). 매일 L ~ N(1, σ²)`;
}
function onSigma() { sigmaLabel(); rerun(); }

function init() {
  $("meta").innerHTML = `${timeLabel(0).slice(0, -6)} ~ ${timeLabel(NT - 1)} · 응급실 ${NH}곳 · 5분 간격<br>확률 계산 ${DATA.generated}`;
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === "v-" + b.dataset.tab));
    state.tab = b.dataset.tab;
    if (state.playing) togglePlay();
    render();
  }));
  $("level").onchange = () => { cfg.level = +$("level").value; rerun(); };
  $("sigma").oninput = onSigma;
  $("reps").onchange = () => { cfg.reps = +$("reps").value; rerun(); };
  $("reroll").onclick = () => { state.seed += 100003; runEngine(); };

  buildMap("mapReal"); buildMap("mapSim");
  $("scrub").max = NT - 1;
  $("scrub").oninput = () => { state.tf = +$("scrub").value; showTime(state.tf); };
  $("play").onclick = togglePlay;
  state.t = Math.max(0, [...Array(NT).keys()].find((t) => DAY[t] === 7 && HOUR[t] === 15) ?? 0);

  WEEKS.forEach((d, i) => $("week").insertAdjacentHTML("beforeend", `<option value="${i}">${dayLabel(d)}부터 일주일</option>`));
  $("week").value = state.week;
  $("week").onchange = () => { state.week = +$("week").value; renderTimeline(); };
  document.querySelector("#v-engine details:last-of-type").addEventListener("toggle", renderDetails);
  hookTip($("tlReal"), "진짜"); hookTip($("tlSim"), "가짜");
  $("sweepBtn").onclick = runSweep;

  q1Reset();
  ["q1a", "q1b", "q1s"].forEach((id) => ($(id).oninput = () => { clearTimeout(q1.t); q1.t = setTimeout(renderQ1, 80); }));
  $("q1slot").onchange = () => { q1Reset(); renderQ1(); };
  $("q1fit").onclick = () => { q1Reset(); renderQ1(); };
  $("q1auto").onclick = q1Auto;

  $("q2origin").innerHTML = `<option value="-1">서울 전체 (25개 구청에서 번갈아)</option>` + DATA.origins.map((o, i) => `<option value="${i}">${o.name}</option>`).join("");
  $("q2h").value = q2.H; $("q2v").value = q2.speed;
  $("q2origin").onchange = () => { q2.origin = +$("q2origin").value; renderQ2(); };
  $("q2strat").onchange = () => { q2.strat = $("q2strat").value; renderQ2(); };
  $("q2h").oninput = () => { q2.H = +$("q2h").value; $("q2hV").textContent = q2.H + "분"; };
  $("q2h").onchange = renderQ2;
  $("q2v").oninput = () => { q2.speed = +$("q2v").value; $("q2vV").textContent = q2.speed + " km/h"; };
  $("q2v").onchange = renderQ2;
  $("q2roll").onclick = () => { q2.roll++; if (state.q2cache) renderQ2Map(); };

  $("q3move").value = MOVE0;
  ["q3hosp", "q3slot", "q3mix", "q3one"].forEach((id) => ($(id).onchange = durChange));
  ["q3pi", "q3ms"].forEach((id) => { $(id).oninput = () => renderQ3(); $(id).onchange = durChange; });
  $("q3move").oninput = renderQ3;
  $("q3auto").onclick = q3Auto;

  sigmaLabel();
  durSummary();
  renderQ3();
  const first = document.querySelector(`#tabs button[data-tab="${location.hash.slice(1)}"]`);
  if (first) first.click();
  window.addEventListener("resize", () => { clearTimeout(state.rs); state.rs = setTimeout(render, 200); });
  runEngine();
}
init();
