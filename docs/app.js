/* 응급실 도착 확률 — 시연용 프로토타입
 * 확률 추정치는 analysis/export_app.py 가 만든 app_data.js(window.APP_DATA)에서 읽고,
 * '지금 상태'는 공개 저장소의 최신 수집 CSV에서 직접 읽는다.
 */
"use strict";

const REPO = "tjdalsrb0428-cloud/seoul-er-beds";
const CANDIDATES = 8;     // 추천 순서를 고를 후보 수 (가까운 순)
const PREFIX = 4;         // 앞 4곳은 모든 순서를 비교, 나머지는 p/(이동+확인) 지수 순
const LIST_N = 15;        // 주변 응급실 목록 길이
const SEVERE_TYPES = ["권역응급의료센터", "지역응급의료센터"];

const D = window.APP_DATA;
const A = D.assumptions;
const H = A.handling_min;
const state = { origin: null, originName: "", mode: "now", hour: 21, live: null, useLive: true, severe: false };
let map, layer, hospitals;

// ---------------------------------------------------------------- 수학
function km(a, b) {
  const R = 6371, rad = Math.PI / 180;
  const dLa = (b.lat - a.lat) * rad, dLo = (b.lon - a.lon) * rad;
  const x = Math.sin(dLa / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLo / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}
const travel = (a, b) => km(a, b) * A.road_factor / A.speed_kmh * 60;   // 분

function startTime() {
  const now = new Date();
  if (state.mode === "now") return now;
  const t = new Date(now); t.setHours(state.hour, 0, 0, 0); return t;
}

/** 도착 시 병상 확률 p_i. elapsed = 출발 후 경과 분 */
function pArrive(h, elapsed) {
  const start = startTime();
  const live = state.useLive && state.mode === "now" && state.live ? state.live.byId[h.hpid] : undefined;
  if (live !== undefined) {
    const horizon = state.live.ageMin + elapsed;            // 관측 시점부터 도착까지
    if (horizon <= D.horizons_min[D.horizons_min.length - 1]) {
      const k = Math.max(0, Math.round(horizon / 5) - 1);
      return live >= 1 ? h.cond.from_avail[k] : h.cond.from_full[k];
    }
  }
  const arr = new Date(start.getTime() + elapsed * 60000);
  return h.p_by_hour[arr.getHours()];
}

/** 방문 순서 하나의 E[T], E[N], N의 PMF (병원 간 독립 가정) */
function evalRoute(order) {
  let pos = state.origin, t = 0, S = 1, ET = 0, EN = 0;
  const pmf = [], steps = [];
  for (const h of order) {
    const d = travel(pos, h), arrive = t + d, p = pArrive(h, arrive);
    ET += S * d;               // k번째로 이동하는 건 앞이 모두 실패했을 때만
    EN += S;                   // E[N] = Σ P[N ≥ k]
    pmf.push(S * p);
    steps.push({ h, d, p, arrive, reach: S });
    S *= 1 - p;
    ET += S * H;               // 실패하면 확인·거절 시간
    t = arrive + H; pos = h;
  }
  let cum = 0, n95 = null;
  pmf.forEach((q, i) => { cum += q; if (n95 === null && cum >= 0.95) n95 = i + 1; });
  return { order, ET, EN, pmf, steps, fail: S, n95, p2: (pmf[0] || 0) + (pmf[1] || 0) };
}

function permutations(arr, k) {
  if (k === 0) return [[]];
  const out = [];
  arr.forEach((x, i) => {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permutations(rest, k - 1)) out.push([x, ...p]);
  });
  return out;
}

/** 앞부분은 전수 비교, 뒷부분은 p/(이동+확인) 지수 순으로 이어 붙여 E[T] 최소 순서 */
function bestRoute(cands) {
  let best = null;
  for (const pre of permutations(cands, Math.min(PREFIX, cands.length))) {
    const rest = cands.filter(h => !pre.includes(h));
    const order = [...pre];
    let pos = pre[pre.length - 1];
    while (rest.length) {
      let bi = 0, bs = -1;
      rest.forEach((h, i) => { const s = h.p_all / (travel(pos, h) + H); if (s > bs) { bs = s; bi = i; } });
      pos = rest.splice(bi, 1)[0]; order.push(pos);
    }
    const r = evalRoute(order);
    if (!best || r.ET < best.ET) best = r;
  }
  return best;
}

// ---------------------------------------------------------------- 실시간 데이터
async function loadLive() {
  const kst = off => new Date(Date.now() + 9 * 3600e3 - off * 86400e3).toISOString().slice(0, 10);
  for (const day of [kst(0), kst(1)]) {
    try {
      const res = await fetch(`https://raw.githubusercontent.com/${REPO}/main/data/${day}.csv`, { cache: "no-store" });
      if (!res.ok) continue;
      const lines = (await res.text()).replace(/^﻿/, "").trim().split(/\r?\n/);
      const head = lines[0].split(","), it = head.indexOf("collected_at"), ih = head.indexOf("hpid"), iv = head.indexOf("hvec");
      const rows = lines.slice(1).map(l => l.split(","));
      const lastT = rows.reduce((m, r) => (r[it] > m ? r[it] : m), "");
      const byId = {};
      rows.filter(r => r[it] === lastT).forEach(r => { if (r[iv] !== "") byId[r[ih]] = Number(r[iv]); });
      const when = new Date(lastT.replace(" ", "T") + "+09:00");
      return { byId, when, ageMin: Math.max(0, (Date.now() - when) / 60000) };
    } catch (e) { /* 오프라인 등 → 다음 날짜 시도 */ }
  }
  return null;
}

// ---------------------------------------------------------------- 화면
const pct = p => `${Math.round(p * 100)}%`;
const lvl = p => (p >= 0.7 ? "hi" : p >= 0.4 ? "mid" : "lo");
const mins = m => `${Math.round(m)}분`;
const color = p => getComputedStyle(document.documentElement).getPropertyValue(`--${lvl(p)}`).trim();
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const $ = id => document.getElementById(id);

function liveBadge(h) {
  if (!state.live || !(h.hpid in state.live.byId)) return "";
  const b = state.live.byId[h.hpid];
  return b >= 1 ? `<span class="badge ok">지금 ${b}병상</span>`
                : `<span class="badge full">지금 만실${b < 0 ? ` (${-b}명 초과)` : ""}</span>`;
}

function render() {
  const pool = hospitals.filter(h => !state.severe || SEVERE_TYPES.includes(h.type));
  const byDist = pool.map(h => ({ h, d: travel(state.origin, h) })).sort((a, b) => a.d - b.d);
  const cands = byDist.slice(0, CANDIDATES).map(x => x.h);

  const best = bestRoute(cands);
  const near = evalRoute(cands);                                         // 가까운 순
  const prob = evalRoute([...cands].sort((a, b) => pArrive(b, travel(state.origin, b)) - pArrive(a, travel(state.origin, a))));

  $("kET").textContent = mins(best.ET);
  $("kEN").textContent = `${best.EN.toFixed(2)}곳`;
  $("kN95").textContent = best.n95 ? `${best.n95}곳` : `${CANDIDATES}곳+`;
  const gain = near.ET - best.ET;
  $("compareLine").innerHTML = gain >= 0.5
    ? `가까운 순서로 가면 E[T] = ${mins(near.ET)} → 추천 순서가 평균 <b>${gain.toFixed(1)}분</b> 빠름`
    : `이 위치·시각에서는 가까운 순서가 이미 최선에 가까움 (E[T] = ${mins(near.ET)})`;

  const rows = [["추천 (E[T] 최소)", best], ["가까운 순", near], ["확률 높은 순", prob]];
  $("stratTable").innerHTML = `<tr><th>전략</th><th>E[T]</th><th>E[N]</th><th>P[N≤2]</th><th>95% 확보</th></tr>` +
    rows.map(([n, r], i) => `<tr class="${i === 0 ? "best" : ""}"><td>${n}</td><td>${mins(r.ET)}</td><td>${r.EN.toFixed(2)}</td>` +
      `<td>${pct(r.p2)}</td><td>${r.n95 ? r.n95 + "곳" : "–"}</td></tr>`).join("");
  $("pmfChart").innerHTML = pmfSvg(best.pmf);

  $("routeList").innerHTML = best.steps.slice(0, 5).map((s, i) => `
    <li class="b-${lvl(s.p)}" data-id="${s.h.hpid}">
      <span class="num">${i + 1}</span>
      <div><div class="rname">${esc(s.h.name)} ${liveBadge(s.h)}</div>
        <div class="rmeta">${esc(s.h.type)} · 출발 후 ${mins(s.arrive)} 도착</div>
        <div class="reach">여기까지 올 확률 ${pct(s.reach)}</div></div>
      <div class="pbig ${lvl(s.p)}-t">${pct(s.p)}<small>도착 시 병상</small></div>
    </li>`).join("");

  $("hospList").innerHTML = byDist.slice(0, LIST_N).map(({ h, d }) => {
    const p = pArrive(h, d);
    return `<li class="b-${lvl(p)}" data-id="${h.hpid}">
      <div><div class="rname">${esc(h.name)} ${liveBadge(h)}</div><div class="rmeta">${esc(h.type)} · ${mins(d)}</div></div>
      <div class="pbig ${lvl(p)}-t">${pct(p)}</div></li>`;
  }).join("");
  document.querySelectorAll("[data-id]").forEach(el => el.onclick = () => showDetail(el.dataset.id));

  drawMap(pool, best);
}

function drawMap(pool, best) {
  layer.clearLayers();
  L.marker([state.origin.lat, state.origin.lon], {
    icon: L.divIcon({ className: "", html: '<div class="origin-pin">출발</div>', iconSize: [0, 0] }),
    zIndexOffset: 2000,
  }).bindTooltip(`출발: ${esc(state.originName)}`).addTo(layer);
  const inRoute = new Map(best.steps.slice(0, 5).map((s, i) => [s.h.hpid, i + 1]));
  for (const h of pool) {
    if (inRoute.has(h.hpid)) continue;
    const p = pArrive(h, travel(state.origin, h));
    L.circleMarker([h.lat, h.lon], { radius: 7, color: "#fff", weight: 1.5, fillColor: color(p), fillOpacity: 0.9 })
      .bindTooltip(`${esc(h.name)} · ${pct(p)}`).on("click", () => showDetail(h.hpid)).addTo(layer);
  }
  const path = [[state.origin.lat, state.origin.lon], ...best.steps.slice(0, 5).map(s => [s.h.lat, s.h.lon])];
  L.polyline(path, { color: "#1e3a5f", weight: 3, opacity: 0.8, dashArray: "6 6" }).addTo(layer);
  best.steps.slice(0, 5).forEach((s, i) => {
    L.marker([s.h.lat, s.h.lon], {
      icon: L.divIcon({ className: "", html: `<div class="pin" style="background:${color(s.p)}">${i + 1}</div>`, iconSize: [24, 24] }),
      zIndexOffset: 1000 - i,
    }).bindTooltip(`${i + 1}. ${esc(s.h.name)} · 도착 시 ${pct(s.p)}`).on("click", () => showDetail(s.h.hpid)).addTo(layer);
  });
}

function showDetail(id) {
  const h = hospitals.find(x => x.hpid === id);
  const d0 = travel(state.origin, h), p = pArrive(h, d0);
  const arrHour = new Date(startTime().getTime() + d0 * 60000).getHours();
  const W = h.mean_full_min;
  const live = state.live && h.hpid in state.live.byId ? state.live.byId[h.hpid] : null;

  // 만실이라면? 기다리기 E[W] vs 가까운 병원으로 옮기기 D + (1-p)E[W_j]
  let waitHtml = `<p class="note">만실 지속시간 데이터가 아직 부족해 비교할 수 없어요.</p>`;
  if (W) {
    const alts = hospitals.filter(x => x !== h && (!state.severe || SEVERE_TYPES.includes(x.type)))
      .map(x => { const d = travel(h, x), q = pArrive(x, d0 + H + d), w = x.mean_full_min || W;
                  return { x, d, q, cost: d + (1 - q) * w }; })
      .sort((a, b) => a.cost - b.cost).slice(0, 2);
    const best = Math.min(W, ...alts.map(a => a.cost));
    waitHtml = `
      <div class="opt ${W <= best ? "win" : ""}"><b>여기서 기다리기</b>${W <= best ? '<span class="tagwin">추천</span>' : ""}
        <div class="t">${mins(W)}</div><p>남은 대기시간 기댓값 = E[W] (지수분포의 무기억성) · 30분 안에 풀릴 확률 ${pct(1 - Math.exp(-30 / W))}</p></div>` +
      alts.map(a => `
      <div class="opt ${a.cost === best ? "win" : ""}"><b>${esc(a.x.name)}(으)로 옮기기</b>${a.cost === best ? '<span class="tagwin">추천</span>' : ""}
        <div class="t">${mins(a.cost)}</div><p>이동 ${mins(a.d)} + (1 − ${pct(a.q)}) × 그곳 E[W] ${mins(a.x.mean_full_min || W)}</p></div>`).join("");
  }

  $("detail").innerHTML = `
    <div class="dhead"><h2>${esc(h.name)}</h2><p>${esc(h.type)} · 응급실 정원 ${h.capacity ?? "?"}병상 · 관측 ${h.n_obs}회</p></div>
    <div class="card bigp"><b class="${lvl(p)}-t">${pct(p)}</b>
      <span>${mins(d0)} 뒤 도착했을 때 병상이 있을 확률 P[A = 1]</span></div>
    <div class="card"><dl class="rows">
      <dt>최근 관측</dt><dd>${live === null ? "–" : live >= 1 ? `${live}병상` : `만실${live < 0 ? ` (${-live}명 초과)` : ""}`}</dd>
      <dt>전체 기간 병상 있음 비율</dt><dd>${pct(h.p_all)}</dd>
      <dt>평균 만실 지속시간 E[W]</dt><dd>${W ? mins(W) + (h.mean_full_src === "pooled" ? " (전체 평균)" : "") : "–"}</dd>
    </dl></div>
    <div class="card"><h2>시간대별 병상 있을 확률</h2>${hourSvg(h.p_by_hour, arrHour)}
      <p class="note">진한 막대 = 도착 예정 시각(${arrHour}시)</p></div>
    <div class="card"><h2>만실이라면? 기다리기 vs 옮기기</h2>${waitHtml}</div>`;
  $("mainView").classList.add("hidden");
  $("detailView").classList.remove("hidden");
  $("side").scrollTop = 0;
  map.panTo([h.lat, h.lon]);
}

// ---------------------------------------------------------------- 작은 SVG 차트
function pmfSvg(pmf) {
  const w = 400, h = 150, pad = 26, n = Math.min(pmf.length, 6), bw = (w - pad * 2) / n;
  let cum = 0, bars = "", line = "", labels = "";
  for (let i = 0; i < n; i++) {
    cum += pmf[i];
    const x = pad + i * bw, bh = pmf[i] * (h - pad * 2), y = h - pad - bh;
    bars += `<rect x="${x + 6}" y="${y}" width="${bw - 12}" height="${bh}" rx="3" fill="#4f7cac"><title>P[N=${i + 1}] = ${pct(pmf[i])}</title></rect>`;
    bars += `<text x="${x + bw / 2}" y="${y - 4}" text-anchor="middle" font-size="11" fill="#1f2937">${pct(pmf[i])}</text>`;
    line += `${i ? "L" : "M"}${x + bw / 2},${h - pad - cum * (h - pad * 2)} `;
    labels += `<text x="${x + bw / 2}" y="${h - 8}" text-anchor="middle" font-size="11" fill="#6b7280">${i + 1}곳</text>`;
  }
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="N의 분포">
    <line x1="${pad}" x2="${w - pad}" y1="${h - pad}" y2="${h - pad}" stroke="#e5e7eb"/>${bars}
    <path d="${line}" fill="none" stroke="#c8102e" stroke-width="2"/>${labels}</svg>`;
}

function hourSvg(ph, hl) {
  const w = 400, h = 120, pad = 18, bw = (w - pad * 2) / 24;
  let s = "";
  ph.forEach((p, i) => {
    const bh = p * (h - pad * 2), x = pad + i * bw;
    s += `<rect x="${x + 1}" y="${h - pad - bh}" width="${bw - 2}" height="${bh}" fill="${color(p)}" opacity="${i === hl ? 1 : 0.45}"><title>${i}시: ${pct(p)}</title></rect>`;
    if (i % 6 === 0) s += `<text x="${x}" y="${h - 4}" font-size="10" fill="#6b7280">${i}시</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="시간대별 확률">
    <line x1="${pad}" x2="${w - pad}" y1="${h - pad - 0.5 * (h - pad * 2)}" y2="${h - pad - 0.5 * (h - pad * 2)}" stroke="#d1d5db" stroke-dasharray="3 3"/>${s}
    <text x="${w - pad}" y="${h - pad - 0.5 * (h - pad * 2) - 3}" font-size="10" fill="#9ca3af" text-anchor="end">50%</text></svg>`;
}

function renderStatus() {
  const [a, b] = D.data_range.map(s => s.slice(5, 16));
  const thin = D.n_snapshots < 288;   // 하루치 미만
  let html = `학습 데이터 <b>${a} ~ ${b}</b> · 스냅샷 ${D.n_snapshots}회`;
  if (thin) html += ` <span class="warn">(수집 초기 · 추정 불확실)</span>`;
  if (state.live) html += `<br>실시간 <b>${state.live.when.toTimeString().slice(0, 5)}</b> 기준 (${Math.round(state.live.ageMin)}분 전)`;
  $("status").innerHTML = html;
  $("liveInfo").textContent = state.live ? `(${state.live.when.toTimeString().slice(0, 5)} 관측)` : "(불러오지 못함)";
  $("useLive").disabled = !state.live;
}

// ---------------------------------------------------------------- 시작
function setOrigin(lat, lon, name) {
  state.origin = { lat, lon }; state.originName = name;
  render();
}

function init() {
  hospitals = D.hospitals.filter(h => h.lat && h.lon);
  map = L.map("map", { zoomControl: true }).setView([37.55, 126.99], 11);
  // OSM·CARTO 타일 서버는 file:// 로 연 페이지(Referer 없음)를 차단 → Esri 타일 사용
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: "Tiles &copy; Esri — Esri, HERE, Garmin, OpenStreetMap contributors",
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
  // 그리드 레이아웃이 자리 잡은 뒤 지도 크기 재계산 (안 하면 일부가 회색으로 비어 보임)
  new ResizeObserver(() => map.invalidateSize()).observe(document.getElementById("map"));

  const sel = $("origin");
  sel.innerHTML = Object.keys(D.origins).map(g => `<option>${g}청</option>`).join("") + `<option value="custom" hidden>지도에서 선택한 위치</option>`;
  sel.value = "마포구청";
  sel.onchange = () => {
    if (sel.value === "custom") return;
    const [lat, lon] = D.origins[sel.value.replace(/청$/, "")];
    setOrigin(lat, lon, sel.value); map.setView([lat, lon], 12);
  };
  map.on("click", e => { sel.value = "custom"; setOrigin(e.latlng.lat, e.latlng.lng, "지도에서 선택한 위치"); });

  $("modeNow").onclick = () => { state.mode = "now"; $("modeNow").classList.add("on"); $("modeHour").classList.remove("on"); $("hourRow").classList.add("hidden"); render(); };
  $("modeHour").onclick = () => { state.mode = "hour"; $("modeHour").classList.add("on"); $("modeNow").classList.remove("on"); $("hourRow").classList.remove("hidden"); render(); };
  $("hour").oninput = e => { state.hour = +e.target.value; $("hourOut").textContent = `${state.hour}시`; render(); };
  $("useLive").onchange = e => { state.useLive = e.target.checked; render(); };
  $("severe").onchange = e => { state.severe = e.target.checked; render(); };
  $("back").onclick = () => { $("detailView").classList.add("hidden"); $("mainView").classList.remove("hidden"); };
  $("assumpLine").textContent = `가정값: 구급차 ${A.speed_kmh}km/h, 직선→도로 ×${A.road_factor}, 병원 도착 후 확인·거절 ${H}분 (측정값 아님)`;

  const [lat, lon] = D.origins["마포구"];
  setOrigin(lat, lon, "마포구청");
  map.setView([lat, lon], 12);
  renderStatus();
  loadLive().then(live => {
    state.live = live; renderStatus(); render();
    // 시연용 바로가기: index.html#h=병원ID → 해당 병원 상세 화면
    const m = location.hash.match(/^#h=(\w+)/);
    if (m && hospitals.some(h => h.hpid === m[1])) showDetail(m[1]);
  });
}

init();
