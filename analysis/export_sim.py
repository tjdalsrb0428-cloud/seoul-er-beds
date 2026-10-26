"""시뮬레이션 앱(sim/)이 읽을 데이터 → sim/sim_data.js

앱은 여기서 넘겨준 '실제 데이터에서 구한 확률'로 가짜 데이터를 만들고, 실제 타임라인과 같은 방식으로 비교한다.
- timeline : 병원별 5분 격자 상태 문자열  '1' 만실, '0' 병상 있음, '.' 수집 실패(모름)
- P        : P[만실 | 평일/주말, 시간대]  병원별 (2 × 4)
- D        : 병원별 평균 만실 지속시간 (분), 만실 5번 미만이면 전체 평균
- q1       : 지금 병상 b개일 때 15분 뒤 만실 비율 (b = 1..5, 전체·시간대별)
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from common import (HANDLING_MIN, ORIGINS, ROAD_FACTOR, ROOT, SLOT_BINS, SPEED_KMH,
                    load_hospitals, load_snapshots)

APP_DIR = ROOT / "sim"
START = pd.Timestamp("2026-10-05 00:00")
STEP = 5
MIN_RUNS = 5
LOOKUP_MAX = 10


def short_name(n):
    n = re.sub(r"^(학교법인|의료법인|재단법인|사회복지법인)\s?\S*?(재단|학원|의료원)?\s", "", n)
    n = re.sub(r"^(재단법인아산사회복지재단|의료법인\S+재단|학교법인\S+학원)\s?", "", n)
    n = n.replace("의과대학부속", "").replace("의과대학", "").replace("대학교", "대 ")
    n = n.replace("서울특별시", "").replace("(안암병원)", " 안암").replace("부속", "")
    n = re.sub(r"^(의료법인|\S*의료재단|\S*의료공단\s?|한국원자력의학원)", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n.replace("대 병원", "대병원")


def run_lengths(col):
    out, k, n = [], 1, len(col)
    while k < n:
        if col[k] == 1 and col[k - 1] == 0:
            st = k
            while k < n and col[k] == 1:
                k += 1
            if k < n and col[k] == 0:
                out.append((k - st) * STEP)
        k += 1
    return out


def main():
    snap = load_snapshots()
    snap = snap[snap["t"] >= START].dropna(subset=["A"])
    hosp = load_hospitals(snap).set_index("hpid")

    # 5분 격자
    snap["g"] = snap["t"].dt.round(f"{STEP}min")
    wide = snap.drop_duplicates(["g", "hpid"]).pivot(index="g", columns="hpid", values="A")
    grid = pd.date_range(wide.index[0], wide.index[-1], freq=f"{STEP}min")
    F = 1 - wide.reindex(grid)
    F = F[[h for h in F.columns if h in hosp.index]]
    X = F.to_numpy()
    slot = np.digitize(grid.hour, SLOT_BINS[1:-1])
    wkend = (grid.dayofweek >= 5).astype(int)

    runs_all, rows = [], []
    for j, h in enumerate(F.columns):
        r = run_lengths(X[:, j])
        runs_all += r
        P = [[float(np.nanmean(X[(wkend == w) & (slot == k), j])) for k in range(4)] for w in (0, 1)]
        rows.append(dict(id=h, name=short_name(hosp.loc[h, "dutyName"]),
                         lat=round(float(hosp.loc[h, "lat"]), 5), lon=round(float(hosp.loc[h, "lon"]), 5),
                         P=[[round(v, 4) for v in p] for p in P], nruns=len(r),
                         D=None if len(r) < MIN_RUNS else round(float(np.mean(r)), 1),
                         timeline="".join("." if np.isnan(v) else str(int(v)) for v in X[:, j])))
    D_all = float(np.mean(runs_all))
    for r in rows:
        r["D"] = r["D"] or round(D_all, 1)

    # Q1: 지금 병상 b개 → 15분 뒤 만실 (원본 스냅샷, 15분 뒤 값은 10분 이내 관측만)
    q1 = {"all": {}, "slot": [{} for _ in range(4)]}
    for h, g in snap.groupby("hpid"):
        g = g.sort_values("t")
        tt = g["t"].to_numpy(dtype="datetime64[ns]")
        q = tt + np.timedelta64(15, "m")
        idx = np.searchsorted(tt, q, side="right") - 1
        okk = (idx >= 0) & (tt[np.clip(idx, 0, None)] > tt) & \
              ((q - tt[np.clip(idx, 0, None)]) / np.timedelta64(1, "m") <= LOOKUP_MAX)
        b_now = g["B"].to_numpy()[okk]
        b_fut = g["B"].to_numpy()[idx[okk]]
        sl = np.digitize(g["hour"].to_numpy()[okk], SLOT_BINS[1:-1])
        for b in range(1, 6):
            m = b_now == b
            for key, mm in [("all", m)] + [(k, m & (sl == k)) for k in range(4)]:
                tgt = q1["all"] if key == "all" else q1["slot"][key]
                n0, f0 = tgt.get(b, (0, 0))
                tgt[b] = (n0 + int(mm.sum()), f0 + int((b_fut[mm] == 0).sum()))
    fmt = lambda d: [dict(b=b, n=n, p=round(f / n, 4) if n else None) for b, (n, f) in sorted(d.items())]
    q1 = dict(all=fmt(q1["all"]), slot=[fmt(d) for d in q1["slot"]])

    data = dict(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        start=grid[0].strftime("%Y-%m-%dT%H:%M"), step=STEP, n=len(grid),
        D_all=round(D_all, 1), n_runs=len(runs_all),
        speed=SPEED_KMH, road=ROAD_FACTOR, handling=HANDLING_MIN,
        origins=[dict(name=k, lat=v[0], lon=v[1]) for k, v in ORIGINS.items()],
        hospitals=rows, q1=q1,
    )
    APP_DIR.mkdir(exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (APP_DIR / "sim_data.js").write_text("window.SIM_DATA = " + body + ";\n", encoding="utf-8")
    print(f"sim_data.js: 병원 {len(rows)}곳, {len(grid)}칸, {len(body) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
