"""앱(docs/)이 읽을 확률 요약 → docs/app_data.json + docs/app_data.js

병원별로 앱에 필요한 추정치를 전부 여기서 계산한다 (앱은 계산된 확률만 사용).
- p_all       : P[A=1] 라플라스 보정 (k+1)/(n+2)  ← 균등 사전분포의 베이즈 추정
- p_by_hour   : 시간대별 P[A=1 | hour], 관측이 적은 시간대는 p_all 쪽으로 당김
- cond        : P[A(t+Δ)=1 | A(t)=1], P[A(t+Δ)=1 | A(t)=0]  (Δ = 5..60분)
                → '지금 상태'를 알 때 도착 시 확률 (Q1 베이즈와 같은 양)
- mean_full_min : 평균 만실 지속시간 E[W] (Q3), 지수분포 가정 시 기다리기 기대시간
"""
import json
from datetime import datetime

import numpy as np

from common import (HANDLING_MIN, ORIGINS, ROAD_FACTOR, ROOT, SPEED_KMH, StateTable,
                    load_hospitals, load_snapshots)
from q3_waiting import full_runs

APP_DIR = ROOT / "docs"
HORIZONS = list(range(5, 65, 5))   # 분
SHRINK = 6                         # 관측 수가 이만큼일 때 '자기 값'과 'p_all' 을 반반 섞음
MIN_RUNS = 3


def shrink(k, n, prior):
    return (k + SHRINK * prior) / (n + SHRINK)


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main():
    snap = load_snapshots()
    hosp = load_hospitals(snap)
    st = StateTable(snap, "A")
    runs = full_runs(snap)
    pooled_w = float(runs["W"].mean()) if len(runs) else None
    ew = runs.groupby("hpid")["W"].agg(["mean", "size"]) if len(runs) else None
    last = snap.sort_values("t").groupby("hpid").tail(1).set_index("hpid")

    hospitals = []
    for _, h in hosp.iterrows():
        i = h["hpid"]
        g = snap[(snap["hpid"] == i)].dropna(subset=["A"])
        k, n = g["A"].sum(), len(g)
        p_all = (k + 1) / (n + 2)
        byh = g.groupby("hour")["A"].agg(["sum", "size"]).reindex(range(24), fill_value=0)
        p_hour = [shrink(byh.loc[hr, "sum"], byh.loc[hr, "size"], p_all) for hr in range(24)]

        now = st.at(i, st.times)
        cond_a, cond_f = [], []
        for d in HORIZONS:
            fut = st.at(i, st.times + np.timedelta64(d, "m"))
            ok = ~np.isnan(now) & ~np.isnan(fut)
            a1, a0 = ok & (now == 1), ok & (now == 0)
            cond_a.append(r(shrink(fut[a1].sum(), a1.sum(), p_all)))
            cond_f.append(r(shrink(fut[a0].sum(), a0.sum(), p_all)))

        if ew is not None and i in ew.index and ew.loc[i, "size"] >= MIN_RUNS:
            mw, mw_src = float(ew.loc[i, "mean"]), "hospital"
        else:
            mw, mw_src = pooled_w, "pooled"

        lt = last.loc[i] if i in last.index else None
        hospitals.append({
            "hpid": i, "name": h["dutyName"], "type": h["dutyEmclsName"],
            "lat": r(h["lat"], 6), "lon": r(h["lon"], 6),
            "capacity": r(h["capacity"], 0),
            "n_obs": int(n), "p_all": r(p_all),
            "p_by_hour": [r(p) for p in p_hour],
            "cond": {"from_avail": cond_a, "from_full": cond_f},
            "mean_full_min": r(mw, 1), "mean_full_src": mw_src,
            "n_full_runs": int(ew.loc[i, "size"]) if ew is not None and i in ew.index else 0,
            "last": None if lt is None else {"t": str(lt["t"]), "B": r(lt["hvec"], 0)},
        })

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_range": [str(snap["t"].min()), str(snap["t"].max())],
        "n_snapshots": int(snap["t"].nunique()),
        "assumptions": {"speed_kmh": SPEED_KMH, "road_factor": ROAD_FACTOR,
                        "handling_min": HANDLING_MIN, "shrink": SHRINK},
        "horizons_min": HORIZONS,
        "pooled_mean_full_min": r(pooled_w, 1),
        "origins": {k: list(v) for k, v in ORIGINS.items()},
        "hospitals": hospitals,
    }
    APP_DIR.mkdir(exist_ok=True)
    js = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    (APP_DIR / "app_data.json").write_text(js, encoding="utf-8")
    # file:// 로 열어도 동작하도록 같은 내용을 스크립트로도 저장
    (APP_DIR / "app_data.js").write_text("window.APP_DATA = " + js + ";\n", encoding="utf-8")
    print(f"[앱] 병원 {len(hospitals)}곳, 스냅샷 {out['n_snapshots']}회 → {APP_DIR / 'app_data.js'}")


if __name__ == "__main__":
    main()
