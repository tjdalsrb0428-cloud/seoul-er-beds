"""앱(기말 시연용)이 읽을 요약 데이터 → analysis/output/app_data.json

병원별: 좌표, 정원, 시간대별 P[A=1], 평균 만실 지속시간 E[W], 최신 스냅샷.
앱은 이 파일만 읽으면 되도록 확률 추정은 전부 여기서 끝냄.
"""
import json
from datetime import datetime

import numpy as np

from common import (HANDLING_MIN, OUT, ORIGINS, ROAD_FACTOR, SPEED_KMH, load_hospitals,
                    load_snapshots, p_by_hour)
from q3_waiting import full_runs


def clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def main():
    snap = load_snapshots()
    hosp = load_hospitals(snap)
    ph = p_by_hour(snap)
    runs = full_runs(snap)
    ew = runs.groupby("hpid")["W"].mean() if not runs.empty else {}
    ew_pool = float(runs["W"].mean()) if not runs.empty else None
    last = snap.sort_values("t").groupby("hpid").tail(1).set_index("hpid")
    n_obs = snap.groupby("hpid")["A"].count()

    hospitals = []
    for _, h in hosp.iterrows():
        i = h["hpid"]
        hospitals.append({
            "hpid": i, "name": h["dutyName"], "type": h["dutyEmclsName"],
            "lat": h["lat"], "lon": h["lon"], "capacity": clean(float(h["capacity"])),
            "p_by_hour": [round(float(x), 4) for x in ph.loc[i]] if i in ph.index else None,
            "mean_full_min": clean(round(float(ew[i]), 1)) if i in ew else ew_pool,
            "n_obs": int(n_obs.get(i, 0)),
            "last": {"t": str(last.loc[i, "t"]), "B": clean(float(last.loc[i, "hvec"]))} if i in last.index else None,
        })
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_range": [str(snap["t"].min()), str(snap["t"].max())],
        "assumptions": {"speed_kmh": SPEED_KMH, "road_factor": ROAD_FACTOR, "handling_min": HANDLING_MIN},
        "origins": {k: list(v) for k, v in ORIGINS.items()},
        "hospitals": hospitals,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "app_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[앱] 병원 {len(hospitals)}곳 → {OUT / 'app_data.json'}")


if __name__ == "__main__":
    main()
