"""Q3 (Ch4) — 기다릴까, 옮길까?

1. 만실 지속시간 W_i: 지수분포 적합 (λ̂ = 1/W̄), 무기억성 P[W>s+t | W>s] vs P[W>t]
2. 관측 간격 Δ 에 의한 측정오차: 양 끝이 Uniform → Var ≈ 2·Δ²/12
3. 시간당 '풀린 병상 수' 가 Poisson(α) 인가 (분산/평균 ≈ 1 ?)
4. Erlang: x분 안에 병상 k개가 풀릴 확률 = 1 - Σ_{n<k} Poisson(λx) PMF
5. 만실 병원에서 기다리기 E[W_i] vs 옆 병원으로 옮기기 D_ij + (1-p_j)·E[W_j]
"""
from math import factorial

import numpy as np
import pandas as pd

from common import (MAX_GAP_MIN, hospital_dist_matrix, load_hospitals, load_snapshots,
                    plt, save_fig, save_table, travel_min)

MIN_RUNS = 5                    # 병원별 추정에 필요한 최소 만실 구간 수
ERLANG_K = [1, 2, 3]
ERLANG_X = [15, 30, 60]         # 분
MEMORYLESS_S = [10, 20, 30]
MEMORYLESS_T = [15, 30]
ALT_NEAREST = 5                 # 옮길 후보: 가까운 5곳


def segments(g):
    """관측 공백(> MAX_GAP_MIN)으로 끊긴 연속 관측 구간들."""
    gap = g["t"].diff().dt.total_seconds().div(60).gt(MAX_GAP_MIN)
    return [s for _, s in g.groupby(gap.cumsum())]


def full_runs(snap):
    """완전히 관측된 만실 구간만 추출 (앞뒤가 모두 '병상 있음'으로 관측된 것).

    W = (만실 뒤 처음 병상 있음 관측 시각) - (만실 첫 관측 시각). 양 끝이 잘린 구간은 제외(검열).
    """
    rows = []
    for (hpid, name), g in snap.dropna(subset=["A"]).groupby(["hpid", "dutyName"]):
        for seg in segments(g.sort_values("t")):
            a, t = seg["A"].to_numpy(), seg["t"].to_numpy()
            k = 1
            while k < len(a):
                if a[k] == 0 and a[k - 1] == 1:
                    start = k
                    while k < len(a) and a[k] == 0:
                        k += 1
                    if k < len(a):   # 끝까지 관측됨
                        rows.append(dict(hpid=hpid, dutyName=name, start=t[start],
                                         W=(t[k] - t[start]) / np.timedelta64(1, "m")))
                k += 1
    return pd.DataFrame(rows)


def freed_per_hour(snap):
    """정각 단위 1시간 동안 늘어난 병상 수 합 Σ max(ΔB, 0). 관측이 충분한 시간만 사용."""
    dt = snap.sort_values("t")["t"].drop_duplicates().diff().dt.total_seconds().div(60).median()
    need = 0.8 * 60 / dt if dt and dt > 0 else 1
    rows = []
    for hpid, g in snap.dropna(subset=["B"]).groupby("hpid"):
        for seg in segments(g.sort_values("t")):
            inc = seg["B"].diff().clip(lower=0)
            hour = seg["t"].dt.floor("h")
            per = pd.DataFrame({"hour": hour, "inc": inc}).iloc[1:]
            agg = per.groupby("hour").agg(freed=("inc", "sum"), n=("inc", "size"))
            agg = agg[agg["n"] >= need]
            rows.append(agg.assign(hpid=hpid).reset_index())
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), dt


def poisson_pmf(x, a):
    return np.exp(-a) * a ** x / np.array([factorial(int(v)) for v in np.atleast_1d(x)])


def erlang_table(fh):
    """병원별 λ (분당 풀린 병상 수) → P[x분 안에 k개 풀림] (Erlang CDF)."""
    lam = fh.groupby("hpid")["freed"].mean() / 60
    rows = []
    for hpid, l in lam.items():
        row = {"hpid": hpid, "lambda_per_hour": l * 60}
        for k in ERLANG_K:
            for x in ERLANG_X:
                n = np.arange(k)
                row[f"P[{k}개 ≤{x}분]"] = 1 - poisson_pmf(n, l * x).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def memoryless_table(W):
    rows = []
    for s in MEMORYLESS_S:
        for t in MEMORYLESS_T:
            surv_s = (W > s).sum()
            rows.append(dict(s=s, t=t, n_survived_s=surv_s,
                             P_W_gt_s_plus_t_given_gt_s=(W > s + t).sum() / surv_s if surv_s else np.nan,
                             P_W_gt_t=(W > t).mean(),
                             exp_model=np.exp(-t / W.mean())))
    return pd.DataFrame(rows)


def wait_or_move(runs, snap, hosp):
    """만실 병원 i 에서: 기다리기 E[W_i] vs 가까운 j 로 옮기기 D_ij + (1-p_j)E[W_j] (모두 분)."""
    EW = runs.groupby("hpid")["W"].agg(["mean", "size"])
    EW_pool = runs["W"].mean()
    ew = lambda h: EW.loc[h, "mean"] if h in EW.index and EW.loc[h, "size"] >= MIN_RUNS else EW_pool
    p = snap.groupby("hpid")["A"].mean()
    D = travel_min(hospital_dist_matrix(hosp))
    ids, names = hosp["hpid"].tolist(), dict(zip(hosp["hpid"], hosp["dutyName"]))
    rows = []
    for a, hi in enumerate(ids):
        best = None
        for b in np.argsort(D[a])[1:ALT_NEAREST + 1]:
            hj = ids[b]
            cost = D[a, b] + (1 - p.get(hj, 0.5)) * ew(hj)
            if best is None or cost < best[1]:
                best = (hj, cost, D[a, b])
        wait = ew(hi)
        rows.append(dict(hospital=names[hi], E_wait=wait, best_alt=names[best[0]],
                         travel=best[2], E_move=best[1],
                         decision="기다리기" if wait <= best[1] else "옮기기"))
    return pd.DataFrame(rows)


def plot_W(W, lam):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(W, bins=30, density=True, alpha=0.6, label="실제 만실 지속시간")
    x = np.linspace(0, W.max(), 200)
    a1.plot(x, lam * np.exp(-lam * x), "r", label=f"Exponential(λ={lam:.3f}/분)")
    a1.set_xlabel("W (분)"); a1.set_ylabel("밀도"); a1.set_title("Q3. 만실 지속시간 PDF"); a1.legend()
    xs = np.sort(W)
    a2.semilogy(xs, 1 - np.arange(1, len(xs) + 1) / (len(xs) + 1), ".", ms=3, label="실제 P[W > w]")
    a2.semilogy(x, np.exp(-lam * x), "r", label="지수분포면 직선")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))   # 10^-k 대신 0.01 표기
    a2.set_xlabel("w (분)"); a2.set_title("생존함수 (로그축)"); a2.legend()
    save_fig(fig, "q3_W_exponential.png")


def plot_poisson(counts):
    a = counts.mean()
    xs = np.arange(0, int(counts.max()) + 1)
    emp = np.array([(counts == v).mean() for v in xs])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs - 0.2, emp, 0.4, label="실제")
    ax.bar(xs + 0.2, poisson_pmf(xs, a), 0.4, label=f"Poisson(α={a:.2f})")
    ax.set_xlabel("1시간 동안 풀린 병상 수"); ax.set_ylabel("확률")
    ax.set_title(f"Q3. Poisson 적합 (분산/평균 = {counts.var() / a:.2f})"); ax.legend()
    save_fig(fig, "q3_freed_poisson.png")


def main():
    snap = load_snapshots()
    hosp = load_hospitals(snap)
    runs = full_runs(snap)
    if runs.empty:
        print("[Q3] 완전히 관측된 만실 구간이 아직 없음 (데이터가 더 쌓여야 함)")
        return
    save_table(runs, "q3_full_runs.csv")
    W = runs["W"]
    lam = 1 / W.mean()
    plot_W(W, lam)
    save_table(memoryless_table(W), "q3_memoryless.csv")

    fh, dt = freed_per_hour(snap)
    print(f"[Q3] 완전 관측 만실 구간 {len(W)}개, E[W] = {W.mean():.1f}분, 표준편차 {W.std():.1f}분 "
          f"(지수분포면 평균 = 표준편차)")
    print(f"     관측 간격 중앙값 Δ = {dt:.1f}분 → 측정오차 표준편차 ≈ {np.sqrt(2 * dt ** 2 / 12):.1f}분")
    if not fh.empty:
        plot_poisson(fh["freed"])
        save_table(erlang_table(fh), "q3_erlang.csv")
        print(f"     시간당 풀린 병상 수 평균 {fh['freed'].mean():.2f}, 분산/평균 {fh['freed'].var() / fh['freed'].mean():.2f}")
    wm = save_table(wait_or_move(runs, snap, hosp), "q3_wait_or_move.csv")
    print(f"     기다리기가 유리한 병원 {(wm['decision'] == '기다리기').sum()}곳 / 옮기기 {(wm['decision'] == '옮기기').sum()}곳")


if __name__ == "__main__":
    main()
