"""Q2 (Ch3) — 몇 곳을 돌아야 병상이 나올까?

과거 스냅샷을 재현(replay): 출발지(구청)에서 시각 t0 에 출발 → 방문 순서대로 이동하며
'도착 시각'의 상태 A_i(t0 + 누적 이동시간)를 확인. 처음 A=1 인 병원에서 멈춤.

1. N (시도 병원 수) 경험적 PMF/CDF vs Geometric(p) vs 독립·이질 p_i 모형
2. P[N <= n] >= 0.95 가 되는 최소 n
3. K = 가까운 5곳 중 병상 있는 곳 수: Binomial / 독립 모형 대비 분산 (과대산포 = 종속)
4. Pascal: 병상 3개(환자 3명) 확보까지 시도 수
5. Jensen: E[1/p] >= 1/E[p] — 평균 확률로 계산하면 시도 수를 과소평가
6. MMSE: 도착 시 병상 수 예측 — 지금 숫자 vs 시간대 평균
7. 방문 전략 비교 (가까운 순 / 확률 순 / p÷시간 지수 순)
"""
from math import comb

import numpy as np
import pandas as pd

from common import (HANDLING_MIN, StateTable, hospital_dist_matrix, load_hospitals,
                    load_snapshots, origin_dist, p_by_hour, plt, save_fig, save_table,
                    split_time, travel_min)

MAX_TRY = 10           # 10곳까지 시도해도 실패면 '실패(검열)'로 기록
POOL = 10              # 확률 순·지수 순 전략은 가까운 10곳 안에서만 고름
REPLAY_EVERY_MIN = 30  # 출발 시각 간격 (연속 시각끼리 너무 비슷한 표본 줄이기)
K_NEAREST = 5          # Binomial 분석에 쓸 가까운 병원 수
PASCAL_K = 3
MMSE_DELTA = 20        # 분


# ---- 재현(replay) ----------------------------------------------------------
class Replayer:
    def __init__(self, test_snap, hosp, p_hour):
        self.st = StateTable(test_snap, "A")
        self.ids = hosp["hpid"].to_numpy()
        self.hd = travel_min(hospital_dist_matrix(hosp))         # 병원 → 병원 (분)
        self.od = travel_min(origin_dist(hosp))                   # 구청 → 병원 (분)
        self.p_hour = p_hour.reindex(self.ids).fillna(0.5).to_numpy()  # (병원, 24)
        times = pd.Series(self.st.times)
        self.starts = times[times.dt.floor(f"{REPLAY_EVERY_MIN}min") == times].to_numpy()
        if len(self.starts) == 0:
            self.starts = self.st.times

    def _next(self, strategy, origin_row, cur, remaining, hour):
        """다음 방문 병원 index 선택."""
        rem = np.fromiter(remaining, int)
        d = origin_row[rem] if cur is None else self.hd[cur, rem]
        if strategy == "nearest":
            return rem[np.argmin(d)]
        p = self.p_hour[rem, hour]
        if strategy == "prob":
            return rem[np.argmax(p)]
        if strategy == "index":   # p / (이동 + 확인 시간) 최대 — 순차 탐색 최적 지수
            return rem[np.argmax(p / (d + HANDLING_MIN))]
        raise ValueError(strategy)

    def run(self, strategy, k_success=1):
        rows = []
        for origin, origin_row in self.od.iterrows():
            orow = origin_row.to_numpy()
            pool = np.argsort(orow)[: (len(orow) if strategy == "nearest" else POOL)]
            for t0 in self.starts:
                hour = pd.Timestamp(t0).hour
                remaining, cur, clock = set(pool.tolist()), None, 0.0
                got, n, ok, p_seq = 0, 0, True, []
                while remaining and n < MAX_TRY:
                    nxt = self._next(strategy, orow, cur, remaining, hour)
                    remaining.discard(nxt)
                    clock += orow[nxt] if cur is None else self.hd[cur, nxt]
                    a = self.st.at(self.ids[nxt], t0 + np.timedelta64(int(clock * 60), "s"))
                    if np.isnan(a):
                        ok = False
                        break
                    n += 1
                    p_seq.append(self.p_hour[nxt, hour])
                    if a == 1:
                        got += 1
                        if got == k_success:
                            break
                    clock += HANDLING_MIN   # 거절(또는 다음 환자용 추가 탐색) 처리 시간
                    cur = nxt
                if not ok:
                    continue
                success = got == k_success
                rows.append(dict(origin=origin, t0=t0, hour=hour, strategy=strategy,
                                 N=n if success else np.nan, T=clock if success else np.nan,
                                 success=success, p_seq=p_seq))
        return pd.DataFrame(rows)


# ---- 분포 비교 -------------------------------------------------------------
def pmf_table(trials, max_n=MAX_TRY):
    """경험적 PMF vs Geometric(1/N̄) vs 독립·이질 모형 Π(1-p_j) p_n."""
    n_vals = np.arange(1, max_n + 1)
    emp = np.array([(trials["N"] == n).mean() for n in n_vals])
    N_ok = trials["N"].dropna()
    p_mle = 1 / N_ok.mean() if len(N_ok) else np.nan
    geom = (1 - p_mle) ** (n_vals - 1) * p_mle
    het = np.zeros(max_n)
    for seq in trials["p_seq"]:
        seq = list(seq) + [seq[-1] if seq else 0.5] * (max_n - len(seq))
        surv = 1.0
        for k in range(max_n):
            het[k] += surv * seq[k]
            surv *= 1 - seq[k]
    het /= max(len(trials), 1)
    return pd.DataFrame({"n": n_vals, "empirical": emp, "geometric": geom, "independent_hetero": het,
                         "emp_CDF": emp.cumsum(), "geom_CDF": geom.cumsum()}), p_mle


def binomial_K(snap, hosp, p_train):
    """K = 출발지에서 가까운 5곳 중 A=1 인 곳 수. 경험 vs 독립 모형(시간대 p 사용)."""
    st = StateTable(snap, "A")
    od = origin_dist(hosp)
    p_hour = p_by_hour(p_train).reindex(hosp["hpid"]).fillna(0.5)
    hours = pd.DatetimeIndex(st.times).hour.to_numpy()
    emp_counts = np.zeros(K_NEAREST + 1)
    model = np.zeros(K_NEAREST + 1)
    for _, row in od.iterrows():
        near = row.sort_values().index[:K_NEAREST]
        cols = [st.cols[h] for h in near if h in st.cols]
        if len(cols) < K_NEAREST:
            continue
        M = st.mat[:, cols]
        ok = ~np.isnan(M).any(axis=1)
        K = M[ok].sum(axis=1).astype(int)
        emp_counts += np.bincount(K, minlength=K_NEAREST + 1)
        for hr, cnt in zip(*np.unique(hours[ok], return_counts=True)):
            pmf = np.array([1.0])
            for h in near:                        # 포아송-이항: 독립 Bernoulli 합의 PMF
                p = p_hour.loc[h, hr]
                pmf = np.convolve(pmf, [1 - p, p])
            model += cnt * pmf
    emp = emp_counts / emp_counts.sum() if emp_counts.sum() else emp_counts
    model = model / model.sum() if model.sum() else model
    k = np.arange(K_NEAREST + 1)
    mean_e, mean_m = (k * emp).sum(), (k * model).sum()
    p_bar = mean_e / K_NEAREST
    binom =np.array([comb(K_NEAREST, x) * p_bar ** x * (1 - p_bar) ** (K_NEAREST - x) for x in k])
    stats = dict(E_emp=mean_e, Var_emp=((k - mean_e) ** 2 * emp).sum(),
                 E_indep=mean_m, Var_indep=((k - mean_m) ** 2 * model).sum(),
                 Var_binom=K_NEAREST * p_bar * (1 - p_bar))
    return pd.DataFrame({"k": k, "empirical": emp, "independent_by_hour": model, "binomial": binom}), stats


def jensen_table(p_train):
    """병원별 E_h[1/p_h] vs 1/E_h[p_h]. Geometric 평균 시도 수 1/p 에 Jensen 적용."""
    ph = p_by_hour(p_train).clip(lower=0.02)    # p=0 이면 1/p 무한대 → 하한 0.02
    names = p_train.drop_duplicates("hpid").set_index("hpid")["dutyName"]
    out = pd.DataFrame({"dutyName": names.reindex(ph.index),
                        "inv_mean_p": 1 / ph.mean(axis=1),          # 1/E[p]
                        "mean_inv_p": (1 / ph).mean(axis=1)})        # E[1/p]
    out["underestimate"] = out["mean_inv_p"] - out["inv_mean_p"]
    return out.sort_values("underestimate", ascending=False).reset_index()


def mmse_table(train, test):
    """B(t+Δ) 예측: 지금 값 B(t) / 시간대 평균 / 전체 평균 의 MSE 비교."""
    stB = StateTable(test, "B")
    hour_mean = train.groupby(["hpid", "hour"])["B"].mean()
    all_mean = train.groupby("hpid")["B"].mean()
    err = {"snapshot_B(t)": [], "hour_mean": [], "overall_mean": []}
    hours = pd.DatetimeIndex(stB.times).hour
    for h in stB.cols:
        now = stB.at(h, stB.times)
        fut = stB.at(h, stB.times + np.timedelta64(MMSE_DELTA, "m"))
        ok = ~np.isnan(now) & ~np.isnan(fut)
        hm = np.array([hour_mean.get((h, hr), all_mean.get(h, np.nan)) for hr in hours])
        err["snapshot_B(t)"].append((now - fut)[ok])
        err["hour_mean"].append((hm - fut)[ok])
        err["overall_mean"].append((all_mean.get(h, np.nan) - fut)[ok])
    return pd.DataFrame([{"predictor": k, "MSE": np.nanmean(np.concatenate(v) ** 2)}
                         for k, v in err.items()])


def strategy_summary(trials):
    def agg(g):
        T = g["T"].dropna()
        return pd.Series({"trials": len(g), "fail_rate": 1 - g["success"].mean(),
                          "E[N]": g["N"].mean(), "Var[N]": g["N"].var(),
                          "E[T]": T.mean(), "median_T": T.median(), "T_90%": T.quantile(0.9),
                          "P[N<=1]": (g["N"] <= 1).mean(), "P[N<=2]": (g["N"] <= 2).mean(),
                          "P[N<=3]": (g["N"] <= 3).mean()})
    return trials.groupby("strategy").apply(agg).reset_index()


# ---- 그림 ------------------------------------------------------------------
def plot_pmf(tab, p_mle, n95):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.27
    a1.bar(tab["n"] - w, tab["empirical"], w, label="실제 재현")
    a1.bar(tab["n"], tab["geometric"], w, label=f"Geometric(p={p_mle:.2f})")
    a1.bar(tab["n"] + w, tab["independent_hetero"], w, label="독립·병원별 p_i 모형")
    a1.set_xlabel("n (시도한 응급실 수)"); a1.set_ylabel("P[N = n]"); a1.legend()
    a1.set_title("Q2. N 의 PMF")
    a2.step(tab["n"], tab["emp_CDF"], where="post", label="실제")
    a2.step(tab["n"], tab["geom_CDF"], where="post", label="Geometric")
    a2.axhline(0.95, color="r", ls="--", lw=1)
    if n95:
        a2.axvline(n95, color="r", ls=":", lw=1)
    a2.set_xlabel("n"); a2.set_ylabel("P[N ≤ n]"); a2.set_title(f"CDF  (95% 확보: {n95}곳)")
    a2.legend()
    save_fig(fig, "q2_N_pmf_cdf.png")


def plot_K(tab, stats):
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.27
    ax.bar(tab["k"] - w, tab["empirical"], w, label=f"실제 (Var={stats['Var_emp']:.2f})")
    ax.bar(tab["k"], tab["independent_by_hour"], w, label=f"독립 모형 (Var={stats['Var_indep']:.2f})")
    ax.bar(tab["k"] + w, tab["binomial"], w, label=f"Binomial (Var={stats['Var_binom']:.2f})")
    ax.set_xlabel(f"K = 가까운 {K_NEAREST}곳 중 병상 있는 곳 수"); ax.set_ylabel("확률")
    ax.set_title("Q2. Binomial 대비 과대산포 = 병원 간 종속"); ax.legend()
    save_fig(fig, "q2_K_binomial.png")


def plot_jensen(jt):
    top = jt.head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(top))
    ax.bar(x - 0.2, top["inv_mean_p"], 0.4, label="1 / E[p]  (평균 확률로 계산)")
    ax.bar(x + 0.2, top["mean_inv_p"], 0.4, label="E[1/p]  (시간대별로 계산 후 평균)")
    ax.set_xticks(x, top["dutyName"].str.slice(0, 8), rotation=45, ha="right")
    ax.set_ylabel("기대 시도 횟수"); ax.set_title("Q2. Jensen 부등식: 평균 p 는 시도 수를 과소평가")
    ax.legend()
    save_fig(fig, "q2_jensen.png")


def main():
    snap = load_snapshots()
    hosp = load_hospitals(snap)
    train, test = split_time(snap)
    if train.empty:
        train = test = snap
    p_hour = p_by_hour(train)

    rp = Replayer(test, hosp, p_hour)
    trials = pd.concat([rp.run(s) for s in ("nearest", "prob", "index")], ignore_index=True)
    if trials.empty:
        print("[Q2] 재현 가능한 표본이 아직 없음 (데이터가 더 쌓여야 함)")
        return
    save_table(trials.drop(columns="p_seq"), "q2_trials.csv")
    summary = save_table(strategy_summary(trials), "q2_strategies.csv")

    near = trials[trials["strategy"] == "nearest"]
    tab, p_mle = pmf_table(near)
    save_table(tab, "q2_N_pmf.csv")
    hit = tab.index[tab["emp_CDF"] >= 0.95]
    n95 = int(tab.loc[hit[0], "n"]) if len(hit) else None
    plot_pmf(tab, p_mle, n95)

    ktab, kstats = binomial_K(test, hosp, train)
    save_table(ktab, "q2_K.csv")
    plot_K(ktab, kstats)

    pas = rp.run("nearest", k_success=PASCAL_K)
    if not pas.empty:
        save_table(pas.drop(columns="p_seq"), "q2_pascal_trials.csv")

    jt = save_table(jensen_table(train), "q2_jensen.csv")
    plot_jensen(jt)
    mm = save_table(mmse_table(train, test), "q2_mmse.csv")

    print(f"[Q2] 재현 표본 {len(near)}건 (가까운 순), Geometric p̂ = {p_mle:.3f}, 95% 확보 n = {n95}")
    print(summary[["strategy", "E[N]", "E[T]", "T_90%", "P[N<=2]", "fail_rate"]].round(3).to_string(index=False))
    print(f"     K 분산: 실제 {kstats['Var_emp']:.2f} / 독립 모형 {kstats['Var_indep']:.2f} / Binomial {kstats['Var_binom']:.2f}")
    if not pas.empty:
        print(f"     Pascal(k={PASCAL_K}): E[N] = {pas['N'].mean():.2f}, 실패율 {1 - pas['success'].mean():.3f}")
    print(f"     Jensen 과소평가 최대 {jt['underestimate'].max():.2f}곳 ({jt.loc[0, 'dutyName']})")
    print(mm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
