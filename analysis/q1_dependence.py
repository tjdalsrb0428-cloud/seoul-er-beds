"""Q1 (Ch1) — 옆 병원도 꽉 찼을까? 앱 숫자를 믿어도 될까?

1. 병원별 만실 확률 P[F_i] (상대도수)
2. 인접 병원 쌍의 조건부확률 P[F_j | F_i] vs P[F_j], 독립 검사 P[F_i∩F_j] vs P[F_i]P[F_j]
3. 전확률 정리: P[A] = Σ_slot P[A | slot] P[slot]
4. 베이즈: 지금 '병상 있음'으로 보일 때 Δ분 뒤 도착하면 실제로 있을 확률
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import (StateTable, hospital_dist_matrix, load_hospitals, load_snapshots,
                    plt, save_fig, save_table)

K_NEIGHBORS = 3          # 각 병원마다 가장 가까운 3곳과 짝지음
BAYES_DELTAS = [10, 20, 30]  # 도착까지 걸리는 시간 Δ (분)


def full_prob(snap):
    """병원별 P[F_i] = 만실 관측 수 / 유효 관측 수, 과밀(음수) 비율도 함께."""
    g = snap.dropna(subset=["A"]).groupby(["hpid", "dutyName"])
    out = g.agg(n=("A", "size"), p_full=("A", lambda a: 1 - a.mean()),
                p_over=("over", "mean"), mean_B=("B", "mean"), var_B=("B", "var"))
    return out.reset_index().sort_values("p_full", ascending=False)


def pair_dependence(snap, hosp):
    """인접 병원 쌍 (i, j) 마다 조건부확률·독립성 비교."""
    F = 1 - StateTable(snap, "A").wide          # 만실 지시변수 (시각 × 병원)
    dist = hospital_dist_matrix(hosp)
    ids = hosp["hpid"].tolist()
    names = dict(zip(hosp["hpid"], hosp["dutyName"]))
    rows, seen = [], set()
    for a, hi in enumerate(ids):
        for b in np.argsort(dist[a])[1:K_NEIGHBORS + 1]:
            hj = ids[b]
            if (hj, hi) in seen or hi not in F or hj not in F:
                continue
            seen.add((hi, hj))
            both = F[[hi, hj]].dropna()
            if len(both) < 20:
                continue
            fi, fj = both[hi].astype(bool), both[hj].astype(bool)
            p_i, p_j, p_ij = fi.mean(), fj.mean(), (fi & fj).mean()
            p_j_given_i = (fi & fj).sum() / fi.sum() if fi.sum() else np.nan
            table = pd.crosstab(fi, fj).reindex(index=[False, True], columns=[False, True], fill_value=0)
            try:
                pval = chi2_contingency(table)[1]
            except ValueError:   # 한쪽이 항상 같은 값 → 검정 불가
                pval = np.nan
            rows.append(dict(hosp_i=names[hi], hosp_j=names[hj], km=round(dist[a, b], 2), n=len(both),
                             P_Fi=p_i, P_Fj=p_j, P_FiFj=p_ij, P_Fi_x_P_Fj=p_i * p_j,
                             P_Fj_given_Fi=p_j_given_i,
                             lift=p_j_given_i / p_j if p_j else np.nan, chi2_p=pval))
    return pd.DataFrame(rows)


def total_probability(snap):
    """병원별로 Σ_slot P[A|slot]P[slot] 이 전체 P[A] 와 같은지 확인."""
    s = snap.dropna(subset=["A"])
    rows = []
    for (hpid, name), g in s.groupby(["hpid", "dutyName"]):
        cond = g.groupby("slot", observed=True)["A"].mean()       # P[A | slot]
        prior = g["slot"].value_counts(normalize=True)            # P[slot]
        row = {"dutyName": name, "P_A_direct": g["A"].mean(),
               "P_A_total_prob": float((cond * prior.reindex(cond.index)).sum())}
        row.update({f"P_A|{k}": v for k, v in cond.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def bayes_display(snap):
    """S = 지금 보이는 상태 A(t), A' = 도착 시 상태 A(t+Δ).

    P[A'=1 | S=1] 를 (1) 데이터로 직접 세고 (2) 베이즈 정리 P[S=1|A'=1]P[A'=1]/P[S=1] 로 계산.
    """
    st = StateTable(snap, "A")
    rows = []
    for delta in BAYES_DELTAS:
        S_all, A_all = [], []
        for hpid in st.cols:
            s = st.at(hpid, st.times)
            a = st.at(hpid, st.times + np.timedelta64(delta, "m"))
            ok = ~np.isnan(s) & ~np.isnan(a)
            S_all.append(s[ok]); A_all.append(a[ok])
        S, A = np.concatenate(S_all), np.concatenate(A_all)
        if len(S) == 0 or S.sum() == 0:
            continue
        direct = A[S == 1].mean()
        p_s, p_a = S.mean(), A.mean()
        p_s_given_a = S[A == 1].mean() if A.sum() else np.nan
        rows.append(dict(delta_min=delta, n=len(S), P_S1=p_s, P_A1=p_a,
                         P_A1_given_S1_direct=direct,
                         P_A1_given_S1_bayes=p_s_given_a * p_a / p_s,
                         P_A1_given_S0=A[S == 0].mean() if (S == 0).any() else np.nan))
    return pd.DataFrame(rows)


def plot_pairs(pairs):
    if pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pairs["P_Fj"], pairs["P_Fj_given_Fi"], s=18, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="독립이면 y = x")
    ax.set_xlabel("P[F_j]  (병원 j 만실)")
    ax.set_ylabel("P[F_j | F_i]  (옆 병원 i 가 만실일 때)")
    ax.set_title("Q1. 인접 응급실 만실의 종속성")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    save_fig(fig, "q1_pairs.png")


def main():
    snap = load_snapshots()
    hosp = load_hospitals(snap)
    fp = save_table(full_prob(snap), "q1_full_prob.csv")
    pairs = save_table(pair_dependence(snap, hosp), "q1_pairs.csv")
    save_table(total_probability(snap), "q1_total_prob.csv")
    bayes = save_table(bayes_display(snap), "q1_bayes.csv")
    plot_pairs(pairs)

    print(f"[Q1] 병원 {len(fp)}곳, 평균 만실 확률 {fp['p_full'].mean():.3f}")
    if not pairs.empty:
        print(f"     인접 쌍 {len(pairs)}개, lift 중앙값 {pairs['lift'].median():.2f} "
              f"(1보다 크면 '같이 만실' 경향)")
    if not bayes.empty:
        print(bayes[["delta_min", "P_A1", "P_A1_given_S1_direct", "P_A1_given_S1_bayes"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
