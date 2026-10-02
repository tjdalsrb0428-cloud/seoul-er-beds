"""공통 모듈: 데이터 로드·전처리, 병원 좌표, 이동시간, 상태 조회.

용어 (보고서와 동일하게 사용)
- B_i(t) : 시각 t 병원 i 응급실 가용 병상 수 (hvec, 음수는 0으로 = 만실)
- A_i(t) : 병상 유무 = 1{B_i(t) >= 1}  → Bernoulli
- F_i    : 만실 사건 {B_i = 0}
"""
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "output"

# ---- 모델 가정 (발표 때 "가정값"이라고 명시할 것) -------------------------
SPEED_KMH = 30        # 구급차 도심 평균 속도
ROAD_FACTOR = 1.3     # 직선거리 → 실제 도로거리 보정
HANDLING_MIN = 5      # 병원 도착 후 수용 여부 확인·거절까지 걸리는 시간
MAX_GAP_MIN = 15      # 이보다 오래 관측이 비면 그 시각 상태는 '모름'(결측)
TRAIN_FRAC = 0.7      # 앞 70% 기간으로 추정, 뒤 30%로 평가

SLOT_BINS = [0, 6, 12, 18, 24]
SLOT_LABELS = ["새벽(0-6)", "오전(6-12)", "오후(12-18)", "저녁(18-24)"]

# 환자 출발지: 서울 25개 구청 (근사 좌표)
ORIGINS = {
    "종로구": (37.5735, 126.9790), "중구": (37.5641, 126.9979),
    "용산구": (37.5324, 126.9900), "성동구": (37.5634, 127.0369),
    "광진구": (37.5385, 127.0823), "동대문구": (37.5744, 127.0400),
    "중랑구": (37.6066, 127.0927), "성북구": (37.5894, 127.0167),
    "강북구": (37.6396, 127.0257), "도봉구": (37.6688, 127.0471),
    "노원구": (37.6542, 127.0568), "은평구": (37.6027, 126.9291),
    "서대문구": (37.5791, 126.9368), "마포구": (37.5663, 126.9019),
    "양천구": (37.5170, 126.8664), "강서구": (37.5509, 126.8495),
    "구로구": (37.4955, 126.8875), "금천구": (37.4569, 126.8955),
    "영등포구": (37.5264, 126.8962), "동작구": (37.5124, 126.9393),
    "관악구": (37.4784, 126.9516), "서초구": (37.4837, 127.0324),
    "강남구": (37.5172, 127.0473), "송파구": (37.5145, 127.1059),
    "강동구": (37.5301, 127.1238),
}

def _korean_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in have:
            return name
    return "DejaVu Sans"   # 한글 폰트가 없으면 글자가 네모로 나옴


logging.getLogger("matplotlib").setLevel(logging.ERROR)   # 로그축 '−' 글리프 경고 숨김
plt.rcParams["font.family"] = _korean_font()
plt.rcParams["axes.unicode_minus"] = False


# ---- 데이터 로드 -----------------------------------------------------------
def load_snapshots() -> pd.DataFrame:
    """data/YYYY-MM-DD.csv 전체를 읽어 한 행 = (수집 시각, 병원) 으로 정리."""
    files = sorted(DATA.glob("20*.csv"))
    if not files:
        raise FileNotFoundError(f"{DATA} 에 수집 데이터가 없음")
    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig", dtype={"hpid": str, "hvidate": str}) for f in files],
        ignore_index=True,
    )
    df["t"] = pd.to_datetime(df["collected_at"])
    df = df.drop_duplicates(["t", "hpid"]).sort_values(["hpid", "t"])

    df["hvec"] = pd.to_numeric(df["hvec"], errors="coerce")
    df["capacity"] = pd.to_numeric(df.get("hvs01"), errors="coerce")
    df["over"] = df["hvec"] < 0                      # 정원 초과(과밀)
    df["B"] = df["hvec"].clip(lower=0)               # 음수 → 0 (만실)
    df["A"] = np.where(df["B"].isna(), np.nan, (df["B"] >= 1).astype(float))

    upd = pd.to_datetime(df["hvidate"], format="%Y%m%d%H%M%S", errors="coerce")
    df["stale_min"] = (df["t"] - upd).dt.total_seconds() / 60   # 병원 입력 후 경과 시간

    df["hour"] = df["t"].dt.hour
    df["dow"] = df["t"].dt.dayofweek                 # 0=월
    df["weekend"] = df["dow"] >= 5
    df["slot"] = pd.cut(df["hour"], SLOT_BINS, right=False, labels=SLOT_LABELS)
    return df.reset_index(drop=True)


def load_hospitals(snap: pd.DataFrame) -> pd.DataFrame:
    """실시간 병상 데이터에 등장하는 병원만, 좌표 포함."""
    h = pd.read_csv(DATA / "hospitals.csv", encoding="utf-8-sig", dtype={"hpid": str})
    h = h.drop_duplicates("hpid")
    h["lat"] = pd.to_numeric(h["wgs84Lat"], errors="coerce")
    h["lon"] = pd.to_numeric(h["wgs84Lon"], errors="coerce")
    cap = snap.groupby("hpid")["capacity"].median()
    h = h[h["hpid"].isin(snap["hpid"].unique())].dropna(subset=["lat", "lon"]).copy()
    h["capacity"] = h["hpid"].map(cap)
    return h[["hpid", "dutyName", "dutyEmclsName", "lat", "lon", "capacity"]].reset_index(drop=True)


def split_time(snap: pd.DataFrame, frac: float = TRAIN_FRAC):
    """시간순 분할: 앞부분(추정용), 뒷부분(평가용)."""
    times = np.sort(snap["t"].unique())
    cut = times[int(len(times) * frac)] if len(times) > 1 else times[-1]
    return snap[snap["t"] < cut], snap[snap["t"] >= cut]


# ---- 거리·이동시간 ---------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def travel_min(km):
    """거리(km) → 예상 이동시간(분). D_j 의 가정 모형."""
    return km * ROAD_FACTOR / SPEED_KMH * 60


def hospital_dist_matrix(hosp: pd.DataFrame) -> np.ndarray:
    lat, lon = hosp["lat"].to_numpy(), hosp["lon"].to_numpy()
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def origin_dist(hosp: pd.DataFrame) -> pd.DataFrame:
    """행 = 출발지(구), 열 = 병원 hpid, 값 = km."""
    rows = {g: haversine_km(la, lo, hosp["lat"].to_numpy(), hosp["lon"].to_numpy())
            for g, (la, lo) in ORIGINS.items()}
    return pd.DataFrame(rows, index=hosp["hpid"]).T


# ---- 상태 조회 (시각 t 에 병원 i 의 상태) ----------------------------------
class StateTable:
    """스냅샷을 (시각 × 병원) 행렬로 만들어 임의 시각의 상태를 빠르게 조회.

    조회 규칙: 질의 시각 이전의 가장 최근 스냅샷 값. 단 MAX_GAP_MIN 보다 오래됐으면 NaN.
    """

    def __init__(self, snap: pd.DataFrame, value: str = "A"):
        wide = snap.pivot_table(index="t", columns="hpid", values=value, aggfunc="first")
        self.times = wide.index.to_numpy(dtype="datetime64[ns]")
        self.cols = {h: k for k, h in enumerate(wide.columns)}
        self.mat = wide.to_numpy(dtype=float)
        self.wide = wide

    def at(self, hpid, when):
        """when: numpy datetime64 (스칼라 또는 배열)."""
        when = np.asarray(when, dtype="datetime64[ns]")
        idx = np.searchsorted(self.times, when, side="right") - 1
        col = self.cols.get(hpid)
        out = np.full(when.shape, np.nan)
        if col is None:
            return out
        ok = idx >= 0
        age = np.where(ok, (when - self.times[np.clip(idx, 0, None)]) / np.timedelta64(1, "m"), np.inf)
        ok &= age <= MAX_GAP_MIN
        out[ok] = self.mat[idx[ok], col]
        return out


def p_by_hour(snap: pd.DataFrame) -> pd.DataFrame:
    """행 = hpid, 열 = 0..23시, 값 = P[A=1 | 시간대] 추정치. 관측 없는 시간은 전체 평균으로 채움."""
    tab = snap.groupby(["hpid", "hour"])["A"].mean().unstack("hour").reindex(columns=range(24))
    overall = snap.groupby("hpid")["A"].mean()
    return tab.apply(lambda r: r.fillna(overall[r.name]), axis=1)


def save_fig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def save_table(df, name, index=False):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=index, encoding="utf-8-sig")
    return df
