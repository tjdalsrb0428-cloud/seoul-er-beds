"""서울특별시 응급실 실시간 가용 병상 수집기 (GitHub Actions에서 주기 실행)

- 실행 1번 = 서울 전체 응급실 스냅샷 1번 → data/YYYY-MM-DD.csv 에 이어 붙임
- 병원 위치(hospitals.csv)가 없으면 처음 한 번 같이 받아 둠
- 첫 실행 때 API 원본 응답을 data/sample_response.xml 로 남겨 필드 확인용으로 씀
- 인증키는 환경변수 SERVICE_KEY 에서 읽음 (코드에 직접 쓰지 않기!)
"""
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

SERVICE_KEY = os.environ.get("SERVICE_KEY", "").strip()
BASE = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/"
RLTM_URL = BASE + "getEmrrmRltmUsefulSckbdInfoInqire"  # 실시간 가용병상
LIST_URL = BASE + "getEgytListInfoInqire"              # 응급의료기관 목록(좌표 포함)

DATA_DIR = "data"
HOSP_PATH = os.path.join(DATA_DIR, "hospitals.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_response.xml")
KST = timezone(timedelta(hours=9))  # 한국은 서머타임 없음

# 저장할 실시간 필드. 이름이 틀리거나 응답에 없으면 빈 칸으로 저장됨
# (sample_response.xml 보고 나중에 추가/수정)
RLTM_FIELDS = [
    "hpid", "dutyName", "hvidate",
    "hvec",    # 응급실 가용 병상 (핵심 B_i)
    "hvoc",    # 수술실
    "hvgc",    # 입원실
    "hvicc",   # 일반 중환자실
    "hvcc",    # 신경 중환자실
    "hvncc",   # 신생아 중환자실
    "hvccc",   # 흉부 중환자실
    "hvctayn", "hvmriayn", "hvangioayn", "hvventiayn",  # 장비 가용 Y/N
    "hvamyn",  # 구급차 가용 Y/N
]
HOSP_FIELDS = ["hpid", "dutyName", "dutyEmclsName", "dutyAddr",
               "wgs84Lat", "wgs84Lon", "dutyTel3"]


def call(url, params, retries=3):
    """API 호출 + 실패 시 재시도. 성공하면 XML root 반환."""
    params = {"serviceKey": SERVICE_KEY, **params}
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            root = ET.fromstring(r.content)
            code = root.findtext("header/resultCode")
            if code != "00":
                raise RuntimeError(f"API 오류 {code}: {root.findtext('header/resultMsg')}")
            return root, r.content
        except Exception as e:  # 네트워크/파싱/API 오류 모두 재시도
            last = e
            print(f"  시도 {attempt}/{retries} 실패: {e}")
            if attempt < retries:
                time.sleep(15)
    raise RuntimeError(f"호출 최종 실패: {last}")


def fetch_all(url, params):
    """페이지를 넘기며 전체 item 수집."""
    items, page = [], 1
    while True:
        root, raw = call(url, {**params, "pageNo": page, "numOfRows": 100})
        if page == 1 and not os.path.exists(SAMPLE_PATH) and url == RLTM_URL:
            with open(SAMPLE_PATH, "wb") as f:
                f.write(raw)
        batch = list(root.iterfind("body/items/item"))
        items.extend(batch)
        total = int(root.findtext("body/totalCount") or 0)
        if not batch or len(items) >= total:
            return items
        page += 1


def append_csv(path, header, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerows(rows)


def collect_hospitals():
    items = fetch_all(LIST_URL, {"Q0": "서울특별시"})
    rows = [[it.findtext(k) or "" for k in HOSP_FIELDS] for it in items]
    append_csv(HOSP_PATH, HOSP_FIELDS, rows)
    print(f"병원 목록 {len(rows)}곳 저장 → {HOSP_PATH}")


def collect_snapshot():
    now = datetime.now(KST)
    items = fetch_all(RLTM_URL, {"STAGE1": "서울특별시"})
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    rows = [[stamp] + [it.findtext(k) or "" for k in RLTM_FIELDS] for it in items]
    path = os.path.join(DATA_DIR, now.strftime("%Y-%m-%d") + ".csv")
    append_csv(path, ["collected_at"] + RLTM_FIELDS, rows)
    print(f"[{stamp} KST] {len(rows)}곳 저장 → {path}")
    return len(rows)


def main():
    if not SERVICE_KEY:
        sys.exit("환경변수 SERVICE_KEY가 없음 (GitHub Secrets 또는 $env:SERVICE_KEY 설정 필요)")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(HOSP_PATH):
        try:
            collect_hospitals()
        except Exception as e:  # 위치 정보 실패해도 병상 수집은 계속
            print("병원 목록 수집 실패(다음 실행 때 재시도):", e)

    if collect_snapshot() == 0:
        sys.exit("응답에 병원이 0곳 — 파라미터/키 확인 필요")


if __name__ == "__main__":
    main()
