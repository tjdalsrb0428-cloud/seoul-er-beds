# 서울 응급실 가용 병상 자동 수집

확률 및 랜덤변수 텀프로젝트용. GitHub Actions가 5분마다 공공데이터포털
「전국 응급의료기관 정보 조회 서비스」를 호출해 서울특별시 전체 응급실의
실시간 가용 병상을 `data/` 폴더에 쌓는다. PC를 켜 둘 필요 없음.

## 파일
| 파일 | 내용 |
|---|---|
| `collect.py` | 1회 실행 = 서울 전체 스냅샷 1회 저장 |
| `.github/workflows/collect.yml` | 5분마다 `collect.py` 실행 후 자동 커밋 |
| `data/YYYY-MM-DD.csv` | 날짜별 수집 데이터 (KST 기준) |
| `data/hospitals.csv` | 병원 목록·좌표 (첫 실행 때 1회 생성) |
| `data/sample_response.xml` | API 원본 응답 예시 (필드 확인용) |

## 데이터 컬럼
- `collected_at`: 우리가 수집한 시각 (KST)
- `hpid`, `dutyName`: 병원 ID, 병원명
- `hvidate`: 병원이 마지막으로 정보를 입력한 시각
- `hvec`: 응급실 가용 병상 수 → 분석의 B_i(t). **음수 = 정원 초과(과밀)**, 만실로 처리
- 그 외: 수술실·입원실·중환자실 가용 수, 장비/구급차 가용 여부

## 시연 앱 (`docs/`)
- `docs/index.html` 을 브라우저로 열면 바로 실행 (인터넷 연결 필요: 지도·실시간 데이터)
- 확률 추정치 `docs/app_data.js` 는 `update-app.yml` 이 6시간마다 자동 갱신
  (수동: `python analysis/export_app.py`)
- '지금 상태'는 이 저장소의 최신 `data/*.csv` 를 앱이 직접 읽음
- 시연용 바로가기: `index.html#h=병원ID` → 해당 병원 상세 화면

## 주의
- 인증키는 코드에 쓰지 말고 저장소 Settings → Secrets 의 `SERVICE_KEY` 로만 관리
- GitHub 예약 실행은 몇 분씩 밀리거나 가끔 빠짐 → 분석 시 실제 `collected_at` 간격 사용
- 이 데이터는 교육용 분석 목적이며 실제 응급 상황에서는 119에 연락할 것
