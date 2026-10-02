# 분석 코드

```bash
pip install -r analysis/requirements.txt
git pull                      # 최신 수집 데이터 받기
python analysis/run_all.py    # 결과 → analysis/output/
```

| 파일 | 챕터 | 내용 |
|---|---|---|
| `common.py` | – | 데이터 로드·전처리 (음수 병상 → 0 = 만실), 좌표·이동시간, 상태 조회, **가정값** |
| `q1_dependence.py` | Ch1 | 만실 확률, 인접 병원 조건부확률·독립성, 전확률 정리, 베이즈 |
| `q2_attempts.py` | Ch3 | 과거 재현으로 N·T, Geometric/Binomial/Pascal 비교, Jensen, MMSE, 방문 전략 비교 |
| `q3_waiting.py` | Ch4 | 만실 지속시간 W 지수분포·무기억성, Poisson, Erlang, 기다리기 vs 옮기기 |
| `export_app.py` | – | 기말 시연 앱용 `app_data.json` |

가정값(속도 30km/h, 도로 보정 1.3, 확인 시간 5분 등)은 `common.py` 맨 위에서 바꾼다.
발표·보고서에는 이 값들이 측정값이 아니라 가정값임을 명시할 것.
