"""전체 분석 실행:  python analysis/run_all.py
결과(표·그림·앱 데이터)는 analysis/output/ 에 저장됨.
"""
import q1_dependence
import q2_attempts
import q3_waiting
import export_app
from common import load_snapshots

if __name__ == "__main__":
    snap = load_snapshots()
    print(f"데이터: {snap['t'].min()} ~ {snap['t'].max()}, 스냅샷 {snap['t'].nunique()}회, "
          f"병원 {snap['hpid'].nunique()}곳, 음수(과밀) 비율 {snap['over'].mean():.1%}\n")
    for mod in (q1_dependence, q2_attempts, q3_waiting, export_app):
        mod.main()
        print()
