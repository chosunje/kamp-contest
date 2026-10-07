"""전체 파이프라인 실행 스크립트.

사용법:
    python run_all.py

모델 비교(model.py)는 기본 설정 24개 x 시드 3개 x 월별 폴드를 반복 학습한다.
실행 시간은 환경에 따라 달라진다. 전체 파이프라인은 27단계다.
전처리·EDA만 다시 만들려면 STEPS 앞쪽 5개만 개별 실행하면 된다.

이 스크립트 하나로 outputs/ 를 전부 다시 만들 수 있어야 한다. 중간 산출물을
저장소에서 지워도 심사위원이 소스와 원자료만으로 재현할 수 있게 하기 위해서다.
"""
import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
STEPS = [
    ('전처리', 'preprocess.py'),
    ('EDA 시각화', 'eda.py'),
    ('기저부하·피크 통계', 'eda3.py'),
    ('반복 패턴 분석', 'pattern.py'),
    ('피처 생성', 'features.py'),
    ('베이스라인 (목표선)', 'baseline.py'),
    ('모델 비교 실험', 'model.py'),
    # report_ch2.py 가 model_results_fix.csv 를 읽는다. 여기서 안 만들면
    # 중간 산출물을 지운 저장소에서 24번 단계가 파일을 못 찾고 멈춘다
    ('약점 공략 실험 (보고서 2장 표 2-9)', 'model.py', ['--fix']),
    ('예측오차 분석', 'error_analysis.py'),
    ('영향요인 분석 (SHAP)', 'shap_analysis.py'),
    # report_ch2.py 는 아래 두 결과가 없으면 해당 표를 조용히 건너뛴다. 빠지면
    # 표 2-7·2-11·2-12, 그림 2-3·2-4 의 CSV 가 재현되지 않는다
    ('변수 중요도 + 피크 구간 SHAP (보고서 2장 표 2-11·2-12)', 'importance.py'),
    ('하이퍼파라미터 민감도 (보고서 2장 표 2-7)', 'hparam_check.py'),
    ('최대피크 위험조건 분석', 'peak_driver.py'),
    ('15분 최대 경보 검증 (D10)', 'peak_alarm_max15.py'),
    ('피크 경보 검증', 'peak_alarm.py'),
    ('분위 비교·누수 검증', 'quantile_analysis.py'),
    ('피크전력 저감방안', 'peak_reduce.py'),
    ('성능 체감 그림', 'viz_perf.py'),
    ('1주 앞 예측 + 피크 경보', 'forecast.py'),
    ('테스트 예측 파일 생성', 'predict_test.py'),
    ('최종 성능 한 장 요약', 'report_final.py'),
    ('최종 모델·저감 일정·HTML 화면', 'final_app.py'),
    ('최대피크 한 장 요약', 'peak_report.py'),
    ('보고서 1장 표·그림 데이터', 'report_ch1.py'),
    ('보고서 2장 표·그림 데이터', 'report_ch2.py'),
    ('보고서 3장 표·그림 데이터', 'report_ch3.py'),
    ('보고서 4장 표·그림 데이터', 'report_ch4.py'),
]

# 파이프라인에 넣지 않은 것 (오래 걸리거나 일회성 실험이라 따로 돌린다)
#   reinforce.py          6차 강화 실험               약 25분
#   report_gain.py        개선폭 비교 (01_dashboard)   약 6분
#   peak_schedule.py      하루 안 배분 조정 실험       약 3분
#   operational_correction.py  D23 운영규칙 실험 (팀원)


def main():
    sys.path.insert(0, str(SRC))
    for i, step in enumerate(STEPS, 1):
        name, script = step[0], step[1]
        args = step[2] if len(step) > 2 else []
        print(f'\n===== [{i}/{len(STEPS)}] {name} '
              f'({" ".join([script, *args])}) =====')
        saved = sys.argv                    # 스크립트가 sys.argv 를 보는 경우가 있다
        sys.argv = [str(SRC / script), *args]
        try:
            runpy.run_path(str(SRC / script), run_name='__main__')
        finally:
            sys.argv = saved
    print('\n완료: 결과는 outputs/ 에 저장됨')


if __name__ == '__main__':
    main()
