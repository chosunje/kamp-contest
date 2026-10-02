"""전체 파이프라인 실행 스크립트.

사용법:
    python run_all.py

모델 비교(model.py)는 설정 17개 x 시드 3개를 돌아 약 3분 걸린다.
전처리·EDA만 다시 만들려면 STEPS 앞쪽 5개만 개별 실행하면 된다.
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
    ('예측오차 분석', 'error_analysis.py'),
    ('영향요인 분석 (SHAP)', 'shap_analysis.py'),
    ('최대피크 위험조건 분석', 'peak_driver.py'),
    ('피크 경보 검증', 'peak_alarm.py'),
    ('성능 체감 그림', 'viz_perf.py'),
    ('1주 앞 예측 + 피크 경보', 'forecast.py'),
    ('테스트 예측 파일 생성', 'predict_test.py'),
    ('최종 성능 한 장 요약', 'report_final.py'),
]


def main():
    sys.path.insert(0, str(SRC))
    for i, (name, script) in enumerate(STEPS, 1):
        print(f'\n===== [{i}/{len(STEPS)}] {name} ({script}) =====')
        runpy.run_path(str(SRC / script), run_name='__main__')
    print('\n완료: 결과는 outputs/ 에 저장됨')


if __name__ == '__main__':
    main()
