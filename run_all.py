"""전체 파이프라인 실행 스크립트.

사용법:
    python run_all.py
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
    ('베이스라인 평가', 'baseline.py'),
    ('피처 생성', 'features.py'),
    ('모델 비교 실험', 'model.py'),
    ('영향요인·오차 분석', 'error_analysis.py'),
    ('1주 앞 예측', 'forecast.py'),
]


def main():
    sys.path.insert(0, str(SRC))
    for i, (name, script) in enumerate(STEPS, 1):
        print(f'\n===== [{i}/{len(STEPS)}] {name} ({script}) =====')
        runpy.run_path(str(SRC / script), run_name='__main__')
    print('\n완료: 결과는 outputs/ 에 저장됨')


if __name__ == '__main__':
    main()
