"""하이퍼파라미터 "단일 변경" 민감도 — model.py 의 FIX 29 to 43 번과 짝이다.

  ★ 이 스크립트만 보고 "튜닝은 효과 없다" 고 결론내면 안 된다.
    여기서는 한 번에 한 항목만 바꾼다 → 변동이 ±0.8 이내로 작게 보인다.
    그러나 FIX 실험에서 얕은 나무(num_leaves 15 + min_data_in_leaf 40)와
    느린 학습을 함께 적용하면 CORE MAE 7.15 → 6.79 로 분명히 내려간다.
    즉 이 데이터에서 파라미터 효과는 단독이 아니라 조합에서 나온다.
    두 결과를 같이 제시하는 것이 정직하다 (보고서에서도 그렇게 쓴다).

본 연구는 하이퍼파라미터를 탐색하지 않고 기본값 수준으로 고정했다. 그 선택이
타당했는지를 보이려면 "조정해도 크게 달라지지 않는다"를 측정해야 한다.
한 번에 한 항목씩만 바꿔 최종 후보 설정에서 돌린다 (model.py 의 비교 원칙과 동일).

실행: python src/hparam_check.py  →  outputs/hparam_results.csv
"""
import numpy as np, pandas as pd

from features import build, FEATURES
from model import CORE, CORE_FOLDS, SEEDS, LGBM_BASE, rolling_eval, score
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'hparam_results.csv'

# 최종 후보 설정 (error_analysis.py 와 동일). 여기서 파라미터만 흔든다
SETTING = dict(cols=FEATURES['h7_full'], model='lgbm',
               train_flag='train_ok_strict', clone='weight', clone_w=0.3)

GRID = [
    ('기본값 (현재 설정)', {}),
    ('트리 수 200', {'n_estimators': 200}),
    ('트리 수 800', {'n_estimators': 800}),
    ('학습률 0.03', {'learning_rate': .03}),
    ('학습률 0.10', {'learning_rate': .10}),
    ('리프 수 15', {'num_leaves': 15}),
    ('리프 수 63', {'num_leaves': 63}),
    ('리프 최소 표본 10', {'min_data_in_leaf': 10}),
    ('리프 최소 표본 40', {'min_data_in_leaf': 40}),
    ('목적함수 L2 (참고)', {'objective': 'l2'}),
]


def main():
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    rows = []
    for name, over in GRID:
        per_seed = []
        for sd in SEEDS:
            P = rolling_eval(X, seed=sd, params=over, **SETTING)
            per_seed.append(score(P[P.fold.isin(CORE_FOLDS)]))
        s = pd.DataFrame(per_seed)
        rows.append({'설정': name,
                     '변경 항목': ', '.join(f'{k}={v}' for k, v in over.items()) or '-',
                     'MAE': round(s.MAE.mean(), 2), '흔들림': round(s.MAE.std(), 2),
                     'RMSE': round(s.RMSE.mean(), 2), 'peakMAE': round(s.peakMAE.mean(), 2),
                     'n': int(s.n.iloc[0])})
        print(f'  {name:20s} MAE {rows[-1]["MAE"]:.2f}', flush=True)

    R = pd.DataFrame(rows)
    R['기본값 대비'] = (R.MAE - R.MAE.iloc[0]).round(2)
    R = R[['설정', '변경 항목', 'MAE', '흔들림', '기본값 대비', 'RMSE', 'peakMAE', 'n']]
    R.to_csv(OUT, index=False, encoding='utf-8-sig')

    print(f'\n[{CORE} 기준] 최종 후보 설정에서 파라미터만 변경 · 시드 {list(SEEDS)} 평균')
    print(R.to_string(index=False))
    spread = R.MAE.max() - R.MAE.min()
    print(f'\n전체 변동 폭 {spread:.2f} (최저 {R.MAE.min():.2f} to 최고 {R.MAE.max():.2f})')
    print(f'기본 설정 LightGBM: {LGBM_BASE}')
    print('→', OUT)


if __name__ == '__main__':
    main()
