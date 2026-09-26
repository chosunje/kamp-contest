"""LightGBM: 결정나무를 차례로 쌓는 부스팅 모델 (주력). 결측값은 그대로 입력."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightgbm import LGBMRegressor
from common import load_features, run_cv, save, dev_summary, GRID   # noqa: E402

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)
importance = []


def fit_predict(tr, va, w, cols):
    m = LGBMRegressor(**PARAMS).fit(tr[cols], tr.target, sample_weight=w)
    importance.append(pd.DataFrame({'feature': cols, 'gain': m.booster_.feature_importance('gain'),
                                    'n_cols': len(cols), 'fold_end': va.dt.max()}))
    return m.predict(va[cols])


X = load_features()
pred = run_cv(X, 'lgbm', fit_predict, GRID)
d = save(pred, 'lgbm')
imp = pd.concat(importance)
imp = imp[imp.n_cols == imp.n_cols.max()]   # full 세트 기준
imp = imp.groupby('feature').gain.mean().sort_values(ascending=False)
(imp / imp.sum()).rename('gain_share').to_csv(d / 'feature_importance.csv', encoding='utf-8-sig')
print('저장 →', d)
print(dev_summary(pred).round(1).to_string())
print('\n변수 중요도 상위 10 (full 세트, 전 폴드 평균 gain 비율)\n', (imp / imp.sum()).head(10).round(3).to_string())
