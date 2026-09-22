"""Ridge 선형회귀: 단순 모델 대표. 시각·요일 등은 원-핫, 수치형은 결측 대체 + 결측 표시 + 표준화."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import load_features, run_cv, save, dev_summary, GRID   # noqa: E402

CATEGORICAL = ['hour', 'dow', 'shift', 'month']


def fit_predict(tr, va, w, cols):
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
    ])
    m = make_pipeline(pre, Ridge(alpha=1.0))
    m.fit(tr[cols], tr.target, ridge__sample_weight=w)
    return m.predict(va[cols])


X = load_features()
pred = run_cv(X, 'ridge', fit_predict, GRID)
print('저장 →', save(pred, 'ridge'))
print(dev_summary(pred).round(1).to_string())
