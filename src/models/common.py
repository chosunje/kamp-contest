"""모델 공통 평가 틀: 시간순 폴드, 복제일 처리 방식, 예측 저장, 지표.
개발 폴드 5 to 8월(모델 선택·조정용) + 최종 확인 9월(모든 선택이 끝난 뒤 한 번만 확인)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from features import build, FEATURES   # noqa: E402
from preprocess import ROOT             # noqa: E402

OUT = ROOT / 'outputs' / 'models'
DEV_FOLDS = ['2021-05', '2021-06', '2021-07', '2021-08']
HOLDOUT = '2021-09'
CLONE_WEIGHT = {'keep': 1.0, 'down': 0.3, 'drop': 0.0}
GRID = [{'fset': f, 'clone': c} for f in ('full', 'no_plan') for c in ('keep', 'down', 'drop')]


def load_features() -> pd.DataFrame:
    X = build()
    X['ym'] = X.dt.dt.to_period('M').astype(str)
    return X


def run_cv(X, model, fit_predict, configs):
    """fit_predict(tr, va, w, cols) -> 예측 배열. 학습·평가 모두 train_ok 행만 사용."""
    out = []
    for cfg in configs:
        cols = FEATURES.get(cfg['fset'], [])
        for fold in DEV_FOLDS + [HOLDOUT]:
            va = X[(X.ym == fold) & X.train_ok]
            tr = X[(X.dt < va.dt.min()) & X.train_ok]
            thr = tr.target.quantile(.95)
            w = np.where(tr.is_clone, CLONE_WEIGHT[cfg['clone']], 1.0)
            tr, w = tr[w > 0], w[w > 0]
            p = np.clip(np.asarray(fit_predict(tr, va, w, cols), float), 0, None)
            out.append(pd.DataFrame({
                'model': model, 'fset': cfg['fset'], 'clone': cfg['clone'], 'fold': fold,
                'dt': va.dt.values, 'y': va.target.values, 'p': p,
                'peak': va.target.values >= thr, 'is_clone': va.is_clone.values}))
    return pd.concat(out, ignore_index=True)


def save(pred: pd.DataFrame, name: str):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / 'predictions.csv', index=False, encoding='utf-8-sig')
    return d


def score(s: pd.DataFrame) -> pd.Series:
    e = s.y - s.p
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()),
                      'peakMAE': e[s.peak].abs().mean(), 'n': len(s)})


def dev_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """개발 폴드(5 to 8월) 합산, 고유일 기준 성적."""
    s = pred[pred.fold.isin(DEV_FOLDS) & ~pred.is_clone]
    return s.groupby(['model', 'fset', 'clone']).apply(score, include_groups=False).sort_values('MAE')
