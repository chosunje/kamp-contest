"""베이스라인 4종 (규칙 기반, 학습 없음 또는 평균만)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_features, run_cv, save, dev_summary   # noqa: E402

X = load_features()
X['raw_lag24'] = X.target.shift(24)     # 규칙 그대로의 lag (휴무·중단 결측 처리 없음)
X['raw_lag168'] = X.target.shift(168)


def profile(tr, va, w, cols):
    prof = tr.groupby(['dow', 'hour']).target.mean()
    return [prof.get((d, h), tr.target.mean()) for d, h in zip(va.dow, va.hour)]


RULES = {
    'lag168': lambda tr, va, w, cols: va.raw_lag168.values,
    '요일x시간 평균': profile,
    'lag24': lambda tr, va, w, cols: va.raw_lag24.values,
    '전체 평균': lambda tr, va, w, cols: np.full(len(va), tr.target.mean()),
}
cfg = [{'fset': '-', 'clone': 'keep'}]
pred = pd.concat([run_cv(X, f'baseline_{k}', f, cfg) for k, f in RULES.items()], ignore_index=True)
print('저장 →', save(pred, 'baseline'))
print(dev_summary(pred).round(1).to_string())
