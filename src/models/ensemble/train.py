"""앙상블: Ridge와 LightGBM 예측의 가중 평균 (학습 없음, 저장된 예측을 섞음).
Ridge 비율은 개발 폴드(5 to 8월) 고유일 MAE로만 고른다 (9월은 보지 않음)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, DEV_FOLDS, save, score, dev_summary   # noqa: E402

KEY = ['fset', 'clone', 'fold', 'dt']
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)   # Ridge 비율

lg = pd.read_csv(OUT / 'lgbm' / 'predictions.csv', parse_dates=['dt'])
rd = pd.read_csv(OUT / 'ridge' / 'predictions.csv', parse_dates=['dt'])
m = lg.merge(rd[KEY + ['p']].rename(columns={'p': 'p_ridge'}), on=KEY, validate='one_to_one')

search = []
for w in WEIGHTS:
    s = m.assign(p=(1 - w) * m.p + w * m.p_ridge)
    s = s[s.fold.isin(DEV_FOLDS) & ~s.is_clone]
    for (f, c), g in s.groupby(['fset', 'clone']):
        search.append({'fset': f, 'clone': c, 'ridge_w': w, **score(g)})
search = pd.DataFrame(search)
best = search.loc[search.groupby(['fset', 'clone']).MAE.idxmin()].set_index(['fset', 'clone']).ridge_w

w = m.set_index(['fset', 'clone']).index.map(best).values
pred = m.assign(model='ensemble', p=(1 - w) * m.p + w * m.p_ridge).drop(columns='p_ridge')
d = save(pred, 'ensemble')
search.to_csv(d / 'weight_search.csv', index=False, encoding='utf-8-sig')

print('저장 →', d)
print('선택된 Ridge 비율 (개발 폴드 고유일 MAE 최소)\n', best.to_string())
fk = search[(search.fset == 'full') & (search.clone == 'keep')]
print('\n[full·keep] Ridge 비율별 성적 (0 = LightGBM 단독, 1 = Ridge 단독)\n',
      fk[['ridge_w', 'MAE', 'RMSE', 'peakMAE']].round(1).to_string(index=False))
print('\n', dev_summary(pred).round(1).to_string())
