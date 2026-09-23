"""베이스라인 4종 시간순 롤링 평가, 복제일/고유일 분리. 예측 마감: 전날 24시 (lag24 이상 사용 가능)."""
import pandas as pd, numpy as np
from preprocess import load, ROOT

df = load()
t = 'target'
df['lag24'] = df[t].shift(24); df['lag168'] = df[t].shift(168)
df['ym'] = df.dt.dt.to_period('M').astype(str)
FOLDS = ['2021-05', '2021-06', '2021-07', '2021-08', '2021-09']
GROUPS = {'전체': lambda s: s, '복제일': lambda s: s[s.is_clone], '고유일': lambda s: s[~s.is_clone]}

preds = []
for m in FOLDS:   # 각 월을 검증, 그 이전 전체를 학습. 가동 중단·기록 손상 행은 학습·평가 모두 제외
    va = df[(df.ym == m) & df.train_ok]
    tr = df[(df.dt < va.dt.min()) & df.train_ok]
    thr = tr[t].quantile(.95)   # 피크 임계: 학습 구간 기준 상위 5%
    prof = tr.groupby(['day', '시간'])[t].mean()
    cand = {
        'lag24': va.lag24.values,
        'lag168': va.lag168.values,
        '요일x시간 평균': np.array([prof.get((d, h), tr[t].mean()) for d, h in zip(va.day, va.시간)]),
        '전체 평균': np.full(len(va), tr[t].mean()),
    }
    for k, p in cand.items():
        preds.append(pd.DataFrame({'model': k, 'fold': m, 'y': va[t].values, 'p': p,
                                   'peak': va[t].values >= thr, 'is_clone': va.is_clone.values}))
P = pd.concat(preds, ignore_index=True)


def score(s):
    e = s.y - s.p
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()),
                      'peakMAE': e[s.peak].abs().mean(), 'n': len(s)})


rows = []
for g, f in GROUPS.items():
    for (k, m), s in f(P).groupby(['model', 'fold']):
        rows.append({'model': k, 'group': g, 'fold': m, **score(s)})
    for k, s in f(P).groupby('model'):
        rows.append({'model': k, 'group': g, 'fold': 'ALL', **score(s)})
res = pd.DataFrame(rows)
res.to_csv(ROOT / 'outputs' / 'baseline_results.csv', index=False, encoding='utf-8-sig')

order = ['lag168', '요일x시간 평균', 'lag24', '전체 평균']
for g in GROUPS:
    r = res[res.group == g]
    piv = r.pivot(index='model', columns='fold', values='MAE').reindex(order).round(1)
    piv['n'] = r[r.fold == 'ALL'].set_index('model').n.reindex(order).astype(int)
    print(f'\n[{g}] MAE (ALL = 전체 폴드 합산)\n', piv.to_string())
summ = res[res.fold == 'ALL'].pivot(index='model', columns='group', values=['MAE', 'RMSE', 'peakMAE'])
print('\n[폴드 합산 요약]\n', summ.reindex(order).round(1).to_string())
