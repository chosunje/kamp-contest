"""검증 전략(시간순 롤링 폴드) + 베이스라인 평가. 가정: 예측 시점 기준 24h 이상 앞(day-ahead)."""
import pandas as pd, numpy as np
df = pd.read_csv('outputs/eda/clean_preview.csv', parse_dates=['dt']).sort_values('dt').reset_index(drop=True)
t = 'target'
df['lag24'] = df[t].shift(24); df['lag168'] = df[t].shift(168)
df['ym'] = df.dt.dt.to_period('M')
thr = df[t].quantile(.95)   # 피크 기준(전체 기준 근사, 추후 학습구간 기준으로 재정의)

def metrics(y, p):
    e = y - p; pk = y >= thr
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()),
                peakMAE=np.abs(e[pk]).mean(), n=len(y))

rows = []
for m in ['2021-05', '2021-06', '2021-07', '2021-08', '2021-09']:   # 각 월을 검증, 그 이전 전체를 학습
    va = df[df.ym == m]; tr = df[df.dt < va.dt.min()]
    prof = tr.groupby(['day', '시간'])[t].mean()
    preds = {
        'lag24': va.lag24,
        'lag168': va.lag168,
        'profile(요일×시간 평균)': [prof.get((d, h), tr[t].mean()) for d, h in zip(va.day, va.시간)],
        'global_mean': np.full(len(va), tr[t].mean()),
    }
    for k, p in preds.items():
        r = metrics(va[t].values, np.asarray(p, float)); r.update(model=k, fold=m); rows.append(r)
res = pd.DataFrame(rows)
piv = res.pivot(index='model', columns='fold', values='MAE').round(1)
piv['평균'] = piv.mean(axis=1).round(1); print('MAE\n', piv)
print('RMSE평균\n', res.groupby('model').RMSE.mean().round(1).to_dict())
print('피크MAE평균\n', res.groupby('model').peakMAE.mean().round(1).to_dict())
res.to_csv('outputs/baseline_results.csv', index=False)
