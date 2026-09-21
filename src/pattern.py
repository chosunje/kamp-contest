"""반복 패턴 분석: 동일한 24시간 전력 프로파일이 어떤 조건에서 반복되는가."""
import pandas as pd, numpy as np
df = pd.read_csv('outputs/eda/clean_preview.csv', parse_dates=['dt'])
d = df.groupby('날짜').agg(prof=('target', lambda s: tuple(s)), day=('day', 'first'), m=('m', 'first'),
                          prod=('생산량', 'sum'), ppl=('공장인원', 'sum'), tmean=('기온', 'mean'),
                          pmean=('target', 'mean'), pmax=('target', 'max')).reset_index()
d['pid'] = d.prof.map({p: i for i, p in enumerate(d.prof.drop_duplicates())})
sz = d.pid.value_counts()
d['cnt'] = d.pid.map(sz)
print('고유 프로파일 수', d.pid.nunique(), '/ 일수', len(d))
print('반복(2회+) 프로파일 수', (sz > 1).sum(), '  크기 분포', sz.value_counts().sort_index().to_dict())
top = sz.head(8).index
g = d[d.pid.isin(top)].groupby('pid').agg(일수=('날짜', 'size'), 요일=('day', lambda s: dict(s.value_counts().sort_index())),
    월=('m', lambda s: sorted(set(s))), 평균전력=('pmean', 'mean'), 최대=('pmax', 'max'),
    생산량합_평균=('prod', 'mean'), 생산량합_표준편차=('prod', 'std'), 시작=('날짜', 'min'), 끝=('날짜', 'max'))
print(g.round(1).to_string())
# 같은 프로파일 안에서 생산량이 서로 다른가? (전력이 생산량을 따르지 않는 증거)
rep = d[d.cnt > 1]
print('반복 프로파일 내 일 생산량합 변동계수 평균', (rep.groupby('pid')['prod'].std() / rep.groupby('pid')['prod'].mean()).mean().round(2))
# 연속 반복인가 (인접일)
d = d.sort_values('날짜'); d['same_prev'] = d.pid == d.pid.shift(1)
print('전날과 동일 프로파일 일수', d.same_prev.sum())
# 날짜 기준 프로파일 전환점
print('프로파일 변경 횟수', (d.pid != d.pid.shift(1)).sum() - 1)
# 반복 여부 vs 요일
print(d.groupby('day').apply(lambda x: (x.cnt > 1).mean()).round(2).to_dict())
# 반복일 vs 고유일 평균전력
print(d.groupby(d.cnt > 1).pmean.mean().round(1).to_dict())
# 고유(1회) 일수 월별
print(d[d.cnt == 1].groupby('m').size().to_dict(), ' / 반복', d[d.cnt > 1].groupby('m').size().to_dict())
d.drop(columns='prof').to_csv('outputs/daily_pattern.csv', index=False)
