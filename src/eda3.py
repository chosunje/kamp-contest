"""EDA 3차: 생산량=0 구간 전력 구조, 피크와 생산/인원 관계."""
import pandas as pd, numpy as np
df = pd.read_csv('outputs/eda/clean_preview.csv', parse_dates=['dt'])
t = 'target'
z = df[df.생산량 == 0]
print('생산량=0 행수', len(z), '전력 분위수\n', z[t].quantile([.1,.25,.5,.75,.9]).to_dict())
print('생산량=0, 요일별 평균전력', z.groupby('day')[t].mean().round(1).to_dict())
print('생산량=0, 시간별 평균전력', z.groupby('시간')[t].mean().round(0).to_dict())
print('생산량=0 & 공장인원>0 & 전력>100:', ((df.생산량==0)&(df.공장인원>0)&(df[t]>100)).sum())
# 생산량=0인데 전력 높은 케이스 (인원/요일)
hi = z[z[t] > 100]; print('생산0 전력>100:', len(hi), '요일', hi.day.value_counts().sort_index().to_dict(),
      '평균인원', hi.공장인원.mean().round(2))
# 피크 vs 비피크
thr = df[t].quantile(.95); df['peak'] = df[t] >= thr
print(df.groupby('peak')[['생산량','공장인원','기온','습도','풍속']].mean().round(2))
# 생산량 구간별 평균 전력 (가동시)
b = pd.qcut(df[df.생산량>0].생산량, 5, duplicates='drop')
print(df[df.생산량>0].groupby(b)[t].agg(['mean','max']).round(1))
# 인원 구간별
p = df[df.공장인원>0]; bp = pd.qcut(p.공장인원, 5, duplicates='drop')
print(p.groupby(bp)[t].agg(['mean','max']).round(1))
# 일 단위: 일 최대 전력과 일 총 생산량
d = df.groupby('날짜').agg(pmax=(t,'max'), prod=('생산량','sum'), ppl=('공장인원','sum'), t=('기온','mean'))
print(d.corr(method='spearman')['pmax'].round(3).to_dict())
# 동시가동: 피크 시점 생산량 분포
print('피크 시 생산량 분위', df[df.peak].생산량.quantile([.1,.5,.9]).to_dict())
print('생산량 상위10%인데 피크 아님 비율', (df[df.생산량>=df.생산량.quantile(.9)].peak==False).mean().round(3))
