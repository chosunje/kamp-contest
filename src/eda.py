"""EDA: 시간 복구/결측 처리 후 패턴 시각화 및 통계 출력."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
OUT = 'outputs/eda/'

df = pd.read_csv('dataset/okm_augumented_2021.csv')
df['시간'] = df.groupby('날짜').cumcount()          # 손상된 시간 복구 (행 순서)
df['dt'] = pd.to_datetime(df['날짜'].astype(str)) + pd.to_timedelta(df['시간'], unit='h')
df = df.sort_values('dt').reset_index(drop=True)
df[['풍속', '강수량']] = df[['풍속', '강수량']].interpolate()
df['공장인원'] = df['공장인원'].fillna(0)
df['target'] = df['평균']
df.to_csv('outputs/eda/clean_preview.csv', index=False)

# 1) 전체 시계열 + 일 최대
fig, ax = plt.subplots(2, 1, figsize=(14, 7))
ax[0].plot(df.dt, df.target, lw=.4); ax[0].set_title('시간별 전력(평균)')
d = df.groupby(df.dt.dt.date).target.max()
ax[1].plot(d.index, d.values); ax[1].set_title('일 최대 전력')
plt.tight_layout(); plt.savefig(OUT + '01_timeseries.png', dpi=110); plt.close()

# 2) 시간대 x 요일 히트맵
h = df.pivot_table(index='day', columns='시간', values='target', aggfunc='mean')
fig, ax = plt.subplots(figsize=(12, 4)); im = ax.imshow(h, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(7)); ax.set_yticklabels(['월','화','수','목','금','토','일'])
ax.set_xticks(range(24)); ax.set_title('요일×시간 평균 전력'); plt.colorbar(im)
plt.tight_layout(); plt.savefig(OUT + '02_heatmap.png', dpi=110); plt.close()

# 3) 생산량-전력 산점, 기온-전력
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(df.생산량, df.target, s=3, alpha=.4); ax[0].set_title('생산량 vs 전력')
ax[1].scatter(df.기온, df.target, s=3, alpha=.4); ax[1].set_title('기온 vs 전력')
plt.tight_layout(); plt.savefig(OUT + '03_scatter.png', dpi=110); plt.close()

# 4) 분포
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df.target, bins=60); ax[0].set_title('전력 분포')
ax[1].hist(df.생산량[df.생산량 > 0], bins=60); ax[1].set_title('생산량 분포(>0)')
plt.tight_layout(); plt.savefig(OUT + '04_dist.png', dpi=110); plt.close()

# 통계
num = ['target', '생산량', '기온', '풍속', '습도', '강수량', '공장인원', '전기요금(계절)']
print(df[num].corr(method='spearman')['target'].round(3))
print('생산량0 & 전력>0 비율:', ((df.생산량 == 0) & (df.target > 20)).mean().round(3))
print('전력 lag1 상관:', df.target.corr(df.target.shift(1)).round(3),
      ' lag24:', df.target.corr(df.target.shift(24)).round(3),
      ' lag168:', df.target.corr(df.target.shift(168)).round(3))
q = df.target.quantile([.9, .95, .99]); print('분위수\n', q)
p = df[df.target >= q[.95]]
print('상위5% 요일분포\n', p.day.value_counts(normalize=True).sort_index().round(3).to_dict())
print('상위5% 시간분포\n', p.시간.value_counts().sort_index().to_dict())
print('상위5% 월분포\n', p.m.value_counts().sort_index().to_dict())
print('생산량 상관(생산량>0):', df[df.생산량 > 0][['생산량', 'target']].corr(method='spearman').iloc[0, 1].round(3))
print('시간 복구 확인:', df.groupby('날짜').시간.nunique().eq(24).all())
