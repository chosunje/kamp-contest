"""EDA 시각화: 모든 그림을 복제일(증강으로 전력곡선이 복사된 날) vs 고유일로 나눠 그린다."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from preprocess import load, ROOT

OUT = ROOT / 'outputs' / 'eda'
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {True: '#2a78d6', False: '#eb6834'}
NAME = {True: '복제일', False: '고유일'}
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#c43a39', '#f0efec', '#1c5cab'])
WEEK = ['월', '화', '수', '목', '금', '토', '일']

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': BASE, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


NOTE = dict(fontsize=9.5, color=INK2, bbox=dict(boxstyle='round,pad=.3', fc=SURF, ec='none'),
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def head(fig, title, sub):
    fig.text(.01, 1.0, title, fontsize=16, fontweight='bold', va='bottom')
    fig.text(.01, .985, sub, fontsize=11, color=INK2, va='top')


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


df = load()
days = df.groupby('날짜').agg(date=('dt', 'min'), mx=('target', 'max'), mean=('target', 'mean'),
                             clone=('is_clone', 'first'), m=('m', 'first'), prod=('생산량', 'sum'),
                             gid=('clone_gid', 'first'), n=('clone_n', 'first'))
cnt = days.clone.value_counts()
LEG = {k: f'{NAME[k]} ({cnt[k]}일)' for k in (True, False)}

# 01 일 최대전력 추이 + 월별 복제 구성
fig, (a, b) = plt.subplots(2, 1, figsize=(15, 8.5), gridspec_kw={'height_ratios': [2.2, 1], 'hspace': .45})
a.plot(days.date, days.mx, color=BASE, linewidth=1, zorder=1)
for k in (True, False):
    s = days[days.clone == k]
    a.scatter(s.date, s.mx, s=26, color=COLOR[k], edgecolor=SURF, linewidth=1, label=LEG[k], zorder=2)
a.set_ylim(0, 235); a.set_ylabel('일 최대전력'); grid(a)
a.axvspan(pd.Timestamp('2021-07-31'), pd.Timestamp('2021-08-09'), color=GRID, alpha=.6, zorder=0)
a.annotate('7/31 to 8/8 연속 휴무\n(전력 23 수준, 하계휴가 추정)', (pd.Timestamp('2021-08-04'), 30),
           xytext=(pd.Timestamp('2021-08-04'), 60), ha='center', **NOTE)
a.annotate('7/27 to 7/30 연중 최대 207 to 208', (pd.Timestamp('2021-07-28'), 208),
           xytext=(pd.Timestamp('2021-06-15'), 226), **NOTE)
a.annotate('1월 화·수, 3월 금요일 평일 무가동 (모두 복제일)', (pd.Timestamp('2021-01-13'), 24),
           xytext=(pd.Timestamp('2021-01-20'), 50), **NOTE)
a.legend(loc='upper left', ncol=2, bbox_to_anchor=(0, 1.02))
a.set_title('일 최대전력 (점 1개 = 하루)', pad=26)

mc = days.groupby(['m', 'clone']).size().unstack(fill_value=0)
x = mc.index.values
b.bar(x, mc[True], width=.62, color=COLOR[True], edgecolor=SURF, linewidth=2, label=NAME[True])
b.bar(x, mc[False], width=.62, bottom=mc[True], color=COLOR[False], edgecolor=SURF, linewidth=2, label=NAME[False])
for xi in x:
    t, f = mc.loc[xi, True], mc.loc[xi, False]
    if t >= 3: b.text(xi, t / 2, t, ha='center', va='center', color='white', fontsize=9.5)
    if f >= 3: b.text(xi, t + f / 2, f, ha='center', va='center', color='white', fontsize=9.5)
b.set_xticks(x, [f'{i}월' for i in x]); b.set_ylabel('일수'); grid(b)
b.set_title('월별 복제일·고유일 구성')
head(fig, '1 to 6월은 대부분 복제일, 7월부터는 거의 고유일',
     '복제일: 24시간 전력곡선이 다른 날과 완전히 같은 날 (증강 데이터, 원본 구분 불가)')
save(fig, '01_timeseries.png')

# 02 요일×시간 평균 전력
fig, axes = plt.subplots(2, 1, figsize=(15, 8.2), gridspec_kw={'hspace': .45})
vmax = df.target.max()
for ax, k in zip(axes, (True, False)):
    s = df[df.is_clone == k]
    h = s.pivot_table(index='day', columns='시간', values='target', aggfunc='mean').reindex(range(1, 8))
    ax.imshow(h.values, cmap=SEQ, vmin=0, vmax=vmax, aspect='auto')
    for i in range(7):
        for j in range(24):
            v = h.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if v > vmax * .55 else INK)
    ax.set_xticks(range(0, 24), [str(i) for i in range(24)])
    ax.set_yticks(range(7), [f'{w} ({(s.drop_duplicates("날짜").day == i + 1).sum()}일)' for i, w in enumerate(WEEK)])
    ax.set_xticks(np.arange(-.5, 24), minor=True); ax.set_yticks(np.arange(-.5, 7), minor=True)
    ax.grid(which='minor', color=SURF, linewidth=1.5); ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f'{LEG[k]}', color=COLOR[k])
axes[1].set_xlabel('시각')
head(fig, '요일×시각 평균 전력: 두 그룹 모두 평일 8 to 11시, 13 to 19시에 부하 집중',
     '셀 숫자 = 평균 전력 · 12시 점심에 부하 하락 · 고유일은 화 to 토 새벽 1 to 6시 야간 가동(약 90), 월 새벽은 기저부하 · 같은 색 척도')
save(fig, '02_heatmap.png')

# 03 생산량·기온 vs 전력 (그룹별 분할)
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharey=True, gridspec_kw={'hspace': .5, 'wspace': .08})
XMAX = 4000
pbins = [0, 1, 100, 250, 500, 750, 1000, 1500, 2000, 3000, XMAX]
tbins = np.arange(-14, 36, 3)
for col, k in enumerate((True, False)):
    s = df[df.is_clone == k]
    for row, (v, bins, xl) in enumerate([('생산량', pbins, '시간당 생산량'), ('기온', tbins, '기온 (°C)')]):
        ax = axes[row, col]
        ax.scatter(s[v].clip(upper=XMAX) if v == '생산량' else s[v], s.target, s=7, alpha=.3,
                   color=COLOR[k], linewidth=0)
        # 기온은 가동/무가동이 섞이면 중앙값이 두 덩어리 사이를 오가므로 가동 시간(전력>40)만 사용
        base = s if v == '생산량' else s[s.target > 40]
        med = base.groupby(pd.cut(base[v], bins, right=False)).agg(x=(v, 'median'), y=('target', 'median'), n=('target', 'size'))
        med = med[med.n >= 30]
        ax.plot(med.x, med.y, color=INK, linewidth=2, marker='o', markersize=5,
                markerfacecolor=SURF, markeredgewidth=1.5,
                label='구간 중앙값' if v == '생산량' else '가동 시간 구간 중앙값')
        r = s[v].corr(s.target, method='spearman')
        ax.set_title(f'{NAME[k]} · {xl}   ρ = {r:.2f}', color=COLOR[k])
        ax.set_xlabel(xl); grid(ax, 'both')
        if col == 0: ax.set_ylabel('전력 (평균)')
        if v == '생산량':
            over = (s[v] > XMAX).sum()
            ax.text(.99, .03, f'{XMAX:,} 초과 {over}건은 오른쪽 끝에 표시', transform=ax.transAxes,
                    ha='right', fontsize=9, color=MUTED)
axes[0, 0].legend(loc='lower right', bbox_to_anchor=(1, .08))
axes[1, 0].legend(loc='upper left')
head(fig, '생산량 약 800에서 전력 포화 · 가동 시간 기준 기온 22°C를 넘으면 전력 상승 (냉방 부하 추정)',
     'ρ = 스피어만 상관 (가동·무가동 시간이 섞여 기온 ρ는 낮게 나옴) · 생산량 0 근처 세로 띠 = 생산 없이 전력을 쓰는 시간 · '
     '고유일은 7 to 9월 위주라 기온 범위가 좁음')
save(fig, '03_scatter.png')

# 04 분포
fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'wspace': .22})
for k in (True, False):
    s = df[df.is_clone == k]
    a.hist(s.target, bins=np.arange(0, 215, 5), density=True, histtype='step', linewidth=2,
           color=COLOR[k], label=LEG[k])
    p = s.생산량[s.생산량 > 0]
    b.hist(np.log10(p), bins=40, density=True, histtype='step', linewidth=2, color=COLOR[k],
           label=f'{NAME[k]} (생산 0 비율 {(s.생산량 == 0).mean():.0%})')
a.set_xlabel('전력 (평균)'); a.set_ylabel('밀도'); grid(a); a.legend(loc='upper right')
a.set_title('전력 분포')
a.annotate('기저부하 약 23 (무가동)', (25, .045), xytext=(45, .05), **NOTE)
a.annotate('중부하 약 105', (105, .0135), xytext=(80, .025), **NOTE)
a.annotate('고부하 약 165', (165, .0105), xytext=(160, .022), **NOTE)
b.set_xticks([0, 1, 2, 3, 4], ['1', '10', '100', '1,000', '10,000'])
b.set_xlabel('시간당 생산량 (로그 척도, 0 제외)'); grid(b); b.legend(loc='upper left')
b.set_title('생산량 분포')
head(fig, '전력은 기저부하·중부하·고부하 세 덩어리로 나뉨: 연속값보다 "가동 상태" 전환이 핵심',
     '밀도 기준이라 그룹 크기가 달라도 모양을 직접 비교 가능')
save(fig, '04_dist.png')

# 05 전력과의 상관 (그룹 비교)
VARS = ['생산량', '공장인원', '인건비', '기온', '습도', '풍속', '강수량', '전기요금(계절)']
LBL = {'전기요금(계절)': '전기요금', 'target': '전력(평균)'}
corr = {k: df[df.is_clone == k][['target'] + VARS].corr('spearman') for k in (True, False)}
t = pd.DataFrame({k: corr[k]['target'].drop('target') for k in (True, False)}).sort_values(False)
fig, ax = plt.subplots(figsize=(11, 6.2))
y = np.arange(len(t)); hgt = .36
for off, k in [(hgt / 2 + .02, True), (-hgt / 2 - .02, False)]:
    ax.barh(y + off, t[k], height=hgt, color=COLOR[k], label=LEG[k])
    for yi, v in zip(y + off, t[k]):
        ax.text(v + (.015 if v >= 0 else -.015), yi, f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9.5, color=INK)
ax.set_yticks(y, [LBL.get(c, c) for c in t.index]); ax.axvline(0, color=BASE, linewidth=1)
ax.set_xlim(-.45, 1); ax.set_xlabel('스피어만 상관계수 (전력과)'); grid(ax, 'x')
ax.legend(loc='lower right')
head(fig, '생산량·공장인원은 고유일에서 전력과 더 강하게 연결됨',
     '복제일은 전력곡선만 복사돼 생산 기록과 어긋난 행이 섞여 있음 · 전력 15분 세부값·시간·요일·월 제외')
save(fig, '05_corr_target.png')

# 06 변수 간 상관 행렬 (그룹별)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': .3})
cols = ['target'] + VARS; names = [LBL.get(c, c) for c in cols]; n = len(cols)
for ax, k in zip(axes, (True, False)):
    m = corr[k].values
    for i in range(1, n):
        for j in range(i):
            v = m[i, j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=DIV((v + 1) / 2), edgecolor=SURF, linewidth=2))
            ax.text(j + .5, i + .5, f'{v:.2f}', ha='center', va='center', fontsize=9.5,
                    color='white' if abs(v) >= .55 else INK)
    ax.set_xlim(0, n - 1); ax.set_ylim(n, 1)
    ax.set_xticks(np.arange(n - 1) + .5, names[:-1], rotation=40, ha='right')
    ax.set_yticks(np.arange(1, n) + .5, names[1:])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(LEG[k], color=COLOR[k])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIV, norm=plt.Normalize(-1, 1)), ax=axes, fraction=.02, pad=.02,
                  ticks=[-1, -.5, 0, .5, 1])
cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
head(fig, '변수 간 상관 (스피어만): 생산량과 공장인원은 두 그룹 모두 0.99로 사실상 같은 정보',
     '파랑 = 같이 커짐 · 빨강 = 반대로 움직임 · 회색 = 관계 없음')
save(fig, '06_corr_matrix.png')

# 07 복제 증거: 전력곡선은 같은데 생산량은 다른 날들
cand = days[(days.n >= 4) & (days['mean'] > 80)].groupby('gid')['prod'].agg(lambda s: s.std() / s.mean())
g = cand.idxmax()
grp = days[days.gid == g].sort_index()
hi_day, lo_day = grp['prod'].idxmax(), grp['prod'].idxmin()
md = lambda d: f'{str(d)[4:6].lstrip("0")}/{str(d)[6:].lstrip("0")}'
fig, (a, b) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'hspace': .35})
for d in grp.index:
    s = df[df.날짜 == d]
    hl = d in (hi_day, lo_day)
    b.plot(s.시간, s.생산량, color=INK if hl else BASE, linewidth=2.4 if hl else 1.4,
           linestyle='--' if d == lo_day else '-', zorder=3 if hl else 2)
s0 = df[df.날짜 == grp.index[0]]
a.plot(s0.시간, s0.target, color=COLOR[True], linewidth=2.4)
dates = ', '.join(f'{md(d)}({WEEK[r.date.dayofweek]})' for d, r in grp.iterrows())
a.set_title(f'전력: {dates} 모두 완전히 같은 곡선'); a.set_ylabel('전력 (평균)'); grid(a)
b.set_title('생산량: 날마다 다름'); b.set_ylabel('시간당 생산량'); b.set_xlabel('시각'); grid(b)
top = df[df.날짜 == hi_day].set_index('시간').생산량
b.annotate(f'{md(hi_day)} (일 생산량 {grp["prod"].max():,.0f})', (top.idxmax(), top.max()),
           xytext=(top.idxmax() + 1.5, top.max() * .9), **NOTE)
b.annotate(f'{md(lo_day)} (일 생산량 {grp["prod"].min():,.0f}), 그런데 전력은 위와 동일', (15, 0),
           xytext=(14, top.max() * .45), **NOTE)
b.text(.99, .96, '회색 = 같은 그룹의 나머지 날', transform=b.transAxes, ha='right', fontsize=9.5, color=MUTED)
b.set_xticks(range(0, 24, 2))
same_d = days[days.n > 1].groupby('gid').agg(k=('date', lambda s: s.dt.day.nunique() == 1), n=('n', 'size'))
head(fig, '복제의 증거: 전력곡선은 통째로 복사됐지만 생산량은 복사되지 않음',
     f'복제 그룹 {len(same_d)}개 중 {same_d.k.sum()}개({same_d[same_d.k].n.sum()}일)는 "다른 달의 같은 날짜"끼리 복사됨 · '
     '예: 매월 11일')
save(fig, '07_clone_evidence.png')

for k in (True, False):
    s = df[df.is_clone == k]
    print(f'{NAME[k]}: {s.날짜.nunique()}일, 전력 평균 {s.target.mean():.1f}, '
          f'생산0&전력>100 {((s.생산량 == 0) & (s.target > 100)).sum()}행, '
          f'ρ(생산량) {s.생산량.corr(s.target, method="spearman"):.2f}')
print('saved →', OUT)
