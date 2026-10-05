"""최대피크 예측·위험조건·경보를 한 장으로 — outputs/final/02_dashboard.png

  00_dashboard  시간별 예측 모델의 성능 (전반부)
  01_dashboard  지난 라운드에서 무엇이 좋아졌나
  02_dashboard  ★ 최대피크 예측 / 위험조건 / 경보 / 저감 (과제 제목의 뒤쪽 절반)

이 파일은 보여 주기만 하는 것이 아니라 이번에 보강한 네 가지를 같이 계산한다.

  보강 1  위험조건 분석      과제 제목의 "최대피크 위험조건" 에 직접 답한다.
                             기온 x 생산 교차표와 단일조건별 피크일 비율을 낸다.
                             상관 한 숫자가 아니라 "어떤 날이 위험한가" 로 답한다
  보강 2  경보 임계값 스윕    지금 임계는 "학습 구간 상위 5%" 라는 잠정값이다 (D10 미확정).
                             임계를 바꿔 가며 정밀도·재현율을 내어 두면 D10 이 정해지는
                             즉시 값만 바꿔 쓸 수 있다. 절대값(전력)도 같이 적는다
  보강 3  예측 시계 비교      경보를 1주 전에 낼 때와 하루 전에 낼 때가 얼마나 다른가.
                             현장은 하루 전이면 대응할 수 있으므로 둘을 같이 제시한다
  보강 4  15분 최대 기준 경보 요금은 시간 평균이 아니라 15분 최대로 매겨진다 ([6-6] 8번).
                             그 기준으로도 경보를 내 보고 시간 평균 기준과 비교한다

판단 기준은 전부 CORE(2021년 7-9월) 고유일 · 시드 3개 평균이다.

실행  python src/peak_report.py              캐시가 있으면 몇 초
      python src/peak_report.py --refresh    처음부터 계산 (약 4분)
출력  outputs/final/02_dashboard.png
      outputs/final/peak_predictions.csv     예측 캐시
      outputs/final/peak_summary.txt         그림의 숫자를 텍스트로
      outputs/peak_risk_cross.csv            위험조건 교차표 (기온 x 생산)
      outputs/peak_risk_single.csv           단일 조건별 피크일 비율
      outputs/peak_alarm_sweep.csv           임계값 스윕 — D10 ② 확정 시 여기서 고른다
      outputs/peak_alarm_leadtime.csv        예측 시계·기준별 경보 성능
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from features import FEATURES, build
from model import (ALARM_CFG, BASE, CLONE_W, CORE_FOLDS, FINAL_CFG, PEAK_W, SEEDS,
                   SHALLOW, SLOW, rolling_eval)
from peak_reduce import day_table, month_peak, ramp
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'final'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'peak_predictions.csv'
FIG = OUT / '02_dashboard.png'
TXT = OUT / 'peak_summary.txt'

SURF, INK, INK2, MUTED, GRID, BASEC = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
PRED, PEAK, HI = '#2a78d6', '#eb6834', '#1baf7a'    # 검증된 팔레트 1·2·3 슬롯
ALARM_Q = .95          # 지금 쓰는 잠정 임계 (D10 확정 전)
SWEEP = (.80, .85, .88, .90, .92, .95, .97, .98)

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10.5,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': BASEC, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 9, 'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def head(ax, title, sub):
    """패널 제목과 그 아래 설명 한 줄. 제목 여백을 넉넉히 줘서 둘이 겹치지 않게 한다."""
    ax.set_title(title, pad=27)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=MUTED)


# ── 설정 ────────────────────────────────────────────────────────────────────
# D-7(1주 앞)은 제출물 기준이다. D-1 은 전날 실적을 쓸 수 있어 피처가 더 많다
#   full  = lag24·전날 평균/최대 포함  → 하루 전에만 쓸 수 있다
#   h7_lag= 그것들을 뺀 세트           → 1주 전에도 쓸 수 있다  [4] 3번
D1_PT = dict(BASE, clone='weight', clone_w=CLONE_W, cols=FEATURES['full'], peak_w=PEAK_W,
             params={**SHALLOW, **SLOW, 'feature_fraction': .6})
D1_HI = dict(BASE, clone='weight', clone_w=CLONE_W, cols=FEATURES['full'],
             params={**SHALLOW, 'objective': 'quantile', 'alpha': .9})
M15_HI = dict(ALARM_CFG, target='target_max15')

CFGS = {'d7_pt': FINAL_CFG, 'd7_hi': ALARM_CFG, 'd1_pt': D1_PT, 'd1_hi': D1_HI,
        'm15_hi': M15_HI}


def predictions(X, refresh=False):
    """설정별 시드 평균 예측을 행 단위로 모아 돌려준다 (캐시)."""
    if CACHE.exists() and not refresh:
        d = pd.read_csv(CACHE, encoding='utf-8-sig')
        print(f'  캐시 사용: {CACHE.name} ({len(d)}행). 다시 계산하려면 --refresh')
        return d
    print('  예측 계산 중 (설정 5개 x 시드 3개 x 폴드 3개)...')
    base = None
    for name, cfg in CFGS.items():
        parts = [rolling_eval(X, folds=CORE_FOLDS, seed=s, **cfg).set_index('idx')
                 for s in SEEDS]
        p = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
        if base is None:
            base = parts[0][['fold', 'y']].copy()
            base['y15'] = X.loc[base.index, 'target_max15'].values
        base[name] = p
        print(f'    {name} 완료', flush=True)
    for c in ['날짜', 'hour', 'temp', 'prod', 'is_off', 'is_weekend', 'is_holiday',
              'day_prod', 'day_prod_hours', 'month']:
        base[c] = X.loc[base.index, c].values
    base = base.reset_index()
    base.to_csv(CACHE, index=False, encoding='utf-8-sig')
    print(f'  → {CACHE}')
    return base


def thresholds(X, q):
    """폴드마다 학습 구간에서만 임계값을 정한다 (평가 구간을 보고 정하면 누수).
    시간 평균 기준과 15분 최대 기준을 함께 낸다 — D10 ① 이 어느 쪽으로 정해져도 쓰게."""
    t = {}
    for m in CORE_FOLDS:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
        t[m] = (tr.target.quantile(q), tr.target_max15.quantile(q))
    return t


# ── 보강 1. 위험조건 ────────────────────────────────────────────────────────
def risk_conditions(X, q=ALARM_Q):
    """"어떤 날이 위험한가" 를 조건별 피크일 비율로 낸다.

    피크일 정의  그날 일 최대가 임계(학습 구간 상위 5% 시간 전력) 이상인 날
    왜 상관이 아니라 비율인가  [8-1] 결과 3 에서 봤듯이 상관 한 숫자는 피어슨/스피어만
      선택에 따라 뒤집힌다. "기온 26도 넘고 평일이면 10일 중 7일이 위험" 같은 서술이
      보고서에도 현장에도 쓸 수 있고 재현도 쉽다.
    예측 시점에 아는 값만 쓴다 (기온=예보, 생산=계획, 달력)
    """
    d = X[X.train_ok & ~X.is_clone].copy()
    thr = thresholds(X, q)
    day = d.groupby('날짜').agg(일최대=('target', 'max'), 기온=('temp', 'mean'),
                               최고기온=('temp', 'max'), 일생산=('prod', 'sum'),
                               생산시간=('day_prod_hours', 'first'),
                               주말=('is_weekend', 'first'), 공휴일=('is_holiday', 'first'),
                               월=('month', 'first'), ym=('ym', 'first'), n=('hour', 'size'))
    day = day[day.n == 24]
    day['임계'] = day.ym.map({k: v[0] for k, v in thr.items()})
    day = day.dropna(subset=['임계'])
    day['피크일'] = day.일최대 >= day.임계
    base_rate = day.피크일.mean()

    # 단일 조건별 비율
    rows = []
    conds = [
        ('평일 (월-금)', day.주말 == 0), ('주말', day.주말 == 1),
        ('최고기온 30도 초과', day.최고기온 > 30),
        ('최고기온 26-30도', day.최고기온.between(26, 30, inclusive='left')),
        ('최고기온 26도 이하', day.최고기온 <= 26),
        ('일생산 상위 1/3', day.일생산 >= day.일생산.quantile(2 / 3)),
        ('일생산 하위 1/3', day.일생산 <= day.일생산.quantile(1 / 3)),
        ('생산시간 18시간 이상', day.생산시간 >= 18),
        ('공휴일', day.공휴일 == 1),
    ]
    for lab, m in conds:
        if m.sum() == 0:
            continue
        rows.append({'조건': lab, '해당일': int(m.sum()), '피크일': int(day[m].피크일.sum()),
                     '피크일 비율(%)': round(100 * day[m].피크일.mean(), 1),
                     '기준대비(배)': round(day[m].피크일.mean() / base_rate, 2)
                     if base_rate else np.nan})
    single = pd.DataFrame(rows).sort_values('피크일 비율(%)', ascending=False)

    # 기온 x 생산 교차표 — 두 축을 같이 봐야 "포화" 가 보인다
    day['기온대'] = pd.cut(day.최고기온, [-99, 26, 30, 99], labels=['26도 이하', '26-30도', '30도 초과'])
    day['생산대'] = pd.qcut(day.일생산, 3, labels=['하위 1/3', '중위 1/3', '상위 1/3'])
    cross = day.pivot_table(index='기온대', columns='생산대', values='피크일',
                            aggfunc='mean', observed=True) * 100
    cnt = day.pivot_table(index='기온대', columns='생산대', values='피크일',
                          aggfunc='size', observed=True)
    return single, cross, cnt, day, base_rate


# ── 보강 2·3·4. 경보 ────────────────────────────────────────────────────────
def alarm_table(X, P, q, col='d7_hi', tcol=0):
    """임계 q 에서 일 단위 경보 성능. tcol=0 시간평균 기준, 1 은 15분 최대 기준."""
    thr = thresholds(X, q)
    t = P.fold.map({k: v[tcol] for k, v in thr.items()})
    y = P.y if tcol == 0 else P.y15
    d = pd.DataFrame({'날짜': P.날짜, 'pred': P[col] >= t, 'real': y >= t})
    g = d.groupby('날짜').agg(pred=('pred', 'any'), real=('real', 'any'))
    tp = int((g.pred & g.real).sum()); fp = int((g.pred & ~g.real).sum())
    fn = int((~g.pred & g.real).sum()); tn = int((~g.pred & ~g.real).sum())
    f = lambda a, b: 100 * a / b if b else np.nan
    n = tp + fp + fn + tn
    return {'임계분위': q, '임계전력': round(np.mean([v[tcol] for v in thr.values()]), 1),
            '정밀도(%)': round(f(tp, tp + fp), 1), '재현율(%)': round(f(tp, tp + fn), 1),
            # 경보율이 너무 높으면 정밀도·재현율이 좋아도 쓸모가 없다 (매일 경보 = 경보 없음)
            '경보율(%)': round(f(tp + fp, n), 1),
            '경보일': tp + fp, '실제위험일': tp + fn, '전체일': n,
            '일치율(%)': round(f(tp + tn, n), 1)}


def sweep(X, P):
    """임계값을 바꿔 가며 경보 성능을 낸다 (D10 확정 전 대비)."""
    return pd.DataFrame([alarm_table(X, P, q) for q in SWEEP])


def leadtime(X, P):
    """경보를 1주 전에 낼 때와 하루 전에 낼 때 (보강 3), 그리고 15분 최대 기준 (보강 4)."""
    rows = []
    for lab, col, tcol in [('D-7 · 분위 0.9', 'd7_hi', 0),
                           ('D-1 · 분위 0.9', 'd1_hi', 0),
                           ('D-7 · 15분 최대', 'm15_hi', 1),
                           ('D-7 · 점 예측', 'd7_pt', 0),
                           ('D-1 · 점 예측', 'd1_pt', 0)]:
        r = alarm_table(X, P, ALARM_Q, col, tcol)
        r['설정'] = lab
        rows.append(r)
    return pd.DataFrame(rows)[['설정', '임계전력', '정밀도(%)', '재현율(%)', '경보율(%)',
                               '경보일', '실제위험일', '일치율(%)']]


def daily_max(P):
    """일 최대 추정 성능 (가동일만). 분위 0.9 와 점 예측을 함께."""
    d = P[~P.is_off.astype(bool)]
    g = d.groupby('날짜').agg(실제=('y', 'max'), 분위=('d7_hi', 'max'), 점예측=('d7_pt', 'max'),
                             n=('hour', 'size'))
    g = g[g.n >= 20]
    out = []
    for c in ('분위', '점예측'):
        e = g.실제 - g[c]
        out.append({'추정': '분위 0.9' if c == '분위' else '점 예측의 최대',
                    'MAE': e.abs().mean(), '편향': e.mean(),
                    'RMSE': np.sqrt((e ** 2).mean())})
    return g, pd.DataFrame(out)


def peak_hour_dist(P):
    """일 최대가 난 시각의 분포 — 실제와 모델이 찍은 시각. 왜 시각을 못 맞히나 [8-3]."""
    d = P[~P.is_off.astype(bool)]
    g = d.groupby('날짜')
    act = g.apply(lambda s: s.loc[s.y.idxmax(), 'hour'], include_groups=False)
    prd = g.apply(lambda s: s.loc[s.d7_pt.idxmax(), 'hour'], include_groups=False)
    a = act.value_counts().reindex(range(24), fill_value=0)
    p = prd.value_counts().reindex(range(24), fill_value=0)
    return a, p, float((act == prd).mean() * 100)


def report(X, P):
    """그림에 들어갈 숫자를 전부 텍스트로도 낸다 (보고서에 그대로 옮길 수 있게)."""
    L = []
    pr = L.append
    pr('최대피크 예측 · 위험조건 · 경보 · 저감   (02_dashboard 의 숫자)')
    pr(f'평가 구간 CORE {CORE_FOLDS} · 고유일 {len(P)}행 · 시드 {list(SEEDS)} 평균')
    pr('')

    g, dm = daily_max(P)
    pr(f'[1] 일 최대 추정 (가동일 {len(g)}일)')
    pr(dm.round(2).to_string(index=False))
    pr('    점 예측의 최대는 구조적으로 낮다 → 일 최대·경보는 분위 0.9 로 낸다  [6-5]')
    pr('')

    single, cross, cnt, day, base = risk_conditions(X)
    pr(f'[2] 최대피크 위험조건 — 전체 {len(day)}일 중 피크일 {int(day.피크일.sum())}일'
       f' (기준 비율 {100 * base:.1f}%)')
    pr(single.to_string(index=False))
    pr('')
    pr('    기온 x 생산 교차표 (칸 = 피크일 비율 %, 괄호 = 해당 일수)')
    for i in cross.index:
        row = '      %-9s' % i
        for c in cross.columns:
            v, n = cross.loc[i, c], cnt.loc[i, c]
            row += f'  {c} {v:5.1f}% ({int(n):2d}일)' if pd.notna(v) else f'  {c}    -     '
        pr(row)
    pr('')
    pr('    ★ 이 표가 D16(동인=생산) vs D24(동인=기온) 논쟁의 답이다  [8-1] 결과 3')
    pr('      생산 하위 1/3 에서는 기온이 30도를 넘어도 피크일이 0% 다 (23일 전부).')
    pr('      생산이 중위 이상일 때만 기온이 비율을 올린다 (66.7 → 83.3 → 100).')
    pr('      즉 둘 중 하나가 동인인 게 아니라 순서가 있다 —')
    pr('        생산은 필요조건(없으면 피크가 안 난다), 기온은 그 위에서 위험을 올리는 증폭 요인.')
    pr('      상관 한 숫자로는 이 순서가 안 보여서 피어슨/스피어만에 따라 뒤집혔던 것이다')
    pr(f'    ※ 임계가 "시간 단위 상위 {100 * (1 - ALARM_Q):.0f}%" 라 일 단위로는 피크일이'
       f' {100 * base:.0f}% 로 흔하다.')
    pr('      위험조건의 방향은 이 임계와 무관하게 유지되지만, 경보를 선택적으로 만들려면')
    pr('      절대 임계(계약전력)가 필요하다 → D10 ②')
    pr('')

    cross_long = (cross.stack().rename('피크일비율(%)').reset_index()
                  .merge(cnt.stack().rename('일수').reset_index(),
                         on=['기온대', '생산대']))
    cross_long.round(2).to_csv(OUT.parent / 'peak_risk_cross.csv', index=False,
                               encoding='utf-8-sig')
    single.to_csv(OUT.parent / 'peak_risk_single.csv', index=False, encoding='utf-8-sig')

    lt = leadtime(X, P)
    lt.to_csv(OUT.parent / 'peak_alarm_leadtime.csv', index=False, encoding='utf-8-sig')
    pr(f'[3] 경보 성능 (임계 = 학습 구간 상위 {100 * (1 - ALARM_Q):.0f}%, 일 단위)')
    pr(lt.to_string(index=False))
    pr('')

    sw = sweep(X, P)
    # D10 ② 가 정해지면 이 표에서 값만 골라 ALARM_Q 를 바꾸면 된다
    sw.to_csv(OUT.parent / 'peak_alarm_sweep.csv', index=False, encoding='utf-8-sig')
    pr('[4] 경보 임계값 스윕 (D-7 분위 0.9 · 일 단위) — D10 확정 시 값만 바꿔 쓰면 된다')
    pr(sw.to_string(index=False))
    lo_rate = sw.loc[sw.임계분위 <= .90, '경보율(%)'].max()
    now_rate = sw.loc[sw.임계분위 == ALARM_Q, '경보율(%)'].iloc[0]
    pr('    ★ 낮은 임계는 숫자만 좋다. 분위 0.90 이하는 정밀도·재현율이 100% 지만')
    pr(f'      경보율이 {lo_rate:.0f}% 다 — 이틀에 한 번 이상 경보를 내면 현장은 무시한다.')
    pr(f'      지금 쓰는 {ALARM_Q} 도 경보율 {now_rate:.0f}% 로 높은 편이다.'
       ' 임계를 학습 분위가 아니라')
    pr('      계약전력 같은 절대값으로 바꿔야 경보가 선택적이 된다 → D10 ②  [0] B-1')
    pr('')

    D, dd = day_table(X)
    R, M = ramp(dd), month_peak(D)
    a, p, hit = peak_hour_dist(P)
    top = a.nlargest(4).index.tolist()
    pr('[5] 일 최대가 난 시각')
    pr(f'    실제 상위 4개 시각 {sorted(top)} 에 {100 * a[top].sum() / a.sum():.1f}%')
    pr(f'    모델이 찍은 시각 적중률 {hit:.1f}% — 경보를 "몇 시" 로 단정하면 안 된다  [8-3]')
    pr('')
    pr('[6] 피크전력 저감방안 (실측 근거. 모델 반사실이 아니다)')
    r8 = R.loc[R.시각 == 8, '상승폭'].iloc[0]
    r12 = R.loc[R.시각 == 12, '상승폭'].iloc[0]
    r13 = R.loc[R.시각 == 13, '상승폭'].iloc[0]
    k = M[M.피크시각.isin([7, 8, 9, 13])]
    pr(f'    동시 기동 시각  8시 {r8:+.1f} · 13시 {r13:+.1f}'
       f'  (12시 {r12:+.1f} 로 세웠다가 한꺼번에 재기동)')
    pr(f'    월 최대일 {len(M)}개월 중 {len(k)}개월이 기동 시각(7-9시·13시)에 났다')
    pr(f'    그 봉우리만 그날 2위 수준으로 눌러도 월 최대 -{M.간격.mean():.1f}'
       f' ~ -{M.간격.max():.1f} (실측 간격)')
    pr('    생산 총량도 시간별 배분도 안 바꾸므로 생산 손실 0 · 인건비 증가 0')
    pr('    ※ 축을 섞지 말 것 — 기각된 것은 "하루 안의 배분 조정" 하나다 (48.2%).')
    pr('      날짜 간 축은 모델이 읽지만(64.3%, p=0.044) 포화 때문에 효과가 작다  [8-7]')
    return '\n'.join(L), (g, dm, single, cross, cnt, day, base, lt, sw, a, p, hit, R, M)


# ── 그림 ────────────────────────────────────────────────────────────────────
def tile(fig, x, y, w, h, val, lab, sub, color):
    """한눈에 보는 수치 타일."""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='white',
                               edgecolor=GRID, linewidth=1))
    ax.text(.5, .66, val, ha='center', va='center', fontsize=20, fontweight='bold',
            color=color, transform=ax.transAxes)
    ax.text(.5, .34, lab, ha='center', va='center', fontsize=9.5, color=INK,
            transform=ax.transAxes)
    ax.text(.5, .13, sub, ha='center', va='center', fontsize=8, color=MUTED,
            transform=ax.transAxes)


def draw(parts):
    g, dm, single, cross, cnt, day, base, lt, sw, a, p, hit, R, M = parts
    fig = plt.figure(figsize=(17.5, 19.8))
    gs = GridSpec(4, 2, figure=fig, left=.078, right=.975, top=.835, bottom=.048,
                  hspace=.40, wspace=.21)

    fig.text(.078, .975, '최대피크 예측 · 위험조건 · 경보 · 저감', fontsize=19, fontweight='bold')
    fig.text(.078, .956, '과제 제목의 뒤쪽 절반.  CORE 2021년 7-9월 · 고유일 · 시드 3개 평균'
                         ' · 예측은 모두 구간 외(시간순 롤링 폴드)',
             fontsize=10.5, color=INK2)

    d7 = lt[lt.설정 == 'D-7 · 분위 0.9'].iloc[0]
    m15 = lt[lt.설정.str.contains('15분')].iloc[0]
    q09 = dm[dm.추정 == '분위 0.9'].iloc[0]
    tile(fig, .078, .872, .205, .066, f"{d7['재현율(%)']:.1f}%", '위험일 경보 재현율',
         f"1주 전 · 정밀도 {d7['정밀도(%)']:.1f}%", HI)
    tile(fig, .307, .872, .205, .066, f"{m15['일치율(%)']:.1f}%", '15분 최대 기준 일치율',
         '요금이 매겨지는 기준', HI)
    tile(fig, .537, .872, .205, .066, f'{q09.MAE:.2f}', '일 최대 MAE',
         f'편향 {q09.편향:+.2f} (거의 0)', PRED)
    tile(fig, .767, .872, .205, .066, '0%', '생산 하위 1/3 의 피크일 비율',
         '기온 30도 초과에도 0/23일', PEAK)

    # ① 위험조건 교차표 — 이 그림의 주인공
    ax = fig.add_subplot(gs[0, 0])
    Z = cross.values.astype(float)
    ax.imshow(Z, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(cross.columns)), [f'생산 {c}' for c in cross.columns])
    ax.set_yticks(range(len(cross.index)), list(cross.index))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                continue
            c = 'white' if Z[i, j] > 55 else INK
            ax.text(j, i - .09, f'{Z[i, j]:.0f}%', ha='center', va='center', fontsize=15,
                    fontweight='bold', color=c)
            ax.text(j, i + .22, f'{int(cnt.values[i, j])}일', ha='center', va='center',
                    fontsize=8.5, color='white' if Z[i, j] > 55 else MUTED)
    head(ax, '① 최대피크 위험조건 — 생산이 먼저, 기온이 그 위에서 올린다',
         '칸 = 그 조건에서 피크일이었던 비율.  생산 하위 1/3 은 기온과 무관하게 0%')
    ax.set_xlabel('일 생산 총량')
    ax.set_ylabel('그날 최고기온')
    for s in ax.spines.values():
        s.set_visible(False)

    # ② 단일 조건별 피크일 비율
    ax = fig.add_subplot(gs[0, 1])
    s2 = single.iloc[::-1].reset_index(drop=True)
    cols = [PEAK if v == 0 else (HI if v >= 90 else PRED) for v in s2['피크일 비율(%)']]
    ax.barh(range(len(s2)), s2['피크일 비율(%)'], color=cols, height=.66)
    ax.set_yticks(range(len(s2)), s2.조건)
    ax.axvline(100 * base, color=MUTED, linewidth=1.4, linestyle='--')
    ax.text(100 * base + 1.5, -.78, f'전체 평균 {100 * base:.0f}%', color=MUTED, fontsize=8.5)
    for i, (v, n) in enumerate(zip(s2['피크일 비율(%)'], s2.해당일)):
        ax.text(v + 1.5, i, f'{v:.0f}%  ({n}일)', va='center', fontsize=8.5, color=INK2)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 25, 50, 75, 100])
    head(ax, '② 조건 하나씩 — 어떤 날이 위험한가',
         '주말과 저생산일은 피크일이 한 번도 없다 → 경보 대상에서 먼저 제외할 수 있다')
    grid(ax, 'x')

    # ③ 일 최대 예측 vs 실제
    ax = fig.add_subplot(gs[1, 0])
    lim = [min(g.실제.min(), g.점예측.min()) - 8, max(g.실제.max(), g.분위.max()) + 8]
    ax.plot(lim, lim, color=BASEC, linewidth=1.4, zorder=1)
    ax.scatter(g.실제, g.점예측, s=34, color=PEAK, alpha=.75, zorder=2, label='점 예측의 최대')
    ax.scatter(g.실제, g.분위, s=34, color=HI, alpha=.85, zorder=3, label='분위 0.9 (채택)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('실제 일 최대')
    ax.set_ylabel('예측 일 최대')
    head(ax, '③ 일 최대 추정 — 점 예측은 대각선 아래로 쏠린다',
         f'가동일 {len(g)}일.  대각선 아래 = 낮게 봄.  '
                      f'MAE {q09.MAE:.2f} vs {dm.iloc[1].MAE:.2f}')
    ax.legend(loc='upper left', fontsize=9)
    grid(ax, 'both')

    # ④ 경보 임계값 스윕
    ax = fig.add_subplot(gs[1, 1])
    x = sw.임계전력
    ax.plot(x, sw['재현율(%)'], marker='o', ms=6, color=HI, linewidth=2, label='재현율')
    ax.plot(x, sw['정밀도(%)'], marker='s', ms=6, color=PRED, linewidth=2, label='정밀도')
    ax.plot(x, sw['경보율(%)'], marker='^', ms=6, color=PEAK, linewidth=2, label='경보 발령률')
    now = sw[sw.임계분위 == ALARM_Q].iloc[0]
    ax.axvline(now.임계전력, color=MUTED, linewidth=1.2, linestyle='--')
    ax.text(now.임계전력 - 1.5, 42, f'지금 쓰는 임계\n상위 5% = {now.임계전력:.0f}',
            fontsize=8.5, color=MUTED, va='top', ha='right')
    ax.set_xlabel('경보 임계 전력')
    ax.set_ylabel('%')
    ax.set_ylim(0, 112)
    head(ax, '④ 경보 임계값 — D10 이 정해지면 값만 바꾸면 된다',
         '임계를 낮추면 재현율은 100% 가 되지만 경보 발령률이 60% 를 넘어 쓸모가 없다')
    ax.legend(loc='lower left', fontsize=9)
    grid(ax, 'both')

    # ⑤ 예측 시계·기준별 경보 성능
    ax = fig.add_subplot(gs[2, 0])
    L = lt.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(L))
    ax.barh(yy + .19, L['재현율(%)'], height=.34, color=HI, label='재현율')
    ax.barh(yy - .19, L['정밀도(%)'], height=.34, color=PRED, label='정밀도')
    for i in yy:
        ax.text(L['재현율(%)'][i] + 1.2, i + .19, f"{L['재현율(%)'][i]:.1f}", va='center',
                fontsize=8.5, color=INK2)
        ax.text(L['정밀도(%)'][i] + 1.2, i - .19, f"{L['정밀도(%)'][i]:.1f}", va='center',
                fontsize=8.5, color=INK2)
    ax.set_yticks(yy, L.설정)
    ax.set_xlim(0, 122)
    head(ax, '⑤ 언제 · 무엇을 기준으로 경보를 낼 것인가',
         '1주 전과 하루 전이 거의 같다 → 미리 내도 성능을 잃지 않는다.'
                      ' 점 예측은 재현율이 낮다')
    ax.legend(loc='lower right', fontsize=9)
    grid(ax, 'x')

    # ⑥ 일 최대가 난 시각
    ax = fig.add_subplot(gs[2, 1])
    w = .4
    ax.bar(np.arange(24) - w / 2, a.values, width=w, color=PEAK, label='실제')
    ax.bar(np.arange(24) + w / 2, p.values, width=w, color=PRED, label='모델이 찍은 시각')
    ax.set_xticks(range(0, 24, 2), [f'{h}시' for h in range(0, 24, 2)])
    ax.set_xlabel('시각')
    ax.set_ylabel('일수')
    head(ax, '⑥ "몇 시가 최대인가" 는 맞힐 수 없다 — 한계로 명시한다',
         f'모델이 찍은 시각 적중률 {hit:.1f}%.  실제는 오전에 몰려 있는데'
                      ' 모델은 오후로 퍼뜨린다')
    ax.legend(loc='upper right', fontsize=9)
    grid(ax)

    # ⑦ 기동 램프 — 동시 기동이 언제 일어나나
    ax = fig.add_subplot(gs[3, 0])
    cols = [PEAK if v > 0 else PRED for v in R.상승폭]
    ax.bar(R.시각, R.상승폭, color=cols, width=.72)
    ax.axhline(0, color=BASEC, linewidth=1.2)
    for _, r in R.iterrows():
        if abs(r.상승폭) >= 20:
            ax.text(r.시각, r.상승폭 + (3 if r.상승폭 > 0 else -3), f'{r.상승폭:+.0f}',
                    ha='center', va='bottom' if r.상승폭 > 0 else 'top',
                    fontsize=9, fontweight='bold', color=INK)
    ax.set_xticks(range(0, 24, 2), [f'{h}시' for h in range(0, 24, 2)])
    ax.set_xlabel('시각')
    ax.set_ylabel('전 시각 대비 전력 상승폭')
    m = float(R.상승폭.abs().max())
    ax.set_ylim(-m * 1.28, m * 1.28)      # 값 라벨이 제목·축에 닿지 않게
    head(ax, '⑦ 동시 기동이 언제 일어나는가 — 저감 1순위의 근거',
         '12시에 세웠다가 13시에 한꺼번에 재기동한다.  아침 8시도 같은 모양')
    grid(ax)

    # ⑧ 월 최대일의 봉우리와 2위 간격 = 기동 분산으로 얻을 수 있는 양
    ax = fig.add_subplot(gs[3, 1])
    xx = np.arange(len(M))
    ax.bar(xx, M['2위'], color=PRED, width=.6, label='그날 2위 시각 전력')
    ax.bar(xx, M.간격, bottom=M['2위'], color=PEAK, width=.6, label='봉우리가 더 높은 폭')
    for i, r in M.reset_index(drop=True).iterrows():
        lab = f'-{r.간격:.0f}' if r.간격 >= .5 else '0'
        ax.text(i, r.월최대 + 2.5, lab, ha='center', fontsize=9.5,
                fontweight='bold', color=PEAK if r.간격 >= .5 else MUTED)
        ax.text(i, r['2위'] / 2, f'{int(r.피크시각)}시', ha='center', va='center',
                fontsize=9, color='white')
    ax.set_xticks(xx, [f'{int(m)}월' for m in M.월])
    ax.set_ylim(0, M.월최대.max() * 1.30)
    ax.set_ylabel('전력')
    head(ax, '⑧ 기동 분산으로 얻을 수 있는 양 — 월 최대일 기준',
         '주황 = 그 봉우리만 2위 수준으로 누르면 월 최대가 내려가는 폭 (실측).'
         '  막대 안 숫자는 피크 시각')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    grid(ax)

    fig.text(.078, .017,
             '저감 3축 중 "하루 안의 생산 배분 조정" 만 기각됐다 (모델이 그 축을 못 읽는다, 48.2%).'
             '  날짜 간 축은 읽지만(64.3%) 포화로 효과가 작고, 기동 분산은 실측만으로 근거가 선다.',
             fontsize=9, color=MUTED)
    fig.text(.078, .004,
             '재현: python src/peak_report.py  /  python src/peak_reduce.py'
             '  /  python src/peak_schedule.py --validate',
             fontsize=8.5, color=MUTED)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(f'→ {FIG}')


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    print('[최대피크 보고] 02_dashboard')
    P = predictions(X, refresh='--refresh' in sys.argv)
    P['ym'] = P.fold
    txt, parts = report(X, P)
    print()
    print(txt)
    TXT.write_text(txt, encoding='utf-8')
    print(f'\n→ {TXT}')
    draw(parts)
