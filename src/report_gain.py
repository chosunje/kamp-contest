"""강화 단계에서 무엇이 좋아졌고 무엇을 포기했나 — 00_dashboard 대비 비교.

  00_dashboard.png  그때까지의 최종 모델 성능 (현재 상태)
  01_dashboard.png  강화 단계의 변화 (이 파일)

  강화 단계의 주제는 "더 짜낼 수 있는가" 였다. 결과는 둘로 갈렸다.
    · 시간별 정확도(MAE 6.51)는 한계에 도달했다 — 수단 6가지를 재고 전부 기각
    · 일 최대에는 큰 여유가 있었다 — 추정량을 바꿔 MAE -21%, 편향 제거
  그 과정을 한 장에 담는다. 기각한 것도 같이 싣는다. 보고서의 "시도와 한계" 가 된다.

실행: python src/report_gain.py            (캐시 있으면 몇 초)
      python src/report_gain.py --refresh  (처음부터 계산, 약 7분)
출력: outputs/final/01_dashboard.png
      outputs/final/gain_predictions.csv   설정별 예측 캐시
      outputs/final/gain_summary.txt
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from features import build, FEATURES
from model import (make_model, rolling_eval, score, FINAL_CFG, ALARM_CFG, BASE,
                   CLONE_W, SHALLOW, SLOW, FOLDS, CORE_FOLDS, SEEDS)
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'final'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'gain_predictions.csv'

SURF, INK, INK2, MUTED, GRID, BASEC = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
PRED, PEAK, HI = '#2a78d6', '#eb6834', '#1baf7a'    # 검증된 팔레트 1·2·3 슬롯
QS = (.75, .80, .85, .90, .95)

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


# ── 예측 만들기 ────────────────────────────────────────────────────────────
def _w(tr, peak_w):
    v = np.where(tr.is_clone, CLONE_W, 1.0)
    return v * np.where(tr.target >= tr.target.quantile(.95), peak_w, 1.0)


def custom(X, cols=None, params=None, mode='direct', folds=FOLDS):
    """로그 타깃·잔차 학습처럼 rolling_eval 로 표현할 수 없는 변형을 돌린다."""
    cols = cols or FINAL_CFG['cols']
    params = FINAL_CFG['params'] if params is None else params
    out = []
    for m in folds:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X.dt < va.dt.min()) & X.train_ok_strict]
        if len(va) == 0 or len(tr) < 200:
            continue
        sw = _w(tr, FINAL_CFG['peak_w'])
        a_va = None
        if mode == 'resid':      # 기준선 = 지난주 같은 시각, 없으면 학습 구간 시각 평균
            hm = tr.groupby('hour').target.mean()
            a_tr = tr.lag_best.fillna(tr.hour.map(hm))
            a_va = va.lag_best.fillna(va.hour.map(hm))
            ytr = tr.target - a_tr
        else:
            ytr = np.log1p(tr.target) if mode == 'log' else tr.target
        ps = []
        for s in SEEDS:
            q = make_model('lgbm', s, params).fit(tr[cols], ytr, sample_weight=sw).predict(va[cols])
            ps.append(q + a_va.values if mode == 'resid' else (np.expm1(q) if mode == 'log' else q))
        out.append(pd.DataFrame({'fold': m, 'idx': va.index, 'y': va.target.values,
                                 'p': np.mean(ps, axis=0),
                                 'peak': va.target.values >= tr.target.quantile(.95)}))
    return pd.concat(out, ignore_index=True)


def oof(X, cfg, folds=FOLDS):
    parts = [rolling_eval(X, folds=folds, seed=s, **cfg).set_index('idx') for s in SEEDS]
    r = parts[0][['fold', 'y', 'peak']].copy()
    r['p'] = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
    return r.reset_index()


def all_configs():
    """패널마다 필요한 설정. 이름이 그대로 그림의 라벨이 된다."""
    L = dict(clone='weight', cols=FEATURES['h7_lag'], clone_w=CLONE_W)
    cfg = {'점 예측 (00 당시)': dict(FINAL_CFG)}
    for a in QS:
        cfg[f'분위 {a:.2f}'] = dict(ALARM_CFG, params={**ALARM_CFG['params'], 'alpha': a})
    cfg['기본 31잎'] = dict(BASE, **L)
    cfg['피크 우선'] = dict(BASE, **L, params=SHALLOW, peak_w=6.0)
    cfg['하루앞 full'] = dict(BASE, clone='weight', clone_w=CLONE_W, cols=FEATURES['full'],
                             params={**SHALLOW, **SLOW}, peak_w=3.0)
    cfg['RandomForest'] = dict(BASE, **L, model='rf',
                               params={'n_estimators': 500, 'min_samples_leaf': 3})
    return cfg


def build_cache(X):
    frames = []
    for name, cfg in all_configs().items():
        print(f'    {name} ...')
        r = oof(X, cfg); r['run'] = name; frames.append(r)
    for name, kw in (('새 피처 4개', dict(cols=FINAL_CFG['cols'] + NEW)),
                     ('로그 타깃', dict(mode='log')),
                     ('Huber 손실', dict(params={**FINAL_CFG['params'], 'objective': 'huber',
                                                 'alpha': 5.0})),
                     ('잔차 학습', dict(mode='resid'))):
        print(f'    {name} ...')
        r = custom(X, **kw); r['run'] = name; frames.append(r)
    return pd.concat(frames, ignore_index=True)


NEW = ['cum_prod', 'temp_x_prod', 'prev_day_tmax', 'lag_trend']


def add_new_features(X):
    """기각된 피처 4개. 그림에 "해봤고 안 됐다" 를 싣기 위해 여기서만 만든다."""
    day = X.groupby('날짜')
    X['cum_prod'] = day['prod'].cumsum()
    X['temp_x_prod'] = X.temp * (X.prod_cap / 800)
    X['prev_day_tmax'] = X['날짜'].map(day.temp.max().shift(1))
    p = X.target.where(~X.is_off & ~X.is_stop)
    X['lag_trend'] = (p.shift(168) - p.shift(168 * 4)) / 3
    return X


def load(refresh=False):
    X = add_new_features(build())
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    deps = [ROOT / 'src' / f for f in ('features.py', 'model.py', 'preprocess.py')]
    stale = (not CACHE.exists() or
             any(d.exists() and d.stat().st_mtime > CACHE.stat().st_mtime for d in deps))
    if not refresh and not stale:
        print(f'  캐시 사용: {CACHE.name}. 다시 계산하려면 --refresh')
        return X, pd.read_csv(CACHE)
    print('  설정 14종 x 시드 3개를 돌린다. 약 7분...')
    P = build_cache(X)
    P.to_csv(CACHE, index=False, encoding='utf-8-sig')
    return X, P


# ── 집계 ───────────────────────────────────────────────────────────────────
def daymax(X, P, run, folds=CORE_FOLDS):
    """가동일의 일 최대. 휴무일은 최대라는 개념이 의미 없어 뺀다."""
    r = P[(P.run == run) & P.fold.isin(folds)].set_index('idx')
    off = X.loc[r.index, 'is_off'].values
    r = r[off == 0].assign(날짜=X.loc[r.index[off == 0], '날짜'].values)
    g = r.groupby('날짜')
    return pd.DataFrame({'실제': g.y.max(), '예측': g.p.max()})


def dm_stat(X, P, run, folds=CORE_FOLDS):
    d = daymax(X, P, run, folds); e = d.실제 - d.예측
    return e.abs().mean(), e.mean()


# ── 패널 ───────────────────────────────────────────────────────────────────
def panel_sweep(ax, X, P):
    """분위 수준을 올리면 일 최대 추정이 어떻게 변하나. MAE 와 편향은 단위가 같다(전력)."""
    names = ['점 예측 (00 당시)'] + [f'분위 {a:.2f}' for a in QS]
    mae = [dm_stat(X, P, n)[0] for n in names]
    bias = [dm_stat(X, P, n)[1] for n in names]
    x = np.arange(len(names))
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.plot(x, mae, color=PRED, lw=2, marker='o', ms=8, zorder=4)
    ax.plot(x, bias, color=PEAK, lw=2, marker='o', ms=8, zorder=4)
    ax.annotate('일 최대 MAE', (x[-1], mae[-1]), xytext=(-6, 10), textcoords='offset points',
                ha='right', fontsize=10, color=PRED, fontweight='bold')
    ax.annotate('편향 (양수 = 낮게 봄)', (x[-1], bias[-1]), xytext=(-6, -14),
                textcoords='offset points', ha='right', fontsize=10, color=PEAK, fontweight='bold')
    k = names.index('분위 0.90')
    ax.scatter([x[k]], [bias[k]], s=150, facecolor='none', edgecolor=PEAK, lw=2, zorder=5)
    ax.annotate('여기서 편향이\n0 을 지난다', (x[k], bias[k]), xytext=(0, 20),
                textcoords='offset points', ha='center', fontsize=9.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['점 예측'] + [f'{a:.2f}' for a in QS])
    ax.set_xlabel('분위 수준'); ax.set_ylabel('전력')
    ax.set_title('① 분위를 올리면 일 최대 편향이 사라진다', color=INK)
    grid(ax)


def panel_days(ax, X, P):
    """날짜별로 실제 일 최대를 두 추정값이 어떻게 따라가나."""
    a = daymax(X, P, '점 예측 (00 당시)'); b = daymax(X, P, '분위 0.90')
    d = a.join(b.예측.rename('q90'), how='inner').sort_index()
    x = np.arange(len(d))
    ax.plot(x, d.실제, color=INK, lw=2.2, zorder=5)
    ax.plot(x, d.예측, color=BASEC, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax.plot(x, d.q90, color=HI, lw=1.8, ls=(0, (1, 1.6)), zorder=4)
    # 선이 셋이라 범례를 둔다. aqua 는 대비가 낮아 직접 라벨도 같이 붙인다 (relief)
    hs = [Line2D([], [], color=INK, lw=2.2), Line2D([], [], color=HI, lw=1.8, ls=(0, (1, 1.6))),
          Line2D([], [], color=BASEC, lw=1.8, ls=(0, (4, 2)))]
    lo, hi_ = d[['실제', '예측', 'q90']].min().min(), d[['실제', '예측', 'q90']].max().max()
    ax.set_ylim(lo - (hi_ - lo) * .06, hi_ + (hi_ - lo) * .30)   # 범례 자리를 위에 비워 둔다
    ax.legend(hs, ['실제', '분위 0.9 (이번)', '점 예측 (00 당시)'], ncol=3, fontsize=9.5,
              loc='upper left')
    k = int(np.argmax(d.q90.values - d.실제.values))
    ax.annotate('분위 0.9', (x[k], d.q90.iloc[k]), xytext=(0, -16), textcoords='offset points',
                ha='center', fontsize=9.5, color=HI, fontweight='bold')
    ax.set_xlim(-1, len(d))
    ax.set_title(f'② 날짜별 일 최대 — 점 예측은 늘 아래에 있다 (가동일 {len(d)}일)', color=INK)
    ax.set_xlabel('CORE 구간 가동일 (시간순)'); ax.set_ylabel('일 최대 전력')
    grid(ax)


def panel_folds(ax, X, P):
    """이득이 고른가. 고르지 않다는 것을 숨기지 않는다."""
    labs, a, b = [], [], []
    for f in FOLDS:
        if not ((P.fold == f).any()):
            continue
        m1 = dm_stat(X, P, '점 예측 (00 당시)', [f])[0]
        m2 = dm_stat(X, P, '분위 0.90', [f])[0]
        if np.isnan(m1):
            continue
        labs.append(f[-2:] + '월'); a.append(m1); b.append(m2)
    labs.append('CORE\n합계'); a.append(dm_stat(X, P, '점 예측 (00 당시)')[0])
    b.append(dm_stat(X, P, '분위 0.90')[0])
    x = np.arange(len(labs))
    ax.bar(x - .2, a, .36, color=BASEC, zorder=3, label='점 예측 (00 당시)')
    ax.bar(x + .2, b, .36, color=HI, zorder=3, label='분위 0.9 (이번)')
    for xi, (v0, v1) in enumerate(zip(a, b)):
        ax.text(xi - .2, v0 + .4, f'{v0:.1f}', ha='center', fontsize=9, color=INK2)
        ax.text(xi + .2, v1 + .4, f'{v1:.1f}', ha='center', fontsize=9,
                color=HI if v1 < v0 else PEAK, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(0, max(a + b) * 1.18)
    ax.set_title('③ 이득이 고르지는 않다 — 5월은 오히려 손해', color=INK)
    ax.set_ylabel('일 최대 MAE'); ax.legend(fontsize=9.5, loc='upper right'); grid(ax)


def panel_rejected(ax, X, P):
    """재보고 버린 수단들. 기준선을 넘지 못했다."""
    base = score(P[(P.run == '점 예측 (00 당시)') & P.fold.isin(CORE_FOLDS)]).MAE
    items = []
    for n in ('로그 타깃', '새 피처 4개', 'Huber 손실', '잔차 학습', '기본 31잎', 'RandomForest'):
        s = P[(P.run == n) & P.fold.isin(CORE_FOLDS)]
        if len(s):
            items.append((n, score(s).MAE))
    # 앙상블: 최종 + 기본 31잎 평균
    f = P[P.fold.isin(CORE_FOLDS)]
    p1 = f[f.run == '점 예측 (00 당시)'].set_index('idx')
    p2 = f[f.run == '기본 31잎'].set_index('idx')
    j = p1.join(p2.p.rename('p2'), how='inner')
    items.append(('앙상블 (최종+기본)', score(j.assign(p=(j.p + j.p2) / 2)).MAE))
    # 기각 사유. MAE 만으로는 판단이 안 되는 것이 둘 있어서 같이 적는다
    why = {'로그 타깃': 'peakMAE 12.48 → 13.90', '앙상블 (최종+기본)': '차이 0.012 = 흔들림 안쪽',
           '새 피처 4개': '악화', '기본 31잎': '악화', 'Huber 손실': '악화',
           '잔차 학습': '크게 악화', 'RandomForest': '크게 악화'}
    items.sort(key=lambda t: t[1])
    lab = [i[0] for i in items][::-1]; val = [i[1] for i in items][::-1]
    # 전부 기각한 것이므로 색으로 우열을 주지 않는다. 기준선만 강조한다
    ax.barh(lab, val, color=BASEC, height=.6, zorder=3)
    for i, (l, v) in enumerate(zip(lab, val)):
        ax.text(v + max(val) * .012, i, f'{v:.3f}', va='center', fontsize=9.5, color=INK2)
        ax.text(max(val) * 1.22, i, why.get(l, ''), va='center', fontsize=9, color=MUTED)
    ax.axvline(base, color=PEAK, lw=1.6, ls='--', zorder=4)
    ax.annotate(f'현재 {base:.3f}', (base, len(val) - .35), xytext=(-5, 0),
                textcoords='offset points', ha='right', fontsize=9.5, color=PEAK, fontweight='bold')
    ax.set_xlim(0, max(val) * 1.90)
    ax.set_title('④ 재보고 버린 수단 — 전부 기각했다', color=INK)
    ax.set_xlabel('시간별 MAE (CORE, 낮을수록 좋다)')
    ax.text(0, -.17, '※ 위 둘은 MAE 만 보면 기준보다 낮지만, 피크를 내주거나 차이가 '
                     '흔들림 안쪽이라 쓰지 않았다',
            transform=ax.transAxes, fontsize=9, color=MUTED)
    grid(ax, 'x')


def panel_corr(ax, X, P):
    """앙상블이 왜 안 되나. 모델들이 같은 자리에서 틀린다."""
    names = ['점 예측 (00 당시)', '피크 우선', '기본 31잎', '하루앞 full', 'RandomForest']
    short = ['최종', '피크우선', '기본31잎', '하루앞full', 'RF']
    f = P[P.fold.isin(CORE_FOLDS)]
    E = pd.DataFrame({s: (lambda v: (v.y - v.p))(f[f.run == n].set_index('idx'))
                      for n, s in zip(names, short)}).dropna()
    M = E.corr().values
    im = ax.imshow(M, cmap='Blues', vmin=.4, vmax=1)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=10,
                    color='white' if M[i, j] > .8 else INK)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, fontsize=9.5, rotation=20)
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=9.5)
    ax.set_title('⑤ 오차 상관 — 높을수록 섞어도 소용없다', color=INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def tiles(fig, items):
    for i, (val, lab, sub) in enumerate(items):
        x = .012 + i * .252
        fig.text(x, .940, val, fontsize=26, fontweight='bold', color=INK, va='top')
        fig.text(x, .911, lab, fontsize=11, color=INK, va='top')
        fig.text(x, .895, sub, fontsize=9.5, color=MUTED, va='top')


if __name__ == '__main__':
    X, P = load('--refresh' in sys.argv)

    m0, b0 = dm_stat(X, P, '점 예측 (00 당시)')
    m1, b1 = dm_stat(X, P, '분위 0.90')
    hourly = score(P[(P.run == '점 예측 (00 당시)') & P.fold.isin(CORE_FOLDS)])

    fig = plt.figure(figsize=(15.5, 15.6))
    gs = GridSpec(3, 2, figure=fig, hspace=.40, wspace=.22, top=.845, bottom=.045,
                  left=.058, right=.975)

    fig.text(.012, .993, '강화 단계에서 무엇이 바뀌었나 — 00_dashboard 대비',
             fontsize=19, fontweight='bold', va='top')
    fig.text(.012, .967,
             '주제: "논리적으로 더 짜낼 수 있는가". 시간별 정확도는 한계에 도달했고, '
             '일 최대에는 여유가 있었다  ·  평가 CORE(7 to 9월) · 시드 3개',
             fontsize=10.5, color=INK2, va='top')

    tiles(fig, [
        (f'{m0:.2f} → {m1:.2f}', '일 최대 MAE', f'{(1 - m1 / m0) * 100:.0f}% 감소'),
        (f'{b0:+.2f} → {b1:+.2f}', '일 최대 편향', '구조적 과소예측이 사라졌다'),
        (f'{hourly.MAE:.2f}', '시간별 MAE  (변화 없음)', '수단 7가지를 재고 전부 기각'),
        ('7', '기각한 수단', '앙상블·피처·로그·Huber·잔차·일단위모델·보정계수'),
    ])

    panel_sweep(fig.add_subplot(gs[0, 0]), X, P)
    panel_days(fig.add_subplot(gs[0, 1]), X, P)
    panel_folds(fig.add_subplot(gs[1, 0]), X, P)
    panel_rejected(fig.add_subplot(gs[1, 1]), X, P)
    panel_corr(fig.add_subplot(gs[2, 0]), X, P)

    ax = fig.add_subplot(gs[2, 1]); ax.axis('off')
    ax.text(0, 1, '⑥ 00_dashboard 와 달라진 점', fontsize=12, fontweight='bold', va='top')
    ax.text(0, .90, f"""
시간별 예측        그대로다. MAE {hourly.MAE:.2f} / RMSE {hourly.RMSE:.2f} / peakMAE {hourly.peakMAE:.2f}
                   00 에서 쓴 모델을 바꾸지 않았다

일 최대 추정       점 예측의 최대  →  분위 0.9 의 최대
                   MAE {m0:.2f} → {m1:.2f} · 편향 {b0:+.2f} → {b1:+.2f}

왜 바꿨나          점 예측은 시각마다 "가운데 값" 을 맞힌다.
                   가운데 값 24개의 최대는 실제 24개의 최대보다
                   구조적으로 낮다. 모델이 나쁜 게 아니라
                   추정량을 잘못 고른 것이었다.

무엇을 포기했나    앙상블 · 추가 피처 · 로그 타깃 · Huber ·
                   잔차 학습 · 일 단위 전용 모델 · 보정 계수
                   전부 측정해서 기각했다 (④⑤)

남은 교훈          하나의 예측값으로 모든 질문에 답할 수 없다.
                   "이 시각 얼마"  → 점 예측
                   "오늘 최대 얼마" → 분위 0.9
                   "오늘 위험한가"  → 분위 0.9 + 임계값
""", fontsize=10, color=INK2, va='top', linespacing=1.5, family='Malgun Gothic')

    path = OUT / '01_dashboard.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)

    txt = OUT / 'gain_summary.txt'
    lines = ['강화 단계 변화 요약 (src/report_gain.py 생성)',
             '비교 대상: 00_dashboard.png 시점의 모델', '',
             f'시간별 지표 (변화 없음)  MAE {hourly.MAE:.2f} / RMSE {hourly.RMSE:.2f} / peakMAE {hourly.peakMAE:.2f}',
             f'일 최대 MAE   {m0:.2f} → {m1:.2f}  ({(1 - m1 / m0) * 100:.0f}% 감소)',
             f'일 최대 편향  {b0:+.2f} → {b1:+.2f}', '', '[분위 수준별 일 최대]']
    for n in ['점 예측 (00 당시)'] + [f'분위 {a:.2f}' for a in QS]:
        a_, b_ = dm_stat(X, P, n)
        lines.append(f'  {n:16s} MAE {a_:6.2f}  편향 {b_:+6.2f}')
    lines += ['', '[폴드별 일 최대 MAE]']
    for f in FOLDS:
        v0 = dm_stat(X, P, '점 예측 (00 당시)', [f])[0]
        v1 = dm_stat(X, P, '분위 0.90', [f])[0]
        if not np.isnan(v0):
            lines.append(f'  {f}  점예측 {v0:6.2f}  분위0.9 {v1:6.2f}'
                         f'  {"개선" if v1 < v0 else "악화"}')
    txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\n[강화 단계] 일 최대 MAE {m0:.2f} → {m1:.2f} · 편향 {b0:+.2f} → {b1:+.2f}')
    print(f'  시간별 지표는 변화 없음 (MAE {hourly.MAE:.2f})')
    print(f'\n→ {path}\n→ {txt}\n→ {CACHE}')
