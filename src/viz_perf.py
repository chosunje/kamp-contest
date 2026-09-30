"""모델 성능 체감용 그림. 숫자 표가 아니라 "얼마나 맞히는지"를 눈으로 보는 용도.

선행: python src/error_analysis.py  (outputs/error_predictions.csv 생성)
실행: python src/viz_perf.py        →  outputs/perf/01 to 06 png, captions.txt

평가 대상은 error_predictions.csv (최종 후보의 OOF 예측, 고유일 2,092행).
판단 기준은 CORE(7 to 9월, 1,708행)를 우선한다 (D22).
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from features import build
from preprocess import ROOT
from model import CORE_FOLDS, CORE

OUT = ROOT / 'outputs' / 'perf'
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
PRED, PEAK = '#2a78d6', '#eb6834'
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
CAPS = []


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def head(fig, title, sub):
    fig.text(.01, 1.0, title, fontsize=16, fontweight='bold', va='bottom')
    fig.text(.01, .965, sub, fontsize=10.5, color=INK2, va='bottom')


def save(fig, name, caption):
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    CAPS.append((name, caption))
    print(f'  → {name}')


def load_pred():
    P = pd.read_csv(ROOT / 'outputs' / 'error_predictions.csv', parse_dates=['dt'])
    X = build()[['dt', 'lag168']]
    P = P.merge(X, on='dt', how='left')          # 목표선(전주 같은 시각)을 같은 그림에 얹으려고
    P['is_core'] = P.fold.isin(CORE_FOLDS)
    P['날짜'] = P['날짜'].astype(int)
    return P


def m(s):
    """한 덩어리의 성적. e = 실제 - 예측 이므로 bias>0 이면 과소예측."""
    e = s.target - s.pred
    peak = s.is_peak.astype(bool)
    r2 = 1 - (e ** 2).sum() / ((s.target - s.target.mean()) ** 2).sum()
    return dict(n=len(s), MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()),
                peakMAE=e[peak].abs().mean() if peak.any() else np.nan,
                bias=e.mean(), R2=r2, within10=(e.abs() <= 10).mean())


# ── 01. 대표적인 날 4개의 24시간 곡선 ────────────────────────────────────
def fig_curves(P):
    C = P[P.is_core]
    d = C.groupby('날짜').agg(mae=('abs_error', 'mean'), mx=('target', 'max'),
                              off=('is_off', 'first'), dprod=('day_prod', 'first'))
    op = d[(d.dprod > 0) & (~d.off.astype(bool))]
    picks = [
        (int(op.mae.sub(op.mae.median()).abs().idxmin()), '대표 평일 (오차가 중간인 가동일)'),
        (int(d.mx.idxmax()), '최대 피크일'),
        (int(d[d.off.astype(bool)].mae.sub(d[d.off.astype(bool)].mae.median()).abs().idxmin()), '휴무일'),
        (int(d.mae.idxmax()), '가장 못 맞힌 날'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    for ax, (day, label) in zip(axes.ravel(), picks):
        s = C[C['날짜'] == day].sort_values('hour')
        dow = WEEK[s.dt.iloc[0].dayofweek]
        ax.fill_between(s.hour, s.target, color=INK, alpha=.07, zorder=1)
        ax.plot(s.hour, s.target, color=INK, lw=2.4, label='실제', zorder=4)
        ax.plot(s.hour, s.pred, color=PRED, lw=2.2, ls=(0, (4, 2)), label='예측', zorder=5)
        ax.plot(s.hour, s.lag168, color=MUTED, lw=1.2, ls=':', label='목표선 (전주 같은 시각)', zorder=3)
        ax.axhline(s.peak_threshold.iloc[0], color=PEAK, lw=.9, ls='--', alpha=.7, zorder=2)
        pk = s[s.is_peak.astype(bool)]
        if len(pk):
            ax.scatter(pk.hour, pk.target, s=26, color=PEAK, zorder=6)
        mae = s.abs_error.mean()
        base = (s.target - s.lag168).abs().mean()
        ax.set_title(f'{label}   {day} ({dow})', color=INK)
        ax.annotate(f'모델 MAE {mae:.1f}   목표선 MAE {base:.1f}\n일 생산 {int(s.day_prod.iloc[0]):,}',
                    xy=(.015, .965), xycoords='axes fraction', va='top', **{k: v for k, v in NOTE.items() if k != 'arrowprops'})
        ax.set_xlim(-.5, 23.5); ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel('시각'); ax.set_ylabel('전력 (시간 평균)')
        grid(ax)
    hs = [Line2D([], [], color=INK, lw=2.4), Line2D([], [], color=PRED, lw=2.2, ls=(0, (4, 2))),
          Line2D([], [], color=MUTED, lw=1.2, ls=':'),
          Line2D([], [], color=PEAK, lw=0, marker='o', ms=6)]
    fig.legend(hs, ['실제', '예측', '목표선 (전주 같은 시각)', '피크 구간'],
               loc='upper right', bbox_to_anchor=(1, 1.035), ncol=4, fontsize=10)
    head(fig, '하루를 얼마나 따라가는가', f'{CORE} 고유일 · 점선이 예측 · 주황 수평선은 피크 임계(학습구간 상위 5%)')
    fig.tight_layout(rect=[0, 0, 1, .955])
    save(fig, '01_day_curves.png',
         '대표 4일의 24시간 곡선. 실제(검정)와 예측(파란 점선)이 거의 겹치고, 전주 같은 시각을 그대로 쓰는 '
         '목표선(회색 점선)은 가동 패턴이 바뀌는 날 크게 벗어난다(휴무일 MAE 1.7 vs 121.3). '
         '다만 피크 구간(주황 점)에서는 예측이 실제보다 아래에 놓이는 경향이 보인다.')


# ── 02. 실제 vs 예측 산점도 ─────────────────────────────────────────────
def fig_scatter(P):
    C = P[P.is_core]
    s = m(C)
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    lim = [0, max(C.target.max(), C.pred.max()) * 1.05]
    ax.fill_between(lim, [lim[0] - 10, lim[1] - 10], [lim[0] + 10, lim[1] + 10],
                    color=PRED, alpha=.08, lw=0, zorder=1, label='오차 ±10 구간')
    ax.plot(lim, lim, color=INK, lw=1, zorder=2)
    nb = C[~C.is_peak.astype(bool)]
    pk = C[C.is_peak.astype(bool)]
    ax.scatter(nb.target, nb.pred, s=13, color=PRED, alpha=.35, lw=0, zorder=3, label='일반 구간')
    ax.scatter(pk.target, pk.pred, s=22, color=PEAK, alpha=.85, lw=0, zorder=4, label='피크 구간 (상위 5%)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('실제 전력'); ax.set_ylabel('예측 전력')
    ax.annotate(f'n = {s["n"]:,}\nMAE {s["MAE"]:.2f}   RMSE {s["RMSE"]:.2f}\n'
                f'R² {s["R2"]:.3f}   오차 ±10 이내 {s["within10"]*100:.1f}%\n'
                f'피크 구간 MAE {s["peakMAE"]:.2f} (평균 {s["bias"]:+.2f})',
                xy=(.03, .97), xycoords='axes fraction', va='top',
                **{k: v for k, v in NOTE.items() if k != 'arrowprops'})
    ax.legend(loc='lower right', fontsize=9.5)
    grid(ax, 'both')
    head(fig, '실제와 예측이 얼마나 붙어 있는가', f'{CORE} 고유일 {s["n"]:,}행 · 대각선에 가까울수록 정확')
    fig.tight_layout(rect=[0, 0, 1, .945])
    save(fig, '02_scatter.png',
         f'예측-실제 산점도. R² {s["R2"]:.3f}, 오차 ±10 이내가 {s["within10"]*100:.0f}%다. '
         '주황색 피크 구간이 대각선 위쪽(과소예측)으로 치우쳐 있는 것이 현재 남은 약점이다.')


# ── 03. 2주 연속 시계열 ─────────────────────────────────────────────────
def fig_timeline(P):
    C = P[P.is_core].sort_values('dt')
    # 하계휴가(7/31 to 8/8) 직후 2주. 휴무 복귀·주말 전환이 모두 들어 있어 난이도가 높은 구간
    s = C[(C.dt >= '2021-08-09') & (C.dt < '2021-08-23')]
    if len(s) < 100:
        s = C[C.dt >= C.dt.max() - pd.Timedelta(days=14)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7.4), height_ratios=[2.6, 1], sharex=True)
    for d, g in s.groupby(s.dt.dt.normalize()):
        if d.dayofweek >= 5:
            ax.axvspan(d, d + pd.Timedelta(days=1), color=BASE, alpha=.25, lw=0, zorder=0)
            ax2.axvspan(d, d + pd.Timedelta(days=1), color=BASE, alpha=.25, lw=0, zorder=0)
    ax.plot(s.dt, s.target, color=INK, lw=1.9, label='실제', zorder=3)
    ax.plot(s.dt, s.pred, color=PRED, lw=1.7, ls=(0, (4, 2)), label='예측', zorder=4)
    ax.set_ylabel('전력 (시간 평균)')
    ax.legend(loc='upper right', fontsize=9.5, ncol=2)
    grid(ax)
    ax.annotate('회색 배경은 주말', xy=(.005, .96), xycoords='axes fraction', va='top',
                **{k: v for k, v in NOTE.items() if k != 'arrowprops'})
    e = s.target - s.pred
    ax2.axhline(0, color=INK, lw=.9)
    ax2.bar(s.dt, e, width=.03, color=np.where(e >= 0, PEAK, PRED), lw=0)
    ax2.set_ylabel('오차\n(실제-예측)')
    grid(ax2)
    ax2.annotate('위(주황) = 과소예측 · 아래(파랑) = 과대예측', xy=(.005, .94), xycoords='axes fraction',
                 va='top', **{k: v for k, v in NOTE.items() if k != 'arrowprops'})
    sm = m(s)
    head(fig, '2주를 연속으로 예측하면', f'{s.dt.min():%Y-%m-%d} to {s.dt.max():%m-%d} · 하계휴가 직후 구간 '
                                        f'· MAE {sm["MAE"]:.2f}')
    fig.tight_layout(rect=[0, 0, 1, .95])
    save(fig, '03_timeline_2weeks.png',
         '하계휴가(7/31 to 8/8) 직후 2주간의 연속 예측. 휴무 복귀와 주말 전환이 반복되는 구간인데 '
         f'MAE {sm["MAE"]:.2f}로 따라간다. 아래 막대에서 보듯 오차는 가동 시작·종료 시각에 몰린다.')


# ── 04. 목표선 대비 ─────────────────────────────────────────────────────
def fig_vs_baseline(P):
    B = pd.read_csv(ROOT / 'outputs' / 'baseline_results.csv')
    B = B[B.group == '고유일']
    rows = []
    for scope, fold in [(CORE, CORE), ('5 to 9월', 'ALL')]:
        b = B[B.fold == fold]
        s = m(P[P.is_core] if fold == CORE else P)
        rows.append({'scope': scope,
                     'MAE': (b.MAE.min(), s['MAE']), 'RMSE': (b.RMSE.min(), s['RMSE']),
                     'peakMAE': (b.peakMAE.min(), s['peakMAE'])})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)
    for ax, r in zip(axes, rows):
        keys = ['MAE', 'RMSE', 'peakMAE']
        x = np.arange(3)
        base = [r[k][0] for k in keys]
        mod = [r[k][1] for k in keys]
        ax.bar(x - .19, base, .36, color=BASE, label='목표선 (최적 단순규칙)')
        ax.bar(x + .19, mod, .36, color=PRED, label='모델 (최종 후보)')
        top = max(base) * 1.3                      # 값·개선율 라벨이 제목을 침범하지 않도록 여유
        ax.set_ylim(0, top)
        for i, (b0, m0) in enumerate(zip(base, mod)):
            ax.text(i - .19, b0 + top * .015, f'{b0:.1f}', ha='center', fontsize=10, color=INK2)
            ax.text(i + .19, m0 + top * .015, f'{m0:.1f}', ha='center', fontsize=10.5, color=PRED, fontweight='bold')
            ax.text(i, max(b0, m0) + top * .075, f'{(m0/b0-1)*100:.0f}%', ha='center', fontsize=12,
                    color=INK, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(['MAE', 'RMSE', '피크구간 MAE'])
        ax.set_title(f'{r["scope"]} 기준' + ('   ★ 우선 기준' if r['scope'] == CORE else ''), color=INK)
        grid(ax)
    axes[0].set_ylabel('오차 (낮을수록 좋음)')
    axes[0].legend(loc='upper left', fontsize=9.5)
    head(fig, '단순 규칙이라면 얼마나 틀렸을까', '목표선 = 전체평균·전일동시각·전주동시각·요일x시간평균 4종 중 지표별 최저값')
    fig.tight_layout(rect=[0, 0, 1, .945])
    c = rows[0]
    save(fig, '04_vs_baseline.png',
         f'목표선 대비 개선폭. {CORE} 기준 MAE {c["MAE"][0]:.1f} → {c["MAE"][1]:.1f}, '
         f'피크구간 MAE {c["peakMAE"][0]:.1f} → {c["peakMAE"][1]:.1f}. 목표선과 모델은 학습 행·폴드·평가셋·'
         '평가 함수를 공유하므로 차이는 모델 능력에서만 나온다.')


# ── 05. 시간대별로 어디서 틀리는가 ──────────────────────────────────────
def fig_hour(P):
    C = P[P.is_core]
    g = C.groupby('hour').apply(lambda s: pd.Series(m(s)), include_groups=False)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.bar(g.index, g.MAE, .66, color=PRED, alpha=.75, label='MAE')
    ax.set_xticks(range(24)); ax.set_xlabel('시각'); ax.set_ylabel('MAE')
    grid(ax)
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(BASE)
    ax2.axhline(0, color=PEAK, lw=.8, ls=':', alpha=.6)   # 오른쪽 축의 0선 (막대 높이와 무관)
    ax2.plot(g.index, g.bias, color=PEAK, lw=2, marker='o', ms=4, label='평균 오차(실제-예측)')
    ax2.set_ylabel('평균 오차 (+면 과소예측)', color=PEAK)
    ax2.tick_params(colors=PEAK)
    lim = max(abs(g.bias).max() * 1.35, 3)
    ax2.set_ylim(-lim, lim)
    worst = g.MAE.idxmax()
    ax.annotate(f'{worst}시가 가장 어렵다 (MAE {g.MAE.max():.1f})',
                xy=(worst, g.MAE.max()), xytext=(worst + 1.6, g.MAE.max() * 1.02), **NOTE)
    ax.legend(loc='upper left', fontsize=9.5)
    ax2.legend(loc='upper right', fontsize=9.5)
    head(fig, '하루 중 어느 시각이 어려운가', f'{CORE} 고유일 · 막대는 오차 크기, 선은 오차 방향')
    fig.tight_layout(rect=[0, 0, 1, .945])
    save(fig, '05_by_hour.png',
         '시간대별 오차. 가동이 오르는 오전 8 to 11시에 오차가 집중되고 11시가 가장 크다. 심야 기저부하 '
         '구간(0 to 5시)은 거의 틀리지 않는다. 주황선이 주간(8 to 16시)에서 양수라는 것은 그 시간대를 '
         '일관되게 낮게 예측한다는 뜻으로, 피크 과소예측과 같은 현상이다.')


# ── 06. 일 최대와 피크 시각 ─────────────────────────────────────────────
def fig_daily_peak(P):
    C = P[P.is_core]
    def day(s):
        yh = int(s.loc[s.target.idxmax(), 'hour'])
        ph = int(s.loc[s.pred.idxmax(), 'hour'])
        return pd.Series({'dt': s.dt.iloc[0].normalize(), 'y_max': s.target.max(), 'p_max': s.pred.max(),
                          'y_h': yh, 'p_h': ph,
                          'in3': yh in set(s.nlargest(3, 'pred').hour),   # 예측 상위 3시간 안에 실제 피크가 있는가
                          'keep': s.loc[s.pred.idxmax(), 'target'] / s.target.max(),  # 예측 피크 시각의 실제 수준
                          'off': bool(s.is_off.iloc[0])})
    d = C.groupby('날짜').apply(day, include_groups=False)
    op = d[~d.off].copy()
    exact = (op.y_h == op.p_h).mean()
    near = ((op.y_h - op.p_h).abs() <= 1).mean()
    top3 = op.in3.mean()
    keep = op.keep.median()
    err = (op.y_max - op.p_max)
    # 가동이 없던 날(하계휴가 등)에서 선이 직선으로 이어지지 않도록 날짜 축을 채운다
    op = op.set_index('dt').reindex(pd.date_range(op.dt.min(), op.dt.max(), freq='D')).reset_index(names='dt')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), width_ratios=[2.1, 1])
    ax.plot(op.dt, op.y_max, color=INK, lw=1.7, marker='o', ms=3.5, label='실제 일 최대')
    ax.plot(op.dt, op.p_max, color=PRED, lw=1.6, ls=(0, (4, 2)), marker='o', ms=3.5, label='예측 일 최대')
    ax.set_ylabel('일 최대 전력 (시간 평균 기준)')
    ax.legend(loc='lower left', fontsize=9.5, ncol=2)
    grid(ax)
    ax.set_title(f'가동일 {int(op.y_max.notna().sum())}일의 일 최대   '
                 f'MAE {err.abs().mean():.2f}  (평균 {err.mean():+.2f})', color=INK)
    for lb in ax.get_xticklabels():
        lb.set_rotation(0)
    vals = [('예측 피크 시각이\n정확히 일치', exact), ('±1시간 이내', near),
            ('예측 상위 3시간 안에\n실제 피크가 있음', top3)]
    y = np.arange(3)[::-1]
    ax2.barh(y, [v for _, v in vals], .55, color=[PRED, PRED, BASE])
    ax2.set_yticks(y); ax2.set_yticklabels([k for k, _ in vals], fontsize=10)
    ax2.set_xlim(0, 1.12); ax2.set_xticks([0, .5, 1]); ax2.set_xticklabels(['0%', '50%', '100%'])
    ax2.set_xlabel('가동일 중 비율')
    for yv, (_, v) in zip(y, vals):
        ax2.text(v + .025, yv, f'{v*100:.0f}%', va='center', fontsize=12, fontweight='bold', color=INK)
    ax2.set_title('피크가 몇 시에 오는지 맞혔는가', color=INK)
    ax2.annotate(f'예측이 지목한 시각의 실제 전력은\n그날 실제 최대의 {keep*100:.1f}% (중앙값)',
                 xy=(.0, -.34), xycoords='axes fraction', va='top',
                 **{k: v for k, v in NOTE.items() if k != 'arrowprops'})
    grid(ax2, 'x')
    head(fig, '피크를 실제로 쓸 수 있는가', f'{CORE} 가동일 기준 · 피크 저감 스케줄링은 "언제 최대가 오는가"가 핵심')
    fig.tight_layout(rect=[0, 0, 1, .945])
    save(fig, '06_daily_peak.png',
         f'일 최대 전력의 예측값과 실제값 (MAE {err.abs().mean():.2f}, 평균 {err.mean():+.2f}로 일관된 과소예측), '
         f'그리고 하루 중 최대가 오는 시각을 맞힌 비율 (정확히 일치 {exact*100:.0f}%, ±1시간 이내 '
         f'{near*100:.0f}%, 예측 상위 3시간 안에 포함 {top3*100:.0f}%). 시각 적중률이 낮은 이유는 이 공장의 '
         '주간 곡선이 평탄해 비슷한 높이의 봉우리가 여러 개이기 때문이다. 실제로 예측이 지목한 시각의 전력은 '
         f'그날 최대의 {keep*100:.1f}% 수준이라, 어느 시간대를 덜어낼지 고르는 용도로는 쓸 수 있다.')


def main():
    P = load_pred()
    print(f'[성능 체감 그림] {CORE} n={P.is_core.sum():,} / 전체 n={len(P):,}')
    for scope, s in [(CORE, P[P.is_core]), ('5 to 9월', P)]:
        r = m(s)
        print(f'  {scope:14s} MAE {r["MAE"]:.2f}  RMSE {r["RMSE"]:.2f}  peakMAE {r["peakMAE"]:.2f}  '
              f'R² {r["R2"]:.3f}  ±10 이내 {r["within10"]*100:.1f}%')
    fig_curves(P); fig_scatter(P); fig_timeline(P); fig_vs_baseline(P); fig_hour(P); fig_daily_peak(P)
    txt = ['성능 체감 그림 설명 (src/viz_perf.py 생성)',
           f'평가: 최종 후보(LightGBM / h7_full / train_ok_strict / 복제일 가중치 0.3) 의 OOF 예측',
           f'기준: {CORE} 우선 (D22). 전체는 5 to 9월.', '']
    for n, c in CAPS:
        txt += [f'[{n}]', c, '']
    (OUT / 'captions.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
