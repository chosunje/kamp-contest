"""최종 모델 한 장 요약 — "이 모델이 어느 정도인가" 를 눈으로 보는 용도.

  viz_perf.py 와 역할이 다르다.
    viz_perf.py     팀원 작성. error_predictions.csv (h7_full, 피크 가중치 없음) 기준
    report_final.py 내 최종 설정(FINAL_CFG)와 기존 시간 평균 경보 비교의 한 장 요약
  현재 D10 기본 경보는 15분 최대 P90/Q95이며 peak_alarm_max15.py에서 별도로 검증한다.

  왜 따로 만들었나: viz_perf.py 가 쓰는 예측은 피크 가중치를 넣기 전 설정이라
  "일 최대를 평균 +9.62 낮게 본다" 로 되어 있다. 최종 설정에서는 이 부호가 뒤집혔다.
  팀원 파일을 건드리지 않고 최종 설정의 그림을 따로 낸다.

실행: python src/report_final.py            (예측이 캐시에 있으면 재사용, 몇 초)
      python src/report_final.py --refresh  (예측부터 다시 계산, 약 3분)
출력: outputs/final/00_dashboard.png   한 장 요약
      outputs/final/predictions.csv    행 단위 예측 (캐시)
      outputs/final/summary.txt        숫자 요약
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from features import build
from model import rolling_eval, score, FINAL_CFG, ALARM_CFG, CORE_FOLDS, SEEDS
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'final'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'predictions.csv'

# 색은 dataviz 기준 팔레트의 1·2·3번 슬롯 (전수쌍 검증 통과).
# aqua 는 밝은 배경에서 대비가 낮아 반드시 직접 라벨을 붙인다.
SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
PRED, PEAK, HI = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10.5,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': BASE, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 9, 'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


# ── 예측 만들기 ────────────────────────────────────────────────────────────
def oof(X, cfg, seeds=SEEDS):
    """시드별 롤링 폴드 예측을 평균낸다 (시드 하나에 운을 걸지 않는다)."""
    parts = [rolling_eval(X, seed=s, **cfg).set_index('idx') for s in seeds]
    r = parts[0][['fold', 'y', 'peak']].copy()
    r['p'] = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
    return r


def build_predictions(X):
    """최종 설정·경보 상한·비교 기준(피크 가중치 전) 셋을 한 표로 모은다."""
    base_cfg = dict(FINAL_CFG, peak_w=1.0)      # 피크 가중치만 끈 것 = 개선 전
    out = oof(X, FINAL_CFG).rename(columns={'p': 'pred'})
    out['pred_hi'] = oof(X, ALARM_CFG).p        # 경보 상한 (분위 0.9)
    out['pred_base'] = oof(X, base_cfg).p       # 피크 보정 전
    keep = ['dt', '날짜', 'hour', 'dow', 'is_off', 'is_holiday', 'day_prod', 'lag168']
    return X.loc[out.index, keep].join(out)


def stale():
    """피처나 모델 설정이 캐시보다 새 것이면 캐시를 믿으면 안 된다.
    이걸 안 보면 설정을 바꾸고도 옛 그림을 보게 된다."""
    if not CACHE.exists():
        return True
    t = CACHE.stat().st_mtime
    deps = [ROOT / 'src' / 'features.py', ROOT / 'src' / 'model.py',
            ROOT / 'src' / 'preprocess.py', ROOT / 'outputs' / 'features.csv']
    return any(d.exists() and d.stat().st_mtime > t for d in deps)


def load_predictions(refresh=False):
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    if not refresh and not stale():
        P = pd.read_csv(CACHE, parse_dates=['dt'])
        print(f'  캐시 사용: {CACHE.name} ({len(P)}행). 다시 계산하려면 --refresh')
        return X, P
    why = '캐시 없음/설정 변경 감지' if not refresh else '--refresh'
    print(f'  예측 계산 중 ({why}). 설정 3종 x 시드 3개, 약 1분 30초...')
    P = build_predictions(X)
    P.to_csv(CACHE, index=False, encoding='utf-8-sig')
    return X, P


# ── 패널 ───────────────────────────────────────────────────────────────────
def panel_day(ax, s, title):
    """하루 24시간 곡선. 실제가 예측·경보상한과 어떻게 놓이는지 본다."""
    ax.fill_between(s.hour, s.y, color=INK, alpha=.06, zorder=1)
    ax.plot(s.hour, s.lag168, color=MUTED, lw=1.1, ls=':', zorder=2)
    ax.plot(s.hour, s.pred_hi, color=HI, lw=1.6, ls=(0, (1, 1.6)), zorder=4)
    ax.plot(s.hour, s.y, color=INK, lw=2.3, zorder=5)
    ax.plot(s.hour, s.pred, color=PRED, lw=2.0, ls=(0, (4, 2)), zorder=6)
    pk = s[s.peak]
    if len(pk):
        ax.scatter(pk.hour, pk.y, s=24, color=PEAK, zorder=7)
    # aqua 는 밝은 배경에서 대비가 낮아 직접 라벨을 붙인다 (relief 규칙)
    ax.annotate('시간 평균 P90', (1, s.pred_hi.iloc[1]), textcoords='offset points',
                xytext=(2, 9), fontsize=9, color=HI, fontweight='bold')
    e = (s.y - s.pred).abs().mean()
    ax.set_title(f'{title}   ·   MAE {e:.1f}', color=INK)
    ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel('시각'); grid(ax)


def panel_scatter(ax, C):
    """실제와 예측이 얼마나 붙어 있나. 피크 구간을 따로 칠해서 약점을 같이 본다."""
    lim = [0, max(C.y.max(), C.pred.max()) * 1.04]
    ax.fill_between(lim, [v - 10 for v in lim], [v + 10 for v in lim],
                    color=PRED, alpha=.08, lw=0, zorder=1)
    ax.plot(lim, lim, color=INK, lw=1, zorder=2)
    nb, pk = C[~C.peak], C[C.peak]
    ax.scatter(nb.y, nb.pred, s=11, color=PRED, alpha=.3, lw=0, zorder=3, label='일반 구간')
    ax.scatter(pk.y, pk.pred, s=20, color=PEAK, alpha=.85, lw=0, zorder=4, label='피크 (상위 5%)')
    r2 = 1 - ((C.y - C.pred) ** 2).sum() / ((C.y - C.y.mean()) ** 2).sum()
    w10 = ((C.y - C.pred).abs() <= 10).mean() * 100
    ax.set_title('실제 vs 예측', color=INK)
    ax.text(.03, .96, f'R² {r2:.3f}\n오차 ±10 이내 {w10:.0f}%', transform=ax.transAxes,
            va='top', fontsize=10, color=INK2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('실제 전력'); ax.set_ylabel('예측 전력')
    ax.legend(loc='lower right', fontsize=9.5); grid(ax, 'both')


def panel_ladder(ax, rows):
    """베이스라인에서 최종까지 얼마나 내려왔나. 막대 하나짜리 비교라 색은 강조용으로만 쓴다."""
    lab = [r[0] for r in rows][::-1]
    val = [r[1] for r in rows][::-1]
    cols = [PRED if '최종' in l else BASE for l in lab]
    b = ax.barh(lab, val, color=cols, height=.62)
    for rect, v in zip(b, val):
        ax.text(rect.get_width() + max(val) * .015, rect.get_y() + rect.get_height() / 2,
                f'{v:.2f}', va='center', fontsize=10, color=INK2)
    ax.set_title('MAE — 목표선에서 최종까지', color=INK)
    ax.set_xlabel('MAE (낮을수록 좋다)')
    ax.set_xlim(0, max(val) * 1.16); grid(ax, 'x')


def panel_peakband(ax, C):
    """실제 전력 구간별 평균오차. 피크 보정이 무엇을 바꿨는지가 여기서 보인다."""
    b = pd.cut(C.y, [0, 100, 150, 180, 999], labels=['~100', '100-150', '150-180', '180+'])
    g = C.groupby(b, observed=True)
    before = g.apply(lambda s: (s.y - s.pred_base).mean(), include_groups=False)
    after = g.apply(lambda s: (s.y - s.pred).mean(), include_groups=False)
    x = np.arange(len(before))
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.bar(x - .2, before, .36, color=BASE, zorder=3, label='피크 보정 전')
    ax.bar(x + .2, after, .36, color=PEAK, zorder=3, label='피크 보정 후 (최종)')
    span = max(before.max(), after.max()) - min(before.min(), after.min(), 0)
    for xi, (v0, v1) in enumerate(zip(before, after)):
        for dx, v in ((-.2, v0), (.2, v1)):
            ax.text(xi + dx, v + span * .035 * (1 if v >= 0 else -1), f'{v:+.1f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9, color=INK2)
    ax.set_ylim(min(before.min(), after.min(), 0) - span * .2, max(before.max(), after.max()) + span * .14)
    ax.set_xticks(x); ax.set_xticklabels(before.index)
    ax.set_title('실제 전력 구간별 평균오차', color=INK)
    ax.set_xlabel('실제 전력 구간'); ax.set_ylabel('평균오차 (양수 = 낮게 봄)')
    ax.legend(fontsize=9.5, loc='upper left'); grid(ax)


def panel_alarm(ax, C, X):
    """경보를 점 예측으로 내면 왜 안 되는가. 재현율 = 실제 피크를 미리 잡은 비율."""
    thr = {}
    for m in CORE_FOLDS:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
        thr[m] = tr.target.quantile(.95)
    t = C.fold.map(thr)
    real = C.y >= t
    out = []
    for lab, col in (('점 예측', 'pred'), ('시간 평균 P90', 'pred_hi')):
        pred = C[col] >= t
        tp = (pred & real).sum()
        out.append((lab, tp / real.sum() * 100, tp / max(pred.sum(), 1) * 100))
    x = np.arange(len(out))
    ax.bar(x - .2, [o[1] for o in out], .36, color=HI, zorder=3, label='재현율 (놓치지 않음)')
    ax.bar(x + .2, [o[2] for o in out], .36, color=BASE, zorder=3, label='정밀도 (헛경보 아님)')
    for xi, o in enumerate(out):
        ax.text(xi - .2, o[1] + 1.5, f'{o[1]:.0f}%', ha='center', fontsize=10,
                color=HI, fontweight='bold')
        ax.text(xi + .2, o[2] + 1.5, f'{o[2]:.0f}%', ha='center', fontsize=10, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([o[0] for o in out])
    ax.set_title('기존 시간 평균 경보 비교', color=INK)
    ax.set_ylabel('%'); ax.set_ylim(0, 116)
    ax.legend(fontsize=9.5, loc='upper left'); grid(ax)


def tiles(fig, items):
    """헤드라인 숫자. 그림을 안 읽어도 이 줄만 보면 성능을 알 수 있게."""
    for i, (val, lab, sub) in enumerate(items):
        x = .012 + i * .252
        fig.text(x, .938, val, fontsize=27, fontweight='bold', color=INK, va='top')
        fig.text(x, .909, lab, fontsize=11, color=INK, va='top')
        fig.text(x, .893, sub, fontsize=9.5, color=MUTED, va='top')


if __name__ == '__main__':
    X, P = load_predictions('--refresh' in sys.argv)
    C = P[P.fold.isin(CORE_FOLDS)].reset_index(drop=True)      # 판단 기준은 CORE (D22)

    s = score(C.rename(columns={'pred': 'p'}))
    sb = score(C.rename(columns={'pred_base': 'p'}))
    base_mae = sb.MAE
    bl = pd.read_csv(ROOT / 'outputs' / 'baseline_results.csv')
    bl = bl[(bl.group == '고유일') & (bl.fold == 'CORE(7 to 9월)')].set_index('model')
    mr = pd.read_csv(ROOT / 'outputs' / 'model_results.csv')
    mr = mr[mr.fold == 'CORE(7 to 9월)'].groupby('run', sort=False).MAE.mean()

    fig = plt.figure(figsize=(15.5, 16.4))
    gs = GridSpec(3, 2, figure=fig, hspace=.40, wspace=.22, top=.836, bottom=.042,
                  left=.055, right=.975)

    fig.text(.012, .993, '전력사용량 예측 모델 — 최종 성능 요약', fontsize=19,
             fontweight='bold', va='top')
    fig.text(.012, .967,
             f'LightGBM · 1주 앞 예측 · 평가 CORE(7 to 9월) 고유일 {int(s.n):,}행 · '
             f'시드 {len(SEEDS)}개 예측 평균 · 복제일 가중 0.3 · 피크행 가중 3배',
             fontsize=10.5, color=INK2, va='top')

    tiles(fig, [
        (f'{s.MAE:.2f}', 'MAE  평균 오차', f'목표선 {bl.MAE.min():.1f} 대비 '
                                          f'{(1 - s.MAE / bl.MAE.min()) * 100:.0f}% 감소'),
        (f'{s.RMSE:.2f}', 'RMSE  큰 오차 가중', f'목표선 {bl.RMSE.min():.1f} 대비 '
                                               f'{(1 - s.RMSE / bl.RMSE.min()) * 100:.0f}% 감소'),
        (f'{s.peakMAE:.2f}', 'peakMAE  피크 구간', f'보정 전 {sb.peakMAE:.1f} → 요금이 걸린 구간'),
        (f'{((C.y - C.pred).abs() <= 10).mean() * 100:.0f}%', '오차 ±10 이내',
         f'전체 {int(s.n):,}행 중'),
    ])

    # 대표일 2개. 잘 맞은 날만 고르면 그림이 거짓말을 한다.
    # 가장 어려운 날(부하 최대)과 가장 흔한 날(오차가 중앙값)을 같이 놓는다
    op = C[C.is_off == 0]
    day_mae = (op.y - op.pred).abs().groupby(op['날짜']).mean()
    hi_day = op.groupby('날짜').y.max().idxmax()
    mid_day = (day_mae - day_mae.median()).abs().drop(hi_day, errors='ignore').idxmin()
    for ax, d, t in ((fig.add_subplot(gs[0, 0]), hi_day, '가장 어려운 날 (부하 최대)'),
                     (fig.add_subplot(gs[0, 1]), mid_day, '보통 날 (오차가 중앙값)')):
        panel_day(ax, C[C['날짜'] == d].reset_index(drop=True), f'{t}  {d}')

    panel_scatter(fig.add_subplot(gs[1, 0]), C)
    panel_ladder(fig.add_subplot(gs[1, 1]), [
        ('목표선 (전주 같은 시각)', bl.loc['lag168', 'MAE']),
        ('기본 설정', mr.iloc[0]),
        ('1주 앞 + 복제일 가중', mr.get('8 조합 (가중치+1주앞)', np.nan)),
        ('피크 보정 전', base_mae),
        ('★ 최종 설정', s.MAE)])
    panel_peakband(fig.add_subplot(gs[2, 0]), C)
    panel_alarm(fig.add_subplot(gs[2, 1]), C, X)

    hs = [Line2D([], [], color=INK, lw=2.3), Line2D([], [], color=PRED, lw=2, ls=(0, (4, 2))),
          Line2D([], [], color=HI, lw=1.6, ls=(0, (1, 1.6))),
          Line2D([], [], color=MUTED, lw=1.1, ls=':'),
          Line2D([], [], color=PEAK, lw=0, marker='o', ms=6)]
    fig.legend(hs, ['실제', '예측', '시간 평균 P90', '목표선 (전주 같은 시각)',
                    '피크 구간 (상위 5%)'],
               loc='upper left', bbox_to_anchor=(.010, .872), ncol=5, fontsize=10)

    path = OUT / '00_dashboard.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)

    txt = OUT / 'summary.txt'
    dm = C.groupby('날짜').apply(lambda t: t.y.max() - t.pred.max(), include_groups=False)
    txt.write_text(
        f"""최종 모델 성능 요약  (src/report_final.py 생성)
평가 기준: CORE(7 to 9월) · 고유일 {int(s.n):,}행 · 시드 {list(SEEDS)} 예측 평균
설정: LightGBM / h7_lag / train_ok_strict / 복제일 가중 0.3 / 피크행 가중 3배

[전체 정확도]
  MAE      {s.MAE:.2f}   (목표선 lag168 {bl.loc['lag168', 'MAE']:.1f})
  RMSE     {s.RMSE:.2f}   (목표선 {bl.RMSE.min():.1f})
  peakMAE  {s.peakMAE:.2f}   (목표선 {bl.peakMAE.min():.1f})
  오차 ±10 이내 {((C.y - C.pred).abs() <= 10).mean() * 100:.1f}%

[피크 보정 효과]  피크행 가중치를 끈 설정과 비교
  MAE      {base_mae:.2f} → {s.MAE:.2f}
  일 최대 평균오차 {(C.groupby('날짜').apply(lambda t: t.y.max() - t.pred_base.max(), include_groups=False)).mean():+.2f} → {dm.mean():+.2f}
  (양수 = 일 최대를 낮게 본다는 뜻. 줄었지만 아직 0은 아니다)
  ※ 팀원이 기록한 +9.62 는 피크 가중치 이전 설정의 값이다. 여기까지 내려왔다.

[남은 약점]
  하루 중 어느 시각이 최대인지는 맞히지 못한다 (적중률 약 20%).
  경보는 "몇 시"가 아니라 "그날 위험한가"로 내야 한다. 근거는 작업내역(조선제).txt [9-3]

[D10 현재 프로젝트 경보]
  15분 최대 직접 P90 >= 학습 15분 최대 Q95. 하루에 한 번이라도 경보이면 위험일.
  위 그림의 시간 평균 P90 비교와 구분하며 상세 검증은 peak_alarm_max15_summary.txt에 있다.
  계약전력 초과 판정과 별도이며 P90의 90% 포함률을 보장하지 않는다.
""", encoding='utf-8')

    print(f'\n[최종 모델] CORE {int(s.n):,}행 · MAE {s.MAE:.2f} / RMSE {s.RMSE:.2f} / '
          f'peakMAE {s.peakMAE:.2f}')
    print(f'  피크 보정 전 MAE {base_mae:.2f} · 일 최대 평균오차 {dm.mean():+.2f}')
    print(f'\n→ {path}\n→ {txt}\n→ {CACHE}')
