"""테스트 구간 예측 파일 생성 (인수인계(2026-09-30) [3] 5번, 제출물 필수).

  forecast.py 는 하루를 보기 좋게 찍어 주는 도구다. 제출물은 그것과 다르다.
  여러 날을 한 번에, 정해진 열 이름으로, 실적이 없어도 돌아가야 한다.

  ── 어떻게 동작하나 ────────────────────────────────────────────────────
  예측 시점(cutoff)을 정하면 그 시각 이전 행만 학습에 쓴다. 대상 구간의 전력은
  한 번도 보지 않는다. 임계값·환산계수도 전부 학습 구간에서만 뽑는다.
  대상 구간 행에 전력이 비어 있어도(= 진짜 테스트 데이터) 그대로 돌아간다.

  ── 테스트 데이터가 오면 해야 할 일 ────────────────────────────────────
  1. 테스트 기간의 생산계획·기상 행을 dataset 에 이어 붙인다 (전력 열은 비워 둔다)
  2. python src/preprocess.py 로 다시 읽히는지 확인한다
  3. python src/predict_test.py 20210915 20210930
     → outputs/submission_20210915_20210930.csv
  ※ 생산계획이 안 오면 --no-plan 을 붙인다 (피처 세트가 h7_no_plan 으로 바뀐다).
    성능이 크게 떨어지므로(MAE 약 9 → 18) 보고서에 반드시 명시할 것.

실행: python src/predict_test.py 20210915 20210930
      python src/predict_test.py 20210915 20210930 --no-plan
      python src/predict_test.py                      (실적이 있는 마지막 7일로 시연)
"""
import sys
import numpy as np, pandas as pd

from features import build, FEATURES, peak_ratio
from model import make_model, CLONE_W, FINAL_CFG, ALARM_CFG
from preprocess import ROOT

ALARM_Q = .95    # 경보 임계값 (D10 확정 전 임시. 학습 구간 상위 5%)
SEEDS = (0, 1, 2)


def _weights(tr, peak_w):
    """학습 가중치. model.rolling_eval 과 같은 방식 (복제일 하향 x 피크행 상향)."""
    w = np.where(tr.is_clone, CLONE_W, 1.0)
    if peak_w != 1.0:
        w = w * np.where(tr.target >= tr.target.quantile(.95), peak_w, 1.0)
    return w


def _fit_predict(cfg, tr, te, peak_w, seeds=SEEDS):
    """시드를 바꿔 여러 번 학습한 뒤 예측을 평균낸다 (시드 하나에 운을 걸지 않는다)."""
    w = _weights(tr, peak_w)
    ps = [make_model(cfg['model'], s, cfg.get('params')).fit(
        tr[cfg['cols']], tr.target, sample_weight=w).predict(te[cfg['cols']]) for s in seeds]
    return np.mean(ps, axis=0)


def predict_range(X, start, end, no_plan=False, seeds=SEEDS):
    """start 부터 end 까지(양끝 포함)를 예측한다. 학습은 start 이전 행만 쓴다."""
    te = X[(X['날짜'] >= start) & (X['날짜'] <= end)].copy()
    if te.empty:
        raise SystemExit(f'{start} to {end} 구간 행이 없다. 테스트 기간의 생산계획·기상 행을 '
                         f'먼저 dataset 에 넣고 preprocess.py 를 돌릴 것.')

    cutoff = te['dt'].min()                       # 이 시각 이후 정보는 일절 쓰지 않는다
    tr = X[(X['dt'] < cutoff) & X['train_ok_strict']]
    if len(tr) < 200:
        raise SystemExit(f'학습 행이 {len(tr)}개뿐이다. 예측 시점이 너무 이르다.')

    point, alarm = dict(FINAL_CFG), dict(ALARM_CFG)
    if no_plan:                                   # 생산계획이 안 오는 경우
        point['cols'] = alarm['cols'] = FEATURES['h7_no_plan']

    ratio = peak_ratio(tr).ratio_a                # 환산 계수도 학습 구간에서만
    thr = tr.target.quantile(ALARM_Q)

    out = te[['dt', '날짜', 'hour']].copy()
    out['pred'] = _fit_predict(point, tr, te, point['peak_w'], seeds)
    out['pred_hi'] = _fit_predict(alarm, tr, te, 1.0, seeds)
    out['pred_max15'] = out.pred * out.hour.map(ratio)
    out['pred_hi_max15'] = out.pred_hi * out.hour.map(ratio)
    out['alarm'] = out.pred_hi >= thr
    out['actual'] = te.target.values              # 진짜 테스트라면 비어 있다
    return out, cutoff, len(tr), thr


def day_summary(out):
    """일 단위 요약. 현장이 실제로 보는 것은 "그날 최대가 얼마고 언제인가" 이다."""
    g = out.groupby('날짜')
    d = pd.DataFrame({
        '예측_일최대': g.pred.max().round(1),
        '예측_피크시각': g.apply(lambda s: int(s.loc[s.pred.idxmax(), 'hour']), include_groups=False),
        '예측_15분최대': g.pred_max15.max().round(1),
        '경보상한_일최대': g.pred_hi.max().round(1),
        '경보': g.alarm.any(),
    })
    if out.actual.notna().any():
        d['실제_일최대'] = g.actual.max()
        d['실제_피크시각'] = g.apply(lambda s: int(s.loc[s.actual.idxmax(), 'hour'])
                                   if s.actual.notna().any() else -1, include_groups=False)
    return d


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    no_plan = '--no-plan' in sys.argv
    X = build()
    dates = sorted(X['날짜'].unique())
    start, end = (int(args[0]), int(args[1])) if len(args) >= 2 else (dates[-7], dates[-1])

    out, cutoff, n_tr, thr = predict_range(X, start, end, no_plan=no_plan)
    tag = 'noplan' if no_plan else 'full'
    path = ROOT / 'outputs' / f'submission_{start}_{end}_{tag}.csv'
    out.to_csv(path, index=False, encoding='utf-8-sig')

    print(f'[테스트 예측] {start} to {end} · {len(out)}행 ({out["날짜"].nunique()}일)')
    print(f'  예측 시점 {cutoff} 이전 행만 학습 ({n_tr}행) · 피처 '
          f'{"h7_no_plan (생산계획 없음)" if no_plan else "h7_lag"} · 시드 {list(SEEDS)} 평균')

    D = day_summary(out)
    print('\n[일 단위 요약]')
    print(D.to_string())

    if out.actual.notna().any():
        e = out.actual - out.pred
        print(f'\n[정확도] MAE {e.abs().mean():.2f} · RMSE {np.sqrt((e ** 2).mean()):.2f}')
        dm = D['실제_일최대'] - D['예측_일최대']
        print(f'[일 최대] 평균오차 {dm.mean():+.2f} (양수 = 낮게 봄) · MAE {dm.abs().mean():.2f}')
        hit = (D['예측_피크시각'] == D['실제_피크시각']).mean() * 100
        print(f'[피크 시각 적중] {hit:.1f}%')
    else:
        print('\n실적 열이 비어 있다 = 진짜 테스트 구간. 정확도는 계산하지 않는다.')
    print(f'\n[경보] 임계 {thr:.0f} · 위험일 {int(D.경보.sum())}일 / {len(D)}일')
    print('→', path)
