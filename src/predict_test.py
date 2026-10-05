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
from model import make_model, CLONE_W, FINAL_CFG, ALARM_CFG, MAX15_CFG
from preprocess import ROOT

ALARM_Q = .95    # 경보 임계값 (D10 확정 전 임시. 학습 구간 상위 5%)
SEEDS = (0, 1, 2)


def _weights(tr, peak_w, target='target'):
    """학습 가중치. model.rolling_eval 과 같은 방식 (복제일 하향 x 피크행 상향)."""
    w = np.where(tr.is_clone, CLONE_W, 1.0)
    if peak_w != 1.0:
        w = w * np.where(tr[target] >= tr[target].quantile(.95), peak_w, 1.0)
    return w


def _fit_predict(cfg, tr, te, peak_w, seeds=SEEDS):
    """시드를 바꿔 여러 번 학습한 뒤 예측을 평균낸다 (시드 하나에 운을 걸지 않는다).
    cfg['target'] 이 있으면 그 열을 학습한다 (15분 최대 직접 학습용)."""
    tgt = cfg.get('target', 'target')
    w = _weights(tr, peak_w, tgt)
    ps = [make_model(cfg['model'], s, cfg.get('params')).fit(
        tr[cfg['cols']], tr[tgt], sample_weight=w).predict(te[cfg['cols']]) for s in seeds]
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

    point, alarm, max15 = dict(FINAL_CFG), dict(ALARM_CFG), dict(MAX15_CFG)
    # 15분 최대를 분위 0.9 로 직접 학습한 것. 요금 기준으로 경보를 낼 때 쓴다
    max15_hi = dict(ALARM_CFG, target='target_max15')
    if no_plan:                                   # 생산계획이 안 오는 경우
        for c in (point, alarm, max15, max15_hi):
            c['cols'] = FEATURES['h7_no_plan']

    ratio = peak_ratio(tr).ratio_a                # 환산 계수도 학습 구간에서만
    thr = tr.target.quantile(ALARM_Q)             # 시간 평균 기준 임계
    thr15 = tr.target_max15.quantile(ALARM_Q)     # 15분 최대 기준 임계

    out = te[['dt', '날짜', 'hour']].copy()
    out['pred'] = _fit_predict(point, tr, te, point['peak_w'], seeds)
    out['pred_hi'] = _fit_predict(alarm, tr, te, 1.0, seeds)
    # 15분 최대(요금 기준). 타깃을 바꿔 직접 학습하는 쪽이 계수 환산보다 낫다
    # (CORE MAE 7.32 vs 7.88). 환산값은 비교용으로 _환산 열에 남긴다
    out['pred_max15'] = _fit_predict(max15, tr, te, max15['peak_w'], seeds)
    out['pred_max15_hi'] = _fit_predict(max15_hi, tr, te, 1.0, seeds)
    out['pred_max15_환산'] = out.pred * out.hour.map(ratio)
    out['pred_hi_max15'] = out.pred_hi * out.hour.map(ratio)
    # D10 ① 이 미확정이라 두 기준의 경보를 모두 낸다 ([0] B-1).
    # CORE 에서는 15분 최대 기준이 정밀도 93.2% 로 시간평균(86.4%)보다 낫다  [8-6] 3번
    out['alarm'] = out.pred_hi >= thr
    out['alarm_max15'] = out.pred_max15_hi >= thr15
    out['actual'] = te.target.values              # 진짜 테스트라면 비어 있다
    return out, cutoff, len(tr), (thr, thr15)


def day_summary(out):
    """일 단위 요약. 현장이 실제로 보는 것은 "그날 최대가 얼마고 언제인가" 이다.

    ★ 일 최대는 점 예측이 아니라 분위 0.9 로 추정한다 ([7-5]).
      점 예측은 시각마다 "가운데 값" 을 맞히므로, 그 24개의 최대는 실제 일 최대보다
      체계적으로 낮다 (편향 +6.8). 분위 0.9 의 최대는 편향이 거의 0 이다 (-0.16).
      시간별 정확도는 점 예측이, 일 최대는 분위 0.9 가 담당한다 — 용도가 다르다."""
    g = out.groupby('날짜')
    d = pd.DataFrame({
        '일최대_추정': g.pred_hi.max().round(1),            # ★ 권장 추정값 (분위 0.9)
        '일최대_점예측': g.pred.max().round(1),             # 참고용. 낮게 나온다
        '예측_피크시각': g.apply(lambda s: int(s.loc[s.pred.idxmax(), 'hour']), include_groups=False),
        '일최대_15분': g.pred_max15_hi.max().round(1),       # 15분 최대를 직접 학습한 상한
        '일최대_15분환산': g.pred_hi_max15.max().round(1),    # 참고. 시간평균 x 계수
        '경보_시간평균': g.alarm.any(),
        '경보_15분최대': g.alarm_max15.any(),
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

    out, cutoff, n_tr, (thr, thr15) = predict_range(X, start, end, no_plan=no_plan)
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
        for lb, c in (('분위 0.9 (권장)', '일최대_추정'), ('점 예측 (참고)', '일최대_점예측')):
            dm = D['실제_일최대'] - D[c]
            print(f'[일 최대 · {lb}] 평균오차 {dm.mean():+.2f} (양수 = 낮게 봄) '
                  f'· MAE {dm.abs().mean():.2f}')
        hit = (D['예측_피크시각'] == D['실제_피크시각']).mean() * 100
        print(f'[피크 시각 적중] {hit:.1f}%')
    else:
        print('\n실적 열이 비어 있다 = 진짜 테스트 구간. 정확도는 계산하지 않는다.')
    print(f'\n[경보] D10 ① 이 미확정이라 두 기준을 함께 낸다  [0] B-1')
    print(f'  시간평균 기준  임계 {thr:.0f} · 위험일 {int(D.경보_시간평균.sum())}일 / {len(D)}일')
    print(f'  15분최대 기준  임계 {thr15:.0f} · 위험일 {int(D.경보_15분최대.sum())}일 / {len(D)}일'
          f'   ← 요금 기준. CORE 에서 정밀도 93.2% 로 더 낫다')
    print('→', path)
