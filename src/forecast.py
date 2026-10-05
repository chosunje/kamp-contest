"""특정 날짜 하루(24시간)를 1주 앞에서 예측하고, 피크 위험 경보를 낸다.

  예측 시점 = 대상일의 7일 전 0시. 그 시점까지의 전력 실적만 학습에 쓴다.
  대상일의 생산계획·기상예보는 주어진다고 가정한다 (h7_full). 계획이 없으면 h7_no_plan.
  피크는 시간 평균을 예측한 뒤 시각별 계수로 15분 최대를 환산한다 (15분 컬럼 A안, [10]).

  ★ 경보 (작업내역(최치훈).txt [15] 1순위)
    점 예측(pred)은 피크를 낮게 본다. 이것을 그대로 임계값에 대면 경보가 늦는다.
    그래서 분위 0.9 모델을 따로 학습해 상한선(pred_hi)을 같이 낸다.
    "이 시각은 임계값을 넘을 수 있다"는 판단은 pred 가 아니라 pred_hi 로 한다.
    근거: 분위 0.9 는 점 예측으로는 나쁘지만(MAE 14.12) 피크만 보면 peakMAE 8.13 으로
          모든 설정 중 가장 정확하다 (작업내역(조선제).txt [17] 13-2)

실행: python src/forecast.py               (마지막 예측 가능일을 자동 선택)
      python src/forecast.py 20210914      (날짜 지정)
      python src/forecast.py 20210914 190  (경보 임계값도 지정)
"""
import sys
import numpy as np, pandas as pd
from features import build, peak_ratio
from model import make_model, CLONE_W, FINAL_CFG, ALARM_CFG
from preprocess import ROOT

HORIZON = 7      # 일. 대상일 7일 전에 예측한다
ALARM_Q = .95    # 경보 임계값을 따로 안 주면 학습 구간 상위 5% 를 쓴다 (D10 확정 전 임시)


def _weights(tr, clone, peak_w):
    """학습 가중치. model.rolling_eval 과 같은 방식으로 만든다 (복제일 하향 x 피크행 상향)."""
    w = np.where(tr.is_clone, CLONE_W, 1.0) if clone == 'weight' else np.ones(len(tr))
    if peak_w != 1.0:   # 피크 과소예측 보정: 학습 구간 상위 5% 행을 더 무겁게 학습한다
        w = w * np.where(tr.target >= tr.target.quantile(.95), peak_w, 1.0)
    return w


def forecast(X, target_date, cols=None, model='lgbm', train_flag='train_ok_strict',
             clone='weight', seed=0, params=None, peak_w=None, thr=None):
    """대상일 24행을 예측해 돌려준다. 학습에는 예측 시점 이전 행만 쓴다.
    기본값은 model.py 에서 고른 최종 설정(FINAL_CFG)과 같다.

    돌려주는 열
      pred          시간 평균 전력의 점 예측
      pred_hi       분위 0.9 상한 (경보 판단은 이 값으로 한다)
      pred_max15    pred 를 시각별 계수로 15분 최대로 환산한 값 (요금 기준)
      pred_hi_max15 pred_hi 를 같은 방식으로 환산한 값
      alarm         pred_hi 가 임계값을 넘는가
    """
    cols = cols or FINAL_CFG['cols']
    params = FINAL_CFG['params'] if params is None else params
    peak_w = FINAL_CFG['peak_w'] if peak_w is None else peak_w
    day = X[X['날짜'] == target_date]
    if day.empty:
        raise SystemExit(f'{target_date} 행이 없다. 미래 날짜를 예측하려면 그날의 '
                         f'생산계획·기상 행을 먼저 데이터에 넣어야 한다.')

    cutoff = day['dt'].min() - pd.Timedelta(days=HORIZON)     # 이 시각 이후 정보는 쓰지 않는다
    tr = X[(X['dt'] < cutoff) & X[train_flag]]
    g = make_model(model, seed, params).fit(tr[cols], tr.target,
                                            sample_weight=_weights(tr, clone, peak_w))
    # 경보용 상한. 분위 0.9 를 별도로 학습한다 (점 예측과 목적이 달라 같은 모델을 못 쓴다)
    gh = make_model(ALARM_CFG['model'], seed, ALARM_CFG['params']).fit(
        tr[ALARM_CFG['cols']], tr.target, sample_weight=_weights(tr, 'weight', 1.0))

    # 임계값도 학습 구간에서만 정한다 (대상일을 보고 정하면 누수)
    thr = tr.target.quantile(ALARM_Q) if thr is None else float(thr)
    ratio = peak_ratio(tr).ratio_a                  # 환산 계수도 학습 구간에서만 뽑는다

    out = day[['dt', '날짜', 'hour']].copy()
    out['pred'] = g.predict(day[cols])
    out['pred_hi'] = gh.predict(day[ALARM_CFG['cols']])
    out['pred_max15'] = out.pred * out.hour.map(ratio)
    out['pred_hi_max15'] = out.pred_hi * out.hour.map(ratio)
    out['alarm'] = out.pred_hi >= thr
    out['actual'] = day.target.values                          # 실적이 있으면 채워진다
    out['actual_max15'] = day.target_max15.values
    return out, cutoff, len(tr), thr


if __name__ == '__main__':
    X = build()
    dates = sorted(X['날짜'].unique())
    target = int(sys.argv[1]) if len(sys.argv) > 1 else dates[-1]
    thr_arg = float(sys.argv[2]) if len(sys.argv) > 2 else None

    out, cutoff, n_tr, thr = forecast(X, target, thr=thr_arg)
    path = ROOT / 'outputs' / f'forecast_{target}.csv'
    out.to_csv(path, index=False, encoding='utf-8-sig')

    print(f'대상일 {target} · 예측 시점 {cutoff} (7일 전) · 학습 {n_tr}행')
    print(out[['hour', 'pred', 'pred_hi', 'pred_max15', 'alarm',
               'actual', 'actual_max15']].round(1).to_string(index=False))

    e = out.actual - out.pred
    print(f'\n[정확도] MAE {e.abs().mean():.2f} · RMSE {np.sqrt((e ** 2).mean()):.2f} '
          f'· 최대오차 {e.abs().max():.1f}')
    i, j, k = out.pred.idxmax(), out.actual.idxmax(), out.pred_hi.idxmax()
    # 일 최대는 분위 0.9 로 추정한다. 점 예측의 최대는 체계적으로 낮다 ([7-5])
    print(f'[일 최대] 추정 {out.pred_hi[k]:.0f} (분위 0.9, 15분 환산 {out.pred_hi_max15[k]:.0f}) '
          f'· 점 예측 {out.pred[i]:.0f} (참고, 낮게 나온다) '
          f'/ 실제 {out.actual[j]:.0f} (15분 최대 {out.actual_max15[j]:.0f}, {out.hour[j]}시)')

    # ★ 경보. 현장이 실제로 쓰는 산출물은 이 줄이다
    hit = out[out.alarm]
    print(f'\n[피크 경보] 임계 {thr:.0f} (학습 구간 상위 {(1 - ALARM_Q) * 100:.0f}%)')
    if len(hit):
        print(f'  위험 시각 {list(hit.hour)} · 상한 최대 {hit.pred_hi.max():.0f} '
              f'({hit.loc[hit.pred_hi.idxmax(), "hour"]}시)')
        real = out[out.actual >= thr]
        print(f'  실제 임계 초과 시각 {list(real.hour)} '
              f'→ 맞힌 시각 {sorted(set(hit.hour) & set(real.hour))}')
    else:
        print('  없음 (이 날은 임계값을 넘을 시각이 없다고 본다)')
        real = out[out.actual >= thr]
        if len(real):
            print(f'  ※ 실제로는 {list(real.hour)} 시가 임계를 넘었다 — 놓친 경보')
    print('→', path)
