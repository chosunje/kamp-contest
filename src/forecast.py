"""특정 날짜 하루(24시간)를 1주 앞에서 예측한다 (근거: 작업내역(조선제).txt [11]).

  예측 시점 = 대상일의 7일 전 0시. 그 시점까지의 전력 실적만 학습에 쓴다.
  대상일의 생산계획·기상예보는 주어진다고 가정한다 (h7_full). 계획이 없으면 h7_no_plan.
  피크는 시간 평균을 예측한 뒤 시각별 계수로 15분 최대를 환산한다 (15분 컬럼 A안, 같은 문서 [10]).

실행: python src/forecast.py            (마지막 예측 가능일을 자동 선택)
      python src/forecast.py 20210914   (날짜 지정)
"""
import sys
import numpy as np, pandas as pd
from features import build, FEATURES, peak_ratio
from model import make_model, CLONE_W
from preprocess import ROOT

HORIZON = 7   # 일. 대상일 7일 전에 예측한다


def forecast(X, target_date, cols=None, model='lgbm', train_flag='train_ok_strict',
             clone='weight', seed=0):
    """대상일 24행을 예측해 돌려준다. 학습에는 예측 시점 이전 행만 쓴다."""
    cols = cols or FEATURES['h7_full']
    day = X[X['날짜'] == target_date]
    if day.empty:
        raise SystemExit(f'{target_date} 행이 없다. 미래 날짜를 예측하려면 그날의 '
                         f'생산계획·기상 행을 먼저 데이터에 넣어야 한다.')

    cutoff = day['dt'].min() - pd.Timedelta(days=HORIZON)     # 이 시각 이후 정보는 쓰지 않는다
    tr = X[(X['dt'] < cutoff) & X[train_flag]]
    w = np.where(tr.is_clone, CLONE_W, 1.0) if clone == 'weight' else None
    g = make_model(model, seed).fit(tr[cols], tr.target, sample_weight=w)

    out = day[['dt', '날짜', 'hour']].copy()
    out['pred'] = g.predict(day[cols])
    # 환산 계수도 학습 구간에서만 뽑는다 (대상일 정보를 쓰면 누수)
    out['pred_max15'] = out.pred * out.hour.map(peak_ratio(tr).ratio_a)
    out['actual'] = day.target.values                          # 실적이 있으면 채워진다
    out['actual_max15'] = day.target_max15.values
    return out, cutoff, len(tr)


if __name__ == '__main__':
    X = build()
    dates = sorted(X['날짜'].unique())
    target = int(sys.argv[1]) if len(sys.argv) > 1 else dates[-1]

    out, cutoff, n_tr = forecast(X, target)
    path = ROOT / 'outputs' / f'forecast_{target}.csv'
    out.to_csv(path, index=False, encoding='utf-8-sig')

    print(f'대상일 {target} · 예측 시점 {cutoff} (7일 전) · 학습 {n_tr}행')
    print(out[['hour', 'pred', 'pred_max15', 'actual', 'actual_max15']].round(1).to_string(index=False))

    e = out.actual - out.pred
    print(f'\n[정확도] MAE {e.abs().mean():.2f} · RMSE {np.sqrt((e ** 2).mean()):.2f} '
          f'· 최대오차 {e.abs().max():.1f}')
    i = out.pred.idxmax()
    j = out.actual.idxmax()
    print(f'[일 피크] 예측 {out.hour[i]}시 {out.pred[i]:.0f} (15분 최대 환산 {out.pred_max15[i]:.0f}) '
          f'/ 실제 {out.hour[j]}시 {out.actual[j]:.0f} (15분 최대 {out.actual_max15[j]:.0f})')
    print('→', path)
