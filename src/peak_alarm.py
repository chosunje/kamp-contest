"""피크 위험 경보의 성능 검증 (인수인계(2026-09-30) [2] 1순위의 "검증" 항목).

  경보는 맞히는 것만으로 평가할 수 없다. 두 가지를 같이 봐야 한다.
    재현율  실제로 임계를 넘은 시각 중 몇 %를 미리 잡았나  (놓치면 요금을 맞는다)
    정밀도  경보를 낸 시각 중 몇 %가 실제로 넘었나          (헛경보가 잦으면 현장이 무시한다)
  둘은 맞바꾸는 관계다. 상한선을 높게 잡으면 재현율이 오르고 정밀도가 떨어진다.

  ★ 점 예측으로 경보를 내면 안 된다는 것을 수치로 보이는 것이 이 스크립트의 요점이다.
    점 예측은 피크를 낮게 보므로 임계값에 닿지 못해 경보가 늦는다.

실행: python src/peak_alarm.py
출력: outputs/peak_alarm_eval.csv   설정별·임계별 정밀도/재현율
      outputs/peak_alarm_days.csv   일 단위 경보 결과 (현장이 받는 형태)
"""
import numpy as np, pandas as pd

from features import build
from model import (rolling_eval, FINAL_CFG, ALARM_CFG, ALT_CFG, CORE_FOLDS, SEEDS)
from preprocess import ROOT

OUT_EVAL = ROOT / 'outputs' / 'peak_alarm_eval.csv'
OUT_DAYS = ROOT / 'outputs' / 'peak_alarm_days.csv'

# 임계값을 학습 구간 상위 몇 %로 잡을 것인가. D10 이 정해지면 하나로 확정한다
QS = (.95, .90)


def oof(X, cfg, folds=CORE_FOLDS, seeds=SEEDS):
    """시드 평균 예측을 행 단위로 돌려준다 (폴드별 임계값도 같이)."""
    parts = [rolling_eval(X, folds=folds, seed=s, **cfg).set_index('idx') for s in seeds]
    r = parts[0][['fold', 'y']].copy()
    r['p'] = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
    return r


def thresholds(X, folds, q):
    """폴드마다 학습 구간에서만 임계값을 정한다 (평가 구간을 보고 정하면 누수)."""
    t = {}
    for m in folds:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
        t[m] = tr.target.quantile(q)
    return t


def evaluate(r, thr, label, q):
    """시각 단위와 일 단위로 정밀도·재현율을 낸다."""
    t = r.fold.map(thr)
    pred, real = r.p >= t, r.y >= t
    tp, fp, fn = (pred & real).sum(), (pred & ~real).sum(), (~pred & real).sum()
    # 일 단위: 그날 한 시각이라도 넘으면 "위험일". 현장이 받는 단위는 이쪽이다
    d = pd.DataFrame({'날짜': r.index.map(lambda i: i), 'pred': pred, 'real': real,
                      'fold': r.fold}).groupby(r.day_key).agg(pred=('pred', 'any'),
                                                              real=('real', 'any'))
    dtp, dfp, dfn = (d.pred & d.real).sum(), (d.pred & ~d.real).sum(), (~d.pred & d.real).sum()
    f = lambda a, b: round(a / b * 100, 1) if b else np.nan
    return {'설정': label, '임계': f'상위 {(1 - q) * 100:.0f}%',
            '시각_정밀도(%)': f(tp, tp + fp), '시각_재현율(%)': f(tp, tp + fn),
            '시각_경보수': int(tp + fp), '시각_실제피크': int(tp + fn),
            '일_정밀도(%)': f(dtp, dtp + dfp), '일_재현율(%)': f(dtp, dtp + dfn),
            '일_경보수': int(dtp + dfp), '일_위험일': int(dtp + dfn)}


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    cfgs = {'점 예측 (강화 최종)': FINAL_CFG, '피크 우선 (대안)': ALT_CFG,
            '★ 경보 상한 (분위 0.9)': ALARM_CFG}
    preds = {k: oof(X, c) for k, c in cfgs.items()}
    for r in preds.values():
        r['day_key'] = X.loc[r.index, '날짜'].values

    rows = []
    for q in QS:
        thr = thresholds(X, CORE_FOLDS, q)
        for k, r in preds.items():
            rows.append(evaluate(r, thr, k, q))
    E = pd.DataFrame(rows)
    E.to_csv(OUT_EVAL, index=False, encoding='utf-8-sig')

    print(f'[피크 경보 검증] CORE {CORE_FOLDS} · 고유일 · 시드 {list(SEEDS)} 평균')
    print('  재현율 = 실제 피크 중 미리 잡은 비율 / 정밀도 = 경보 중 실제로 맞은 비율\n')
    print(E.to_string(index=False))

    print('\n  → 점 예측으로 경보를 내면 재현율이 낮다. 피크를 낮게 보기 때문에')
    print('    임계값에 닿지 못하고 그대로 지나간다. 경보에는 분위 상한을 써야 한다.')

    # 현장이 받는 형태: 날짜별 "위험일 여부 + 가장 위험한 시각"
    q = QS[0]
    thr = thresholds(X, CORE_FOLDS, q)
    r = preds['★ 경보 상한 (분위 0.9)'].copy()
    r['hour'] = X.loc[r.index, 'hour'].values
    r['t'] = r.fold.map(thr)
    D = r.groupby('day_key').apply(lambda s: pd.Series({
        '임계': round(s.t.iloc[0], 1), '예측상한최대': round(s.p.max(), 1),
        '위험시각': int(s.loc[s.p.idxmax(), 'hour']), '경보': bool((s.p >= s.t).any()),
        '실제최대': round(s.y.max(), 1), '실제피크시각': int(s.loc[s.y.idxmax(), 'hour']),
        '실제초과': bool((s.y >= s.t).any())}), include_groups=False)
    D.to_csv(OUT_DAYS, encoding='utf-8-sig')
    ok = (D.경보 == D.실제초과).mean() * 100
    print(f'\n[일 단위 경보표] {len(D)}일 중 경보/실제가 일치한 날 {ok:.1f}%')
    print(D.head(12).to_string())
    print(f'\n→ {OUT_EVAL}\n→ {OUT_DAYS}')
