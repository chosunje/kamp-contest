"""변수 중요도 + 전체 구간 대비 피크 구간 SHAP 비교.

  shap_analysis.py 와 역할을 나눈다 (둘 다 필요하다).
    shap_analysis.py  전체 구간의 기여도 한 벌 (무엇으로 맞히는가)
    importance.py     같은 값을 피크 구간과 나눠서 낸다 (피크에서 무엇이 달라지는가)
  후자가 1장의 "피크 동인은 생산량이 아니라 기온" 과 모델을 잇는 근거가 된다.

보고서에서 "피크는 생산량이 아니라 기온이 만든다"는 데이터 진단 결론을,
모델도 같은 구조로 학습했는지 확인하는 근거로 쓴다.
그래서 전체 구간뿐 아니라 피크 구간만 따로 떼어 기여도를 비교한다.

모델·학습 조건은 error_analysis.py 와 동일하다 (h7_full / strict / 복제일 0.3).
SHAP 은 시드 0 하나로만 계산한다 (중요도 순위는 시드에 거의 영향받지 않는다).

실행: python src/importance.py
  → outputs/importance_gain.csv   폴드·시드 평균 gain 중요도
  → outputs/importance_shap.csv   전체 구간 vs 피크 구간 평균 |SHAP|
"""
import numpy as np, pandas as pd

from features import build, FEATURES
from model import FOLDS, SEEDS, make_model, FINAL_CFG
from preprocess import ROOT

OUT_GAIN = ROOT / 'outputs' / 'importance_gain.csv'
OUT_SHAP = ROOT / 'outputs' / 'importance_shap.csv'

# 설정은 model.py 의 FINAL_CFG 를 따른다 (shap_analysis.py · error_analysis.py 와 동일)
COLS = FINAL_CFG['cols']
TRAIN_FLAG, CLONE_W = FINAL_CFG['train_flag'], FINAL_CFG['clone_w']
PARAMS, PEAK_W = FINAL_CFG['params'], FINAL_CFG['peak_w']

# 피처를 어느 정보군에서 왔는지로 묶는다 (피처목록.txt A to D 군)
GROUP = {}
for c in COLS:
    if c in ('prod', 'prod_cap', 'prod_log', 'prod_zero', 'day_prod', 'day_prod_hours',
             'day_prod_zero', 'prod_prev', 'prod_next', 'prod_diff', 'is_off',
             'after_off', 'off_run_prev', 'days_since_off'):
        GROUP[c] = '생산계획·휴무'
    elif c in ('cool', 'temp', 'humid', 'thi', 'wind', 'rain'):
        GROUP[c] = '기상'
    elif c.startswith('lag') or c.startswith('last_week'):
        GROUP[c] = '과거 전력'
    else:
        GROUP[c] = '달력'


def main():
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    gains, shaps = [], []
    for fold in FOLDS:
        va = X[(X.ym == fold) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X[TRAIN_FLAG]]
        if len(va) == 0 or len(tr) < 200:
            continue
        thr = tr.target.quantile(.95)              # 피크 정의는 model.py 와 동일 (학습 구간 상위 5%)
        w = np.where(tr.is_clone, CLONE_W, 1.0) * np.where(tr.target >= thr, PEAK_W, 1.0)

        for seed in SEEDS:
            g = make_model('lgbm', seed, PARAMS).fit(tr[COLS], tr.target, sample_weight=w)
            b = g.booster_.feature_importance(importance_type='gain')
            gains.append(pd.Series(b / b.sum(), index=COLS, name=f'{fold}_{seed}'))

        # SHAP: 시드 0 모델로 그 폴드의 평가 행을 설명한다
        g = make_model('lgbm', 0, PARAMS).fit(tr[COLS], tr.target, sample_weight=w)
        import shap
        sv = shap.TreeExplainer(g).shap_values(va[COLS])
        s = pd.DataFrame(np.abs(sv), columns=COLS)
        s['is_peak'] = (va.target >= thr).values
        shaps.append(s)
        print(f'  {fold}: 학습 {len(tr):,}행 / 평가 {len(va):,}행 (피크 {int(s.is_peak.sum())}행)', flush=True)

    G = pd.concat(gains, axis=1)
    gain = (G.mean(axis=1) * 100).sort_values(ascending=False)
    gi = pd.DataFrame({'피처': gain.index, '정보군': [GROUP[c] for c in gain.index],
                       'gain 중요도(%)': gain.round(2).values,
                       '폴드간 편차(%)': (G.loc[gain.index].std(axis=1) * 100).round(2).values})
    gi.insert(0, '순위', range(1, len(gi) + 1))
    gi.to_csv(OUT_GAIN, index=False, encoding='utf-8-sig')

    S = pd.concat(shaps, ignore_index=True)
    allm, pkm = S[COLS].mean(), S.loc[S.is_peak, COLS].mean()
    si = pd.DataFrame({'피처': COLS, '정보군': [GROUP[c] for c in COLS],
                       '전체 평균|SHAP|': allm.values.round(3),
                       '피크 구간 평균|SHAP|': pkm.values.round(3)})
    si['피크/전체 배율'] = (si['피크 구간 평균|SHAP|'] / si['전체 평균|SHAP|']).round(2)
    si = si.sort_values('피크 구간 평균|SHAP|', ascending=False).reset_index(drop=True)
    si.to_csv(OUT_SHAP, index=False, encoding='utf-8-sig')

    print('\n[gain 중요도 상위 10]')
    print(gi.head(10).to_string(index=False))
    print(f'\n[정보군별 gain 합계(%)]')
    print(gi.groupby('정보군')['gain 중요도(%)'].sum().sort_values(ascending=False).round(1).to_string())
    print(f'\n[SHAP] 전체 n={len(S):,} / 피크 n={int(S.is_peak.sum()):,}')
    print(si.head(10).to_string(index=False))
    print(f'\n→ {OUT_GAIN}\n→ {OUT_SHAP}')


if __name__ == '__main__':
    main()
