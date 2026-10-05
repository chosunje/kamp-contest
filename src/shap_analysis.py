"""영향요인 분석 — SHAP (배점 3번 "영향요인 및 오류분석" 중 영향요인 쪽).

  error_analysis.py 와 역할을 나눈다.
    error_analysis.py  오차가 큰 조건이 무엇인가 (언제 틀리는가)
    shap_analysis.py   예측값을 만든 것이 무엇인가 (무엇으로 맞히는가)

  SHAP 은 "이 피처가 예측값을 얼마나 밀어올리거나 내렸는가"를 행마다 계산한다.
  LightGBM 자체의 중요도(gain)와 달리 방향과 크기를 같이 볼 수 있어 보고서에 쓰기 좋다.

  피처군(A 달력 / B 생산계획 / C 기상 / D 과거전력)별로 묶은 표도 함께 낸다.
  "생산계획이 없으면 얼마나 손해인가" 같은 질문에 바로 답할 수 있다.

실행: python src/shap_analysis.py
출력: outputs/shap_importance.csv   피처별 SHAP 기여도(%)
      outputs/shap_by_group.csv     피처군별 합계
"""
import numpy as np, pandas as pd, shap

from features import build, FEATURES, A, B, C, D, A7, B7, D7, H
from model import make_model, FINAL_CFG
from preprocess import ROOT

OUT_IMP = ROOT / 'outputs' / 'shap_importance.csv'
OUT_GRP = ROOT / 'outputs' / 'shap_by_group.csv'

# 피처가 어느 군에 속하는지. h7 세트는 A7/B7/D7 로 재배치되어 있으므로 원래 군으로 되돌린다
GROUP = {**{c: 'A 달력' for c in A + A7}, **{c: 'B 생산계획' for c in B + B7},
         **{c: 'C 기상' for c in C}, **{c: 'D 과거전력' for c in D + D7},
         **{c: 'H 공휴일x생산' for c in H}}


def shap_importance(X, cols=None, seed=0):
    """전체 학습 구간으로 한 번 학습해 SHAP 기여도를 본다 (전역 영향요인용).

    폴드마다 따로 보지 않는 이유: 여기서 알고 싶은 것은 "이 공장의 전력을 무엇이
    설명하는가"라는 구조이고, 그것은 폴드마다 달라질 성질이 아니다.
    성능 숫자가 아니므로 롤링 폴드로 나눌 필요가 없다.
    """
    cfg = FINAL_CFG
    cols = cols or cfg['cols']
    tr = X[X[cfg['train_flag']]]
    # 학습 가중치는 rolling_eval 과 똑같이 만든다 (복제일 하향 x 피크행 상향)
    w = np.where(tr.is_clone, cfg['clone_w'], 1.0)
    w = w * np.where(tr.target >= tr.target.quantile(.95), cfg.get('peak_w', 1.0), 1.0)
    g = make_model(cfg['model'], seed, cfg.get('params')).fit(tr[cols], tr.target, sample_weight=w)
    sv = shap.TreeExplainer(g).shap_values(tr[cols])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    return (imp / imp.sum() * 100).round(2)


if __name__ == '__main__':
    X = build()
    imp = shap_importance(X)
    imp.to_csv(OUT_IMP, header=['기여도(%)'], encoding='utf-8-sig')

    grp = imp.groupby(imp.index.map(GROUP)).sum().sort_values(ascending=False)
    grp.to_csv(OUT_GRP, header=['기여도(%)'], encoding='utf-8-sig')

    print('[영향요인] SHAP 기여도 상위 15 (%)  ·  최종 설정 기준')
    print(imp.head(15).to_string())
    print('\n[피처군별 합계 (%)]')
    print(grp.round(2).to_string())
    print(f'\n→ {OUT_IMP}\n→ {OUT_GRP}')
