"""생산계획 기반 운영 규칙 후처리 실험.

오류분석에서 공휴일/부분가동일의 '마지막 생산시간 이후'에 모델이 평일 주간부하를
과대예측하는 현상이 확인됐다. 생산계획에서 마지막 양(+)의 생산 시각은 예측 전에
알 수 있으므로, 그 이후의 예측값이 과도하게 높을 때 학습구간의 동일 조건 전력
중앙값을 상한(cap)으로 적용한다.

초기 h7_full 비교에서 CORE(7 to 9월)의 MAE·RMSE가 악화되어 이 규칙의
일괄 적용은 채택하지 않았다. 코드와 산출물은 비교 실험 근거로 보존한다.

선행:
    python src/error_analysis.py
실행:
    python src/operational_correction.py
"""
import numpy as np
import pandas as pd

from error_analysis import add_plan_window_features
from features import build
from preprocess import ROOT

PRED_IN = ROOT / 'outputs' / 'error_predictions.csv'
OUT_PRED = ROOT / 'outputs' / 'operational_correction_predictions.csv'
OUT_RES = ROOT / 'outputs' / 'operational_correction_results.csv'
OUT_SUMMARY = ROOT / 'outputs' / 'operational_correction_summary.txt'


def calc(s, pred_col):
    e = s.target - s[pred_col]
    peak = s.is_peak.astype(bool)
    return {
        'n': len(s),
        'MAE': e.abs().mean(),
        'RMSE': np.sqrt((e ** 2).mean()),
        'peakMAE': e[peak].abs().mean(),
        'bias': e.mean(),
    }


def main():
    if not PRED_IN.exists():
        raise SystemExit('error_predictions.csv가 없다. 먼저 python src/error_analysis.py 를 실행할 것.')

    X = add_plan_window_features(build())
    P = pd.read_csv(PRED_IN, parse_dates=['dt'])
    P['pred_corrected'] = P['pred']
    P['correction_cap'] = np.nan
    P['corrected'] = False

    for fold, s in P.groupby('fold', sort=False):
        idx = s.index
        cutoff = s.dt.min()
        tr = X[(X.dt < cutoff) & X.train_ok_strict]
        hist = tr.loc[tr.after_prod_end == 1, 'target']
        if hist.empty:
            continue
        cap = float(hist.median())
        mask = (P.index.isin(idx)) & (P.after_prod_end == 1) & (P['pred'] > cap)
        P.loc[mask, 'pred_corrected'] = cap
        P.loc[P.index.isin(idx), 'correction_cap'] = cap
        P.loc[mask, 'corrected'] = True

    P['error_corrected'] = P.target - P.pred_corrected
    P['abs_error_corrected'] = P.error_corrected.abs()

    from model import CORE_FOLDS, CORE
    rows = []
    scopes = [('ALL', P), (CORE, P[P.fold.isin(CORE_FOLDS)]),
              *[(f, g) for f, g in P.groupby('fold', sort=False)]]
    for scope, s in scopes:
        a = calc(s, 'pred')
        b = calc(s, 'pred_corrected')
        rows.append({
            'fold': scope,
            'n': len(s),
            'changed': int(s.corrected.sum()),
            'MAE_base': a['MAE'], 'MAE_corrected': b['MAE'], 'MAE_delta': b['MAE'] - a['MAE'],
            'RMSE_base': a['RMSE'], 'RMSE_corrected': b['RMSE'], 'RMSE_delta': b['RMSE'] - a['RMSE'],
            'peakMAE_base': a['peakMAE'], 'peakMAE_corrected': b['peakMAE'],
            'bias_base': a['bias'], 'bias_corrected': b['bias'],
        })
    R = pd.DataFrame(rows)

    P.to_csv(OUT_PRED, index=False, encoding='utf-8-sig')
    R.to_csv(OUT_RES, index=False, encoding='utf-8-sig')

    a = R.iloc[0]
    c = R[R.fold == CORE].iloc[0]
    lines = [
        '==============================================================',
        ' 생산종료 이후 운영규칙 보정 실험',
        '==============================================================',
        '규칙: 생산계획상 마지막 생산시간 이후이고 모델 예측이 학습구간 동일조건 중앙값보다',
        '      높으면 그 중앙값으로 상한 제한한다. 현재 각 폴드의 중앙값은 23 수준이다.',
        '',
        f'[전체 5 to 9월] 변경 {int(a.changed):,}/{int(a.n):,}행',
        f'MAE  {a.MAE_base:.2f} → {a.MAE_corrected:.2f} ({a.MAE_delta:+.2f})',
        f'RMSE {a.RMSE_base:.2f} → {a.RMSE_corrected:.2f} ({a.RMSE_delta:+.2f})',
        f'peakMAE {a.peakMAE_base:.2f} → {a.peakMAE_corrected:.2f}',
        '',
        f'★ [{CORE}] 테스트 구간과 성격이 같은 구간 · 이 수치를 우선 본다',
        f'   변경 {int(c.changed):,}/{int(c.n):,}행',
        f'   MAE  {c.MAE_base:.2f} → {c.MAE_corrected:.2f} ({c.MAE_delta:+.2f})',
        f'   RMSE {c.RMSE_base:.2f} → {c.RMSE_corrected:.2f} ({c.RMSE_delta:+.2f})',
        '',
        '[폴드별]',
    ]
    for _, r in R[~R.fold.isin(['ALL', CORE])].iterrows():
        lines.append(
            f'- {r.fold}: 변경 {int(r.changed)}행, MAE {r.MAE_base:.2f} → {r.MAE_corrected:.2f}, '
            f'RMSE {r.RMSE_base:.2f} → {r.RMSE_corrected:.2f}'
        )
    decision = ('CORE MAE·RMSE가 함께 개선되어 추가 검토할 실험 후보로 남긴다.'
                if c.MAE_delta < 0 and c.RMSE_delta <= 0
                else 'CORE MAE·RMSE의 동시 개선이 없어 이 규칙의 일괄 적용을 채택하지 않는다.')
    lines += [
        '',
        f'판단: {decision}',
        '      현재 결과의 전체 개선은 대부분 5월에 집중되어 있으며 CORE 기준을 우선한다.',
        '출처: outputs/error_predictions.csv. 입력 모델·평가 조건은 outputs/error_summary.txt에 기록한다.',
    ]
    OUT_SUMMARY.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(f'→ {OUT_PRED}')
    print(f'→ {OUT_RES}')
    print(f'→ {OUT_SUMMARY}')


if __name__ == '__main__':
    main()
