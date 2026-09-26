"""영향요인 분석과 예측오차 다발 조건 분석 (배점 3번: 영향요인 및 오류분석 15점).

  과제 요구사항 ② "예측오차가 크게 발생하는 생산조건 분석" 에 대응한다.
  최종 후보 설정으로 시간순 롤링 폴드 예측을 만든 뒤, 그 오차를 조건별로 쪼개 본다.
  영향요인은 SHAP(각 피처가 예측값을 얼마나 밀어올리거나 내렸는지)으로 본다.

실행: python src/error_analysis.py
출력: outputs/error_rows.csv       행 단위 예측·오차 (조건 분석 원자료)
      outputs/error_by_cond.csv    조건별 오차 집계
      outputs/shap_importance.csv  피처별 SHAP 중요도
"""
import numpy as np, pandas as pd, shap
from features import build, FEATURES
from model import make_model, rolling_eval, BASE, CLONE_W
from preprocess import ROOT

OUT_ROWS = ROOT / 'outputs' / 'error_rows.csv'
OUT_COND = ROOT / 'outputs' / 'error_by_cond.csv'
OUT_SHAP = ROOT / 'outputs' / 'shap_importance.csv'

# 현재 최종 후보 = 8번 설정 (1주 앞 피처 + 복제일 가중치 0.3 + strict)
FINAL = dict(BASE, cols=FEATURES['h7_full'], clone='weight', clone_w=CLONE_W)


def predict_rows(X, seeds=(0, 1, 2)):
    """롤링 폴드 예측을 시드별로 만들어 평균낸다. 행 단위 오차를 돌려준다."""
    parts = [rolling_eval(X, seed=s, **FINAL).set_index('idx') for s in seeds]
    p = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
    r = parts[0][['fold', 'y', 'peak']].copy()
    r['p'] = p
    r['err'] = r.y - r.p                 # 양수 = 과소예측 (실제가 더 높았다)
    r['abs_err'] = r.err.abs()
    return X.loc[r.index].join(r[['fold', 'y', 'p', 'err', 'abs_err', 'peak']])


def by_cond(R):
    """조건별 평균 오차를 한 표로 모은다. n 이 작은 조건은 참고용이므로 함께 싣는다."""
    rows = []

    def add(group, label, s):
        g = R.groupby(s, observed=True)
        d = g.agg(n=('abs_err', 'size'), MAE=('abs_err', 'mean'),
                  평균오차=('err', 'mean'), 실제평균=('y', 'mean'))
        for k, v in d.iterrows():
            rows.append({'구분': group, '조건': f'{label}={k}', **v.round(2).to_dict()})

    add('시각', 'h', R.hour)
    add('요일', 'dow', R.dow.map({1: '월', 2: '화', 3: '수', 4: '목', 5: '금', 6: '토', 7: '일'}))
    add('교대 전환 시각', '7·12·17시', R.is_transition.map({1: '해당', 0: '그 외'}))
    add('휴무 여부', 'is_off', R.is_off.map({1: '휴무', 0: '가동'}))
    add('휴무 직후', 'after_off', R.after_off.map({1: '휴무 다음날', 0: '그 외'}))
    add('직전 연속휴무', 'off_run_prev', pd.cut(R.off_run_prev, [-1, 0, 1, 2, 99],
                                                labels=['0일', '1일', '2일', '3일 이상']))
    add('생산량 구간', 'prod', pd.cut(R['prod'], [-1, 0, 300, 900, 99999],
                                      labels=['0', '1-300', '300-900', '900+']))
    add('생산0 고부하', 'prod=0 & 실제>100', ((R['prod'] == 0) & (R.y > 100)).map({True: '해당', False: '그 외'}))
    add('기온 구간', 'temp', pd.cut(R.temp, [-20, 5, 15, 22, 26, 40],
                                    labels=['~5', '5-15', '15-22', '22-26', '26+']))
    add('피크 여부', 'peak', R.peak.map({True: '피크(상위5%)', False: '비피크'}))
    add('생산량 급변', '|전시간 대비|', pd.cut(R.prod_diff.abs(), [-1, 100, 500, 1500, 99999],
                                               labels=['0-100', '100-500', '500-1500', '1500+']))
    return pd.DataFrame(rows)


def shap_importance(X, cols=None, seed=0):
    """전체 학습 구간으로 한 번 학습해 SHAP 기여도를 본다 (전역 영향요인용)."""
    cols = cols or FINAL['cols']
    tr = X[X[FINAL['train_flag']]]
    w = np.where(tr.is_clone, FINAL['clone_w'], 1.0)
    g = make_model(FINAL['model'], seed).fit(tr[cols], tr.target, sample_weight=w)
    sv = shap.TreeExplainer(g).shap_values(tr[cols])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    return (imp / imp.sum() * 100).round(2), sv, tr[cols]


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    R = predict_rows(X)
    R.to_csv(OUT_ROWS, index=False, encoding='utf-8-sig')
    print(f'예측 행 {len(R)}개 (고유일) · MAE {R.abs_err.mean():.2f} → {OUT_ROWS}\n')

    C = by_cond(R)
    C.to_csv(OUT_COND, index=False, encoding='utf-8-sig')

    print('=' * 72)
    print('[1] 오차가 큰 조건 상위 12 (n>=30 인 것만)')
    top = C[C.n >= 30].nlargest(12, 'MAE')
    print(top.to_string(index=False))

    print('\n' + '=' * 72)
    print('[2] 과소예측이 심한 조건 상위 8  (평균오차 양수 = 실제가 예측보다 높았다 = 피크를 놓침)')
    print(C[C.n >= 30].nlargest(8, '평균오차').to_string(index=False))

    print('\n' + '=' * 72)
    print('[3] 오차 상위 1% 행은 어떤 날인가')
    w = R.nlargest(max(1, len(R) // 100), 'abs_err')
    print(f'  기준 |오차| >= {w.abs_err.min():.1f} · {len(w)}행')
    print('  날짜별 건수 상위:', w['날짜'].value_counts().head(6).to_dict())
    print('  시각 분포 상위 :', w.hour.value_counts().head(6).to_dict())
    print(f'  이 행들의 평균오차 {w.err.mean():+.1f} (양수면 과소예측)')

    imp, sv, xtr = shap_importance(X)
    imp.to_csv(OUT_SHAP, header=['기여도(%)'], encoding='utf-8-sig')
    print('\n' + '=' * 72)
    print('[4] SHAP 영향요인 상위 12 (%)')
    print(imp.head(12).to_string())
    print(f'\n→ {OUT_COND}\n→ {OUT_SHAP}')
