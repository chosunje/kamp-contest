"""최종 후보 모델의 예측오차 다발 조건 분석.

기준 모델
- 피처: h7_full (1주 앞에서도 사용 가능한 피처)
- 학습: train_ok_strict
- 복제일: sample_weight=0.3
- 모델: LightGBM, model.py의 SEEDS 예측값 평균(간단 앙상블)
- 평가: model.py와 동일한 시간순 롤링 폴드 + 고유일 & train_ok

주의
- model_results.csv의 8번 수치(MAE 약 8.77)는 "시드별 지표의 평균"이다.
- 이 스크립트는 행별 예측값을 시드 평균한 뒤 오차를 계산하므로 MAE가 약 8.52로 더 낮다.
  즉 오류분석에는 실제 앙상블 예측을 사용하며, 둘은 같은 계산이 아니다.

출력
- outputs/error_predictions.csv       행 단위 OOF 예측/오차
- outputs/error_by_condition.csv     조건별 오차 요약
- outputs/error_by_date.csv          날짜별 오차 요약
- outputs/error_top_cases.csv        절대오차 상위 사례
- outputs/error_summary.txt          보고서용 핵심 요약

실행:
    python src/error_analysis.py
"""
import numpy as np
import pandas as pd

from features import build, FEATURES
from model import FOLDS, SEEDS, make_model
from preprocess import ROOT

OUT_PRED = ROOT / 'outputs' / 'error_predictions.csv'
OUT_COND = ROOT / 'outputs' / 'error_by_condition.csv'
OUT_DATE = ROOT / 'outputs' / 'error_by_date.csv'
OUT_TOP = ROOT / 'outputs' / 'error_top_cases.csv'
OUT_SUMMARY = ROOT / 'outputs' / 'error_summary.txt'

COLS = FEATURES['h7_full']
MODEL = 'lgbm'
TRAIN_FLAG = 'train_ok_strict'
CLONE_W = 0.3


def add_plan_window_features(X: pd.DataFrame) -> pd.DataFrame:
    """생산계획만으로 알 수 있는 하루의 생산 시작/종료 위치를 분석용으로 붙인다.

    모델 입력에는 넣지 않는다. 실제 실험에서 이 두 플래그를 h7_full에 직접 추가하면
    전체 MAE가 소폭 악화되어 채택하지 않았다. 오류 조건 분석과 별도 후처리 실험에만 사용한다.
    """
    X = X.copy()
    positive = X['prod'] > 0
    first = X['hour'].where(positive).groupby(X['날짜']).transform('min')
    last = X['hour'].where(positive).groupby(X['날짜']).transform('max')
    has_plan = X['day_prod'] > 0
    X['before_prod_start'] = (has_plan & (X['hour'] < first)).astype(int)
    X['after_prod_end'] = (has_plan & (X['hour'] > last)).astype(int)
    return X


def rolling_predictions(X: pd.DataFrame) -> pd.DataFrame:
    """model.py와 같은 평가셋에서 시드 평균 OOF 예측을 만든다."""
    rows = []
    total = len(FOLDS)
    for idx, fold in enumerate(FOLDS, 1):
        va = X[(X['ym'] == fold) & X.train_ok & ~X.is_clone].copy()
        if va.empty:
            print(f'  [{idx}/{total}] {fold}: 평가 행이 없어 건너뜀', flush=True)
            continue
        tr = X[(X['dt'] < va['dt'].min()) & X[TRAIN_FLAG]].copy()
        if len(tr) < 200:
            print(f'  [{idx}/{total}] {fold}: 학습 행 {len(tr):,}개로 부족해 건너뜀', flush=True)
            continue

        print(
            f'  [{idx}/{total}] {fold}: 학습 {len(tr):,}행 / 평가 {len(va):,}행 / '
            f'시드 {len(SEEDS)}개 학습 시작',
            flush=True,
        )
        w = np.where(tr.is_clone, CLONE_W, 1.0)
        preds = []
        for seed in SEEDS:
            g = make_model(MODEL, seed).fit(tr[COLS], tr.target, sample_weight=w)
            preds.append(g.predict(va[COLS]))
        p = np.mean(preds, axis=0)
        print(f'      {fold}: OOF 예측 완료', flush=True)
        peak_thr = tr.target.quantile(.95)

        keep = [
            'dt', '날짜', 'target', 'target_max15', 'hour', 'dow', 'month',
            'is_weekend', 'is_holiday', 'after_off', 'off_run_prev',
            'days_since_off', 'shift', 'is_transition', 'is_off',
            'prod', 'prod_cap', 'prod_zero', 'day_prod', 'day_prod_hours',
            'before_prod_start', 'after_prod_end',
            'temp', 'cool', 'humid', 'wind', 'rain',
        ]
        z = va[keep].copy()
        z['fold'] = fold
        z['pred'] = p
        z['error'] = z.target - z.pred             # +면 과소예측
        z['abs_error'] = z.error.abs()
        z['sq_error'] = z.error ** 2
        z['underpredict'] = z.error > 0
        z['peak_threshold'] = peak_thr
        z['is_peak'] = z.target >= peak_thr
        rows.append(z)
    return pd.concat(rows, ignore_index=True)


def metrics(s: pd.DataFrame) -> dict:
    if len(s) == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, bias=np.nan,
                    p90_abs_error=np.nan, under_rate=np.nan,
                    peak_rate=np.nan, actual_mean=np.nan, pred_mean=np.nan)
    return {
        'n': len(s),
        'MAE': s.abs_error.mean(),
        'RMSE': np.sqrt(s.sq_error.mean()),
        'bias': s.error.mean(),
        'p90_abs_error': s.abs_error.quantile(.90),
        'under_rate': s.underpredict.mean(),
        'peak_rate': s.is_peak.mean(),
        'actual_mean': s.target.mean(),
        'pred_mean': s.pred.mean(),
    }


def add_group(out: list, P: pd.DataFrame, axis: str, series: pd.Series):
    tmp = P.copy()
    tmp['_group'] = series
    for name, s in tmp.groupby('_group', observed=True, dropna=False):
        out.append({'axis': axis, 'condition': str(name), **metrics(s)})


def condition_table(P: pd.DataFrame) -> pd.DataFrame:
    """보고서에서 바로 쓸 수 있도록 일반 그룹 + 핵심 가설 조건을 같은 표로 만든다."""
    out = []

    add_group(out, P, 'fold', P.fold)
    add_group(out, P, 'month', P.month.map(lambda x: f'{int(x)}월'))
    add_group(out, P, 'dow', P.dow.map({1:'월', 2:'화', 3:'수', 4:'목', 5:'금', 6:'토', 7:'일'}))
    add_group(out, P, 'hour', P.hour.map(lambda x: f'{int(x):02d}시'))
    add_group(out, P, 'shift', P['shift'].map({0:'야간', 1:'점심', 2:'주간'}))
    add_group(out, P, 'transition', P.is_transition.map({0:'일반시각', 1:'전환시각(7·12·17시)'}))
    add_group(out, P, 'after_off', P.after_off.map({0:'일반일', 1:'휴무 직후'}))
    add_group(out, P, 'holiday', P.is_holiday.map({0:'비공휴일', 1:'법정공휴일'}))
    add_group(out, P, 'after_prod_end', P.after_prod_end.map({0:'생산종료 이전/해당없음', 1:'마지막 생산시간 이후'}))

    prod_band = pd.cut(
        P['prod'], [-np.inf, 0, 300, 900, np.inf],
        labels=['생산 0', '저생산(1~300)', '중생산(301~900)', '고생산(900+)'],
        include_lowest=True,
    )
    add_group(out, P, 'production_band', prod_band)

    temp_band = pd.cut(
        P['temp'], [-np.inf, 22, 26, np.inf],
        labels=['22℃ 이하', '22~26℃', '26℃ 초과'],
        include_lowest=True,
    )
    add_group(out, P, 'temperature_band', temp_band)

    off_band = pd.cut(
        P['off_run_prev'], [-np.inf, 0, 1, 3, np.inf],
        labels=['직전휴무 없음', '1일 휴무 후', '2~3일 휴무 후', '4일+ 휴무 후'],
        include_lowest=True,
    )
    add_group(out, P, 'previous_off_run', off_band)

    special = {
        '전체': np.ones(len(P), dtype=bool),
        '피크 구간(학습구간 상위5% 기준)': P.is_peak,
        '생산0 & 실제전력>100': (P['prod'] == 0) & (P.target > 100),
        '휴무 직후': P.after_off == 1,
        '전환시각(7·12·17시)': P.is_transition == 1,
        '고온 가동(기온>22℃ & 생산>0)': (P.temp > 22) & (P['prod'] > 0),
        '고생산(900+)': P['prod'] > 900,
        '주말': P.is_weekend == 1,
        '법정공휴일': P.is_holiday == 1,
        '마지막 생산시간 이후': P.after_prod_end == 1,
        '공휴일 & 생산종료 이후': (P.is_holiday == 1) & (P.after_prod_end == 1),
    }
    for name, mask in special.items():
        out.append({'axis': 'special', 'condition': name, **metrics(P[mask])})

    R = pd.DataFrame(out)
    R['n'] = R['n'].astype(int)
    return R


def date_table(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, s in P.groupby('날짜'):
        row = {
            '날짜': d,
            'fold': s.fold.iloc[0],
            'dow': int(s.dow.iloc[0]),
            'is_holiday': int(s.is_holiday.iloc[0]),
            'day_prod': float(s.day_prod.iloc[0]),
            'after_prod_end_hours': int(s.after_prod_end.sum()),
            **metrics(s),
        }
        rows.append(row)
    R = pd.DataFrame(rows).sort_values('MAE', ascending=False)
    R['n'] = R['n'].astype(int)
    return R


def main():
    print('[예측오차 분석] 피처 생성 시작', flush=True)
    X = add_plan_window_features(build())
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    print(f'[예측오차 분석] 피처 생성 완료: {len(X):,}행 / OOF 예측 시작', flush=True)
    P = rolling_predictions(X)
    print(f'[예측오차 분석] OOF 예측 완료: {len(P):,}행 / 조건별 통계 집계 시작', flush=True)

    q90 = P.abs_error.quantile(.90)
    P['large_error'] = P.abs_error >= q90

    C = condition_table(P)
    D = date_table(P)
    top_cols = [
        'dt', '날짜', 'fold', 'hour', 'dow', 'target', 'pred', 'error', 'abs_error',
        'is_peak', 'prod', 'day_prod', 'temp', 'is_holiday', 'after_off', 'off_run_prev',
        'is_transition', 'after_prod_end', 'is_off',
    ]
    T = P.nlargest(100, 'abs_error')[top_cols].copy()

    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT_PRED, index=False, encoding='utf-8-sig')
    C.to_csv(OUT_COND, index=False, encoding='utf-8-sig')
    D.to_csv(OUT_DATE, index=False, encoding='utf-8-sig')
    T.to_csv(OUT_TOP, index=False, encoding='utf-8-sig')

    overall = metrics(P)
    special = C[C.axis == 'special'].set_index('condition')
    worst_hours = (C[C.axis == 'hour'].sort_values('MAE', ascending=False).head(5)
                   [['condition', 'n', 'MAE', 'bias']])
    worst_folds = (C[C.axis == 'fold'].sort_values('MAE', ascending=False)
                   [['condition', 'n', 'MAE', 'RMSE', 'bias']])
    worst_dates = D.head(8)[['날짜', 'fold', 'is_holiday', 'day_prod', 'MAE', 'RMSE', 'bias']]

    lines = [
        '==============================================================',
        ' 최종 후보 모델 예측오차 분석',
        '==============================================================',
        f'모델: LightGBM / h7_full / {TRAIN_FLAG} / 복제일 가중치 {CLONE_W}',
        f'시드: {list(SEEDS)} 예측값 평균 / 평가: 고유일 & train_ok / 폴드: {FOLDS}',
        '',
        f'[전체] n={overall["n"]:,}  MAE={overall["MAE"]:.2f}  RMSE={overall["RMSE"]:.2f}  '
        f'bias={overall["bias"]:+.2f}  |오차|90%={overall["p90_abs_error"]:.2f}',
        f'과소예측 비율={overall["under_rate"]:.1%}  피크 비율={overall["peak_rate"]:.1%}',
        '',
        '[핵심 조건]',
    ]
    for name in special.index:
        r = special.loc[name]
        lines.append(
            f'- {name}: n={int(r.n):,}, MAE={r.MAE:.2f}, RMSE={r.RMSE:.2f}, '
            f'bias={r.bias:+.2f}, 과소예측={r.under_rate:.1%}'
        )
    lines += ['', '[폴드별 오차]']
    for _, r in worst_folds.iterrows():
        lines.append(
            f'- {r.condition}: n={int(r.n):,}, MAE={r.MAE:.2f}, RMSE={r.RMSE:.2f}, bias={r.bias:+.2f}'
        )
    lines += ['', '[MAE가 큰 시간대 상위 5개]']
    for _, r in worst_hours.iterrows():
        lines.append(f'- {r.condition}: n={int(r.n):,}, MAE={r.MAE:.2f}, bias={r.bias:+.2f}')
    lines += ['', '[MAE가 큰 날짜 상위 8일]']
    for _, r in worst_dates.iterrows():
        lines.append(
            f'- {int(r["날짜"])} ({r.fold}, 공휴일={int(r.is_holiday)}): '
            f'일생산={r.day_prod:.0f}, MAE={r.MAE:.2f}, RMSE={r.RMSE:.2f}, bias={r.bias:+.2f}'
        )
    lines += [
        '',
        f'[대오차 기준] 전체 |오차| 상위 10% 임계값 = {q90:.2f}',
        f'상위 100건 상세 → {OUT_TOP.name}',
        f'행 단위 예측 → {OUT_PRED.name}',
        f'조건별 표 → {OUT_COND.name}',
        f'날짜별 표 → {OUT_DATE.name}',
    ]
    OUT_SUMMARY.write_text('\n'.join(lines), encoding='utf-8')

    print('[예측오차 분석] 조건별 통계 및 파일 저장 완료', flush=True)
    print('\n'.join(lines), flush=True)
    for p in [OUT_PRED, OUT_COND, OUT_DATE, OUT_TOP, OUT_SUMMARY]:
        print(f'→ {p}')


if __name__ == '__main__':
    main()
