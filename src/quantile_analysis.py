"""L1 점 예측과 분위수 0.9 비교. 실행: python src/quantile_analysis.py

reference: 기존 h7_full·복제 가중치 0.3 실험 재현. 당일 실제 전력에서 만든
           is_off가 입력에 있어 미래 예측 성능으로 해석하지 않는다.
d1: h7_full에서 is_off 제외. 폴드 첫 평가일 0시에 모델을 학습한 뒤 고정하고,
    각 예측일 0시 이전 전력 실적으로 lag를 갱신하는 D-1 백테스트.
    학습 복제 가중치는 학습 시점 이전 전력 곡선으로만 판정한다.

두 프로토콜 안에서는 학습·평가 행, 피처, 가중치, 시드가 동일하다.
기상 실측·생산 실적을 예보·계획으로 가정한다. 독립 홀드아웃은 없다.
P90은 초과확률 90%가 아니며 시간별 P90의 최대도 일 최대의 90% 상한이 아니다.
"""
import sys
import lightgbm
import numpy as np
import pandas as pd

from features import build, FEATURES, Q15
from model import make_model, FOLDS, CORE_FOLDS, CORE, SEEDS, CLONE_W
from preprocess import load, ROOT, RAW

OUT = ROOT / 'outputs'
ALPHA = .9
D1_COLS = [c for c in FEATURES['h7_full'] if c != 'is_off']
PROTOCOLS = {'reference': FEATURES['h7_full'], 'd1': D1_COLS}
MODELS = {'L1': 'lgbm', 'P90': 'lgbm_q90'}
POWER_COLUMNS = ['평균', *Q15]


def historical_clone_weights(history, tr):
    """학습 시점 이전의 완전한 일별 곡선으로 복제를 판정한다."""
    curves = history.groupby('날짜').target.apply(tuple)
    counts = curves.map(curves.value_counts())
    clone = tr['날짜'].map(counts).gt(1)
    return np.where(clone, CLONE_W, 1.0)


def compare(X):
    """동일 평가 행에 두 손실함수의 시드별·앙상블 예측을 생성한다."""
    frames, fitted = [], {}
    for fold in FOLDS:
        va = X[(X.ym == fold) & X.train_ok & ~X.is_clone]
        if va.empty:
            continue
        for protocol, cols in PROTOCOLS.items():
            train_cutoff = (va.dt.min().normalize() if protocol == 'd1' else va.dt.min())
            history = X[X.dt < train_cutoff]
            tr = history[history.train_ok_strict]
            if len(tr) < 200:
                continue
            threshold = tr.target.quantile(.95)
            w = (historical_clone_weights(history, tr) if protocol == 'd1'
                 else np.where(tr.is_clone, CLONE_W, 1.0))
            print(f'[{protocol} {fold}] 학습 {len(tr):,} / 평가 {len(va):,}', flush=True)
            for label, name in MODELS.items():
                ps = []
                for seed in SEEDS:
                    model = make_model(name, seed).fit(tr[cols], tr.target, sample_weight=w)
                    p = model.predict(va[cols])
                    ps.append(p)
                    fitted[(protocol, fold, label, seed)] = model
                    frames.append(prediction_frame(va, protocol, fold, label, str(seed),
                                                   p, train_cutoff, threshold))
                frames.append(prediction_frame(va, protocol, fold, label, 'ensemble',
                                               np.mean(ps, axis=0), train_cutoff, threshold))
    if not frames:
        raise ValueError('비교 가능한 평가 행이 없다.')
    P = pd.concat(frames, ignore_index=True)
    d1 = P[P.protocol == 'd1']
    assert (d1.train_cutoff <= d1.prediction_cutoff).all(), 'D-1 학습 시점이 예측 시점보다 늦다'
    # 모든 비교군의 정답·평가시각·피크 임계값이 동일한지 검산한다.
    for _, group in P.groupby(['protocol', 'fold']):
        groups = list(group.groupby(['model', 'seed']))
        keys = ['idx', 'target', 'peak_threshold', 'train_cutoff', 'prediction_cutoff']
        base = groups[0][1][keys].reset_index(drop=True)
        for _, rows in groups[1:]:
            pd.testing.assert_frame_equal(base, rows[keys].reset_index(drop=True))
    return P, fitted


def prediction_frame(va, protocol, fold, label, seed, p, train_cutoff, threshold):
    z = va[['dt', '날짜', 'hour', 'target', 'is_off']].copy()
    z.insert(0, 'idx', va.index)
    z['protocol'], z['fold'], z['model'], z['seed'] = protocol, fold, label, seed
    z['train_cutoff'] = train_cutoff
    z['prediction_cutoff'] = z.dt.dt.normalize() if protocol == 'd1' else pd.NaT
    z['pred'] = p
    z['peak_threshold'] = threshold
    z['is_peak'] = z.target >= threshold
    return z.reset_index(drop=True)


def metrics(s):
    e = s.target - s.pred
    peak = e[s.is_peak]
    # 일 최대는 가동일 중 평가 행이 24개인 날만 사용한다.
    active = s[~s.is_off.astype(bool)]
    complete = active.groupby('날짜').size().eq(24)
    daily = active[active['날짜'].isin(complete[complete].index)].groupby('날짜')
    dm = daily.target.max() - daily.pred.max()
    return {
        'n': len(s), 'MAE': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()),
        'bias': e.mean(), 'peak_n': len(peak), 'peakMAE': peak.abs().mean(),
        'peak_bias': peak.mean(), 'peak_under_pct': peak.gt(0).mean() * 100,
        'coverage_pct': e.le(0).mean() * 100,
        'pinball_90': np.maximum(ALPHA * e, (ALPHA - 1) * e).mean(),
        'active_complete_days': len(dm), 'daily_max_MAE': dm.abs().mean(),
        'daily_max_bias': dm.mean(),
        'incomplete_active_days': int((~complete).sum()),
    }


def result_table(P):
    rows = []
    for (protocol, label, seed), group in P.groupby(['protocol', 'model', 'seed'], sort=False):
        periods = [('ALL', group), (CORE, group[group.fold.isin(CORE_FOLDS)])]
        periods.extend(group.groupby('fold', sort=False))
        for period, s in periods:
            if len(s):
                rows.append({'protocol': protocol, 'model': label, 'seed': seed,
                             'period': period, **metrics(s)})
    R = pd.DataFrame(rows)
    # 시드별 지표의 평균과 예측을 먼저 평균한 앙상블 지표를 별도 행으로 보존한다.
    single = R[R.seed != 'ensemble']
    keys = ['protocol', 'model', 'period']
    avg = single.groupby(keys, sort=False).mean(numeric_only=True).reset_index()
    avg['seed'] = 'seed_metric_mean'
    avg['MAE_seed_std'] = single.groupby(keys, sort=False).MAE.std().to_numpy()
    return pd.concat([R, avg], ignore_index=True)


def audit_future_power(raw, X, fitted):
    """모든 평가일에서 cutoff 이후 전력 삭제/변경에 대한 D-1 불변성을 검사한다.

    정답·평가행은 원본 X로 고정한다. 변형 데이터의 정답 유래 메타데이터로
    평가 대상을 다시 고르면 미래 전력을 삭제했을 때 평가셋 자체가 바뀐다.
    """
    rows = []
    eval_rows = X[X.ym.isin(FOLDS) & X.train_ok & ~X.is_clone]
    for date, group in eval_rows.groupby('날짜', sort=True):
        fold = group.ym.iloc[0]
        if ('d1', fold, 'L1', SEEDS[0]) not in fitted:
            continue
        cutoff = group.dt.min().normalize()
        idx = X.index[X['날짜'] == date]
        past_idx = X.index[X.dt < cutoff]
        for mode, value in [('deleted', np.nan), ('changed', 10000.0)]:
            changed = raw.copy(deep=True)
            changed.loc[changed['날짜'] >= date, POWER_COLUMNS] = value
            altered = build(load(changed))
            pd.testing.assert_frame_equal(X.loc[idx, D1_COLS], altered.loc[idx, D1_COLS])
            pd.testing.assert_frame_equal(X.loc[past_idx, D1_COLS],
                                          altered.loc[past_idx, D1_COLS])
            prediction_delta = 0.0
            for label in MODELS:
                models = [fitted[('d1', fold, label, seed)] for seed in SEEDS]
                before = np.mean([m.predict(X.loc[idx, D1_COLS]) for m in models], axis=0)
                after = np.mean([m.predict(altered.loc[idx, D1_COLS]) for m in models], axis=0)
                np.testing.assert_allclose(before, after, rtol=0, atol=0)
                prediction_delta = max(prediction_delta, float(np.abs(before - after).max()))
            train_cutoff = X[(X.ym == fold) & X.train_ok & ~X.is_clone].dt.min().normalize()
            history, changed_history = X[X.dt < train_cutoff], altered[altered.dt < train_cutoff]
            tr = history[history.train_ok_strict]
            changed_tr = changed_history[changed_history.train_ok_strict]
            pd.testing.assert_index_equal(tr.index, changed_tr.index)
            np.testing.assert_array_equal(historical_clone_weights(history, tr),
                                          historical_clone_weights(changed_history, changed_tr))
            # 기존 reference의 당일 실제 휴무 피처는 달라지는지 별도로 기록한다.
            reference_changed = not X.loc[idx, 'is_off'].equals(altered.loc[idx, 'is_off'])
            rows.append({'날짜': date, 'prediction_cutoff': cutoff, 'mode': mode,
                         'feature_equal': True, 'prediction_max_delta': prediction_delta,
                         'reference_is_off_changed': reference_changed})
    A = pd.DataFrame(rows)
    if A.empty:
        raise ValueError('미래 전력 변경 검증을 수행할 평가일이 없다.')
    return A


def summary_text(R, A):
    lines = [
        'L1 점 예측 vs 분위수 0.9 비교 (src/quantile_analysis.py 생성)',
        f'실행 환경: Python {sys.version.split()[0]}, numpy {np.__version__}, '
        f'pandas {pd.__version__}, lightgbm {lightgbm.__version__}',
        f'학습: strict / 복제 가중치 {CLONE_W} / 시드 {list(SEEDS)} / 같은 나무 설정',
        '평가: 고유일 & train_ok. 아래 수치는 시드 예측을 평균한 앙상블 기준.',
        'reference: 기존 h7_full 재현용. 당일 실제 is_off 포함 → 미래 운영 성능 아님.',
        'd1: is_off 제외, 매일 0시 이전 전력으로 입력 갱신. 폴드 첫 평가일 학습 모델 고정.',
        'd1의 학습 복제 가중치는 학습 시점 이전 곡선만으로 판정.',
        '일 최대 지표: 가동일 중 평가 행이 24개인 날만 사용.',
        '',
    ]
    for protocol in PROTOCOLS:
        lines.append(f'[{protocol}]')
        for period in (CORE, 'ALL', *FOLDS):
            s = R[(R.protocol == protocol) & (R.period == period) & (R.seed == 'ensemble')]
            if s.empty:
                continue
            lines.append(f'  {period}')
            for _, r in s.iterrows():
                lines.append(
                    f'    {r.model}: n={int(r.n):,} MAE={r.MAE:.2f} RMSE={r.RMSE:.2f} '
                    f'peakMAE={r.peakMAE:.2f} peak_bias={r.peak_bias:+.2f} '
                    f'피크과소={r.peak_under_pct:.1f}% 포함률={r.coverage_pct:.1f}% '
                    f'pinball90={r.pinball_90:.2f}')
                lines.append(
                    f'      일최대 MAE={r.daily_max_MAE:.2f} bias={r.daily_max_bias:+.2f} '
                    f'가동일={int(r.active_complete_days)} '
                    f'불완전가동일제외={int(r.incomplete_active_days)}')
        lines.append('')
    lines += [
        '[미래 전력 변경 검증]',
        f'  {A["날짜"].nunique()}일 × 삭제/변경 = {len(A)}건 통과.',
        f'  D-1 피처 동일, 예측 최대 변화 {A.prediction_max_delta.max():.1f}.',
        f'  reference 당일 is_off가 달라진 검증 {int(A.reference_is_off_changed.sum())}건.',
        '',
        '[해석 및 한계]',
        '  P90은 경보용 상한 후보이며 점 예측을 대체하지 않는다.',
        '  P90 포함률은 실제 관측 지표다. 90%가 보장되거나 초과확률 90%라는 뜻이 아니다.',
        '  시간별 P90 최대를 일 최대의 90% 상한으로 해석하지 않는다.',
        '  이 결과는 시간 평균 전력 기준이며 15분 최대 경보 검증이 아니다.',
        '  생산 실적·기상 실측을 계획·예보로 가정했다. 실제 예보 오차는 반영되지 않았다.',
        '  고유일 판별·품질 제외는 고정된 사후 평가 기준이며 운영 입력이 아니다.',
        '  독립 홀드아웃은 없다. D-7·월 전체 일괄예측 성능으로 해석하지 않는다.',
        '  reference와 d1은 입력·복제 판정 조건이 다르므로 프로토콜 안에서 모델을 비교한다.',
        '  시드별 지표 평균은 quantile_results.csv의 seed_metric_mean 행에 따로 저장한다.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    raw = pd.read_csv(RAW)
    X = build(load(raw))
    X['ym'] = X.dt.dt.to_period('M').astype(str)
    P, fitted = compare(X)
    R = result_table(P)
    print('[검증] 평가일 이후 전력 삭제/변경에 대한 피처·예측 불변성', flush=True)
    A = audit_future_power(raw, X, fitted)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('quantile_predictions.csv', P), ('quantile_results.csv', R),
                        ('quantile_leakage_audit.csv', A)]:
        frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    summary = summary_text(R, A)
    (OUT / 'quantile_summary.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
