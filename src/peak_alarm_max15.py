"""피크 경보 정책 검증. 실행: python src/peak_alarm_max15.py

15분 최대 직접 L1/P90을 동일 조건으로 비교한다. 실제 상대 피크는 학습 Q95로
고정하고 예측 경보 임계만 Q95/Q97/Q98로 비교한다. 프로젝트 기본은 사전에 Q95로
정했으며 이 민감도 표로 비용 최적 임계값이나 계약전력 초과 성능을 주장하지 않는다.
학습은 폴드 첫 평가일 00시 이전, 입력은 매일 00시 이전 전력으로 갱신한다.
당일 실측 is_off 제외, 학습 시점 이전 곡선만으로 복제 가중치를 판정한다.
"""
import json
import platform
from time import perf_counter

import lightgbm
import numpy as np
import pandas as pd

import alarm_policy as policy
from features import build, Q15
from model import CORE, CORE_FOLDS, FOLDS, SEEDS
from preprocess import load, ROOT, RAW
from quantile_analysis import historical_clone_weights, result_table

OUT = ROOT / 'outputs'
QS = (.95, .97, .98)
LABELS = ('L1', 'P90')
POWER_COLUMNS = ['평균', *Q15]


def compare(X):
    """시드별·앙상블 OOF와 학습 임계값, 감사용 학습 모델을 반환한다."""
    frames, thresholds, fitted = [], [], {}
    for fold in FOLDS:
        va = X[(X.ym == fold) & X.train_ok & ~X.is_clone]
        if va.empty:
            continue
        train_cutoff = va.dt.min().normalize()
        meta = None
        for label in LABELS:
            print(f'[{fold} {label}] 학습 마감 {train_cutoff}', flush=True)
            models, current = policy.fit_models(X, train_cutoff, label=label)
            fitted[(fold, label)] = models
            if meta is not None:
                for key in ('train_n', 'train_cutoff', 'threshold_q95',
                            'threshold_q97', 'threshold_q98'):
                    assert meta[key] == current[key], f'L1/P90 조건 불일치: {key}'
            meta = current
            for date, day in va.groupby('날짜', sort=True):
                prediction_cutoff = day.dt.min().normalize()
                assert train_cutoff <= prediction_cutoff
                inputs = policy.available_inputs(X, prediction_cutoff).loc[day.index, policy.ALARM_COLS]
                ps = [model.predict(inputs) for model in models]
                for seed, prediction in [*zip(map(str, SEEDS), ps),
                                         ('ensemble', np.mean(ps, axis=0))]:
                    z = day[['dt', '날짜', 'hour', 'is_off']].copy()
                    z.insert(0, 'idx', day.index)
                    z['target'] = day[policy.ALARM_TARGET].to_numpy()
                    z['pred'] = prediction
                    z['fold'], z['protocol'], z['model'], z['seed'] = fold, 'd1_max15', label, seed
                    z['train_cutoff'], z['prediction_cutoff'] = train_cutoff, prediction_cutoff
                    z['peak_threshold'] = current['threshold_q95']
                    z['is_peak'] = z.target >= z.peak_threshold
                    frames.append(z.reset_index(drop=True))
        thresholds.append({'fold': fold, **{key: meta[key] for key in (
            'train_cutoff', 'train_n', 'threshold_q95', 'threshold_q97', 'threshold_q98')}})
    if not frames:
        raise ValueError('경보 정책을 검증할 평가 행이 없다.')
    P = pd.concat(frames, ignore_index=True)
    if not np.isfinite(P.pred).all() or P.duplicated(['idx', 'model', 'seed']).any():
        raise ValueError('비유한 예측 또는 중복 예측 키가 있다.')
    keys = ['idx', 'target', 'fold', 'train_cutoff', 'prediction_cutoff', 'peak_threshold']
    groups = list(P.groupby(['model', 'seed'], sort=False))
    reference = groups[0][1].sort_values('idx')[keys].reset_index(drop=True)
    for _, rows in groups[1:]:
        pd.testing.assert_frame_equal(reference, rows.sort_values('idx')[keys].reset_index(drop=True))
    return P, pd.DataFrame(thresholds), fitted


def _percent(a, b):
    return float(a / b * 100) if b else np.nan


def alarm_metrics(frame):
    """시간별 유효행, 일별 모든 완전일(휴무 포함)에서 경보 지표를 계산한다."""
    pred, real = frame.alarm.astype(bool), frame.real_peak.astype(bool)
    tp, fp, fn = int((pred & real).sum()), int((pred & ~real).sum()), int((~pred & real).sum())
    group = frame.groupby('날짜', sort=True)
    days = group.agg(n=('alarm', 'size'), alarm=('alarm', 'any'), real_peak=('real_peak', 'any'),
                     actual_max15=('target', 'max'), predicted_max15=('pred', 'max'),
                     real_threshold=('peak_threshold', 'first'), alarm_threshold=('alarm_threshold', 'first'))
    excluded = int((days.n != 24).sum())
    days = days[days.n == 24].copy()
    dtp = int((days.alarm & days.real_peak).sum())
    dfp = int((days.alarm & ~days.real_peak).sum())
    dfn = int((~days.alarm & days.real_peak).sum())
    values = {
        'hour_n': len(frame), 'hour_tp': tp, 'hour_fp': fp, 'hour_fn': fn,
        'hour_precision_pct': _percent(tp, tp + fp), 'hour_recall_pct': _percent(tp, tp + fn),
        'hour_alarm_rate_pct': _percent(int(pred.sum()), len(frame)),
        'hour_coverage_pct': _percent(int((frame.target <= frame.pred).sum()), len(frame)),
        'complete_days': len(days), 'incomplete_days_excluded': excluded,
        'day_tp': dtp, 'day_fp': dfp, 'day_fn': dfn,
        'day_precision_pct': _percent(dtp, dtp + dfp), 'day_recall_pct': _percent(dtp, dtp + dfn),
        'day_alarm_rate_pct': _percent(int(days.alarm.sum()), len(days)),
        'day_actual_peak_days': int(days.real_peak.sum()),
    }
    return values, days.reset_index()


def evaluate(P, T):
    """실제 피크 Q95를 고정하고 예측 경보만 높인 민감도 표를 만든다."""
    rows, daily = [], []
    for label in LABELS:
        predictions = P[(P.model == label) & (P.seed == 'ensemble')]
        for q in QS:
            frame = predictions.copy()
            frame['alarm_threshold'] = frame.fold.map(T.set_index('fold')[f'threshold_q{round(q * 100)}'])
            frame['alarm'] = frame.pred >= frame.alarm_threshold
            frame['real_peak'] = frame.target >= frame.peak_threshold
            periods = [('ALL', frame), (CORE, frame[frame.fold.isin(CORE_FOLDS)])]
            periods.extend(frame.groupby('fold', sort=False))
            for period, part in periods:
                if part.empty:
                    continue
                values, days = alarm_metrics(part)
                rows.append({'model': label, 'alarm_q': q, 'real_peak_q': .95,
                             'period': period, **values})
                if period == 'ALL':
                    days['model'], days['alarm_q'], days['real_peak_q'] = label, q, .95
                    days['fold'] = days['날짜'].map(frame.groupby('날짜').fold.first())
                    daily.append(days)
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def audit_future_power(raw, X, fitted, T):
    """모든 평가일의 미래 전력 삭제·변경 후 입력, 예측, 경보가 불변인지 검사한다."""
    rows = []
    evaluation = X[X.ym.isin(FOLDS) & X.train_ok & ~X.is_clone]
    thresholds = T.set_index('fold')
    for date, day in evaluation.groupby('날짜', sort=True):
        fold = day.ym.iloc[0]
        cutoff = day.dt.min().normalize()
        idx = X.index[X['날짜'] == date]
        train_cutoff = thresholds.loc[fold, 'train_cutoff']
        history = X[X.dt < train_cutoff]
        tr = history[history.train_ok_strict & history[policy.ALARM_TARGET].notna()]
        for mode, value in [('deleted', np.nan), ('changed', 10000.)]:
            changed = raw.copy(deep=True)
            changed.loc[changed['날짜'] >= date, POWER_COLUMNS] = value
            altered = build(load(changed))
            before_inputs = policy.available_inputs(X, cutoff).loc[idx, policy.ALARM_COLS]
            after_inputs = policy.available_inputs(altered, cutoff).loc[idx, policy.ALARM_COLS]
            pd.testing.assert_frame_equal(before_inputs, after_inputs)
            changed_history = altered[altered.dt < train_cutoff]
            changed_tr = changed_history[changed_history.train_ok_strict & changed_history[policy.ALARM_TARGET].notna()]
            pd.testing.assert_index_equal(tr.index, changed_tr.index)
            pd.testing.assert_frame_equal(tr[policy.ALARM_COLS + [policy.ALARM_TARGET]],
                                          changed_tr[policy.ALARM_COLS + [policy.ALARM_TARGET]],
                                          check_dtype=False, check_exact=True)
            np.testing.assert_array_equal(historical_clone_weights(history, tr),
                                          historical_clone_weights(changed_history, changed_tr))
            thr = thresholds.loc[fold, 'threshold_q95']
            assert policy.threshold(changed_tr) == thr
            maximum = 0.
            for label in LABELS:
                before = policy.predict(fitted[(fold, label)], X, cutoff, idx)
                after = policy.predict(fitted[(fold, label)], altered, cutoff, idx)
                np.testing.assert_array_equal(before, after)
                np.testing.assert_array_equal(policy.alarm_flags(before, thr), policy.alarm_flags(after, thr))
                maximum = max(maximum, float(np.abs(before - after).max()))
            rows.append({'날짜': date, 'prediction_cutoff': cutoff, 'mode': mode,
                         'feature_equal': True, 'prediction_max_delta': maximum,
                         'alarm_equal': True, 'train_and_threshold_equal': True})
            if len(rows) % 40 == 0:
                print(f'  미래 전력 변경 검증 {len(rows)}건 통과', flush=True)
    if not rows:
        raise ValueError('미래 전력 변경 검증 대상이 없다.')
    return pd.DataFrame(rows)


def summary_text(E, T, A):
    lines = [
        '기본 피크 경보 정책과 15분 최대 검증',
        f'정책: {policy.POLICY_ID}',
        '시간 경보: 15분 최대 직접 학습 P90 예측 >= 학습 target_max15 Q95.',
        '일 경보: 하루에 시간 경보가 하나라도 있으면 위험일.',
        'Q95는 strict·타깃 유효 학습행(복제 포함)의 비가중 분위값. 숫자는 학습 기간마다 갱신.',
        'Q97/Q98 비교는 실제 피크 Q95를 고정하고 예측 경보 임계만 올린 민감도 분석.',
        '계약전력·전력 단위 확인 자료가 없어 계약전력 초과나 요금 발생 판정으로 사용하지 않는다.',
        '15분 최대: 원본 네 15분 값의 최대. 생산 실적·기상 실측은 계획·예보로 가정.',
        '학습: h7_lag - is_off 33피처, shallow, 복제 가중치 0.3(과거 곡선 판정), 시드 0/1/2.',
        '검증: 폴드 첫 평가일 00시 학습 고정, 매일 00시 이전 전력으로 입력 갱신(D-1).',
        '일 지표는 휴무 포함 완전 24행 평가일, 불완전일 제외. 실제 휴무로 경보를 억제하지 않는다.',
        'P90은 포함률 미보정 후보이며 90% 확률·일 최대 상한을 보장하지 않는다.',
        '', '[CORE(7 to 9월) 경보 임계 민감도 — 시드 예측 평균]',
    ]
    for _, row in E[E.period == CORE].iterrows():
        lines.append(f'{row.model} 경보 Q{row.alarm_q * 100:.0f}: '
                     f'시간 정밀도 {row.hour_precision_pct:.1f}% / 재현율 {row.hour_recall_pct:.1f}% / '
                     f'경보율 {row.hour_alarm_rate_pct:.1f}% / 관측 포함률 {row.hour_coverage_pct:.1f}%')
        lines.append(f'  일 정밀도 {row.day_precision_pct:.1f}% / 재현율 {row.day_recall_pct:.1f}% / '
                     f'경보율 {row.day_alarm_rate_pct:.1f}% / 완전일 {int(row.complete_days)} / '
                     f'불완전일 제외 {int(row.incomplete_days_excluded)}')
    lines.extend(['', '[학습 폴드별 임계값]', T.to_string(index=False), '', '[미래 전력 변경 검증]',
                  f'{A["날짜"].nunique()}일 × 삭제/변경 = {len(A)}건 통과. '
                  f'예측 최대 변화 {A.prediction_max_delta.max():.1f}, 경보 불변.',
                  '', '[적용 범위]',
                  '프로젝트 기본 경보 정책은 Q95로 확정했다. 비용 최적화 또는 현장 채택 성능 보장은 아니다.',
                  'CORE를 이미 여러 설정 선택에 사용했으며 독립 홀드아웃은 없다.',
                  '고유일 판별·품질 제외는 사후 평가용이다. 실제 미래 가동 상태를 확정하지 않는다.',
                  '위 성능은 D-1·생산계획 제공 조건이다. D-7·no-plan 성능으로 확대하지 않는다.',
                  '기존 hourly 경보 표, D-1 h7_full L1/P90, FINAL_CFG 가중치 표와 학습·타깃 조건이 다르다.'])
    return '\n'.join(lines) + '\n'


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    started = perf_counter()
    raw = pd.read_csv(RAW)
    X = build(load(raw))
    X['ym'] = X.dt.dt.to_period('M').astype(str)
    P, T, fitted = compare(X)
    R = result_table(P)
    E, D = evaluate(P, T)
    print('[감사] 모든 평가일 이후 전력 삭제/변경', flush=True)
    A = audit_future_power(raw, X, fitted, T)
    snapshot = {
        'policy_id': policy.POLICY_ID, 'status': 'project_default', 'target': policy.ALARM_TARGET,
        'interpretation': 'D20_A', 'model_alpha': policy.ALPHA, 'alarm_quantile': policy.ALARM_Q,
        'comparison': '>=', 'day_rule': 'any_hour_alarm', 'quantile_weighted': False,
        'absolute_contract_limit': None, 'power_unit': 'same_as_source_columns',
        'model_config': policy.make_config(), 'model_params': fitted[(FOLDS[-1], 'P90')][0].get_params(),
        'seeds': list(SEEDS), 'folds': list(FOLDS), 'validation_protocol': 'd1_max15',
        'thresholds': T.to_dict(orient='records'),
        'sensitivity_truth': 'fixed_training_Q95',
        'selection_rule': 'predeclared_Q95_default; Q97/Q98_are_sensitivity_only',
        'core_default_metrics': E[(E.period == CORE) & (E.model == 'P90') & (E.alarm_q == policy.ALARM_Q)].iloc[0].to_dict(),
        'audit_cases': len(A), 'elapsed_seconds': perf_counter() - started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'lightgbm': lightgbm.__version__},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('predictions', P), ('model_results', R), ('eval', E),
                        ('days', D), ('thresholds', T), ('leakage_audit', A)]:
        frame.to_csv(OUT / f'peak_alarm_max15_{name}.csv', index=False, encoding='utf-8-sig')
    (OUT / 'alarm_policy.json').write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + '\n',
        encoding='utf-8')
    summary = summary_text(E, T, A)
    (OUT / 'peak_alarm_max15_summary.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)
    print(f'완료: {perf_counter() - started:.1f}초', flush=True)


if __name__ == '__main__':
    main()
