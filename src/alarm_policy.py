"""상대 피크 경보 정책과 예측 시점에 이용 가능한 입력.

15분 컬럼의 최대(네 값을 15분 구간 평균으로 보는 해석)를 직접 학습한 P90을 학습 Q95와 비교한다.
계약전력 초과 규칙이나 보정된 90% 신뢰상한을 뜻하지 않는다.
당일 실제 is_off는 입력에서 제외하고, cutoff 이후 전력은 lag에 넣지 않는다.
생산계획과 기상예보가 제공된다는 가정은 기존 피처 생성과 동일하다.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from features import FEATURES
from model import ALARM_CFG, SEEDS, make_model
from quantile_analysis import historical_clone_weights


ALARM_TARGET = 'target_max15'
ALARM_Q = .95
ALPHA = .9
POLICY_ID = 'D10_max15_p90_q95'
ALARM_COLS = [c for c in ALARM_CFG['cols'] if c != 'is_off']
NO_PLAN_COLS = list(FEATURES['h7_no_plan'])


def _midnight(cutoff):
    value = pd.Timestamp(cutoff)
    if pd.isna(value) or value != value.normalize():
        raise ValueError('예측 cutoff는 유효한 날짜의 00시여야 한다.')
    return value


def available_inputs(X, cutoff):
    """cutoff 이전 전력만으로 lag를 다시 만들고 원본 인덱스를 보존한다.

cutoff 당일 휴무 이력은 전일까지의 관측으로 알 수 있다. 그 이후 날의
직전 휴무 이력은 아직 관측하지 않았으므로 생산계획으로 추정하지 않는다.
target와 품질 메타데이터는 정답/사후 평가용으로 보존되지만 입력에 쓰지 않는다.
"""
    cutoff = _midnight(cutoff)
    result = X.copy(deep=True)
    observed = X.dt < cutoff
    power = X.target.where(observed & ~X.is_off.astype(bool) & ~X.is_stop.astype(bool))
    result['lag168'] = power.shift(168)
    result['lag336'] = power.shift(336)
    result['lag_week_mean4'] = pd.concat(
        [power.shift(168 * k) for k in range(1, 5)], axis=1,
    ).mean(axis=1)
    daily = power.groupby(X['날짜']).agg(['mean', 'max'])
    result['last_week_day_mean'] = X['날짜'].map(daily['mean'].shift(7))
    result['last_week_day_max'] = X['날짜'].map(daily['max'].shift(7))
    result['lag168_na'] = result.lag168.isna().astype(int)
    result['lag_best'] = result.lag168.fillna(result.lag336).fillna(result.lag_week_mean4)
    future_days = X.dt.dt.normalize() > cutoff
    for col in ('after_off', 'off_run_prev', 'days_since_off'):
        result[col] = result[col].astype(float)
        result.loc[future_days, col] = np.nan
    return result


def make_config(no_plan=False, label='P90'):
    """L1/P90 비교에서는 목적함수만 바꾸고 나무 설정과 입력을 맞춘다."""
    if label not in {'L1', 'P90'}:
        raise ValueError(f'지원하지 않는 경보 모델: {label}')
    cfg = deepcopy(ALARM_CFG)
    cfg['cols'] = list(NO_PLAN_COLS if no_plan else ALARM_COLS)
    cfg['target'] = ALARM_TARGET
    cfg.pop('peak_w', None)
    cfg['params']['objective'] = 'l1' if label == 'L1' else 'quantile'
    if label == 'L1':
        cfg['params'].pop('alpha', None)
    else:
        cfg['params']['alpha'] = ALPHA
    return cfg


def threshold(tr, value=None, q=ALARM_Q):
    """명시값 또는 학습 max15의 비가중 분위값을 반환한다."""
    if value is not None:
        result = float(value)
    else:
        if not np.isfinite(q) or not 0 < q < 1:
            raise ValueError('상대 임계의 q는 0과 1 사이여야 한다.')
        result = float(tr[ALARM_TARGET].quantile(q))
    if not np.isfinite(result) or result <= 0:
        raise ValueError('경보 임계값은 유한한 양수여야 한다.')
    return result


def fit_models(X, cutoff, label='P90', no_plan=False, seeds=SEEDS):
    """cutoff 이전 strict 행으로 모델을 학습하고 임계 스냅샷을 반환한다."""
    cutoff = _midnight(cutoff)
    cfg = make_config(no_plan=no_plan, label=label)
    history = X[X.dt < cutoff]
    tr = history[history.train_ok_strict & history[ALARM_TARGET].notna()]
    if len(tr) < 200:
        raise ValueError('경보 모델을 학습할 과거 유효 행이 200개 미만이다.')
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('학습할 모델의 시드가 없다.')
    weights = historical_clone_weights(history, tr)
    inputs = available_inputs(X, cutoff).loc[tr.index, cfg['cols']]
    models = [make_model(cfg['model'], seed, cfg['params']).fit(
        inputs, tr[ALARM_TARGET], sample_weight=weights,
    ) for seed in seeds]
    meta = {
        'train_n': int(len(tr)), 'train_cutoff': cutoff,
        'cols': list(cfg['cols']), 'target_name': ALARM_TARGET,
        'threshold_q95': threshold(tr, q=.95),
        'threshold_q97': threshold(tr, q=.97),
        'threshold_q98': threshold(tr, q=.98),
        'config': cfg,
    }
    return models, meta


def predict(models, X, cutoff, idx, no_plan=False):
    """예측 cutoff에 이용 가능한 입력으로 시드 예측을 평균한다."""
    if not models:
        raise ValueError('예측할 학습 모델이 없다.')
    cols = NO_PLAN_COLS if no_plan else ALARM_COLS
    inputs = available_inputs(X, cutoff).loc[idx, cols]
    result = np.mean([m.predict(inputs) for m in models], axis=0)
    if not np.isfinite(result).all():
        raise ValueError('경보 예측값에 유한하지 않은 값이 있다.')
    return result


def alarm_flags(pred, threshold_value):
    """임계값과 같은 예측도 경보에 포함한다."""
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('경보 예측값에 유한하지 않은 값이 있다.')
    return pred >= threshold(None, value=threshold_value)
