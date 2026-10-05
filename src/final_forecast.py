"""최종 설정의 운영 입력을 고정한 예측 묶음. 기존 실험 결과는 변경하지 않는다.

시간 평균 L1 / 15분 최대 L1 / 15분 최대 P90을 각각 시드 0·1·2로 학습한다.
FINAL_CFG의 설정은 유지하되 당일 실제 is_off를 제거하고 cutoff 이후 전력을
lag에 쓰지 않는다. 기존 34피처 사후 평가 수치를 이 경로의 성능으로 인용하지 않는다.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

import alarm_policy as policy
from features import FEATURES, build
from model import FINAL_CFG, MAX15_CFG, SEEDS, make_model
from preprocess import RAW
from quantile_analysis import historical_clone_weights


SCHEMA_VERSION = 1


def source_fingerprint():
    """모델·입력 정책이 바뀐 뒤 과거 예측 캐시를 재사용하지 않는다."""
    source = Path(__file__).resolve().parent
    digest = sha256()
    for name in ('final_forecast.py', 'model.py', 'features.py', 'preprocess.py',
                 'alarm_policy.py', 'quantile_analysis.py'):
        digest.update(name.encode('ascii'))
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def _day(value):
    text = str(value)
    date = pd.to_datetime(text, format='%Y%m%d' if len(text) == 8 else '%Y-%m-%d', errors='raise')
    return policy._midnight(date)


def operational_config(kind, no_plan=False):
    if kind == 'alarm':
        return policy.make_config(no_plan=no_plan)
    if kind not in {'point', 'max15'}:
        raise ValueError('지원하지 않는 최종 모델 종류입니다.')
    cfg = deepcopy(FINAL_CFG if kind == 'point' else MAX15_CFG)
    cfg['cols'] = list(FEATURES['h7_no_plan'] if no_plan else policy.ALARM_COLS)
    cfg['target'] = 'target' if kind == 'point' else policy.ALARM_TARGET
    return cfg


def forecast_bundle(X, start, end, no_plan=False, seeds=SEEDS, threshold_value=None):
    """기간 첫날 00시 이전에 한 번 학습하고 모든 예측에 같은 cutoff를 적용한다."""
    start, end = _day(start), _day(end)
    if end < start:
        raise ValueError('종료일은 시작일보다 빠를 수 없습니다.')
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('시드를 하나 이상 지정해야 합니다.')
    te = X[(X.dt >= start) & (X.dt < end + pd.Timedelta(days=1))]
    expected = pd.date_range(start, end + pd.Timedelta(hours=23), freq='h')
    if len(te) != len(expected) or not np.array_equal(te.dt.to_numpy(), expected.to_numpy()):
        raise ValueError('예측 기간에 연속된 하루 24행의 계획·기상 입력이 필요합니다.')
    history = X[X.dt < start]
    safe = policy.available_inputs(X, start)
    predictions, models, configs, training = {}, {}, {}, {}
    for kind in ('point', 'max15', 'alarm'):
        cfg = operational_config(kind, no_plan)
        target = cfg['target']
        tr = history[history.train_ok_strict & history[target].notna()]
        if len(tr) < 200:
            raise ValueError(f'{kind}: 과거 유효 학습행이 200개 미만입니다.')
        weights = historical_clone_weights(history, tr)
        peak_weight = cfg.get('peak_w', 1.)
        if peak_weight != 1.:
            weights = weights * np.where(tr[target] >= tr[target].quantile(.95), peak_weight, 1.)
        fitted = []
        for seed in seeds:
            model = make_model(cfg['model'], seed, cfg.get('params'))
            model.fit(safe.loc[tr.index, cfg['cols']], tr[target], sample_weight=weights)
            fitted.append(model)
        ps = np.mean([m.predict(safe.loc[te.index, cfg['cols']]) for m in fitted], axis=0)
        if not np.isfinite(ps).all():
            raise ValueError('최종 예측에 유한하지 않은 값이 있습니다.')
        predictions[kind], models[kind], configs[kind] = ps, fitted, cfg
        training[kind] = len(tr)
    threshold_training = history[history.train_ok_strict & history[policy.ALARM_TARGET].notna()]
    thr = policy.threshold(threshold_training, value=threshold_value)
    rows = pd.DataFrame({
        'date': te.dt.dt.strftime('%Y-%m-%d').to_numpy(), 'hour': te.hour.to_numpy(),
        'pred': predictions['point'], 'pred_max15': predictions['max15'],
        'pred_hi_max15': predictions['alarm'], 'alarm_threshold': thr,
        'alarm': policy.alarm_flags(predictions['alarm'], thr),
    })
    meta = {
        'schema_version': SCHEMA_VERSION, 'model': f'LightGBM · 시드 {"/".join(map(str, seeds))} 평균',
        'train_cutoff': start.isoformat(), 'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'), 'seeds': list(seeds), 'train_rows': training,
        'feature_count': len(configs['point']['cols']), 'configs': configs,
        'source_fingerprint': source_fingerprint(),
        'policy_id': policy.POLICY_ID,
        'threshold_basis': 'relative_q95' if threshold_value is None else 'provided_absolute',
        'threshold_value': thr, 'unit': '원본 전력 단위', 'no_plan': bool(no_plan),
        'mode': 'historical_demo' if te.target.notna().any() else 'provided_plan',
        'data_source': RAW.name,
        'assumptions': [
            '생산 실적·기상 실측을 계획·예보로 가정한 시연입니다.',
            '기간 첫날 00시 이후의 전력을 모델 입력에 사용하지 않습니다.',
            '15분 컬럼은 D20 A안이며 전력 단위와 계약전력은 확인되지 않았습니다.',
            'P90은 보정된 신뢰상한이 아니며 저감 시뮬레이션은 실제 효과를 보장하지 않습니다.',
            '점 예측은 is_off를 제외한 운영용 입력으로 재학습했으며 기존 사후 MAE와 구분합니다.',
        ],
        'library_versions': {'lightgbm': lightgbm.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    return rows, meta, models


def write_bundle(directory, rows, meta, models):
    """CSV·명세·학습 모델을 함께 저장한다. 모델 파일은 실행 파일이 아닌 텍스트다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = directory / 'models'
    model_dir.mkdir(exist_ok=True)
    files = []
    for kind, fitted in models.items():
        for seed, model in zip(meta['seeds'], fitted):
            path = model_dir / f'{kind}_seed{seed}.txt'
            model.booster_.save_model(str(path))
            files.append({'kind': kind, 'seed': seed, 'path': path.relative_to(directory).as_posix(),
                          'sha256': sha256(path.read_bytes()).hexdigest()})
    meta = deepcopy(meta)
    meta['model_files'] = files
    meta['raw_sha256'] = sha256(RAW.read_bytes()).hexdigest()
    rows.to_csv(directory / 'forecast.csv', index=False, encoding='utf-8-sig')
    payload = {'meta': meta, 'dates': sorted(rows.date.unique().tolist()), 'forecast': rows.to_dict('records')}
    (directory / 'forecast.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (directory / 'model_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return payload


def build_bundle(directory, start=None, end=None, no_plan=False, threshold_value=None):
    X = build()
    dates = sorted(X['날짜'].unique())
    if (start is None) != (end is None):
        raise ValueError('시작일과 종료일을 함께 지정해야 합니다.')
    start, end = (dates[-7], dates[-1]) if start is None else (start, end)
    rows, meta, models = forecast_bundle(X, start, end, no_plan=no_plan, threshold_value=threshold_value)
    return write_bundle(directory, rows, meta, models)
