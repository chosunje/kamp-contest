"""현재 강화 모델의 복제일 가중치 0.1/0.3 재비교.

실행: python src/clone_weight_analysis.py
FINAL_CFG의 피처·나무·피크 가중치를 고정하고 clone_w만 바꾼다.
시간 평균과 15분 최대는 별도 타깃으로 학습·평가한다.
기존 rolling_eval을 재현하는 사후 비교이며 실제 is_off와 전체 자료의
복제 판정이 포함돼 D-1 운영 검증 결과로 해석하지 않는다.
"""
from copy import deepcopy
import json
import platform
from time import perf_counter

import lightgbm
import numpy as np
import pandas as pd

from features import build
from model import CORE, CORE_FOLDS, FINAL_CFG, FOLDS, SEEDS, make_model, rolling_eval
from preprocess import ROOT
from quantile_analysis import metrics

WEIGHTS = (.1, .3)
TARGETS = ('target', 'target_max15')
OUT = ROOT / 'outputs'


def compare(X):
    """동일 평가 행에서 가중치만 다른 시드별·앙상블 예측을 만든다."""
    frames = []
    total = len(TARGETS) * len(WEIGHTS) * len(SEEDS) * len(FOLDS)
    step = 0
    for target in TARGETS:
        for weight in WEIGHTS:
            cfg = deepcopy(FINAL_CFG)
            cfg.update(clone_w=weight, target=target)
            for seed in SEEDS:
                for fold in FOLDS:
                    step += 1
                    va = X[(X.ym == fold) & X.train_ok & ~X.is_clone]
                    if va.empty:
                        continue
                    cutoff = va.dt.min()
                    tr = X[(X.dt < cutoff) & X[cfg['train_flag']]]
                    if len(tr) < 200:
                        continue
                    started = perf_counter()
                    print(f'[{step}/{total}] {target} clone_w={weight} '
                          f'seed={seed} {fold}: 학습 {len(tr):,}, 평가 {len(va):,}',
                          flush=True)
                    p = rolling_eval(X, seed=seed, folds=[fold], **cfg)
                    pd.testing.assert_index_equal(pd.Index(p.idx), pd.Index(va.index),
                                                  check_names=False)
                    np.testing.assert_array_equal(p.y, va[target])
                    threshold = tr[target].quantile(.95)
                    np.testing.assert_array_equal(p.peak, va[target].ge(threshold))
                    z = p.rename(columns={'y': 'target', 'p': 'pred', 'peak': 'is_peak'})
                    for column in ('dt', '날짜', 'hour', 'is_off'):
                        z[column] = X.loc[p.idx, column].to_numpy()
                    z['target_name'], z['clone_w'], z['seed'] = target, weight, str(seed)
                    z['train_cutoff'], z['train_n'] = cutoff, len(tr)
                    z['peak_threshold'] = threshold
                    frames.append(z)
                    print(f'  완료 {perf_counter() - started:.1f}초', flush=True)
    if not frames:
        raise ValueError('비교 가능한 평가 행이 없다.')
    P = pd.concat(frames, ignore_index=True)
    keys = ['idx', 'fold', 'target', 'train_cutoff', 'train_n', 'peak_threshold']
    for _, group in P.groupby('target_name', sort=False):
        comparisons = list(group.groupby(['clone_w', 'seed'], sort=False))
        base = comparisons[0][1].sort_values('idx')[keys].reset_index(drop=True)
        for _, part in comparisons[1:]:
            pd.testing.assert_frame_equal(base, part.sort_values('idx')[keys].reset_index(drop=True))
    ensembles = []
    for _, group in P.groupby(['target_name', 'clone_w'], sort=False):
        parts = [s.sort_values('idx').reset_index(drop=True)
                 for _, s in group.groupby('seed', sort=False)]
        ensemble = parts[0].copy()
        ensemble['pred'] = np.mean([s.pred.to_numpy() for s in parts], axis=0)
        ensemble['seed'] = 'ensemble'
        ensembles.append(ensemble)
    P = pd.concat([P, *ensembles], ignore_index=True)
    if not np.isfinite(P.pred).all():
        raise ValueError('유한하지 않은 예측값이 있다.')
    return P


def result_table(P):
    rows = []
    for (target, weight, seed), group in P.groupby(['target_name', 'clone_w', 'seed'], sort=False):
        periods = [('ALL', group), (CORE, group[group.fold.isin(CORE_FOLDS)])]
        periods.extend(group.groupby('fold', sort=False))
        for period, part in periods:
            if part.empty:
                continue
            values = metrics(part)
            # 이번 비교는 L1 모델이며 P90의 포함률/손실을 평가하는 실험이 아니다.
            values.pop('coverage_pct')
            values.pop('pinball_90')
            rows.append({'target_name': target, 'clone_w': weight, 'seed': seed,
                         'period': period, **values})
    R = pd.DataFrame(rows)
    keys = ['target_name', 'clone_w', 'period']
    singles = R[R.seed != 'ensemble']
    avg = singles.groupby(keys, sort=False).mean(numeric_only=True).reset_index()
    avg['seed'] = 'seed_metric_mean'
    avg['MAE_seed_std'] = singles.groupby(keys, sort=False).MAE.std().to_numpy()
    return pd.concat([R, avg], ignore_index=True)


def summary_text(R, snapshot, elapsed):
    lines = [
        'D07 복제일 가중치 0.1/0.3 재비교 (src/clone_weight_analysis.py 생성)',
        f'실행 환경: Python {platform.python_version()}, numpy {np.__version__}, '
        f'pandas {pd.__version__}, lightgbm {lightgbm.__version__}',
        f'실행 시간: {elapsed:.1f}초',
        f'설정: {len(FINAL_CFG["cols"])}개 피처(h7_lag), strict, '
        f'피크행 가중 {FINAL_CFG["peak_w"]}배, 시드 {list(SEEDS)}',
        f'나무 설정: {json.dumps(snapshot["model_params"], ensure_ascii=False, sort_keys=True)}',
        '변경한 조건: 복제일 가중치만 0.1/0.3. 타깃별로 별도 모델을 학습했다.',
        '평가: 고유일 & train_ok. CORE와 전체 결과를 함께 기록한다.',
        '피크: 해당 타깃의 폴드별 학습 구간 상위 5%. 계약전력 초과 기준이 아니다.',
        '일 최대: 평가 행이 24개인 가동일만 사용한다.',
        'seed_metric_mean은 시드별 지표 평균, ensemble은 예측을 평균한 지표다.',
        '',
        '[2026-10-05 채택 결정]',
        'D07: 현재 강화 L1 점 예측의 복제일 가중치 0.3을 유지한다.',
        '시간 평균과 15분 최대 모두 CORE·ALL MAE/RMSE는 0.3이 낮다.',
        '0.1은 피크 MAE가 낮고 15분 최대의 일 최대 MAE도 낮아 피크 측면의 이점을 기록한다.',
        'D10: 시간 평균은 점 예측, 15분 최대는 별도 피크 분석 타깃으로 사용한다.',
        '각 타깃의 학습 상위 5%는 임시 분석 기준이며 공식 경보 임계는 미확정이다.',
        '이 비교는 P90의 최적 가중치 재선정이나 기존 경보의 15분 최대 전환이 아니다.',
        '',
    ]
    for target in TARGETS:
        lines.append(f'[{target}]')
        for period in (CORE, 'ALL', *FOLDS):
            rows = R[(R.target_name == target) & (R.period == period)]
            if rows.empty:
                continue
            lines.append(f'  {period}')
            for weight in WEIGHTS:
                a = rows[(rows.clone_w == weight) & (rows.seed == 'seed_metric_mean')].iloc[0]
                e = rows[(rows.clone_w == weight) & (rows.seed == 'ensemble')].iloc[0]
                lines.append(f'    {weight}: 시드평균 MAE={a.MAE:.4f} '
                             f'시드표준편차={a.MAE_seed_std:.4f} '
                             f'RMSE={a.RMSE:.4f} peakMAE={a.peakMAE:.4f}')
                lines.append(f'         앙상블 n={int(e.n):,} MAE={e.MAE:.4f} '
                             f'RMSE={e.RMSE:.4f} peakMAE={e.peakMAE:.4f} '
                             f'피크bias={e.peak_bias:+.4f} 피크과소={e.peak_under_pct:.1f}%')
                lines.append(f'         일최대 MAE={e.daily_max_MAE:.4f} '
                             f'bias={e.daily_max_bias:+.4f} 완전가동일={int(e.active_complete_days)} '
                             f'불완전가동일제외={int(e.incomplete_active_days)}')
        lines.append('')
    lines.extend([
        '[해석의 범위]',
        '기존 FINAL_CFG의 동일 조건 비교다. 당일 실측 is_off와 전체 자료 복제 판정이 포함된다.',
        '앞서 누수 검증한 quantile_analysis의 D-1 프로토콜과 다른 결과다.',
        '생산 실적과 기상 실측을 계획·예보로 가정했다. 독립 홀드아웃은 없다.',
        '시간 평균과 15분 최대의 피크 집합이 다르므로 타깃 안에서 가중치를 비교한다.',
        '시드 표준편차는 학습 난수 안정성 지표이며 통계적 유의성 검정이 아니다.',
        '미래 테스트 구간의 부분가동 여부 또는 공식 경보 임계값을 이 결과로 확정하지 않는다.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    started = perf_counter()
    X = build()
    X['ym'] = X.dt.dt.to_period('M').astype(str)
    snapshot = {
        'base_config': deepcopy(FINAL_CFG),
        'model_params': make_model(FINAL_CFG['model'], params=FINAL_CFG['params']).get_params(),
        'weights': WEIGHTS, 'targets': TARGETS, 'seeds': SEEDS, 'folds': FOLDS,
        'protocol': 'reference_final', 'peak_quantile': .95,
    }
    P = compare(X)
    R = result_table(P)
    summary = summary_text(R, snapshot, perf_counter() - started)
    P.to_csv(OUT / 'clone_weight_predictions.csv', index=False, encoding='utf-8-sig')
    R.to_csv(OUT / 'clone_weight_results.csv', index=False, encoding='utf-8-sig')
    (OUT / 'clone_weight_config.json').write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'clone_weight_summary.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
