"""강화 실험 — 추가 개선 수단의 측정·기각 수치를 재현한다.

앞선 강화 단계(report_gain.py)에서 "더 짜낼 수단" 7가지를 기각했고, 이 실험은
남은 축을 다시 전수 확인한 것이다. 결론부터: 8가지를 더 기각하고 1가지를 채택했다.
  채택   15분 최대 직접 학습 (model.MAX15_CFG)
  조건부 휴지 구간 규칙 — MAE -0.40 이지만 테스트 구간 생산계획이 완전해야 쓸 수 있다
  기각   단조제약 · L2규제 · 분할최소이득 · 가지치기 · 최근가중 · 기온축적 ·
         생산축적 · 휴지기준값(N군)

판단 기준은 전부 CORE(7-9월) 고유일 1,708행 · 시드 3개다.
"기준대비" 차이가 "흔들림"(시드 3개 표준편차)보다 작으면 차이 없다고 본다.

실행  python src/reinforce.py            전부 (약 25분)
      python src/reinforce.py tune       단조제약·규제·가지치기        (약 7분)
      python src/reinforce.py recency    최근 데이터 가중              (약 2분)
      python src/reinforce.py feat       축적 피처·휴지 기준값(N군)     (약 4분)
      python src/reinforce.py align      생산량 기록 시각 (C-4)        (즉시)
      python src/reinforce.py bias       피크 편향의 정체              (즉시)
      python src/reinforce.py idle       휴지 구간 — 여지와 위험       (약 2분)
      python src/reinforce.py max15      15분 최대 직접 학습 vs 환산    (약 3분)
      python src/reinforce.py seeds      시드 개수 3 vs 10             (약 4분)

align·bias·idle 의 일부는 outputs/final/predictions.csv 를 읽는다.
없으면 python src/report_final.py 를 먼저 돌릴 것.
"""
import sys

import numpy as np
import pandas as pd

from features import build, FEATURES, peak_ratio
from model import (ALARM_CFG, CLONE_W, CORE_FOLDS, FINAL_CFG, FOLDS, MAX15_CFG,
                   PEAK_W, SEEDS, SHALLOW, SLOW, make_model, rolling_eval, score)
from preprocess import ROOT, load

OUT = ROOT / 'outputs'
PRED = OUT / 'final' / 'predictions.csv'
P_FINAL = {**SHALLOW, **SLOW, 'feature_fraction': 0.6}   # 강화 최종의 나무 설정
COLS = FEATURES['h7_lag']
IDLE_LEVEL = 30     # 실제 전력이 이 값 미만이면 "휴지" 로 본다 (휴지 평균 21.8)


# ── 공용 ────────────────────────────────────────────────────────────────────
def _core(X, cfg, seeds=SEEDS):
    """설정 하나를 시드별로 돌려 CORE 성적을 시드 평균 + 흔들림으로 돌려준다."""
    rows = []
    for sd in seeds:
        P = rolling_eval(X, seed=sd, **cfg)
        c = P[P.fold.isin(CORE_FOLDS)]
        s = score(c).to_dict()
        lo = c.y < IDLE_LEVEL                     # 휴지 구간 성적도 같이 본다
        s['휴지MAE'] = (c.y - c.p)[lo].abs().mean()
        s['휴지편향'] = (c.y - c.p)[lo].mean()
        rows.append(s)
    d = pd.DataFrame(rows)
    return {'MAE': d.MAE.mean(), '흔들림': d.MAE.std(), 'RMSE': d.RMSE.mean(),
            'peakMAE': d.peakMAE.mean(), '휴지MAE': d.휴지MAE.mean(),
            '휴지편향': d.휴지편향.mean()}


def _table(X, runs, name, cols=None):
    """설정 여러 개를 돌려 표로 찍고 csv 로 남긴다. 첫 항목이 기준이다."""
    rows = []
    for label, cfg in runs.items():
        r = {'설정': label, **_core(X, cfg)}
        rows.append(r)
        print('  %-24s MAE %7.3f (+-%.3f)  RMSE %7.3f  peakMAE %7.3f'
              % (label, r['MAE'], r['흔들림'], r['RMSE'], r['peakMAE']), flush=True)
    t = pd.DataFrame(rows)
    t['기준대비'] = t.MAE - t.MAE.iloc[0]
    t = t[['설정', 'MAE', '흔들림', '기준대비', 'RMSE', 'peakMAE'] + (cols or [])]
    path = OUT / f'reinforce_{name}.csv'
    t.round(4).to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  → {path}')
    return t


def add_extra(X):
    """강화 실험에서 시험한 피처를 X 에 붙인다 (전부 기각됐다. 재현용).

    축적군  전력은 생산에 한 박자 늦게 반응하므로 순간값보다 쌓인 양이
            맞을 수 있다는 가정. 생산계획·기상은 예측 시점에 아는 값이라
            당일 안에서 누적해도 누수가 아니다
    N군     휴지 시간의 전력 수준. 과거전력 피처가 휴무·중단 시간을 전부 결측
            처리하기 때문에(features.py) 참조할 값이 아예 없다는 문제를 겨냥했다.
            1주 앞 예측이므로 반드시 7일 이상 지연시킨다
    """
    X = X.copy()
    d = X.groupby('날짜')
    X['prod_roll3'] = X['prod'].rolling(3, min_periods=1).mean()
    X['prod_cum'] = d['prod'].cumsum()
    X['temp_roll3'] = X['temp'].rolling(3, min_periods=1).mean()
    X['temp_roll24'] = X['temp'].rolling(24, min_periods=1).mean()
    X['day_temp_max'] = X['날짜'].map(d['temp'].max())
    X['cool_cum'] = d['cool'].cumsum()

    df = load()
    off = df['is_off'] | df['is_stop']
    day_off = off.groupby(df['날짜']).any()
    idle_day = df['target'].where(off).groupby(df['날짜']).mean().where(day_off)
    X['prev_off_mean'] = df['날짜'].map(idle_day.shift(7).ffill()).values
    X['base_load'] = df['날짜'].map(idle_day.shift(7).expanding().mean()).values
    grid = (df['target'].to_frame('v').assign(d=df['날짜'], h=df['시간'])
            .pivot(index='d', columns='h', values='v'))
    X['lag_prev_off'] = grid.where(day_off, axis=0).shift(7).ffill().stack().reindex(
        pd.MultiIndex.from_arrays([df['날짜'], df['시간']])).values
    X['lag168_was_off'] = off.shift(168).fillna(False).astype(int).values
    return X


def _fold_ratio(X):
    """폴드별 (15분최대 / 시간평균) 시각별 비율. 각 폴드의 학습 구간에서만 뽑는다.
    전 구간에서 뽑으면 평가 구간이 섞여 들어가 환산 쪽이 부당하게 유리해진다."""
    out = {}
    for m in FOLDS:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        if len(va) == 0:
            continue
        tr = X[(X['dt'] < va['dt'].min()) & X.train_ok]
        if len(tr) < 200:
            continue
        out[m] = peak_ratio(tr).ratio_a
    return out


def _apply_ratio(ratio, folds, hours):
    """행마다 그 폴드의 환산 계수를 붙인다 (없으면 1.0)."""
    return np.array([ratio[f].get(h, 1.0) if f in ratio else 1.0
                     for f, h in zip(folds, hours)])


ACC_T = ['temp_roll3', 'temp_roll24', 'day_temp_max', 'cool_cum']
ACC_P = ['prod_roll3', 'prod_cum']
N_GRP = ['base_load', 'prev_off_mean', 'lag_prev_off', 'lag168_was_off']
# SHAP 기여도 합계 1% 미만 (outputs/shap_importance.csv)
DROP_TINY = ['lag168_na', 'is_weekend', 'off_run_prev', 'after_off', 'rain', 'wind', 'dow']
DROP_MORE = DROP_TINY + ['humid', 'days_since_off', 'is_transition', 'cool', 'is_holiday']


# ── 1. 단조제약·규제·가지치기 ──────────────────────────────────────────────
def sec_tune(X):
    """공정 지식을 제약으로 걸거나 모델을 더 규제하면 나아지나.

    단조제약은 "생산량이 늘면 전력도 늘어야 한다" 를 나무에 직접 거는 것이다.
    그런데 LightGBM 은 objective='l1' 에서 이를 금지한다 (경사가 상수라서).
    목적함수를 바꿔야 쓸 수 있으므로, 목적함수 효과와 분리되게 짝을 지어 잰다.
    """
    up = {'prod', 'prod_cap', 'prod_log', 'day_prod', 'cool', 'temp', 'lag168',
          'lag336', 'lag_week_mean4', 'lag_best', 'last_week_day_mean', 'last_week_day_max'}
    down = {'prod_zero', 'day_prod_zero', 'is_off'}
    mono = [1 if c in up else (-1 if c in down else 0) for c in COLS]

    print('\n[1] 단조제약·규제·가지치기')
    return _table(X, {
        '기준 (l1)': dict(FINAL_CFG),
        'l2': dict(FINAL_CFG, params={**P_FINAL, 'objective': 'l2'}),
        'l2 + 단조제약': dict(FINAL_CFG, params={**P_FINAL, 'objective': 'l2',
                                               'monotone_constraints': mono}),
        'huber': dict(FINAL_CFG, params={**P_FINAL, 'objective': 'huber'}),
        'huber + 단조제약': dict(FINAL_CFG, params={**P_FINAL, 'objective': 'huber',
                                                  'monotone_constraints': mono}),
        'L2 규제 1.0': dict(FINAL_CFG, params={**P_FINAL, 'lambda_l2': 1.0}),
        'L2 규제 5.0': dict(FINAL_CFG, params={**P_FINAL, 'lambda_l2': 5.0}),
        '분할 최소이득 0.1': dict(FINAL_CFG, params={**P_FINAL, 'min_gain_to_split': 0.1}),
        '가지치기 7개 제거': dict(FINAL_CFG, cols=[c for c in COLS if c not in DROP_TINY]),
        '가지치기 12개 제거': dict(FINAL_CFG, cols=[c for c in COLS if c not in DROP_MORE]),
    }, 'tune')


# ── 2. 최근 데이터 가중 ────────────────────────────────────────────────────
def sec_recency(X):
    """1-4월은 계절이 다르다. 오래된 행의 무게를 줄이면 나아지나."""
    print('\n[2] 최근 데이터 가중 (반감기)')
    return _table(X, {
        '기준 (가중 없음)': dict(FINAL_CFG),
        '반감기 30일': dict(FINAL_CFG, decay_days=30),
        '반감기 60일': dict(FINAL_CFG, decay_days=60),
        '반감기 120일': dict(FINAL_CFG, decay_days=120),
    }, 'recency')


# ── 3. 축적 피처·휴지 기준값 ───────────────────────────────────────────────
def sec_feat(X):
    print('\n[3] 축적 피처 · 휴지 기준값(N군)')
    Z = add_extra(X)
    return _table(Z, {
        '기준': dict(FINAL_CFG),
        '+기온 축적 4개': dict(FINAL_CFG, cols=COLS + ACC_T),
        '+생산 축적 2개': dict(FINAL_CFG, cols=COLS + ACC_P),
        '+축적 전부': dict(FINAL_CFG, cols=COLS + ACC_T + ACC_P),
        '+기저부하 1개': dict(FINAL_CFG, cols=COLS + ['base_load']),
        '+N군 전체 4개': dict(FINAL_CFG, cols=COLS + N_GRP),
    }, 'feat', cols=['휴지MAE', '휴지편향'])


# ── 4. 생산량 기록 시각 (C-4) ──────────────────────────────────────────────
def sec_align(X):
    """[0] C-4: 19시 생산이 가장 많은데 전력은 중간이다. 기록 시각이 어긋났나.

    생산계획군이 SHAP 기여도의 64% 니, 어긋나 있으면 가장 중요한 피처가 밀려 있다는
    뜻이다. 생산량을 앞뒤로 밀어 가며 전력과의 상관이 어디서 최대인지 본다.
    """
    print('\n[4] 생산량 기록 시각이 어긋났나 (C-4)')
    s = X[X.train_ok_strict & ~X.is_off & ~X.is_stop]
    print(f'  대상 {len(s)}행 (가동일 · 학습 가능 행)')
    rows = []
    for k in range(-4, 5):
        b = X['prod'].shift(k).loc[s.index]
        a, ok = X.loc[s.index, 'target'], b.notna()
        rows.append({'k': k, '피어슨': np.corrcoef(a[ok], b[ok])[0, 1],
                     '스피어만': a[ok].corr(b[ok], 'spearman')})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    best = int(t.loc[t.피어슨.idxmax(), 'k'])
    print(f'  → 상관 최대는 k={best}.', '어긋나지 않았다' if best == 0 else '어긋남 의심')
    g = s.groupby('hour').agg(생산량=('prod', 'mean'), 전력=('target', 'mean'))
    print(f'  전력 최대 시각 {int(g.전력.idxmax())}시 ({g.전력.max():.0f}) / '
          f'생산 최대 시각 {int(g.생산량.idxmax())}시 ({g.생산량.max():.0f})')
    print('  두 최대 시각이 다른 것은 기록 오류가 아니라 "피크는 기동에서 온다" 자체다')
    t.round(4).to_csv(OUT / 'reinforce_align.csv', index=False, encoding='utf-8-sig')
    return t


# ── 5. 피크 편향의 정체 ────────────────────────────────────────────────────
def sec_bias():
    """피크행 편향 +11.9 가 모델이 피크를 깎는 것인가, 선택 효과인가.

    처방이 반대라서 반드시 갈라야 한다. 모델 결함이면 예측을 올려야 하고,
    선택 효과면 올리는 순간 전체 MAE 가 나빠진다.
    같은 상위 5% 를 (가) 실제값으로 (나) 예측값으로 골라 편향을 비교한다.
    선택 효과라면 두 편향의 부호가 반대로 나온다.
    """
    print('\n[5] 피크 편향은 모델 결함인가 선택 효과인가')
    if not PRED.exists():
        print('  predictions.csv 가 없다. python src/report_final.py 를 먼저 돌릴 것')
        return None
    d = pd.read_csv(PRED)
    d = d[d.fold.isin(CORE_FOLDS)].copy()
    d['err'] = d.y - d.pred                       # +면 과소예측
    print(f'  CORE {len(d)}행 · MAE {d.err.abs().mean():.3f} · 전체 편향 {d.err.mean():+.3f}')
    rows = []
    for q in (.95, .90):
        for lab, m in [('실제값으로 고름', d.y >= d.y.quantile(q)),
                       ('예측값으로 고름', d.pred >= d.pred.quantile(q))]:
            s = d[m]
            rows.append({'상위': f'{100 * (1 - q):.0f}%', '고르는 기준': lab, 'n': len(s),
                         '실제평균': s.y.mean(), '예측평균': s.pred.mean(),
                         '편향': s.err.mean(), 'MAE': s.err.abs().mean()})
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    print('  → 부호가 반대면 선택 효과다 (평균 회귀). 예측을 올려 고칠 문제가 아니다')
    d['예측구간'] = pd.qcut(d.pred, 10, labels=[f'{i + 1}분위' for i in range(10)])
    q = d.groupby('예측구간', observed=True).agg(
        n=('y', 'size'), 예측평균=('pred', 'mean'), 실제평균=('y', 'mean'),
        편향=('err', 'mean'), MAE=('err', lambda e: e.abs().mean()))
    print('\n  예측값 구간별 편향 (이 표가 진짜 편향을 보여준다)')
    print(q.round(2).to_string())
    t.round(4).to_csv(OUT / 'reinforce_bias.csv', index=False, encoding='utf-8-sig')
    return t


# ── 6. 휴지 구간 — 여지와 위험 ─────────────────────────────────────────────
def sec_idle(X):
    """남은 오차가 어디 있나. 그리고 그걸 규칙으로 줄일 수 있나.

    휴무·가동중단 시간의 전력은 24시각 전부 21.8 로 거의 상수인데 모델은 높게 본다.
    그 구간을 상수로 답하면 큰 이득인데, 문제는 "휴지인지 미리 알 수 있는가" 다.
    생산계획이 0 인 날 64일 중 15일은 실제로 고부하다 (생산기록 누락 의심일).
    is_off·is_prod_missing 은 타깃에서 유도한 값이라 예측 시점에는 쓸 수 없다.
    """
    print('\n[6] 휴지 구간 — 남은 여지와 그 위험')
    df = load()
    off = df['is_off'] | df['is_stop']
    print(f'  휴무·중단 {int(off.sum())}행의 전력: 평균 {df.loc[off, "target"].mean():.1f} · '
          f'표준편차 {df.loc[off, "target"].std():.1f} · '
          f'범위 {df.loc[off, "target"].min():.0f}-{df.loc[off, "target"].max():.0f}')

    if PRED.exists():
        d = pd.read_csv(PRED)
        d = d[d.fold.isin(CORE_FOLDS)].merge(
            X.assign(날짜=X['날짜'].astype(int))[
                ['날짜', 'hour', 'day_prod_zero', 'hrs_since_prod_end', 'hrs_to_prod_start']],
            on=['날짜', 'hour'], how='left')
        d['ae'] = (d.y - d.pred).abs()
        lo = d.y < IDLE_LEVEL
        base = d.ae.mean()
        fixed = (np.abs(d.loc[lo, 'y'] - 22).sum() + d.loc[~lo, 'ae'].sum()) / len(d)
        print(f'  CORE {len(d)}행 중 휴지 {int(lo.sum())}행 ({100 * lo.mean():.0f}%) · '
              f'그 구간 MAE {d.loc[lo, "ae"].mean():.2f} · 편향 {(d.y - d.pred)[lo].mean():+.2f}')
        print(f'  휴지 구간을 전부 22 로 답하면 전체 MAE {base:.3f} → {fixed:.3f}'
              f' ({fixed - base:+.3f})  ← 남은 여지의 크기')

        print('\n  규칙별 정밀도 (정밀도가 100% 가 아니면 가동 중인 행을 깎아 버린다)')
        rows = []
        win = (d.hrs_to_prod_start > 0) | (d.hrs_since_prod_end > 0)
        for lab, rule in [('① 일 생산계획 0', d.day_prod_zero == 1),
                          ('② 생산창 밖', win),
                          ('③ ① 또는 ②', (d.day_prod_zero == 1) | win),
                          ('④ ③ & 예측 60 미만', ((d.day_prod_zero == 1) | win) & (d.pred < 60))]:
            tp = int((rule & lo).sum()); fp = int((rule & ~lo).sum())
            mae = (d.y - d.pred.where(~rule, 22.0)).abs().mean()
            rows.append({'규칙': lab, '적용행': int(rule.sum()),
                         '정밀도': tp / max(tp + fp, 1), '재현율': tp / max(int(lo.sum()), 1),
                         '전체MAE': mae})
            print('    %-20s 적용 %4d행  정밀도 %5.1f%%  재현율 %5.1f%%  전체 MAE %.3f'
                  % (lab, rule.sum(), 100 * rows[-1]['정밀도'],
                     100 * rows[-1]['재현율'], mae))
        pd.DataFrame(rows).round(4).to_csv(OUT / 'reinforce_idle_rule.csv',
                                           index=False, encoding='utf-8-sig')

    # 위험: 생산계획 0 인 날을 달력만으로 가를 수 있나
    day = df.groupby('날짜').agg(일생산=('생산량', 'sum'), dow=('day', 'first'),
                                공휴일=('is_holiday', 'first'), m=('m', 'first'),
                                휴무=('is_off', 'first'))
    z0 = day.일생산 == 0
    day['연속0'] = z0.shift(1).fillna(False) | z0.shift(-1).fillna(False)
    z = day[z0]
    print(f'\n  생산계획 0 인 날 {len(z)}일 = 휴무 {int(z.휴무.sum())}일 + '
          f'누락 의심 {int((~z.휴무).sum())}일 (누락일 전력 평균 100-136)')
    rows = []
    for lab, rule in [('전부 적용', pd.Series(True, index=z.index)),
                      ('주말만', z.dow >= 6), ('일요일만', z.dow == 7),
                      ('주말 또는 공휴일', (z.dow >= 6) | (z.공휴일 == 1)),
                      ('연속 0', z.연속0), ('주말 & 연속 0', (z.dow >= 6) & z.연속0)]:
        tp = int((rule & z.휴무).sum()); fp = int((rule & ~z.휴무).sum())
        rows.append({'기준': lab, '대상일': int(rule.sum()), '그중 누락': fp,
                     '정밀도': tp / max(tp + fp, 1)})
        print('    %-18s 대상 %2d일  그중 누락 %2d일  정밀도 %5.1f%%'
              % (lab, rule.sum(), fp, 100 * rows[-1]['정밀도']))
    print('  → 정밀도 100% 인 기준이 없다. 예측 시점 정보로는 가를 수 없다 → 규칙 채택 불가')
    print('    테스트 구간 생산계획이 완전하다고 확인되면 그때 ① 을 켜면 된다')
    pd.DataFrame(rows).round(4).to_csv(OUT / 'reinforce_idle_sep.csv',
                                       index=False, encoding='utf-8-sig')

    # 안전장치 시험: 누락일에 모델이 뭐라고 예측하나
    print('\n  안전장치("모델 예측이 낮을 때만 적용")가 누락일을 걸러내나')
    rows = []
    for dt in day[z0 & ~day.휴무].index[-3:]:     # 테스트 구간에 가까운 최근 3일만
        te = X[X['날짜'].astype(int) == dt]
        tr = X[(X['dt'] < te['dt'].min()) & X['train_ok_strict']]
        if len(tr) < 200:
            continue
        w = (np.where(tr.is_clone, CLONE_W, 1.0)
             * np.where(tr.target >= tr.target.quantile(.95), PEAK_W, 1.0))
        p = np.mean([make_model('lgbm', s, P_FINAL).fit(tr[COLS], tr.target, sample_weight=w)
                     .predict(te[COLS]) for s in SEEDS], axis=0)
        rows.append({'날짜': dt, '실제평균': te.target.mean(), '예측평균': p.mean(),
                     '예측 60미만 시각': int((p < 60).sum())})
        print('    %s  실제 %5.1f  예측 %5.1f  (24시각 중 %d개가 60 미만)'
              % (dt, te.target.mean(), p.mean(), (p < 60).sum()))
    print('  → 60 미만 시각이 많으면 안전장치가 작동하지 않는다는 뜻이다')
    pd.DataFrame(rows).round(2).to_csv(OUT / 'reinforce_idle_hedge.csv',
                                       index=False, encoding='utf-8-sig')


# ── 7. 15분 최대 ───────────────────────────────────────────────────────────
def sec_max15(X):
    """요금은 시간 평균이 아니라 15분 최대로 매겨진다.
    두 경로를 같은 기준에서 비교한다. 환산 계수는 각 폴드의 학습 구간에서만 뽑는다.
    """
    print('\n[7] 15분 최대 — 직접 학습 vs 계수 환산')
    ratio = _fold_ratio(X)                                     # 폴드별 환산 계수
    rows = []
    for sd in SEEDS:
        a = rolling_eval(X, seed=sd, **MAX15_CFG)              # 직접 학습
        b = rolling_eval(X, seed=sd, **FINAL_CFG)              # 시간 평균
        a, b = [t[t.fold.isin(CORE_FOLDS)] for t in (a, b)]
        hour = X.loc[b.idx, 'hour'].values
        y15 = X.loc[b.idx, 'target_max15'].values
        conv = b.p.values * _apply_ratio(ratio, b.fold.values, hour)
        for lab, p in [('직접 학습', a.p.values), ('시각별 계수 환산', conv),
                       ('(참고) 시간평균 그대로', b.p.values)]:
            yy = a.y.values if lab == '직접 학습' else y15
            e = yy - p
            rows.append({'방법': lab, 'seed': sd, 'MAE': np.abs(e).mean(),
                         'RMSE': np.sqrt((e ** 2).mean()),
                         'peakMAE': np.abs(e[a.peak.values]).mean(), '편향': e.mean()})
    t = (pd.DataFrame(rows).groupby('방법', sort=False)
         .agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'), RMSE=('RMSE', 'mean'),
              peakMAE=('peakMAE', 'mean'), 편향=('편향', 'mean')))
    print('  실제값 = target_max15 (A안) · 편향 +면 낮게 본 것')
    print(t.round(3).to_string())
    print('  → 차이가 흔들림보다 크면 직접 학습을 쓴다 (model.MAX15_CFG)')

    # 일 최대는 경로가 다를 수 있다 — 네 가지를 함께 본다
    print('\n  "일 최대" 는 따로 봐야 한다 (추정량은 용도별로)')
    def avg(cfg):
        ps = [rolling_eval(X, seed=s, **cfg).set_index('idx') for s in SEEDS]
        return pd.concat([x.p for x in ps], axis=1).mean(axis=1), ps[0].fold
    p_pt, fold = avg(FINAL_CFG)
    p_hi, _ = avg(ALARM_CFG)
    p_m, _ = avg(MAX15_CFG)
    p_mhi, _ = avg(dict(ALARM_CFG, target='target_max15'))
    rr = _apply_ratio(ratio, fold.values, X.loc[p_pt.index, 'hour'].values)
    g = pd.DataFrame({'fold': fold, '날짜': X.loc[p_pt.index, '날짜'].values,
                      '실제': X.loc[p_pt.index, 'target_max15'].values,
                      '직접_점': p_m, '직접_분위': p_mhi,
                      '환산_점': p_pt.values * rr, '환산_분위': p_hi.values * rr})
    g = g[g.fold.isin(CORE_FOLDS)]
    day = g.groupby('날짜').agg(['max', 'size'])
    day = day[(day[('실제', 'size')] >= 20) & (day[('실제', 'max')] > 60)]   # 가동일만
    rows = []
    for k, lab in [('직접_점', '15분 직접 학습, 점 예측의 최대'),
                   ('직접_분위', '15분 직접 학습, 분위 0.9 의 최대'),
                   ('환산_점', '시간평균 점예측 x 계수의 최대'),
                   ('환산_분위', '시간평균 분위0.9 x 계수의 최대')]:
        e = day[('실제', 'max')] - day[(k, 'max')]
        rows.append({'방법': lab, 'MAE': e.abs().mean(), '편향': e.mean(),
                     'RMSE': np.sqrt((e ** 2).mean())})
    d = pd.DataFrame(rows)
    print(f'  가동일 {len(day)}일 · 실제 일 최대 평균 {day[("실제", "max")].mean():.1f}')
    print(d.round(2).to_string(index=False))
    t.round(4).to_csv(OUT / 'reinforce_max15.csv', encoding='utf-8-sig')
    d.round(4).to_csv(OUT / 'reinforce_max15_day.csv', index=False, encoding='utf-8-sig')
    return t


# ── 8. 시드 개수 ───────────────────────────────────────────────────────────
def sec_seeds(X, k=10):
    """시드를 몇 개 평균해야 하나.
    제출물(predict_test.py)은 시드 평균을 쓰므로, 행 단위로 재야 의미가 있다.
    """
    print(f'\n[8] 시드 개수 (1 ~ {k})')
    out = []
    for m in FOLDS:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
        if len(va) == 0 or len(tr) < 200:
            continue
        w = (np.where(tr.is_clone, CLONE_W, 1.0)
             * np.where(tr.target >= tr.target.quantile(.95), PEAK_W, 1.0))
        ps = {f'p{s}': make_model('lgbm', s, P_FINAL)
              .fit(tr[COLS], tr.target, sample_weight=w).predict(va[COLS]) for s in range(k)}
        out.append(pd.DataFrame({'fold': m, 'y': va.target.values,
                                 'peak': va.target.values >= tr.target.quantile(.95), **ps}))
    c = pd.concat(out, ignore_index=True)
    c = c[c.fold.isin(CORE_FOLDS)]
    rows = []
    for n in (1, 2, 3, 5, 7, k):
        e = c.y - c[[f'p{s}' for s in range(n)]].mean(axis=1)
        rows.append({'시드 수': n, 'MAE': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()),
                     'peakMAE': e[c.peak].abs().mean()})
    t = pd.DataFrame(rows)
    t['3개 대비'] = t.MAE - t.MAE[t['시드 수'] == 3].iloc[0]
    print('  행 단위 지표 (시드 n개 평균 예측) · CORE')
    print(t.round(4).to_string(index=False))
    single = [np.abs(c.y - c[f'p{s}']).mean() for s in range(k)]
    print(f'  시드 1개씩 따로 쓰면 {min(single):.3f} - {max(single):.3f} '
          f'(폭 {max(single) - min(single):.3f}) → 평균을 쓰는 이유')
    print(f'  ★ 늘릴수록 수렴한다. {k}개 평균 {t.MAE.iloc[-1]:.3f} 가 참값에 가깝고, '
          f'3개 값 {t.MAE[t["시드 수"] == 3].iloc[0]:.3f} 은 시드 운이 섞인 값이다')
    t.round(4).to_csv(OUT / 'reinforce_seeds.csv', index=False, encoding='utf-8-sig')
    return t


# ── 9. 분위 모델의 보정 ────────────────────────────────────────────────────
def sec_calib():
    """경보 상한(분위 0.9)이 이름값만큼 보수적인가.
    이름대로라면 실제값이 그 선 아래에 있을 확률이 90% 여야 한다. 실제로 재 본다.
    보고서에 "90% 신뢰 상한" 처럼 쓰면 과장이 되는지 확인하는 목적이다.
    """
    print('\n[9] 분위 0.9 상한의 실제 포함률 (이론값 90%)')
    if not PRED.exists():
        print('  predictions.csv 가 없다. python src/report_final.py 를 먼저 돌릴 것')
        return None
    d = pd.read_csv(PRED)
    c = d[d.fold.isin(CORE_FOLDS)]
    rows = [{'구간': 'CORE 전체', 'n': len(c), '포함률': (c.y <= c.pred_hi).mean()}]
    for k, g in c.groupby('fold'):
        rows.append({'구간': k, 'n': len(g), '포함률': (g.y <= g.pred_hi).mean()})
    for lab, m in [('휴지 행(실제 30 미만)', c.y < IDLE_LEVEL),
                   ('실제 상위 10% 행', c.y >= c.y.quantile(.9))]:
        g = c[m]
        rows.append({'구간': lab, 'n': len(g), '포함률': (g.y <= g.pred_hi).mean()})
    t = pd.DataFrame(rows)
    t['포함률'] = (100 * t.포함률).round(1)
    print(t.to_string(index=False))
    print('  → 명목 90% 보다 낮으면 "90% 신뢰 상한" 이라고 쓸 수 없다.')
    print('    실제가 높은 행에서 포함률이 크게 떨어지는 것은 선택 효과와 같은 현상이다')
    t.to_csv(OUT / 'reinforce_calib.csv', index=False, encoding='utf-8-sig')
    return t


SECTIONS = {'tune': sec_tune, 'recency': sec_recency, 'feat': sec_feat,
            'align': sec_align, 'bias': lambda X: sec_bias(), 'idle': sec_idle,
            'max15': sec_max15, 'seeds': sec_seeds, 'calib': lambda X: sec_calib()}

if __name__ == '__main__':
    want = [a for a in sys.argv[1:] if not a.startswith('-')] or list(SECTIONS)
    bad = [w for w in want if w not in SECTIONS]
    if bad:
        raise SystemExit(f'모르는 구간: {bad}. 쓸 수 있는 것: {list(SECTIONS)}')
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    print(f'강화 실험 · 판단 기준 CORE(7-9월) 고유일 · 시드 {list(SEEDS)}')
    print('"기준대비" 가 "흔들림" 보다 작으면 차이 없다고 본다')
    for w in want:
        SECTIONS[w](X)
    print('\n해석은 보고서 제2장(분석의 한계 및 후속 과제)에 정리했다')
