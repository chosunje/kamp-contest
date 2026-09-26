"""예측 모델. 검증 규약은 baseline.py 와 동일하다 (D12: 시간순 롤링 폴드).
  학습 = 검증월 이전 전체 / 평가 = 고유일만 (복제일을 평가에 넣으면 점수가 부풀려진다)
  개발 폴드는 5 to 8월. 9월은 최종 확인용으로 남겨둔다 (--final 로 한 번만 확인).
  같은 5 to 8월 기준 베이스라인 lag168 은 MAE 26.8 / RMSE 52.5 / peakMAE 44.6

실행: python src/model.py  →  outputs/model_results.csv
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from features import build, FEATURES
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'model_results.csv'
OUT_FIX = ROOT / 'outputs' / 'model_results_fix.csv'
# 개발 폴드는 5 to 8월만 쓰고 9월(9/1 to 9/14)은 최종 확인용으로 남겨둔다 (팀 결정 D17).
# 설정을 고르는 동안 9월을 보면, 마지막에 진짜 실력을 확인할 데이터가 남지 않는다.
FOLDS = ['2021-05', '2021-06', '2021-07', '2021-08']
FINAL_FOLD = '2021-09'
SEEDS = (0, 1, 2)    # 같은 설정을 시드만 바꿔 여러 번 돌린다 (차이가 흔들림보다 큰지 보려고)
CLONE_W = 0.3        # clone='weight' 일 때 복제일에 줄 가중치
NEEDS_FILL = {'rf'}  # 결측을 직접 못 다루는 모델. 트리라서 -999 로 채우면 분기로 갈라낸다

# ── 비교 실험 ──────────────────────────────────────────────────────────────
# BASE 를 기준으로, 각 run 은 "한 가지만" 덮어쓴다. 두 가지를 동시에 바꾸면
# 무엇 때문에 좋아졌는지 알 수 없으므로 반드시 한 번에 하나만 바꿀 것.
BASE = dict(cols=FEATURES['full'], model='lgbm', train_flag='train_ok_strict', clone='keep')

RUNS = {
    '0 기준 (lgbm·full·strict·keep)': {},
    # 축 1. 복제일 처리 (D07 미결) — 161일을 어떻게 다룰 것인가
    '1 복제일 학습 제외':             dict(clone='drop'),
    f'2 복제일 가중치 {CLONE_W}':      dict(clone='weight'),
    # 축 2. 학습 행 (작업내역(조선제).txt [3] 보강 1) — 생산기록 누락 의심일 15일을 뺄 것인가
    '3 누락일 포함 (train_ok)':       dict(train_flag='train_ok'),
    # 축 3. 피처 세트 — 테스트에 생산계획이 안 올 경우 대비
    '4 피처 no_plan (달력+과거전력)':  dict(cols=FEATURES['no_plan']),
    # 축 4. 모델 (과제 요건: 2종 이상 비교)
    '5 RandomForest':                 dict(model='rf'),
    # 축 5. 예측 시계 (같은 문서 [11]) — 1주 앞 세트는 full 에서 최근 lag 4개를 뺀 것이다.
    #        빼면 오히려 좋아진다 ([9] 7번). lag24 계열이 요일 패턴을 흐리기 때문
    '6 1주 앞 (h7_full)':             dict(cols=FEATURES['h7_full']),
    '7 1주 앞 (h7_no_plan)':          dict(cols=FEATURES['h7_no_plan']),

    # ── 여기부터는 "한 번에 하나" 규칙의 예외 ─────────────────────────────
    # 위에서 따로 좋았던 2번과 6번을 합치면 효과가 더해지는지 겹치는지 본다.
    # 8 을 2번·6번과 비교하면 각 요소가 조합 안에서도 여전히 기여하는지 알 수 있다.
    '8 조합 (가중치+1주앞)':           dict(clone='weight', cols=FEATURES['h7_full']),
    '9 조합 + 누락일 포함':            dict(clone='weight', cols=FEATURES['h7_full'],
                                          train_flag='train_ok'),

    # ── 복제일 가중치 값 탐색 ────────────────────────────────────────────
    # 0 에 가까울수록 "제외"(1번), 1.0 이면 "그대로"(0번)와 같아진다.
    # 1.0 은 0번과 같은 값이 나와야 한다 — 가중치 구현이 맞는지 확인하는 용도
    '10 가중치 0.1':                  dict(clone='weight', clone_w=0.1),
    '11 가중치 0.5':                  dict(clone='weight', clone_w=0.5),
    '12 가중치 0.7':                  dict(clone='weight', clone_w=0.7),
    '13 가중치 1.0 (= 0번 검산)':      dict(clone='weight', clone_w=1.0),

    # ── 가장 좋았던 조합(8번)에 가장 좋았던 가중치(0.1)를 합치면 ────────────
    '14 조합 + 가중치 0.1':            dict(clone='weight', clone_w=0.1, cols=FEATURES['h7_full']),
    '15 조합 + 가중치 0.05':           dict(clone='weight', clone_w=0.05, cols=FEATURES['h7_full']),
    # 생산계획이 없는 쪽에도 같은 조합이 통하는지 (7번과 비교할 것)
    '16 no_plan 조합':                dict(clone='weight', clone_w=0.1, cols=FEATURES['h7_no_plan']),
}

# ── 2차 강화: 오차 분석에서 찾은 두 약점을 정면으로 공략한다 ─────────────────
#   약점 A 공휴일 과대예측  공휴일 MAE 36.7 vs 비공휴일 7.6 (작업내역(조선제).txt [16] 12-1)
#   약점 B 피크 과소예측    실제 180 이상 구간에서 평균 +19.2 낮게 본다
# 기준은 8번 조합(현재 최고). 여기서 한 가지씩만 바꿔 효과를 잰다.
BEST = dict(clone='weight', cols=FEATURES['h7_full'])
# 실험으로 고른 나무 설정 (12-3). 기본값(num_leaves 31)은 이 데이터에 비해 너무 컸다
SHALLOW = {'num_leaves': 15, 'min_data_in_leaf': 40}
TINY = {'num_leaves': 7, 'min_data_in_leaf': 60}
SLOW = {'learning_rate': 0.02, 'n_estimators': 1200}
PEAK_W = 3.0
FIX = {
    '17 [기준] 8번 조합': dict(BEST),
    # ── 약점 B: 피크 ──────────────────────────────────────────────────────
    # B-1 목적함수. l1 은 "중앙값"을 맞추므로 높은 쪽을 깎는다.
    #     l2 는 평균, quantile(alpha>0.5)은 위쪽 분위를 맞춰 예측을 끌어올린다
    '18 목적함수 l2':      dict(BEST, params={'objective': 'l2'}),
    '19 분위 0.6':         dict(BEST, params={'objective': 'quantile', 'alpha': 0.6}),
    '20 분위 0.7':         dict(BEST, params={'objective': 'quantile', 'alpha': 0.7}),
    # B-2 피크 행 가중치. 상위 5% 행을 더 무겁게 학습시킨다
    '21 피크행 가중 3배':  dict(BEST, peak_w=3.0),
    '22 피크행 가중 6배':  dict(BEST, peak_w=6.0),
    # B-3 Ridge. 트리와 달리 학습 최대값 밖으로도 예측할 수 있다 (팀원 관찰: 피크 MAE 12.7)
    '23 Ridge 단독':       dict(BEST, model='ridge'),
    '24 LGBM+Ridge 5:5':   dict(BEST, model='lgbm+ridge'),
    '25 LGBM+Ridge 7:3':   dict(BEST, model='lgbm:0.7+ridge:0.3'),
    # ── 약점 A: 공휴일 ────────────────────────────────────────────────────
    # A-1 상호작용 피처. "공휴일 & 생산 있음" 을 나무가 찾지 않아도 되게 직접 넣는다
    '26 공휴일 피처':      dict(BEST, cols=FEATURES['h7_full_hol']),
    # A-2 공휴일 행 가중치. 학습 구간에 2일뿐이라 묻히는 것을 막는다
    '27 공휴일 가중 5배':  dict(BEST, hol_w=5.0),
    '28 공휴일 피처+가중': dict(BEST, cols=FEATURES['h7_full_hol'], hol_w=5.0),
    # ── 하이퍼파라미터 (여기까지 전부 기본값이었다) ────────────────────────
    '29 나무 더 깊게':     dict(BEST, params={'num_leaves': 63, 'min_data_in_leaf': 10}),
    '30 나무 더 얕게':     dict(BEST, params=SHALLOW),
    '31 천천히 오래':      dict(BEST, params=SLOW),
    # ── 여기서 효과가 있던 것끼리 합쳐 본다 (얕은 나무 · 피크 가중 · 느린 학습) ──
    '32 얕게+피크3':        dict(BEST, params=SHALLOW, peak_w=3.0),
    '33 얕게+피크6':        dict(BEST, params=SHALLOW, peak_w=6.0),
    '34 아주 얕게':         dict(BEST, params=TINY),
    '35 아주얕게+피크3':     dict(BEST, params=TINY, peak_w=3.0),
    '36 얕게+천천히':       dict(BEST, params={**SHALLOW, **SLOW}),
    '37 얕게+피크3+공휴일':  dict(BEST, params=SHALLOW, peak_w=3.0, cols=FEATURES['h7_full_hol']),
    '38 얕게+분위0.6':      dict(BEST, params={**SHALLOW, 'objective': 'quantile', 'alpha': 0.6}),
    '39 얕게+천천히+피크3':   dict(BEST, params={**SHALLOW, **SLOW}, peak_w=3.0),
    '40 얕게+천천히+피크6':   dict(BEST, params={**SHALLOW, **SLOW}, peak_w=6.0),
    '41 아주얕게+천천히+피크3': dict(BEST, params={**TINY, **SLOW}, peak_w=3.0),
    '42 아주얕게+천천히':     dict(BEST, params={**TINY, **SLOW}),
    '43 얕게+천천히+피크3+ff6': dict(BEST, params={**SHALLOW, **SLOW, 'feature_fraction': 0.6},
                                    peak_w=3.0),
}

# ── 위 실험으로 고른 최종 설정 ────────────────────────────────────────────
# 35번. 기준(8번) 대비 MAE 는 같고 RMSE·peakMAE·공휴일 오차가 모두 낮다 (12-4)
FINAL_CFG = dict(BASE, **BEST, clone_w=CLONE_W, params=TINY, peak_w=PEAK_W)
# 36번. MAE 만 놓고 보면 가장 낮지만 피크는 덜 잡는다. 지표 우선순위(D10) 확정 전까지 같이 본다
ALT_CFG = dict(BASE, **BEST, clone_w=CLONE_W, params={**SHALLOW, **SLOW})


LGBM_BASE = dict(objective='l1', n_estimators=400, learning_rate=.05, num_leaves=31,
                 min_data_in_leaf=20, feature_fraction=.8, bagging_fraction=.8,
                 bagging_freq=1, verbose=-1)


def make_model(name, seed=0, params=None):
    """모델 1종을 만들어 돌려준다. params 로 기본 설정을 덮어쓸 수 있다."""
    p = dict(params or {})
    if name == 'lgbm':
        return lgb.LGBMRegressor(**{**LGBM_BASE, **p, 'seed': seed})
    if name == 'rf':
        return RandomForestRegressor(**{'n_estimators': 300, 'min_samples_leaf': 5,
                                        'n_jobs': -1, **p, 'random_state': seed})
    if name == 'ridge':
        # 선형 모델이라 학습에서 본 최대값을 넘는 값도 낼 수 있다 (트리는 못 한다) → 피크 담당.
        # 결측은 중앙값으로 메우고 (트리처럼 -999 를 넣으면 직선이 망가진다) 스케일을 맞춘다.
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             Ridge(**{'alpha': 10.0, **p}))
    raise ValueError(f'모르는 모델: {name}')


def _fit_predict(name, seed, params, xtr, ytr, w, xva):
    """모델 1종을 학습해 검증 행 예측을 돌려준다 (결측 처리 방식이 모델마다 다르다)."""
    if name in NEEDS_FILL:
        xtr, xva = xtr.fillna(-999), xva.fillna(-999)
    g = make_model(name, seed, params)
    # 파이프라인(ridge)은 sample_weight 를 마지막 단계 이름으로 받아야 한다
    kw = {} if w is None else {f'{g.steps[-1][0]}__sample_weight'
                               if isinstance(g, Pipeline) else 'sample_weight': w}
    return g.fit(xtr, ytr, **kw).predict(xva)


def rolling_eval(X, cols, model='lgbm', train_flag='train_ok_strict', clone='keep',
                 clone_w=CLONE_W, seed=0, folds=None, params=None,
                 peak_w=1.0, peak_q=.95, hol_w=1.0):
    """시간순 롤링 폴드로 학습·예측한 결과를 행 단위로 돌려준다.

    cols       사용할 피처 목록 (FEATURES['full'] 등)
    model      모델 이름. '+' 로 이으면 예측을 평균낸다 ('lgbm+ridge').
               'lgbm:0.7+ridge:0.3' 처럼 뒤에 가중치를 붙일 수도 있다
    train_flag 학습에 쓸 행 (train_ok 또는 train_ok_strict)
    clone      복제일 처리 (D07 미결). keep 그대로 / drop 학습 제외 / weight 가중치 하향
    clone_w    clone='weight' 일 때 복제일에 줄 가중치. 1.0 이면 keep 과 같고 0 에 가까울수록 drop 에 가깝다
    params     모델별 설정 덮어쓰기. {'lgbm': {...}} 형태이거나 단일 모델이면 {...} 그대로
    peak_w     학습 구간 상위 peak_q 분위 행에 곱할 가중치 (피크 과소예측 대응)
    hol_w      학습 구간 공휴일 행에 곱할 가중치 (공휴일 사례가 적어서 묻히는 문제 대응)
    seed       난수 시드. 같은 설정을 여러 시드로 돌려 "차이가 흔들림보다 큰지" 본다
    """
    names, ws = zip(*[(s.split(':')[0], float(s.split(':')[1]) if ':' in s else 1.0)
                      for s in model.split('+')])
    params = params or {}
    if names and not set(params) <= set(names):     # 단일 모델이면 {'alpha':..} 처럼 바로 줘도 되게
        params = {names[0]: params}
    out = []
    for m in (folds or FOLDS):
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]      # 평가셋은 어떤 설정에서도 고정
        tr = X[(X['dt'] < va['dt'].min()) & X[train_flag]]
        if clone == 'drop':
            tr = tr[~tr.is_clone]
        if len(va) == 0 or len(tr) < 200:
            continue                                        # 6월은 고유일이 2일뿐이라 건너뛸 수 있다
        thr = tr.target.quantile(.95)                       # 피크 임계: 학습 구간 상위 5% (D10 확정 전 임시)
        # 가중치는 곱해서 쌓는다. 전부 1.0 이면 None 으로 넘겨 기존 동작과 완전히 같게 둔다
        w = np.ones(len(tr))
        if clone == 'weight':
            w *= np.where(tr.is_clone, clone_w, 1.0)
        if peak_w != 1.0:
            w *= np.where(tr.target >= tr.target.quantile(peak_q), peak_w, 1.0)
        if hol_w != 1.0:
            w *= np.where(tr.is_holiday == 1, hol_w, 1.0)
        w = None if np.allclose(w, 1.0) else w
        xtr, xva = tr[cols], va[cols]
        p = np.average([_fit_predict(n, seed, params.get(n), xtr, tr.target, w, xva)
                        for n in names], axis=0, weights=ws)
        # idx = 검증 행의 원본 인덱스. 오차 분석에서 조건별로 되짚어 보려고 같이 들고 나간다
        out.append(pd.DataFrame({'fold': m, 'idx': va.index, 'y': va.target.values,
                                 'p': p, 'peak': va.target.values >= thr}))
    return pd.concat(out, ignore_index=True)


def score(s):
    e = s.y - s.p
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()),
                      'peakMAE': e[s.peak].abs().mean(), 'n': len(s)})


def run_all(X, runs, base, seeds=SEEDS, folds=None):
    """설정 여러 개를 시드별로 돌려 폴드별 + 합산(ALL) 성적표를 만든다.
    각 run 은 base 에서 지정한 항목만 덮어쓴 것이다 (= 한 번에 한 가지만 다르다)."""
    rows = []
    for name, over in runs.items():
        kw = {**base, **over}
        for sd in seeds:
            P = rolling_eval(X, seed=sd, folds=folds, **kw)
            rows.append({'run': name, 'seed': sd, 'fold': 'ALL', **score(P)})
            for f, s in P.groupby('fold'):
                rows.append({'run': name, 'seed': sd, 'fold': f, **score(s)})
    res = pd.DataFrame(rows)
    res['n'] = res['n'].astype(int)
    return res


# 강화 실험 결과를 본 실험표에도 올려 둔다 (run_all.py 는 --fix 없이 돌기 때문에)
RUNS['★ 강화 최종 (아주얕은나무+피크가중)'] = FINAL_CFG
RUNS['★ 강화 대안 (얕은나무+느린학습)'] = ALT_CFG

# --final 로 9월을 확인할 때 평가할 후보. 모든 선택이 끝난 뒤 한 번만 돌린다
FINALISTS = {k: RUNS[k] for k in ['0 기준 (lgbm·full·strict·keep)', '8 조합 (가중치+1주앞)',
                                  '★ 강화 최종 (아주얕은나무+피크가중)',
                                  '★ 강화 대안 (얕은나무+느린학습)']}


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    if '--final' in sys.argv:
        # 최종 확인: 미개봉으로 남겨둔 9월에서 후보를 한 번만 평가한다.
        # 여기 숫자를 보고 설정을 다시 고르면 9월도 개발 데이터가 되어 버리므로 그러지 말 것.
        f = run_all(X, FINALISTS, BASE, folds=[FINAL_FOLD])
        g = f[f.fold == 'ALL'].groupby('run', sort=False).agg(
            MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'), RMSE=('RMSE', 'mean'),
            peakMAE=('peakMAE', 'mean'), n=('n', 'first'))
        print(f'[최종 확인] {FINAL_FOLD} · 고유일 · 시드 {list(SEEDS)} 평균')
        print(g.round(2).to_string())
        print('\n※ 이 결과를 보고 설정을 바꾸면 9월이 더 이상 미개봉이 아니게 된다.')
        raise SystemExit

    # --fix : 오차 분석에서 찾은 약점(공휴일·피크)을 공략하는 2차 실험만 돌린다
    runs, out = (FIX, OUT_FIX) if '--fix' in sys.argv else (RUNS, OUT)
    res = run_all(X, runs, BASE)
    res.to_csv(out, index=False, encoding='utf-8-sig')

    a = res[res.fold == 'ALL']
    g = a.groupby('run', sort=False).agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'),
                                         RMSE=('RMSE', 'mean'), peakMAE=('peakMAE', 'mean'),
                                         n=('n', 'first'))
    g['기준대비'] = g.MAE - g.MAE.iloc[0]
    print(f'[폴드 합산] 고유일 기준 · 시드 {list(SEEDS)} 평균')
    print(g[['MAE', '흔들림', '기준대비', 'RMSE', 'peakMAE', 'n']].round(2).to_string())
    print('\n[폴드별 MAE] (시드 평균)')
    print(res[res.fold != 'ALL'].pivot_table(index='run', columns='fold', values='MAE',
                                             aggfunc='mean', sort=False).round(1).to_string())
    # 학습 행이 모자라 폴드가 통째로 건너뛰어지면 평가 행 수가 달라져 비교가 성립하지 않는다
    bad = g[g.n != g.n.iloc[0]]
    if len(bad):
        print('\n[주의] 아래 설정은 평가 행 수가 기준과 달라 MAE 를 직접 비교할 수 없다.')
        print('       학습 데이터가 200행 미만인 폴드가 건너뛰어진 것이다 (폴드별 표의 NaN).')
        for r, v in bad.n.items():
            print(f'       {r}: {v}행 (기준 {g.n.iloc[0]}행)')
        print('       → 공통 폴드(NaN 아닌 열)끼리만 비교할 것')

    print('\n목표선 (고유일 베이스라인): MAE 23.6 / RMSE 37.6 / peakMAE 37.1')
    print('판단 기준: "기준대비" 차이가 "흔들림"보다 작으면 차이 없다고 본다')
    print('→', out)
