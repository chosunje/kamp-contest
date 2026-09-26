"""예측 모델. 검증 규약은 baseline.py 와 동일하다 (D12: 시간순 롤링 폴드).
  학습 = 검증월 이전 전체 / 평가 = 고유일만 (복제일을 평가에 넣으면 점수가 부풀려진다)
  개발 폴드는 5 to 8월. 9월은 최종 확인용으로 남겨둔다 (--final 로 한 번만 확인).
  같은 5 to 8월 기준 베이스라인 lag168 은 MAE 26.8 / RMSE 52.5 / peakMAE 44.6

실행: python src/model.py  →  outputs/model_results.csv
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from features import build, FEATURES
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'model_results.csv'
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


def make_model(name, seed=0):
    """모델 1종을 만들어 돌려준다. 새 모델은 여기에 추가한다."""
    if name == 'lgbm':
        return lgb.LGBMRegressor(objective='l1', n_estimators=400, learning_rate=.05,
                                 num_leaves=31, min_data_in_leaf=20, feature_fraction=.8,
                                 bagging_fraction=.8, bagging_freq=1, verbose=-1, seed=seed)
    if name == 'rf':
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     n_jobs=-1, random_state=seed)
    raise ValueError(f'모르는 모델: {name}')


def rolling_eval(X, cols, model='lgbm', train_flag='train_ok_strict', clone='keep',
                 clone_w=CLONE_W, seed=0, folds=None):
    """시간순 롤링 폴드로 학습·예측한 결과를 행 단위로 돌려준다.

    cols       사용할 피처 목록 (FEATURES['full'] 또는 FEATURES['no_plan'])
    train_flag 학습에 쓸 행 (train_ok 또는 train_ok_strict)
    clone      복제일 처리 (D07 미결). keep 그대로 / drop 학습 제외 / weight 가중치 하향
    clone_w    clone='weight' 일 때 복제일에 줄 가중치. 1.0 이면 keep 과 같고 0 에 가까울수록 drop 에 가깝다
    seed       난수 시드. 같은 설정을 여러 시드로 돌려 "차이가 흔들림보다 큰지" 본다
    """
    out = []
    for m in (folds or FOLDS):
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]      # 평가셋은 어떤 설정에서도 고정
        tr = X[(X['dt'] < va['dt'].min()) & X[train_flag]]
        if clone == 'drop':
            tr = tr[~tr.is_clone]
        if len(va) == 0 or len(tr) < 200:
            continue                                        # 6월은 고유일이 2일뿐이라 건너뛸 수 있다
        w = np.where(tr.is_clone, clone_w, 1.0) if clone == 'weight' else None
        thr = tr.target.quantile(.95)                       # 피크 임계: 학습 구간 상위 5% (D10 확정 전 임시)
        xtr, xva = tr[cols], va[cols]
        if model in NEEDS_FILL:
            xtr, xva = xtr.fillna(-999), xva.fillna(-999)
        g = make_model(model, seed).fit(xtr, tr.target, sample_weight=w)
        # idx = 검증 행의 원본 인덱스. 오차 분석에서 조건별로 되짚어 보려고 같이 들고 나간다
        out.append(pd.DataFrame({'fold': m, 'idx': va.index, 'y': va.target.values,
                                 'p': g.predict(xva), 'peak': va.target.values >= thr}))
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


# --final 로 9월을 확인할 때 평가할 후보. 모든 선택이 끝난 뒤 한 번만 돌린다
FINALISTS = {k: RUNS[k] for k in ['0 기준 (lgbm·full·strict·keep)', '8 조합 (가중치+1주앞)']}


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

    res = run_all(X, RUNS, BASE)
    res.to_csv(OUT, index=False, encoding='utf-8-sig')

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
    print('→', OUT)
