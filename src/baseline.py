"""베이스라인 4종 — 모델이 넘어야 할 목표선. 규칙만으로 예측했을 때의 오차다.

검증 규약은 model.py 를 그대로 따른다 (D12: 시간순 롤링 폴드, 고유일 평가).
폴드 목록·지표 정의·학습 행 플래그를 전부 model.py 에서 가져다 쓴다.
model.py 의 BASE 를 바꾸면 목표선도 같은 조건으로 따라 움직인다 — 정의가 갈리면 비교가 성립하지 않는다.

실행: python src/baseline.py  →  outputs/baseline_results.csv
"""
import numpy as np, pandas as pd
from features import build
from model import FOLDS, score, BASE
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'baseline_results.csv'
TRAIN_FLAG = BASE['train_flag']   # model.py 와 같은 학습 행을 쓴다 (현재 train_ok_strict, D18)
GROUPS = {'고유일': lambda s: s[~s.is_clone], '복제일': lambda s: s[s.is_clone], '전체': lambda s: s}


def predict(tr, va):
    """규칙 4종. 학습이라 할 것은 평균을 내는 것뿐이다."""
    prof = tr.groupby(['dow', 'hour']).target.mean()
    return {
        'lag168': va.raw_lag168.values,                       # 지난주 같은 요일·시각
        '요일x시간 평균': np.array([prof.get((d, h), tr.target.mean())
                                    for d, h in zip(va.dow, va.hour)]),
        'lag24': va.raw_lag24.values,                         # 어제 같은 시각
        '전체 평균': np.full(len(va), tr.target.mean()),
    }


def rolling_eval(X):
    out = []
    for m in FOLDS:
        va = X[(X.ym == m) & X.train_ok]          # 평가셋은 model.py 와 동일하게 고정
        tr = X[(X['dt'] < va['dt'].min()) & X[TRAIN_FLAG]]
        if len(va) == 0 or len(tr) < 200:
            continue
        thr = tr.target.quantile(.95)   # 피크 임계: 학습 구간 상위 5% (D10 확정 전 임시)
        for k, p in predict(tr, va).items():
            out.append(pd.DataFrame({'model': k, 'fold': m, 'y': va.target.values, 'p': p,
                                     'peak': va.target.values >= thr, 'is_clone': va.is_clone.values}))
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    # 규칙 그대로의 lag (휴무·중단 결측 처리를 하지 않은 원본 값)
    X['raw_lag24'] = X.target.shift(24)
    X['raw_lag168'] = X.target.shift(168)

    P = rolling_eval(X)
    rows = []
    for g, f in GROUPS.items():
        s = f(P)
        for k, v in s.groupby('model'):
            rows.append({'model': k, 'group': g, 'fold': 'ALL', **score(v)})
        for (k, m), v in s.groupby(['model', 'fold']):
            rows.append({'model': k, 'group': g, 'fold': m, **score(v)})
    res = pd.DataFrame(rows)
    res['n'] = res['n'].astype(int)
    res.to_csv(OUT, index=False, encoding='utf-8-sig')

    order = ['lag168', '요일x시간 평균', 'lag24', '전체 평균']
    a = res[(res.group == '고유일') & (res.fold == 'ALL')].set_index('model').reindex(order)
    print(f'[고유일 폴드 합산] 모델이 넘어야 할 목표선  (학습 행 {TRAIN_FLAG})')
    print(a[['MAE', 'RMSE', 'peakMAE', 'n']].round(2).to_string())
    print('\n[고유일 폴드별 MAE]')
    print(res[(res.group == '고유일') & (res.fold != 'ALL')]
          .pivot(index='model', columns='fold', values='MAE').reindex(order).round(1).to_string())
    print('\n[복제일 비교] 복제일 성적은 복제 덕분이라 실력이 아니다 (MAE)')
    print(res[(res.fold == 'ALL')].pivot(index='model', columns='group', values='MAE')
          .reindex(order).round(1).to_string())
    print(f'\n목표선: MAE {a.MAE.min():.1f} / RMSE {a.RMSE.min():.1f} / peakMAE {a.peakMAE.min():.1f}'
          '  (지표별 최저 베이스라인)')
    print('→', OUT)
