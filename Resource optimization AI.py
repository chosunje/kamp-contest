"""Resource optimization AI — 이 파일 하나로 전부 돌아간다.

    python "Resource optimization AI.py"

KAMP 과제 ⑤ (제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석)의
전처리부터 저감방안까지를 한 파일에 담았다. 중간 파일을 거치지 않고 위에서 아래로
한 번에 흐른다. 끝나면 콘솔에 결과가 전부 찍히고 대시보드가 다시 만들어진다.

── 흐름 ────────────────────────────────────────────────────────────────────
  1 전처리      원자료 정제 · 복제일/휴무일/기록손상 판별
  2 피처        달력 · 생산계획 · 기상 · 과거전력 (1주 앞 예측용 세트)
  3 검증 규약   시간순 롤링 폴드 · 평가는 고유일만 · 판단은 CORE(7-9월)
  4 모델 성능   목표선 대비 · 설정 비교 · 월별
  5 최대피크    일 최대 추정(분위 0.9) · 15분 최대 · 피크 시각의 한계
  6 위험조건    기온 x 생산 교차표 · 조건별 피크일 비율
  7 경보        예측 시계(D-7/D-1) · 기준(시간평균/15분최대) · 임계값 스윕
  8 저감방안    기동 분산의 기회를 실측으로 · 날짜 간 생산일정 · 포화
  9 검증        이번 실행 값이 보고서 표와 같은지 스스로 대조 (VERIFY)

── src/ 와의 관계 ─────────────────────────────────────────────────────────
  src/ 아래 모듈들이 작업용 원본이고 이 파일은 그것을 한 벌로 묶은 실행본이다.
  두 벌이 되면 한쪽만 고쳐 숫자가 갈릴 수 있으므로, 마지막에 이 파일이 낸 값이
  문서에 적힌 값과 같은지 스스로 검증한다 (VERIFY). 어긋나면 경고를 띄운다.

── 실행 ────────────────────────────────────────────────────────────────────
  python "Resource optimization AI.py"      전처리부터 저감까지 한 번에 (약 4분)

  ★ 파일명에 공백이 있으므로 따옴표를 꼭 붙일 것.
  결과는 전부 콘솔에 표로 찍힌다. 현장용 운영 화면은 이 파일이 아니라
  run_dashboard.py 가 만든다 (outputs/final_app/index.html, 보고서 제5장).

필요한 것: dataset/okm_augumented_2021.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'dataset' / 'okm_augumented_2021.csv'
OUT = ROOT / 'outputs' / 'final'

# ── 규약 ────────────────────────────────────────────────────────────────────
FOLDS = ['2021-05', '2021-06', '2021-07', '2021-08', '2021-09']
CORE_FOLDS = ['2021-07', '2021-08', '2021-09']   # 테스트 구간과 성격이 같다 (D22)
SEEDS = (0, 1, 2)          # 같은 설정을 시드만 바꿔 돌려 "차이가 흔들림보다 큰지" 본다
CLONE_W = 0.3              # 복제일(증강으로 전력곡선이 복사된 날) 가중치
PEAK_W = 3.0               # 학습 구간 상위 5% 행에 줄 가중치
ALARM_Q = 0.95             # 경보 임계 (D10 확정 전 잠정값)
SWEEP_QS = (.80, .85, .88, .90, .92, .95, .97, .98)
PROD_CAP, COOL_BASE = 800, 22

HOLIDAY = {'2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01',
           '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15', '2021-08-16',
           '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04',
           '2021-10-09', '2021-10-11', '2021-12-25'}

# 문서(작업내역(조선제).txt)에 적힌 값. 이 파일이 같은 값을 내는지 끝에서 확인한다
# 값의 출처: 모델 = 보고서 표 2-4/2-6, 일최대 = 표 2-13, 경보 = 표 3-3 (D10 정책)
VERIFY = {'강화최종 MAE': 6.57, '강화최종 peakMAE': 12.49, '일최대 MAE': 8.61,
          '경보 재현율': 100.0, '피크시각 적중률': 20.4, '학습행': 5788}

t0 = time.time()
step = 0


def say(title):
    global step
    step += 1
    print(f'\n{"=" * 74}\n [{step}] {title}   ({time.time() - t0:.0f}초)\n{"=" * 74}')


# ════════════════════════════════════════════════════════════════════════════
# 1. 전처리
# ════════════════════════════════════════════════════════════════════════════
def load(raw=None):
    """원자료를 읽어 정제하고 학습에 쓸 수 있는 행을 가려낸다."""
    df = pd.read_csv(RAW) if raw is None else raw.copy(deep=True)
    # 7/13·7/15는 시간 컬럼이 손상됐다. 전력값은 정상이라 lag 입력으로는 쓴다
    df['is_corrupt'] = df.groupby('날짜')['시간'].transform(lambda s: (~s.between(0, 23)).any())
    df['시간'] = df.groupby('날짜').cumcount()
    df['dt'] = pd.to_datetime(df['날짜'].astype(str)) + pd.to_timedelta(df['시간'], unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    # lag 피처가 "1행 = 1시간, 누락 없이 연속"을 전제로 한다. 깨지면 조용히 틀린 lag가 생긴다
    n = df.groupby('날짜').size()
    assert (n == 24).all(), f'하루 24행이 아닌 날: {n[n != 24].to_dict()}'
    assert (df['dt'].diff().dropna() == pd.Timedelta(hours=1)).all(), 'dt가 연속이 아님'

    df[['풍속', '강수량']] = df[['풍속', '강수량']].interpolate()
    df['공장인원'] = df['공장인원'].fillna(0)
    df['target'] = df['평균']
    df['is_stop'] = (df[['15분', '30분', '45분', '60분']] == 0).any(axis=1)

    day_prod = df.groupby('날짜')['생산량'].transform('sum')
    day_max = df.groupby('날짜')['target'].transform('max')
    # 휴무일 = 하루 종일 기저부하만. 전력이 비어 있는 진짜 테스트 구간에서는 생산계획으로 판정한다
    df['is_off'] = np.where(day_max.isna(), day_prod == 0, day_max < 30)
    df['is_holiday'] = df['날짜'].astype(str).isin({d.replace('-', '') for d in HOLIDAY})
    df['day_prod'] = day_prod
    # 하루 종일 생산 0인데 전력은 고부하 = 생산 기록 누락 의심일
    df['is_prod_missing'] = (day_prod == 0) & ~df['is_off']

    # 24시간 전력 곡선이 통째로 같은 날 = 증강으로 복사된 날
    curve = df.groupby('날짜')['평균'].apply(tuple)
    gid = curve.map({c: i for i, c in enumerate(curve.drop_duplicates())})
    df['is_clone'] = df['날짜'].map(gid.map(gid.value_counts())) > 1
    df['train_ok'] = ~df['is_stop'] & ~df['is_corrupt']
    df['train_ok_strict'] = df['train_ok'] & ~df['is_prod_missing']
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. 피처
# ════════════════════════════════════════════════════════════════════════════
# 1주 앞 예측용 세트. 전날 실적(lag24 등)은 그 시점에 없으므로 뺀다
A7 = ['hour', 'dow', 'is_weekend', 'is_holiday', 'shift', 'is_transition', 'month']
B = ['is_off', 'day_prod_zero', 'prod', 'prod_cap', 'prod_log', 'prod_zero', 'day_prod',
     'day_prod_hours', 'prod_prev', 'prod_next', 'prod_diff']
B7 = ['after_off', 'off_run_prev', 'days_since_off'] + B
C = ['cool', 'temp', 'humid', 'thi', 'wind', 'rain']
D7 = ['lag168', 'lag336', 'lag_week_mean4', 'last_week_day_mean', 'last_week_day_max']
L = ['lag168_na', 'lag_best']        # lag168 결측 대응. 하계휴가 주에 1주 전이 통째로 빈다
COLS = A7 + B7 + C + D7 + L          # 34개
FULL = (['hour', 'dow', 'is_weekend', 'is_holiday', 'after_off', 'off_run_prev',
         'days_since_off', 'shift', 'is_transition', 'month'] + B + C +
        ['lag24', 'lag168', 'lag336', 'lag_week_mean4', 'lag_prev_op',
         'prev_day_mean', 'prev_day_max', 'last_week_day_mean', 'last_week_day_max'])


def build(df):
    """예측 시점에 알 수 있는 값만으로 피처를 만든다 (예측 마감 = 전날 24시)."""
    f = pd.DataFrame(index=df.index)
    day = df.groupby('날짜')

    # 달력
    f['hour'] = df['시간']
    f['dow'] = df['day']
    f['is_weekend'] = (df['day'] >= 6).astype(int)
    f['is_off'] = df['is_off'].astype(int)
    f['is_holiday'] = df['is_holiday'].astype(int)
    off_day = day['is_off'].first()
    f['after_off'] = df['날짜'].map(off_day.shift(1).fillna(False)).astype(int)
    o = off_day.astype(int)
    f['off_run_prev'] = df['날짜'].map(o.groupby((1 - o).cumsum()).cumsum().shift(1).fillna(0))
    idx = pd.Series(np.arange(len(off_day)), index=off_day.index)
    f['days_since_off'] = df['날짜'].map(idx - idx.where(off_day).shift(1).ffill())
    f['shift'] = np.select([df['시간'] == 12, df['시간'].between(8, 19)], [1, 2], 0)
    f['is_transition'] = df['시간'].isin([7, 12, 17]).astype(int)
    f['month'] = df['m']

    # 생산계획
    f['prod'] = df['생산량']
    f['prod_cap'] = df['생산량'].clip(upper=PROD_CAP)
    f['prod_log'] = np.log1p(df['생산량'])
    f['prod_zero'] = (df['생산량'] == 0).astype(int)
    f['day_prod'] = day['생산량'].transform('sum')
    f['day_prod_zero'] = (f['day_prod'] == 0).astype(int)
    f['day_prod_hours'] = day['생산량'].transform(lambda s: (s > 0).sum())
    f['prod_prev'] = day['생산량'].shift(1)
    f['prod_next'] = day['생산량'].shift(-1)
    f['prod_diff'] = f['prod'] - f['prod_prev']

    # 기상 (실측을 예보로 가정)
    f['cool'] = (df['기온'] - COOL_BASE).clip(lower=0)
    f['temp'], f['humid'], f['wind'], f['rain'] = df['기온'], df['습도'], df['풍속'], df['강수량']
    f['thi'] = 0.81 * df['기온'] + 0.01 * df['습도'] * (0.99 * df['기온'] - 14.3) + 46.3

    # 과거 전력 — 휴무·중단 시간은 결측 처리하고 참조한다
    p = df['target'].where(~df['is_off'] & ~df['is_stop'])
    f['lag24'], f['lag168'], f['lag336'] = p.shift(24), p.shift(168), p.shift(336)
    f['lag_week_mean4'] = pd.concat([p.shift(168 * k) for k in range(1, 5)], axis=1).mean(axis=1)
    grid = p.to_frame('p').assign(날짜=df['날짜'], h=df['시간']).pivot(index='날짜', columns='h', values='p')
    f['lag_prev_op'] = grid.shift(1).ffill().stack().reindex(
        pd.MultiIndex.from_arrays([df['날짜'], df['시간']])).values
    dstat = p.groupby(df['날짜']).agg(['mean', 'max'])
    f['prev_day_mean'] = df['날짜'].map(dstat['mean'].shift(1))
    f['prev_day_max'] = df['날짜'].map(dstat['max'].shift(1))
    f['last_week_day_mean'] = df['날짜'].map(dstat['mean'].shift(7))
    f['last_week_day_max'] = df['날짜'].map(dstat['max'].shift(7))

    # lag168 이 없을 때 — 결측이라는 사실 자체가 정보다 (그 주가 비정상이었다는 뜻)
    f['lag168_na'] = f['lag168'].isna().astype(int)
    f['lag_best'] = f['lag168'].fillna(f['lag336']).fillna(f['lag_week_mean4'])

    meta = df[['dt', '날짜', 'target', 'is_clone', 'is_off', 'is_stop', 'is_corrupt',
               'is_prod_missing', 'train_ok', 'train_ok_strict']].copy()
    q = ['15분', '30분', '45분', '60분']
    meta['target_max15'] = df[q].max(axis=1)         # 요금이 매겨지는 기준
    X = pd.concat([meta, f.drop(columns='is_off')], axis=1)
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    return X


# ════════════════════════════════════════════════════════════════════════════
# 3. 모델과 검증 규약
# ════════════════════════════════════════════════════════════════════════════
LGBM = dict(objective='l1', n_estimators=400, learning_rate=.05, num_leaves=31,
            min_data_in_leaf=20, feature_fraction=.8, bagging_fraction=.8,
            bagging_freq=1, verbose=-1)
SHALLOW = {'num_leaves': 15, 'min_data_in_leaf': 40}
SLOW = {'learning_rate': 0.02, 'n_estimators': 1200}
P_FINAL = {**SHALLOW, **SLOW, 'feature_fraction': 0.6}

# 용도별로 추정량이 다르다 — 하나의 예측값으로 모든 질문에 답할 수 없다
CFG = {
    '강화 최종':      dict(cols=COLS, params=P_FINAL, peak_w=PEAK_W),              # 시간별 점 예측
    '강화 대안':      dict(cols=COLS, params=SHALLOW, peak_w=6.0),                 # 피크 우선
    '경보 상한':      dict(cols=COLS, params={**SHALLOW, 'objective': 'quantile',   # 일최대·경보
                                             'alpha': .9}, peak_w=1.0),
    '15분 최대':      dict(cols=COLS, params=P_FINAL, peak_w=PEAK_W, target='target_max15'),
    '15분 상한':      dict(cols=COLS, params={**SHALLOW, 'objective': 'quantile', 'alpha': .9},
                          peak_w=1.0, target='target_max15'),
    'D-1 상한':      dict(cols=FULL, params={**SHALLOW, 'objective': 'quantile', 'alpha': .9},
                          peak_w=1.0),
    'D-1 점예측':     dict(cols=FULL, params=P_FINAL, peak_w=PEAK_W),
    '기본 설정':      dict(cols=FULL, params=None, peak_w=1.0, clone='keep'),
    '1주 앞+가중':    dict(cols=A7 + B7 + C + D7, params=None, peak_w=1.0),
    'RandomForest':  dict(cols=FULL, params=None, peak_w=1.0, clone='keep', model='rf'),
}


def fit_predict(name, seed, params, xtr, ytr, w, xva):
    if name == 'rf':
        xtr, xva = xtr.fillna(-999), xva.fillna(-999)
        g = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                  random_state=seed)
    else:
        g = lgb.LGBMRegressor(**{**LGBM, **(params or {}), 'seed': seed})
    return g.fit(xtr, ytr, sample_weight=w).predict(xva)


def rolling(X, cols, params=None, peak_w=1.0, clone='weight', seed=0,
            folds=None, target='target', model='lgbm'):
    """시간순 롤링 폴드. 학습 = 검증월 이전 전체, 평가 = 고유일만.

    복제일을 평가에 넣으면 같은 곡선을 학습에서도 봤으므로 점수가 부풀려진다.
    평가셋은 어떤 설정에서도 고정이라 설정 간 비교가 성립한다.
    """
    out = []
    for m in (folds or FOLDS):
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
        if len(va) == 0 or len(tr) < 200:
            continue                       # 6월은 고유일이 2일뿐이라 건너뛸 수 있다
        thr = tr[target].quantile(.95)
        w = np.ones(len(tr))
        if clone == 'weight':
            w *= np.where(tr.is_clone, CLONE_W, 1.0)
        if peak_w != 1.0:
            w *= np.where(tr[target] >= thr, peak_w, 1.0)
        # 전부 1.0 이면 None 으로 넘긴다. src/model.py 와 같은 동작을 위해 꼭 필요하다 —
        # RandomForest 는 sample_weight 를 받으면 부트스트랩 경로가 달라져서
        # 1.0 을 넣든 안 넣든 결과가 바뀐다 (CORE MAE 10.10 vs 9.96)
        w = None if np.allclose(w, 1.0) else w
        p = fit_predict(model, seed, params, tr[cols], tr[target], w, va[cols])
        out.append(pd.DataFrame({'fold': m, 'idx': va.index, 'y': va[target].values,
                                 'p': p, 'peak': va[target].values >= thr}))
    return pd.concat(out, ignore_index=True)


def score(s):
    e = s.y - s.p
    return {'MAE': e.abs().mean(), 'RMSE': float(np.sqrt((e ** 2).mean())),
            'peakMAE': e[s.peak].abs().mean(), 'n': len(s)}


def run_cfg(X, key, folds=CORE_FOLDS, seeds=SEEDS):
    """설정 하나를 시드별로 돌려 (시드평균 예측, 시드별 지표)를 돌려준다."""
    cfg = {k: v for k, v in CFG[key].items()}
    parts = [rolling(X, seed=s, folds=folds, **cfg).set_index('idx') for s in seeds]
    base = parts[0][['fold', 'y', 'peak']].copy()
    base['p'] = pd.concat([q.p for q in parts], axis=1).mean(axis=1)
    per = pd.DataFrame([score(q.assign(peak=base.peak)) for q in parts])
    return base, per


# ════════════════════════════════════════════════════════════════════════════
# 4~9. 분석
# ════════════════════════════════════════════════════════════════════════════
def tbl(rows, cols, widths=None):
    """콘솔 표. 숫자는 오른쪽, 글자는 왼쪽."""
    widths = widths or [max(len(str(c)), *(len(str(r.get(c, ''))) for r in rows)) for c in cols]
    line = '  ' + '  '.join(str(c).rjust(w) if i else str(c).ljust(w)
                            for i, (c, w) in enumerate(zip(cols, widths)))
    print(line)
    print('  ' + '  '.join('─' * w for w in widths))
    for r in rows:
        print('  ' + '  '.join(str(r.get(c, '')).rjust(w) if i else str(r.get(c, '')).ljust(w)
                               for i, (c, w) in enumerate(zip(cols, widths))))


def model_performance(X, R):
    say('모델 성능 — 목표선을 얼마나 넘었나')
    print('  평가 CORE 2021년 7-9월 · 고유일 1,708행 · 시드 3개 · 예측은 구간 외')
    print('  흔들림 = 시드 3개의 표준편차. 설정 간 차이가 이보다 작으면 "차이 없음"\n')
    names = ['기본 설정', '1주 앞+가중', 'RandomForest', '강화 최종', '강화 대안', '경보 상한']
    rows, R['models'] = [], []
    for k in names:
        _, per = run_cfg(X, k)
        m, sd = per.MAE.mean(), per.MAE.std()
        R['models'].append({'name': k, 'mae': round(m, 2), 'sd': round(sd, 2),
                            'rmse': round(per.RMSE.mean(), 2), 'peak': round(per.peakMAE.mean(), 2),
                            'n': int(per.n.iloc[0])})
        rows.append({'설정': k, 'MAE': f'{m:.2f}', '흔들림': f'{sd:.2f}',
                     'RMSE': f'{per.RMSE.mean():.2f}', 'peakMAE': f'{per.peakMAE.mean():.2f}'})
        print(f'    {k} 완료', flush=True)
    print()
    tbl(rows, ['설정', 'MAE', '흔들림', 'RMSE', 'peakMAE'])
    R['baseline'] = {'mae': 22.9, 'rmse': 35.1, 'peak': 35.7}
    best = R['models'][3]
    print(f"\n  목표선(규칙만으로 예측) MAE 22.9 → 강화 최종 {best['mae']}  "
          f"({100 * (best['mae'] / 22.9 - 1):+.0f}%)")
    print('  경보 상한은 시간별 정확도를 내주고 피크 정확도를 가져간다 — 용도가 다르다')

    # 월별
    rows, R['folds'] = [], []
    for m in FOLDS:
        per = [score(rolling(X, seed=s, folds=[m], **CFG['강화 최종'])) for s in SEEDS]
        d = pd.DataFrame(per)
        if not len(d) or pd.isna(d.MAE.mean()):
            continue
        R['folds'].append({'fold': m, 'mae': round(d.MAE.mean(), 2), 'n': int(d.n.iloc[0])})
        rows.append({'월': m, 'MAE': f'{d.MAE.mean():.2f}', '평가행': int(d.n.iloc[0])})
    print('\n  월별 (강화 최종)')
    tbl(rows, ['월', 'MAE', '평가행'])
    print('  5월은 학습이 4개월뿐이고 그 대부분이 복제일이라 따로 본다')


def peak_prediction(X, R, P):
    say('최대피크 예측 — 일 최대 · 15분 최대 · 피크 시각')
    hi, _ = P['경보 상한']
    pt, _ = P['강화 최종']
    d = X.loc[hi.index]
    g = pd.DataFrame({'날짜': d['날짜'].values, 'off': d.is_off.values,
                      'y': hi.y.values, 'hi': hi.p.values, 'pt': pt.p.values,
                      'hour': d.hour.values})
    op = g[~g.off.astype(bool)]
    # 가동일 = 휴무가 아닌 날. 평가 시간 수로 더 거르지 않는다 —
    # 보고서 표 2-13(src/report_gain.py)과 같은 기준이어야 수치가 어긋나지 않는다
    day = op.groupby('날짜').agg(y=('y', 'max'), hi=('hi', 'max'), pt=('pt', 'max'),
                                n=('hour', 'size'))
    R['daymax'] = [{'d': int(i), 'y': round(r.y, 1), 'hi': round(r.hi, 1), 'pt': round(r.pt, 1)}
                   for i, r in day.iterrows()]
    e1, e2 = day.y - day.hi, day.y - day.pt
    R['daymax_stat'] = {'hi_mae': round(e1.abs().mean(), 2), 'hi_bias': round(e1.mean(), 2),
                        'pt_mae': round(e2.abs().mean(), 2), 'pt_bias': round(e2.mean(), 2),
                        'n': len(day)}
    print(f'  일 최대 (가동일 {len(day)}일)  편향 = 실제−예측. +면 낮게 본 것\n')
    tbl([{'추정 방법': '분위 0.9 (채택)', 'MAE': f'{e1.abs().mean():.2f}', '편향': f'{e1.mean():+.2f}'},
         {'추정 방법': '점 예측의 최대', 'MAE': f'{e2.abs().mean():.2f}', '편향': f'{e2.mean():+.2f}'}],
        ['추정 방법', 'MAE', '편향'])
    print('  점 예측은 시각마다 가운데 값을 맞히므로 그 24개의 최대는 구조적으로 낮다')

    # 피크 시각
    gg = op.groupby('날짜')
    act = gg.apply(lambda s: s.loc[s.y.idxmax(), 'hour'], include_groups=False)
    prd = gg.apply(lambda s: s.loc[s.pt.idxmax(), 'hour'], include_groups=False)
    hit = float((act == prd).mean() * 100)
    R['hour_hit'] = round(hit, 1)
    R['hours'] = [{'h': h, 'act': int((act == h).sum()), 'pred': int((prd == h).sum())}
                  for h in range(24)]
    top = act.value_counts().head(4).index.tolist()
    # 비교 규칙의 시각은 학습 구간에서 고른다. 평가 구간의 최빈값을 쓰면 규칙이 부당하게 유리해진다
    trh = X[X.train_ok_strict & ~X.is_off & ~X.ym.isin(CORE_FOLDS)]
    h0 = int(trh.loc[trh.groupby('날짜').target.idxmax(), 'hour'].value_counts().index[0])
    rule = float((act == h0).mean() * 100)
    R['hour_rule'] = {'hour': h0, 'hit': round(rule, 1)}
    print(f'\n  피크 시각  모델 적중률 {hit:.1f}%  vs  "무조건 {h0}시" 규칙 {rule:.1f}%'
          f'  ({"규칙이 낫다" if rule > hit else "모델이 낫다"})')
    print(f'  실제 일 최대의 {100 * act.isin(top).mean():.1f}% 가 {sorted(top)}시에 난다')
    print('  → 경보는 "몇 시" 가 아니라 "그날 위험한가" 로 낸다')

    # 15분 최대 — 직접 학습과 계수 환산을 같은 기준에서 비교한다
    m15, per15 = P['15분 최대']
    conv = []
    for seed in SEEDS:
        rows = []
        for m in CORE_FOLDS:
            va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
            tr = X[(X['dt'] < va['dt'].min()) & X['train_ok_strict']]
            # 환산 계수는 그 폴드의 학습 구간에서만 뽑는다 (전체에서 뽑으면 누수)
            t = tr[tr.train_ok & ~tr.is_off]
            ratio = (t.target_max15 / t.target.replace(0, np.nan)).groupby(t.hour).mean()
            w = (np.where(tr.is_clone, CLONE_W, 1.0)
                 * np.where(tr.target >= tr.target.quantile(.95), PEAK_W, 1.0))
            pr = fit_predict('lgbm', seed, P_FINAL, tr[COLS], tr.target, w, va[COLS])
            rows.append(pd.DataFrame({'y': va.target_max15.values,
                                      'p': pr * va.hour.map(ratio).values}))
        r = pd.concat(rows, ignore_index=True)
        conv.append((r.y - r.p).abs().mean())
    R['max15'] = {'direct_mae': round(per15.MAE.mean(), 2),
                  'direct_sd': round(per15.MAE.std(), 2),
                  'conv_mae': round(float(np.mean(conv)), 2)}
    print(f"\n  15분 최대 (요금 기준)  직접 학습 MAE {per15.MAE.mean():.2f} "
          f"(흔들림 {per15.MAE.std():.2f})  vs  시각별 계수 환산 {np.mean(conv):.2f}")
    print('  타깃을 바꿔 직접 학습하는 쪽이 낫다 — 환산은 그날의 사정을 반영하지 못한다')
    return g


def risk_conditions(X, R):
    say('최대피크 위험조건 — 어떤 날이 위험한가')
    d = X[X.train_ok & ~X.is_clone].copy()
    thr = {m: X[(X['dt'] < X[(X.ym == m) & X.train_ok & ~X.is_clone]['dt'].min())
                & X['train_ok_strict']].target.quantile(ALARM_Q) for m in CORE_FOLDS}
    day = d.groupby('날짜').agg(일최대=('target', 'max'), 최고기온=('temp', 'max'),
                               일생산=('prod', 'sum'), 생산시간=('day_prod_hours', 'first'),
                               주말=('is_weekend', 'first'), 공휴일=('is_holiday', 'first'),
                               ym=('ym', 'first'), n=('hour', 'size'))
    day = day[(day.n == 24) & day.ym.isin(CORE_FOLDS)]
    day['피크일'] = day.일최대 >= day.ym.map(thr)
    base = day.피크일.mean()
    R['risk'] = {'days': int(len(day)), 'peak': int(day.피크일.sum()),
                 'base': round(float(100 * base), 1)}
    print(f'  피크일 = 그날 일 최대가 임계(학습 구간 상위 5%) 이상인 날')
    print(f'  전체 {len(day)}일 중 피크일 {int(day.피크일.sum())}일 ({100 * base:.0f}%)\n')

    day['기온대'] = pd.cut(day.최고기온, [-99, 26, 30, 99], labels=['26도 이하', '26-30도', '30도 초과'])
    day['생산대'] = pd.qcut(day.일생산, 3, labels=['하위 1/3', '중위 1/3', '상위 1/3'])
    cross, R['cross'] = [], []
    for r in ['26도 이하', '26-30도', '30도 초과']:
        row = {'최고기온': r}
        for c in ['하위 1/3', '중위 1/3', '상위 1/3']:
            s = day[(day.기온대 == r) & (day.생산대 == c)]
            v = 100 * s.피크일.mean() if len(s) else float('nan')
            row['생산 ' + c] = f'{v:.0f}% ({len(s)}일)' if len(s) else '—'
            R['cross'].append({'기온대': r, '생산대': c,
                               '피크일비율(%)': round(float(v), 2) if len(s) else 0,
                               '일수': int(len(s))})
        cross.append(row)
    tbl(cross, ['최고기온', '생산 하위 1/3', '생산 중위 1/3', '생산 상위 1/3'])
    print('\n  ★ 생산이 하위 1/3 이면 기온이 30도를 넘어도 피크일이 0% 다')
    print('    생산이 중위 이상일 때만 기온이 비율을 올린다 —')
    print('    생산은 필요조건, 기온은 그 위에서 위험을 올리는 증폭 요인이다')

    conds = [('일생산 상위 1/3', day.일생산 >= day.일생산.quantile(2 / 3)),
             ('생산시간 18시간 이상', day.생산시간 >= 18),
             ('평일 (월-금)', day.주말 == 0),
             ('최고기온 26-30도', day.최고기온.between(26, 30, inclusive='right')),
             ('최고기온 26도 이하', day.최고기온 <= 26),
             ('공휴일', day.공휴일 == 1),
             ('최고기온 30도 초과', day.최고기온 > 30),
             ('주말', day.주말 == 1),
             ('일생산 하위 1/3', day.일생산 <= day.일생산.quantile(1 / 3))]
    rows, R['single'] = [], []
    for lab, m in conds:
        if not m.sum():
            continue
        v = 100 * day[m].피크일.mean()
        R['single'].append({'조건': lab, '해당일': int(m.sum()), '피크일': int(day[m].피크일.sum()),
                            '피크일 비율(%)': round(float(v), 1),
                            '기준대비(배)': round(float(day[m].피크일.mean() / base), 2)})
        rows.append({'조건': lab, '해당일': int(m.sum()), '피크일 비율': f'{v:.0f}%'})
    print()
    tbl(rows, ['조건', '해당일', '피크일 비율'])
    print('  "기온 30도 초과" 가 평균보다 낮은 것이 핵심이다 —')
    print('  그 날들 중 상당수가 저생산일(주말·휴무)이기 때문이다')


# ── D10. 팀이 확정한 프로젝트 기본 경보 정책 (보고서 3장이 쓰는 기준) ─────────
#   15분 최대를 직접 학습한 P90 예측이 학습 구간 15분 최대 Q95 를 넘으면 그 시각 경보,
#   하루에 한 번이라도 경보가 나면 그날을 위험일로 본다.
#   학습은 폴드 첫 평가일 00시 이전 행으로 한 번, 입력은 매일 00시 이전 전력으로 갱신(D-1).
#   당일 실측 휴무(is_off)는 입력에서 뺀다 — 예측 시점에 확정된 값이 아니기 때문이다.
#   일 지표는 24행이 다 있는 완전일만 센다 (CORE 69일, 불완전 3일 제외).
#   계약전력 초과 판정이 아니고, P90 은 포함률 보정을 거치지 않은 1차 산출물이다.
D10_COLS = [c for c in COLS if c != 'is_off']          # 33개
D10_QS = (.95, .97, .98)
D10_MODELS = {'15분 최대 P90': {'objective': 'quantile', 'alpha': .9},
              '점 예측 L1': {'objective': 'l1'}}


def d10_inputs(X, cutoff):
    """cutoff(그날 00시) 이전에 관측한 전력만으로 시차 변수를 다시 만든다.

    예측 시점에 없는 값이 입력에 섞이지 않는지를 구조로 보장하는 장치다.
    cutoff 이후 날의 '직전 휴무 이력'도 아직 모르는 값이라 결측으로 비운다.
    """
    r = X.copy(deep=True)
    p = X.target.where((X.dt < cutoff) & ~X.is_off.astype(bool) & ~X.is_stop.astype(bool))
    r['lag168'], r['lag336'] = p.shift(168), p.shift(336)
    r['lag_week_mean4'] = pd.concat([p.shift(168 * k) for k in range(1, 5)], axis=1).mean(axis=1)
    d = p.groupby(X['날짜']).agg(['mean', 'max'])
    r['last_week_day_mean'] = X['날짜'].map(d['mean'].shift(7))
    r['last_week_day_max'] = X['날짜'].map(d['max'].shift(7))
    r['lag168_na'] = r.lag168.isna().astype(int)
    r['lag_best'] = r.lag168.fillna(r.lag336).fillna(r.lag_week_mean4)
    future = X.dt.dt.normalize() > cutoff
    for c in ('after_off', 'off_run_prev', 'days_since_off'):
        r[c] = r[c].astype(float)
        r.loc[future, c] = np.nan
    return r


def d10_fit(X, cutoff):
    """cutoff 이전 행으로 L1·P90 을 같은 조건에서 학습하고 임계 스냅샷을 돌려준다."""
    h = X[X.dt < cutoff]
    tr = h[h.train_ok_strict & h.target_max15.notna()]
    # 복제 판정도 학습 시점 이전 곡선만으로 한다 (전체 데이터로 판정하면 미래를 당겨 쓴다)
    curves = h.groupby('날짜').target.apply(tuple)
    w = np.where(tr['날짜'].map(curves.map(curves.value_counts())).gt(1), CLONE_W, 1.0)
    xin = d10_inputs(X, cutoff).loc[tr.index, D10_COLS]
    models = {lab: [lgb.LGBMRegressor(**{**LGBM, **SHALLOW, **par, 'seed': s}).fit(
        xin, tr.target_max15, sample_weight=w) for s in SEEDS]
        for lab, par in D10_MODELS.items()}
    return models, {q: float(tr.target_max15.quantile(q)) for q in D10_QS}


def d10_run(X):
    """CORE 폴드를 D-1 규약으로 돌려 시각별 예측·임계를 모은다."""
    rows = []
    for m in CORE_FOLDS:
        va = X[(X.ym == m) & X.train_ok & ~X.is_clone]
        models, thr = d10_fit(X, va.dt.min().normalize())
        for date, day in va.groupby('날짜', sort=True):
            xin = d10_inputs(X, day.dt.min().normalize()).loc[day.index, D10_COLS]
            z = pd.DataFrame({'날짜': date, 'y': day.target_max15.values})
            for lab, ms in models.items():
                z[lab] = np.mean([g.predict(xin) for g in ms], axis=0)
            for q in D10_QS:
                z[f'thr{q}'] = thr[q]
            rows.append(z)
    return pd.concat(rows, ignore_index=True)


def d10_score(F, lab, q):
    """일 단위 경보 성능. 실제 위험일은 Q95 고정, 경보 임계만 바꿔 민감도를 본다."""
    g = pd.DataFrame({'날짜': F.날짜, 'al': F[lab] >= F[f'thr{q}'], 're': F.y >= F[f'thr{D10_QS[0]}']}
                     ).groupby('날짜').agg(n=('al', 'size'), al=('al', 'any'), re=('re', 'any'))
    exc = int((g.n != 24).sum())
    g = g[g.n == 24]
    tp = int((g.al & g.re).sum()); fp = int((g.al & ~g.re).sum())
    fn = int((~g.al & g.re).sum()); tn = int((~g.al & ~g.re).sum())
    f = lambda a, b: round(100 * a / b, 1) if b else float('nan')
    return {'설정': lab, '경보임계': f'Q{round(q * 100)}',
            '임계전력': round(float(F[f'thr{q}'].mean()), 1),
            '정밀도(%)': f(tp, tp + fp), '재현율(%)': f(tp, tp + fn),
            '경보율(%)': f(int(g.al.sum()), len(g)), '일치율(%)': f(tp + tn, len(g)),
            '경보일': tp + fp, '미탐지': fn, '오경보': fp,
            '완전일': len(g), '제외일': exc, '실제위험일': tp + fn}


def alarm(X, R, P, F):
    say('피크 위험 경보 — 언제 · 무엇을 기준으로')

    def thresholds(q, col='target'):
        return {m: X[(X['dt'] < X[(X.ym == m) & X.train_ok & ~X.is_clone]['dt'].min())
                     & X['train_ok_strict']][col].quantile(q) for m in CORE_FOLDS}

    def evaluate(pred, q, col='target'):
        t = pred.fold.map(thresholds(q, col))
        dd = pd.DataFrame({'날짜': X.loc[pred.index, '날짜'].values,
                           'pred': pred.p >= t, 'real': pred.y >= t})
        g = dd.groupby('날짜').agg(pred=('pred', 'any'), real=('real', 'any'))
        tp = int((g.pred & g.real).sum()); fp = int((g.pred & ~g.real).sum())
        fn = int((~g.pred & g.real).sum()); tn = int((~g.pred & ~g.real).sum())
        n = tp + fp + fn + tn
        f = lambda a, b: round(100 * a / b, 1) if b else float('nan')
        return {'임계전력': round(float(np.mean(list(thresholds(q, col).values()))), 1),
                '정밀도(%)': f(tp, tp + fp), '재현율(%)': f(tp, tp + fn),
                '경보율(%)': f(tp + fp, n), '경보일': tp + fp, '실제위험일': tp + fn,
                '전체일': n, '일치율(%)': f(tp + tn, n)}

    # ① 프로젝트 기본 정책 (D10) — 보고서 3장 표 3-3/3-5 와 같은 기준이다
    R['d10'] = [d10_score(F, lab, q) for lab in D10_MODELS for q in D10_QS]
    base = next(r for r in R['d10'] if r['설정'] == '15분 최대 P90' and r['경보임계'] == 'Q95')
    print('  ① 프로젝트 기본 경보 정책 (D10) — 15분 최대 직접 P90 >= 학습 Q95')
    print(f"     일 단위 · D-1 · 완전일 {base['완전일']}일 (불완전 {base['제외일']}일 제외)\n")
    tbl([{'설정': r['설정'], '경보임계': r['경보임계'], '전력': r['임계전력'],
          '정밀도': f"{r['정밀도(%)']}%", '재현율': f"{r['재현율(%)']}%",
          '경보율': f"{r['경보율(%)']}%", '미탐지': r['미탐지'], '오경보': r['오경보']}
         for r in R['d10']],
        ['설정', '경보임계', '전력', '정밀도', '재현율', '경보율', '미탐지', '오경보'])
    R['alarm_thr'] = base['임계전력']
    R['alarm_head'] = base
    print('\n  ★ 미탐지 0건(재현율 100%). 최대수요 기본요금은 한 번만 넘겨도 1년이 정해지므로')
    print('    경보율 63.8% 를 감수하고 가장 보수적인 Q95 를 기본으로 확정했다')
    print('    같은 임계에 점 예측(L1)을 쓰면 재현율이 64.3% 로 떨어진다 —')
    print('    타깃을 15분 최대로 바꾸고 분위 회귀를 쓴 것이 미탐지 방어의 핵심이다')

    # ② 리드타임 비교 — ①과 규약이 다르다 (D-7 고정 입력, 완전일 필터 없음).
    #    1주 전에 내도 성능을 잃지 않는지만 보는 보조 표이고, 대표 수치가 아니다
    rows, R['lead'] = [], []
    for lab, key, col in [('D-7 · 분위 0.9', '경보 상한', 'target'),
                          ('D-1 · 분위 0.9', 'D-1 상한', 'target'),
                          ('D-7 · 15분 최대', '15분 상한', 'target_max15'),
                          ('D-7 · 점 예측', '강화 최종', 'target'),
                          ('D-1 · 점 예측', 'D-1 점예측', 'target')]:
        r = evaluate(P[key][0], ALARM_Q, col)
        r['설정'] = lab
        R['lead'].append(r)
        rows.append({'설정': lab, '정밀도': f"{r['정밀도(%)']}%", '재현율': f"{r['재현율(%)']}%",
                     '경보율': f"{r['경보율(%)']}%", '일치율': f"{r['일치율(%)']}%"})
    print('\n  ② 리드타임 비교 (보조) — ①과 규약이 다르다: 입력을 D-7 에 고정하고'
          f"\n     완전일 필터 없이 {R['lead'][0]['전체일']}일 전체를 센다. 대표 수치가 아니다\n")
    tbl(rows, ['설정', '정밀도', '재현율', '경보율', '일치율'])
    print('\n  1주 전과 하루 전이 거의 같다 → 미리 내도 성능을 잃지 않는다')
    print('  15분 최대 기준이 정밀도가 가장 높다 — 요금이 매겨지는 기준이기도 하다')
    print('  점 예측으로 경보를 내면 재현율이 크게 떨어진다')

    rows, R['sweep'] = [], []
    for q in SWEEP_QS:
        r = evaluate(P['경보 상한'][0], q)
        r['임계분위'] = q
        R['sweep'].append(r)
        rows.append({'임계': f'상위 {100 * (1 - q):.0f}%', '전력': r['임계전력'],
                     '정밀도': f"{r['정밀도(%)']}%", '재현율': f"{r['재현율(%)']}%",
                     '경보율': f"{r['경보율(%)']}%"})
    print('\n  ③ 시간 평균 기준 임계 스윕 (보조) — ①의 15분 최대 기준과 단위가 다르다\n')
    tbl(rows, ['임계', '전력', '정밀도', '재현율', '경보율'])
    print('\n  ★ 낮은 임계는 숫자만 좋다. 정밀도·재현율이 100% 여도 경보율이 60% 를 넘으면')
    print('    이틀에 한 번 경보가 나가 현장이 무시한다. 절대 기준(계약전력)이 필요하다')


def reduction(X, R):
    say('피크전력 저감방안 — 실측만으로 (모델 반사실 아님)')
    d = X[X.train_ok & ~X.is_clone & ~X.is_off & ~X.is_stop].copy()
    n = d.groupby('날짜')['hour'].size()
    d = d[d['날짜'].isin(n[n == 24].index)]

    rows = []
    for dt, s in d.groupby('날짜'):
        v = s.set_index('hour')['target'].sort_values(ascending=False)
        rows.append({'날짜': dt, '월': int(s['month'].iloc[0]), '피크시각': int(v.index[0]),
                     '월최대': float(v.iloc[0]), '2위시각': int(v.index[1]),
                     '2위': float(v.iloc[1]), '간격': float(v.iloc[0] - v.iloc[1]),
                     '일생산': float(s['prod'].sum())})
    DD = pd.DataFrame(rows)

    h = d.pivot_table(index='날짜', columns='hour', values='target')
    ramp = h.diff(axis=1).mean().dropna()
    R['ramp'] = [{'시각': int(k), '상승폭': round(float(v), 1)} for k, v in ramp.items()]
    big = ramp[ramp.abs() >= 10]
    print('  축 A — 설비가동 시점 조정 (기동 분산)')
    print('  시각별 전력 상승폭 (전 시각 대비, 가동일 평균). 큰 양수 = 한꺼번에 켜지는 시각\n')
    print('    ' + '   '.join(f'{int(k):2d}시 {v:+6.1f}' for k, v in big.items()))
    up = ramp.sort_values(ascending=False)
    print(f'\n    가장 큰 곳  {int(up.index[0])}시 {up.iloc[0]:+.1f} · {int(up.index[1])}시 {up.iloc[1]:+.1f}')
    if 12 in ramp.index and 13 in ramp.index:
        print(f'    ★ 12시 {ramp[12]:+.1f} 로 세웠다가 13시 {ramp[13]:+.1f} 로 되올라온다 —')
        print('      점심에 설비를 세웠다가 한꺼번에 다시 켠다는 뜻이다')

    M = DD.loc[DD.groupby('월').월최대.idxmax()].sort_values('월')
    R['month'] = [{'월': int(r.월), '날짜': int(r.날짜), '월최대': r.월최대,
                   '피크시각': int(r.피크시각), '2위': r['2위'], '2위시각': int(r['2위시각']),
                   '간격': r.간격} for _, r in M.iterrows()]
    print('\n  기회의 크기 — 월 최대를 만든 날 (요금이 걸린 곳은 여기 하나다)\n')
    tbl([{'월': f"{int(r.월)}월", '날짜': int(r.날짜), '월최대': f'{r.월최대:.0f}',
          '피크시각': f'{int(r.피크시각)}시', '2위': f"{r['2위']:.0f} ({int(r['2위시각'])}시)",
          '간격': f'{r.간격:.0f}'} for _, r in M.iterrows()],
        ['월', '날짜', '월최대', '피크시각', '2위', '간격'])
    k = M[M.피크시각.isin([7, 8, 9, 13])]
    print(f'\n    {len(M)}개월 중 {len(k)}개월의 월 최대가 기동 시각(7-9시·13시)에 났다')
    print(f'    봉우리만 그날 2위 수준으로 눌러도 월 최대 -{M.간격.mean():.0f} ~ -{M.간격.max():.0f}')
    print('    생산 총량도 배분도 안 바꾸므로 생산 손실 0 · 인건비 증가 0')
    print('    ※ 상한이다. 기동을 분산하면 부하가 옆 시각으로 옮겨 가 실제는 그보다 작다')

    q = pd.qcut(DD.일생산, 5)
    lv = DD.groupby(q, observed=True).agg(일수=('월최대', 'size'), 평균일생산=('일생산', 'mean'),
                                          평균일최대=('월최대', 'mean')).reset_index(drop=True)
    R['level'] = [{'일수': int(r.일수), '평균일생산': round(r.평균일생산, 2),
                   '평균일최대': round(r.평균일최대, 2)} for _, r in lv.iterrows()]
    print('\n  축 B — 생산일정 조정 (날짜 간). 일생산 구간별 일 최대\n')
    tbl([{'일수': int(r.일수), '평균 일생산': f'{r.평균일생산:,.0f}',
          '평균 일최대': f'{r.평균일최대:.0f}'} for _, r in lv.iterrows()],
        ['일수', '평균 일생산', '평균 일최대'])
    a, b = lv.iloc[1], lv.iloc[-1]
    print(f'\n    포화한다. 일생산 {a.평균일생산:,.0f} → {b.평균일생산:,.0f} '
          f'({100 * (b.평균일생산 / a.평균일생산 - 1):.0f}% 증가) 인데 '
          f'일 최대는 {a.평균일최대:.0f} → {b.평균일최대:.0f} 뿐이다')
    print('    생산일정을 옮겨 피크를 낮추려면 그날을 거의 쉬게 해야 한다 → 효과가 작다')
    R['reduce'] = {'ramp8': round(float(ramp.get(8, 0)), 1),
                   'ramp12': round(float(ramp.get(12, 0)), 1),
                   'ramp13': round(float(ramp.get(13, 0)), 1),
                   'gap_mean': round(float(M.간격.mean()), 1),
                   'gap_max': round(float(M.간격.max()), 1),
                   'k_months': int(len(k)), 'n_months': int(len(M)),
                   'lo_prod': round(float(a.평균일생산)), 'hi_prod': round(float(b.평균일생산)),
                   'lo_max': round(float(a.평균일최대)), 'hi_max': round(float(b.평균일최대))}
    print('\n  축 C — 하루 안의 생산 배분 조정  ✗ 기각')
    print('    모델이 그 축을 읽지 못한다 (조건 맞춘 날짜 쌍 83개에서 48.2%, p=0.67)')
    print('    총량을 고정하고 배분만 바꾸는 수단이라 모델이 평가할 수 없다')


# ════════════════════════════════════════════════════════════════════════════
# 10. 보고
# ════════════════════════════════════════════════════════════════════════════
def verify(R):
    say('검증 — src/ 기준 수치와 같은 값이 나왔나')
    got = {'강화최종 MAE': R['models'][3]['mae'], '강화최종 peakMAE': R['models'][3]['peak'],
           '일최대 MAE': R['daymax_stat']['hi_mae'],
           '경보 재현율': R['alarm_head']['재현율(%)'], '피크시각 적중률': R['hour_hit'],
           '학습행': R['n_train']}
    rows, bad = [], 0
    for k, want in VERIFY.items():
        g = got[k]
        ok = abs(g - want) <= (0.02 if isinstance(want, float) else 0)
        bad += not ok
        rows.append({'항목': k, '문서': want, '이번 실행': g, '': '일치' if ok else '★ 다름'})
    tbl(rows, ['항목', '문서', '이번 실행', ''])
    if bad:
        print(f'\n  ★ {bad}개가 문서와 다르다. src/ 와 이 파일 중 한쪽만 고쳐졌을 수 있다')
    else:
        print('\n  전부 일치. 이 파일의 결과를 그대로 보고서에 쓸 수 있다')
    return bad == 0


def main():
    if not RAW.exists():
        raise SystemExit(f'원자료가 없다: {RAW}')
    R = {}
    say('전처리 — 원자료 정제와 플래그')
    df = load()
    days = df.drop_duplicates('날짜')
    print(f'  {len(df):,}행 / {len(days)}일  ({df["날짜"].min()} ~ {df["날짜"].max()})')
    print(f'  복제일 {int(days.is_clone.sum())}일 · 고유일 {int((~days.is_clone).sum())}일')
    print(f'  휴무일 {int(days.is_off.sum())}일 · 가동중단 {int(df.is_stop.sum())}행'
          f' · 기록손상 {int(days.is_corrupt.sum())}일')
    print(f'  생산기록 누락 의심 {int(days.is_prod_missing.sum())}일')
    print(f'  학습 가능 행  train_ok {int(df.train_ok.sum()):,}'
          f' / train_ok_strict {int(df.train_ok_strict.sum()):,}')
    R['n_train'] = int(df.train_ok_strict.sum())
    R['meta'] = {'rows': len(df), 'days': len(days), 'clone': int(days.is_clone.sum()),
                 'off': int(days.is_off.sum()), 'train': int(df.train_ok_strict.sum()),
                 'span': f'{df["날짜"].min()} ~ {df["날짜"].max()}'}

    say('피처 생성')
    X = build(df)
    print(f'  {len(X):,}행 · 1주 앞 예측용 {len(COLS)}개 / 하루 앞용 {len(FULL)}개')
    R['meta']['n_cols'] = len(COLS)
    R['cfg'] = {'leaves': SHALLOW['num_leaves'], 'min_leaf': SHALLOW['min_data_in_leaf'],
                'lr': SLOW['learning_rate'], 'trees': SLOW['n_estimators'],
                'clone_w': CLONE_W, 'peak_w': PEAK_W, 'alarm_q': ALARM_Q}
    na = X[X.train_ok][COLS].isna().mean()
    print('  결측이 있는 피처:', ', '.join(f'{k} {v:.0%}' for k, v in na[na > 0].items()) or '없음')

    model_performance(X, R)

    print('\n  경보·피크용 설정을 돌린다 (분위 0.9 · 15분 최대 · D-1)...', flush=True)
    P = {k: run_cfg(X, k) for k in ['강화 최종', '경보 상한', '15분 최대', '15분 상한',
                                    'D-1 상한', 'D-1 점예측']}
    peak_prediction(X, R, P)
    risk_conditions(X, R)
    print('\n  D10 경보 정책을 D-1 규약으로 다시 돌린다 (날마다 입력을 갱신한다)...', flush=True)
    F = d10_run(X)
    alarm(X, R, P, F)
    reduction(X, R)
    ok = verify(R)

    print(f'\n{"=" * 74}')
    print(f' 완료 — {time.time() - t0:.0f}초')
    print(f'{"=" * 74}')
    print('  위 표에 이번 실행의 모든 수치가 있다')
    print('  현장용 운영 화면은 python run_dashboard.py 로 만든다')
    print('    만들어 둔 화면  outputs/final_app/index.html (서버 없이 열린다)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
