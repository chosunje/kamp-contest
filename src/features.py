"""피처 생성 (피처목록.txt 기준). 예측 마감: 전날 24시 → 당일 전력은 쓰지 않고 lag 24 이상만 사용.
당일 생산계획·기상·휴무 여부는 전날 알 수 있다고 가정."""
import numpy as np, pandas as pd
from preprocess import load, ROOT

OUT = ROOT / 'outputs' / 'features.csv'
RATIO = ROOT / 'outputs' / 'peak_ratio.csv'
Q15 = ['15분', '30분', '45분', '60분']
PROD_CAP = 800   # 생산량 약 800에서 전력 포화
COOL_BASE = 22   # 가동 시간 기준 22°C 이상에서 전력 상승

# A는 달력과 "지난 휴무 이력"만 쓴다. 당일 휴무 여부(is_off)는 타깃에서 유도한 값이라
# 생산계획을 받아야 알 수 있으므로 B로 옮겼다 (no_plan 세트에 들어가면 모순)
A = ['hour', 'dow', 'is_weekend', 'is_holiday', 'after_off', 'off_run_prev', 'days_since_off',
     'shift', 'is_transition', 'month']
B = ['is_off', 'day_prod_zero', 'prod', 'prod_cap', 'prod_log', 'prod_zero', 'day_prod',
     'day_prod_hours', 'prod_prev', 'prod_next', 'prod_diff']
C = ['cool', 'temp', 'humid', 'thi', 'wind', 'rain']
D = ['lag24', 'lag168', 'lag336', 'lag_week_mean4', 'lag_prev_op',
     'prev_day_mean', 'prev_day_max', 'last_week_day_mean', 'last_week_day_max']
# 1주 앞(168시간) 예측용 세트 (근거: 작업내역(조선제).txt [11]). 예측 시점이 대상일 7일 전이라 최근 값이 존재하지 않는다.
#   빠지는 것  lag24, lag_prev_op, prev_day_mean, prev_day_max  (전날 실적 = 아직 미래)
#   옮기는 것  after_off, off_run_prev, days_since_off  → 대상일 직전 휴무 정보라
#              1주 앞 시점에는 생산계획이 있어야 알 수 있다 (달력군 → 계획군)
A7 = ['hour', 'dow', 'is_weekend', 'is_holiday', 'shift', 'is_transition', 'month']
B7 = ['after_off', 'off_run_prev', 'days_since_off'] + B
D7 = ['lag168', 'lag336', 'lag_week_mean4', 'last_week_day_mean', 'last_week_day_max']

FEATURES = {'full': A + B + C + D, 'no_plan': A + D,
            'h7_full': A7 + B7 + C + D7, 'h7_no_plan': A7 + D7}
META = ['dt', '날짜', 'target', 'target_max15', 'target_max15_trapz', 'is_clone', 'is_off',
        'is_stop', 'is_corrupt', 'is_prod_missing', 'train_ok', 'train_ok_strict']


def build() -> pd.DataFrame:
    df = load()
    f = pd.DataFrame(index=df.index)
    day = df.groupby('날짜')

    # A. 달력
    f['hour'] = df['시간']
    f['dow'] = df['day']
    f['is_weekend'] = (df['day'] >= 6).astype(int)
    f['is_off'] = df['is_off'].astype(int)
    f['is_holiday'] = df['is_holiday'].astype(int)   # 근거는 약함(공휴일 10일 중 8일 정상 가동), 추석 대비용
    off_day = day['is_off'].first()
    f['after_off'] = df['날짜'].map(off_day.shift(1).fillna(False)).astype(int)
    # 직전 연속 휴무 길이. 복귀일 일 최대는 휴무 1일 뒤 182, 9일 하계휴가 뒤 193으로
    # 일반 가동일 평균 165보다 높다 → 0/1 플래그로는 1일 휴무와 장기 휴가를 구분 못 함
    o = off_day.astype(int)
    run = o.groupby((1 - o).cumsum()).cumsum()
    f['off_run_prev'] = df['날짜'].map(run.shift(1).fillna(0))
    idx = pd.Series(np.arange(len(off_day)), index=off_day.index)
    f['days_since_off'] = df['날짜'].map(idx - idx.where(off_day).shift(1).ffill())
    f['shift'] = np.select([df['시간'] == 12, df['시간'].between(8, 19)], [1, 2], 0)  # 0 야간, 1 점심, 2 주간
    f['is_transition'] = df['시간'].isin([7, 12, 17]).astype(int)
    f['month'] = df['m']

    # B. 생산계획 (당일 안에서만 앞뒤 시간 참조)
    f['prod'] = df['생산량']
    f['prod_cap'] = df['생산량'].clip(upper=PROD_CAP)
    f['prod_log'] = np.log1p(df['생산량'])
    f['prod_zero'] = (df['생산량'] == 0).astype(int)
    f['day_prod'] = day['생산량'].transform('sum')
    # 휴무 49일은 전부 일 생산량 0 (포함관계 확인). 생산계획만으로 만드는 is_off 근사값이라
    # 타깃에서 유도한 is_off 대신 쓸 수 있다. 역은 성립 안 함: 일 생산량 0인 64일 중 15일은 고부하
    f['day_prod_zero'] = (f['day_prod'] == 0).astype(int)
    f['day_prod_hours'] = day['생산량'].transform(lambda s: (s > 0).sum())
    f['prod_prev'] = day['생산량'].shift(1)
    f['prod_next'] = day['생산량'].shift(-1)
    f['prod_diff'] = f['prod'] - f['prod_prev']

    # C. 기상 (실측을 예보로 가정)
    f['cool'] = (df['기온'] - COOL_BASE).clip(lower=0)
    f['temp'] = df['기온']
    f['humid'] = df['습도']
    f['thi'] = 0.81 * df['기온'] + 0.01 * df['습도'] * (0.99 * df['기온'] - 14.3) + 46.3
    f['wind'] = df['풍속']
    f['rain'] = df['강수량']

    # D. 과거 전력: 휴무일·가동 중단 값은 결측 처리 후 참조 (행 = 1시간, 누락 없이 연속)
    p = df['target'].where(~df['is_off'] & ~df['is_stop'])
    f['lag24'] = p.shift(24)
    f['lag168'] = p.shift(168)
    f['lag336'] = p.shift(336)
    f['lag_week_mean4'] = pd.concat([p.shift(168 * k) for k in range(1, 5)], axis=1).mean(axis=1)
    grid = p.to_frame('p').assign(날짜=df['날짜'], h=df['시간']).pivot(index='날짜', columns='h', values='p')
    prev_op = grid.shift(1).ffill()   # 전날까지 중 가장 최근 가동일의 같은 시각 값
    f['lag_prev_op'] = prev_op.stack().reindex(pd.MultiIndex.from_arrays([df['날짜'], df['시간']])).values
    dstat = p.groupby(df['날짜']).agg(['mean', 'max'])
    f['prev_day_mean'] = df['날짜'].map(dstat['mean'].shift(1))
    f['prev_day_max'] = df['날짜'].map(dstat['max'].shift(1))
    f['last_week_day_mean'] = df['날짜'].map(dstat['mean'].shift(7))
    f['last_week_day_max'] = df['날짜'].map(dstat['max'].shift(7))

    meta = df[['dt', '날짜', 'target', 'is_clone', 'is_off', 'is_stop', 'is_corrupt',
               'is_prod_missing', 'train_ok', 'train_ok_strict']].copy()
    # 15분 컬럼의 의미가 확정되지 않아(작업내역(조선제).txt [10]) 두 해석의 피크 타깃을 모두 만들어 둔다.
    # A안: 네 값이 각 15분 구간의 평균 → 그 최대가 곧 요금 기준 최대수요전력
    meta['target_max15'] = df[Q15].max(axis=1)
    # B안: 네 값이 :15 :30 :45 :60 순간값 → 인접 두 값의 평균으로 구간 평균을 추정한 뒤 최대
    #      첫 구간(0 to 15분)은 직전 시간의 60분 값을 0분 시점으로 쓴다
    prev60 = df['60분'].astype(float).shift(1)
    q = [prev60] + [df[c].astype(float) for c in Q15]
    meta['target_max15_trapz'] = pd.concat([(q[i] + q[i + 1]) / 2 for i in range(4)], axis=1).max(axis=1)
    return pd.concat([meta[META], f.drop(columns='is_off')], axis=1)


def peak_ratio(X: pd.DataFrame) -> pd.DataFrame:
    """시각별 (15분 최대 / 시간 평균) 비율. 시간 평균 예측값을 요금 기준 피크로 환산할 때 쓴다.
    A안 기준 7시 1.30, 17시 1.17, 5·12시 약 1.15로 교대 전환 시각에 크고 나머지는 약 1.05.
    15분 컬럼 해석(작업내역(조선제).txt [10])이 미확정이라 두 안을 모두 낸다. 0시와 12시 외에는 차이가 0.05 미만이다."""
    s = X[X.train_ok & ~X.is_off]
    base = s.target.replace(0, np.nan)
    return pd.DataFrame({'ratio_a': (s.target_max15 / base).groupby(s.hour).mean(),
                         'ratio_b': (s.target_max15_trapz / base).groupby(s.hour).mean()})


if __name__ == '__main__':
    X = build()
    X.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'{len(X)}행, 피처 full {len(FEATURES["full"])}개 / no_plan {len(FEATURES["no_plan"])}개 → {OUT}')
    r = peak_ratio(X)
    r.round(4).to_csv(RATIO, encoding='utf-8-sig')
    print('15분 최대 환산 계수 →', RATIO, '| A안 상위:', r.ratio_a.nlargest(4).round(2).to_dict())
    tr = X[X.train_ok]
    na = tr[FEATURES['full']].isna().mean()
    print('결측 비율(학습 가능 행 기준, 0 초과만)\n', na[na > 0].round(3).to_string())
    r = tr[FEATURES['full'] + ['target']].corr('spearman')['target'].drop('target')
    print('타깃과 스피어만 상관 (고유일, 절댓값 상위 12)')
    ru = X[X.train_ok & ~X.is_clone][FEATURES['full'] + ['target']].corr('spearman')['target'].drop('target')
    print(ru.reindex(ru.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
