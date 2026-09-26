"""원본 로드 + 정제 + 복제일 판별. 모든 분석·학습은 이 모듈의 load()를 사용한다."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset' / 'okm_augumented_2021.csv'
CLEAN = ROOT / 'outputs' / 'eda' / 'clean_preview.csv'

# 2021 법정공휴일 (대체공휴일 8/16·10/4·10/11 포함). 학습 구간은 9/14까지지만
# 테스트 구간이 추석(9/20 to 22)을 포함할 수 있어 연말까지 정의해 둔다.
HOLIDAY = {
    '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01',
    '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15', '2021-08-16',
    '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04',
    '2021-10-09', '2021-10-11', '2021-12-25',
}


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    # 7/13, 7/15: 시간 컬럼 손상 + 생산량 기록 누락 → 학습 제외 (전력값은 정상이라 lag 입력으로는 사용)
    df['is_corrupt'] = df.groupby('날짜')['시간'].transform(lambda s: (~s.between(0, 23)).any())
    df['시간'] = df.groupby('날짜').cumcount()
    df['dt'] = pd.to_datetime(df['날짜'].astype(str)) + pd.to_timedelta(df['시간'], unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    # lag 피처는 "1행 = 1시간, 누락 없이 연속"을 전제로 shift(24) 등을 쓴다. 전제가 깨지면 조용히
    # 틀린 lag가 만들어지므로 여기서 막는다 (테스트 데이터를 이어 붙일 때 특히 중요)
    n = df.groupby('날짜').size()
    assert (n == 24).all(), f'하루 24행이 아닌 날: {n[n != 24].to_dict()}'
    assert (df['dt'].diff().dropna() == pd.Timedelta(hours=1)).all(), 'dt가 1시간 간격으로 연속이 아님'

    df[['풍속', '강수량']] = df[['풍속', '강수량']].interpolate()
    df['공장인원'] = df['공장인원'].fillna(0)
    df['target'] = df['평균']
    # 가동 중단(8/28 17시 to 8/29 11시): 15분 값 중 하나라도 0인 시간, 학습에서 제외
    df['is_stop'] = (df[['15분', '30분', '45분', '60분']] == 0).any(axis=1)
    # 휴무일: 하루 종일 기저부하만(일 최대 30 미만). 실제 경계는 26 vs 97로 넓게 벌어져 있어 임계 30은 안전
    # ※ 타깃에서 유도한 값이라 예측 시점에는 모른다. 휴무 49일은 전부 일 생산량 0이므로(포함관계 확인)
    #   생산계획을 받으면 알 수 있다는 가정(D09). 계획이 없는 시나리오에서는 쓰면 안 됨 → features.py B군으로 분류
    df['is_off'] = df.groupby('날짜')['target'].transform('max') < 30
    df['is_holiday'] = df['날짜'].astype(str).isin(
        {d.replace('-', '') for d in HOLIDAY})
    # 하루 종일 생산량 0인데 일 최대전력이 기저부하를 한참 넘는 날 = 생산 기록 누락 의심.
    # 7/13, 7/15(is_corrupt)는 시간 컬럼까지 손상돼 우연히 발견된 것이고, 같은 성격의 날이 더 있다.
    day_prod = df.groupby('날짜')['생산량'].transform('sum')
    df['day_prod'] = day_prod
    df['is_prod_missing'] = (day_prod == 0) & ~df['is_off']

    # 24시간 전력 곡선이 다른 날과 완전히 같은 날 = 증강으로 복사된 날(원본 포함, 구분 불가)
    curve = df.groupby('날짜')['평균'].apply(tuple)
    gid = curve.map({c: i for i, c in enumerate(curve.drop_duplicates())})
    size = gid.map(gid.value_counts())
    df['clone_gid'] = df['날짜'].map(gid)
    df['clone_n'] = df['날짜'].map(size)
    df['is_clone'] = df['clone_n'] > 1
    df['train_ok'] = ~df['is_stop'] & ~df['is_corrupt']
    # 생산 기록 누락 의심일까지 뺀 엄격 버전. 기존 결과와 비교할 수 있도록 train_ok 는 그대로 두고 따로 둔다
    # ("생산 0인데 고부하"를 학습하면 생산량 피처의 기울기가 눌린다)
    df['train_ok_strict'] = df['train_ok'] & ~df['is_prod_missing']
    return df


if __name__ == '__main__':
    df = load()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False, encoding='utf-8-sig')
    days = df.drop_duplicates('날짜')
    print('복제일', days.is_clone.sum(), '/ 고유일', (~days.is_clone).sum(),
          '/ 휴무일', days.is_off.sum(), '/ 가동 중단', df.is_stop.sum(), '행',
          '/ 기록 손상', days.is_corrupt.sum(), '일 →', CLEAN)
    pm = days[days.is_prod_missing]
    print('생산 기록 누락 의심', len(pm), '일 (복제', int(pm.is_clone.sum()), '/ 고유', int((~pm.is_clone).sum()), ')',
          pm['날짜'].tolist())
    print('학습 가능 행  train_ok', int(df.train_ok.sum()), '/ train_ok_strict', int(df.train_ok_strict.sum()))
    h = days[days.is_holiday]
    print('공휴일', len(h), '일 중 실제 휴무', int(h.is_off.sum()), '일 →',
          h.loc[h.is_off, '날짜'].tolist(), '(나머지는 정상 가동)')
