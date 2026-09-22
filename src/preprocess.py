"""원본 로드 + 정제 + 복제일 판별. 모든 분석·학습은 이 모듈의 load()를 사용한다."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset' / 'okm_augumented_2021.csv'
CLEAN = ROOT / 'outputs' / 'eda' / 'clean_preview.csv'


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df['시간'] = df.groupby('날짜').cumcount()  # 7/13, 7/15 시간 컬럼 손상 → 행 순서로 복구
    df['dt'] = pd.to_datetime(df['날짜'].astype(str)) + pd.to_timedelta(df['시간'], unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df[['풍속', '강수량']] = df[['풍속', '강수량']].interpolate()
    df['공장인원'] = df['공장인원'].fillna(0)
    df['target'] = df['평균']
    # 가동 중단(8/28 17시 to 8/29 11시): 15분 값 중 하나라도 0인 시간, 학습에서 제외
    df['is_stop'] = (df[['15분', '30분', '45분', '60분']] == 0).any(axis=1)
    # 휴무일: 하루 종일 기저부하만(일 최대 30 미만, 30 to 100 사이 날은 없음). 예측 시점에 생산계획으로 알 수 있다고 가정
    df['is_off'] = df.groupby('날짜')['target'].transform('max') < 30

    # 24시간 전력 곡선이 다른 날과 완전히 같은 날 = 증강으로 복사된 날(원본 포함, 구분 불가)
    curve = df.groupby('날짜')['평균'].apply(tuple)
    gid = curve.map({c: i for i, c in enumerate(curve.drop_duplicates())})
    size = gid.map(gid.value_counts())
    df['clone_gid'] = df['날짜'].map(gid)
    df['clone_n'] = df['날짜'].map(size)
    df['is_clone'] = df['clone_n'] > 1
    return df


if __name__ == '__main__':
    df = load()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False, encoding='utf-8-sig')
    days = df.drop_duplicates('날짜')
    print('복제일', days.is_clone.sum(), '/ 고유일', (~days.is_clone).sum(),
          '/ 휴무일', days.is_off.sum(), '/ 가동 중단', df.is_stop.sum(), '행 →', CLEAN)
