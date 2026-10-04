"""보고서 제1장(데이터 이해 및 진단)에 넣을 표·그림용 데이터를 CSV로 만든다.

표와 그래프는 Word 기능으로 직접 그리므로, 여기서는 "그릴 값"만 내보낸다.
파일마다 어느 모집단에서 계산했는지가 중요하므로 outputs/report/ch1/캡션.txt 에
기준·캡션·배치 위치를 함께 적는다.

실행: python src/report_ch1.py  →  outputs/report/ch1/*.csv, 캡션.txt
"""
import numpy as np, pandas as pd

from features import build
from preprocess import load, ROOT

OUT = ROOT / 'outputs' / 'report' / 'ch1'
OUT.mkdir(parents=True, exist_ok=True)
WEEK = ['월', '화', '수', '목', '금', '토', '일']


def save(df, name, caption):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')   # Excel·Word 에서 한글이 깨지지 않게
    CAPS.append((name, caption))
    print(f'  → {name:28s} {len(df):3d}행')


CAPS = []


def main():
    R = load()
    X = build()
    ok = R[R.train_ok]
    uniq = ok[~ok.is_clone]

    # ── 표1. 변수 구성 ────────────────────────────────────────────────
    t1 = pd.DataFrame([
        ('전력', '15분·30분·45분·60분', '0 to 208', '15분 간격 측정값. 타깃과 같은 값이라 피처에서 제외'),
        ('전력', '평균', f"0 to {ok.target.max():.0f} (평균 {ok.target.mean():.1f})", '네 값의 산술평균. 예측 타깃'),
        ('생산', '생산량', f"0 to {R['생산량'].max():,.0f}, 43.1%가 0", '시간당 생산 실적'),
        ('생산', '공장인원', '0 to 48.4 (실수)', '생산량과 순위상관 0.994 → 설비 투입 비율로 해석'),
        ('생산', '인건비', '1.0 / 1.5', '9 to 17시 1.0, 그 외 1.5 → 주·야간 할증 계수'),
        ('기상', '기온', f"{R['기온'].min():.1f} to {R['기온'].max():.1f}℃", '22℃ 이상에서 냉방부하 발생'),
        ('기상', '풍속·습도·강수량', '-', '복제되지 않은 실측 기상'),
        ('날짜·단가', '날짜 / 시간', '20210101 to 20210914 / 0 to 23', '시간 컬럼은 7/13·7/15 손상 후 복구'),
        ('날짜·단가', 'day / d / m', '요일(1=월) / 일 / 월', '달력 정보'),
        ('날짜·단가', '전기요금(계절)', '109.8 / 167.2 / 191.6', '1 to 2월 / 3 to 5월·9월 / 6 to 8월. 시간대별 차등 없음'),
    ], columns=['변수군', '변수', '값 범위', '해석'])
    save(t1, 't1_변수구성.csv', '표. 변수 구성과 해석 (18개)')

    # ── 그림1. 시간대별 평균 전력 (가동일 vs 휴무일) ──────────────────
    # 가동일 = 일 생산량 5,000 초과(156일). 토요일·부분가동일을 섞으면 고부하 수준이 희석된다
    full = ok[ok.groupby('날짜')['생산량'].transform('sum') > 5000]
    f1 = pd.DataFrame({'시각': range(24),
                       '가동일 평균 전력': full.groupby('시간')['target'].mean().round(1).values,
                       '휴무일 평균 전력': ok[ok.is_off].groupby('시간')['target'].mean().round(1).values,
                       '가동일 평균 생산량': full.groupby('시간')['생산량'].mean().round(0).values})
    save(f1, 'f1_시간대별_전력.csv',
         '그림. 시간대별 평균 전력 — 꺾은선 (가동일 = 일 생산 5,000 초과 156일 / 휴무일 49일)')

    # ── 그림2. 전력값 분포 (3봉 구조) ─────────────────────────────────
    b = np.arange(0, 221, 10)
    c = pd.cut(ok.target, b, right=False)
    f2 = pd.DataFrame({'구간 시작': b[:-1], '구간': [f'{l} to {l+10}' for l in b[:-1]],
                       '시간 수': c.value_counts(sort=False).values})
    save(f2, 'f2_전력분포.csv', '그림. 전력 분포 — 세로 막대 (기저 23 / 중부하 105 / 고부하 165 세 덩어리)')

    # ── 그림3. 생산량 구간별 평균 전력 (포화) ─────────────────────────
    o = uniq[uniq['생산량'] > 0]
    bins = [0, 200, 400, 600, 800, 1000, 1200, 1600, 10000]
    lab = ['0 to 200', '200 to 400', '400 to 600', '600 to 800', '800 to 1000',
           '1000 to 1200', '1200 to 1600', '1600 초과']
    gg = o.groupby(pd.cut(o['생산량'], bins, labels=lab), observed=True)
    f3 = pd.DataFrame({'생산량 구간': lab,
                       '평균 생산량': gg['생산량'].mean().round(0).values,
                       '평균 전력': gg['target'].mean().round(1).values,
                       '시간 수': gg.size().values})
    save(f3, 'f3_생산량구간별_전력.csv', '그림. 생산량 구간별 평균 전력 — 세로 막대 (800 부근에서 포화)')

    # ── 그림4. 기온 구간별 평균 전력 (22℃ 임계) ──────────────────────
    tb = np.arange(-10, 34, 2)
    lab2 = [f'{t} to {t+2}' for t in tb[:-1]]
    run = uniq[uniq.target > 40]            # 가동 시간만 (무가동이 섞이면 관계가 가려진다)
    a = uniq.groupby(pd.cut(uniq['기온'], tb, labels=lab2), observed=False)['target'].mean()
    r = run.groupby(pd.cut(run['기온'], tb, labels=lab2), observed=False)['target'].mean()
    n = run.groupby(pd.cut(run['기온'], tb, labels=lab2), observed=False).size()
    f4 = pd.DataFrame({'기온 구간': lab2, '전체 시간 평균 전력': a.round(1).values,
                       '가동 시간 평균 전력': r.round(1).values, '가동 시간 수': n.values})
    f4 = f4[f4['가동 시간 수'] >= 30]
    save(f4, 'f4_기온구간별_전력.csv', '그림. 기온 구간별 평균 전력 — 꺾은선 (가동 시간 기준, n>=30)')

    # ── 표2a. 모집단별 상관 (지배 변수 역전) ★ ────────────────────────
    d = X[X.train_ok_strict]
    day = d.groupby(d.dt.dt.date).agg(prod=('day_prod', 'first'), mx=('target', 'max'),
                                      temp=('temp', 'mean'))
    opd = day[day['prod'] > 5000]
    # 계수 종류에 따라 결론이 달라지지 않는다는 것까지 보여야 한다 → 피어슨·스피어만 병기
    rows = []
    for name, g2 in [('일 단위 전체 (휴무일 포함)', day), ('일 단위 가동일만 (일 생산 5,000 초과)', opd)]:
        pe, sp = g2.corr(), g2.corr(method='spearman')
        rows.append((name, len(g2),
                     round(pe.loc['prod', 'mx'], 3), round(sp.loc['prod', 'mx'], 3),
                     round(pe.loc['temp', 'mx'], 3), round(sp.loc['temp', 'mx'], 3)))
    t2a = pd.DataFrame(rows, columns=['모집단', '일수',
                                      '일생산량-일최대전력 (피어슨)', '일생산량-일최대전력 (스피어만)',
                                      '일평균기온-일최대전력 (피어슨)', '일평균기온-일최대전력 (스피어만)'])
    save(t2a, 't2a_상관_모집단별.csv',
         '표. 모집단에 따른 상관 역전 ★ — 일 단위 집계값의 상관. 대상은 학습 기준 행(생산 기록 '
         '누락 의심일 제외) 242일, 가동일은 일 생산량 5,000 초과 156일. 두 계수 모두 같은 방향이다')

    # ── 표2b. 피크 시간 vs 고부하 시간 ★ ──────────────────────────────
    s = X[(X.dt >= '2021-07-01') & (X.dt < '2021-09-15') & X.train_ok]
    hi = s.nlargest(30, 'target')
    rest = s[(s.target > 135) & (~s.index.isin(hi.index))]
    hb = s[s.target > 135].copy()
    hb['is_day_max'] = hb.groupby(hb.dt.dt.date)['target'].transform('max') == hb.target
    rows = []
    for name, g2 in [('상위 30개 피크 시간', hi), ('나머지 고부하 시간 (135 이상)', rest),
                     ('[같은 날 비교] 일 최대가 난 시간', hb[hb.is_day_max]),
                     ('[같은 날 비교] 같은 날 다른 고부하 시간', hb[~hb.is_day_max])]:
        rows.append((name, len(g2), round(g2.target.mean(), 1), round(g2['prod'].mean()),
                     round(g2.temp.mean(), 1), round(g2.humid.mean(), 1)))
    t2b = pd.DataFrame(rows, columns=['구분', '시간 수', '평균 전력', '평균 생산량', '평균 기온(℃)', '평균 습도'])
    save(t2b, 't2b_피크시간_비교.csv', '표. 피크 시간은 생산이 더 적고 기온만 높다 ★ (7 to 9월)')

    # ── 그림5. 월별 복제일·고유일 구성 ★ ──────────────────────────────
    dd = R.groupby('날짜').agg(m=('m', 'first'), clone=('is_clone', 'first'))
    f5 = dd.groupby(['m', 'clone']).size().unstack(fill_value=0)
    f5 = pd.DataFrame({'월': f5.index, '복제일 수': f5.get(True, 0).values,
                       '고유일 수': f5.get(False, 0).values}).reset_index(drop=True)
    save(f5, 'f5_월별_복제구성.csv', '그림. 월별 복제일·고유일 구성 — 누적 세로 막대 ★ (1 to 4월이 거의 전부 복제)')

    # ── 표3. 증강 데이터의 평가 왜곡 ★ ────────────────────────────────
    B = pd.read_csv(ROOT / 'outputs' / 'baseline_results.csv')
    B = B[B.fold == 'ALL'].pivot_table(index='model', columns='group', values='MAE', sort=False)
    t3 = pd.DataFrame({'단순 예측 규칙': B.index, '복제일 MAE': B['복제일'].round(2).values,
                       '고유일 MAE': B['고유일'].round(2).values}).reset_index(drop=True)
    t3['배율'] = (t3['고유일 MAE'] / t3['복제일 MAE']).round(2)
    save(t3, 't3_증강_평가왜곡.csv', '표. 같은 규칙인데 복제일에서 오차가 절반 ★ (평가 과대평가의 증거)')

    # ── 표4. 데이터 품질 이슈와 처리 ─────────────────────────────────
    t4 = pd.DataFrame([
        ('중복 기록', '날짜+시간 중복 없음', '0행', '-'),
        ('결측', '풍속 3 / 강수량 1 / 공장인원 17', '21행', '선형보간, 공장인원은 0 (가동 중단 구간과 일치)'),
        ('시간 컬럼 손상', '7/13·7/15 시간 값 70 to 188', '2일 48행', '날짜별 행 순서로 0 to 23 복구'),
        ('가동 중단', '15분 값 중 0 포함 (8/28 17시 to 8/29 11시, 9/8 12시)', '20행', '학습·평가 모두 제외'),
        ('생산 기록 누락', '하루 종일 생산 0인데 일 최대가 기저부하를 크게 초과', '15일', '학습만 제외, 평가는 유지'),
        ('휴무일', '일 최대전력 30 미만 (일요일 29일, 하계휴가 7/31 to 8/8 등)', '49일', '학습 포함 + 플래그, 시차 입력에서는 결측'),
        ('공휴일 부분가동', '공휴일 10일 중 8일 가동, 조기 종료', '10일', '별도 조건으로 분리'),
        ('증강(복제)', '24시간 전력 곡선이 완전히 동일 (45개 그룹)', '161일', '평가에서 제외, 학습은 가중치 하향'),
    ], columns=['이슈', '내용', '규모', '처리'])
    save(t4, 't4_품질이슈_처리.csv', '표. 데이터 품질 이슈와 처리 방침')

    # ── 표5. 학습·평가 행 구성 ───────────────────────────────────────
    t5 = pd.DataFrame([
        ('전체', len(R), '2021-01-01 to 09-14, 257일 x 24시간'),
        ('평가 대상', int(R.train_ok.sum()), '가동 중단 20행, 시간 손상 48행 제외'),
        ('학습 대상', int(R.train_ok_strict.sum()), '위에서 생산 기록 누락 15일(360행) 추가 제외'),
        ('실제 평가 (5 to 9월 고유일)', 2092, '복제일 제외, 롤링 폴드 평가 구간'),
        ('핵심 판단 기준 (7 to 9월 고유일)', 1708, '테스트 구간과 성격이 같은 구간'),
    ], columns=['구분', '행 수', '설명'])
    save(t5, 't5_학습평가_행구성.csv', '표. 학습·평가 대상 행 구성')

    # ── 그림6. 복제 예시 (전력은 같고 생산은 다른 두 날) ──────────────
    # 전력 곡선이 평범한 가동일 모양인 쌍만 후보로 둔다. 무가동일끼리 짝지으면 그림이 밋밋하다
    cl = R[R.is_clone & (R.clone_n == 2)]
    pair, gap = None, -1
    for gid, g2 in cl.groupby('clone_gid'):
        ds = g2.groupby('날짜')['생산량'].sum()
        mx = g2.groupby('날짜')['target'].max()
        if len(ds) == 2 and mx.min() > 150 and abs(ds.iloc[0] - ds.iloc[1]) > gap:
            gap, pair = abs(ds.iloc[0] - ds.iloc[1]), (int(ds.index[0]), int(ds.index[1]))
    a1, a2 = [R[R['날짜'] == p].sort_values('시간') for p in pair]
    f6 = pd.DataFrame({'시각': a1['시간'].values,
                       f'{pair[0]} 전력': a1.target.values, f'{pair[1]} 전력': a2.target.values,
                       f'{pair[0]} 생산량': a1['생산량'].values, f'{pair[1]} 생산량': a2['생산량'].values})
    save(f6, 'f6_복제예시.csv',
         f'그림. 복제의 증거 — 두 날의 전력 곡선이 완전히 동일한데 생산량은 다르다 '
         f'({pair[0]} 일생산 {a1["생산량"].sum():,.0f} / {pair[1]} 일생산 {a2["생산량"].sum():,.0f})')

    txt = ['보고서 제1장 표·그림 데이터 (src/report_ch1.py 생성)',
           '표·그래프는 Word 기능으로 직접 만든다. 여기 CSV 는 그릴 값만 담았다.',
           '모든 파일은 utf-8-sig 라 엑셀에서 바로 열린다.', '',
           '※ 공통 기준: 특별히 적지 않은 한 train_ok(6,100행) 기준이다.',
           '   상관계수는 종류(피어슨/스피어만)와 모집단을 반드시 캡션에 적을 것.', '']
    for n, c in CAPS:
        txt += [f'[{n}]', f'  {c}', '']
    (OUT / '캡션.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
