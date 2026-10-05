"""피크전력 저감방안 — 과제 마지막 문장에 답한다.

  과제  "생산일정 또는 설비가동 시점 조정을 통한 피크전력 저감방안을 제안하시오"

  요구가 "제안" 이라는 점이 중요하다. 반사실 모델로 효과를 증명하라는 것이 아니다.
  그래서 이 모듈은 모델을 쓰지 않고 실측값만으로 기회의 크기를 잰다.
  모델이 필요한 것은 "어느 날이 위험한가"(경보)까지고, 저감방안의 근거는 관측이다.

── 세 가지 축을 따로 재야 한다 (이걸 섞으면 틀린 결론이 난다) ─────────────
  축 A  설비가동 시점 조정 (기동 분산)   ★ 가장 크고 근거가 확실하다
        하루 안에서 봉우리 하나를 눌러 평탄화. 생산 총량도 배분도 안 바꾼다
  축 B  생산일정 조정 (날짜 간)          모델이 읽는 축이지만 효과가 작다
        어느 날에 많이 생산할지를 바꾼다. 일 단위 총량이 바뀐다
  축 C  하루 안의 생산 배분 조정         ✗ 근거 없음 → 기각  [8-5]
        총량을 고정하고 시간별 배분만 바꾼다. 모델이 이 축을 읽지 못한다 (48.2%)

  [8-5] 에서 기각한 것은 축 C 뿐이다. 축 A·B 는 따로 재야 하고, 실제로 결과가 다르다.

── 왜 축 A 에 모델이 필요 없나 ──────────────────────────────────────────
  "그날 최대가 기동 시각에 났고, 그 봉우리를 그날 2위 시각 수준까지만 눌렀다면
   일 최대가 얼마나 내려갔겠나" 는 실측값 두 개의 뺄셈이다. 예측이 아니다.
  이것이 기동 분산으로 얻을 수 있는 양의 상한이다.

실행  python src/peak_reduce.py
출력  outputs/peak_reduce_ramp.csv     시각별 전력 상승폭 (동시 기동 지점)
      outputs/peak_reduce_month.csv    월 최대일의 봉우리와 2위 간격 (기회 크기)
      outputs/peak_reduce_level.csv    일생산 구간별 일 최대 (축 B 의 포화)
"""
import numpy as np
import pandas as pd

from features import build
from preprocess import ROOT

OUT = ROOT / 'outputs'


def day_table(X):
    """가동일마다 일 최대와 그날 2위 시각을 낸다. 24행이 온전한 날만."""
    d = X[X.train_ok & ~X.is_clone & ~X.is_off & ~X.is_stop].copy()
    n = d.groupby('날짜')['hour'].size()
    d = d[d['날짜'].isin(n[n == 24].index)]
    rows = []
    for dt, s in d.groupby('날짜'):
        v = s.set_index('hour')['target'].sort_values(ascending=False)
        rows.append({'날짜': dt, '월': int(s['month'].iloc[0]),
                     '피크시각': int(v.index[0]), '일최대': float(v.iloc[0]),
                     '2위시각': int(v.index[1]), '2위': float(v.iloc[1]),
                     '간격': float(v.iloc[0] - v.iloc[1]),
                     '일생산': float(s['prod'].sum())})
    return pd.DataFrame(rows), d


def ramp(d):
    """시각별 전력 상승폭 (전 시각 대비, 가동일 평균).
    가장 큰 시각이 설비가 한꺼번에 켜지는 시각이다 — 어떤 설비인지는 몰라도
    '언제' 동시 기동이 일어나는지는 이걸로 특정된다  ([8-4] 2순위가 비워 둔 부분)."""
    h = d.pivot_table(index='날짜', columns='hour', values='target')
    r = h.diff(axis=1).mean()
    return pd.DataFrame({'시각': r.index, '상승폭': r.values}).dropna()


def month_peak(D):
    """월 최대를 만든 날 — 요금이 걸린 곳은 여기 하나다."""
    rows = []
    for m, s in D.groupby('월'):
        t = s.loc[s.일최대.idxmax()]
        rows.append({'월': int(m), '날짜': int(t.날짜), '월최대': t.일최대,
                     '피크시각': int(t.피크시각), '2위': t['2위'], '2위시각': int(t['2위시각']),
                     '간격': t.간격, '일생산': t.일생산})
    return pd.DataFrame(rows)


def level(D, k=5):
    """축 B — 일생산을 바꾸면 일 최대가 얼마나 움직이나 (관측값).
    포화하면 생산일정을 옮겨도 피크가 안 내려간다는 뜻이다."""
    q = pd.qcut(D.일생산, k)
    g = D.groupby(q, observed=True).agg(일수=('일최대', 'size'), 평균일생산=('일생산', 'mean'),
                                        평균일최대=('일최대', 'mean'), 최대일최대=('일최대', 'max'))
    return g.reset_index(drop=True)


def main():
    X = build()
    D, d = day_table(X)
    R, M, L = ramp(d), month_peak(D), level(D)
    morn = D[D.피크시각.isin([7, 8, 9])]
    up = R.sort_values('상승폭', ascending=False).head(3)

    print(f'[피크전력 저감방안]  가동일 {len(D)}일 · 실측만 사용 (모델 반사실 아님)\n')

    print('── 축 A. 설비가동 시점 조정 (기동 분산) ──────────────────────────')
    print('  ① 동시 기동이 언제 일어나는가 — 시각별 전력 상승폭 (전 시각 대비 평균)')
    print('    ' + '  '.join(f'{int(r.시각):2d}시 {r.상승폭:+6.1f}' for _, r in
                             R[R.상승폭.abs() >= 10].iterrows()))
    print(f'    상승폭 최대 {int(up.iloc[0].시각)}시 {up.iloc[0].상승폭:+.1f}'
          f' · 그다음 {int(up.iloc[1].시각)}시 {up.iloc[1].상승폭:+.1f}')
    drop12 = R.loc[R.시각 == 12, '상승폭'].iloc[0]
    up13 = R.loc[R.시각 == 13, '상승폭'].iloc[0]
    print(f'    ★ 12시 {drop12:+.1f} 로 떨어졌다가 13시 {up13:+.1f} 로 되올라온다 —')
    print('      점심에 설비를 세웠다가 한꺼번에 다시 켠다는 뜻이다. 아침 8시와 같은 현상')
    print()
    print('  ② 기회의 크기 — 월 최대를 만든 날의 봉우리와 2위의 간격')
    print(M[['월', '날짜', '월최대', '피크시각', '2위', '2위시각', '간격']]
          .to_string(index=False))
    k = M[M.피크시각.isin([7, 8, 9, 13])]
    print(f'    월 최대일 {len(M)}개월 중 {len(k)}개월이 기동 시각(7-9시·13시)에 났다')
    print(f'    간격 평균 {M.간격.mean():.1f} · 최대 {M.간격.max():.1f}')
    print('    → 그 봉우리만 그날 2위 수준으로 눌러도 월 최대가 그만큼 내려간다.')
    print('      생산 총량도 시간별 배분도 건드리지 않는다 = 생산 손실 0, 인건비 증가 0')
    print()
    print(f'  ③ 오전 기동형은 전체의 {100 * len(morn) / len(D):.1f}% ({len(morn)}일)')
    print()

    print('── 축 B. 생산일정 조정 (날짜 간) ─────────────────────────────────')
    print('  일생산 구간별 일 최대 (관측값)')
    print(L.round(0).to_string(index=False))
    lo, hi = L.iloc[1], L.iloc[-1]
    print(f'    ★ 포화한다. 일생산 {lo.평균일생산:,.0f} → {hi.평균일생산:,.0f}'
          f' ({100 * (hi.평균일생산 / lo.평균일생산 - 1):.0f}% 증가) 인데'
          f' 일 최대는 {lo.평균일최대:.0f} → {hi.평균일최대:.0f} 뿐이다')
    print(f'    큰 차이는 최하위 구간에서만 난다 (일생산 {L.iloc[0].평균일생산:,.0f}'
          f' → 일 최대 {L.iloc[0].평균일최대:.0f})')
    print('    → 생산일정을 옮겨 피크를 낮추려면 그날을 거의 쉬게 해야 한다.')
    print('      그건 일정 조정이 아니라 생산 포기다 → 효과가 작은 수단으로 분류한다')
    print()

    print('── 축 C. 하루 안의 생산 배분 조정 ────────────────────────────────')
    print('  ✗ 기각. 모델이 이 축을 읽지 못한다 (조건 맞춘 날짜 쌍 83개에서 48.2%)  [8-5]')
    print('    참고: 날짜 간 축(축 B)에서는 읽는다 — 일생산 40% 이상 차이 나는 쌍에서')
    print('    64.3% (p=0.044). 축에 따라 결론이 다르므로 섞어서 말하면 안 된다')
    print()

    print('── 제안 (보고서에 이 순서로) ─────────────────────────────────────')
    print('  1순위  아침 기동과 점심 후 재기동을 분산한다                     ★')
    print(f'         근거  8시 +{up13 if False else R.loc[R.시각 == 8, "상승폭"].iloc[0]:.0f},'
          f' 13시 +{up13:.0f} 의 동시 상승. 월 최대일 {len(k)}/{len(M)}개월이 그 시각')
    print(f'         기대  월 최대 -{M.간격.mean():.0f} ~ -{M.간격.max():.0f}'
          ' (실측 간격). 생산 손실·인건비 증가 없음')
    print('         한계  어떤 설비를 몇 분 어긋나게 할지는 설비별 계량이 없어 못 정한다')
    print('               → 현장과 함께 정할 사항. 이 분석은 "언제" 까지만 특정한다')
    print('  2순위  월 최대를 만드는 날의 생산량을 다른 날로 분산한다')
    print('         근거  모델이 날짜 간 축을 읽는다 (64.3%, p=0.044)')
    print('         한계  포화 때문에 효과가 작다. 크게 줄여야 의미가 있는데 그건 생산 포기')
    print('  3순위  냉방 선행 운전')
    print('         한계  냉방이 따로 계량되지 않아 검증 불가. 제안까지만  [8-4]')
    print('  기각   하루 안의 생산 배분 조정 — 근거 없음  [8-5]')
    print()
    print('  ※ 저감의 기준은 일 최대가 아니라 월 최대다. 기본요금은 한 달에 하루로')
    print('    정해지므로 여러 날을 조금씩 낮춰도 최대일을 못 낮추면 절감은 0이다')

    R.round(3).to_csv(OUT / 'peak_reduce_ramp.csv', index=False, encoding='utf-8-sig')
    M.round(2).to_csv(OUT / 'peak_reduce_month.csv', index=False, encoding='utf-8-sig')
    L.round(2).to_csv(OUT / 'peak_reduce_level.csv', index=False, encoding='utf-8-sig')
    print(f'\n→ {OUT / "peak_reduce_ramp.csv"}')
    print(f'→ {OUT / "peak_reduce_month.csv"}')
    print(f'→ {OUT / "peak_reduce_level.csv"}')


if __name__ == '__main__':
    main()
