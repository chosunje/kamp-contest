"""보고서 4장 표·그림 데이터 — 피크 저감효과의 금액 환산.

4장 마지막 단의 "그래서 얼마인가" 를 계산한다. 숫자를 눈으로 적지 않고
preprocess.load() 에서 다시 뽑아 만든다.

── 왜 기본요금 하나만 보나 ───────────────────────────────────────────────
  전기요금(계절) 열은 109.8 / 167.2 / 191.6 세 값뿐이고 시간대 차등이 없다(1장).
  kWh 를 밤으로 옮겨서 얻는 이득이 0이라는 뜻이다. 생산 총량을 줄이는 과제도 아니므로
  금액이 움직이는 통로는 최대수요전력 기본요금 하나다.
  (같은 판단이 src/peak_schedule.py 머리말과 520행에 이미 있다)

── 왜 월 최대가 아니라 연 최대를 보나 ───────────────────────────────────
  요금적용전력은 당월과 직전 11개월 중 가장 큰 최대수요전력으로 정해진다.
  3장이 "1회 초과가 1년 기본요금을 결정한다" 로 이미 전제한 구조다.
  그래서 연 절감액의 기준은 연 최대의 감소분이고, 월별 감소는 참고로만 쓴다.

── 두 조치를 나눠 계산하는 이유 ─────────────────────────────────────────
  축 A 상시 기동 분산   7·8·9시를 그 날 비기동 최대까지 끌어내린다. 무비용이지만
                        연 최대일(2021-07-19)이 11시 피크라 연 최대를 못 내린다.
  축 B 경보일 피크 억제  그 날 최대 시각 1시간만 2위 시각 수준까지 누른다.
                        경보 재현율 100%(3장)라 대상일을 놓치지 않는다.
  기준은 15분 최대(target_max15 = Q15 열 최대, D20 A안)로 기본요금 산정 기준과 같다.
  복제일은 제외한다(1장 161일). 고유 가동일 5일 미만인 달은 표본 부족으로 표시한다.

── k 와 U 는 왜 대입값인가 ──────────────────────────────────────────────
  k : 전력 열 → kW 환산계수. 15분 열 해석이 미확정이다
      (features.py:154, final_forecast.py:111)
  U : 기본요금 단가. 한전 고시값은 알려져 있으나 이 사업장의 계약 종별이
      데이터에 없다 (peak_alarm_max15.py:188). 세 종별을 모두 낸다.

실행  python src/report_ch4.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess                                                    # noqa: E402

OUT = Path('outputs/report/ch4')
OUT5 = Path('outputs/report/ch5')   # 운영 화면·일정 조정 권고는 5장으로 옮겼다
Q15 = ['15분', '30분', '45분', '60분']
# 기동 구간. 아침 기동(8시 +48.6)과 점심 복귀(13시 +50.1) 둘 다 넣는다.
# 13시를 빼도 결과는 같다 — 월 최대일의 피크가 8·9·11시라 13시가 월 최대를 안 건드린다.
RAMP = (7, 8, 9, 13)
CORE = (7, 8, 9)
MIN_DAYS = 5                     # 고유 가동일이 이보다 적은 달은 표본 부족으로 본다
# 2021-01-01 시행 한전 기본요금 단가(원/kW·월).
# 최대수요전력이 222 라 계약전력 300kW 미만 구간(갑I 저압)이 기준이고,
# 고압 수전이면 갑II 고압A 가 된다. 세 종별을 모두 두고 고지서로 읽는다.
UNITS = ((5550, '산업용(갑)I 저압'), (6490, '산업용(갑)II 고압A 선택I'),
         (7470, '산업용(갑)II 고압A 선택II'))
K_BASE, K_ALT = 1, 4             # 단위 환산계수. 15분 열이 사용량이면 kW 로 4배
SCEN = (('A 상시 기동 분산', 'ramp', 0.3), ('A 상시 기동 분산', 'ramp', 0.5),
        ('A 상시 기동 분산', 'ramp', 1.0), ('B 경보일 피크 시각 억제', 'peak', 0.5),
        ('B 경보일 피크 시각 억제', 'peak', 1.0), ('A + B 병행', 'both', 0.5),
        ('A + B 병행', 'both', 1.0))


def load():
    d = preprocess.load()
    d['q15'] = d[Q15].max(axis=1).astype(float)
    d['월'] = d['날짜'].astype(str).str[4:6].astype(int)
    return d[~d.is_clone & d.train_ok].copy()


def simulate(u, kind, frac):
    """조치 후 일 최대를 다시 계산한다. 월 2위일도 같이 내려가므로
    '한 날만 낮추면 절감 0' 문제를 피한다."""
    rows = []
    for (m, dt), g in u.groupby(['월', '날짜']):
        v = g.q15.values.copy()
        if kind in ('ramp', 'both'):
            inr = g.시간.isin(RAMP).values
            if inr.any() and (~inr).any():
                base, hi = v[~inr].max(), v[inr]
                v[inr] = np.where(hi > base, hi - (hi - base) * frac, hi)
        if kind in ('peak', 'both'):
            s = np.sort(v)[::-1]
            if len(s) > 1:
                v = np.minimum(v, s[0] - (s[0] - s[1]) * frac)
        rows.append((m, dt, g.q15.max(), v.max()))
    D = pd.DataFrame(rows, columns=['월', '날짜', '전', '후'])
    return D.groupby('월').agg(전=('전', 'max'), 후=('후', 'max'))


# 표 4-1·4-4 는 다른 장의 결론을 한 표로 모으는 구성표다. 근거 수치는 모두
# 본문·CSV 에 출처가 있고, 여기서는 그 조합만 고정해 둔다.
PRODUCTS = [
    ['시간별 전력 예측', 'LightGBM h7_lag · L1', '다음 날 24시간 부하 곡선',
     'MAE 6.57 (2장)', '생산·교대 일정 검토'],
    ['일 최대 전력 추정', 'LightGBM 분위 0.9', '그날 최고점의 높이',
     'MAE 8.61 · 편향 -0.16 (2장)', '기본요금 영향 판단'],
    ['피크 위험 경보', '15분 최대 직접 P90', '그날이 위험일인지 여부',
     '재현율 100% · 정밀도 95.5% (3장)', '당일 대응 발동'],
    ['일정 조정 권고', '분기한정 탐색', '어느 작업을 몇 시로 옮길지',
     '피크 -4.01 (2021-09-09)', '작업 재배치 지시 (화면·알고리즘은 5장)'],
]
AXES = [
    ['A 상시 기동 분산', '7·8시와 13시의 동시 기동을 분산',
     '8시 상승폭 +48.6 · 13시 +50.1 (그림 4-1)', '인건비 0 · 생산 손실 0', '채택 (상시)'],
    ['B 경보일 피크 시각 억제', '그날 최대 시각 1시간을 2위 시각 수준까지 억제',
     '연 최대 222 → 215.5 (50% 억제, 표 4-7)', '인건비 0 · 생산 손실 0', '채택 (경보일)'],
    ['C 날짜 간 생산일정', '고생산일을 분산 배치',
     '일생산 17,000 이상에서 일 최대가 포화 (그림 4-2)', '생산 계획 변경 필요', '효과 작음'],
    ['D 하루 안 생산 배분', '시간별 생산량을 재배분',
     '반사실 검증 83쌍 적중률 48.2% (동전 던지기 수준)', '모델이 배분 차이를 구분 못 함', '기각'],
]


def body_tables(u):
    """4장 본문(표 4-1 to 4-5, 그림 4-1·4-2)과 5장 표 5-1 의 데이터."""
    pd.DataFrame(PRODUCTS, columns=['산출물', '담당 모델', '무엇을 내는가', '검증된 성능',
                                    '현장 용도']).to_csv(
        OUT / 't1_산출물별_담당모델.csv', index=False, encoding='utf-8-sig')

    # 표 4-2  기온 x 생산 교차표 — 칸은 피크일 비율, 괄호는 일수
    c = pd.read_csv('outputs/peak_risk_cross.csv')
    c['칸'] = c.apply(lambda r: f"{r['피크일비율(%)']:.0f}% ({r['일수']}일)", axis=1)
    (c.pivot(index='기온대', columns='생산대', values='칸')
      .reindex(['26도 이하', '26-30도', '30도 초과'])
      .reindex(columns=['하위 1/3', '중위 1/3', '상위 1/3'])
      .to_csv(OUT / 't2_기온x생산_교차표.csv', encoding='utf-8-sig'))

    # 표 4-3  단일 조건별 피크일 비율 (교차표의 두 축이 왜 그 둘인지의 근거)
    pd.read_csv('outputs/peak_risk_single.csv').to_csv(
        OUT / 't3_조건별_피크일비율.csv', index=False, encoding='utf-8-sig')

    pd.DataFrame(AXES, columns=['저감 축', '무엇을 바꾸나', '측정 결과', '현장 비용',
                                '판정']).to_csv(
        OUT / 't4_저감축_판정.csv', index=False, encoding='utf-8-sig')

    # 표 4-5  월 최대일의 봉우리와 2위 시각 간격 = 하루 안에서 낮출 수 있는 폭.
    # outputs/peak_reduce_month.csv 는 시간 평균 기준이라 7월 최대일이 20210727(207)로
    # 잡힌다. 4장은 요금 기준(15분 최대)으로 쓰므로 여기서 다시 계산한다.
    rows = []
    for m, g in u.groupby('월'):
        dt = g.loc[g.q15.idxmax(), '날짜']
        s_ = g[g.날짜 == dt].sort_values('q15', ascending=False)
        h1, h2 = s_.iloc[0], s_.iloc[1]
        rows.append({'월': m, '월 최대일': int(dt), '월 최대': h1.q15, '피크 시각': int(h1.시간),
                     '2위': h2.q15, '2위 시각': int(h2.시간), '간격': h1.q15 - h2.q15,
                     '일생산': g[g.날짜 == dt]['생산량'].sum(),
                     '고유 가동일수': g['날짜'].nunique()})
    mm = pd.DataFrame(rows)
    mm['표본'] = np.where(mm['고유 가동일수'] >= MIN_DAYS, '충분', '부족')
    mm.round(1).to_csv(OUT / 't5_월최대일_봉우리간격.csv', index=False, encoding='utf-8-sig')

    # 표 5-1  일정 조정 권고 예시 — 스케줄러 실행 결과를 그대로 옮긴다.
    # 운영 화면과 분기한정 알고리즘 설명은 5장으로 옮기기로 해 출력도 ch5 로 보낸다.
    OUT5.mkdir(parents=True, exist_ok=True)
    sch = json.loads(Path('outputs/final_app/schedule.json').read_text(encoding='utf-8'))
    pd.DataFrame([{
        '작업': t['name'], '설비': t['resource'], '부하': t['load'],
        '원래 시작': f"{t['original_start']}시", '권고 시작': f"{t['recommended_start']}시",
        '이동': f"{t['recommended_start'] - t['original_start']:+d}시간" if t['changed'] else '-',
        '판정': '이동' if t['changed'] else ('고정 작업' if not t['movable'] else '유지'),
    } for t in sch['tasks']]).to_csv(OUT5 / 't1_일정조정_권고예시.csv',
                                     index=False, encoding='utf-8-sig')

    # 그림 4-1  시각별 전력 상승폭 — 동시 기동 지점
    pd.read_csv('outputs/peak_reduce_ramp.csv').to_csv(
        OUT / 'f1_시각별_상승폭.csv', index=False, encoding='utf-8-sig')

    # 그림 4-2  일생산 구간별 일 최대 — 축 B 를 "기각" 이 아니라 "작음" 으로 판정한 근거
    lv = pd.read_csv('outputs/peak_reduce_level.csv')
    lv.insert(0, '생산 구간', [f'{i + 1}분위' for i in range(len(lv))])
    lv.to_csv(OUT / 'f2_일생산구간별_일최대.csv', index=False, encoding='utf-8-sig')
    return sch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = load()
    sch = body_tables(u)
    ndays = u.groupby('월')['날짜'].nunique()
    sim = {(k, f): simulate(u, k, f) for _, k, f in SCEN}

    # ── 표 4-6  월별 저감폭 ------------------------------------------------
    base = sim[('ramp', 0.5)]['전']
    t = pd.DataFrame({'고유 가동일수': ndays, '월 최대수요전력': base})
    for label, kind, frac in SCEN:
        t[f'{label[0]} {frac * 100:.0f}%'] = (sim[(kind, frac)]['전']
                                              - sim[(kind, frac)]['후']).round(1)
    top = (u.loc[u.groupby('월')['q15'].idxmax(), ['월', '날짜', '시간']]
             .set_index('월').rename(columns={'날짜': '월 최대일', '시간': '피크 시각'}))
    t = t.join(top)
    t['표본'] = np.where(t['고유 가동일수'] >= MIN_DAYS, '충분', '부족')
    cols = ['월', '고유 가동일수', '표본', '월 최대일', '피크 시각', '월 최대수요전력']
    t = t.reset_index()[cols + [c for c in t.columns if c.endswith('%')]]
    t.to_csv(OUT / 't7_월별_저감폭.csv', index=False, encoding='utf-8-sig')

    # ── 표 4-7  모델식 입력값 ----------------------------------------------
    pd.DataFrame([
        ['ΔP', '연 최대수요전력 저감폭', '조치별 0.0 to 13.0',
         '표 4-7 — 15분 최대 기준 시뮬레이션', '계산됨'],
        ['k', '전력 열 → kW 환산계수', f'{K_BASE} (대안 {K_ALT})',
         '15분 열 해석 미확정 (features.py:154)', '현장 확인'],
        ['U', '기본요금 단가(원/kW·월)', ' / '.join(f'{v:,}' for v, _ in UNITS),
         '2021-01-01 시행 한전 고시. 계약 종별은 데이터에 없음', '종별 확인'],
        ['n', '적용 개월', '12', '요금적용전력 = 당월·직전 11개월 중 최대', '제도 전제'],
    ], columns=['기호', '뜻', '값', '출처', '구분']).to_csv(
        OUT / 't6_모델식_입력값.csv', index=False, encoding='utf-8-sig')

    # ── 표 4-8  조건별 연 절감액 -------------------------------------------
    rows = []
    for label, kind, frac in SCEN:
        S = sim[(kind, frac)]
        dP_year = S['전'].max() - S['후'].max()
        dP_core = (S['전'] - S['후']).loc[S.index.isin(CORE)].mean()
        for k in (K_BASE, K_ALT):
            r = {'조치': f'{label} ({frac * 100:.0f}%)',
                 '연 최대 ΔP': round(dP_year, 1), 'CORE 월평균 ΔP': round(dP_core, 2),
                 'k': k, 'ΔP·k (kW)': round(dP_year * k, 1)}
            for Uv, lab in UNITS:
                r[f'{lab} ({Uv:,}원)'] = int(round(dP_year * k * Uv * 12))
            rows.append(r)
    money = pd.DataFrame(rows)
    money.to_csv(OUT / 't8_조건별_절감액.csv', index=False, encoding='utf-8-sig')

    # ── 표 4-9  경보 리드타임 (부가 효과의 근거) --------------------------
    pd.read_csv('outputs/peak_alarm_leadtime.csv').to_csv(
        OUT / 't9_경보_리드타임.csv', index=False, encoding='utf-8-sig')

    # ── 그림 4-3  조치 전/후 월 최대 ---------------------------------------
    pd.DataFrame({'월': base.index, '현재': base.values,
                  'A 상시 기동 분산': sim[('ramp', 0.5)]['후'].values,
                  'B 경보일 피크 억제': sim[('peak', 0.5)]['후'].values,
                  'A + B 병행': sim[('both', 0.5)]['후'].values}).round(1).to_csv(
        OUT / 'f3_조치_전후_월최대.csv', index=False, encoding='utf-8-sig')

    cap = [
        '표 4-1  산출물별 담당 모델과 현장 용도 (구성표 · 근거는 2·3장)', '',
        '표 4-2  기온 x 생산 교차표 — 칸은 피크일 비율, 괄호는 해당 일수',
        '  생산은 필요조건(관문), 기온은 증폭 요인이다. 생산 하위 1/3 은 기온과 무관하게 0% 다.', '',
        '표 4-3  단일 조건별 피크일 발생 비율 (outputs/peak_risk_single.csv)', '',
        '표 4-4  저감 축 4종의 검증 결과와 판정 (구성표)', '',
        '표 4-5  월 최대일의 봉우리와 2위 시각 간격 (15분 최대 기준 · 복제일 제외)',
        '  하루 안에서 낮출 수 있는 상한이다. 월 2위일이 받치므로 월 최대에 실제로',
        '  반영되는 폭은 표 4-6 쪽이 작거나 같다.', '',
        '표 5-1  일정 조정 권고 예시 — outputs/final_app/schedule.json 실행 결과  [5장]',
        f'  {sch["date"]} · 피크 {sch["metrics"]["before_peak"]:.1f} -> '
        f'{sch["metrics"]["after_peak"]:.1f} · 경보 시간 '
        f'{sch["metrics"]["before_alarm_hours"]} -> {sch["metrics"]["after_alarm_hours"]}', '',
        '그림 4-1  시각별 전력 상승폭 — 동시 기동 지점 (8시 +48.6 · 13시 +50.1)', '',
        '그림 4-2  일생산 구간별 일 최대 전력 (포화 구조)', '',
        '표 4-6  절감액 모델식의 입력값',
        '  ΔP 는 데이터로 계산된다. k(단위 환산)와 U(계약 종별)는 현장 확인 항목이다.', '',
        '표 4-7  조치별 월 최대수요전력 저감폭 (15분 최대 기준 · 복제일 제외)',
        '  월 2위일도 같이 내려가므로 "한 날만 낮추면 절감 0" 문제를 피한다.',
        f'  고유 가동일 {MIN_DAYS}일 미만인 달은 표본 부족으로 표시했다.', '',
        '표 4-8  조치별 연 절감액',
        '  연 절감액 = ΔP x k x U x 12. ΔP 는 연 최대수요전력의 감소분이다.',
        '  축 A 는 연 최대일(2021-07-19)이 11시 피크라 연 최대를 내리지 못해 0 이다.', '',
        '표 4-9  경보 리드타임 (outputs/peak_alarm_leadtime.csv)',
        '  분위 0.9 정책 기준이며 3장의 D10 확정 정책과 모집단·타깃이 다른 별도 측정이다.', '',
        '그림 4-3  조치 전/후 월 최대수요전력 (각 조치 50% 수준)',
        '  운영 화면(옛 그림 4-3)은 5장으로 옮겼고, 그 자리를 이 그림이 받는다.', '']
    (OUT / '캡션.txt').write_text('\n'.join(cap), encoding='utf-8')

    print('── 표 4-6 월별 저감폭 ──')
    print(t.to_string(index=False))
    i = t['월 최대수요전력'].idxmax()
    print(f'\n연 최대 {base.max():.0f} ({t.loc[i, "월 최대일"]} {t.loc[i, "피크 시각"]}시)')
    print('\n── 표 4-8 조건별 연 절감액 (원) ──')
    print(money.to_string(index=False))
    print('\n→', OUT)


if __name__ == '__main__':
    main()
