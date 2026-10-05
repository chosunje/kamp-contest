"""보고서 제3장(영향요인 및 오류분석)에 넣을 표·그림용 데이터를 CSV로 만든다.

표와 그래프는 Word 기능으로 직접 그리므로 "그릴 값"만 내보낸다.
수치는 모두 outputs/ 의 실행 결과에서 읽어 오며, 본문에 직접 타이핑한 값이 없도록 한다.

제2장과 집계 방식이 다르다는 점에 주의한다.
  2장  설정 비교가 목적이라 시드별 지표를 평균한다 (CORE MAE 6.57)
  3장  오차를 행 단위로 쪼개야 하므로 시드 3개의 예측값을 평균한 단일 예측 집합을 쓴다 (6.51)
차이는 시드 흔들림(0.19) 안쪽이며, 캡션에도 같은 내용을 적어 둔다.

선행: python src/error_analysis.py, python src/peak_alarm_max15.py, python src/peak_driver.py
실행: python src/report_ch3.py  →  outputs/report/ch3/*.csv, 캡션.txt
"""
import pandas as pd

from preprocess import ROOT

OUT = ROOT / 'outputs' / 'report' / 'ch3'
OUT.mkdir(parents=True, exist_ok=True)
CAPS = []

CORE_MONTHS = ('07', '08', '09')
ALARM_POLICY = ('P90', 0.95)          # D10 확정 정책 (alarm_policy.py 와 동일)


def save(df, name, caption):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    CAPS.append((name, caption))
    print(f'  → {name:30s} {len(df):3d}행')


E = pd.read_csv(ROOT / 'outputs' / 'error_by_condition.csv')
P = pd.read_csv(ROOT / 'outputs' / 'error_predictions.csv')
D = pd.read_csv(ROOT / 'outputs' / 'peak_alarm_max15_days.csv')
V = pd.read_csv(ROOT / 'outputs' / 'peak_alarm_max15_eval.csv')
A = pd.read_csv(ROOT / 'outputs' / 'peak_alarm_max15_leakage_audit.csv')
W = pd.read_csv(ROOT / 'outputs' / 'peak_driver_when.csv')

core = P[P.dt.str.slice(5, 7).isin(CORE_MONTHS)]


def main():
    # ── 표1. 조건별 오차 구조 ──────────────────────────────────────────
    # 평균 하나로는 "어디서 믿고 쓸 수 있는가"를 답할 수 없어 조건을 안정/실패로 나눠 보인다.
    S = E[E.axis == 'special'].set_index('condition')
    rows = [('기준', '전체 (5 to 9월)'),
            ('기준', 'CORE (7 to 9월)'),
            ('안정', '주말'),
            ('안정', '휴무 직후'),
            ('실패', '고온 가동(기온>22℃ & 생산>0)'),
            ('실패', '전환시각(7·12·17시)'),
            ('실패', '고생산(900+)'),
            ('실패', '피크 구간(학습구간 상위5% 기준)'),
            ('실패', '마지막 생산시간 이후'),
            ('실패', '법정공휴일'),
            ('실패', '공휴일 & 생산종료 이후')]
    out = []
    for group, cond in rows:
        if cond == 'CORE (7 to 9월)':                    # period 축에 있어 따로 집계한다
            e = core.pred - core.target
            r = dict(n=len(core), MAE=e.abs().mean(), bias=-e.mean(),
                     under_rate=(e < 0).mean())
        else:
            key = '전체' if cond.startswith('전체') else cond
            r = S.loc[key]
        out.append({'구분': group, '조건': cond, '행 수': int(r['n']),
                    'MAE': round(float(r['MAE']), 2),
                    '편향': round(float(r['bias']), 2),
                    '과소예측 비율(%)': round(float(r['under_rate']) * 100, 1)})
    t1 = pd.DataFrame(out)
    save(t1, 't1_조건별_오차.csv',
         '표. 조건별 오차 구조 ★ — 편향 양수 = 실제보다 낮게 봄. '
         '주말·휴무 직후는 안정적이고, 공휴일과 피크 구간에서 무너진다')

    # 그림1. 위 표를 막대로 (가로 막대 1계열)
    f1 = t1[t1.구분 != '기준'][['조건', 'MAE']].sort_values('MAE')
    save(f1, 'f1_조건별_MAE.csv',
         '그림. 조건별 MAE — 가로 막대 1계열. 전체 평균선(8.59)을 참조선으로 겹치면 좋다')

    # ── 표2. 공휴일 일자별 (최대 실패 요인의 해부) ──────────────────────
    # 공휴일이라 틀리는 게 아니라 '부분 가동' 공휴일만 틀린다는 것을 보이는 표.
    h = P[P.is_holiday == 1]
    t2 = (h.groupby('날짜')
            .agg(**{'일 생산량': ('prod', 'sum'), '실제 평균 전력': ('target', 'mean'),
                    '예측 평균 전력': ('pred', 'mean'), 'MAE': ('abs_error', 'mean')})
            .round(1).reset_index())
    t2['판정'] = ['완전 휴무 (정확)' if p == 0 else
                  '부분 가동 (과대예측)' if p < 5000 else '정상 조업에 가까움'
                  for p in t2['일 생산량']]
    save(t2, 't2_공휴일_일자별.csv',
         '표. 공휴일 일자별 실제 대 예측 ★ — 생산 0인 완전 휴무는 맞히고, '
         '생산은 있으나 조기 종료하는 부분 가동일에서만 크게 과대예측한다')

    # ── 표3. 피크 경보 혼동행렬 + 오경보 상세 ──────────────────────────
    model, q = ALARM_POLICY
    d = D[(D.model == model) & (D.alarm_q == q)
          & (D.fold.str.slice(5, 7).isin(CORE_MONTHS)) & (D.n == 24)]
    tp = int((d.alarm & d.real_peak).sum())
    fp = int((d.alarm & ~d.real_peak).sum())
    fn = int((~d.alarm & d.real_peak).sum())
    tn = int((~d.alarm & ~d.real_peak).sum())
    t3 = pd.DataFrame([
        {'구분': '경보 발령', '실제 위험일': tp, '실제 안전일': fp},
        {'구분': '경보 없음', '실제 위험일': fn, '실제 안전일': tn}])
    save(t3, 't3_경보_혼동행렬.csv',
         f'표. 피크 경보 혼동행렬 ★ — CORE 완전일 {len(d)}일, '
         f'미탐지 {fn}일 / 오경보 {fp}일 (정밀도 {tp/(tp+fp)*100:.1f}%, 재현율 {tp/(tp+fn)*100:.1f}%)')

    miss = d[d.alarm != d.real_peak].copy()
    miss['유형'] = ['오경보' if a else '미탐지' for a in miss.alarm]
    miss['임계 대비 차이'] = (miss.actual_max15 - miss.real_threshold).round(1)
    t3b = miss[['날짜', '유형', 'actual_max15', 'predicted_max15',
                'real_threshold', '임계 대비 차이']].rename(columns={
        'actual_max15': '실제 15분 최대', 'predicted_max15': '예측 P90',
        'real_threshold': '위험 임계'}).round(1)
    save(t3b, 't3b_오경보_상세.csv',
         '표. 틀린 날의 상세 — 두 오경보 모두 실제 최대가 임계 바로 아래였던 경계 사례다')

    # ── 표4. 경보 임계 민감도 (왜 Q95인가) ─────────────────────────────
    v = V[V.period.str.startswith('CORE')].copy()
    cols = ['day_precision_pct', 'day_recall_pct', 'day_alarm_rate_pct',
            'hour_precision_pct', 'hour_recall_pct']
    t4 = v[['model', 'alarm_q'] + cols].copy()
    t4[cols] = t4[cols].round(1)                      # 임계는 반올림하면 안 되므로 지표만 round
    t4['alarm_q'] = [f'Q{int(round(q * 100))}' for q in t4.alarm_q]
    t4['model'] = t4['model'].replace({'L1': '점 예측 (L1)', 'P90': '15분 최대 P90 ★'})
    t4 = t4.rename(columns={
        'model': '예측 모델', 'alarm_q': '경보 임계',
        'day_precision_pct': '일 정밀도(%)', 'day_recall_pct': '일 재현율(%)',
        'day_alarm_rate_pct': '일 경보율(%)',
        'hour_precision_pct': '시간 정밀도(%)', 'hour_recall_pct': '시간 재현율(%)'})
    save(t4.sort_values(['예측 모델', '경보 임계'], ascending=[False, True]),
         't4_경보_임계민감도.csv',
         '표. 경보 임계 민감도 ★ — 임계를 올려도 정밀도는 거의 그대로인데 재현율만 급락한다. '
         '미탐지 비용이 오경보 비용보다 크므로 Q95를 채택했다')

    # ── 그림2. 일 최대가 발생한 시각 분포 ──────────────────────────────
    # 시간 단위 경보가 어려운 이유이자, 일 경보를 받은 뒤 어디를 볼지 알려주는 근거.
    f2 = W[['hour', '일수', '비율(%)']].rename(columns={'hour': '시각'}).sort_values('시각')
    top4 = W.nlargest(4, '비율(%)')
    save(f2, 'f2_피크발생시각.csv',
         f'그림. 일 최대가 발생한 시각 분포 — 세로 막대. '
         f'상위 4개 시각({", ".join(str(h) + "시" for h in sorted(top4.hour))})이 '
         f'전체의 {top4["비율(%)"].sum():.0f}%를 차지한다')

    # ── 캡션 ──────────────────────────────────────────────────────────
    e = core.pred - core.target
    txt = ['보고서 제3장 표·그림 데이터 (src/report_ch3.py 생성)',
           '수치는 outputs/ 의 error_by_condition.csv, error_predictions.csv,',
           'peak_alarm_max15_*.csv, peak_driver_when.csv 에서 읽었다.',
           '',
           '※ 집계 기준이 제2장과 다르다.',
           '   2장은 설정 비교가 목적이라 시드별 지표를 평균한다 (CORE MAE 6.57).',
           f'   3장은 오차를 행 단위로 쪼개야 하므로 시드 3개의 예측값을 평균한',
           f'   단일 예측 집합을 쓴다 (CORE n={len(core):,} MAE {e.abs().mean():.2f}).',
           '   차이는 시드 흔들림(0.19) 안쪽이며 본문에도 같은 각주를 달았다.',
           '',
           '※ 편향(bias)은 양수가 "실제보다 낮게 봤다"는 뜻이다 (실제 - 예측).',
           f'※ 경보 정책은 {ALARM_POLICY[0]} · 임계 Q{int(ALARM_POLICY[1]*100)} '
           '(alarm_policy.py 의 D10 확정 정책과 동일).',
           f'※ 미래 정보 누수 검증: {A.날짜.nunique()}일 × 삭제/변경 {len(A)}건에서 '
           f'피처·예측·경보가 모두 불변이었다 (예측 최대 변화 {A.prediction_max_delta.max():.1f}).',
           '']
    for name, cap in CAPS:
        txt += [f'[{name}]', f'  {cap}', '']
    (OUT / '캡션.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
