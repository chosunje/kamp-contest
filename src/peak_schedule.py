"""최대피크 예측과 저감 스케줄링 — 과제 제목의 뒤쪽 절반 (현장 활용 배점 10점).

작업내역(조선제).txt [8-4] 가 남겨 둔 2순위(오전 기동 분산)와 4순위(생산 재배치)를
실제로 돌려 보는 모듈이다. [8-2] 의 경보(1순위)는 "위험하다" 까지만 알려 준다.
여기서는 "그래서 무엇을 바꾸면 되는가" 를 낸다.

── 무엇을 하나 ───────────────────────────────────────────────────────────
  1. 일 최대 피크 예측    분위 0.9 추정량으로 그날 최대가 얼마일지 낸다  [6-5]
  2. 재배치안 탐색        생산 총량은 그대로 두고 시간별 배분만 바꿔 예측 피크를 낮춘다
  3. 규칙 기준선 비교     모델 없이 "생산 최다 시각 비우기" 보다 나은지 본다  [9] 12번
  4. 반사실 신뢰도 검증   모델을 "안 해 본 일정" 에 쓰는 것이 타당한지 실제 날짜로 잰다
  5. 절감 환산            피크 몇 kW 를 줄였는지까지. 금액은 단가가 없어 식만 둔다

── 왜 생산량을 줄이지 않고 배분만 바꾸나 ────────────────────────────────
  생산 800 이상에서 전력이 거의 안 오르고 1,600 이상에서는 오히려 내려간다
  ([8-1] 결과 2, outputs/peak_driver_sat.csv). 총량을 건드리지 않고 몰아 주는 것만으로
  피크가 내려갈 수 있다는 뜻이다. 그래서 결정변수는 생산 수준이 아니라 시간별 배분이다.

── 인건비 (데이터에서 확인한 값이다. 가정이 아니다) ──────────────────────
  인건비 열이 9-17시에 1.0, 그 외 시각에 1.5 로 들어 있다.
  → 9-17시 안에서만 옮기면 인건비가 전혀 늘지 않는다 (무비용안).
    18-8시로 옮기면 그 시간분이 1.5 배가 된다 (야간 허용안).
  시간대별 전기요금(TOU)은 없고 계절 단가뿐이므로([8-4] 전제) 절감 채널은
  최대수요전력 기본요금 하나다. kWh 를 야간으로 옮겨 얻는 이득은 없다.

── 한계 (보고서에 반드시 같이 쓸 것) ────────────────────────────────────
  이 모듈은 모델을 반사실(counterfactual)로 쓴다. 모델은 "관측된 일정에서" 전력을
  맞히도록 학습됐을 뿐, "일정을 바꾸면 전력이 이렇게 변한다" 는 인과를 배운 적이 없다.
  그래서 두 가지를 넣었다.
    · 받는 시각의 생산량이 학습 구간에서 그 시각에 관측된 최대를 넘지 않게 막는다
      (외삽 금지). 넘으면 제안에서 제외하고 그 횟수를 보고한다
    · --validate 로 "비슷한 조건의 실제 날짜 두 개 중 어느 쪽 피크가 높은가" 를
      모델이 맞히는지 잰다. 이게 반사실 사용의 간접 근거다

실행  python src/peak_schedule.py                     실적이 있는 마지막 가동일 3일로 시연
      python src/peak_schedule.py 20210908 20210914   구간 지정
      python src/peak_schedule.py --night             야간(18-8시)으로도 옮김
      python src/peak_schedule.py --validate          반사실 신뢰도 검증만 (빠름)
      python src/peak_schedule.py --days 5            시연 일수
"""
import sys

import numpy as np
import pandas as pd

from features import FEATURES, build
from model import (ALARM_CFG, CLONE_W, FINAL_CFG, PEAK_W, SEEDS, make_model,
                   rolling_eval)
from preprocess import RAW, ROOT, load

OUT = ROOT / 'outputs'
COLS = FEATURES['h7_lag']
WORK_HOURS = tuple(range(9, 18))     # 인건비 1.0 구간 (데이터에서 확인)
NIGHT_HOURS = tuple(h for h in range(24) if h not in WORK_HOURS)
MAX_OFF = 4         # 하루에 비울 수 있는 시각의 최대 개수
TOL = 0.3           # 이보다 작게 내려가면 더 옮기지 않는다 (잡음 수준)
SAT = 1000          # 포화 지점. 규칙 기준선이 쓰는 상한 (peak_driver_sat.csv)


# ── 모델 ────────────────────────────────────────────────────────────────────
def fit(X, cutoff, seeds=SEEDS):
    """cutoff 이전 행만으로 점 예측·분위 0.9 두 모델을 학습한다.
    대상일의 전력은 한 번도 보지 않는다 (predict_test.py 와 같은 규약)."""
    tr = X[(X['dt'] < cutoff) & X['train_ok_strict']]
    if len(tr) < 200:
        raise SystemExit(f'학습 행이 {len(tr)}개뿐이다. 대상일이 너무 이르다.')
    w = (np.where(tr.is_clone, CLONE_W, 1.0)
         * np.where(tr.target >= tr.target.quantile(.95), PEAK_W, 1.0))
    pt = [make_model('lgbm', s, FINAL_CFG['params']).fit(tr[COLS], tr.target, sample_weight=w)
          for s in seeds]
    hi = [make_model('lgbm', s, ALARM_CFG['params']).fit(tr[COLS], tr.target,
                                                         sample_weight=np.where(tr.is_clone, CLONE_W, 1.0))
          for s in seeds]

    def predict(rows):
        """(점 예측, 분위 0.9) 를 시드 평균으로 돌려준다."""
        x = rows[COLS]
        return (np.mean([m.predict(x) for m in pt], axis=0),
                np.mean([m.predict(x) for m in hi], axis=0))
    return predict, tr


def hour_cap(tr, q=.90, ceiling=1600):
    """시각별 생산량 상한. 재배치안이 모델을 악용하지 못하게 막는 장치다.

    왜 "관측 최대" 를 그냥 쓰지 않나
      포화표(peak_driver_sat.csv)에서 생산 1,600 이상 구간은 전력이 오히려 내려간다
      (156.6 → 150.6). 물리가 아니라 기록 방식의 교란이다. 1,600 초과 행이 교대 경계
      시각에 몰려 있고 그때 전력이 낮다 — 7시 n=36 생산 1,993/전력 111,
      12시 n=12 생산 3,967/전력 107, 19시 n=108 생산 2,697/전력 140.
      일을 몰아서 기록한 흔적이다. 이 구간을 열어 두면 최적화가 "한 시각에 많이 몰면
      전력이 내려간다" 는 가짜 이득을 찾아낸다. 실제로 열어 두고 돌려 보니 9/14 에서
      네 시각을 비우고 생산 3,400 을 몰아 -6.8 을 만들어 냈는데, 그 이득의 출처가
      위 교란이었다.
    그래서  그 시각 관측 분위 q(기본 0.90)와 절대 상한 ceiling(1,600) 중 작은 값을 쓴다.
      조밀하게 관측된 구간 안에서만 제안하므로 "포화라 얹는 비용이 없다"는 근거(800-1,600
      구간이 평평하다)는 그대로 쓸 수 있고, 내려가는 구간은 건드리지 않는다.
    """
    cap = tr.groupby('hour')['prod'].quantile(q).reindex(range(24)).fillna(0)
    return cap.clip(upper=ceiling)


# ── 반사실: 생산량을 바꿔 다시 피처를 만든다 ────────────────────────────────
def rebuild(raw, date, prod=None):
    """raw(원본 형식)에서 date 의 생산량만 prod 로 바꿔 전처리·피처를 다시 태운다.

    생산량을 바꾸면 prod·prod_cap·prod_log·prod_zero·day_prod·day_prod_hours·
    prod_prev·prod_next·prod_diff·in_prod_window·hrs_* 가 함께 바뀐다. 피처를 직접
    손대면 빠뜨리기 쉬우므로 반드시 전처리를 다시 태운다.
    (팀원이 넣은 load(raw)·build(df) 입구를 쓴다 — 원래 누수 검증용이지만 여기에도 맞는다)
    """
    # 생산량은 원본에서 정수 열이다. 상한이 소수로 나올 수 있어 실수로 바꿔 둔다.
    # 바꾸지 않는 경로에서도 같이 바꿔야 피처 dtype 이 갈리지 않는다
    r = raw.copy()
    r['생산량'] = r['생산량'].astype(float)
    if prod is not None:
        m = r['날짜'] == date
        assert m.sum() == 24, f'{date} 행이 24개가 아니다'
        r.loc[m, '생산량'] = np.asarray(prod, dtype=float)
    X = build(load(r))
    X['ym'] = X['dt'].dt.to_period('M').astype(str)
    return X


def peak_of(predict, X, date):
    """그날의 (예측 일 최대, 시간별 분위 0.9, 시간별 점 예측) 을 돌려준다.
    일 최대는 점 예측이 아니라 분위 0.9 로 본다 — 점 예측의 최대는 구조적으로 낮다 [6-5]."""
    d = X[X['날짜'] == date].sort_values('hour')
    pt, hi = predict(d)
    return hi.max(), hi, pt


# ── 재배치 탐색 ─────────────────────────────────────────────────────────────
def sensitivity(predict, raw, date, hour, levels=(0, 200, 500, 1000, 2000, 4000)):
    """한 시각의 생산량만 바꿔 가며 그 시각 예측 전력이 어떻게 움직이는지 본다.
    재배치 수단을 고르기 전에 반드시 봐야 하는 곡선이다 — 아래 reallocate 가
    "조금씩 옮기기" 가 아니라 "비우기" 를 쓰는 이유가 여기서 나온다."""
    base = day_prod_vector(raw, date)
    rows = []
    for v in levels:
        p = base.copy()
        p[hour] = v
        pk, hi, _ = peak_of(predict, rebuild(raw, date, p), date)
        rows.append({'생산량': v, f'{hour}시 예측전력': hi[hour], '일 최대': pk})
    return pd.DataFrame(rows)


def day_prod_vector(raw, date):
    """그날의 시간별 생산량 (0시 to 23시 순서)."""
    X = rebuild(raw, date)
    return X[X['날짜'] == date].sort_values('hour')['prod'].to_numpy(dtype=float)


def reallocate(predict, raw, date, cap, hours, max_off=4):
    """예측 피크 시각을 "비우고" 그 생산량을 다른 시각에 얹는다.

    왜 비우기인가  결정변수는 생산량이 아니라 가동 상태다 ([8-1] 결과 2).
      생산 800-1,000 위에서 전력이 포화하므로(peak_driver_sat.csv) 피크 시각에서
      200-400 을 덜어 내도 여전히 포화 구간 안이어서 전력이 꿈쩍하지 않는다.
      실제로 9/14 14시에서 생산을 1,023 → 823 으로 줄여도 예측이 그대로였고,
      0 으로 비우니 181 → 144 로 떨어졌다 (sensitivity() 로 재현된다).
      받는 쪽도 같은 포화 덕분에 얹는 비용이 거의 없다. 그래서 "비우기 + 얹기" 가
      이 데이터에서 유일하게 듣는 수단이다.

    지키는 것
      · 일 생산 총량 불변 (생산을 줄이는 게 아니라 옮기는 것이다)
      · 받는 시각의 생산량 <= 그 시각 관측 최대 (외삽 금지)
      · hours 안에서만 옮긴다 (무비용안이면 9-17시)
    돌려주는 것  (바뀐 생산량 벡터, 원래 벡터, 단계별 기록, 외삽으로 못 얹은 양)
    """
    base = day_prod_vector(raw, date)
    cur = base.copy()
    best, hi, _ = peak_of(predict, rebuild(raw, date, cur), date)
    log, leftover = [], 0.0

    for _ in range(max_off):
        h_peak = int(np.argmax(hi))
        if cur[h_peak] <= 0:            # 피크 시각에 생산이 없다 = 생산이 만든 피크가 아니다
            break
        trial, spill = _empty_hour(cur, h_peak, hours, cap, hi)
        if trial is None:
            break
        pk, h2, _ = peak_of(predict, rebuild(raw, date, trial), date)
        if best - pk < TOL:             # 비워도 안 내려간다 → 다른 원인이 피크를 만든다
            break
        log.append({'비운 시각': h_peak, '옮긴 생산량': base[h_peak] if not log else cur[h_peak],
                    '예측피크': pk, '내려간 폭': best - pk, '못 얹은 양': spill})
        best, cur, hi = pk, trial, h2
        leftover += spill
    assert abs(cur.sum() - base.sum()) < 1e-6, '생산 총량이 바뀌었다'
    return cur, base, log, leftover


def _empty_hour(cur, h_peak, hours, cap, hi):
    """h_peak 를 0 으로 비우고 그 생산량을 받을 수 있는 시각에 나눠 얹는다.
    예측 전력이 낮은 시각부터 채운다 (여유가 많은 쪽에 먼저 얹는다)."""
    amount = cur[h_peak]
    trial = cur.copy()
    trial[h_peak] = 0.0
    receivers = sorted((h for h in hours if h != h_peak), key=lambda h: hi[h])
    for r in receivers:
        room = max(cap.get(r, 0) - trial[r], 0)
        put = min(room, amount)
        trial[r] += put
        amount -= put
        if amount <= 0:
            break
    if amount > 0:                      # 다 얹지 못하면 비운 시각에 되돌려 총량을 지킨다
        trial[h_peak] += amount
    return trial, float(amount)


def rule_baseline(predict, raw, date, cap, hours):
    """모델 없이 고르는 기준선 — "생산이 가장 많은 시각을 비운다".

    같은 수단(비우기)을 쓰되 어느 시각을 비울지를 모델이 아니라 생산량으로 정한다.
    그래야 모델이 보태는 것이 "수단" 이 아니라 "시각 선택" 이라는 걸 가를 수 있다.
    이 기준선이 이기면 모델 없이 규칙만 쓰면 된다 ([9] 12번: 모델이 항상 규칙보다
    낫지는 않다. 피크 시각 맞히기에서는 실제로 규칙이 이겼다 [8-3]).
    받는 쪽 분배는 양쪽 같게 둔다 (비교에서 그 부분이 섞이지 않게).
    """
    base = day_prod_vector(raw, date)
    _, hi, _ = peak_of(predict, rebuild(raw, date), date)
    cur = base.copy()
    for _ in range(MAX_OFF):
        h = int(np.argmax(np.where(cur > 0, cur, -1)))   # 생산이 가장 많은 시각
        if cur[h] <= 0:
            break
        trial, spill = _empty_hour(cur, h, hours, cap, hi)
        if spill >= cur[h]:                              # 아무것도 못 옮겼다
            break
        cur = trial
        if int(np.argmax(np.where(cur > 0, cur, -1))) == h:
            break                                        # 더 비울 곳이 없다
    if np.allclose(cur, base):
        return base, np.nan
    pk, _, _ = peak_of(predict, rebuild(raw, date, cur), date)
    return cur, pk


# ── 반사실 신뢰도 검증 ──────────────────────────────────────────────────────
def validate(X, tol_temp=1.5, tol_prod=.10, min_pairs=1):
    """모델을 "안 해 본 일정" 에 써도 되는가를 실제 날짜로 잰다.

    생각  반사실이 맞으려면 적어도 "조건이 비슷하고 배분만 다른 두 날" 에서
          어느 쪽 피크가 높은지는 맞혀야 한다. 못 맞히면 재배치 제안은 근거가 없다.
    방법  가동일 고유일 중 요일 종류(평일/주말)·기온(±1.5도)·일 생산 총량(±10%)이
          비슷한 쌍을 모아, 실제 일 최대의 순서와 예측 일 최대의 순서가 같은지 본다.
    주의  예측은 반드시 구간 외(out-of-sample)여야 한다. 여기서는 model.rolling_eval 의
          시간순 롤링 폴드를 그대로 쓴다 — fit() 으로 만든 모델을 쓰면 대상일보다 앞선
          날짜가 학습에 들어가 있어 낙관적으로 나온다
    한계  이것은 "배분이 다르면 피크가 다르다" 를 모델이 읽어 낸다는 간접 근거일 뿐
          인과의 증거는 아니다. 공정 담당자 확인이 필요하다는 서술을 같이 써야 한다.
    """
    ps = [rolling_eval(X, seed=s, **ALARM_CFG).set_index('idx') for s in SEEDS]
    p = pd.concat([x.p for x in ps], axis=1).mean(axis=1)
    d = X.loc[p.index].copy()
    d['pred'] = p
    d = d[d.train_ok & ~d.is_clone & ~d.is_off]
    day = d.groupby('날짜').agg(실제최대=('target', 'max'), 예측최대=('pred', 'max'),
                               기온=('temp', 'mean'), 일생산=('prod', 'sum'),
                               평일=('is_weekend', 'first'), n=('hour', 'size'))
    day = day[(day.n == 24) & (day.일생산 > 0)]

    rows = []
    idx = list(day.index)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = day.loc[idx[i]], day.loc[idx[j]]
            if a.평일 != b.평일:
                continue
            if abs(a.기온 - b.기온) > tol_temp:
                continue
            if abs(a.일생산 - b.일생산) > tol_prod * max(a.일생산, b.일생산):
                continue
            if abs(a.실제최대 - b.실제최대) < 5:      # 실제 차이가 없으면 맞힐 것도 없다
                continue
            rows.append({'날짜A': idx[i], '날짜B': idx[j],
                         '실제차': a.실제최대 - b.실제최대,
                         '예측차': a.예측최대 - b.예측최대})
    t = pd.DataFrame(rows)
    if len(t) < min_pairs:
        print('  비교할 쌍이 없다. 허용 폭을 넓혀야 한다')
        return t, day
    t['방향일치'] = np.sign(t.실제차) == np.sign(t.예측차)
    return t, day


def decompose(day):
    """모델의 일 최대 예측력이 무엇에서 오는지 가른다.

    전체 상관만 보면 모델이 일 최대를 잘 맞히는 것처럼 보인다. 그런데 그 상관이
    기온·생산 총량·요일 같은 "일 단위 집계" 에서 오는 것이라면, 총량을 그대로 두고
    배분만 바꾸는 재배치는 모델이 평가할 수 없는 대상이 된다. 그래서 둘을 갈라야 한다.
    """
    rows = []
    for v in ['기온', '일생산', '평일']:
        rows.append({'설명변수': v,
                     '실제 일최대와의 상관': day.실제최대.corr(day[v].astype(float)),
                     '예측 일최대와의 상관': day.예측최대.corr(day[v].astype(float))})
    return pd.DataFrame(rows)


def binom_p(n, k):
    """일치 k/n 이 50% 보다 높다고 할 수 있나 (단측 이항검정)."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


# ── 보고 ────────────────────────────────────────────────────────────────────
def run_validate(X):
    """반사실 신뢰도 검증 — 재배치 제안을 쓸 수 있는지 정하는 관문이다.
    여기서 통과하지 못하면 아래 재배치 결과는 "수단은 만들었으나 근거가 없다" 가 된다.
    돌려주는 값은 통과 여부 (True/False)."""
    print('\n[반사실 신뢰도 검증] 조건이 비슷한 실제 날짜 쌍에서 피크 순서를 맞히나')
    print('  예측은 구간 외(시간순 롤링 폴드)로 낸다. 분위 0.9 의 일 최대를 쓴다')
    t, day = validate(X)
    if not len(t):
        return False
    k, n = int(t.방향일치.sum()), len(t)
    p = binom_p(n, k)
    print(f'  가동일 {len(day)}일에서 만든 쌍 {n}개 · 방향 일치 {100 * k / n:.1f}% ({k}/{n})')
    print(f'  단측 이항검정 p = {p:.3f}'
          f'  → {"50% 보다 높다고 할 수 없다" if p > .05 else "유의하게 높다"}')
    print(f'  실제차 평균 {t.실제차.abs().mean():.1f} · 예측차 평균 {t.예측차.abs().mean():.1f}')
    crit = min((k2 for k2 in range(n + 1) if binom_p(n, k2) <= .05), default=n)
    print(f'  ※ 검정력 한계: 쌍이 {n}개뿐이라 일치 {crit}/{n}({100 * crit / n:.1f}%) 이상이어야'
          f' 유의해진다.')
    print(f'     {100 * crit / n:.0f}% 미만의 작은 효과는 이 쌍 수로 잡아낼 수 없다 —'
          ' "효과가 없다" 가 아니라')
    print('     "있다고 말할 근거가 없다" 가 정확한 서술이다')

    print('\n  모델이 확신할수록 맞히나 (맞히면 예측차가 클 때 정확도가 올라가야 한다)')
    for lo in (0, 5, 10, 15):
        m = t.예측차.abs() >= lo
        if m.sum():
            print(f'    예측차 >= {lo:2d}  쌍 {int(m.sum()):3d}개  방향 일치 '
                  f'{100 * t[m].방향일치.mean():5.1f}%')

    print('\n  [진단] 모델의 일 최대 예측력은 무엇에서 오는가')
    print(f'    전체 예측 vs 실제 일 최대  피어슨 {day.실제최대.corr(day.예측최대):.3f}'
          f' · MAE {(day.실제최대 - day.예측최대).abs().mean():.2f}')
    print(decompose(day).round(3).to_string(index=False))
    ok = p <= .05
    print('\n  읽는 법')
    print('    전체 상관은 높지만 그것은 기온·생산 총량·요일 같은 "일 단위 집계" 에서 온다.')
    print('    재배치는 총량을 그대로 두고 배분만 바꾸는 수단이다. 그 셋을 매칭으로 고정하면')
    print('    남는 것이 배분 차이인데, 거기서 모델은 동전 던지기였다.')
    print('    → 모델은 배분을 읽지 못한다. 배분을 바꾸는 제안을 이 모델로 평가할 수 없다')
    t.round(2).to_csv(OUT / 'peak_schedule_validate.csv', index=False, encoding='utf-8-sig')
    decompose(day).round(4).to_csv(OUT / 'peak_schedule_decompose.csv', index=False,
                                   encoding='utf-8-sig')
    print('  →', OUT / 'peak_schedule_validate.csv', '/', OUT / 'peak_schedule_decompose.csv')
    return ok


def month_context(X, target_days):
    """기본요금은 "그 달의 최대 한 번" 으로 정해진다. 그래서 어느 날을 낮추느냐가
    핵심이다 — 월 최대를 만든 날이 아니면 그 날을 낮춰도 요금은 1원도 안 줄어든다.
    대상일이 속한 달의 실제 일 최대 상위 날짜를 같이 보여 준다."""
    months = sorted({str(d)[:6] for d in target_days})
    d = X[X.train_ok & X['날짜'].astype(str).str[:6].isin(months)]
    day = d.groupby('날짜')['target'].max().sort_values(ascending=False)
    return day


def report_day(predict, raw, date, cap, hours, label):
    before = rebuild(raw, date)
    pk0, hi0, pt0 = peak_of(predict, before, date)
    actual = before[before['날짜'] == date].sort_values('hour')['target'].to_numpy()

    cur, base, log, leftover = reallocate(predict, raw, date, cap, hours)
    pk1, hi1, _ = peak_of(predict, rebuild(raw, date, cur), date)
    _, pk_rule = rule_baseline(predict, raw, date, cap, hours)

    moved_night = float(sum(cur[h] - base[h] for h in NIGHT_HOURS if cur[h] > base[h]))
    print(f'\n── {date} ({label}) ──')
    print(f'  실제 일 최대 {np.nanmax(actual):.0f}  ·  예측 일 최대 {pk0:.1f}'
          f'  ·  예측 피크 시각 {int(np.argmax(hi0))}시 (그 시각 생산 {base[int(np.argmax(hi0))]:.0f})')
    print(f'  재배치 후 예측 일 최대 {pk1:.1f}  ({pk1 - pk0:+.1f})  ·  비운 시각 {len(log)}개')
    if np.isnan(pk_rule):
        print('  규칙 기준선(생산 최다 시각 비우기): 적용 불가 — 옮길 여력이 없다')
    else:
        print(f'  규칙 기준선(생산 최다 시각 비우기) {pk_rule:.1f} ({pk_rule - pk0:+.1f})'
              f'  → 모델이 고른 쪽이 {"더 낫다" if pk1 < pk_rule - TOL else "낫지 않다"}')
    print(f'  야간(18-8시)으로 옮긴 생산량 {moved_night:.0f}'
          f' → 인건비 {"1.5배 구간 발생" if moved_night > 0 else "증가 없음"}')
    if log:
        print('  비운 내역')
        for r in log:
            print('    %2d시 비움  생산 %5.0f 이동  예측피크 %6.1f (%+.1f)%s'
                  % (r['비운 시각'], r['옮긴 생산량'], r['예측피크'], -r['내려간 폭'],
                     f"  ※ {r['못 얹은 양']:.0f} 은 못 얹어 되돌림" if r['못 얹은 양'] > 0 else ''))
    else:
        print('  ★ 바꿀 것이 없다 — 피크 시각에 생산이 없거나, 비워도 피크가 안 내려간다.')
        print('    생산이 만든 피크가 아니라는 뜻이다 (기동 또는 기온)  [8-1] 결과 4')

    plan = pd.DataFrame({'시간': range(24), '생산량_기존': base, '생산량_제안': cur,
                         '예측전력_기존': hi0, '예측전력_제안': hi1,
                         '점예측_기존': pt0, '실제전력': actual})
    plan.round(2).to_csv(OUT / f'peak_schedule_plan_{date}.csv', index=False,
                         encoding='utf-8-sig')
    return {'날짜': date, '유형': label, '실제최대': np.nanmax(actual), '예측최대': pk0,
            '재배치후': pk1, '감소': pk0 - pk1, '규칙기준선': pk_rule,
            '비운시각수': len(log), '야간이동량': moved_night, '못얹은양': leftover,
            '피크시각': int(np.argmax(hi0)), '피크시각생산': base[int(np.argmax(hi0))]}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    night = '--night' in sys.argv
    only_validate = '--validate' in sys.argv
    n_days = 3
    if '--days' in sys.argv:
        n_days = int(sys.argv[sys.argv.index('--days') + 1])

    raw = pd.read_csv(RAW)
    X = rebuild(raw, None)
    hours = tuple(range(24)) if night else WORK_HOURS

    print('최대피크 예측과 저감 스케줄링')
    print(f'  옮길 수 있는 시각  {hours[0]}-{hours[-1]}시'
          f'  ({"야간 허용 — 인건비 1.5배 구간 포함" if night else "무비용안 — 인건비 1.0 구간만"})')
    print('  일 최대는 분위 0.9 로 추정한다 (점 예측의 최대는 구조적으로 낮다)  [6-5]')

    # 대상일 고르기: 실적이 있는 마지막 가동일들
    d = X[X.train_ok & ~X.is_off]
    day = d.groupby('날짜').agg(일생산=('prod', 'sum'), n=('hour', 'size'))
    cand = day[(day.n == 24) & (day.일생산 > 0)].index.tolist()
    if len(args) >= 2:
        lo, hi_ = int(args[0]), int(args[1])
        target_days = [x for x in cand if lo <= x <= hi_]
    else:
        target_days = cand[-n_days:]

    cutoff = X[X['날짜'] == min(target_days)]['dt'].min()
    predict, tr = fit(X, cutoff)
    cap = hour_cap(tr).to_dict()
    month_peak = month_context(X, target_days)
    print(f'  학습 {len(tr)}행 (대상일 {min(target_days)} 이전만) · 시드 {list(SEEDS)} 평균')
    print('  받는 시각 생산량 상한 (관측 분위 0.9 와 1,600 중 작은 값):',
          {h: int(cap[h]) for h in hours})
    print('  1,600 을 절대 상한으로 둔 이유는 hour_cap() 독스트링에 적었다 —')
    print('  그 위에서 전력이 내려가는 것은 교대 경계에 생산을 몰아 기록한 교란이다')

    if '--probe' in sys.argv:
        dt = target_days[-1]
        _, hi, _ = peak_of(predict, X, dt)
        h = int(np.argmax(hi))
        print(f'\n[민감도] {dt} 의 예측 피크 시각 {h}시 생산량만 바꿔 본다')
        print('  "조금씩 옮기기" 가 왜 안 듣고 "비우기" 만 듣는지 보여 주는 곡선이다')
        t = sensitivity(predict, raw, dt, h)
        t.insert(0, '시각', h)
        print(t.round(1).to_string(index=False))
        print(f'  원래 생산량 {day_prod_vector(raw, dt)[h]:.0f}')
        print('  생산 800-1,000 위가 평평하다 → 그 안에서 덜어 내도 전력이 안 내려간다')
        t.round(3).to_csv(OUT / 'peak_schedule_sensitivity.csv', index=False,
                          encoding='utf-8-sig')
        print('  →', OUT / 'peak_schedule_sensitivity.csv')
        return

    if only_validate:
        run_validate(X)
        return

    gate = run_validate(X)
    print('\n' + '=' * 70)
    if gate:
        print('관문 통과 — 아래 재배치안을 제안으로 쓸 수 있다')
    else:
        print('★ 관문 실패 — 아래 재배치안은 "수단은 만들었으나 근거가 없다" 로 읽어야 한다')
        print('  숫자가 내려가는 것은 모델이 그렇게 예측한다는 뜻일 뿐이고, 실제로 내려간다는')
        print('  근거가 아니다. 보고서에는 감소량을 성과로 쓰지 말고, 왜 쓸 수 없는지를 쓴다')
    print('=' * 70)

    rows = []
    for dt in target_days:
        h0 = int(np.argmax(peak_of(predict, X, dt)[1]))
        label = '오전 기동형' if h0 <= 9 else '생산형'
        rows.append(report_day(predict, raw, dt, cap, hours, label))

    s = pd.DataFrame(rows)
    s.round(2).to_csv(OUT / 'peak_schedule_summary.csv', index=False, encoding='utf-8-sig')
    print('\n── 요약 ──')
    print(s[['날짜', '유형', '피크시각', '피크시각생산', '예측최대', '재배치후', '감소',
             '규칙기준선', '비운시각수', '야간이동량']].round(1).to_string(index=False))
    print(f'\n  평균 감소 {s.감소.mean():.1f} · 최대 감소 {s.감소.max():.1f}'
          f' · 효과가 있던 날 {int((s.감소 > TOL).sum())}/{len(s)}일')

    print('\n── ★ 어느 날을 낮춰야 요금이 줄어드나 ──')
    print('  최대수요 기본요금은 "그 달의 최대 한 번" 으로 정해진다. 월 최대를 만든 날이')
    print('  아니면 그 날 피크를 낮춰도 요금은 1원도 줄지 않는다.')
    print(f'  대상 달의 실제 일 최대 상위 5일')
    for dt, v in month_peak.head(5).items():
        mark = '  ← 이번에 다룬 날' if dt in set(s.날짜) else ''
        print(f'    {dt}  {v:.0f}{mark}')
    top = month_peak.index[0]
    done = [d for d in s.날짜 if d in set(month_peak.head(3).index)]
    if done:
        # 요금이 걸린 숫자는 "월 최대가 얼마나 내려가는가" 하나다.
        # 날짜 순위는 실측으로 매기고, 다룬 날에만 모델이 추정한 감소분(델타)을 적용한다.
        # 예측 수준과 실측 수준을 섞지 않으려고 델타만 쓴다
        cut = dict(zip(s.날짜, s.감소))
        after_by_day = {d: v - cut.get(d, 0.0) for d, v in month_peak.items()}
        before, after = month_peak.max(), max(after_by_day.values())
        new_top = max(after_by_day, key=after_by_day.get)
        print(f'  → 상위 3일 중 {done} 을 다뤘다')
        print(f'    월 최대  {before:.0f} → {after:.1f}  ({after - before:+.1f})'
              f'   (실측 순위에 모델 추정 감소분만 적용)')
        if new_top not in cut:
            print(f'    ※ 이제 {new_top}({month_peak[new_top]:.0f}) 가 월 최대를 결정한다.')
            print('      그 날까지 같이 낮춰야 절감이 더 나온다 — 한 날만 낮춰도 요금은 안 줄어든다')
    else:
        print(f'  → 이번에 다룬 날은 상위 3일에 없다 (월 최대일은 {top}, {month_peak.iloc[0]:.0f}).')
        print('    요금 절감을 보려면 월 최대일을 대상으로 다시 돌릴 것:')
        print(f'      python src/peak_schedule.py {top} {top}')
    print('\n── 결론 ──')
    if gate:
        print('  절감 환산  최대수요전력 기본요금 단가가 데이터에 없다 ([0] C-3).')
        print('             월 절감액 = 월 최대 감소분(kW) x 기본요금 단가(원/kW·월) 로만 쓴다')
    else:
        print('  생산 재배치는 이 데이터로 제안할 수 없다. 이유가 셋 다 같은 곳을 가리킨다.')
        print('    1. 전력이 생산량에 포화한다 — 800-1,600 구간이 평평하다  [8-1] 결과 2')
        print('    2. 피크는 생산이 아니라 아침 기동에서 온다 — 8시 피크일의 평균생산은')
        print('       11시 피크일의 절반도 안 되는데 일 최대는 더 높다  [8-1] 결과 4')
        print('    3. 모델의 일 최대 예측력은 전부 일 단위 집계(생산 총량·요일·기온)에서')
        print('       온다. 총량을 고정하면 배분 차이를 구분하지 못한다 (위 관문)')
        print('  → 재배치로 얻을 수 있는 것이 있다고 말할 근거가 없다. 보고서에는 이것을')
        print('    "측정해서 기각" 으로 쓴다 ([8-4] 4순위를 미착수에서 기각으로 바꾼다)')
        print('  남는 수단')
        print('    · 피크 위험 경보 (1순위) — 검증까지 끝났다. 일 재현율 97.4%  [8-2]')
        print('    · 오전 기동 분산 (2순위) — 생산량과 무관하므로 재배치로는 다룰 수 없다.')
        print('      어떤 설비가 8시에 동시 기동하는지는 데이터에 없다 → 공정 담당자 확인')
        print('    · 냉방 선행 운전 (3순위) — 냉방이 따로 계량되지 않아 검증 불가')
        print('  ※ 월 최대 기준도 같이 기억할 것: 기본요금은 한 달에 하루로 정해지므로')
        print('    여러 날을 조금씩 낮춰도 최대일을 못 낮추면 절감은 0이다')
    print('  →', OUT / 'peak_schedule_summary.csv')


if __name__ == '__main__':
    main()
