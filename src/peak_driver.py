"""최대피크 위험조건 분석 — 피크를 만드는 것은 무엇인가 (과제 제목의 뒤쪽 절반).

  D24(피크의 동인은 생산량이 아니라 기온)의 근거를 재현 가능한 형태로 고정한다.
  인수인계(2026-09-30) [2] 의 수치가 어떤 기준에서 나온 것인지 코드로 남겨 두어야
  보고서를 쓸 때 다시 세지 않아도 된다.

  ★ 이 분석의 결론은 "어떤 기준으로 재느냐"에 크게 흔들린다. 그래서 하나의 숫자를
    내지 않고 기준을 바꿔 가며 전부 낸다. 결론은 여러 기준에서 같은 방향이 나오는
    것만 쓴다 (작업내역(조선제).txt [17] 13-3).

실행: python src/peak_driver.py
출력: outputs/peak_driver_corr.csv   기준별 일 단위 상관 (생산 vs 기온)
      outputs/peak_driver_hours.csv  피크 시간대와 그 외 고부하 시간대 비교
      outputs/peak_driver_sat.csv    생산량 구간별 평균 전력 (포화 확인)
"""
import itertools
import numpy as np, pandas as pd

from features import build
from model import CORE_FOLDS
from preprocess import ROOT

OUT_CORR = ROOT / 'outputs' / 'peak_driver_corr.csv'
OUT_HOUR = ROOT / 'outputs' / 'peak_driver_hours.csv'
OUT_SAT = ROOT / 'outputs' / 'peak_driver_sat.csv'

TOP_N = 30      # "피크 시간대" 로 볼 상위 몇 개 행인가 (인수인계 [2] 와 같은 기준)
HIGH = 150      # "그 외 고부하" 의 하한. 전력 150 이상


def corr_table(X):
    """기준을 바꿔 가며 일 최대전력이 생산량·기온과 얼마나 같이 움직이는지 잰다.

    피어슨은 값 자체의 직선 관계, 스피어만은 순위만 본다. 생산량은 800 부근에서
    포화하므로 직선 관계가 아니고, 그래서 두 방법의 답이 갈린다. 둘 다 낸다.
    """
    rows = []
    ranges = {'전체 1 to 9월': X.ym.notna(), 'CORE 7 to 9월': X.ym.isin(CORE_FOLDS)}
    flags = {'train_ok': X.train_ok, 'strict': X.train_ok_strict}
    subs = {'가동일': ~X.is_off, '가동일＆고유일': (~X.is_off) & (~X.is_clone)}
    for (rn, rm), (fn, fm), (sn, sm) in itertools.product(ranges.items(), flags.items(), subs.items()):
        o = X[rm & fm & sm]
        if len(o) < 100:
            continue
        g = o.groupby('날짜')
        d = pd.DataFrame({'일최대전력': g.target.max(), '일총생산': g['prod'].sum(),
                          '일최고기온': g.temp.max(), '일평균기온': g.temp.mean()}).dropna()
        for meth in ('pearson', 'spearman'):
            c = d.corr(meth)['일최대전력']
            rows.append({'구간': rn, '학습행': fn, '대상': sn, '방법': meth,
                         '생산': round(c['일총생산'], 3), '최고기온': round(c['일최고기온'], 3),
                         '평균기온': round(c['일평균기온'], 3), 'n일': len(d)})
    t = pd.DataFrame(rows)
    # 기온이 생산을 앞지르는 기준이 어디인지 한눈에 보이게 표시한다
    t['기온우세'] = np.where(t[['최고기온', '평균기온']].max(axis=1) > t['생산'], '★', '')
    return t


def hour_table(X):
    """피크가 난 시간대와 그 외 고부하 시간대를 나란히 놓는다.

    피크에서 생산이 오히려 적다면 "생산을 옮겨 피크를 깎는다"는 설계가 성립하지 않는다.
    """
    rows = []
    for rn, rm in [('전체 1 to 9월', X.ym.notna()), ('CORE 7 to 9월', X.ym.isin(CORE_FOLDS))]:
        o = X[rm & X.train_ok & ~X.is_off]
        top = o.nlargest(TOP_N, 'target')
        rest = o[(o.target >= HIGH) & (~o.index.isin(top.index))]
        for lb, s in [(f'피크 상위 {TOP_N}', top), (f'그 외 고부하 ({HIGH}+)', rest)]:
            rows.append({'구간': rn, '집단': lb, 'n': len(s),
                         '평균생산': round(s['prod'].mean(), 0),
                         '평균전력': round(s.target.mean(), 1),
                         '평균기온': round(s.temp.mean(), 1)})
    return pd.DataFrame(rows)


def sat_table(X):
    """생산량을 늘리면 전력이 계속 따라 오르는가 (= 생산 재배치로 피크를 깎을 수 있는가)."""
    o = X[X.train_ok & ~X.is_off]
    b = pd.cut(o['prod'], [-1, 0, 200, 400, 600, 800, 1000, 1200, 1600, 10 ** 9],
               labels=['0', '1-200', '200-400', '400-600', '600-800',
                       '800-1000', '1000-1200', '1200-1600', '1600+'])
    t = o.groupby(b, observed=True).agg(n=('target', 'size'), 평균전력=('target', 'mean'),
                                        평균기온=('temp', 'mean')).round(1)
    t['직전구간대비'] = t.평균전력.diff().round(1)
    return t.reset_index().rename(columns={'prod': '시간당 생산량'})


if __name__ == '__main__':
    X = build()
    X['ym'] = X['dt'].dt.to_period('M').astype(str)

    C = corr_table(X)
    C.to_csv(OUT_CORR, index=False, encoding='utf-8-sig')
    print('[1] 일 최대전력은 무엇과 같이 움직이는가 (기준별)')
    print(C.to_string(index=False))
    n_temp = (C.기온우세 == '★').sum()
    print(f'\n  → {len(C)}개 기준 중 기온이 생산을 앞지른 것은 {n_temp}개.')
    print('    피어슨(값)에서는 생산이 앞서고 스피어만(순위)에서는 기온이 따라붙는다.')
    print('    생산량이 800 부근에서 포화하기 때문이다 ([3] 참조) — 많이 만든다고')
    print('    전력이 비례해 오르지 않으므로 직선 관계가 약해진다.')

    H = hour_table(X)
    H.to_csv(OUT_HOUR, index=False, encoding='utf-8-sig')
    print('\n[2] 피크 시간대 vs 그 외 고부하 시간대')
    print(H.to_string(index=False))
    print('\n  → CORE 구간에서 피크 시간대의 생산량이 오히려 적다.')
    print('    피크를 만든 것은 "더 많이 만들어서"가 아니다.')

    S = sat_table(X)
    S.to_csv(OUT_SAT, index=False, encoding='utf-8-sig')
    print('\n[3] 생산량을 늘리면 전력이 계속 오르는가 (포화 확인)')
    print(S.to_string(index=False))
    print('\n  → 800 이상에서는 생산을 더 해도 전력이 거의 오르지 않는다.')
    print('    피크 저감 모형의 결정변수를 생산량으로 두면 안 되는 이유다 (인수인계 [2] 4순위).')
    print(f'\n→ {OUT_CORR}\n→ {OUT_HOUR}\n→ {OUT_SAT}')
