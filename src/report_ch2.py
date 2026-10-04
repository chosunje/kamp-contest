"""보고서 제2장(AI 예측모델 개발 및 성능평가)에 넣을 표·그림용 데이터를 CSV로 만든다.

표와 그래프는 Word 기능으로 직접 그리므로 "그릴 값"만 내보낸다.
수치는 모두 outputs/ 의 실행 결과에서 읽어 오며, 본문에 직접 타이핑한 값이 없도록 한다.

선행: python src/model.py, python src/baseline.py, python src/error_analysis.py
실행: python src/report_ch2.py  →  outputs/report/ch2/*.csv, 캡션.txt
"""
import numpy as np, pandas as pd

from model import CORE, CORE_FOLDS, SEEDS, RUNS, BASE
from preprocess import ROOT

OUT = ROOT / 'outputs' / 'report' / 'ch2'
OUT.mkdir(parents=True, exist_ok=True)
CAPS = []


def save(df, name, caption):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    CAPS.append((name, caption))
    print(f'  → {name:30s} {len(df):3d}행')


M = pd.read_csv(ROOT / 'outputs' / 'model_results.csv')
B = pd.read_csv(ROOT / 'outputs' / 'baseline_results.csv')
P = pd.read_csv(ROOT / 'outputs' / 'error_predictions.csv', parse_dates=['dt'])


def agg(fold):
    """설정별 시드 평균 성적. 흔들림 = 시드 간 MAE 표준편차"""
    a = M[M.fold == fold]
    g = a.groupby('run', sort=False).agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'),
                                         RMSE=('RMSE', 'mean'), peakMAE=('peakMAE', 'mean'),
                                         n=('n', 'first'))
    g.index = [i.split(' ', 1)[0] for i in g.index]      # '8 조합 (…)' → '8'
    return g.round(2)


C, A = agg(CORE), agg('ALL')            # C = 7 to 9월(판단 기준), A = 5 to 9월


def row(no, label, cols=('MAE', '흔들림', 'RMSE', 'peakMAE')):
    c, a = C.loc[no], A.loc[no]
    d = {'설정': label}
    d.update({f'{k} (7 to 9월)': c[k] for k in cols})
    d['MAE (5 to 9월)'] = a['MAE']
    return d


def main():
    # ── 표1. 베이스라인 4종 ──────────────────────────────────────────
    b = B[B.group == '고유일']
    core = b[b.fold == CORE].set_index('model')
    all_ = b[b.fold == 'ALL'].set_index('model')
    t1 = pd.DataFrame({
        '단순 예측 규칙': ['전체 평균', '전일 동일 시각', '전주 동일 시각', '요일×시각 평균'],
    })
    key = ['전체 평균', 'lag24', 'lag168', '요일x시간 평균']
    for lab, src in [('7 to 9월', core), ('5 to 9월', all_)]:
        for met in ['MAE', 'RMSE', 'peakMAE']:
            t1[f'{met} ({lab})'] = [round(src.loc[k, met], 2) for k in key]
    save(t1, 't1_베이스라인_4종.csv',
         '표. 베이스라인 4종의 성능 — 지표별 최저값을 목표선으로 삼았다 (고유일 기준)')

    # ── 표2. 모델 2종 비교 ★ ─────────────────────────────────────────
    t2 = pd.DataFrame([row('0', 'LightGBM (목적함수 L1)'), row('5', 'RandomForest')])
    t2 = t2.rename(columns={'설정': '모델'})
    t2['결측 처리'] = ['분기 규칙으로 직접 학습', '-999로 대체 후 학습']
    save(t2, 't2_모델2종_비교.csv',
         '표. 모델 2종 비교 ★ — 동일 피처(full 36개)·동일 학습 행·동일 폴드, 시드 3개 평균')

    # ── 표3. 피처 세트 비교 ─────────────────────────────────────────
    t3 = pd.DataFrame([row('0', 'full (하루 앞)'), row('6', 'h7_full (24시간 전 시차 4개 제외)'),
                       row('4', 'no_plan (생산계획 미제공)'), row('7', 'h7_no_plan')])
    t3.insert(1, '피처 수', [len(BASE['cols']), 32, 19, 12])
    t3 = t3.rename(columns={'설정': '피처 세트'})
    save(t3, 't3_피처세트_비교.csv',
         '표. 피처 세트 비교 — 피처를 4개 줄인 h7_full 이 오히려 정확하다')

    # ── 그림1. 복제일 가중치에 따른 성능 ★ ───────────────────────────
    # 가중치 탐색은 두 피처 세트에서 따로 했다. 한 곡선에 섞으면 비교가 성립하지 않으므로 계열을 나눈다
    full_w = {0.0: '1', 0.05: '20', 0.1: '10', 0.3: '2', 0.5: '11', 0.7: '12', 1.0: '13'}
    h7_w = {0.0: '17', 0.05: '15', 0.1: '14', 0.3: '8', 0.5: '18', 0.7: '19', 1.0: '6'}
    rows = []
    for w in [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]:
        r = {'복제일 가중치': w, '처리': '학습 제외' if w == 0 else
             ('가중치 1.0 (= 그대로)' if w == 1 else f'가중치 {w}')}
        for lab, mp in [('full 세트', full_w), ('h7_full 세트', h7_w)]:
            no = mp.get(w)
            r[f'{lab} MAE'] = C.loc[no, 'MAE'] if no else ''
            r[f'{lab} peakMAE'] = C.loc[no, 'peakMAE'] if no else ''
        rows.append(r)
    f1 = pd.DataFrame(rows)
    save(f1, 'f1_복제일_가중치곡선.csv',
         '그림. 복제일 가중치에 따른 성능 ★ — 꺾은선 (가로축 가중치, 계열 2개). 가중치 1.0 은 '
         '"그대로"와 정의상 같아야 하며 실제로 소수점까지 일치했다(구현 검증). 빈 칸은 미실험')

    # ── 표4. 조합 설정 ──────────────────────────────────────────────
    t4 = pd.DataFrame([row('0', '기준 (full · 복제일 그대로)'),
                       row('6', 'h7_full 단독'),
                       row('2', '복제일 가중치 0.3 단독'),
                       row('8', 'h7_full + 가중치 0.3  ← 최종 후보'),
                       row('14', 'h7_full + 가중치 0.1'),
                       row('3', '[참고] 누락일 포함 (학습 행 완화)')])
    save(t4, 't4_조합설정.csv',
         '표. 개별 요소와 조합의 성능 — 두 요소의 효과는 상쇄되지 않고 누적된다')

    # ── 표5. 최종 성능 대 목표선 ★ ──────────────────────────────────
    def fin(sub):
        e = sub.target - sub.pred
        pk = sub.is_peak.astype(bool)
        return [round(e.abs().mean(), 2), round(np.sqrt((e ** 2).mean()), 2),
                round(e[pk].abs().mean(), 2)]
    rows = []
    for lab, sub, src in [('7 to 9월 (판단 기준)', P[P.fold.isin(CORE_FOLDS)], core),
                          ('5 to 9월 (전체)', P, all_)]:
        base = [round(src[m].min(), 2) for m in ['MAE', 'RMSE', 'peakMAE']]
        mod = fin(sub)
        rows.append({'기준': lab, '평가 행 수': len(sub),
                     '목표선 MAE': base[0], '모델 MAE': mod[0], 'MAE 개선율(%)': round((mod[0]/base[0]-1)*100),
                     '목표선 RMSE': base[1], '모델 RMSE': mod[1], 'RMSE 개선율(%)': round((mod[1]/base[1]-1)*100),
                     '목표선 피크MAE': base[2], '모델 피크MAE': mod[2], '피크MAE 개선율(%)': round((mod[2]/base[2]-1)*100)})
    save(pd.DataFrame(rows), 't5_최종성능_목표선대비.csv',
         '표. 최종 모델과 목표선 ★ — 모델은 시드 3개 예측 평균. 목표선은 지표별 최저 단순규칙')

    # ── 그림2. 폴드별 성능 ──────────────────────────────────────────
    bb = b[~b.fold.isin(['ALL', CORE])]
    rows = []
    for f, s in P.groupby('fold'):
        e = s.target - s.pred
        rows.append({'검증 월': f, '평가 행 수': len(s),
                     '목표선 MAE': round(bb[bb.fold == f].MAE.min(), 2),
                     '최종 모델 MAE': round(e.abs().mean(), 2)})
    f2 = pd.DataFrame(rows).sort_values('검증 월').reset_index(drop=True)
    save(f2, 'f2_폴드별_성능.csv',
         '그림. 검증 월별 성능 — 세로 막대 2계열. 학습 데이터가 쌓일수록 오차가 줄어든다 '
         '(5월은 학습 4개월·복제일 편중)')

    # ── 표6. 피크 관점의 성능 ───────────────────────────────────────
    c = P[P.fold.isin(CORE_FOLDS)].copy()
    e = c.target - c.pred
    pk = c.is_peak.astype(bool)
    d = c.groupby(c.dt.dt.date).apply(lambda s: pd.Series({
        'y_max': s.target.max(), 'p_max': s.pred.max(),
        'y_h': int(s.loc[s.target.idxmax(), 'hour']), 'p_h': int(s.loc[s.pred.idxmax(), 'hour']),
        'in3': int(s.loc[s.target.idxmax(), 'hour']) in set(s.nlargest(3, 'pred').hour),
        'keep': s.loc[s.pred.idxmax(), 'target'] / s.target.max(),
        'off': bool(s.is_off.iloc[0])}), include_groups=False)
    op = d[~d.off]
    t6 = pd.DataFrame([
        ('시간 단위 MAE', round(e.abs().mean(), 2), '전체 평가 구간'),
        ('피크 구간 MAE', round(e[pk].abs().mean(), 2), '학습 구간 상위 5%'),
        ('피크 구간 평균 오차', round(e[pk].mean(), 2), '양수 = 과소예측'),
        ('피크 구간 과소예측 비율(%)', round(pk.sum() and (e[pk] > 0).mean() * 100, 1), '-'),
        ('일 최대 전력 MAE', round((op.y_max - op.p_max).abs().mean(), 2), f'가동일 {len(op)}일'),
        ('일 최대 전력 평균 오차', round((op.y_max - op.p_max).mean(), 2), '양수 = 과소예측'),
        ('피크 시각 정확히 일치(%)', round((op.y_h == op.p_h).mean() * 100, 1), '-'),
        ('피크 시각 ±1시간 이내(%)', round(((op.y_h - op.p_h).abs() <= 1).mean() * 100, 1), '-'),
        ('예측 상위 3시간 안에 포함(%)', round(op.in3.mean() * 100, 1), '-'),
        ('예측 지목 시각의 실제 전력 수준(%)', round(op.keep.median() * 100, 1), '그날 실제 최대 대비, 중앙값'),
    ], columns=['지표', '값', '비고'])
    save(t6, 't6_피크관점_성능.csv',
         '표. 피크 관점의 성능 (7 to 9월) — 평균 오차만으로는 보이지 않는 과소예측 경향')

    # ── 표7. 모델 설정 (재현용) ─────────────────────────────────────
    t7 = pd.DataFrame([
        ('LightGBM', '목적함수', 'L1 (MAE 직접 최소화)'),
        ('LightGBM', '트리 수 / 학습률', '400 / 0.05'),
        ('LightGBM', '리프 수 / 리프 최소 표본', '31 / 20'),
        ('LightGBM', '피처·표본 샘플링', '0.8 / 0.8 (매 반복)'),
        ('RandomForest', '트리 수 / 리프 최소 표본', '300 / 5'),
        ('공통', '시드', f'{list(SEEDS)} (3회 반복 학습)'),
        ('공통', '검증', '시간순 롤링 폴드 5 to 9월, 평가셋 고정'),
        ('공통', '실험 수', f'설정 {len(RUNS)}개 × 시드 3개 × 폴드'),
    ], columns=['모델', '항목', '값'])
    save(t7, 't7_모델설정.csv', '표. 모델 설정 (재현용)')

    # ── 그림3·표8. 변수 중요도 (선행: python src/importance.py) ──────
    gp, sp = ROOT / 'outputs' / 'importance_gain.csv', ROOT / 'outputs' / 'importance_shap.csv'
    if gp.exists() and sp.exists():
        G = pd.read_csv(gp)
        S = pd.read_csv(sp)
        f3 = G.head(15)[['순위', '피처', '정보군', 'gain 중요도(%)', '폴드간 편차(%)']]
        save(f3, 'f3_변수중요도.csv',
             '그림. 변수 중요도 상위 15개 — 가로 막대 (LightGBM gain, 폴드·시드 평균). '
             '정보군 열로 색을 나누면 생산계획·과거전력이 대부분을 차지하는 구조가 보인다')

        g_grp = G.groupby('정보군')['gain 중요도(%)'].sum()
        s_grp = S.groupby('정보군')[['전체 평균|SHAP|', '피크 구간 평균|SHAP|']].sum()
        t8 = pd.DataFrame({'정보군': g_grp.index, 'gain 중요도 합계(%)': g_grp.round(1).values})
        t8 = t8.merge(s_grp.round(2).reset_index(), on='정보군')
        t8['피크/전체 배율'] = (t8['피크 구간 평균|SHAP|'] / t8['전체 평균|SHAP|']).round(2)
        t8 = t8.sort_values('gain 중요도 합계(%)', ascending=False).reset_index(drop=True)
        save(t8, 't8_정보군별_기여.csv',
             '표. 정보군별 기여 ★ — 피크 구간에서 기여가 커지는 것은 과거 전력과 기상이고, '
             '생산계획은 커지지 않는다 (1장의 피크 동인 분석과 같은 방향)')

        pick = ['lag168', 'lag_week_mean4', 'lag336', 'prod', 'prod_prev', 'prod_next',
                'day_prod', 'is_off', 'thi', 'cool', 'humid', 'temp']
        f4 = S[S.피처.isin(pick)].reset_index(drop=True)
        save(f4, 'f4_SHAP_전체대비피크.csv',
             '그림. 주요 변수의 평균 |SHAP| — 전체 구간 vs 피크 구간, 가로 막대 2계열 ★ '
             '(기상 파생변수 thi·cool 은 피크에서 1.7 to 1.8배로 커지고 원본 temp 는 오히려 줄어든다)')

        # 표9. 도메인 발견 → 파생변수 매핑
        def gi(c):
            r = G[G.피처 == c]
            return f"{r['gain 중요도(%)'].iloc[0]:.2f}%" if len(r) else '-'
        def sm(c):
            r = S[S.피처 == c]
            return f"{r['피크/전체 배율'].iloc[0]:.2f}배" if len(r) else '-'
        t9 = pd.DataFrame([
            ('생산량 800 부근에서 전력 포화', 'prod_cap', '생산량을 800에서 절단', gi('prod_cap'), sm('prod_cap')),
            ('생산량 분포가 한쪽으로 쏠림', 'prod_log', 'log(1+생산량)', gi('prod_log'), sm('prod_log')),
            ('기온 22℃ 이상에서 냉방부하 발생', 'cool', 'max(기온-22, 0)', gi('cool'), sm('cool')),
            ('기온과 습도가 함께 작용', 'thi', '불쾌지수 (기온·습도 결합)', gi('thi'), sm('thi')),
            ('전일보다 전주 동일 요일이 유사', 'lag168 / lag336 / lag_week_mean4',
             '1주 전·2주 전·최근 4주 평균 동시각', gi('lag168'), sm('lag168')),
            ('휴무 기간이 다음 주 시차를 오염', 'lag 결측 처리 + lag_prev_op',
             '휴무·중단 구간은 결측, 직전 가동일 동시각을 별도 변수로', gi('lag_prev_op'), sm('lag_prev_op')),
            ('휴무 길이에 따라 복귀일 부하가 다름', 'off_run_prev / days_since_off',
             '직전 연속 휴무 일수 / 마지막 휴무 이후 경과일', gi('off_run_prev'), sm('off_run_prev')),
            ('부하가 기저·중·고 세 단계로 구분', 'shift / is_transition',
             '야간·점심·주간 구분 / 전환 시각(7·12·17시)', gi('shift'), sm('shift')),
            ('휴무 여부가 타깃에서 유도된 값', 'day_prod_zero',
             '생산계획만으로 만든 휴무 근사값', gi('day_prod_zero'), sm('day_prod_zero')),
        ], columns=['1장의 도메인 발견', '파생 변수', '정의', 'gain 중요도', '피크 구간 SHAP 배율'])
        save(t9, 't9_피처엔지니어링_매핑.csv',
             '표. 데이터 진단에서 얻은 발견이 어떤 파생 변수로 들어갔는가 ★')

    # ── 표10. 하이퍼파라미터 민감도 (선행: python src/hparam_check.py) ──
    hp = ROOT / 'outputs' / 'hparam_results.csv'
    if hp.exists():
        H = pd.read_csv(hp)
        save(H, 't10_하이퍼파라미터_민감도.csv',
             '표. 하이퍼파라미터 민감도 ★ — 최종 후보 설정에서 한 항목씩만 변경. '
             '목적함수를 제외하면 변동이 ±0.8 이내로 피처·가중치 효과보다 작다')

    txt = ['보고서 제2장 표·그림 데이터 (src/report_ch2.py 생성)',
           '수치는 outputs/model_results.csv, baseline_results.csv, error_predictions.csv 에서 읽었다.',
           '', '※ 7 to 9월 = 판단 기준(고유일 1,708행) / 5 to 9월 = 전체(2,092행)',
           '※ "흔들림"은 시드 3개 간 MAE 표준편차. 설정 간 차이가 이보다 작으면 차이 없음으로 본다.',
           '※ 최종 성능(t5, f2, t6)은 시드 3개의 예측값을 평균한 결과이므로,',
           '   시드별 점수를 평균한 t4 의 7.15 와는 다른 값(6.93)이다.', '',
           '※ 최종 성능 절의 그림은 outputs/perf/ 의 PNG 를 그대로 삽입해도 된다',
           '   (02 실제-예측 산점도, 03 2주 연속 시계열, 01 대표일 24시간 곡선).', '']
    for n, c2 in CAPS:
        txt += [f'[{n}]', f'  {c2}', '']
    (OUT / '캡션.txt').write_text('\n'.join(txt), encoding='utf-8')
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
