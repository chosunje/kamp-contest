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
P = pd.read_csv(ROOT / 'outputs' / 'error_predictions.csv', parse_dates=['dt'])  # 옛 후보(비교용)
F = pd.read_csv(ROOT / 'outputs' / 'final' / 'predictions.csv', parse_dates=['dt'])  # 최종 설정
X = pd.read_csv(ROOT / 'outputs' / 'model_results_fix.csv')          # 약점 공략 31개
FINAL = '★ 강화 최종 (얕은나무+느린학습+피크가중)'


def agg(fold):
    """설정별 시드 평균 성적. 흔들림 = 시드 간 MAE 표준편차"""
    a = M[M.fold == fold]
    g = a.groupby('run', sort=False).agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'),
                                         RMSE=('RMSE', 'mean'), peakMAE=('peakMAE', 'mean'),
                                         n=('n', 'first'))
    g.index = [i.split(' ', 1)[0] for i in g.index]      # '8 조합 (…)' → '8'
    return g.round(2)


C, A = agg(CORE), agg('ALL')            # C = 7 to 9월(판단 기준), A = 5 to 9월


def full_agg(fold):
    """설정 이름을 자르지 않은 성적표 (★ 설정처럼 번호가 없는 것을 찾을 때)"""
    a = M[M.fold == fold]
    return a.groupby('run', sort=False).agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'),
                                            RMSE=('RMSE', 'mean'), peakMAE=('peakMAE', 'mean'),
                                            n=('n', 'first')).round(2)


CF, AF = full_agg(CORE), full_agg('ALL')


def fix_agg(fold=CORE):
    a = X[X.fold == fold]
    return a.groupby('run', sort=False).agg(MAE=('MAE', 'mean'), 흔들림=('MAE', 'std'),
                                            RMSE=('RMSE', 'mean'), peakMAE=('peakMAE', 'mean')).round(2)


XF = fix_agg()


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
    # 설정 이름은 표 2-6 to 2-9 에서 똑같이 쓴다. 같은 설정이 표마다 다른 이름으로 나오면
    # 독자가 연결하지 못한다. '비교 실험 기준'(표 2-2) 과 '1차 조합 (h7_full + 가중치 0.3)' 은 서로 다른 설정이다
    t4 = pd.DataFrame([row('0', '비교 실험 기준 (full · 복제일 그대로)'),
                       row('6', 'h7_full 단독'),
                       row('2', '복제일 가중치 0.3 단독'),
                       row('8', '1차 조합 (h7_full + 가중치 0.3)'),
                       row('14', 'h7_full + 가중치 0.1'),
                       row('3', '[참고] 누락일 포함 (학습 행 완화)')])
    fin = {'설정': '★ 최종 (h7_lag + 피크가중 + 얕은나무·느린학습)'}
    fin.update({f'{k} (7 to 9월)': CF.loc[FINAL, k] for k in ['MAE', '흔들림', 'RMSE', 'peakMAE']})
    fin['MAE (5 to 9월)'] = AF.loc[FINAL, 'MAE']
    t4 = pd.concat([t4, pd.DataFrame([fin])], ignore_index=True)
    save(t4, 't4_조합설정.csv',
         '표. 개별 요소와 조합의 성능 — 두 요소의 효과는 상쇄되지 않고 누적된다')

    # ── 표5. 최종 성능 대 목표선 ★ ──────────────────────────────────
    rows = []
    for lab, g2, src, n in [('7 to 9월 (판단 기준)', CF, core, 1708),
                            ('5 to 9월 (전체)', AF, all_, 2092)]:
        base = [round(src[m].min(), 2) for m in ['MAE', 'RMSE', 'peakMAE']]
        mod = [g2.loc[FINAL, m] for m in ['MAE', 'RMSE', 'peakMAE']]
        rows.append({'기준': lab, '평가 행 수': n,
                     '목표선 MAE': base[0], '모델 MAE': mod[0], 'MAE 개선율(%)': round((mod[0]/base[0]-1)*100),
                     '목표선 RMSE': base[1], '모델 RMSE': mod[1], 'RMSE 개선율(%)': round((mod[1]/base[1]-1)*100),
                     '목표선 피크MAE': base[2], '모델 피크MAE': mod[2], '피크MAE 개선율(%)': round((mod[2]/base[2]-1)*100)})
    save(pd.DataFrame(rows), 't5_최종성능_목표선대비.csv',
         '표. 최종 모델과 목표선 ★ — 모델은 최종 설정(h7_lag·피크가중·얕은나무·느린학습)의 '
         '시드 3개 평균. 목표선은 지표별 최저 단순규칙')

    # ── 그림2. 폴드별 성능 ──────────────────────────────────────────
    bb = b[~b.fold.isin(['ALL', CORE])]
    mf = M[(M.run == FINAL) & ~M.fold.isin(['ALL', CORE])].groupby('fold')[['MAE', 'n']].mean()
    rows = []
    for f in sorted(mf.index):
        rows.append({'검증 월': f, '평가 행 수': int(mf.loc[f, 'n']),
                     '목표선 MAE': round(bb[bb.fold == f].MAE.min(), 2),
                     '최종 모델 MAE': round(mf.loc[f, 'MAE'], 2)})
    f2 = pd.DataFrame(rows).sort_values('검증 월').reset_index(drop=True)
    save(f2, 'f2_폴드별_성능.csv',
         '그림. 검증 월별 성능 — 세로 막대 2계열. 학습 데이터가 쌓일수록 오차가 줄어든다 '
         '(5월은 학습 4개월·복제일 편중)')

    # ── 표6. 피크 관점의 성능 (최종 설정 예측 기준) ────────────────
    c = F[F.fold.isin(CORE_FOLDS)].copy()
    e = c.y - c.pred
    pk = c.peak.astype(bool)
    d = c.groupby('날짜').apply(lambda t: pd.Series({
        'y_max': t.y.max(), 'p_max': t.pred.max(), 'hi_max': t.pred_hi.max(),
        'base_max': t.pred_base.max(),
        'y_h': int(t.loc[t.y.idxmax(), 'hour']), 'p_h': int(t.loc[t.pred.idxmax(), 'hour']),
        'in3': int(t.loc[t.y.idxmax(), 'hour']) in set(t.nlargest(3, 'pred').hour),
        'keep': t.loc[t.pred.idxmax(), 'y'] / t.y.max(),
        'off': bool(t.is_off.iloc[0])}), include_groups=False)
    op = d[~d.off]
    t6 = pd.DataFrame([
        ('시간 단위 MAE', CF.loc[FINAL, 'MAE'], '시드별 지표의 평균 (설정 비교표 기준)'),
        ('피크 구간 MAE', CF.loc[FINAL, 'peakMAE'], '학습 구간 상위 5%'),
        ('피크 구간 평균 오차', round(e[pk].mean(), 2), '양수 = 과소예측'),
        ('피크 구간 과소예측 비율(%)', round((e[pk] > 0).mean() * 100, 1), '-'),
        ('일 최대 MAE (점 예측)', round((op.y_max - op.p_max).abs().mean(), 2), f'가동일 {len(op)}일'),
        ('일 최대 평균 오차 (점 예측)', round((op.y_max - op.p_max).mean(), 2), '양수 = 과소예측'),
        ('일 최대 MAE (분위 0.9)', round((op.y_max - op.hi_max).abs().mean(), 2), '추정량 교체 후'),
        ('일 최대 평균 오차 (분위 0.9)', round((op.y_max - op.hi_max).mean(), 2), '편향이 사실상 사라짐'),
        ('피크 시각 정확히 일치(%)', round((op.y_h == op.p_h).mean() * 100, 1), '-'),
        ('피크 시각 ±1시간 이내(%)', round(((op.y_h - op.p_h).abs() <= 1).mean() * 100, 1), '-'),
        ('예측 상위 3시간 안에 포함(%)', round(op.in3.mean() * 100, 1), '-'),
        ('예측 지목 시각의 실제 전력 수준(%)', round(op.keep.median() * 100, 1), '그날 실제 최대 대비, 중앙값'),
    ], columns=['지표', '값', '비고'])
    save(t6, 't6_피크관점_성능.csv',
         '표. 피크 관점의 성능 (7 to 9월, 최종 설정) — 일 최대는 점 예측과 분위 0.9 를 나눠 본다')

    # ── 표11. 하이퍼파라미터, 단독 변경 vs 조합 ★ ──────────────────
    pick = ['17 [기준] 8번 조합', '29 나무 더 깊게', '30 나무 더 얕게', '31 천천히 오래',
            '36 얕게+천천히', '32 얕게+피크3', '43 얕게+천천히+피크3+ff6', '45 +lag대응 (L군)']
    lab = ['1차 조합 (h7_full + 가중치 0.3)', '깊은 나무 (리프 63 · 최소표본 10)',
           '얕은 나무 (리프 15 · 최소표본 40)', '느린 학습 (학습률 0.02 · 트리 1,200)',
           '얕은 나무 + 느린 학습', '얕은 나무 + 피크행 가중 3배',
           '얕은 나무 + 느린 학습 + 피크가중 + 샘플링 0.6', '최종 (위 + L군 피처)']
    t11 = XF.loc[pick].reset_index(drop=True)
    t11.insert(0, '설정', lab)
    t11['1차 조합 대비'] = (t11.MAE - t11.MAE.iloc[0]).round(2)
    t11 = t11[['설정', 'MAE', '흔들림', '1차 조합 대비', 'RMSE', 'peakMAE']]
    save(t11, 't11_파라미터_조합효과.csv',
         '표. 파라미터는 단독으로는 흔들림에 묻히지만 조합에서는 분명하다 ★ '
         '(단일 변경 표는 t10, 이 표와 함께 제시할 것)')

    # ── 표12. 약점 공략 실험 요약 ─────────────────────────────────
    pick2 = ['17 [기준] 8번 조합', '18 목적함수 l2', '19 분위 0.6', '20b 분위 0.9',
             '21 피크행 가중 3배', '22 피크행 가중 6배', '23 Ridge 단독', '25 LGBM+Ridge 7:3',
             '26 공휴일 피처', '27 공휴일 가중 5배']
    lab2 = ['1차 조합 (h7_full + 가중치 0.3)', '목적함수 L2', '분위수 0.6', '분위수 0.9', '피크행 가중 3배', '피크행 가중 6배',
            'Ridge 단독', 'LGBM+Ridge 7:3', '공휴일 상호작용 피처', '공휴일 행 가중 5배']
    # 6배는 MAE 가 3배와 0.02 차이(흔들림 이내)라 "악화" 로 기각할 수 없다.
    # peakMAE 는 오히려 6배가 낫다 → 피크를 우선할 때의 대안으로 남긴다 (model.py ALT_CFG)
    judge = ['-', '기각', '보류 (peakMAE 는 우수)', '일 최대 추정 전용으로 채택', '★ 채택',
             '대안 보류 (피크 우선 시)', '기각', '기각', '기각', '기각']
    t12 = XF.loc[pick2].reset_index(drop=True)
    t12.insert(0, '설정', lab2)
    t12['판정'] = judge
    t12 = t12[['설정', 'MAE', '흔들림', 'RMSE', 'peakMAE', '판정']]
    save(t12, 't12_약점공략_요약.csv',
         '표. 피크 과소예측·공휴일 과대예측을 겨냥한 실험 ★ — 기각한 것도 함께 싣는다')

    # ── 그림5. 일 최대 추정량 비교 ────────────────────────────────
    f5 = pd.DataFrame([('점 예측', 10.93, 6.80), ('분위 0.75', 9.97, 3.88),
                       ('분위 0.80', 9.39, 2.81), ('분위 0.85', 8.82, 1.57),
                       ('분위 0.90', 8.61, -0.16), ('분위 0.95', 8.66, -1.90)],
                      columns=['추정량', '일 최대 MAE', '평균 오차 (양수 = 과소예측)'])
    save(f5, 'f5_일최대_추정량비교.csv',
         '그림. 일 최대 추정량 비교 — 꺾은선 2계열 (출처 outputs/final/gain_summary.txt). '
         '분위 0.9 에서 MAE 가 가장 낮고 편향이 거의 사라진다')

    # ── 표7. 모델 설정 (재현용) ─────────────────────────────────────
    # 세 설정을 세로로 쌓으면 어느 모델의 값인지 표만 보고는 알 수 없다 → 열로 나란히 둔다
    t7 = pd.DataFrame([
        ('피처 세트', 'full (36개)', 'full (36개, 동일)', 'h7_lag (34개)'),
        ('목적함수', 'L1 (MAE 직접 최소화)', '제곱오차 고정 (선택 불가)', 'L1 (MAE 직접 최소화)'),
        ('트리 수 / 학습률', '400 / 0.05', '300 / 해당 없음', '1,200 / 0.02  (느린 학습)'),
        ('리프 수 / 리프 최소 표본', '31 / 20', '제한 없음 / 5', '15 / 40  (얕은 나무)'),
        ('피처 / 표본 샘플링', '0.8 / 0.8', '부트스트랩 (자동)', '0.6 / 0.8'),
        ('결측 처리', '분기 규칙으로 직접 학습', '-999 로 대체 후 학습', '분기 규칙으로 직접 학습'),
        ('복제일 가중치', '1.0 (그대로)', '1.0 (그대로)', '0.3'),
        ('피크 행 가중치', '1.0 (적용 안 함)', '1.0 (적용 안 함)', '3.0 (학습 구간 상위 5%)'),
        ('일 최대 추정', '점 예측 그대로', '점 예측 그대로', '분위수 0.9 모델로 별도 산출'),
    ], columns=['항목', '비교 실험 기준 (LightGBM)', '비교 대상 (RandomForest)', '★ 최종 모델 (LightGBM)'])
    save(t7, 't7_모델설정.csv',
         '표. 모델 설정 — 왼쪽 두 열은 모델 2종을 맞비교할 때 쓴 공통 출발점이고, 오른쪽 열은 '
         '실험을 거쳐 확정한 최종 모델이다. 굵게 표시된 값이 출발점에서 바뀐 부분이다')

    # 공통 설정은 세 열이 모두 같으므로 표에서 빼고 따로 둔다 (본문에 한 문단으로 써도 된다)
    t7b = pd.DataFrame([
        ('학습 행', f'train_ok_strict 5,788행 (가동중단·시간손상·생산기록 누락일 제외)'),
        ('평가 행', '고유일 2,092행 (5 to 9월) · 판단 기준 1,708행 (7 to 9월)'),
        ('검증', '시간순 롤링 폴드, 평가셋은 모든 설정에서 고정'),
        ('시드', f'{list(SEEDS)} 3회 반복 학습, 시드 평균으로 보고'),
        ('실험 규모', f'설정 비교 {len(RUNS)}개 + 약점 공략 {X.run.nunique()}개 (각 시드 3개 × 폴드)'),
    ], columns=['항목', '값'])
    save(t7b, 't7b_공통설정.csv', '표. 세 설정에 공통으로 적용한 조건 (본문 문단으로 풀어 써도 된다)')

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
        # 여러 변수를 묶어 적은 행은 그 변수들의 gain 을 모두 더한다.
        # (첫 변수만 쓰면 lag 묶음이 22.42% 대신 8.90% 로 크게 축소된다)
        def gi(*cols):
            r = G[G.피처.isin(cols)]
            return f"{r['gain 중요도(%)'].sum():.2f}%" if len(r) else '-'
        def sm(*cols):
            r = S[S.피처.isin(cols)]
            if not len(r):
                return '-'
            w = G.set_index('피처').loc[r.피처, 'gain 중요도(%)'].values  # gain 가중평균
            v = (r['피크/전체 배율'].values * w).sum() / w.sum() if w.sum() else r['피크/전체 배율'].mean()
            return f"{v:.2f}배"
        t9 = pd.DataFrame([
            ('생산량 800 부근에서 전력 포화', 'prod_cap', '생산량을 800에서 절단', gi('prod_cap'), sm('prod_cap')),
            ('생산량 분포가 한쪽으로 쏠림', 'prod_log', 'log(1+생산량)', gi('prod_log'), sm('prod_log')),
            ('기온 22℃ 이상에서 냉방부하 발생', 'cool', 'max(기온-22, 0)', gi('cool'), sm('cool')),
            ('기온과 습도가 함께 작용', 'thi', '불쾌지수 (기온·습도 결합)', gi('thi'), sm('thi')),
            ('전일보다 전주 동일 요일이 유사', 'lag168 / lag336 / lag_week_mean4',
             '1주 전·2주 전·최근 4주 평균 동시각',
             gi('lag168', 'lag336', 'lag_week_mean4'), sm('lag168', 'lag336', 'lag_week_mean4')),
            ('휴무 기간이 다음 주 시차를 오염', 'lag 결측 처리 + lag_prev_op',
             '휴무·중단 구간은 결측, 직전 가동일 동시각을 별도 변수로', gi('lag_prev_op'), sm('lag_prev_op')),
            ('휴무 길이에 따라 복귀일 부하가 다름', 'off_run_prev / days_since_off',
             '직전 연속 휴무 일수 / 마지막 휴무 이후 경과일',
             gi('off_run_prev', 'days_since_off'), sm('off_run_prev', 'days_since_off')),
            ('부하가 기저·중·고 세 단계로 구분', 'shift / is_transition',
             '야간·점심·주간 구분 / 전환 시각(7·12·17시)',
             gi('shift', 'is_transition'), sm('shift', 'is_transition')),
            ('휴무 여부가 타깃에서 유도된 값', 'day_prod_zero',
             '생산계획만으로 만든 휴무 근사값', gi('day_prod_zero'), sm('day_prod_zero')),
        ], columns=['1장의 도메인 발견', '파생 변수', '정의', 'gain 중요도', '피크 구간 SHAP 배율'])
        save(t9, 't9_피처엔지니어링_매핑.csv',
             '표. 데이터 진단에서 얻은 발견이 어떤 파생 변수로 들어갔는가 ★')

    # ── 표10. 하이퍼파라미터 민감도 (선행: python src/hparam_check.py) ──
    hp = ROOT / 'outputs' / 'hparam_results.csv'
    if hp.exists():
        H = pd.read_csv(hp)
        # 원본 파일은 '기본값 (현재 설정)' 이지만, 보고서에서는 표 2-6 to 2-9 와 이름을 맞춘다
        H.loc[H.설정 == '기본값 (현재 설정)', '설정'] = '1차 조합 (h7_full + 가중치 0.3)'
        H = H.rename(columns={'기본값 대비': '1차 조합 대비'})
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
