"""모든 모델 예측을 모아 비교표 작성. 기본은 개발 폴드(5 to 8월)만.
python src/models/compare.py --final  → 9월 최종 확인 결과도 출력 (모델 선택이 끝난 뒤에만 사용)"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, DEV_FOLDS, HOLDOUT, score   # noqa: E402

final = '--final' in sys.argv
P = pd.concat([pd.read_csv(f, parse_dates=['dt']) for f in sorted(OUT.glob('*/predictions.csv'))], ignore_index=True)
folds = DEV_FOLDS + ([HOLDOUT] if final else [])
P = P[P.fold.isin(folds)]
keys = ['model', 'fset', 'clone']
GROUPS = {'고유일': ~P.is_clone, '복제일': P.is_clone, '전체': P.is_clone | ~P.is_clone}

rows = []
for g, m in GROUPS.items():
    for part, s in [('개발 합산', P[m & P.fold.isin(DEV_FOLDS)])] + [(f, P[m & (P.fold == f)]) for f in folds]:
        if len(s):
            r = s.groupby(keys).apply(score, include_groups=False).reset_index()
            rows.append(r.assign(group=g, part=part))
res = pd.concat(rows, ignore_index=True)
res.to_csv(OUT / ('comparison_final.csv' if final else 'comparison.csv'), index=False, encoding='utf-8-sig')

u = res[(res.group == '고유일') & (res.part == '개발 합산')].sort_values('MAE')
print('[고유일, 개발 폴드 5 to 8월 합산] MAE 순\n', u[keys + ['MAE', 'RMSE', 'peakMAE', 'n']].round(1).to_string(index=False))
f = res[(res.group == '고유일') & res.part.isin(folds)].pivot_table(index=keys, columns='part', values='MAE')
print('\n[고유일 폴드별 MAE]\n', f.reindex(u.set_index(keys).index).round(1).to_string())
if final:
    h = res[(res.group == '고유일') & (res.part == HOLDOUT)].sort_values('MAE')
    print(f'\n[최종 확인 {HOLDOUT}, 고유일]\n', h[keys + ['MAE', 'RMSE', 'peakMAE', 'n']].round(1).to_string(index=False))
