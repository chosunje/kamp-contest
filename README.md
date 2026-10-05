# KAMP 경진대회 프로젝트

과제 ⑤ 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석

## 폴더 구조

```
kamp-contest/
├── dataset/
│   └── okm_augumented_2021.csv   # 원본 데이터 (수정하지 않음)
├── src/
│   ├── preprocess.py   # 원본 로드·정제·플래그(복제일/휴무일/가동중단). 모든 단계의 공통 입구
│   ├── eda.py          # EDA 그림 (복제일/고유일 분리)
│   ├── eda3.py         # 기저부하·피크 조건 통계
│   ├── pattern.py      # 날짜별 전력곡선 패턴 ID
│   ├── features.py     # 모델 입력 피처 생성 (피처목록.txt 기준)
│   ├── baseline.py     # 베이스라인 4종 (모델이 넘어야 할 목표선)
│   ├── model.py        # 모델 비교 실험 (LightGBM / RandomForest, 설정 x 시드 x 폴드)
│   ├── error_analysis.py # 조건별 오차 분석 (언제 틀리는가)
│   ├── shap_analysis.py  # 영향요인 SHAP (무엇으로 맞히는가, 전체 구간)
│   ├── importance.py     # 변수 중요도 + 전체 구간 대비 피크 구간 SHAP 비교
│   ├── peak_driver.py    # 최대피크 위험조건 (기준을 바꿔 가며 전수 측정)
│   ├── peak_alarm.py     # 피크 위험 경보의 재현율·정밀도 검증
│   ├── predict_test.py   # 테스트 구간 예측 파일 생성 (제출물)
│   ├── report_final.py   # 최종 설정 기준 성능 한 장 요약
│   ├── report_gain.py    # 라운드 간 변화 비교 (채택·기각 기록)
│   ├── quantile_analysis.py # L1/P90 비교 및 D-1 입력·예측 불변성 검증(실험)
│   ├── reinforce.py      # 6차 강화 실험 (수단 8가지 측정·기각, 15분 최대 채택)
│   ├── hparam_check.py   # 하이퍼파라미터 단일 변경 민감도 (조합 효과의 대조군)
│   ├── viz_perf.py     # 성능 체감용 그림 (예측 vs 실제)
│   ├── report_ch1.py   # 보고서 1장 표·그림용 CSV
│   ├── report_ch2.py   # 보고서 2장 표·그림용 CSV
│   ├── operational_correction.py # 생산종료 이후 운영규칙 보정 실험(선택)
│   └── forecast.py     # 특정 날짜 24시간을 1주 앞 시점에서 예측
├── outputs/
│   ├── eda/            # 그림 01 to 07, captions.txt, clean_preview.csv
│   ├── perf/           # 성능 체감 그림 01 to 06, captions.txt
│   ├── final/          # 최종 설정 요약 (00·01_dashboard.png, summary.txt)
│   ├── report/ch1/     # 보고서 1장 표·그림 데이터 CSV, 캡션.txt
│   ├── report/ch2/     # 보고서 2장 표·그림 데이터 CSV, 캡션.txt
│   ├── daily_pattern.csv
│   ├── features.csv
│   ├── peak_ratio.csv  # 시각별 15분최대/시간평균 환산 계수
│   ├── baseline_results.csv
│   ├── model_results.csv       # 설정 비교 24개
│   ├── model_results_fix.csv   # 약점 공략 실험 31개 (python src/model.py --fix)
│   ├── importance_gain.csv / importance_shap.csv
│   ├── shap_importance.csv / shap_by_group.csv
│   ├── peak_driver_*.csv / peak_alarm_*.csv / reinforce_*.csv
│   ├── hparam_results.csv
│   ├── submission_{구간}.csv   # 제출용 테스트 예측
│   ├── error_predictions.csv
│   ├── error_by_condition.csv
│   ├── error_by_date.csv
│   ├── error_top_cases.csv
│   ├── error_summary.txt
│   ├── quantile_results.csv
│   ├── quantile_predictions.csv
│   ├── quantile_leakage_audit.csv
│   ├── quantile_summary.txt
│   └── forecast_{날짜}.csv
├── run_all.py          # 전체 파이프라인 실행
├── environment.yml     # conda 환경 정의
└── README.md
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate kamp-contest
```

## 실행 방법

```bash
python run_all.py
```

전처리 → EDA 시각화 → 통계 → 패턴 분석 → 피처 생성 → 베이스라인 → 모델 비교 실험 → 예측오차 분석 → 영향요인(SHAP) → 피크 위험조건 → 피크 경보 검증 → 성능 체감 그림 → 1주 앞 예측 → 테스트 예측 파일 → 최종 성능 요약 → 보고서 1·2장 데이터 순으로 17단계가 실행되며, 결과물은 `outputs/`에 저장됩니다. 전체 약 9분이 걸리고, 모델 비교와 오차 분석이 대부분을 차지합니다.

파이프라인에 넣지 않은 보조 스크립트는 필요할 때 개별 실행합니다. `importance.py`(피크 구간 SHAP 비교), `hparam_check.py`(파라미터 단일 변경 민감도), `reinforce.py`(강화 실험 전수), `quantile_analysis.py`(L1/P90 비교), `operational_correction.py`(운영규칙 보정), `model.py --fix`(약점 공략 31개)입니다.

단계별로 실행하려면 `python src/<스크립트>.py` 를 사용합니다. 어느 폴더에서 실행해도 경로는 저장소 기준으로 잡힙니다.
특정 날짜만 예측하려면 `python src/forecast.py 20210914` 처럼 날짜를 넘깁니다.

오차 분석 후 생산계획상 마지막 생산시간 이후의 과대예측을 제한하는 운영규칙 실험은 `python src/operational_correction.py` 로 별도 실행합니다. 이 보정은 전체 점수는 개선되지만 일부 폴드가 악화되어 현재는 최종 모델이 아닌 실험 후보입니다.

L1 점 예측과 분위수 0.9를 비교하려면 `python src/quantile_analysis.py`를 별도로 실행합니다. 기존 `h7_full` 재현용 결과(`reference`)와 당일 실제 전력에서 계산한 `is_off`를 제외한 D-1 결과(`d1`)를 나누어 저장합니다. D-1은 폴드 첫 평가일에 학습한 모델을 고정하고 매일 0시 이전 실적으로 입력을 갱신하는 방식이며, 평가일 이후 전력을 삭제·변경해도 입력과 예측이 동일한지 검사합니다. 생산 실적과 기상 실측을 계획·예보로 가정한 실험이고, P90은 초과확률 90%를 뜻하지 않습니다. 기존 최종 후보와 `forecast.py`의 D-7 동작은 이 실험으로 교체되지 않습니다.

실험 검증은 `python -m unittest discover -s tests -v`로 실행합니다.

모델 검증은 시간순 롤링 폴드(5 to 9월)로 하고, 성능 판단은 **고유일 · 7 to 9월(CORE) 기준**을 우선합니다. 5월은 학습이 4개월뿐이고 그 대부분이 복제일이라 오차가 크므로, 두 기준을 함께 출력합니다.

최종 모델은 LightGBM(`h7_lag` 34개 피처 · 복제일 가중치 0.3 · 피크행 가중치 3배 · 얕은 나무 + 느린 학습)이며, CORE 기준 MAE 6.57 / RMSE 9.54 / 피크구간 MAE 12.48입니다. 일 최대는 분위수 0.9로 따로 추정합니다(`model.py`의 `FINAL_CFG`, `ALARM_CFG`).

결정 사항의 상세 근거(대안·수치·메커니즘)는 `결정근거(보고서용).txt`에 정리되어 있습니다. 보고서를 쓸 때 이 문서를 먼저 봅니다.

문서는 `작업내역(최치훈).txt`가 팀 공용 기준(결정사항 D01 to D24)이고, 실험 원자료와 해석은 `작업내역(조선제).txt`, 점검 기록은 `작업내역(남궁인성).txt`에 있습니다.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib
- jupyter
