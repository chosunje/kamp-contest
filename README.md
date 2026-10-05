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
│   ├── clone_weight_analysis.py # 강화 모델의 복제일 가중치 0.1/0.3 재비교(실험)
│   ├── shap_analysis.py # SHAP 영향요인 분석
│   ├── peak_driver.py  # 최대피크 위험조건 분석
│   ├── peak_alarm.py   # 기존 시간 평균 경보 비교 보존
│   ├── alarm_policy.py # D10 15분 최대 P90·학습 Q95 프로젝트 경보 정책
│   ├── peak_alarm_max15.py # D-1 경보 검증·임계 민감도·미래 전력 변경 감사
│   ├── reinforce.py      # 6차 강화 실험 (수단 8가지 측정·기각, 15분 최대 채택)
│   ├── hparam_check.py   # 하이퍼파라미터 단일 변경 민감도 (조합 효과의 대조군)
│   ├── viz_perf.py     # 성능 체감용 그림 (예측 vs 실제)
│   ├── report_ch1.py   # 보고서 1장 표·그림용 CSV
│   ├── report_ch2.py   # 보고서 2장 표·그림용 CSV
│   ├── operational_correction.py # 생산종료 이후 운영규칙 보정 실험(선택)
│   ├── forecast.py     # 특정 날짜 24시간을 1주 앞 시점에서 예측
│   ├── predict_test.py # 기간별 점 예측·15분 최대·D10 경보 파일 생성
│   └── report_final.py # 강화 최종 모델 성능 요약
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
│   ├── clone_weight_results.csv
│   ├── clone_weight_predictions.csv
│   ├── clone_weight_config.json
│   ├── clone_weight_summary.txt
│   ├── alarm_policy.json # 확정한 프로젝트 정책·학습 설정·폴드 임계·검증 결과
│   ├── peak_alarm_max15_predictions.csv
│   ├── peak_alarm_max15_model_results.csv
│   ├── peak_alarm_max15_eval.csv
│   ├── peak_alarm_max15_days.csv
│   ├── peak_alarm_max15_thresholds.csv
│   ├── peak_alarm_max15_leakage_audit.csv
│   ├── peak_alarm_max15_summary.txt
│   └── forecast_{날짜}.csv
├── run_all.py          # 전체 파이프라인 실행
├── environment.yml     # conda 환경 정의
└── README.md
```

## 최종 모델·저감 스케줄링·HTML 화면

환경 설정 후 아래 명령 하나로 예측과 일정 추천 화면을 실행합니다.

```bash
python run_dashboard.py --open
```

브라우저 주소는 `http://127.0.0.1:8765`입니다. 날짜별 전력 예측·D10 경보를 보고, 작업의 시작 시간·소요 시간·허용 시간대·설비·선행 작업·부하를 입력한 뒤 **일정 추천**을 실행합니다. 기존/추천 시간표와 변경 이유를 비교하고 CSV를 다운로드할 수 있습니다. 기본 작업은 실제 설비 기록이 아닌 **예시 작업·가정 부하**입니다.

처음 실행하면 최종 모델 3종을 시드 0/1/2로 학습해 저장하고, 이후에는 예측 캐시를 사용합니다. 시간 평균·15분 최대 점 예측은 기존 최종 설정을 유지하면서 당일 실측 `is_off`를 제거한 33개 입력으로 재학습합니다. 세 모델 모두 학습 마감 이후 전력을 입력에서 제외합니다. 이 운영 경로에 기존 34피처 사후 평가 MAE를 붙이지 않습니다.

```bash
python run_dashboard.py --build-only            # 예측·예시 일정·오프라인 HTML 생성
python run_dashboard.py --rebuild               # 모델과 예측 재생성
python run_dashboard.py --start 20210908 --end 20210914
python run_dashboard.py --threshold 190         # 사용자가 확인한 원본 단위의 절대 임계값
```

`outputs/final_app/index.html`은 서버 없이 예측과 생성된 예시 추천을 볼 수 있습니다. 작업을 바꾸고 추천을 재계산할 때는 로컬 서버를 실행합니다. 실제 테스트 기간 예측에는 해당 기간의 계획·기상 입력 행이 필요합니다. 화면에서 기존 예측 CSV를 불러와 일정 추천에 사용할 수도 있습니다.

스케줄러는 작업 시간·고정 작업·허용 시간대·설비 중복·선행 순서·최대 동시 작업 수를 지킵니다. 모든 작업의 부하를 입력한 경우 기존 작업 부하를 총 예측에서 빼고 추천 시간에 다시 더해 비교합니다. 부하가 없거나 총 예측과 맞지 않으면 경보 시간의 작업 노출만 비교하며 저감량을 표시하지 않습니다. **저감 수치는 입력한 부하에 따른 시뮬레이션이며 실측 효과나 요금 절감을 검증한 결과가 아닙니다.**

구성·입출력·시연 절차·해석 범위는 [최종 운영 시연 설명](docs/final_app.md)에 정리했습니다. 구현은 `src/final_forecast.py`, `src/scheduling.py`, `web/index.html`, `run_dashboard.py`입니다.

## 환경 설정

```bash
conda env create -f environment.yml
conda activate kamp-contest
```

## 실행 방법

```bash
python run_all.py
```

전처리 → EDA 시각화 → 통계 → 패턴 분석 → 피처 생성 → 베이스라인 → 모델 비교 → 예측오차 분석 → SHAP 분석 → 피크 위험조건 분석 → 15분 최대 경보 검증 → 성능 그림 → 1주 앞 예측 → 기간별 예측 파일 → 최종 성능 요약 → 최종 모델·저감 일정·HTML·최대피크 요약 → 보고서 1·2장 데이터의 22단계를 실행합니다. 마지막 단계는 로컬 서버를 시작하지 않고 시연 산출물만 생성합니다. 결과물은 `outputs/`에 저장됩니다. 모델 비교와 오차 분석은 여러 시드·폴드를 반복 학습하므로 다른 단계보다 오래 걸립니다. 아래 별도 비교 실험 3종은 `run_all.py`에 포함되지 않습니다.

파이프라인에 넣지 않은 보조 스크립트는 필요할 때 개별 실행합니다. `importance.py`(피크 구간 SHAP 비교), `hparam_check.py`(파라미터 단일 변경 민감도), `reinforce.py`(강화 실험 전수), `quantile_analysis.py`(L1/P90 비교), `operational_correction.py`(운영규칙 보정), `model.py --fix`(약점 공략 31개)입니다.

단계별로 실행하려면 `python src/<스크립트>.py` 를 사용합니다. 어느 폴더에서 실행해도 경로는 저장소 기준으로 잡힙니다.
특정 날짜만 예측하려면 `python src/forecast.py 20210914` 처럼 날짜를 넘깁니다.

오차 분석 후 생산계획상 마지막 생산시간 이후의 과대예측을 제한하는 운영규칙 실험은 `python src/operational_correction.py` 로 별도 실행합니다. 초기 `h7_full` 결과에서 전체 점수는 개선됐지만 CORE MAE는 6.93→6.98, RMSE는 10.58→11.23으로 악화되어 일괄 적용은 채택하지 않았습니다. 코드와 결과는 비교 근거로 보존합니다.

L1 점 예측과 분위수 0.9를 비교하려면 `python src/quantile_analysis.py`를 별도로 실행합니다. 기존 `h7_full` 재현용 결과(`reference`)와 당일 실제 전력에서 계산한 `is_off`를 제외한 D-1 결과(`d1`)를 나누어 저장합니다. D-1은 폴드 첫 평가일에 학습한 모델을 고정하고 매일 0시 이전 실적으로 입력을 갱신하는 방식이며, 평가일 이후 전력을 삭제·변경해도 입력과 예측이 동일한지 검사합니다. 생산 실적과 기상 실측을 계획·예보로 가정한 실험이고, P90은 초과확률 90%를 뜻하지 않습니다. 기존 최종 후보와 `forecast.py`의 D-7 동작은 이 실험으로 교체되지 않습니다.

강화 모델의 복제일 가중치를 비교하려면 `python src/clone_weight_analysis.py`를 실행합니다. `FINAL_CFG`의 피처·나무·피크 가중치를 고정하고 복제일 가중치만 0.1/0.3으로 바꾸며, 시간 평균과 15분 최대를 별도 타깃으로 학습합니다. 각 타깃의 학습 상위 5%를 분석용 피크 기준으로 사용합니다. 현재 조건에서는 0.3의 전체 MAE·RMSE가 낮고 0.1의 피크 MAE가 낮아 점 예측의 0.3을 유지했습니다. 당일 실측 `is_off`가 포함된 사후 비교이며 앞선 D-1 검증 결과와 구분합니다. 실행 설정과 시드별·앙상블 결과를 별도로 저장합니다.

실험 검증은 `python -m unittest discover -s tests -v`로 실행합니다.

모델 검증은 시간순 롤링 폴드(5 to 9월)로 하고, 성능 판단은 **고유일 · 7 to 9월(CORE) 기준**을 우선합니다. 5월은 학습이 4개월뿐이고 그 대부분이 복제일이라 오차가 크므로, 두 기준을 함께 출력합니다.

최종 모델은 LightGBM(`h7_lag` 34개 피처 · 복제일 가중치 0.3 · 피크행 가중치 3배 · 얕은 나무 + 느린 학습)이며, CORE 기준 MAE 6.57 / RMSE 9.54 / 피크구간 MAE 12.48입니다. 일 최대는 분위수 0.9로 따로 추정합니다(`model.py`의 `FINAL_CFG`, `ALARM_CFG`).

작업을 이어받을 때는 `인수인계(2026-09-30).txt`의 상단 10/05 후속 결정을 먼저 봅니다. 본문의 수치·미결 항목은 9/30 당시 기록으로 보존했습니다.

결정 사항의 상세 근거(대안·수치·메커니즘)는 `결정근거(보고서용).txt`에 정리되어 있습니다. 보고서를 쓸 때 이 문서를 먼저 봅니다.

문서는 `작업내역(최치훈).txt`가 팀 공용 기준(결정사항 D01 to D24)이고, 실험 원자료와 해석은 `작업내역(조선제).txt`, 점검 기록은 `작업내역(남궁인성).txt`에 있습니다.

## 현재 반영한 결정 (2026-10-05)

| 항목 | 적용 내용 |
| --- | --- |
| D07 복제일 가중치 | 강화 L1 점 예측의 0.3 유지. CORE·ALL 전체 MAE/RMSE는 0.3이 낮고 피크 MAE는 0.1이 낮습니다. |
| D10 경보 정책 | 15분 최대 직접 P90이 학습 15분 최대 Q95 이상이면 시간 경보, 하루에 한 번이라도 경보이면 위험일. 프로젝트 기본 정책으로 확정했습니다. |
| L1/P90 | L1 점 예측 유지. P90은 포함률 미보정 경보 후보로 보존하며 90% 상한이나 초과확률을 보장하지 않습니다. |
| 생산종료 이후 cap | 초기 h7_full의 CORE MAE/RMSE 악화로 일괄 미채택. 실험 코드·결과는 보존합니다. |
| src/models | 기존 삭제 유지. 현재 실행 경로는 src/baseline.py와 src/model.py입니다. |
| 부분가동일 | P군 피처와 추가 일괄 후처리는 기본 제외. 실제 테스트 구간의 부분가동 여부는 미확인입니다. |

현재 `forecast.py`, `predict_test.py`의 `alarm`은 **15분 최대 직접 P90과 학습 15분 최대 Q95**를 비교합니다. 시간 평균은 점 예측으로 사용합니다. Q95는 학습 마감 전 `train_ok_strict`·15분 타깃 유효행(복제 포함)의 비가중 분위값이며, 임계 숫자는 학습 기간마다 갱신됩니다. `pred_hi_max15`는 직접 학습한 P90이고, 기존 시간 평균 P90의 환산값은 `pred_hi_max15_환산`에 보존합니다. `predict_test.py --threshold <값>` 또는 `forecast.py`의 임계값 인자로 원본 전력값과 같은 단위의 절대값을 지정할 수 있습니다.

**확정한 범위는 프로젝트 기본 경보 정책입니다.** 계약전력·공식 제출 기준·전력 단위의 확인 자료가 없으므로 계약전력 초과나 요금 발생을 판정하는 규칙으로 쓰지 않습니다. 15분 컬럼은 D20 A안으로 해석하며 P90의 90% 포함률도 보장하지 않습니다.

`python src/peak_alarm_max15.py`의 D-1 검증에서 CORE 1,708행, 완전 평가일 69일(휴무 포함; 불완전일 3일 제외)의 P90 Q95 일 정밀도는 95.5%, 재현율은 100.0%, 경보율은 63.8%였습니다. 실제 위험일 42일, 경보 44일 중 오경보 2일이며 관측 시간별 P90 포함률은 83.0%였습니다. 88일의 미래 전력 삭제·변경 176건에서 입력·예측·경보가 동일했습니다. 독립 홀드아웃은 없고 생산 실적·기상 실측을 계획·예보로 가정했습니다. 이 성능을 D-7 또는 생산계획 없는 조건의 성능으로 확대하지 않습니다.

경보 입력은 h7_lag에서 당일 실측 `is_off`를 제외한 33개 피처이며, 예측 마감 이후 전력과 아직 관측하지 않은 휴무 이력을 마스킹합니다. Q97·Q98은 실제 피크 Q95를 고정한 경보 임계 민감도 비교용입니다. `peak_alarm.py`와 기존 시간 평균 경보 표는 이전 비교를 보존하며 새 기본 경보 검증과 구분합니다.

가중치 재비교와 기존 최종 모델 결과는 실제 전력 유래 `is_off` 등을 포함한 사후 비교입니다. 미래 전력 변경 검증을 통과한 `quantile_analysis.py`의 D-1 결과와 구분합니다. 실제 테스트 파일이 없어 미래 구간의 운영 성능과 가동 상태를 확정하지 않았습니다.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib
- jupyter
