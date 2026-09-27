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
│   ├── error_analysis.py # 최종 후보 OOF 예측 및 조건별 오차 분석
│   ├── operational_correction.py # 생산종료 이후 운영규칙 보정 실험(선택)
│   └── forecast.py     # 특정 날짜 24시간을 1주 앞 시점에서 예측
├── outputs/
│   ├── eda/            # 그림 01 to 07, captions.txt, clean_preview.csv
│   ├── daily_pattern.csv
│   ├── features.csv
│   ├── peak_ratio.csv  # 시각별 15분최대/시간평균 환산 계수
│   ├── baseline_results.csv
│   ├── model_results.csv
│   ├── error_predictions.csv
│   ├── error_by_condition.csv
│   ├── error_by_date.csv
│   ├── error_top_cases.csv
│   ├── error_summary.txt
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

전처리 → EDA 시각화 → 통계 → 패턴 분석 → 피처 생성 → 베이스라인 → 모델 비교 실험 → 예측오차 분석 → 1주 앞 예측 순으로 실행되며, 결과물은 `outputs/`에 저장됩니다. 모델 비교와 오차 분석은 여러 시드·폴드를 반복 학습하므로 다른 단계보다 오래 걸립니다.

단계별로 실행하려면 `python src/<스크립트>.py` 를 사용합니다. 어느 폴더에서 실행해도 경로는 저장소 기준으로 잡힙니다.
특정 날짜만 예측하려면 `python src/forecast.py 20210914` 처럼 날짜를 넘깁니다.

오차 분석 후 생산계획상 마지막 생산시간 이후의 과대예측을 제한하는 운영규칙 실험은 `python src/operational_correction.py` 로 별도 실행합니다. 이 보정은 전체 점수는 개선되지만 일부 폴드가 악화되어 현재는 최종 모델이 아닌 실험 후보입니다.

모델 검증은 시간순 롤링 폴드로 하고, 성능 판단은 고유일 기준으로 합니다. 자세한 실험 결과는 `작업내역(조선제).txt` [13] [14] 를 참고하세요.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib
- jupyter
