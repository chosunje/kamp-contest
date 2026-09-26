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
│   └── models/
│       ├── common.py   # 공통 평가 틀: 시간순 폴드, 복제일 처리, 지표
│       ├── baseline/   # 규칙 기반 베이스라인 4종
│       ├── ridge/      # Ridge 선형회귀
│       ├── lgbm/       # LightGBM
│       ├── ensemble/   # Ridge + LightGBM 가중 평균
│       └── compare.py  # 모델 비교표 (--final 옵션: 9월 최종 확인 포함)
├── outputs/
│   ├── eda/            # 그림 01 to 07, captions.txt, clean_preview.csv
│   ├── models/         # 모델별 predictions.csv, comparison.csv
│   ├── daily_pattern.csv
│   ├── baseline_results.csv
│   └── features.csv
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

전처리 → EDA 시각화 → 통계 → 패턴 분석 → 피처 생성 → 모델 학습(베이스라인·Ridge·LightGBM·앙상블) → 모델 비교 순으로 실행되며, 결과물은 `outputs/`에 저장됩니다.

모델 검증은 개발 폴드(5 to 8월)로 하고, 9월(9/1 to 9/14)은 모델 선택이 끝난 뒤 `python src/models/compare.py --final` 로 한 번만 확인합니다.
단계별로 실행하려면 `python src/<스크립트>.py` 를 사용합니다. 어느 폴더에서 실행해도 경로는 저장소 기준으로 잡힙니다.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib
- jupyter
