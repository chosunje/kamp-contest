# KAMP 경진대회 프로젝트

## 폴더 구조

```
kamp-contest/
├── data/
│   ├── 1/ ~ 5/         # 과제 데이터셋 1~5
│   └── raw/            # 원본 데이터 (수정하지 않음)
├── notebooks/          # EDA 및 실험용 Jupyter 노트북
├── src/                # 재사용 가능한 소스 코드 (전처리, 모델, 평가 등)
├── outputs/            # 모델, 예측 결과, 그림, SHAP 결과
├── run_all.py          # 전체 파이프라인 실행 스크립트
├── environment.yml     # conda 환경 정의
└── README.md
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate kamp-contest
```

## 실행 방법

1. 원본 데이터를 `data/raw/` 에 넣습니다.
2. 전체 파이프라인을 실행합니다.

```bash
python run_all.py
```

결과물은 `outputs/` 에 저장됩니다.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib, seaborn
- jupyter
