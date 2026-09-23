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
│   ├── baseline.py     # 베이스라인 4종 시간순 롤링 평가 (복제일/고유일 분리)
│   └── features.py     # 모델 입력 피처 생성 (피처목록.txt 기준)
├── outputs/
│   ├── eda/            # 그림 01 to 07, captions.txt, clean_preview.csv
│   ├── daily_pattern.csv
│   ├── baseline_results.csv
│   ├── features.csv
│   └── peak_ratio.csv  # 시각별 15분최대/시간평균 환산 계수
├── run_all.py          # 전체 파이프라인 실행
├── environment.yml     # conda 환경 정의
├── 작업내역.txt        # 현황과 결정사항 D01 to D19 (가장 먼저 읽을 문서)
├── 피처목록.txt        # 피처 후보·우선순위·구현 열 이름
├── 작업내역(조선제).txt # 파일별 역할, 전처리 보강 내용, 데이터 특성과 상관관계, 모델 실험 결과
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

전처리 → EDA 시각화 → 통계 → 패턴 분석 → 베이스라인 평가 → 피처 생성 순으로 실행되며, 결과물은 `outputs/`에 저장됩니다.
단계별로 실행하려면 `python src/<스크립트>.py` 를 사용합니다. 어느 폴더에서 실행해도 경로는 저장소 기준으로 잡힙니다.

### 문제 해결

`eda.py` 단계에서 오류 메시지 없이 프로세스가 죽는다면 numpy 설치가 깨진 것입니다. 먼저 확인하세요.

```bash
python -c "import numpy as np; print(np.eye(2) @ np.eye(2))"
```

이 명령이 죽으면 해당 환경의 numpy를 다시 설치합니다 (`conda install -n kamp-contest --force-reinstall numpy`). 코드 문제가 아닙니다.

## 주요 라이브러리

- Python 3.11
- pandas, numpy
- scikit-learn, lightgbm
- shap
- matplotlib
- jupyter
