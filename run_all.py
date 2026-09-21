"""전체 파이프라인 실행 스크립트.

사용법:
    python run_all.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"


def load_data():
    """data/raw 에서 원본 데이터를 불러온다."""
    # TODO: 데이터 로딩 구현
    raise NotImplementedError


def preprocess(df):
    """전처리 및 피처 엔지니어링."""
    # TODO: 전처리 구현
    return df


def train(df):
    """모델 학습 (예: LightGBM)."""
    # TODO: 학습 구현
    raise NotImplementedError


def evaluate_and_explain(model, df):
    """평가 지표 계산 및 SHAP 해석 결과를 outputs 에 저장."""
    # TODO: 평가 및 SHAP 구현
    raise NotImplementedError


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df = preprocess(df)
    model = train(df)
    evaluate_and_explain(model, df)


if __name__ == "__main__":
    main()
