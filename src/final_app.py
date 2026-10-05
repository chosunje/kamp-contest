"""전체 파이프라인의 마지막 단계: 최종 예측·예시 일정·오프라인 화면 생성."""
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_dashboard import export_demo, load_payload


if __name__ == '__main__':
    args = Namespace(start=None, end=None, threshold=None, no_plan=False, rebuild=True)
    payload = load_payload(args)
    result = export_demo(payload)
    print(f'최종 운영 시연 생성 완료: outputs/final_app/index.html · 일정 모드 {result["mode"]}')
    print('일정 편집·추천 재계산: python run_dashboard.py')
