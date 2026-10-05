"""최종 예측 → 제약을 지킨 일정 추천 → HTML 화면. python run_dashboard.py"""
from argparse import ArgumentParser
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from scheduling import schedule_day

OUT = ROOT / 'outputs' / 'final_app'
WEB = ROOT / 'web' / 'index.html'
MAX_BODY = 1_000_000


def date_text(value):
    from datetime import datetime
    text = str(value)
    return datetime.strptime(text, '%Y%m%d' if len(text) == 8 else '%Y-%m-%d').strftime('%Y-%m-%d')


def reject_constant(value):
    raise ValueError('JSON에 NaN 또는 Infinity를 넣을 수 없습니다.')


def prepare_payload(payload):
    """예측의 위험 주간 시간에 예시 작업을 놓는다. 실제 설비·생산계획이 아니다."""
    payload = dict(payload)
    daytime = [r for r in payload['forecast'] if 8 <= r['hour'] <= 14]
    chosen = max(daytime, key=lambda r: r['pred_hi_max15'])
    peak = chosen['hour']
    payload['demo_date'] = chosen['date']
    payload['demo_tasks'] = [
        {'task_id': 'A', 'name': '예시 작업 A', 'original_start': peak, 'duration': 2,
         'earliest_start': 8, 'latest_end': 20, 'load': 8., 'movable': True,
         'resource': '설비 A', 'depends_on': []},
        {'task_id': 'B', 'name': '예시 작업 B', 'original_start': peak, 'duration': 2,
         'earliest_start': 8, 'latest_end': 20, 'load': 10., 'movable': True,
         'resource': '설비 B', 'depends_on': []},
        {'task_id': 'C', 'name': '예시 후속 작업 C', 'original_start': peak + 2, 'duration': 2,
         'earliest_start': 8, 'latest_end': 20, 'load': 6., 'movable': True,
         'resource': '설비 A', 'depends_on': ['A']},
        {'task_id': 'D', 'name': '예시 고정 작업 D', 'original_start': peak + 4, 'duration': 1,
         'earliest_start': 8, 'latest_end': 20, 'load': 3., 'movable': False,
         'resource': '설비 B', 'depends_on': []},
    ]
    return payload


def recommend(payload, request):
    if not isinstance(request, dict):
        raise ValueError('일정 요청은 JSON 객체여야 합니다.')
    date = date_text(request.get('date', ''))
    if 'forecast' in request:
        rows = request['forecast']
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError('업로드 예측은 시간별 행의 배열이어야 합니다.')
        rows = [r for r in rows if 'date' not in r or date_text(r['date']) == date]
    else:
        rows = [r for r in payload['forecast'] if r['date'] == date]
    result = schedule_day(rows, request.get('tasks', []), max_parallel=request.get('max_parallel', 2))
    return {'date': date, **result}


def export_demo(payload):
    OUT.mkdir(parents=True, exist_ok=True)
    result = recommend(payload, {'date': payload['demo_date'], 'tasks': payload['demo_tasks'], 'max_parallel': 2})
    (OUT / 'demo_tasks.json').write_text(json.dumps(payload['demo_tasks'], ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'schedule.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    import csv
    for name, rows in (('schedule.csv', result['tasks']), ('schedule_hourly.csv', result['hourly'])):
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['task_id'])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
    def js(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    script = '<script>window.DASHBOARD_DATA=' + js(payload) + ';window.DASHBOARD_SCHEDULE=' + js(result) + ';</script>'
    html = WEB.read_text(encoding='utf-8').replace('</head>', script + '\n</head>')
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    return result


def load_payload(args):
    path = OUT / 'forecast.json'
    if path.exists() and not args.rebuild:
        payload = json.loads(path.read_text(encoding='utf-8'))
        meta = payload.get('meta', {})
        desired_start = date_text(args.start) if args.start else meta.get('start')
        desired_end = date_text(args.end) if args.end else meta.get('end')
        from preprocess import RAW
        from final_forecast import source_fingerprint
        matches = (meta.get('schema_version') == 1 and meta.get('start') == desired_start and
                   meta.get('end') == desired_end and meta.get('no_plan') == args.no_plan and
                   meta.get('raw_sha256') == sha256(RAW.read_bytes()).hexdigest() and
                   meta.get('source_fingerprint') == source_fingerprint() and
                   ((args.threshold is None and meta.get('threshold_basis') == 'relative_q95') or
                    (args.threshold is not None and meta.get('threshold_basis') == 'provided_absolute' and
                     meta.get('threshold_value') == args.threshold)))
        if matches:
            print('저장된 최종 예측을 불러왔습니다. 재학습하려면 --rebuild를 사용하세요.', flush=True)
            return prepare_payload(payload)
    from final_forecast import build_bundle
    print('최종 모델 3종 × 시드 3개를 학습합니다. 이후 실행은 저장된 예측을 사용합니다.', flush=True)
    payload = build_bundle(OUT, args.start, args.end, no_plan=args.no_plan, threshold_value=args.threshold)
    return prepare_payload(payload)


def make_handler(payload):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content_type='application/json; charset=utf-8'):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8') if isinstance(value, (dict, list)) else value
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/index.html'):
                self.send(200, WEB.read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/forecast':
                self.send(200, payload)
            elif path == '/favicon.ico':
                self.send(204, b'')
            else:
                self.send(404, {'error': '요청한 페이지를 찾을 수 없습니다.'})

        def do_POST(self):
            if urlsplit(self.path).path != '/api/schedule':
                self.send(404, {'error': '요청한 기능을 찾을 수 없습니다.'})
                return
            # 다른 웹사이트에서 localhost 추천 API를 호출하지 않는다.
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.send(403, {'error': '현재 화면에서 일정 추천을 요청하세요.'})
                return
            try:
                if self.headers.get('Content-Type', '').split(';', 1)[0] != 'application/json':
                    raise ValueError('일정 입력은 JSON 형식이어야 합니다.')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('일정 입력 크기는 1MB 이하여야 합니다.')
                request = json.loads(self.rfile.read(length).decode('utf-8'), parse_constant=reject_constant)
                result = recommend(payload, request)
                self.send(200, result)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.send(400, {'error': str(exc)})

        def log_message(self, fmt, *args):
            # 브라우저 요청마다 콘솔을 채우지 않는다.
            pass
    return Handler


def main():
    parser = ArgumentParser(description='최종 모델·피크 저감 일정·HTML 운영 시연')
    parser.add_argument('--start', help='예측 시작일 YYYYMMDD 또는 YYYY-MM-DD')
    parser.add_argument('--end', help='예측 종료일 YYYYMMDD 또는 YYYY-MM-DD')
    parser.add_argument('--threshold', type=float, help='사용자가 확인한 절대 경보 임계값(원본 단위)')
    parser.add_argument('--no-plan', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--build-only', action='store_true', help='예측·추천·오프라인 HTML만 생성하고 종료')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='실행 후 브라우저 열기')
    args = parser.parse_args()
    if bool(args.start) != bool(args.end):
        parser.error('--start와 --end를 함께 지정해야 합니다.')
    if not 1 <= args.port <= 65535:
        parser.error('포트는 1~65535여야 합니다.')
    if args.threshold is not None and (not math.isfinite(args.threshold) or args.threshold <= 0):
        parser.error('임계값은 유한한 양수여야 합니다.')
    try:
        payload = load_payload(args)
        result = export_demo(payload)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'실행할 수 없습니다: {exc}\n')
    print(f'최종 예측 {len(payload["forecast"])}행 · 예시 일정 {result["mode"]} · 저장: {OUT}', flush=True)
    if args.build_only:
        return
    url = f'http://127.0.0.1:{args.port}'
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(payload))
    except OSError as exc:
        parser.exit(1, f'화면 서버를 시작할 수 없습니다: {exc}. 다른 --port를 지정하세요.\n')
    print(f'브라우저에서 {url} 을 여세요. 종료: Ctrl+C', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n화면 서버를 종료합니다.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
