"""오프라인 운영 화면(outputs/final_app/index.html)의 계약을 확인한다.

1. 저장본이 보고서 5장 수치(피크 199.1 → 195.1, 경보 8 → 7시간)를 그대로 싣고 있는가
2. 저장본이 외부 리소스 없이 열리고, web/index.html 템플릿과 어긋나지 않았는가
3. 화면의 업로드 검증이 tests/fixtures/dashboard 의 예시 CSV 에 기대대로 반응하는가
4. '추천 일정 계산' 뒤의 스케줄러가 fixtures/dashboard/README.txt 의 시나리오대로 동작하는가

모델은 학습하지 않는다. 3번은 web/index.html 의 실제 JS 함수를 node 로 실행하며,
node 가 없으면 그 테스트만 건너뛴다. outputs/final_app 이 없는 저장소(제출용 최소 구성)에서는
1·2·4번을 건너뛴다. python run_dashboard.py --build-only 로 만든 뒤 다시 돌리면 된다.

실행  python -m unittest tests.test_offline_html -v
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from scheduling import schedule_day

APP = ROOT / 'outputs' / 'final_app'
WEB = ROOT / 'web' / 'index.html'
FIX = Path(__file__).resolve().parent / 'fixtures' / 'dashboard'
DEMO_DATE = '2021-09-09'
# run_dashboard.export_demo 가 '</head>' 앞에 넣는 데이터 스크립트
INJECT = re.compile(r'<script>window\.DASHBOARD_DATA=.*?;</script>\n', re.S)
HAS_APP = all((APP / name).exists() for name in ('index.html', 'forecast.json', 'demo_tasks.json'))
NO_APP = 'outputs/final_app 이 없음 — python run_dashboard.py --build-only 뒤에 실행'


def embedded():
    """저장본에 주입된 DASHBOARD_DATA · DASHBOARD_SCHEDULE 을 꺼낸다."""
    html = (APP / 'index.html').read_text(encoding='utf-8')
    dec = json.JSONDecoder()
    i = html.index('window.DASHBOARD_DATA=') + len('window.DASHBOARD_DATA=')
    data, end = dec.raw_decode(html, i)
    j = html.index('window.DASHBOARD_SCHEDULE=', end) + len('window.DASHBOARD_SCHEDULE=')
    schedule, _ = dec.raw_decode(html, j)
    return html, data, schedule


def moved(result):
    return {t['task_id']: (t['original_start'], t['recommended_start'])
            for t in result['tasks'] if t['changed']}


@unittest.skipUnless(HAS_APP, NO_APP)
class OfflineFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html, cls.data, cls.schedule = embedded()

    def test_embedded_forecast_is_the_saved_forecast(self):
        saved = json.loads((APP / 'forecast.json').read_text(encoding='utf-8'))
        self.assertEqual(self.data['forecast'], saved['forecast'])
        dates = {r['date'] for r in self.data['forecast']}
        self.assertEqual(len(self.data['forecast']), 24 * len(dates))

    def test_embedded_schedule_carries_report_numbers(self):
        # 보고서 5장 「일정 조정 권고 알고리즘의 동작 원리」와 표 5-1 의 값
        s, m = self.schedule, self.schedule['metrics']
        self.assertEqual((s['date'], s['mode']), (DEMO_DATE, 'load_simulation'))
        self.assertAlmostEqual(m['before_peak'], 199.1, places=1)
        self.assertAlmostEqual(m['after_peak'], 195.1, places=1)
        self.assertAlmostEqual(m['peak_reduction'], 4.0, places=1)
        self.assertEqual((m['before_alarm_hours'], m['after_alarm_hours']), (8, 7))
        self.assertEqual(s['search']['nodes'], 42)
        self.assertEqual(moved(s), {'B': (14, 16), 'C': (16, 17)})

    def test_opens_without_network(self):
        external = re.findall(r'(?:src|href)\s*=\s*["\']https?://|url\(\s*["\']?https?://|@import',
                              self.html)
        self.assertEqual(external, [])

    def test_saved_file_is_template_plus_data(self):
        # web/index.html 을 고친 뒤 python run_dashboard.py --build-only 를 다시 돌리지 않으면 걸린다
        self.assertEqual(INJECT.sub('', self.html, count=1), WEB.read_text(encoding='utf-8'))


# fixtures/dashboard/README.txt [1] 표와 같은 내용이다. 둘 중 하나만 고치지 말 것
UPLOADS = {
    '01_기본_7일.csv': {'alarm_hours': {
        '2021-09-08': 6, '2021-09-09': 8, '2021-09-10': 5, '2021-09-11': 0,
        '2021-09-12': 0, '2021-09-13': 1, '2021-09-14': 4}},
    '02_임계193_절대기준.csv': {'alarm_hours': {
        '2021-09-08': 2, '2021-09-09': 2, '2021-09-10': 1, '2021-09-11': 0,
        '2021-09-12': 0, '2021-09-13': 0, '2021-09-14': 0}},
    'E1_시간누락.csv': {'error': '2021-09-09에 0~23시의 서로 다른 24개 예측 행이 필요합니다.'},
    'E2_음수값.csv': {'error': '11번째 행의 pred_hi_max15 값을 확인하세요.'},
    'E3_임계혼재.csv': {'error': '2021-09-09의 경보 기준은 하루 동안 같아야 합니다.'},
}

HARNESS = r'''
const fs = require('fs'), path = require('path');
/*FUNCS*/
const dir = process.argv[2], out = {};
for (const name of fs.readdirSync(dir).filter(n => n.endsWith('.csv'))) {
  try {
    const rows = normalizeForecast(parseCSV(fs.readFileSync(path.join(dir, name), 'utf8')));
    const hours = {};
    for (const r of rows) hours[r.date] = (hours[r.date] || 0) + (r.alarm ? 1 : 0);
    out[name] = {alarm_hours: hours};
  } catch (e) { out[name] = {error: e.message}; }
}
process.stdout.write(JSON.stringify(out));
'''


def js_functions(*names):
    """web/index.html 에서 최상위 함수 정의를 그대로 잘라 온다 (화면과 같은 코드로 검증하기 위해)."""
    src = WEB.read_text(encoding='utf-8')
    out = []
    for name in names:
        m = re.search(rf'\bconst {name}=', src)
        if m:
            out.append(src[m.start():src.index(';', m.end()) + 1])
            continue
        start = src.index(f'function {name}(')
        depth = 0
        for k in range(src.index('{', start), len(src)):
            depth += {'{': 1, '}': -1}.get(src[k], 0)
            if depth == 0:
                out.append(src[start:k + 1])
                break
    return '\n'.join(out)


@unittest.skipUnless(shutil.which('node'), 'node 가 없어 화면 JS 검증을 건너뜀')
class UploadValidationTests(unittest.TestCase):
    def test_fixtures_get_the_documented_response(self):
        code = HARNESS.replace('/*FUNCS*/', js_functions('num', 'dateISO', 'parseCSV', 'normalizeForecast'))
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'harness.js'
            script.write_text(code, encoding='utf-8')
            run = subprocess.run(['node', str(script), str(FIX)], capture_output=True,
                                 text=True, encoding='utf-8', timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        got = json.loads(run.stdout)
        self.assertEqual(set(got), set(UPLOADS), '폴더의 CSV 와 README·테스트 목록이 어긋남')
        for name, want in UPLOADS.items():
            with self.subTest(name):
                self.assertEqual(got[name], want)

    def test_threshold_fixture_keeps_stale_alarm_column(self):
        # 02 파일의 alarm 열은 일부러 옛 기준(187) 그대로다. 화면이 이 열을 믿지 않고
        # P90 >= 기준으로 다시 판정하는지를 보려는 장치라, 열 자체는 01 과 같아야 한다
        def alarm_column(name):
            lines = (FIX / name).read_text(encoding='utf-8-sig').splitlines()[1:]
            return [line.rsplit(',', 1)[1] for line in lines]
        self.assertEqual(alarm_column('02_임계193_절대기준.csv'), alarm_column('01_기본_7일.csv'))


@unittest.skipUnless(HAS_APP, NO_APP)
class ScheduleScenarioTests(unittest.TestCase):
    """fixtures/dashboard/README.txt [2] 의 시나리오 표와 1:1 로 대응한다."""

    @classmethod
    def setUpClass(cls):
        forecast = json.loads((APP / 'forecast.json').read_text(encoding='utf-8'))['forecast']
        cls.day = [r for r in forecast if r['date'] == DEMO_DATE]
        cls.tasks = json.loads((APP / 'demo_tasks.json').read_text(encoding='utf-8'))

    def edit(self, **changes):
        """edit(B={'movable': False}) 처럼 작업 ID 별로 값을 바꾼 복사본."""
        tasks = deepcopy(self.tasks)
        for task in tasks:
            task.update(changes.get(task['task_id'], {}))
        return tasks

    def test_s1_demo_as_is_reproduces_saved_result(self):
        result = schedule_day(self.day, self.tasks, max_parallel=2)
        self.assertEqual(result['metrics'], embedded()[2]['metrics'])
        self.assertEqual(moved(result), {'B': (14, 16), 'C': (16, 17)})

    def test_s2_fixed_b_moves_a_instead(self):
        result = schedule_day(self.day, self.edit(B={'movable': False}), max_parallel=2)
        self.assertAlmostEqual(result['metrics']['after_peak'], 195.1, places=1)
        self.assertEqual(moved(result), {'A': (14, 16), 'C': (16, 18)})

    def test_s3_one_parallel_rejects_overlapping_original_plan(self):
        with self.assertRaisesRegex(ValueError, '최대 동시 작업 수를 초과'):
            schedule_day(self.day, self.tasks, max_parallel=1)

    def test_s3b_one_parallel_after_separating_a_and_b(self):
        result = schedule_day(self.day, self.edit(B={'original_start': 10}), max_parallel=1)
        m = result['metrics']
        self.assertAlmostEqual(m['after_peak'], 196.3, places=1)
        self.assertEqual((m['before_alarm_hours'], m['after_alarm_hours']), (8, 6))
        self.assertEqual(moved(result), {'A': (14, 9), 'B': (10, 11)})

    def test_s4_no_load_falls_back_to_alarm_avoidance(self):
        blank = {'load': None}
        result = schedule_day(self.day, self.edit(A=blank, B=blank, C=blank, D=blank), max_parallel=2)
        m = result['metrics']
        self.assertEqual(result['mode'], 'risk_avoidance')
        self.assertIsNone(m['after_peak'])
        self.assertEqual((m['before_alarm_task_hours'], m['after_alarm_task_hours']), (5, 2))
        self.assertEqual(moved(result), {'A': (14, 16), 'B': (14, 16), 'C': (16, 18)})

    def test_s5_unknown_dependency_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '선행 작업 Z가 없습니다'):
            schedule_day(self.day, self.edit(C={'depends_on': ['Z']}), max_parallel=2)


if __name__ == '__main__':
    unittest.main()
