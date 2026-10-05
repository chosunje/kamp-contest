"""Local HTTP contract and invalid user input, without training any models."""
import contextlib
from copy import deepcopy
from hashlib import sha256
import http.client
from http.server import ThreadingHTTPServer
import json
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dashboard as dashboard
import final_forecast as final


def fixture_payload():
    return {
        'meta': {'schema_version': 1}, 'dates': ['2021-09-14'],
        'forecast': [
            {'date': '2021-09-14', 'hour': hour, 'pred': 90., 'pred_max15': 100.,
             'pred_hi_max15': 150. if hour == 9 else 110., 'alarm_threshold': 120.,
             'alarm': hour == 9}
            for hour in range(24)
        ],
    }


def fixture_task():
    return {'task_id': 'A', 'name': 'Task A', 'original_start': 9, 'duration': 1,
            'earliest_start': 8, 'latest_end': 18, 'load': 10., 'movable': True,
            'resource': 'asset-A', 'depends_on': []}


class DashboardHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = fixture_payload()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.make_handler(cls.payload))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = json.loads(response.read().decode('utf-8'))
            return response.status, data
        finally:
            connection.close()

    def post(self, request):
        return self.request('POST', '/api/schedule', json.dumps(request).encode('utf-8'),
                            {'Content-Type': 'application/json'})

    def test_forecast_get_and_successful_schedule_response(self):
        before = deepcopy(self.payload)
        status, data = self.request('GET', '/api/forecast')
        self.assertEqual(status, 200)
        self.assertEqual(data, self.payload)
        status, result = self.post({'date': '20210914', 'tasks': [fixture_task()], 'max_parallel': 2})
        self.assertEqual(status, 200)
        self.assertEqual(result['date'], '2021-09-14')
        self.assertEqual(result['mode'], 'load_simulation')
        self.assertEqual(len(result['hourly']), 24)
        self.assertEqual(result['tasks'][0]['task_id'], 'A')
        self.assertGreater(result['metrics']['peak_reduction'], 0.)
        self.assertEqual(result['metrics']['modeled_task_energy_before'],
                         result['metrics']['modeled_task_energy_after'])
        self.assertEqual(self.payload, before)

    def test_uploaded_forecast_uses_its_own_date_and_rejects_incomplete_day(self):
        uploaded = deepcopy(self.payload['forecast'])
        for row in uploaded:
            row['date'] = '2021-09-15'
        status, result = self.post({'date': '2021-09-15', 'forecast': uploaded, 'tasks': []})
        self.assertEqual(status, 200)
        self.assertEqual(result['date'], '2021-09-15')
        status, error = self.post({'date': '2021-09-15', 'forecast': uploaded[:-1], 'tasks': []})
        self.assertEqual(status, 400)
        self.assertTrue(error['error'])

    def test_invalid_json_numbers_resource_and_request_size_return_json_errors(self):
        invalid_task = fixture_task()
        invalid_task['resource'] = 15
        bad_inputs = [[], {'date': 'bad-date'},
                      {'date': '2021-09-14', 'tasks': [invalid_task]},
                      {'date': '2021-09-14', 'tasks': [], 'max_parallel': True}]
        for request in bad_inputs:
            with self.subTest(request=request):
                status, error = self.post(request)
                self.assertEqual(status, 400)
                self.assertTrue(error['error'])
        for body in (b'{invalid', b'{"date":"2021-09-14","tasks":[],"value":NaN}',
                     b'{"date":"2021-09-14","tasks":[],"value":Infinity}'):
            status, error = self.request('POST', '/api/schedule', body)
            self.assertEqual(status, 400)
            self.assertTrue(error['error'])
        status, error = self.request('POST', '/api/schedule', b'{}',
                                    {'Content-Length': str(dashboard.MAX_BODY + 1)})
        self.assertEqual(status, 400)
        self.assertTrue(error['error'])

    def test_missing_routes_and_cross_origin_post_return_expected_status(self):
        for method in ('GET', 'POST'):
            status, error = self.request(method, '/not-found', b'{}' if method == 'POST' else None)
            self.assertEqual(status, 404)
            self.assertTrue(error['error'])
        status, error = self.request('POST', '/api/schedule', b'{}',
                                    {'Origin': 'https://example.invalid'})
        self.assertEqual(status, 403)
        self.assertTrue(error['error'])


class DashboardCacheTests(unittest.TestCase):
    def test_matching_cache_avoids_training_and_stale_model_or_data_rebuilds(self):
        payload = fixture_payload()
        payload['meta'].update({
            'start': '2021-09-14', 'end': '2021-09-14', 'no_plan': False,
            'raw_sha256': sha256(final.RAW.read_bytes()).hexdigest(),
            'source_fingerprint': final.source_fingerprint(),
            'threshold_basis': 'relative_q95', 'threshold_value': 120.,
        })
        args = SimpleNamespace(start=None, end=None, no_plan=False,
                               threshold=None, rebuild=False)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(dashboard, 'OUT', Path(directory)), \
             patch.object(final, 'build_bundle', return_value=payload) as builder:
            path = Path(directory) / 'forecast.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                restored = dashboard.load_payload(args)
            builder.assert_not_called()
            self.assertEqual(restored['forecast'], payload['forecast'])
            for key, value in (('raw_sha256', 'changed-data'),
                               ('source_fingerprint', 'changed-model'),
                               ('schema_version', 0), ('no_plan', True),
                               ('threshold_basis', 'provided_absolute')):
                with self.subTest(stale_field=key):
                    stale = deepcopy(payload)
                    stale['meta'][key] = value
                    path.write_text(json.dumps(stale), encoding='utf-8')
                    builder.reset_mock()
                    with contextlib.redirect_stdout(io.StringIO()):
                        dashboard.load_payload(args)
                    builder.assert_called_once_with(
                        Path(directory), None, None, no_plan=False, threshold_value=None,
                    )


if __name__ == '__main__':
    unittest.main()
