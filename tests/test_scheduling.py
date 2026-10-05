"""Scheduling preserves plant constraints and distinguishes assumed load effects."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import scheduling


def forecast(peak_hour=8):
    return [{'hour': hour, 'pred': 80. if hour == peak_hour else 40.,
             'pred_max15': 110. if hour == peak_hour else 50.,
             'pred_hi_max15': 120. if hour == peak_hour else 60.,
             'alarm_threshold': 100.} for hour in range(24)]


def task(task_id='A', start=8, duration=1, earliest=8, latest=12, load=30.,
         movable=True, resource='', depends_on=None):
    return {'task_id': task_id, 'name': task_id, 'original_start': start,
            'duration': duration, 'earliest_start': earliest, 'latest_end': latest,
            'load': load, 'movable': movable, 'resource': resource,
            'depends_on': depends_on or []}


class SchedulingTests(unittest.TestCase):
    def test_simulation_removes_old_load_then_adds_shift_and_conserves_work(self):
        result = scheduling.schedule_day(forecast(), [task()])
        self.assertEqual(result['mode'], 'load_simulation')
        self.assertEqual(result['tasks'][0]['recommended_start'], 9)
        self.assertEqual(result['hourly'][8]['after_pred_hi_max15'], 90.)
        self.assertEqual(result['hourly'][9]['after_pred_hi_max15'], 90.)
        self.assertEqual(result['metrics']['peak_reduction'], 30.)
        self.assertEqual(result['metrics']['modeled_task_energy_before'], 30.)
        self.assertEqual(result['metrics']['modeled_task_energy_after'], 30.)
        self.assertEqual(sum(row['pred_hi_max15'] for row in result['hourly']),
                         sum(row['after_pred_hi_max15'] for row in result['hourly']))
        json.dumps(result, allow_nan=False)

    def test_fixed_jobs_and_dependencies_preserved(self):
        tasks = [task('A', 8, resource='line'),
                 task('B', 10, earliest=10, latest=13, load=5., movable=False, resource='line', depends_on=['A'])]
        result = scheduling.schedule_day(forecast(), tasks)
        by_id = {item['task_id']: item for item in result['tasks']}
        self.assertEqual(by_id['B']['recommended_start'], 10)
        self.assertLessEqual(by_id['A']['recommended_start'] + by_id['A']['duration'], 10)
        self.assertEqual(by_id['A']['recommended_start'], 9)
        self.assertEqual(result['metrics']['workload_hours'], 2)

    def test_shared_resource_and_parallel_limit_respected(self):
        tasks = [task('A', 8, load=None, resource='line'),
                 task('B', 9, earliest=8, latest=12, load=None, resource='line'),
                 task('C', 10, earliest=8, latest=12, load=None)]
        result = scheduling.schedule_day(forecast(), tasks, max_parallel=1)
        self.assertTrue(all(row['recommended_task_count'] <= 1 for row in result['hourly']))
        starts = [item['recommended_start'] for item in result['tasks']]
        self.assertEqual(len(set(starts)), 3)
        self.assertLess(result['metrics']['after_risk_exposure'], result['metrics']['before_risk_exposure'])

    def test_unknown_loads_only_claim_reduced_exposure(self):
        result = scheduling.schedule_day(forecast(), [task(load=None)])
        self.assertEqual(result['mode'], 'risk_avoidance')
        self.assertIsNone(result['metrics']['after_peak'])
        self.assertIsNone(result['metrics']['peak_reduction'])
        self.assertIsNone(result['metrics']['modeled_task_energy_after'])
        self.assertIsNone(result['hourly'][8]['after_pred_hi_max15'])
        self.assertEqual(result['metrics']['after_alarm_task_hours'], 0)

    def test_inconsistent_loads_disable_power_simulation(self):
        result = scheduling.schedule_day(forecast(), [task(load=200.)])
        self.assertEqual(result['mode'], 'risk_avoidance')
        self.assertIsNone(result['metrics']['peak_reduction'])
        self.assertTrue(any('초과' in warning for warning in result['warnings']))

    def test_hourly_mean_not_simulated_if_task_load_exceeds_it(self):
        rows = forecast()
        rows[8]['pred'] = 5.
        result = scheduling.schedule_day(rows, [task()])
        self.assertEqual(result['mode'], 'load_simulation')
        self.assertTrue(all(row['after_pred'] is None for row in result['hourly']))
        self.assertEqual(result['metrics']['peak_reduction'], 30.)

    def test_no_improvement_preserves_baseline(self):
        rows = forecast()
        for row in rows:
            row.update(pred=40., pred_max15=50., pred_hi_max15=60.)
        result = scheduling.schedule_day(rows, [task()])
        self.assertEqual(result['tasks'][0]['recommended_start'], 8)
        self.assertFalse(result['search']['improved'])
        self.assertEqual(result['metrics']['peak_reduction'], 0.)

    def test_bounded_search_keeps_valid_incumbent(self):
        with patch.object(scheduling, 'SEARCH_BUDGET', 1):
            result = scheduling.schedule_day(forecast(), [task()])
        self.assertEqual(result['tasks'][0]['recommended_start'], 8)
        self.assertFalse(result['search']['complete'])
        self.assertEqual(result['metrics']['peak_reduction'], 0.)

    def test_deterministic_and_does_not_mutate_inputs(self):
        rows, tasks = forecast(), [task(), task('B', 10, load=10.)]
        original = deepcopy((rows, tasks))
        self.assertEqual(scheduling.schedule_day(rows, tasks), scheduling.schedule_day(rows, tasks))
        self.assertEqual((rows, tasks), original)

    def test_empty_jobs_return_baseline_without_claiming_effect(self):
        result = scheduling.schedule_day(forecast(), [])
        self.assertEqual(result['mode'], 'risk_avoidance')
        self.assertEqual(result['tasks'], [])
        self.assertEqual(result['metrics']['workload_hours'], 0)
        self.assertIsNone(result['metrics']['peak_reduction'])

    def test_precedence_chain_can_move_as_a_group(self):
        rows = forecast()
        rows[9]['pred_hi_max15'] = 110.
        jobs = [task('A', 8, load=None), task('B', 9, earliest=8, latest=12, load=None, depends_on=['A'])]
        result = scheduling.schedule_day(rows, jobs)
        by_id = {item['task_id']: item for item in result['tasks']}
        self.assertEqual(by_id['A']['recommended_start'], 10)
        self.assertEqual(by_id['B']['recommended_start'], 11)
        self.assertEqual(result['metrics']['after_alarm_task_hours'], 0)

    def test_reject_invalid_original_constraints(self):
        cases = [
            [task(start=7)],
            [task(start=23, duration=2, earliest=23, latest=24)],
            [task('A', 8, resource='line'), task('B', 8, resource='line')],
            [task('A', 8), task('B', 8, depends_on=['A'])],
            [task('A', 8), task('B', 8), task('C', 8)],
            [task('A'), task('A')],
            [task(depends_on=['missing'])],
            [task('A', depends_on=['B']), task('B', 9, depends_on=['A'])],
        ]
        for jobs in cases:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                scheduling.schedule_day(forecast(), jobs)

    def test_reject_malformed_forecasts_and_values(self):
        cases = [forecast()[:23], forecast() + [forecast()[0]]]
        duplicate = forecast()
        duplicate[1]['hour'] = 0
        cases.append(duplicate)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                scheduling.schedule_day(rows, [task()])
        for value in (-1., float('nan'), float('inf'), True, '30'):
            with self.subTest(load=value), self.assertRaises(ValueError):
                scheduling.schedule_day(forecast(), [task(load=value)])
        for key, value in (('duration', 1.5), ('movable', 1), ('depends_on', 'A')):
            job = task()
            job[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                scheduling.schedule_day(forecast(), [job])


if __name__ == '__main__':
    unittest.main()
