"""Alarm evaluation counts complete days and keeps the true peak event fixed."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import peak_alarm_max15 as evaluation


class AlarmEvaluationTests(unittest.TestCase):
    def test_daily_metrics_include_off_days_and_exclude_partial_days(self):
        rows = []
        for date, count, off, predicted, actual in [
            (20210701, 24, False, 200., 200.),
            (20210702, 24, True, 200., 20.),
            (20210703, 23, False, 100., 200.),
        ]:
            for _ in range(count):
                rows.append({'날짜': date, 'target': actual, 'pred': predicted,
                             'is_off': off, 'peak_threshold': 180., 'alarm_threshold': 180.,
                             'alarm': predicted >= 180., 'real_peak': actual >= 180.})
        metrics, days = evaluation.alarm_metrics(pd.DataFrame(rows))
        self.assertEqual(metrics['hour_n'], 71)
        self.assertEqual(metrics['hour_fn'], 23)
        self.assertEqual(metrics['complete_days'], 2)
        self.assertEqual(metrics['incomplete_days_excluded'], 1)
        self.assertEqual(metrics['day_precision_pct'], 50.)
        self.assertEqual(metrics['day_recall_pct'], 100.)
        self.assertEqual(metrics['day_alarm_rate_pct'], 100.)
        self.assertEqual(len(days), 2)

    def test_sensitivity_changes_prediction_cutoff_not_true_peak_definition(self):
        dates = [20210701] * 24 + [20210702] * 24
        frames = []
        for model in ('L1', 'P90'):
            frames.append(pd.DataFrame({
                '날짜': dates, 'fold': '2021-07', 'model': model, 'seed': 'ensemble',
                'target': [185.] * 24 + [170.] * 24,
                'pred': [185.] * 24 + [180.] * 24, 'peak_threshold': 180.,
            }))
        thresholds = pd.DataFrame({'fold': ['2021-07'], 'threshold_q95': [180.],
                                   'threshold_q97': [190.], 'threshold_q98': [200.]})
        metrics, _ = evaluation.evaluate(pd.concat(frames, ignore_index=True), thresholds)
        core = metrics[(metrics.period == evaluation.CORE) & (metrics.model == 'P90')]
        np.testing.assert_array_equal(core.day_actual_peak_days, [1, 1, 1])
        np.testing.assert_array_equal(core.day_recall_pct, [100., 0., 0.])
        self.assertTrue(core.real_peak_q.eq(.95).all())
        self.assertEqual(core.iloc[0].day_precision_pct, 50.)


if __name__ == '__main__':
    unittest.main()
