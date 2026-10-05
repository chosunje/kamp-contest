"""Prediction entry points route alarms to direct max15 P90, including ties."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import forecast
import predict_test
from features import build
from preprocess import load
from test_quantile_analysis import synthetic_raw


class ConstantLegacyModel:
    def fit(self, *args, **kwargs):
        return self

    def predict(self, inputs):
        return np.full(len(inputs), 1000.)


def policy_models(*args, **kwargs):
    return [object()], {'threshold_q95': 186.}


def direct_predictions(models, X, cutoff, idx, no_plan=False):
    return np.resize([185., 186., 187.], len(idx))


class AlarmIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = build(load(synthetic_raw(days=60)))

    def test_forecast_uses_direct_max15_even_when_hourly_p90_is_high(self):
        with patch.object(forecast, 'make_model', return_value=ConstantLegacyModel()), \
             patch.object(forecast.policy, 'fit_models', side_effect=policy_models), \
             patch.object(forecast.policy, 'predict', side_effect=direct_predictions):
            out, _, _, threshold = forecast.forecast(self.X, 20210510)
        self.assertEqual(threshold, 186.)
        self.assertTrue(out.pred_hi.ge(threshold).all())
        np.testing.assert_array_equal(out.alarm, np.resize([False, True, True], 24))
        self.assertTrue(out.alarm_target.eq('target_max15').all())
        self.assertTrue(out.alarm_basis.eq('relative_q95').all())

    def test_range_prediction_and_day_summary_use_direct_max15_alarm(self):
        with patch.object(predict_test, 'make_model', return_value=ConstantLegacyModel()), \
             patch.object(predict_test.policy, 'fit_models', side_effect=policy_models), \
             patch.object(predict_test.policy, 'predict', side_effect=direct_predictions):
            out, _, _, _ = predict_test.predict_range(self.X, 20210510, 20210511)
        np.testing.assert_array_equal(out.alarm, np.resize([False, True, True], 48))
        days = predict_test.day_summary(out)
        self.assertTrue(days['경보'].all())
        self.assertTrue(days['일최대_15분P90'].eq(187.).all())
        self.assertTrue(days['일최대_15분환산'].gt(187.).all())

    def test_explicit_absolute_threshold_is_not_replaced_by_relative_q95(self):
        with patch.object(predict_test, 'make_model', return_value=ConstantLegacyModel()), \
             patch.object(predict_test.policy, 'fit_models', side_effect=policy_models), \
             patch.object(predict_test.policy, 'predict', side_effect=direct_predictions):
            out, _, _, threshold = predict_test.predict_range(self.X, 20210510, 20210510, thr=187.)
        self.assertEqual(threshold, 187.)
        np.testing.assert_array_equal(out.alarm, np.resize([False, False, True], 24))
        self.assertTrue(out.alarm_basis.eq('provided_absolute').all())


if __name__ == '__main__':
    unittest.main()
