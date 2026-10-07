"""경보 정책의 임계 단위, cutoff 입력 가용성 및 예측 경계 검증."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import alarm_policy as policy
from features import FEATURES, Q15, build
from model import ALARM_CFG
from preprocess import load
from test_quantile_analysis import synthetic_raw


ORIGINAL_FACTORY = policy.make_model


def small_factory(name, seed=0, params=None):
    return ORIGINAL_FACTORY(name, seed, params).set_params(n_estimators=8, n_jobs=1)


class AlarmPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_raw(days=70)
        cls.raw['15분'] = cls.raw['평균'] * 1.08
        cls.X = build(load(cls.raw))
        cls.cutoff = pd.Timestamp('2021-05-10')

    def test_loss_and_alpha_are_the_only_model_configuration_difference(self):
        before = deepcopy(ALARM_CFG)
        l1 = policy.make_config(label='L1')
        p90 = policy.make_config(label='P90')
        self.assertEqual(l1['params'].pop('objective'), 'l1')
        self.assertEqual(p90['params'].pop('objective'), 'quantile')
        self.assertEqual(p90['params'].pop('alpha'), .9)
        self.assertEqual(l1, p90)
        self.assertEqual(l1['target'], 'target_max15')
        self.assertEqual(len(l1['cols']), 33)
        self.assertNotIn('is_off', l1['cols'])
        self.assertNotIn('peak_w', l1)
        self.assertEqual(ALARM_CFG, before)
        self.assertEqual(policy.make_config(no_plan=True)['cols'], FEATURES['h7_no_plan'])

    def test_d1_input_matches_existing_features_without_actual_day_off(self):
        supplied = policy.available_inputs(self.X, self.cutoff)
        idx = self.X.index[self.X.dt.dt.normalize() == self.cutoff]
        pd.testing.assert_frame_equal(
            supplied.loc[idx, policy.ALARM_COLS],
            self.X.loc[idx, policy.ALARM_COLS], check_dtype=False,
        )

    def test_future_power_deletion_and_change_leave_available_inputs_unchanged(self):
        expected = policy.available_inputs(self.X, self.cutoff)
        for replacement in (np.nan, 10000.):
            raw = self.raw.copy(deep=True)
            raw.loc[raw['날짜'] >= int(self.cutoff.strftime('%Y%m%d')),
                    ['평균', *Q15]] = replacement
            altered = policy.available_inputs(build(load(raw)), self.cutoff)
            pd.testing.assert_frame_equal(expected[policy.ALARM_COLS],
                                          altered[policy.ALARM_COLS])
            pd.testing.assert_frame_equal(expected[policy.NO_PLAN_COLS],
                                          altered[policy.NO_PLAN_COLS])

    def test_d7_input_masks_unobserved_power_and_future_off_history(self):
        supplied = policy.available_inputs(self.X, self.cutoff)
        day7 = self.cutoff + pd.Timedelta(days=7)
        idx = self.X.index[self.X.dt.dt.normalize() == day7]
        self.assertTrue(supplied.loc[idx, 'lag168'].isna().all())
        self.assertTrue(supplied.loc[idx, 'last_week_day_mean'].isna().all())
        self.assertTrue(supplied.loc[idx, 'last_week_day_max'].isna().all())
        self.assertTrue(supplied.loc[idx, 'lag168_na'].eq(1).all())
        # A week before cutoff is observable, so lag336 remains available.
        np.testing.assert_allclose(supplied.loc[idx, 'lag336'],
                                   self.X.loc[idx, 'lag336'], equal_nan=True)
        off_cols = ['after_off', 'off_run_prev', 'days_since_off']
        future = self.X.dt.dt.normalize() > self.cutoff
        self.assertTrue(supplied.loc[future, off_cols].isna().all().all())
        today = self.X.dt.dt.normalize() == self.cutoff
        pd.testing.assert_frame_equal(supplied.loc[today, off_cols],
                                      self.X.loc[today, off_cols], check_dtype=False)

    def test_available_inputs_preserve_callers_and_reject_non_midnight_cutoff(self):
        before = self.X.copy(deep=True)
        supplied = policy.available_inputs(self.X, self.cutoff)
        pd.testing.assert_frame_equal(self.X, before)
        pd.testing.assert_index_equal(supplied.index, self.X.index)
        for invalid in ('2021-05-10 01:00', pd.NaT):
            with self.assertRaises(ValueError):
                policy.available_inputs(self.X, invalid)

    def test_threshold_uses_max15_and_accepts_only_finite_positive_values(self):
        training = pd.DataFrame({'target': [1., 2., 3.],
                                 'target_max15': [100., 200., 300.]})
        self.assertEqual(policy.threshold(training), 290.)
        self.assertEqual(policy.threshold(training, q=.97), 294.)
        self.assertEqual(policy.threshold(training, value=175.), 175.)
        for invalid in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                policy.threshold(training, value=invalid)
        for invalid in (0., 1., np.nan):
            with self.assertRaises(ValueError):
                policy.threshold(training, q=invalid)

    def test_alarm_flags_include_exact_threshold_ties(self):
        np.testing.assert_array_equal(policy.alarm_flags([174.9, 175., 175.1], 175.),
                                      [False, True, True])
        for invalid in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                policy.alarm_flags([invalid], 175.)

    def test_fit_predict_and_metadata_ignore_future_power_changes(self):
        before = self.X.copy(deep=True)
        idx = self.X.index[self.X.dt.dt.normalize() == self.cutoff]
        with patch.object(policy, 'make_model', side_effect=small_factory), \
             contextlib.redirect_stdout(io.StringIO()):
            models, meta = policy.fit_models(self.X, self.cutoff, seeds=(0, 1))
            expected = policy.predict(models, self.X, self.cutoff, idx)
            raw = self.raw.copy(deep=True)
            raw.loc[raw['날짜'] >= int(self.cutoff.strftime('%Y%m%d')),
                    ['평균', *Q15]] = 10000.
            altered = build(load(raw))
            altered_models, altered_meta = policy.fit_models(altered, self.cutoff,
                                                             seeds=(0, 1))
            actual = policy.predict(altered_models, altered, self.cutoff, idx)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(meta, altered_meta)
        self.assertEqual(meta['target_name'], 'target_max15')
        self.assertEqual(meta['cols'], policy.ALARM_COLS)
        self.assertEqual(meta['train_cutoff'], self.cutoff)
        training = self.X[(self.X.dt < self.cutoff) & self.X.train_ok_strict]
        self.assertEqual(meta['train_n'], len(training))
        self.assertEqual(meta['threshold_q95'], training.target_max15.quantile(.95))
        self.assertEqual(meta['threshold_q97'], training.target_max15.quantile(.97))
        self.assertEqual(meta['threshold_q98'], training.target_max15.quantile(.98))
        pd.testing.assert_frame_equal(self.X, before)

    def test_insufficient_training_data_fails_before_fitting(self):
        with patch.object(policy, 'make_model') as factory:
            with self.assertRaises(ValueError):
                policy.fit_models(self.X, '2021-04-02', seeds=(0,))
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
