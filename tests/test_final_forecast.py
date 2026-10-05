"""Final-model cutoff safety and portable model artifacts, without full training."""
import contextlib
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import final_forecast as final
from features import FEATURES, Q15, build
from model import FINAL_CFG, MAX15_CFG
from preprocess import load
from test_quantile_analysis import synthetic_raw


ORIGINAL_FACTORY = final.make_model
ORIGINAL_WEIGHTS = final.historical_clone_weights


def small_factory(name, seed=0, params=None):
    return ORIGINAL_FACTORY(name, seed, params).set_params(n_estimators=8, n_jobs=1)


class FinalForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_raw(days=70)
        cls.raw[Q15[0]] = cls.raw['평균'] * 1.08
        # One duplicate wholly inside history; another only crosses the cutoff.
        power_cols = ['평균', *Q15]
        for source, destination in ((20210402, 20210403), (20210405, 20210510)):
            cls.raw.loc[cls.raw['날짜'] == destination, power_cols] = (
                cls.raw.loc[cls.raw['날짜'] == source, power_cols].to_numpy()
            )
        cls.X = build(load(cls.raw))
        cls.start, cls.end = '2021-05-10', '2021-05-11'

    def fit(self, X=None, **kwargs):
        with patch.object(final, 'make_model', side_effect=small_factory), \
             contextlib.redirect_stdout(io.StringIO()):
            return final.forecast_bundle(
                self.X if X is None else X, self.start, self.end, **kwargs,
            )

    def test_all_forecast_targets_and_weights_ignore_future_power(self):
        before = self.X.copy(deep=True)
        configs_before = deepcopy((FINAL_CFG, MAX15_CFG))
        recorded = []

        def capture_weights(history, training):
            weights = ORIGINAL_WEIGHTS(history, training)
            recorded.append(weights.copy())
            return weights

        with patch.object(final, 'historical_clone_weights', side_effect=capture_weights):
            expected, expected_meta, expected_models = self.fit(seeds=(0, 1))
            expected_weights = recorded[-3:]
            for replacement in (np.nan, 10000.):
                altered_raw = self.raw.copy(deep=True)
                altered_raw.loc[altered_raw['날짜'] >= 20210510,
                                ['평균', *Q15]] = replacement
                altered = build(load(altered_raw))
                self.assertFalse(self.X.is_clone.equals(altered.is_clone))
                actual, meta, models = self.fit(altered, seeds=(0, 1))
                pd.testing.assert_frame_equal(expected, actual)
                for key in ('train_cutoff', 'train_rows', 'configs', 'threshold_value',
                            'threshold_basis', 'feature_count', 'seeds'):
                    self.assertEqual(expected_meta[key], meta[key])
                for original, changed in zip(expected_weights, recorded[-3:]):
                    np.testing.assert_array_equal(original, changed)
                for kind in expected_models:
                    for original, changed in zip(expected_models[kind], models[kind]):
                        self.assertEqual(original.booster_.model_to_string(),
                                         changed.booster_.model_to_string())
        self.assertEqual((FINAL_CFG, MAX15_CFG), configs_before)
        pd.testing.assert_frame_equal(self.X, before)
        self.assertEqual(len(expected), 48)
        self.assertEqual(expected_meta['feature_count'], 33)
        self.assertNotIn('is_off', expected_meta['configs']['point']['cols'])
        expected_threshold = self.X.loc[
            (self.X.dt < pd.Timestamp(self.start)) & self.X.train_ok_strict,
            'target_max15',
        ].quantile(.95)
        self.assertEqual(expected_meta['threshold_value'], expected_threshold)
        np.testing.assert_array_equal(expected.alarm,
                                      expected.pred_hi_max15 >= expected_threshold)

    def test_nine_saved_models_reproduce_forecasts_and_keep_manifest_hashes(self):
        rows, meta, models = self.fit(seeds=(0, 1, 2))
        meta_before = deepcopy(meta)
        with tempfile.TemporaryDirectory() as directory:
            payload = final.write_bundle(directory, rows, meta, models)
            saved = Path(directory)
            manifest = json.loads((saved / 'model_manifest.json').read_text('utf-8'))
            self.assertEqual(len(manifest['model_files']), 9)
            self.assertEqual(manifest['seeds'], [0, 1, 2])
            self.assertEqual(payload['meta'], manifest)
            self.assertEqual(manifest['raw_sha256'], sha256(final.RAW.read_bytes()).hexdigest())
            self.assertEqual(json.loads((saved / 'forecast.json').read_text('utf-8')), payload)
            restored_csv = pd.read_csv(saved / 'forecast.csv')
            pd.testing.assert_frame_equal(restored_csv, rows, check_exact=False,
                                          rtol=1e-14, atol=1e-14)
            safe = final.policy.available_inputs(self.X, self.start)
            indices = self.X.index[
                (self.X.dt >= pd.Timestamp(self.start)) &
                (self.X.dt < pd.Timestamp(self.end) + pd.Timedelta(days=1))
            ]
            outputs = {'point': 'pred', 'max15': 'pred_max15', 'alarm': 'pred_hi_max15'}
            for kind, column in outputs.items():
                predictions = []
                for file in manifest['model_files']:
                    if file['kind'] != kind:
                        continue
                    path = saved / file['path']
                    self.assertEqual(file['sha256'], sha256(path.read_bytes()).hexdigest())
                    booster = lgb.Booster(model_file=str(path))
                    predictions.append(booster.predict(
                        safe.loc[indices, manifest['configs'][kind]['cols']], num_threads=1,
                    ))
                self.assertEqual(len(predictions), 3)
                np.testing.assert_array_equal(rows[column], np.mean(predictions, axis=0))
        self.assertEqual(meta, meta_before)

    def test_no_plan_and_explicit_threshold_use_requested_operational_contract(self):
        rows, meta, _ = self.fit(no_plan=True, seeds=(0,), threshold_value=175.)
        self.assertEqual(meta['threshold_basis'], 'provided_absolute')
        self.assertEqual(meta['feature_count'], len(FEATURES['h7_no_plan']))
        for cfg in meta['configs'].values():
            self.assertEqual(cfg['cols'], FEATURES['h7_no_plan'])
        self.assertTrue(rows.alarm_threshold.eq(175.).all())
        np.testing.assert_array_equal(rows.alarm, rows.pred_hi_max15 >= 175.)

    def test_invalid_ranges_and_incomplete_rows_fail_before_training(self):
        reversed_range = (self.X, self.end, self.start)
        outside_range = (self.X, '2021-07-01', '2021-07-02')
        missing = self.X.drop(self.X.index[self.X.dt == pd.Timestamp(self.start)][0])
        bad_cases = (reversed_range, outside_range, (missing, self.start, self.end))
        with patch.object(final, 'make_model') as factory:
            for X, start, end in bad_cases:
                with self.assertRaises(ValueError):
                    final.forecast_bundle(X, start, end)
            with self.assertRaises(ValueError):
                final.forecast_bundle(self.X, self.start, self.end, seeds=())
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
