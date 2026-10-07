"""Small regression checks for the quantile experiment; no full pipeline training."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import model as model_module
import quantile_analysis as qa
from features import build
from preprocess import load


def synthetic_raw(days=42):
    """Continuous hourly input with distinct working and nonworking curves."""
    dt = pd.date_range('2021-04-01', periods=days * 24, freq='h')
    day_number = np.arange(len(dt)) // 24
    hour = dt.hour.to_numpy()
    off = dt.dayofweek.to_numpy() == 6
    production = np.where((hour >= 8) & (hour <= 19) & ~off, 1000, 0)
    power = np.where(off, 20 + day_number * .1,
                     45 + day_number * .4 + (production > 0) * 110 + hour * .1)
    return pd.DataFrame({
        '날짜': dt.strftime('%Y%m%d').astype(int), '시간': hour,
        '15분': power, '30분': power, '45분': power, '60분': power,
        '평균': power, '생산량': production, '기온': 22 + day_number * .1,
        '풍속': 2., '습도': 50., '강수량': 0., '공장인원': 5,
        'day': dt.dayofweek + 1, 'd': dt.day, 'm': dt.month,
    })


def small_factory(name, seed=0, params=None):
    # params 는 model.make_model 이 받는 설정 덮어쓰기다.
    # 여기서는 그대로 넘기고 나무 수만 줄여 테스트를 빠르게 만든다.
    return ORIGINAL_FACTORY(name, seed, params).set_params(n_estimators=12, n_jobs=1)


ORIGINAL_FACTORY = model_module.make_model


class QuantileAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_raw()
        cls.X = build(load(cls.raw))
        cls.X['ym'] = cls.X.dt.dt.to_period('M').astype(str)

    def test_loss_is_the_only_model_configuration_difference(self):
        l1 = ORIGINAL_FACTORY('lgbm', seed=2).get_params()
        p90 = ORIGINAL_FACTORY('lgbm_q90', seed=2).get_params()
        self.assertEqual(l1.pop('objective'), 'l1')
        self.assertEqual(p90.pop('objective'), 'quantile')
        self.assertEqual(p90.pop('alpha'), .9)
        self.assertEqual(l1, p90)

    def test_existing_l1_factory_preserves_legacy_predictions(self):
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(240, 5)), columns=list('abcde'))
        x.loc[::13, 'b'] = np.nan
        y = 100 + 8 * x.a + 3 * x.c + rng.normal(size=len(x))
        old = lgb.LGBMRegressor(
            objective='l1', n_estimators=12, learning_rate=.05,
            num_leaves=31, min_data_in_leaf=20, feature_fraction=.8,
            bagging_fraction=.8, bagging_freq=1, verbose=-1, seed=0, n_jobs=1,
        )
        new = small_factory('lgbm', seed=0)
        weights = np.where(np.arange(len(x)) % 3 == 0, .3, 1.)
        old.fit(x.iloc[:200], y.iloc[:200], sample_weight=weights[:200])
        new.fit(x.iloc[:200], y.iloc[:200], sample_weight=weights[:200])
        np.testing.assert_array_equal(old.predict(x.iloc[200:]),
                                      new.predict(x.iloc[200:]))

    def test_reference_predictions_match_existing_rolling_experiment(self):
        with patch.object(qa, 'FOLDS', ['2021-05']), \
             patch.object(qa, 'SEEDS', (0, 1)), \
             patch.object(qa, 'make_model', side_effect=small_factory), \
             patch.object(model_module, 'FOLDS', ['2021-05']), \
             patch.object(model_module, 'make_model', side_effect=small_factory), \
             contextlib.redirect_stdout(io.StringIO()):
            predictions, _ = qa.compare(self.X)
            for seed in (0, 1):
                legacy = model_module.rolling_eval(
                    self.X, qa.PROTOCOLS['reference'], clone='weight', seed=seed,
                )
                current = predictions[
                    (predictions.protocol == 'reference') &
                    (predictions.model == 'L1') & (predictions.seed == str(seed))
                ]
                np.testing.assert_array_equal(legacy.y, current.target)
                np.testing.assert_array_equal(legacy.p, current.pred)
                np.testing.assert_array_equal(legacy.peak, current.is_peak)
            d1 = predictions[predictions.protocol == 'd1']
            pd.testing.assert_series_equal(
                d1.prediction_cutoff.reset_index(drop=True),
                d1.dt.dt.normalize().reset_index(drop=True),
                check_names=False, check_dtype=False,
            )
            self.assertTrue((d1.train_cutoff <= d1.prediction_cutoff).all())

    def test_d1_training_cutoff_precedes_first_valid_hour(self):
        frame = self.X.copy(deep=True)
        invalid_midnight = frame.dt == pd.Timestamp('2021-05-01')
        frame.loc[invalid_midnight, ['train_ok', 'train_ok_strict']] = False
        with patch.object(qa, 'FOLDS', ['2021-05']), \
             patch.object(qa, 'SEEDS', (0,)), \
             patch.object(qa, 'make_model', side_effect=small_factory), \
             contextlib.redirect_stdout(io.StringIO()):
            predictions, _ = qa.compare(frame)
        d1 = predictions[predictions.protocol == 'd1']
        reference = predictions[predictions.protocol == 'reference']
        self.assertTrue(d1.train_cutoff.eq(pd.Timestamp('2021-05-01')).all())
        self.assertTrue(reference.train_cutoff.eq(pd.Timestamp('2021-05-01 01:00')).all())
        self.assertTrue((d1.train_cutoff <= d1.prediction_cutoff).all())

    def test_future_power_audit_preserves_d1_but_detects_reference_leakage(self):
        class InputOnlyModel:
            def predict(self, frame):
                return (frame['prod'] * .03 + frame['hour']).to_numpy()

        fitted = {('d1', '2021-05', label, 0): InputOnlyModel()
                  for label in qa.MODELS}
        with patch.object(qa, 'FOLDS', ['2021-05']), patch.object(qa, 'SEEDS', (0,)):
            audit = qa.audit_future_power(self.raw, self.X, fitted)
        expected_days = self.X.loc[
            (self.X.ym == '2021-05') & self.X.train_ok & ~self.X.is_clone, '날짜'
        ].nunique()
        self.assertEqual(len(audit), expected_days * 2)
        self.assertEqual(set(audit['mode']), {'deleted', 'changed'})
        self.assertTrue(audit.feature_equal.all())
        self.assertTrue(audit.prediction_max_delta.eq(0).all())
        self.assertTrue(audit.reference_is_off_changed.any())
        off_date = 20210502
        # 전력을 '바꾸면' 여전히 is_off 가 흔들린다 — reference 규약의 누수는 그대로 탐지된다.
        # 전력을 '지우면' 더 이상 흔들리지 않는다. preprocess.load() 가 그 경우에만
        # 생산계획(일 생산량 0)으로 휴무를 판정하도록 바뀌었기 때문이다 (생산계획 가용 가정).
        # 진짜 테스트 구간은 전력이 비어 있으므로 이쪽이 실제 운영 조건이다 —
        # 고치기 전에는 휴무일을 가동일로 보고 예측해 그 날 MAE 가 0.8 → 19.0 이었다.
        at = audit[audit['날짜'] == off_date]
        self.assertTrue(at.loc[at['mode'] == 'changed', 'reference_is_off_changed'].all())
        self.assertFalse(at.loc[at['mode'] == 'deleted', 'reference_is_off_changed'].any())

    def test_memory_input_does_not_mutate_callers_or_change_features(self):
        raw_before = self.raw.copy(deep=True)
        prepared = load(self.raw)
        prepared_before = prepared.copy(deep=True)
        actual = build(prepared)
        expected = self.X.drop(columns='ym')
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(self.raw, raw_before)
        pd.testing.assert_frame_equal(prepared, prepared_before)

    def test_historical_clone_weights_ignore_future_duplicate_curve(self):
        raw = synthetic_raw(days=4)
        dates = raw['날짜'].drop_duplicates().to_list()
        for date, offset in zip(dates, (0., 0., 30., 30.)):
            mask = raw['날짜'] == date
            curve = np.arange(24) + 100. + offset
            raw.loc[mask, qa.POWER_COLUMNS] = np.repeat(curve[:, None], 5, axis=1)
        cutoff = pd.Timestamp('2021-04-04')
        before = load(raw[raw['날짜'] < dates[-1]])
        extended = load(raw)
        history_after = extended[extended.dt < cutoff]
        tr_before = before[before.train_ok_strict]
        tr_after = history_after[history_after.train_ok_strict]
        w_before = qa.historical_clone_weights(before, tr_before)
        w_after = qa.historical_clone_weights(history_after, tr_after)
        np.testing.assert_array_equal(w_before, w_after)
        self.assertTrue(np.all(w_before[:48] == qa.CLONE_W))
        self.assertTrue(np.all(w_before[48:] == 1.))
        # A full-data clone flag changes; an as-of-cutoff weight must not.
        self.assertFalse(before.loc[before['날짜'] == dates[2], 'is_clone'].any())
        self.assertTrue(extended.loc[extended['날짜'] == dates[2], 'is_clone'].all())

    def test_daily_maximum_uses_only_complete_active_days(self):
        rows = []
        for date, count, off, base, error in [
            (20210701, 24, False, 100., 3.),
            (20210702, 23, False, 10000., 500.),
            (20210703, 24, True, 20., 5.),
        ]:
            for hour in range(count):
                target = base + hour
                rows.append({'날짜': date, 'target': target, 'pred': target - error,
                             'is_off': off, 'is_peak': target >= 110})
        result = qa.metrics(pd.DataFrame(rows))
        self.assertEqual(result['n'], 71)
        self.assertEqual(result['active_complete_days'], 1)
        self.assertEqual(result['incomplete_active_days'], 1)
        self.assertEqual(result['daily_max_MAE'], 3.)
        self.assertEqual(result['daily_max_bias'], 3.)

    def test_metric_bias_coverage_and_pinball_include_exact_ties(self):
        frame = pd.DataFrame({
            '날짜': [20210701] * 4, 'target': [0., 10., 10., 10.],
            'pred': [0., 8., 12., 9.], 'is_off': [False] * 4,
            'is_peak': [False, True, True, True],
        })
        result = qa.metrics(frame)
        self.assertEqual(result['MAE'], 1.25)
        self.assertEqual(result['RMSE'], 1.5)
        self.assertEqual(result['bias'], .25)
        self.assertEqual(result['coverage_pct'], 50.)
        self.assertAlmostEqual(result['peak_under_pct'], 200 / 3)
        self.assertAlmostEqual(result['pinball_90'], .725)
        self.assertEqual(result['active_complete_days'], 0)
        self.assertTrue(np.isnan(result['daily_max_MAE']))

    def test_seed_metric_mean_is_distinct_from_ensemble_metric(self):
        va = pd.DataFrame({
            'dt': pd.date_range('2021-07-01', periods=24, freq='h'),
            '날짜': [20210701] * 24, 'hour': range(24),
            'target': [100.] * 24, 'is_off': [False] * 24,
        })
        frames = [qa.prediction_frame(
            va, 'd1', '2021-07', 'L1', seed, np.full(24, prediction),
            pd.Timestamp('2021-07-01'), 95.,
        ) for seed, prediction in [('0', 90.), ('1', 110.), ('ensemble', 100.)]]
        table = qa.result_table(pd.concat(frames, ignore_index=True))
        mean = table[(table.period == qa.CORE) &
                     (table.seed == 'seed_metric_mean')].iloc[0]
        ensemble = table[(table.period == qa.CORE) &
                         (table.seed == 'ensemble')].iloc[0]
        self.assertEqual(mean.MAE, 10.)
        self.assertEqual(mean.RMSE, 10.)
        self.assertEqual(mean.coverage_pct, 50.)
        self.assertEqual(mean.MAE_seed_std, 0.)
        self.assertEqual(ensemble.MAE, 0.)
        self.assertEqual(ensemble.RMSE, 0.)
        self.assertEqual(ensemble.coverage_pct, 100.)


if __name__ == '__main__':
    unittest.main()
