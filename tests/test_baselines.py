import copy
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from baselines import prediction_metrics, rolling_ridge, zero_predictions
from features import START, read_prices


class RollingRidgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = read_prices(Path(__file__).parents[1] / "data" / "SZ#399006.txt")

    def test_time_boundaries_and_future_independence(self):
        original = rolling_ridge(self.rows, START, START)
        self.assertEqual(len(original), 1)
        prediction = original[0]
        self.assertEqual(prediction["target_date"], str(START))
        self.assertEqual(prediction["n_train"], 1260)
        self.assertLessEqual(prediction["max_training_label_date"], prediction["fit_asof"])
        self.assertLessEqual(prediction["fit_asof"], prediction["asof_date"])
        self.assertTrue(np.isfinite(prediction["predicted_return"]))

        changed_rows = copy.deepcopy(self.rows)
        for row in changed_rows:
            if str(row["date"]) > prediction["asof_date"]:
                for key in ("open", "high", "low", "close", "volume", "amount"):
                    row[key] *= 7.0
        changed = rolling_ridge(changed_rows, START, START)
        self.assertAlmostEqual(prediction["predicted_return"], changed[0]["predicted_return"], places=14)


class PredictionMetricsTest(unittest.TestCase):
    def test_zero_return_baseline(self):
        rows = [
            {"date": date(2025, 1, 2), "close": 100.0},
            {"date": date(2025, 1, 3), "close": 110.0},
            {"date": date(2025, 1, 6), "close": 99.0},
        ]
        signals = [
            {"model": "any", "asof_date": "2025-01-02", "target_date": "2025-01-03"},
            {"model": "any", "asof_date": "2025-01-03", "target_date": "2025-01-06"},
        ]
        forecasts = zero_predictions(rows, signals)
        self.assertEqual([rec["predicted_close"] for rec in forecasts], [100.0, 110.0])
        metrics = prediction_metrics(rows, forecasts)
        self.assertEqual(metrics["model"], "zero_return")
        self.assertEqual(metrics["n_samples"], 2)
        self.assertAlmostEqual(metrics["return_mae"], 0.1)
        self.assertAlmostEqual(metrics["price_mae_points"], 10.5)
        self.assertEqual(metrics["direction_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
