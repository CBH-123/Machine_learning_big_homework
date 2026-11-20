import copy
import math
import unittest
from datetime import date, timedelta

import numpy as np

from features import build_features


class BuildFeaturesCausalityTest(unittest.TestCase):
    def test_future_market_changes_do_not_affect_past_features(self):
        rows = []
        for i in range(75):
            close = 100.0 + 0.3 * i + math.sin(i)
            open_ = close * (1.0 + 0.001 * math.cos(i))
            volume = 1000.0 + 17.0 * i
            rows.append({
                "date": date(2025, 1, 1) + timedelta(days=i),
                "open": open_,
                "high": max(open_, close) + 1.0,
                "low": min(open_, close) - 1.0,
                "close": close,
                "volume": volume,
                "amount": volume * close,
            })

        cutoff = 65
        original = build_features(rows)
        changed_rows = copy.deepcopy(rows)
        for row in changed_rows[cutoff + 1:]:
            for field in ("open", "high", "low", "close", "volume", "amount"):
                row[field] = row[field] * 7.0 + 123.0
        changed = build_features(changed_rows)

        self.assertTrue(np.isfinite(original[cutoff]).all())
        np.testing.assert_allclose(
            original[:cutoff + 1],
            changed[:cutoff + 1],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
