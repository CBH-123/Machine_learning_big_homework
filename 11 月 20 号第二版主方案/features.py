"""仅用截至信号日的行情构建特征。"""
from __future__ import annotations
import datetime as dates_module
from pathlib import Path
import datetime


import numpy as np

FEE_RATE = 0.0005

VALID_START = dates_module.date(2023, 1, 1)

REFIT_EVERY = 20

TRAIN_WINDOW = 1260

FEATURE_NAMES = ['return_1', 'return_2', 'return_5', 'return_10', 'return_20', 'lag_return_1', 'lag_return_2', 'lag_return_3', 'ma_gap_5', 'ma_gap_10', 'ma_gap_20', 'ma_gap_60', 'volatility_5', 'volatility_10', 'volatility_20', 'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow', 'volume_ratio_5', 'volume_ratio_20', 'log_volume_change', 'amount_ratio_20', 'log_amount_change', 'price_position_20']

def build_features(rows: list[dict]) -> np.ndarray:
    """第 i 行仅访问 rows[:i+1]；保留前60行 NaN，绝不用未来数据回填。"""
    n = len(rows)
    x = np.full((n, len(FEATURE_NAMES)), np.nan)
    arrays = {k: np.array([r[k] for r in rows], dtype=float) for k in ('open', 'high', 'low', 'close', 'volume', 'amount')}
    c, o, h, low, v, a = [arrays[k] for k in ('close', 'open', 'high', 'low', 'volume', 'amount')]
    ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
    for i in range(60, n):
        f = [c[i] / c[i - w] - 1 for w in (1, 2, 5, 10, 20)]
        f += [ret[i - k] for k in (1, 2, 3)]
        f += [c[i] / np.mean(c[i - w + 1:i + 1]) - 1 for w in (5, 10, 20, 60)]
        f += [np.std(ret[i - w + 1:i + 1], ddof=1) for w in (5, 10, 20)]
        f += [(c[i] - o[i]) / o[i], (h[i] - low[i]) / c[i], (h[i] - max(o[i], c[i])) / c[i], (min(o[i], c[i]) - low[i]) / c[i]]
        f += [v[i] / max(np.mean(v[i - w + 1:i + 1]), 1e-12) - 1 for w in (5, 20)]
        f += [np.log((v[i] + 1) / (v[i - 1] + 1)), a[i] / max(np.mean(a[i - 19:i + 1]), 1e-12) - 1, np.log((a[i] + 1) / (a[i - 1] + 1))]
        lo, hi = (np.min(low[i - 19:i + 1]), np.max(h[i - 19:i + 1]))
        f += [(c[i] - lo) / max(hi - lo, 1e-12)]
        x[i] = f
    return x
