from __future__ import annotations
import csv
import hashlib
import math
import re
from datetime import date, datetime
from pathlib import Path
import numpy as np

START = date(2025, 10, 10)
END = date(2026, 10, 23)
FIELDS = ("date", "open", "high", "low", "close", "volume", "amount")


def read_prices(path: Path) -> tuple[list[dict], dict]:
    """跳过通达信说明行；逐条校验，不静默填补或删除异常行情。"""
    raw = path.read_bytes()
    rows = []
    ignored = []
    for number, line in enumerate(raw.decode("gb18030").splitlines(), 1):
        if not re.match(r"^\d{4}/\d{2}/\d{2}", line.strip()):
            if line.strip():
                ignored.append(number)
            continue
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"第 {number} 行应有 7 列")
        row = dict(zip(FIELDS, [datetime.strptime(parts[0], "%Y/%m/%d").date(),
                                *map(float, parts[1:])]))
        values = [row[k] for k in FIELDS[1:]]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"第 {number} 行存在缺失值或无穷值")
        if min(values[:4]) <= 0 or min(values[4:]) < 0:
            raise ValueError(f"第 {number} 行价格或成交数据无效")
        if not (row["low"] <= min(row["open"], row["close"])
                <= max(row["open"], row["close"]) <= row["high"]):
            raise ValueError(f"第 {number} 行开高低收不一致")
        rows.append(row)
    if not rows:
        raise ValueError("没有读到行情")
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("存在重复日期")
    if any(a["date"] >= b["date"] for a, b in zip(rows, rows[1:])):
        raise ValueError("日期未严格递增")
    audit = {
        "source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "gb18030", "raw_rows": len(rows),
        "first_date": rows[0]["date"].isoformat(),
        "last_date": rows[-1]["date"].isoformat(),
        "ignored_non_price_line_numbers": ignored,
        "duplicate_dates": 0, "missing_or_nonfinite_values": 0,
        "invalid_ohlc_rows": 0,
    }
    return rows, audit

def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

TRAIN_WINDOW = 1260
REFIT_EVERY = 20
FEE_RATE = 0.0005  # 单边综合成本 0.05%，教学假设，不是实际市场费率。
FEATURE_INFO = [
    *[(f"return_{w}", f"过去{w}个交易日涨跌幅", f"close[t]/close[t-{w}]-1") for w in (1, 2, 5, 10, 20)],
    *[(f"lag_return_{k}", f"向前{k}天的单日涨跌幅", f"close[t-{k}]/close[t-{k+1}]-1") for k in (1, 2, 3)],
    *[(f"ma_gap_{w}", f"收盘价相对{w}日均价的偏离", f"close[t]/mean(close[t-{w-1}:t])-1") for w in (5, 10, 20, 60)],
    *[(f"volatility_{w}", f"过去{w}日收益波动率", f"std(last {w} daily returns, ddof=1)") for w in (5, 10, 20)],
    ("candle_body", "当天开盘至收盘的涨跌", "(close-open)/open"),
    ("candle_range", "当天最高与最低的振幅", "(high-low)/close"),
    ("upper_shadow", "当天上影线相对长度", "(high-max(open,close))/close"),
    ("lower_shadow", "当天下影线相对长度", "(min(open,close)-low)/close"),
    *[(f"volume_ratio_{w}", f"成交量相对{w}日均值", f"volume[t]/mean(last {w} volumes)-1") for w in (5, 20)],
    ("log_volume_change", "成交量对数变化", "log((volume[t]+1)/(volume[t-1]+1))"),
    ("amount_ratio_20", "成交额相对20日均值", "amount[t]/mean(last 20 amounts)-1"),
    ("log_amount_change", "成交额对数变化", "log((amount[t]+1)/(amount[t-1]+1))"),
    ("price_position_20", "收盘价在近期高低区间的位置", "(close-min(last 20 lows))/(max(last 20 highs)-min(last 20 lows))"),
]
FEATURE_NAMES = [row[0] for row in FEATURE_INFO]


def build_features(rows: list[dict]) -> np.ndarray:
    """第 i 行仅访问 rows[:i+1]；保留前60行 NaN，绝不用未来数据回填。"""
    n = len(rows)
    x = np.full((n, len(FEATURE_NAMES)), np.nan)
    arrays = {k: np.array([r[k] for r in rows], dtype=float)
              for k in ("open", "high", "low", "close", "volume", "amount")}
    c, o, h, low, v, a = [arrays[k] for k in ("close", "open", "high", "low", "volume", "amount")]
    ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
    for i in range(60, n):
        f = [c[i] / c[i-w] - 1 for w in (1, 2, 5, 10, 20)]
        f += [ret[i-k] for k in (1, 2, 3)]
        f += [c[i] / np.mean(c[i-w+1:i+1]) - 1 for w in (5, 10, 20, 60)]
        f += [np.std(ret[i-w+1:i+1], ddof=1) for w in (5, 10, 20)]
        f += [(c[i]-o[i])/o[i], (h[i]-low[i])/c[i],
              (h[i]-max(o[i], c[i]))/c[i], (min(o[i], c[i])-low[i])/c[i]]
        f += [v[i]/max(np.mean(v[i-w+1:i+1]), 1e-12)-1 for w in (5, 20)]
        f += [np.log((v[i]+1)/(v[i-1]+1)),
              a[i]/max(np.mean(a[i-19:i+1]), 1e-12)-1,
              np.log((a[i]+1)/(a[i-1]+1))]
        lo, hi = np.min(low[i-19:i+1]), np.max(h[i-19:i+1])
        f += [(c[i]-lo)/max(hi-lo, 1e-12)]
        x[i] = f
    return x
