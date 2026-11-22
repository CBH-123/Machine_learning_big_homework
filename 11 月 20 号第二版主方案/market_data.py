"""读取教师提供的行情，定义账户和交易记录。"""
from __future__ import annotations
import datetime as dates_module
from pathlib import Path
import datetime


from pathlib import Path

import csv

import re

from datetime import datetime

import math

import hashlib

from datetime import date

from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent / '.'

START = dates_module.date(2025, 10, 10)

FIELDS = ('date', 'open', 'high', 'low', 'close', 'volume', 'amount')

END = dates_module.date(2026, 10, 23)

def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_prices(path: Path) -> tuple[list[dict], dict]:
    """跳过通达信说明行；逐条校验，不静默填补或删除异常行情。"""
    raw = path.read_bytes()
    rows = []
    ignored = []
    for number, line in enumerate(raw.decode('gb18030').splitlines(), 1):
        if not re.match('^\\d{4}/\\d{2}/\\d{2}', line.strip()):
            if line.strip():
                ignored.append(number)
            continue
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f'第 {number} 行应有 7 列')
        row = dict(zip(FIELDS, [datetime.strptime(parts[0], '%Y/%m/%d').date(), *map(float, parts[1:])]))
        values = [row[k] for k in FIELDS[1:]]
        if not all((math.isfinite(x) for x in values)):
            raise ValueError(f'第 {number} 行存在缺失值或无穷值')
        if min(values[:4]) <= 0 or min(values[4:]) < 0:
            raise ValueError(f'第 {number} 行价格或成交数据无效')
        if not row['low'] <= min(row['open'], row['close']) <= max(row['open'], row['close']) <= row['high']:
            raise ValueError(f'第 {number} 行开高低收不一致')
        rows.append(row)
    if not rows:
        raise ValueError('没有读到行情')
    if len({r['date'] for r in rows}) != len(rows):
        raise ValueError('存在重复日期')
    if any((a['date'] >= b['date'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('日期未严格递增')
    audit = {'source': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'encoding': 'gb18030', 'raw_rows': len(rows), 'first_date': rows[0]['date'].isoformat(), 'last_date': rows[-1]['date'].isoformat(), 'ignored_non_price_line_numbers': ignored, 'duplicate_dates': 0, 'missing_or_nonfinite_values': 0, 'invalid_ohlc_rows': 0}
    return (rows, audit)

class LongOnlyT1Account:
    """小型教学账户：持仓非负、不借钱；只有更早交易日买入的份额可卖。

    按批次记录买入日，因此已有老持仓时，当天新增的份额不会冻结老持仓。
    这里只检查日线层面的约束，不模拟涨跌停、交易手数或真实 ETF 成交。
    """

    def __init__(self, capital: float, trading_dates: list[date], fee_rate: float=0):
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError('初始资金必须为正数')
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError('费率应在 [0, 1) 内')
        if not trading_dates or trading_dates != sorted(set(trading_dates)):
            raise ValueError('交易日历必须严格递增且非空')
        self.cash = capital
        self.fee_rate = fee_rate
        self.calendar = set(trading_dates)
        self.lots: list[Lot] = []
        self.last_action_date: date | None = None

    @property
    def units(self) -> float:
        return sum((lot.units for lot in self.lots))

    def sellable(self, on: date) -> float:
        return sum((lot.units for lot in self.lots if lot.bought_on < on))

    def _check_order(self, on: date, price: float, units: float):
        if on not in self.calendar:
            raise ValueError('该日期不在数据交易日历中')
        if self.last_action_date and on < self.last_action_date:
            raise ValueError('不能倒退到过去成交')
        if not all((math.isfinite(x) and x > 0 for x in (price, units))):
            raise ValueError('成交价和份额必须为正数')

    def buy(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        cost = price * units * (1 + self.fee_rate)
        if cost > self.cash + 1e-08:
            raise ValueError('现金不足，禁止借钱买入')
        self.cash = max(0.0, self.cash - cost)
        self.lots.append(Lot(on, units))
        self.last_action_date = on

    def sell(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        if units > self.units + 1e-08:
            raise ValueError('卖出超过总持仓，禁止做空')
        if units > self.sellable(on) + 1e-08:
            raise ValueError('违反 T+1：当天买入的份额不能当天卖出')
        remaining = units
        for lot in self.lots:
            if lot.bought_on < on:
                taken = min(lot.units, remaining)
                lot.units -= taken
                remaining -= taken
        self.lots = [lot for lot in self.lots if lot.units > 1e-10]
        self.cash += price * units * (1 - self.fee_rate)
        self.last_action_date = on

@dataclass
class Lot:
    bought_on: date
    units: float
