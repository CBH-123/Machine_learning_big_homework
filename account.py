from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import math
import numpy as np
import features as m

@dataclass
class Lot:
    bought_on: date
    units: float


class LongOnlyT1Account:
    """小型教学账户：持仓非负、不借钱；只有更早交易日买入的份额可卖。

    按批次记录买入日，因此已有老持仓时，当天新增的份额不会冻结老持仓。
    这里只检查日线层面的约束，不模拟涨跌停、交易手数或真实 ETF 成交。
    """

    def __init__(self, capital: float, trading_dates: list[date], fee_rate: float = 0):
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("初始资金必须为正数")
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("费率应在 [0, 1) 内")
        if not trading_dates or trading_dates != sorted(set(trading_dates)):
            raise ValueError("交易日历必须严格递增且非空")
        self.cash = capital
        self.fee_rate = fee_rate
        self.calendar = set(trading_dates)
        self.lots: list[Lot] = []
        self.last_action_date: date | None = None

    @property
    def units(self) -> float:
        return sum(lot.units for lot in self.lots)

    def sellable(self, on: date) -> float:
        return sum(lot.units for lot in self.lots if lot.bought_on < on)

    def _check_order(self, on: date, price: float, units: float):
        if on not in self.calendar:
            raise ValueError("该日期不在数据交易日历中")
        if self.last_action_date and on < self.last_action_date:
            raise ValueError("不能倒退到过去成交")
        if not all(math.isfinite(x) and x > 0 for x in (price, units)):
            raise ValueError("成交价和份额必须为正数")

    def buy(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        cost = price * units * (1 + self.fee_rate)
        if cost > self.cash + 1e-8:
            raise ValueError("现金不足，禁止借钱买入")
        self.cash = max(0.0, self.cash - cost)
        self.lots.append(Lot(on, units))
        self.last_action_date = on

    def sell(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        if units > self.units + 1e-8:
            raise ValueError("卖出超过总持仓，禁止做空")
        if units > self.sellable(on) + 1e-8:
            raise ValueError("违反 T+1：当天买入的份额不能当天卖出")
        remaining = units
        for lot in self.lots:
            if lot.bought_on < on:
                taken = min(lot.units, remaining)
                lot.units -= taken
                remaining -= taken
        self.lots = [lot for lot in self.lots if lot.units > 1e-10]
        self.cash += price * units * (1 - self.fee_rate)
        self.last_action_date = on

def simulate(rows, forecasts, capital=100_000):
    fee_rate = m.FEE_RATE
    weights = np.array([float(rec["strategy_weight"]) for rec in forecasts])
    if len(weights) != len(forecasts) or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1+1e-12):
        raise ValueError("目标仓位必须是0到1之间的有限数")
    weights = np.clip(weights, 0, 1)
    by_date = {str(r["date"]): r for r in rows}
    days = [by_date[f["target_date"]] for f in forecasts]
    account = LongOnlyT1Account(capital, [r["date"] for r in days], fee_rate)
    curve = [{"timestamp": f'{days[0]["date"]} 09:29:00', "equity": capital,
              "cash": capital, "units": 0, "target_weight": 0, "open_weight_after_trade": 0,
              "close_weight": 0, "daily_return": 0, "drawdown": 0}]
    trades = []
    peak = previous = capital
    for i, (row, f, weight) in enumerate(zip(days, forecasts, weights)):
        on, price = row["date"], row["open"]
        terminal = i == len(days) - 1
        if not f["asof_date"] < str(on):
            raise ValueError("交易信号不能使用当天完整行情")
        if f.get("max_training_label_date") and not f["max_training_label_date"] <= f["fit_asof"] <= f["asof_date"]:
            raise ValueError("训练答案越过了预测时点")
        difference = (account.cash+account.units*price)*float(weight)/price-account.units
        if i and weight == weights[i-1]:
            difference = 0
        elif difference > 0:
            difference /= 1 + float(weight) * fee_rate
        elif difference < 0:
            difference /= 1 - float(weight) * fee_rate

        def record(side, q, p, clock, signal):
            trades.append({"date": str(on), "time": clock, "side": side, "units": q,
                           "price": p, "fee": q*p*fee_rate, "signal_asof": signal,
                           "provenance": "causal"})

        signal = f["asof_date"]
        if difference > 1e-8 and not terminal:
            q = min(difference, account.cash/(price*(1+fee_rate)))
            if q > 1e-8:
                account.buy(on, price, q)
                record("BUY", q, price, "09:30", signal)
        elif difference < -1e-8:
            q = min(-difference, account.sellable(on))
            if q > 1e-8:
                account.sell(on, price, q)
                record("SELL", q, price, "09:30", signal)
        open_equity = account.cash+account.units*price
        open_weight = account.units*price/open_equity
        if terminal and account.units > 1e-8:
            q = account.units
            account.sell(on, row["close"], q)
            record("SELL", q, row["close"], "15:00", "PRECOMMITTED_END")
        equity = account.cash+account.units*row["close"]
        peak = max(peak, equity)
        curve.append({"timestamp": f"{on} 15:00:00", "equity": equity,
                      "cash": account.cash, "units": account.units, "target_weight": float(weight),
                      "open_weight_after_trade": open_weight,
                      "close_weight": account.units*row["close"]/equity,
                      "daily_return": equity/previous-1, "drawdown": equity/peak-1})
        previous = equity
    daily = np.array([r["daily_return"] for r in curve[1:]])
    std = np.std(daily, ddof=1) if len(daily) > 1 else 0
    metrics = {"name": "breadth_half", "provenance": "causal", "trading_days": len(days),
               "fee_rate_each_side": fee_rate, "final_equity": curve[-1]["equity"],
               "total_return": curve[-1]["equity"]/capital-1,
               "annualized_return": (curve[-1]["equity"]/capital)**(252/len(days))-1,
               "max_drawdown": -min(r["drawdown"] for r in curve),
               "sharpe_rf_zero": float(np.mean(daily)/std*np.sqrt(252)) if std > 1e-12 else 0.0,
               "average_open_weight": float(np.mean([r["open_weight_after_trade"] for r in curve[1:]])),
               "average_close_weight": float(np.mean([r["close_weight"] for r in curve[1:]])),
               "order_count": len(trades), "fees_paid": sum(t["fee"] for t in trades)}
    return {"curve": curve, "trades": trades, "metrics": metrics}

def verify_consensus_ledger(rows, decisions, run, liquidate_at_end=True):
    """Rebuild shares and cash from order records, independently of the account."""
    def check(condition, msg):
        if not condition:
            raise ValueError(f"Consensus replay audit: {msg}")

    def equal(a, b):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)

    prices = {str(row["date"]): row for row in rows}
    marks, trades, metrics = run["curve"], run["trades"], run["metrics"]
    check(decisions and len(marks) == len(decisions) + 1, "incorrect daily marks")
    fee_rate = metrics["fee_rate_each_side"]
    check(0 <= fee_rate < 1, "invalid fee rate")
    initial = marks[0]
    check(initial["timestamp"] == f'{decisions[0]["target_date"]} 09:29:00'
          and initial["cash"] > 0 and equal(initial["cash"], initial["equity"])
          and equal(initial["units"], 0), "account must start in cash")
    cash, units, fees = initial["cash"], 0.0, 0.0
    peak = previous_close = cash
    max_drawdown = previous_target = 0.0
    fills = defaultdict(list)
    previous_key = None
    for fill in trades:
        key = (fill["date"], fill["time"])
        check(previous_key is None or previous_key < key, "orders must follow the trading clock")
        fills[fill["date"]].append(fill)
        previous_key = key

    for index, (decision, mark) in enumerate(zip(decisions, marks[1:], strict=True)):
        day, target = decision["target_date"], decision["strategy_weight"]
        row = prices[day]
        open_price, close_price = row["open"], row["close"]
        final_day = liquidate_at_end and index == len(decisions) - 1
        day_fills = {fill["time"]: fill for fill in fills.pop(day, [])}
        check(set(day_fills).issubset({"09:30", "15:00"}), f"{day}: invalid order time")
        # At the open, all existing shares were bought before today. A buy today
        # raises total shares but does not raise today's sellable balance.
        sellable = units
        opening_weight_before = units * open_price / (cash + units * open_price)

        def apply_fill(fill, price):
            nonlocal cash, units, fees, sellable
            quantity, fee = fill["units"], fill["fee"]
            check(fill["provenance"] == "causal" and equal(fill["price"], price)
                  and quantity > 0 and equal(fee, quantity * price * fee_rate),
                  f"{day}: invalid execution or fee")
            if fill["side"] == "BUY":
                cash -= quantity * price + fee
                units += quantity
            else:
                check(fill["side"] == "SELL" and quantity <= sellable + 1e-8,
                      f"{day}: sale exceeds T+1 inventory")
                cash += quantity * price - fee
                units -= quantity
                sellable -= quantity
            fees += fee
            check(cash >= -1e-8 and units >= -1e-8, f"{day}: negative cash or shares")

        if "09:30" in day_fills:
            fill = day_fills["09:30"]
            check(target != previous_target and fill["signal_asof"] == decision["asof_date"] < day
                  and (not final_day or fill["side"] == "SELL"),
                  f"{day}: opening order violates the decision clock or target rule")
            apply_fill(fill, open_price)
        opening_weight = units * open_price / (cash + units * open_price)
        if target != previous_target and (not final_day or target <= opening_weight_before):
            check(equal(opening_weight, target), f"{day}: missed target on the transition")
        check(equal(mark["target_weight"], target)
              and equal(mark["open_weight_after_trade"], opening_weight),
              f"{day}: opening weight or target does not match ledger")

        if "15:00" in day_fills:
            fill = day_fills["15:00"]
            check(final_day and fill["side"] == "SELL" and equal(fill["units"], units)
                  and fill["signal_asof"] == "PRECOMMITTED_END", f"{day}: invalid liquidation")
            apply_fill(fill, close_price)
        check(not final_day or equal(units, 0), f"{day}: shares remain after final close")
        equity = cash + units * close_price
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        max_drawdown = max(-drawdown, max_drawdown)
        check(mark["timestamp"] == f"{day} 15:00:00" and equal(mark["equity"], equity)
              and equal(mark["cash"], cash) and equal(mark["units"], units)
              and equal(mark["close_weight"], units * close_price / equity)
              and equal(mark["daily_return"], equity / previous_close - 1)
              and equal(mark["drawdown"], drawdown), f"{day}: closing marks do not reconcile")
        previous_target, previous_close = target, equity

    check(not fills and len(trades) == metrics["order_count"]
          and equal(fees, metrics["fees_paid"]) and equal(previous_close, metrics["final_equity"])
          and equal(previous_close / initial["cash"] - 1, metrics["total_return"])
          and equal(max_drawdown, metrics["max_drawdown"]), "account totals do not reconcile")
