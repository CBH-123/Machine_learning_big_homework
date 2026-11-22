"""先按各自的规则模拟，再按已完成的近期净收益决定下个开盘跟随哪个策略。"""
from collections import deque
from math import isfinite, log
from statistics import fmean

import step7_net_return as s7
from step2_models import LongOnlyT1Account


TRAILING_SESSIONS = 60
TREND_WINDOW = 20
INITIAL_CASH = 100_000
PLAN = {
    "experts": {
        "tcn": "existing TCN buffer rule",
        "trend": "long when the latest close exceeds the mean of the latest 20 closes",
    },
    "trailing_sessions": TRAILING_SESSIONS,
    "score": "net log equity return over the latest 60 fully observed sessions",
    "shadow_fee_each_side": s7.FEE,
    "tie_and_warmup": "follow TCN",
    "shadow_exit": "follow each shadow strategy's own rules; no liquidation at scoring or evaluation boundaries",
    "execution": "choose after today's close and follow the selected target weight at the next open",
}


def _mark_shadow(account, row, target):
    """只结算传入交易日的既定开盘仓位，并给出当日收盘净值。"""
    day, opening, closing = row["date"], float(row["open"]), float(row["close"])
    if not all(isfinite(price) and price > 0 for price in (opening, closing)):
        raise ValueError(f"Invalid shadow prices on {day}")
    if target == 1 and account.units == 0:
        account.buy(day, opening, account.cash / (opening * (1 + account.fee_rate)))
    elif target == 0 and account.units > 0:
        account.sell(day, opening, account.units)
    return account.cash + account.units * closing


def adaptive_decisions(rows, forecasts, trade_from=None):
    """返回实际账户的目标仓位。更早预测只延续两个模拟账户和滚动评分。"""
    if not forecasts:
        return []
    trade_from = str(trade_from) if trade_from is not None else ""
    dates = [row["date"] for row in rows]
    indices = {str(day): i for i, day in enumerate(dates)}
    shadows = {name: LongOnlyT1Account(INITIAL_CASH, dates, s7.FEE) for name in PLAN["experts"]}
    equity = {name: deque([INITIAL_CASH], maxlen=TRAILING_SESSIONS + 1) for name in shadows}
    previous_targets = None
    previous_execution = None
    decisions = []
    for forecast in s7.weighted_forecasts(forecasts, "buffer"):
        asof, target = forecast["asof_date"], forecast["target_date"]
        i = indices.get(asof)
        if (i is None or asof >= target or (i + 1 < len(dates) and str(dates[i + 1]) != target)
                or (previous_execution is not None and previous_execution != asof)):
            raise ValueError(f"Expected consecutive signal and execution dates: {asof}, {target}")
        if not isfinite(float(forecast["predicted_return"])):
            raise ValueError(f"Non-finite TCN forecast on {asof}")
        if previous_targets is not None:
            for name, account in shadows.items():
                equity[name].append(_mark_shadow(account, rows[i], previous_targets[name]))
        observed = len(equity["tcn"]) - 1
        scores = {name: log(values[-1] / values[0]) if observed == TRAILING_SESSIONS else None
                  for name, values in equity.items()}
        selected = "tcn"
        if observed == TRAILING_SESSIONS and scores["trend"] > scores["tcn"]:
            selected = "trend"
        targets = {"tcn": forecast["strategy_weight"], "trend": 0.0}
        if i + 1 >= TREND_WINDOW:
            mean = fmean(float(row["close"]) for row in rows[i - TREND_WINDOW + 1:i + 1])
            targets["trend"] = float(float(rows[i]["close"]) > mean)
        if target >= trade_from:
            decisions.append({**forecast, "strategy_weight": targets[selected], "selected_expert": selected,
                              "tcn_target_weight": targets["tcn"], "trend_target_weight": targets["trend"],
                              "expert_sessions_observed": observed, "tcn_trailing_log_return": scores["tcn"],
                              "trend_trailing_log_return": scores["trend"]})
        previous_targets = targets
        previous_execution = target
    return decisions
