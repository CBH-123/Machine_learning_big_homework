"""Combine two past-only position signals in one long-only account."""

from collections import defaultdict
from math import isclose


PLAN = {
    "sources": ["original TCN with its buffer rule", "Step 13 turnover utility"],
    "target_weights": {"both_hold": 1.0, "one_holds": 0.5, "neither_holds": 0.0},
    "execution": "single account; adjust to target only when it changes; net change at next open",
    "rebalance_on_weight_change_only": True,
    "uses_calendar_features": False,
}


def _binary_weight(decision, asof, name):
    if decision.get("provenance", "causal") != "causal":
        raise ValueError(f"{name} must contain causal decisions")
    fit_asof = decision.get("fit_asof")
    max_label = decision.get("max_training_label_date")
    if ((fit_asof and str(fit_asof) > asof)
            or (max_label and (not fit_asof or str(max_label) > str(fit_asof)))):
        raise ValueError(f"{name} training information exceeds its as-of date")
    try:
        weight = float(decision["strategy_weight"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires binary target weights") from exc
    if weight not in (0.0, 1.0):
        raise ValueError(f"{name} requires binary target weights")
    return weight


def consensus_decisions(tcn_weights, utility):
    """Match the binary signals by target and as-of; retain source weights for audit."""
    combined = []
    previous_target = None
    for tcn, util in zip(tcn_weights, utility, strict=True):
        asof, target = str(tcn["asof_date"]), str(tcn["target_date"])
        if ((asof, target) != (str(util["asof_date"]), str(util["target_date"]))
                or asof >= target or (previous_target is not None and target <= previous_target)):
            raise ValueError("Consensus requires aligned, strictly ordered, prior-session signals")
        tcn_weight = _binary_weight(tcn, asof, "TCN")
        utility_weight = _binary_weight(util, asof, "Utility")
        fit_dates = [str(d["fit_asof"]) for d in (tcn, util) if d.get("fit_asof")]
        label_dates = [str(d["max_training_label_date"]) for d in (tcn, util)
                       if d.get("max_training_label_date")]
        combined.append({
            "model": "consensus",
            "asof_date": asof,
            "target_date": target,
            "strategy_weight": (tcn_weight + utility_weight) / 2,
            "tcn_holds": bool(tcn_weight),
            "utility_holds": bool(utility_weight),
            "fit_asof": max(fit_dates, default=""),
            "max_training_label_date": max(label_dates, default=""),
        })
        previous_target = target
    if not combined:
        raise ValueError("Consensus requires at least one paired trading session")
    return combined


def verify_consensus_ledger(rows, decisions, run, liquidate_at_end=True):
    """Rebuild shares and cash from order records, independently of the account."""
    def check(condition, msg):
        if not condition:
            raise ValueError(f"Consensus replay audit: {msg}")

    def equal(a, b):
        return isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)

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
