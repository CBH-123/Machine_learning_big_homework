"""Attribute the relative growth of two aligned daily backtest ledgers."""

from datetime import date
from math import expm1, fsum, isfinite, log1p


def _daily_logs(curve):
    """Index ledger rows by timestamp without silently dropping observations."""
    logs = {}
    for row in curve:
        timestamp = str(row["timestamp"]).strip().replace("T", " ", 1)
        if timestamp in logs:
            raise ValueError(f"Duplicate ledger timestamp: {timestamp}")
        try:
            trading_day = date.fromisoformat(timestamp[:10])
        except ValueError as error:
            raise ValueError(f"Expected an ISO ledger date: {timestamp}") from error
        daily_return = float(row["daily_return"])
        if not isfinite(daily_return) or daily_return <= -1.0:
            raise ValueError(f"Invalid daily return at {timestamp}: {daily_return}")
        logs[timestamp] = (trading_day.isoformat()[:7], log1p(daily_return))
    return logs


def month_audit(candidate_curve, baseline_curve):
    """Compound daily returns by month and remove one month's factors at a time.

    This is return attribution; it does not retrain the models or reset positions.
    Monthly excess log returns add to the full-sample relative log growth.
    """
    candidate = _daily_logs(candidate_curve)
    baseline = _daily_logs(baseline_curve)
    if not candidate or candidate.keys() != baseline.keys():
        raise ValueError("Return ledgers must have the same nonempty timestamp set")

    monthly_logs = {}
    for timestamp in sorted(candidate):
        month, candidate_log = candidate[timestamp]
        _, baseline_log = baseline[timestamp]
        bucket = monthly_logs.setdefault(month, ([], [], []))
        bucket[0].append(candidate_log)
        bucket[1].append(baseline_log)
        bucket[2].append(candidate_log - baseline_log)

    months = []
    for month, (candidate_logs, baseline_logs, relative_logs) in monthly_logs.items():
        months.append({
            "month": month,
            "candidate_return": expm1(fsum(candidate_logs)),
            "baseline_return": expm1(fsum(baseline_logs)),
            "relative_log_growth": fsum(relative_logs),
        })

    total_relative_log_growth = fsum(row["relative_log_growth"] for row in months)
    for row in months:
        remaining_log_growth = fsum(
            other["relative_log_growth"] for other in months if other is not row
        )
        row["relative_equity_advantage_excluding_month"] = expm1(remaining_log_growth)

    leave_one_month_out = sorted(
        months, key=lambda row: (row["relative_equity_advantage_excluding_month"], row["month"])
    )
    worst = leave_one_month_out[0]
    positive_logs = [row["relative_log_growth"] for row in months if row["relative_log_growth"] > 0]

    return {
        "month_count": len(months),
        "months": months,
        "leave_one_month_out": leave_one_month_out,
        "total_relative_log_growth": total_relative_log_growth,
        "total_relative_equity_advantage": expm1(total_relative_log_growth),
        "worst_leave_one_month": worst["month"],
        "worst_leave_one_month_relative_equity_advantage": worst[
            "relative_equity_advantage_excluding_month"
        ],
        "largest_positive_month_concentration": max(positive_logs) / fsum(positive_logs)
        if positive_logs else 0.0,
        "months_beating_baseline": len(positive_logs),
        "positive_leave_one_month_out_count": sum(
            row["relative_equity_advantage_excluding_month"] > 0 for row in months
        ),
    }
