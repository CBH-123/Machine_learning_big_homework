"""与 TCN 使用相同时间边界的零预测和 Ridge 回归基线。"""
from math import sqrt
from statistics import fmean

import numpy as np

import features as m

RIDGE_ALPHA = 1.0


def zero_predictions(rows, forecasts):
    """在相同时点上预测零收益，即预测下一收盘价等于当前收盘价。"""
    prices = {str(row["date"]): row for row in rows}
    return [{
        "model": "zero_return",
        "asof_date": rec["asof_date"],
        "target_date": rec["target_date"],
        "close_asof": prices[rec["asof_date"]]["close"],
        "predicted_return": 0.0,
        "predicted_close": prices[rec["asof_date"]]["close"],
    } for rec in forecasts]


def rolling_ridge(rows, first, last, alpha=RIDGE_ALPHA):
    """滚动训练带未惩罚截距的 Ridge，标签与 TCN 一样为次日收盘收益。"""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha 必须为正数")
    x = m.build_features(rows)
    close = np.array([r["close"] for r in rows])
    returns = close[1:] / close[:-1] - 1
    eligible = np.flatnonzero(np.isfinite(x).all(axis=1))
    signals = [i for i in eligible if i + 1 < len(rows) and first <= rows[i + 1]["date"] <= last]
    forecasts = []
    fit_id = 0
    for offset, i in enumerate(signals):
        asof = rows[i]["date"]
        if offset % m.REFIT_EVERY == 0:
            train = eligible[eligible < i][-m.TRAIN_WINDOW:]
            if len(train) != m.TRAIN_WINDOW:
                raise ValueError(f"{asof} has only {len(train)} mature rows")
            mean = x[train].mean(axis=0)
            scale = np.maximum(x[train].std(axis=0), 1e-8)
            train_x = np.clip((x[train] - mean) / scale, -6, 6)
            design = np.column_stack((np.ones(len(train)), train_x))
            train_y = returns[train] * 100
            penalty = np.eye(design.shape[1]) * alpha
            penalty[0, 0] = 0.0
            coefs = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
            fit_id += 1
            fit_asof = str(asof)
            max_training_label_date = str(rows[train[-1] + 1]["date"])
            if max_training_label_date > fit_asof:
                raise ValueError("Ridge 训练标签越过预测时点")
        scaled = np.clip((x[i] - mean) / scale, -6, 6)
        prediction = float(np.r_[1.0, scaled] @ coefs / 100)
        if not np.isfinite(prediction):
            raise ValueError(f"Nonfinite Ridge prediction for {asof}")
        forecasts.append({
            "model": "ridge",
            "asof_date": str(asof),
            "target_date": str(rows[i + 1]["date"]),
            "close_asof": float(close[i]),
            "predicted_return": prediction,
            "predicted_close": float(close[i] * (1 + prediction)),
            "fit_id": fit_id,
            "fit_asof": fit_asof,
            "max_training_label_date": max_training_label_date,
            "n_train": m.TRAIN_WINDOW,
        })
    if not forecasts:
        raise ValueError("No usable signal dates")
    return forecasts


def prediction_metrics(rows, forecasts, model=None):
    """同时报告收益和点位误差；方向口径与 run.py 保持一致。"""
    if not forecasts:
        raise ValueError("预测记录不能为空")
    prices = {str(row["date"]): row for row in rows}
    return_errors, price_errors, directions = [], [], []
    for rec in forecasts:
        asof, target = prices[rec["asof_date"]], prices[rec["target_date"]]
        actual_return = target["close"] / asof["close"] - 1
        return_errors.append(rec["predicted_return"] - actual_return)
        price_errors.append(rec["predicted_close"] - target["close"])
        directions.append((rec["predicted_return"] > 0) == (actual_return > 0))
    return {
        "model": model or forecasts[0].get("model", "unknown"),
        "n_samples": len(forecasts),
        "direction_accuracy": fmean(directions),
        "return_mae": fmean(abs(error) for error in return_errors),
        "return_rmse": sqrt(fmean(error * error for error in return_errors)),
        "price_mae_points": fmean(abs(error) for error in price_errors),
        "price_rmse_points": sqrt(fmean(error * error for error in price_errors)),
    }
