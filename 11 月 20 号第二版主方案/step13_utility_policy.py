"""Train a small position policy directly on simulated returns and turnover costs."""

import step2_models as m
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


FEATURES = ("return_1", "return_5", "return_20", "ma_gap_60", "volatility_20",
            "candle_body", "amount_ratio_20", "price_position_20")
FEE = m.FEE_RATE
TURNOVER_SMOOTHING = 0.02
L2 = 0.0001
CUTOFF = 0.5

PLAN = {
    "features": list(FEATURES),
    "model": "single affine sigmoid with 8 inputs and an intercept",
    "training_target": "maximize mean log growth from next-open to following-open returns, net of smooth turnover costs",
    "training_exposure": "continuous between zero and one; zero initial and final exposure for turnover",
    "fee_each_side": FEE,
    "turnover_smoothing": TURNOVER_SMOOTHING,
    "l2": L2,
    "execution_cutoff": CUTOFF,
    "position_rule": "hold fully when sigmoid score is strictly greater than 0.5; otherwise stay in cash",
    "train_window": m.TRAIN_WINDOW,
    "refit_every": m.REFIT_EVERY,
    "optimizer": "deterministic L-BFGS-B from zero, maximum 100 iterations",
}


def _loss_and_gradient(parameters, design, open_returns):
    weights = expit(design @ parameters)
    changes = np.diff(np.r_[0.0, weights, 0.0])
    smooth = np.hypot(changes, TURNOVER_SMOOTHING)
    # With a flat start/end, paired buy/sell costs equal this log fee times absolute turnover for binary positions.
    turnover_fee = (np.log1p(FEE) - np.log1p(-FEE)) / 2
    loss = (-np.log1p(weights * open_returns).mean()
            + turnover_fee * (smooth - TURNOVER_SMOOTHING).sum() / len(weights)
            + L2 * np.dot(parameters, parameters) / 2)
    turnover_gradient = changes / smooth
    weight_gradient = (-open_returns / (1 + weights * open_returns)
                       + turnover_fee * np.diff(-turnover_gradient)) / len(weights)
    gradient = design.T @ (weight_gradient * weights * (1 - weights)) + L2 * parameters
    return loss, gradient


def _fit(features, returns):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    design = np.column_stack((np.ones(len(features)), np.clip((features - mean) / scale, -5, 5)))
    result = minimize(_loss_and_gradient, np.zeros(design.shape[1]), args=(design, returns), jac=True,
                      method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Utility policy optimizer failed: {result.message}")
    return mean, scale, result.x


def utility_decisions(rows, first, last):
    """Produce one binary decision for each target session in the inclusive date range."""
    dates = [str(r["date"]) for r in rows]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("Expected strictly ordered trading sessions")
    target_indices = [i for i, day in enumerate(dates) if str(first) <= day <= str(last)]
    if not target_indices or target_indices[0] == 0:
        raise ValueError("The requested dates must include each signal's prior session")
    x = m.build_features(rows)[:, [m.FEATURE_NAMES.index(name) for name in FEATURES]]
    opens = np.asarray([r["open"] for r in rows])
    # Row j has known close inputs and a holding window from open j+1 to open j+2.
    open_returns = opens[2:] / opens[1:-1] - 1
    decisions = []
    for offset, target in enumerate(target_indices):
        signal = target - 1
        if offset % m.REFIT_EVERY == 0:
            train_end = signal - 1
            train_start = train_end - m.TRAIN_WINDOW
            if train_start < 0 or not np.isfinite(x[train_start:train_end]).all():
                raise ValueError("Need 1260 consecutive mature feature rows")
            mean, scale, coefs = _fit(x[train_start:train_end], open_returns[train_start:train_end])
            last_mature_label = dates[(train_end - 1) + 2]
            fit_asof = dates[signal]
        scaled = np.clip((x[signal] - mean) / scale, -5, 5)
        score = float(expit(coefs[0] + scaled @ coefs[1:]))
        if not np.isfinite(score):
            raise ValueError(f"Unavailable utility policy inputs on {dates[signal]}")
        decisions.append({"model": "direct_utility", "asof_date": dates[signal], "target_date": dates[target],
                          "strategy_weight": float(score > CUTOFF), "utility_score": score,
                          "fit_asof": fit_asof, "max_training_label_date": last_mature_label,
                          "n_train": m.TRAIN_WINDOW})
    return decisions
