"""Train a position policy that gives exceptional winning periods less influence."""

import step2_models as m
from step13_utility_policy import CUTOFF, FEATURES, FEE, L2, TURNOVER_SMOOTHING
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


BLOCK_DAYS = 21
GAIN_SCALE = 0.02

PLAN = {
    "features": list(FEATURES),
    "model": "single affine sigmoid with 8 inputs and an intercept",
    "training_target": "maximize per-session block utility of net log growth",
    "block_days": BLOCK_DAYS,
    "block_gain_scale": GAIN_SCALE,
    "block_utility": "B if B <= 0; scale * tanh(B / scale) otherwise, where B is block net log growth",
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
    """Losses retain full weight, while the marginal credit for gains falls smoothly."""
    weights = expit(design @ parameters)
    changes = np.diff(np.r_[0.0, weights, 0.0])
    smooth = np.hypot(changes, TURNOVER_SMOOTHING)
    turnover_fee = (np.log1p(FEE) - np.log1p(-FEE)) / 2
    costs = turnover_fee * (smooth - TURNOVER_SMOOTHING)
    # Charge transitions to their destination day; the last day also pays to exit.
    daily_net = np.log1p(weights * open_returns) - costs[:-1]
    daily_net[-1] -= costs[-1]
    block_net = daily_net.reshape(-1, BLOCK_DAYS).sum(axis=1)
    gain_fraction = np.tanh(np.maximum(block_net, 0.0) / GAIN_SCALE)
    utility = np.minimum(block_net, 0.0) + GAIN_SCALE * gain_fraction
    loss = -utility.sum() / len(weights) + L2 * np.dot(parameters, parameters) / 2

    daily_credit = np.repeat(1 - gain_fraction**2, BLOCK_DAYS)
    cost_gradient = turnover_fee * np.r_[daily_credit, daily_credit[-1]] * changes / smooth
    weight_gradient = (-daily_credit * open_returns / (1 + weights * open_returns)
                       - np.diff(cost_gradient)) / len(weights)
    gradient = design.T @ (weight_gradient * weights * (1 - weights)) + L2 * parameters
    return loss, gradient


def _fit(features, returns):
    if len(returns) != m.TRAIN_WINDOW or len(returns) % BLOCK_DAYS:
        raise ValueError("Expected one training window composed of complete blocks")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    design = np.column_stack((np.ones(len(features)), np.clip((features - mean) / scale, -5, 5)))
    result = minimize(_loss_and_gradient, np.zeros(design.shape[1]), args=(design, returns), jac=True,
                      method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Block utility policy optimizer failed: {result.message}")
    return mean, scale, result.x


def robust_decisions(rows, first, last):
    """Produce one causal binary position for each target session, inclusive."""
    dates = [str(row["date"]) for row in rows]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("Expected strictly ordered trading sessions")
    target_indices = [i for i, day in enumerate(dates) if str(first) <= day <= str(last)]
    if not target_indices or target_indices[0] == 0:
        raise ValueError("The requested dates must include each signal's prior session")
    features = m.build_features(rows)[:, [m.FEATURE_NAMES.index(name) for name in FEATURES]]
    opens = np.asarray([row["open"] for row in rows])
    # Row j's close features forecast the return between the next two session opens.
    open_returns = opens[2:] / opens[1:-1] - 1
    decisions = []
    for offset, target in enumerate(target_indices):
        signal = target - 1
        if offset % m.REFIT_EVERY == 0:
            train_end = signal - 1
            train_start = train_end - m.TRAIN_WINDOW
            if train_start < 0 or not np.isfinite(features[train_start:train_end]).all():
                raise ValueError("Need 1260 consecutive mature feature rows")
            mean, scale, coefs = _fit(features[train_start:train_end], open_returns[train_start:train_end])
            fit_asof = dates[signal]
            last_mature_label = dates[train_end + 1]
        scaled = np.clip((features[signal] - mean) / scale, -5, 5)
        score = float(expit(coefs[0] + scaled @ coefs[1:]))
        if not np.isfinite(score):
            raise ValueError(f"Unavailable block utility policy inputs on {dates[signal]}")
        decisions.append({"model": "block_robust_utility", "asof_date": dates[signal],
                          "target_date": dates[target], "strategy_weight": float(score > CUTOFF),
                          "utility_score": score, "fit_asof": fit_asof,
                          "max_training_label_date": last_mature_label, "n_train": m.TRAIN_WINDOW})
    return decisions
