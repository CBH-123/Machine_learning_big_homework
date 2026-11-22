"""根据持有收益与交易成本训练持仓决策模型。"""
from __future__ import annotations
import features

import numpy as np

from scipy.special import expit

from scipy.optimize import minimize

FEATURES = ('return_1', 'return_5', 'return_20', 'ma_gap_60', 'volatility_20', 'candle_body', 'amount_ratio_20', 'price_position_20')

CUTOFF = 0.5

TURNOVER_SMOOTHING = 0.02

FEE = 0.0005

L2 = 0.0001

def utility_decisions(rows, first, last):
    """Produce one binary decision for each target session in the inclusive date range."""
    dates = [str(r['date']) for r in rows]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError('Expected strictly ordered trading sessions')
    target_indices = [i for i, day in enumerate(dates) if str(first) <= day <= str(last)]
    if not target_indices or target_indices[0] == 0:
        raise ValueError("The requested dates must include each signal's prior session")
    x = features.build_features(rows)[:, [features.FEATURE_NAMES.index(name) for name in FEATURES]]
    opens = np.asarray([r['open'] for r in rows])
    open_returns = opens[2:] / opens[1:-1] - 1
    decisions = []
    for offset, target in enumerate(target_indices):
        signal = target - 1
        if offset % features.REFIT_EVERY == 0:
            train_end = signal - 1
            train_start = train_end - features.TRAIN_WINDOW
            if train_start < 0 or not np.isfinite(x[train_start:train_end]).all():
                raise ValueError('Need 1260 consecutive mature feature rows')
            mean, scale, coefs = _fit(x[train_start:train_end], open_returns[train_start:train_end])
            last_mature_label = dates[train_end - 1 + 2]
            fit_asof = dates[signal]
        scaled = np.clip((x[signal] - mean) / scale, -5, 5)
        score = float(expit(coefs[0] + scaled @ coefs[1:]))
        if not np.isfinite(score):
            raise ValueError(f'Unavailable utility policy inputs on {dates[signal]}')
        decisions.append({'model': 'direct_utility', 'asof_date': dates[signal], 'target_date': dates[target], 'strategy_weight': float(score > CUTOFF), 'utility_score': score, 'fit_asof': fit_asof, 'max_training_label_date': last_mature_label, 'n_train': features.TRAIN_WINDOW})
    return decisions

def _fit(features, returns):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-08)
    design = np.column_stack((np.ones(len(features)), np.clip((features - mean) / scale, -5, 5)))
    result = minimize(_loss_and_gradient, np.zeros(design.shape[1]), args=(design, returns), jac=True, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-11, 'gtol': 1e-08})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Utility policy optimizer failed: {result.message}')
    return (mean, scale, result.x)

def _loss_and_gradient(parameters, design, open_returns):
    weights = expit(design @ parameters)
    changes = np.diff(np.r_[0.0, weights, 0.0])
    smooth = np.hypot(changes, TURNOVER_SMOOTHING)
    turnover_fee = (np.log1p(FEE) - np.log1p(-FEE)) / 2
    loss = -np.log1p(weights * open_returns).mean() + turnover_fee * (smooth - TURNOVER_SMOOTHING).sum() / len(weights) + L2 * np.dot(parameters, parameters) / 2
    turnover_gradient = changes / smooth
    weight_gradient = (-open_returns / (1 + weights * open_returns) + turnover_fee * np.diff(-turnover_gradient)) / len(weights)
    gradient = design.T @ (weight_gradient * weights * (1 - weights)) + L2 * parameters
    return (loss, gradient)
