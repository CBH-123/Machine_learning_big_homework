"""时序卷积模型、滚动拟合和次日预测。"""
from __future__ import annotations

import datetime as dates_module

from pathlib import Path

import datetime

import features

import torch

import json

import numpy as np

from time import perf_counter

from torch import nn

import torch.nn.functional as F

MODEL = 'tiny_tcn'

EPOCHS = 12

LOOKBACK = 31

CHANNELS = 8

DILATIONS = (1, 2, 4, 8)

SEED = 2026

LR = 0.001

WEIGHT_DECAY = 0.01

BATCH = 128

DROPOUT = 0.05

class TinyTCN(nn.Module):

    def __init__(self, extra_features=0):
        super().__init__()
        self.input_layer = nn.Conv1d(len(features.FEATURE_NAMES), CHANNELS, kernel_size=1)
        self.temporal_layers = nn.Sequential(*(CausalBlock(d) for d in DILATIONS))
        self.output_layer = nn.Linear(CHANNELS, 1)
        self.aux_layer = None
        if extra_features:
            with torch.random.fork_rng(devices=[]):
                self.aux_layer = nn.Conv1d(extra_features, CHANNELS, kernel_size=1, bias=False)
            nn.init.zeros_(self.aux_layer.weight)

    def forward(self, x):
        z = self.input_layer(x if self.aux_layer is None else x[:, :len(features.FEATURE_NAMES), :])
        if self.aux_layer is not None:
            z = z + self.aux_layer(x[:, len(features.FEATURE_NAMES):, :])
        features = self.temporal_layers(z)
        return self.output_layer(features[:, :, -1]).squeeze(-1)

class CausalBlock(nn.Module):

    def __init__(self, dilation):
        super().__init__()
        self.left_pad = 2 * dilation
        self.conv = nn.Conv1d(CHANNELS, CHANNELS, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        features = self.conv(F.pad(x, (self.left_pad, 0)))
        return x + self.dropout(F.relu(features))

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def rolling(rows, first, last, device, epochs=EPOCHS, checkpoint=None, extra=None, *, model_factory=TinyTCN, model_name=MODEL):
    if checkpoint is not None and (model_factory is not TinyTCN or model_name != MODEL):
        raise ValueError('TCN checkpoints are only supported for the original model')
    x = features.build_features(rows)
    if extra is not None:
        if checkpoint is not None or extra.ndim != 2 or extra.shape[0] != len(rows):
            raise ValueError('无效的辅助特征输入或检查点设置')
        x = np.column_stack((x, extra))
    windows = np.lib.stride_tricks.sliding_window_view(x, LOOKBACK, axis=0)
    shift = LOOKBACK - 1
    close = np.array([r['close'] for r in rows])
    returns = close[1:] / close[:-1] - 1
    eligible_features = np.flatnonzero(np.isfinite(windows).all(axis=(1, 2))) + shift
    signals = [i for i in eligible_features if i + 1 < len(rows) and first <= rows[i + 1]['date'] <= last]
    forecasts, fits, loss_rows = ([], [], [])
    for offset, i in enumerate(signals):
        asof = rows[i]['date']
        if offset % features.REFIT_EVERY == 0:
            train = eligible_features[eligible_features < i][-features.TRAIN_WINDOW:]
            if len(train) != features.TRAIN_WINDOW:
                raise ValueError(f'{asof} has only {len(train)} mature windows')
            started = perf_counter()
            model, mean, scale, losses = fit_window(windows[train - shift], returns[train], device, epochs, model_factory=model_factory)
            fit = {'model': model_name, 'fit_id': len(fits) + 1, 'fit_asof': str(asof), 'n_train': len(train), 'first_training_asof': str(rows[train[0]]['date']), 'last_training_asof': str(rows[train[-1]]['date']), 'max_training_label_date': str(rows[train[-1] + 1]['date']), 'seconds': perf_counter() - started, 'loss_first_epoch': losses[0], 'loss_last_epoch': losses[-1]}
            assert fit['last_training_asof'] < fit['max_training_label_date'] <= str(asof)
            fits.append(fit)
            loss_rows.extend(({'fit_id': fit['fit_id'], 'fit_asof': str(asof), 'epoch': epoch, 'huber_train_loss': loss} for epoch, loss in enumerate(losses, 1)))
            if len(fits) == 1 or len(fits) % 10 == 0:
                print(f"fit {len(fits):2d} | as of {asof} | {fit['seconds']:.2f}s | loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
            if len(fits) == 1 and checkpoint is not None:
                torch.save({'model_state': {k: v.cpu() for k, v in model.state_dict().items()}, 'mean': torch.tensor(mean), 'scale': torch.tensor(scale), 'fit_asof': str(asof), 'target_date': str(rows[i + 1]['date']), 'feature_names': features.FEATURE_NAMES, 'lookback': LOOKBACK}, checkpoint)
        prediction = predict(model, windows[i - shift:i - shift + 1], mean, scale, device)
        if not np.isfinite(prediction):
            raise ValueError(f'Nonfinite prediction for {asof}')
        forecasts.append({'model': model_name, 'asof_date': str(asof), 'prediction_time': f'{asof} 15:00:00', 'target_date': str(rows[i + 1]['date']), 'close_asof': float(close[i]), 'predicted_return': prediction, 'predicted_close': float(close[i] * (1 + prediction)), 'annual_volatility_20': float(x[i, features.FEATURE_NAMES.index('volatility_20')] * np.sqrt(252)), 'ma_gap_20': float(x[i, features.FEATURE_NAMES.index('ma_gap_20')]), **{k: fit[k] for k in ('fit_id', 'fit_asof', 'max_training_label_date')}})
    if not forecasts:
        raise ValueError('No usable signal dates')
    return (forecasts, fits, loss_rows)

def fit_window(windows, returns, device, epochs, *, model_factory=TinyTCN):
    mean = windows.mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(windows.std(axis=(0, 2), keepdims=True), 1e-06)
    x = torch.from_numpy(np.clip((windows - mean) / scale, -6, 6).astype(np.float32)).to(device)
    y = torch.from_numpy((returns * 100).astype(np.float32)).to(device)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = model_factory(windows.shape[1] - len(features.FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses = []
    model.train()
    for _ in range(epochs):
        total_loss = torch.zeros((), device=device)
        for ids in torch.randperm(len(x), generator=generator).split(BATCH):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.huber_loss(model(x[ids]), y[ids], delta=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(ids)
        losses.append(float(total_loss.item() / len(x)))
    return (model.eval(), mean, scale, losses)

def predict(model, window, mean, scale, device):
    x = torch.from_numpy(np.clip((window - mean) / scale, -6, 6).astype(np.float32)).to(device)
    with torch.inference_mode():
        return float(model(x).item() / 100)
