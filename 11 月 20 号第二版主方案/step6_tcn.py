"""以小型因果卷积预测下一交易日收盘收益，比较第二步的既有基准。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import step2_models as m
import step3_robustness as s3
from verify_step2_outputs import verify
import numpy as np

if (m.ROOT / '.deps-tcn').exists():
    sys.path.append(str(m.ROOT / '.deps-tcn'))

import torch
from torch import nn
from torch.nn import functional as F

MODEL = 'tiny_tcn'
DILATIONS = (1, 2, 4, 8)
CHANNELS = 8
LOOKBACK = 1 + 2 * sum(DILATIONS)
EPOCHS = 12
BATCH = 128
LR = 0.001
WEIGHT_DECAY = 0.01
DROPOUT = 0.05
SEED = 2026
TRADE_FIELDS = ['date', 'time', 'side', 'units', 'price', 'fee', 'signal_asof', 'provenance']


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.left_pad = 2 * dilation
        self.conv = nn.Conv1d(CHANNELS, CHANNELS, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # 只在左边补零，每个输出时间点只读取当时和过去的数据。
        features = self.conv(F.pad(x, (self.left_pad, 0)))
        return x + self.dropout(F.relu(features))


class TinyTCN(nn.Module):
    def __init__(self, extra_features=0):
        super().__init__()
        self.input_layer = nn.Conv1d(len(m.FEATURE_NAMES), CHANNELS, kernel_size=1)
        self.temporal_layers = nn.Sequential(*(CausalBlock(d) for d in DILATIONS))
        self.output_layer = nn.Linear(CHANNELS, 1)
        self.aux_layer = None
        if extra_features:
            # 额外特征起初不改变输出，也不改变原网络的随机初始化。
            with torch.random.fork_rng(devices=[]):
                self.aux_layer = nn.Conv1d(extra_features, CHANNELS, kernel_size=1, bias=False)
            nn.init.zeros_(self.aux_layer.weight)

    def forward(self, x):
        z = self.input_layer(x if self.aux_layer is None else x[:, :len(m.FEATURE_NAMES), :])
        if self.aux_layer is not None:
            z = z + self.aux_layer(x[:, len(m.FEATURE_NAMES):, :])
        features = self.temporal_layers(z)
        return self.output_layer(features[:, :, -1]).squeeze(-1)


def fit_window(windows, returns, device, epochs, *, model_factory=TinyTCN):
    # 均值和标准差只来自本次训练窗。每轮从固定种子重新初始化。
    mean = windows.mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(windows.std(axis=(0, 2), keepdims=True), 1e-6)
    x = torch.from_numpy(np.clip((windows - mean) / scale, -6, 6).astype(np.float32)).to(device)
    y = torch.from_numpy((returns * 100).astype(np.float32)).to(device)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = model_factory(windows.shape[1] - len(m.FEATURE_NAMES)).to(device)
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
    return model.eval(), mean, scale, losses


def predict(model, window, mean, scale, device):
    x = torch.from_numpy(np.clip((window - mean) / scale, -6, 6).astype(np.float32)).to(device)
    with torch.inference_mode():
        return float(model(x).item() / 100)


def rolling(rows, first, last, device, epochs=EPOCHS, checkpoint=None, extra=None,
            *, model_factory=TinyTCN, model_name=MODEL):
    if checkpoint is not None and (model_factory is not TinyTCN or model_name != MODEL):
        raise ValueError("TCN checkpoints are only supported for the original model")
    x = m.build_features(rows)
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
    forecasts, fits, loss_rows = [], [], []
    for offset, i in enumerate(signals):
        asof = rows[i]['date']
        if offset % m.REFIT_EVERY == 0:
            train = eligible_features[eligible_features < i][-m.TRAIN_WINDOW:]
            if len(train) != m.TRAIN_WINDOW:
                raise ValueError(f'{asof} has only {len(train)} mature windows')
            started = perf_counter()
            model, mean, scale, losses = fit_window(
                windows[train - shift], returns[train], device, epochs, model_factory=model_factory)
            fit = {'model': model_name, 'fit_id': len(fits) + 1, 'fit_asof': str(asof), 'n_train': len(train),
                   'first_training_asof': str(rows[train[0]]['date']),
                   'last_training_asof': str(rows[train[-1]]['date']),
                   'max_training_label_date': str(rows[train[-1] + 1]['date']),
                   'seconds': perf_counter() - started, 'loss_first_epoch': losses[0], 'loss_last_epoch': losses[-1]}
            assert fit['last_training_asof'] < fit['max_training_label_date'] <= str(asof)
            fits.append(fit)
            loss_rows.extend({'fit_id': fit['fit_id'], 'fit_asof': str(asof), 'epoch': epoch, 'huber_train_loss': loss}
                             for epoch, loss in enumerate(losses, 1))
            if len(fits) == 1 or len(fits) % 10 == 0:
                print(f"fit {len(fits):2d} | as of {asof} | {fit['seconds']:.2f}s | loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
            if len(fits) == 1 and checkpoint is not None:
                torch.save({'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                            'mean': torch.tensor(mean), 'scale': torch.tensor(scale),
                            'fit_asof': str(asof), 'target_date': str(rows[i + 1]['date']),
                            'feature_names': m.FEATURE_NAMES, 'lookback': LOOKBACK}, checkpoint)
        prediction = predict(model, windows[i - shift:i - shift + 1], mean, scale, device)
        if not np.isfinite(prediction):
            raise ValueError(f'Nonfinite prediction for {asof}')
        forecasts.append({'model': model_name, 'asof_date': str(asof), 'prediction_time': f'{asof} 15:00:00',
                          'target_date': str(rows[i + 1]['date']), 'close_asof': float(close[i]),
                          'predicted_return': prediction, 'predicted_close': float(close[i] * (1 + prediction)),
                          'annual_volatility_20': float(x[i, m.FEATURE_NAMES.index('volatility_20')] * np.sqrt(252)),
                          'ma_gap_20': float(x[i, m.FEATURE_NAMES.index('ma_gap_20')]),
                          **{k: fit[k] for k in ('fit_id', 'fit_asof', 'max_training_label_date')}})
    if not forecasts:
        raise ValueError('No usable signal dates')
    return forecasts, fits, loss_rows


def smoke(history, device, out):
    target = next(r['date'] for r in history if r['date'] >= m.VALID_START)
    forecast, fits, losses = rolling(history, target, target, device, epochs=2)
    changed = [dict(row) for row in history]
    for row in changed:
        if row['date'] >= target:
            for key in ('open', 'high', 'low', 'close'):
                row[key] *= 1.05
    repeated, _, _ = rolling(changed, target, target, device, epochs=2)
    assert np.isclose(forecast[0]['predicted_return'], repeated[0]['predicted_return'], rtol=0, atol=1e-7)
    m.write_csv(out/'smoke_forecast.csv', forecast)
    m.write_csv(out/'epoch_losses.csv', losses)
    write_json(out/'smoke_check.json', {'status': 'passed', 'device': str(device), 'fit': fits[0],
                                      'unchanged_when_future_prices_are_perturbed': True})
    print(f'Smoke check passed. See {out}', flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def check_checkpoint(rows, first, path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['fit_asof'] == first['asof_date'] and saved['target_date'] == first['target_date']
    assert saved['lookback'] == LOOKBACK and saved['feature_names'] == m.FEATURE_NAMES
    pos = next(i for i, r in enumerate(rows) if str(r['date']) == saved['fit_asof'])
    x = m.build_features(rows[:pos + 1])[-LOOKBACK:].T[None, :, :]
    model = TinyTCN().to(device).eval()
    model.load_state_dict(saved['model_state'])
    from_checkpoint = predict(model, x, saved['mean'].numpy(), saved['scale'].numpy(), device)
    assert abs(from_checkpoint - first['predicted_return']) < 1e-7
    return from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true', help='只在历史验证段训练两轮并检查未来行情扰动')
    parser.add_argument('--out', type=Path, help='本次输出目录，须为新目录或空目录')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Run --device cpu or install a CUDA PyTorch wheel.')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = args.out or m.ROOT / ('outputs/step6_smoke' if args.smoke else 'outputs/step6')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'{out} is occupied. Choose a fresh --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows_all, source = m.read_prices(m.ROOT/'data/raw/SZ#399006.txt')
    history = [row for row in rows_all if row['date'] < m.START]
    if args.smoke:
        smoke(history, device, out)
        return
    baseline_path = m.ROOT/'outputs/step2/summary.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    assert baseline['source_sha256'] == source['sha256']
    assert baseline['protocol']['fee_rate_each_side'] == m.FEE_RATE
    hardware = {'torch': str(torch.__version__), 'python': platform.python_version(),
                'device': str(device), 'cpu_threads': args.threads,
                'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    parameters = sum(p.numel() for p in TinyTCN().parameters())
    plan = {'source_sha256': source['sha256'], 'step2_summary_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            'model': MODEL, 'parameters': parameters, 'lookback': LOOKBACK, 'channels': CHANNELS,
            'dilations': list(DILATIONS), 'dropout': DROPOUT, 'batch_size': BATCH, 'epochs': EPOCHS,
            'learning_rate': LR, 'weight_decay': WEIGHT_DECAY, 'seed': SEED,
            'objective': 'Huber loss on percentage-point returns, delta 1',
            'target': 'next close / current close - 1', 'initial_training_before': str(m.START),
            'train_window': m.TRAIN_WINDOW, 'refit_every': m.REFIT_EVERY,
            'pretrained_weights': False, 'formal_period_previously_viewed': True,
            'selection': 'TCN strategy grid from step2 using pretest data; compare fixed RF rule too'}
    write_json(out/'research_plan.json', plan)
    print(f'{parameters:,} parameters | {LOOKBACK} trading days | {hardware}', flush=True)
    validation, vf, vl = rolling(history, m.VALID_START, history[-1]['date'], device)
    val_scored, val_metrics = m.evaluate_forecasts(history, validation)
    picked, candidates = m.select_protocol([val_metrics], {MODEL: validation}, history, m.FEE_RATE)
    # 旧选择函数描述的是多模型选择，覆盖成这轮实际实验的协议。
    del picked['protocol_id'], picked['forest'], picked['ridge_alpha']
    picked['model_selection'] = 'fixed tiny TCN architecture; selection of trade rule only'
    picked['training_plan'] = plan
    picked['protocol_id'] = hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()[:16]
    write_json(out/'protocol.json', picked)
    m.write_csv(out/'validation_predictions.csv', val_scored)
    m.write_csv(out/'validation_rule_candidates.csv', candidates)
    print(f"Frozen rule {picked['protocol_id']} | threshold {picked['threshold']:.2%} | vol target {picked['vol_target']}", flush=True)
    rows = [row for row in rows_all if row['date'] <= m.END]
    checkpoint = out/'initial_weights.pt'
    forecasts, tf, tl = rolling(rows, m.START, m.END, device, checkpoint=checkpoint)
    assert tf[0]['max_training_label_date'] < str(m.START)
    assert len(validation) == baseline['validation_model_metrics'][0]['n_predictions']
    assert len(forecasts) == baseline['test_model_metrics'][0]['n_predictions']
    first_check = check_checkpoint(rows, forecasts[0], checkpoint, device)
    scored, test_metrics = m.evaluate_forecasts(rows, forecasts)
    m.write_csv(out/'test_predictions.csv', scored)
    m.write_csv(out/'fit_audit.csv', [{'phase': 'validation', **r} for r in vf] + [{'phase': 'test', **r} for r in tf])
    m.write_csv(out/'epoch_losses.csv', [{'phase': 'validation', **r} for r in vl] + [{'phase': 'test', **r} for r in tl])
    previous = baseline['protocol']
    specs = [{'name': 'tcn_selected', 'threshold': picked['threshold'], 'profile': 'vol15' if picked['vol_target'] else 'raw'},
             {'name': 'tcn_same_rf_rule', 'threshold': previous['threshold'], 'profile': 'vol15' if previous['vol_target'] else 'raw'},
             {'name': 'buy_hold', 'kind': 'fixed', 'weight': 1.0}]
    runs = []
    for spec in specs:
        run = s3.replay(rows, forecasts, spec)
        m.write_csv(out/f"{spec['name']}_daily.csv", run['curve'])
        m.write_csv(out/f"{spec['name']}_trades.csv", run['trades'], TRADE_FIELDS)
        runs.append(run)
    matched = s3.matched_exposure(rows, forecasts, runs[0], 'tcn_matched_exposure')['metrics']
    summary = {'protocol': picked, 'hardware': hardware,
               'timing': {'training_seconds': sum(r['seconds'] for r in vf + tf),
                          'total_seconds_before_ledger_verification': perf_counter() - started,
                          'validation_fits': len(vf), 'formal_fits': len(tf),
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None},
               'validation_model_metrics': [*baseline['validation_model_metrics'], val_metrics],
               'test_model_metrics': [*baseline['test_model_metrics'], test_metrics],
               'test_strategy_metrics': [{'strategy': r['metrics']['name'], **r['metrics']} for r in runs],
               'step2_random_forest_strategy': next(r for r in baseline['test_strategy_metrics'] if r['strategy'] == 'selected'),
               'matched_exposure_diagnostic': matched, 'checkpoint_first_forecast_reproduced': first_check}
    write_json(out/'summary.json', summary)
    m.write_csv(out/'model_metrics.csv', [{'phase': phase, **r}
                 for phase in ('validation', 'test') for r in summary[f'{phase}_model_metrics']])
    verify(out)
    print(f"Complete in {perf_counter() - started:.1f}s | first checkpoint matched | see {out}", flush=True)


if __name__ == '__main__':
    main()
