"""Compare four encoders on pre-course history under one fixed trading rule."""
from __future__ import annotations

import argparse
from itertools import groupby
from pathlib import Path
from time import perf_counter

import step2_models as m
import step3_robustness as replay_engine
import step6_tcn as tcn
from step7_net_return import ENTRY, weighted_forecasts
from step13_utility_policy import utility_decisions
from step14_consensus import verify_consensus_ledger
from step15_trend_vote import trend_vote_decisions

from torch import nn
import torch


class Recurrent(nn.Module):
    def __init__(self, layer, extra=0):
        super().__init__()
        self.encoder = layer(len(m.FEATURE_NAMES) + extra, 16, batch_first=True)
        self.dropout = nn.Dropout(tcn.DROPOUT)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        states, _ = self.encoder(x.transpose(1, 2).contiguous())
        return self.output(self.dropout(states[:, -1])).squeeze(-1)


class PatchTransformer(nn.Module):
    """PatchTST-inspired encoder with a joint embedding of all input features."""
    patch_days = 7
    stride_days = 4

    def __init__(self, extra=0, *, width=16, heads=2, feedforward=32, layers=2):
        super().__init__()
        channels = len(m.FEATURE_NAMES) + extra
        patch_count, remainder = divmod(tcn.LOOKBACK - self.patch_days, self.stride_days)
        if remainder:
            raise ValueError('The last patch must include the latest session')
        self.embedding = nn.Linear(channels * self.patch_days, width)
        self.position = nn.Parameter(torch.empty(1, patch_count + 1, width))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=tcn.DROPOUT,
            batch_first=True, activation='gelu',
        ), num_layers=layers)
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        patches = x.unfold(-1, self.patch_days, self.stride_days).transpose(1, 2).flatten(2)
        encoded = self.encoder(self.embedding(patches) + self.position)
        return self.output(encoded[:, -1]).squeeze(-1)


ENCODERS = {
    'tiny_tcn': tcn.TinyTCN,
    'gru': lambda extra: Recurrent(nn.GRU, extra),
    'lstm': lambda extra: Recurrent(nn.LSTM, extra),
    'patch_transformer': PatchTransformer,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=m.ROOT / 'outputs/step24_sequence_comparison')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.out.exists():
        parser.error(f'Choose a new output directory: {args.out}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA PyTorch is unavailable; use --device cpu')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, source = m.read_prices(m.ROOT / 'data/raw/SZ#399006.txt')
    history = [row for row in rows if row['date'] < m.START]
    first, last = m.VALID_START, history[-1]['date']
    utility = utility_decisions(history, first, last)
    args.out.mkdir(parents=True)
    tcn.write_json(args.out / 'protocol.json', {
        'source_sha256': source['sha256'], 'phase': 'pre-course historical comparison, no formal replay',
        'first_target': utility[0]['target_date'], 'last_target': str(last),
        'lookback': tcn.LOOKBACK, 'features': m.FEATURE_NAMES, 'training_window': m.TRAIN_WINDOW,
        'refit_every': m.REFIT_EVERY, 'seed': tcn.SEED, 'epochs_per_fit': tcn.EPOCHS,
        'optimizer': 'AdamW', 'learning_rate': tcn.LR, 'weight_decay': tcn.WEIGHT_DECAY,
        'batch_size': tcn.BATCH, 'dropout': tcn.DROPOUT, 'objective': 'Huber loss on next close-to-close return',
        'encoders': {'tiny_tcn': 'identical to original', 'gru': 'single layer, hidden size 16',
                     'lstm': 'single layer, hidden size 16',
                     'patch_transformer': '2 layers, width 16, 2 heads, 7-day patches with stride 4'},
        'position': 'identical buffer, utility vote and MA20 gate; trade at next open with T+1',
        'entry_threshold': ENTRY, 'fee_each_side': m.FEE_RATE, 'device': args.device,
        'torch': str(torch.__version__), 'cpu_threads': args.threads,
    })
    summary = []
    for name, factory in ENCODERS.items():
        started = perf_counter()
        print(f'\n{name}', flush=True)
        forecasts, fits, losses = tcn.rolling(history, first, last, device, model_factory=factory, model_name=name)
        if any(not (f['max_training_label_date'] <= f['fit_asof'] <= f['asof_date']
                    < f['target_date'] < str(m.START)) for f in forecasts):
            raise ValueError(f'{name} violated the historical date boundary')
        scored, prediction_metrics = m.evaluate_forecasts(history, forecasts)
        choices = trend_vote_decisions(history, weighted_forecasts(forecasts, 'buffer'), utility)
        result = replay_engine.replay(history, choices, {'name': name, 'kind': 'forecast_weight'},
                                      rebalance_on_weight_change_only=True)
        verify_consensus_ledger(history, choices, result)
        longest_flat = max((sum(1 for _ in group) for flat, group in groupby(
            mark['units'] <= 1e-8 for mark in result['curve'][1:]) if flat), default=0)
        longest_flat_all_day = max((sum(1 for _ in group) for flat, group in groupby(
            mark['units'] <= 1e-8 and mark['open_weight_after_trade'] <= 1e-8
            for mark in result['curve'][1:]) if flat), default=0)
        report = {
            'model': name, 'parameters': sum(p.numel() for p in factory(0).parameters()),
            'refits': len(fits), 'seconds': round(perf_counter() - started, 2),
            'mean_fit_seconds': round(sum(f['seconds'] for f in fits) / len(fits), 2),
            **prediction_metrics, **result['metrics'], 'longest_flat_close_sessions': longest_flat,
            'longest_flat_full_sessions': longest_flat_all_day,
        }
        for suffix, records in [('forecasts', scored), ('fits', fits), ('losses', losses),
                                ('decisions', choices), ('daily', result['curve']), ('trades', result['trades'])]:
            if records:
                m.write_csv(args.out / f'{name}_{suffix}.csv', records)
        summary.append(report)
        print(f"{name}: fee-adjusted return {report['total_return']:+.2%}; "
              f"drawdown {report['max_drawdown']:.2%}; {report['seconds']:.2f}s", flush=True)
    m.write_csv(args.out / 'summary.csv', summary)
    tcn.write_json(args.out / 'summary.json', summary)


if __name__ == '__main__':
    main()
