"""Frozen walk-forward experiment: model capacity and an optional downside-risk cap."""
import argparse
from functools import partial
from itertools import groupby
import json
import math
from pathlib import Path

import step24_sequence_comparison as seq
import step7_net_return as buffer
from step24_sequence_comparison import m, tcn, torch
import numpy as np


LARGER = partial(seq.PatchTransformer, width=48, heads=4, feedforward=96, layers=3)
MODEL = 'medium_transformer'
CAP = 'medium_transformer_cap'
BASE = 'tiny_tcn'
SMALL = 'patch_transformer'
LSTM = 'lstm'


def risk_limits(rows):
    """Today's 20-session downside deviation versus the 252 preceding observations."""
    closes = np.array([row['close'] for row in rows])
    losses = np.minimum(np.diff(closes) / closes[:-1], 0.0)
    deviations = np.full(len(rows), np.nan)
    deviations[20:] = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(losses, 20)**2, axis=1))
    limits = {}
    for index in range(272, len(rows)):
        cutoff = float(np.quantile(deviations[index-252:index], 0.80))
        limits[str(rows[index]['date'])] = (float(deviations[index]), cutoff)
    return limits


def apply_cap(choices, limits):
    results = []
    for decision in choices:
        current, cutoff = limits[decision['asof_date']]
        weight = decision['strategy_weight']
        results.append({**decision, 'uncapped_weight': weight, 'downside_20': current,
                        'prior_q80': cutoff, 'strategy_weight': min(weight, 0.5) if current > cutoff else weight})
    return results


def ledger(rows, name, decisions):
    result = seq.replay_engine.replay(rows, decisions, {'name': name, 'kind': 'forecast_weight'},
                                      rebalance_on_weight_change_only=True)
    seq.verify_consensus_ledger(rows, decisions, result)
    flat = (row['open_weight_after_trade'] <= 1e-8 and row['units'] <= 1e-8 for row in result['curve'][1:])
    longest = max((sum(1 for _ in days) for empty, days in groupby(flat) if empty), default=0)
    return {**result['metrics'], 'longest_flat_full_sessions': longest}, result


def evaluate(rows, first, last, phase, directory, device):
    directory.mkdir()
    reference_dir = m.ROOT / 'outputs/step24_sequence_comparison'
    utility = seq.utility_decisions(rows, first, last)
    limits = risk_limits(rows)
    scores, segments = {}, {}
    for name in (BASE, SMALL, LSTM, MODEL):
        if phase == 'history' and name != MODEL:
            forecasts = buffer.load_forecasts(reference_dir / f'{name}_forecasts.csv', rows, first, last)
        elif name == BASE:
            forecasts = buffer.load_forecasts(m.ROOT / 'outputs/step6/test_predictions.csv', rows, first, last)
        else:
            factory = LARGER if name == MODEL else seq.ENCODERS[name]
            print(f'{phase}: training {name}', flush=True)
            forecasts, fits, _ = tcn.rolling(rows, first, last, device, model_factory=factory, model_name=name)
            m.write_csv(directory / f'{name}_forecasts.csv', forecasts)
            m.write_csv(directory / f'{name}_fits.csv', fits)
        if (forecasts[0]['max_training_label_date'] >= str(first)
                or any(not rec['max_training_label_date'] <= rec['fit_asof'] <= rec['asof_date'] < rec['target_date']
                       for rec in forecasts)):
            raise ValueError(f'Training clock violation: {name}')
        signals = seq.trend_vote_decisions(rows, buffer.weighted_forecasts(forecasts, 'buffer'), utility)
        choices = {name: signals}
        if name == MODEL:
            choices[CAP] = apply_cap(signals, limits)
        for variant, decisions in choices.items():
            metrics, result = ledger(rows, variant, decisions)
            scores[variant] = metrics
            years = sorted({rec['target_date'][:4] for rec in decisions})
            segments[variant] = {year: ledger(rows, variant, [rec for rec in decisions if
                                 rec['target_date'].startswith(year)])[0] for year in years}
            m.write_csv(directory / f'{variant}_signals.csv', decisions)
            m.write_csv(directory / f'{variant}_daily.csv', result['curve'])
            m.write_csv(directory / f'{variant}_trades.csv', result['trades'], buffer.TRADE_FIELDS)
        if phase == 'history' and name != MODEL:
            archive = next(item for item in json.loads((reference_dir / 'summary.json').read_text('utf-8'))
                           if item['model'] == name)
        elif phase == 'course' and name == BASE:
            archive = json.loads((m.ROOT / 'outputs/step15/formal_results.json').read_text('utf-8'))['formal']['trend_vote']
        else:
            continue
        for metric in ('total_return', 'max_drawdown', 'order_count'):
            if not math.isclose(scores[name][metric], archive[metric], abs_tol=1e-10):
                raise ValueError(f'Could not reproduce {phase}/{name}/{metric}')
    tcn.write_json(directory / 'results.json', {'full': scores, 'calendar_slices_fresh_cash': segments})
    m.write_csv(directory / 'comparison.csv', list(scores.values()))
    return scores, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=m.ROOT / 'outputs/step25_medium_transformer')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error('Choose a fresh output directory and positive thread count.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    rows, source = m.read_prices(m.ROOT / 'data/raw/SZ#399006.txt')
    past = [row for row in rows if row['date'] < m.START]
    reference_plan = json.loads((m.ROOT / 'outputs/step24_sequence_comparison/protocol.json').read_text('utf-8'))
    if source['sha256'] != reference_plan['source_sha256']:
        raise ValueError('Source differs from the archived comparison')
    tcn.write_json(args.out / 'protocol.json', {
        'source_sha256': source['sha256'], 'seed': tcn.SEED, 'torch': str(torch.__version__),
        'device': str(device), 'threads': args.threads, 'baseline_protocol': reference_plan,
        'medium': {'layers': 3, 'width': 48, 'heads': 4, 'feedforward': 96,
                   'parameters': sum(weight.numel() for weight in LARGER().parameters())},
        'control': 'Identical data, features, label, rolling fit schedule, optimizer, epochs, fee and vote.',
        'optional_cap': '20-day RMS negative daily return above trailing 252-day 80th percentile '
                        '(excluding current reading): cap target at half. All inputs available at signal close.',
        'course': [str(m.START), str(m.END)], 'initial_fit_strictly_before_start': True,
        'course_dates_previously_viewed': True, 'single_seed': True,
        'screen': 'Of the two predefined medium variants, require higher net return and lower max drawdown '
                  'than BOTH the TCN and small Transformer on the whole history and higher return than TCN '
                  'in at least two of three calendar slices. Pick highest historical return of qualifiers, '
                  'otherwise retain TCN. Replay all frozen variants once as diagnostics.',
    })
    history, segments = evaluate(past, m.VALID_START, past[-1]['date'], 'history', args.out / 'history', device)
    passed = [name for name in (MODEL, CAP) if all(
        history[name]['total_return'] > history[control]['total_return'] and
        history[name]['max_drawdown'] < history[control]['max_drawdown'] for control in (BASE, SMALL))
        and sum(block['total_return'] > segments[BASE][year]['total_return']
                for year, block in segments[name].items()) >= 2]
    selected = max(passed, key=lambda name: history[name]['total_return']) if passed else BASE
    tcn.write_json(args.out / 'selection_before_course.json', {'passed': passed, 'selected': selected})
    course, _ = evaluate([row for row in rows if row['date'] <= m.END], m.START, m.END,
                         'course', args.out / 'course', device)
    for name in course:
        print(name, *(f"{scores[name][metric]:.2%}" for scores in (history, course)
                      for metric in ('total_return', 'max_drawdown')), flush=True)


if __name__ == '__main__':
    main()
