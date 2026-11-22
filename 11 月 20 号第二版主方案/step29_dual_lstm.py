"""Matched ablations of two independently trained LSTMs with different mature labels."""
import argparse
import json
from datetime import date
from pathlib import Path
from statistics import fmean

from step1_baseline import buy_and_hold
from step10_return_alternatives import opening_forecasts
from step25_medium_transformer import buffer, ledger, m, seq, tcn, torch
import numpy as np


OUT = m.ROOT / 'outputs/step29_dual_lstm'
PERIODS = {'early': (date(2018, 1, 1), date(2022, 12, 31)),
           'history': (m.VALID_START, date(2025, 10, 9)), 'course': (m.START, m.END)}
RULES = {'dual': 'Full: two yes votes; half: split and above MA20; otherwise flat.',
         'no_ma': 'Average the two votes; split always yields half.',
         'a_only': 'Remove B. Renormalize A to one whole vote; no split, so no MA gate.',
         'b_only': 'Remove A. Renormalize B to one whole vote; no split, so no MA gate.',
         'ma_only': 'Hold if close > MA20, otherwise cash.'}


def run(phase, all_rows, directory):
    directory.mkdir()
    first, last = PERIODS[phase]
    rows = [rec for rec in all_rows if rec['date'] <= last]
    index = {str(rec['date']): number for number, rec in enumerate(rows)}
    device, factory = torch.device('cpu'), seq.ENCODERS['lstm']
    first_head, first_fits, _ = tcn.rolling(rows, first, last, device, model_factory=factory, model_name='lstm')
    second_head, second_fits = opening_forecasts(rows, first, last, device, model_factory=factory)
    for key, forecasts, fits in [('A', first_head, first_fits), ('B', second_head, second_fits)]:
        m.write_csv(directory / f'{key}_forecasts.csv', forecasts)
        m.write_csv(directory / f'{key}_fits.csv', fits)
    days = [str(rec['date']) for rec in rows if rec['date'] >= first]
    for head in (first_head, second_head):
        if [rec['target_date'] for rec in head] != days or head[0]['max_training_label_date'] >= days[0]:
            raise ValueError('Invalid calendar or initial fit.')
    if phase in ('course', 'history'):
        path = m.ROOT / 'outputs' / ('step25_medium_transformer/course' if phase == 'course' else 'step24_sequence_comparison')
        known = buffer.load_forecasts(path / 'lstm_forecasts.csv', rows, first, last)
        if not np.allclose([rec['predicted_return'] for rec in first_head],
                           [rec['predicted_return'] for rec in known], atol=1e-7, rtol=0):
            raise ValueError('Archived LSTM A did not reproduce.')
    if phase == 'course':
        prefix = [rec for rec in rows if rec['date'] < first]
        masked = prefix + [{**prefix[-1], 'date': first}]
        boundary, _ = opening_forecasts(masked, first, first, device, model_factory=factory)
        if boundary[0] != second_head[0]:
            raise ValueError('First B prediction changed with unknown prices.')
    groups = {name: [] for name in RULES}
    pairs = []
    for head_a, head_b in zip(*(buffer.weighted_forecasts(head, 'buffer') for head in (first_head, second_head)), strict=True):
        asof, target = head_a['asof_date'], head_a['target_date']
        if ((head_b['asof_date'],head_b['target_date']) != (asof,target) or index[target] != index[asof]+1
                or any(not rec['max_training_label_date'] <= rec['fit_asof'] <= asof < target for rec in (head_a,head_b))):
            raise ValueError('Leakage or misaligned pair')
        pos = index[asof]
        above = rows[pos]['close'] > fmean(rec['close'] for rec in rows[pos-19:pos+1])
        vote_a, vote_b = head_a['strategy_weight'], head_b['strategy_weight']
        average = (vote_a + vote_b) / 2
        gate = average if above or average != 0.5 else 0.
        signal = {'asof_date': asof, 'target_date': target, 'vote_a': vote_a, 'vote_b':vote_b, 'above_ma20':above,
                  **{key:max(head_a[key],head_b[key]) for key in ('fit_asof','max_training_label_date')}}
        for name, weight in zip(RULES,(gate, average, vote_a, vote_b, float(above)),strict=True):
            groups[name].append({**signal, 'strategy_weight':weight})
        pairs.append(signal)
    calendar = [rec for rec in rows if rec['date'] >= first]
    hold, hold_metrics, _ = buy_and_hold(calendar, fee_rate=m.FEE_RATE)
    m.write_csv(directory/'hold_daily.csv', hold)
    summary = []
    for name, group in groups.items():
        metrics, result = ledger(rows, name, group)
        m.write_csv(directory/f'{name}_daily.csv',result['curve'])
        m.write_csv(directory/f'{name}_trades.csv',result['trades'],buffer.TRADE_FIELDS)
        summary.append({'phase':phase, **metrics})
    m.write_csv(directory/'signals.csv', groups['dual'])
    tcn.write_json(directory/'hold.json',hold_metrics)
    close = np.asarray([rec['close'] for rec in rows]); opening = np.asarray([rec['open'] for rec in rows])
    indices = np.asarray([index[day] for day in days])
    evidence = {'phase':phase, 'start':days[0], 'end':days[-1], 'days':len(days),
                'split_days':sum(rec['vote_a'] != rec['vote_b'] for rec in pairs),
                'blocked_days':sum(rec['vote_a'] != rec['vote_b'] and not rec['above_ma20'] for rec in pairs)}
    for name, actual, guesses in [('A',close[indices]/close[indices-1]-1,first_head),
                                   ('B',opening[indices[1:]]/opening[indices[:-1]]-1,second_head[:-1])]:
        predicted = np.asarray([rec['predicted_return'] for rec in guesses])
        evidence.update({f'{name}_n':len(actual),f'{name}_mae':float(np.mean(np.abs(actual-predicted))),
                         f'{name}_direction':float(np.mean(np.sign(actual)==np.sign(predicted))),
                         f'{name}_zero_mae':float(np.mean(np.abs(actual)))})
    tcn.write_json(directory/'diagnostics.json', evidence)
    m.write_csv(directory/'metrics.csv',summary)
    print(phase, [(rec['name'],round(100*rec['total_return'],2),round(100*rec['max_drawdown'],2)) for rec in summary], flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows, source = m.read_prices(m.ROOT/'data/raw/SZ#399006.txt')
    tcn.write_json(args.out/'protocol.json', {'rules':RULES,'raw_sha256':source['sha256'],
        'seed':tcn.SEED,'torch':str(torch.__version__),'lookback':tcn.LOOKBACK,'epochs':tcn.EPOCHS,
        'hidden':16,'inputs':m.FEATURE_NAMES,'train_window':m.TRAIN_WINDOW,'refit':m.REFIT_EVERY,
        'periods':{key:[str(day) for day in period] for key,period in PERIODS.items()},
        'fee_each_side':m.FEE_RATE,'parameters_per_network':sum(p.numel() for p in seq.ENCODERS['lstm'](0).parameters()),
        'course_previously_viewed':True,'selection':'Fixed dual is primary. Controls and earlier periods are diagnostic; no tuning.'})
    summaries = []
    for phase in ('course','history','early'):
        summaries.extend(run(phase,rows,args.out/phase))
    m.write_csv(args.out/'comparison.csv',summaries)
    tcn.write_json(args.out/'comparison.json',summaries)


if __name__ == '__main__':
    main()
