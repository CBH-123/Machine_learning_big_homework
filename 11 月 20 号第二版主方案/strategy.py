"""滚动训练与模型切换。收盘后产生信号，在下一交易日执行。"""
from __future__ import annotations
from pathlib import Path
import datetime
import market_data
import features
import tcn
import backtest
import utility as utility_model

from datetime import date

from itertools import groupby

import csv

from math import log1p

from math import expm1

from math import fsum

from datetime import date as training_date

from math import log1p as consensus_evidence_log1p

from math import fsum as consensus_evidence_fsum

from math import sqrt

from statistics import stdev

from statistics import fmean

from datetime import date as refinement_experiment_date

import csv as refinement_experiment_csv

from collections import deque

import numpy as np

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

import math


BASE = 'both_100'

PRIMARY = 'two_year_evidence'

ROOT = Path(__file__).resolve().parent / 'outputs'

PERIODS = {'early': ('2018-01-01', '2022-12-31'), 'development': ('2023-01-01', '2024-12-31'), 'check_2025': ('2025-01-01', '2025-10-09'), 'history': ('2023-01-01', '2025-10-09'), 'course': ('2025-10-10', '2026-10-23'), 'forward': ('2026-10-28', '2026-11-17')}

ARCHIVE = Path(__file__).resolve().parent / 'outputs/adaptive_consensus_from_prices'

BALANCED = 'balanced_fallback'

CONFIG = {'one_year': (252, 0, 1, 1.0), 'two_year': (504, 0, 1, 1.0), 'two_year_half_fallback': (504, 0, 1, 0.5), 'two_year_evidence': (504, 1, 1, 1.0), 'two_year_quarterly': (504, 0, 3, 1.0), 'one_year_annual': (252, 0, 12, 1.0)}

COST = 0.0010000000833333459

CSV = Path(__file__).resolve().parent / 'outputs/step16/external_prices.csv'

RULES = {'daily': '每日比较，预测超过 0.10% 持有', 'buffer': '预测超过 0.10% 入场，降至 0% 或以下才退出', 'mean3': '最近三次预测的平均值超过 0.10% 持有，前两个交易日观望', 'review5': '从区间首日开始每五个交易日按 0.10% 门槛复查'}

ENTRY = 0.001


INPUTS = ('return_1', 'return_5', 'return_20', 'ma_gap_20', 'volatility_20', 'price_position_20', 'candle_range', 'upper_shadow', 'lower_shadow', 'amount_ratio_20')

CALENDAR = [datetime.date(2026, 10, 28), datetime.date(2026, 10, 29), datetime.date(2026, 10, 30), datetime.date(2026, 11, 2), datetime.date(2026, 11, 3), datetime.date(2026, 11, 4), datetime.date(2026, 11, 5), datetime.date(2026, 11, 6), datetime.date(2026, 11, 9), datetime.date(2026, 11, 10), datetime.date(2026, 11, 11), datetime.date(2026, 11, 12), datetime.date(2026, 11, 13), datetime.date(2026, 11, 16)]


def archive_signals(rows):
    warm = [r for r in rows if str(r['date']) < '2018-01-01']
    records = plans_for(warm, panel(warm, 'warm'), (PRIMARY,))[PRIMARY]
    for phase in ('early', 'history', 'course'):
        for rec in read(ARCHIVE / f'{phase}_{PRIMARY}_signals.csv'):
            records.append({**rec, BASE: float(rec[BASE]), 'ridge': float(rec['ridge']), 'switched': rec['switched'] == 'True', 'strategy_weight': float(rec['strategy_weight'])})
    return records

def extension(prices):
    last, = read(ROOT / 'step17/new_session.csv')
    return prices + read_external() + [{k: date.fromisoformat(v) if k == 'date' else float(v) for k, v in last.items()}]

def ledger(rows, name, decisions):
    result = backtest.replay(rows, decisions, {'name': name, 'kind': 'forecast_weight'}, rebalance_on_weight_change_only=True)
    backtest.verify_consensus_ledger(rows, decisions, result)
    flat = (row['open_weight_after_trade'] <= 1e-08 and row['units'] <= 1e-08 for row in result['curve'][1:])
    longest = max((sum((1 for _ in days)) for empty, days in groupby(flat) if empty), default=0)
    return ({**result['metrics'], 'longest_flat_full_sessions': longest}, result)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def score(rows, plans, period, yearly):
    first, last = PERIODS[period]
    results = {}
    for name, all_orders in plans.items():
        orders = [r for r in all_orders if first <= r['target_date'] <= last]
        s, run = ledger(rows, name, orders)
        double = backtest.replay(rows, orders, {'name': name, 'kind': 'forecast_weight'}, rebalance_on_weight_change_only=True, fee_rate=2 * features.FEE_RATE)
        backtest.verify_consensus_ledger(rows, orders, double)
        s.update(phase=period, double_fee_return=double['metrics']['total_return'], guarded=sum((r['guard_active'] for r in orders)), changed=sum((r['guard_active'] and r['strategy_weight'] != r['ridge'] for r in orders)))
        results[name] = s
        if period in ('history', 'early'):
            for year in sorted({r['target_date'][:4] for r in orders}):
                moves = [log1p(row['daily_return']) for row in run['curve'][1:] if row['timestamp'].startswith(year)]
                yearly.append(dict(phase=period, name=name, year=year, total_return=expm1(fsum(moves))))
    return results

def rebuild(prices, end):
    blocks = [('warm', training_date(2016, 1, 1), training_date(2017, 12, 31)), ('early', training_date(2018, 1, 1), training_date(2022, 12, 31)), ('history', features.VALID_START, training_date(2025, 10, 9)), ('course', market_data.START, end)]
    accumulated = []
    for phase, first, last in blocks:
        rows = [r for r in prices if r['date'] <= last]
        forecasts, _, _ = tcn.rolling(rows, first, last, tcn.torch.device('cpu'))
        utility_first = training_date(2019, 1, 1) if phase == 'history' else first
        utility = [s for s in utility_model.utility_decisions(rows, utility_first, last) if s['target_date'] >= str(first)]
        raw = backtest.consensus_decisions(weighted_forecasts(forecasts, 'buffer'), utility)
        records = panel(rows, phase, raw=raw)
        for rec, fc in zip(records, forecasts, strict=True):
            rec['predicted_close'] = fc['predicted_close']
        accumulated.extend(records)
    plans = plans_for(prices, accumulated, (BASE, PRIMARY))
    plans[BALANCED] = [{**r, 'strategy_weight': (r[BASE] + r['ridge']) / 2 if r['switched'] else r[BASE], 'ridge_allocation': 0.5 if r['switched'] else 0.0} for r in plans[PRIMARY]]
    return plans

def plans_for(rows, records, names, configs=None):
    if configs is None:
        configs = CONFIG
    base = [{**r, 'strategy_weight': r[BASE]} for r in records]
    run = backtest.replay(rows, base, {'name': BASE, 'kind': 'forecast_weight'}, rebalance_on_weight_change_only=True, liquidate_at_end=False)
    backtest.verify_consensus_ledger(rows, base, run, liquidate_at_end=False)
    log_returns = [consensus_evidence_log1p(r['daily_return']) for r in run['curve'][1:]]
    output = {name: [] for name in names}
    for name in names:
        blocked = False
        review = records[0]['asof_date']
        score = None
        for i, r in enumerate(records):
            if i and r['asof_date'] != records[i - 1]['target_date']:
                raise ValueError('Gap in the shadow account')
            if name in configs:
                window, buffer, months, allocation = configs[name]
                first_in_month = i == 0 or records[i - 1]['target_date'][:7] != r['target_date'][:7]
                if i == 0 or (first_in_month and (int(r['target_date'][5:7]) - 1) % months == 0):
                    review = r['asof_date']
                    score = None
                    if i >= window:
                        observed = log_returns[i - window:i]
                        score = consensus_evidence_fsum(observed) - buffer * sqrt(window) * stdev(observed)
                    blocked = score is None or score <= 0
                part = allocation if blocked else 0.0
            else:
                part = {BASE: 0.0, 'ridge_only': 1.0, 'fixed_mix': 0.5}[name]
            w = (1 - part) * r[BASE] + part * r['ridge']
            output[name].append({**r, 'strategy_weight': w, 'review_asof': review, 'profit_score': score, 'switched': bool(part), 'ridge_allocation': part})
    return output

def panel(rows, phase, raw=None):
    if raw is None:
        if phase == 'warm':
            raw = [r for r in read(ROOT / 'step17_history/continuous_trend_vote_signals.csv') if r['target_date'] < '2018-01-01']
        else:
            raise ValueError('Historical or coursework panel requires supplied forecasts')
    first, last = (raw[0]['target_date'], raw[-1]['target_date'])
    estimates = forecasts(rows, first, last)
    index = {str(r['date']): i for i, r in enumerate(rows)}
    ridge = 0.0
    records = []
    for r, pred in zip(raw, estimates, strict=True):
        day, target = (r['asof_date'], r['target_date'])
        i = index[day]
        if (day, target) != (pred['asof_date'], pred['target_date']) or index[target] != i + 1:
            raise ValueError('Signals disagree on the next trading date')
        for signal in (r, pred):
            if not signal['max_training_label_date'] <= signal['fit_asof'] <= day < target:
                raise ValueError('Immature labels')
        gain = pred['intraday'] + pred['overnight']
        if gain > COST:
            ridge = 0.95
        elif gain < 0:
            ridge = 0.0
        records.append(dict(asof_date=day, target_date=target, fit_asof=max(r['fit_asof'], pred['fit_asof']), max_training_label_date=max(r['max_training_label_date'], pred['max_training_label_date']), both_100=float(vote(r['tcn_holds']) and vote(r['utility_holds'])), ridge=ridge, trend=float(rows[i]['close'] > fmean((b['close'] for b in rows[i - 199:i + 1]))), cash=0.0))
    return records

def read_external():
    with CSV.open(encoding='utf-8-sig', newline='') as f:
        rows = [{k: refinement_experiment_date.fromisoformat(v) if k == 'date' else float(v) for k, v in row.items()} for row in refinement_experiment_csv.DictReader(f)]
    validate(rows)
    return rows

def weighted_forecasts(forecasts, rule):
    if rule not in RULES:
        raise ValueError(f'Unknown rule: {rule}')
    position = 0.0
    recent = deque(maxlen=3)
    weighted = []
    for i, forecast in enumerate(forecasts):
        score = forecast['predicted_return']
        if rule == 'daily':
            position = float(score > ENTRY)
        elif rule == 'buffer':
            if score > ENTRY:
                position = 1.0
            elif score <= 0:
                position = 0.0
        elif rule == 'mean3':
            recent.append(score)
            position = float(len(recent) == 3 and sum(recent) / 3 > ENTRY)
        elif rule == 'review5' and i % 5 == 0:
            position = float(score > ENTRY)
        weighted.append({**forecast, 'strategy_weight': position})
    return weighted


def forecasts(rows, first, last):
    opening, close = (np.array([r[k] for r in rows]) for k in ('open', 'close'))
    dates = [str(r['date']) for r in rows]
    intra = np.log(close / opening)
    gap = np.r_[np.nan, np.log(opening[1:] / close[:-1])]
    x = features.build_features(rows)[:, [features.FEATURE_NAMES.index(k) for k in INPUTS]]
    extra = []
    for series in (intra, gap):
        for w in (1, 5, 20):
            averages = np.convolve(series, np.ones(w) / w, mode='valid')
            extra.append(np.r_[np.full(w - 1, np.nan), averages])
    x = np.column_stack([x, *extra])
    labels = np.column_stack((intra[1:-1], gap[2:]))
    target_days = [i for i, day in enumerate(dates) if first <= day <= last]
    result = []
    for offset, target in enumerate(target_days):
        i = target - 1
        if offset % features.REFIT_EVERY == 0:
            start, end = (i - 1 - features.TRAIN_WINDOW, i - 1)
            if start < 0 or not np.isfinite(x[start:end]).all() or (not np.isfinite(labels[start:end]).all()):
                raise ValueError('Incomplete mature training window')
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
            model.fit(x[start:end], labels[start:end])
            fitted, label_day = (dates[i], dates[end - 1 + 2])
        daylight, night = model.predict(x[i:i + 1])[0]
        if not label_day <= fitted <= dates[i] < dates[target]:
            raise ValueError('Forecast clock violation')
        result.append(dict(asof_date=dates[i], target_date=dates[target], fit_asof=fitted, max_training_label_date=label_day, intraday=float(daylight), overnight=float(night)))
    return result

def vote(value):
    if value in (True, 'True'):
        return True
    if value in (False, 'False'):
        return False
    raise ValueError(f'Invalid model vote {value!r}')

def validate(rows):
    if [r['date'] for r in rows] != CALENDAR:
        raise ValueError('Source dates differ from the frozen exchange calendar.')
    for r in rows:
        if not all((math.isfinite(r[k]) and r[k] > 0 for k in market_data.FIELDS[1:])) or not r['low'] <= min(r['open'], r['close']) <= max(r['open'], r['close']) <= r['high']:
            raise ValueError(f"Invalid external daily bar: {r['date']}")


