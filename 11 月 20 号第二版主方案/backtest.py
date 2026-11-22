"""指数交易撮合、手续费及 T+1 账本核对。"""
from __future__ import annotations
from pathlib import Path
import market_data
import features

import numpy as np

from math import isclose

from collections import defaultdict


import csv

PROFILES = ('raw', 'vol15', 'trend20', 'trend20_vol15')

def replay(rows, forecasts, spec, fee_rate=features.FEE_RATE, override_weights=None, provenance='causal', capital=100000, rebalance_on_weight_change_only=False, liquidate_at_end=True):
    """与第二步相同的撮合口径，另记录开盘调仓后实际仓位用于对照。

    provenance=ex_post 的外部持仓序列仅用于诊断，不能伪装成当时可执行的决策。
    可选择只在目标仓位变化时调仓；持有期间实际仓位随价格变动。
    默认在最后一天禁增仓并收盘清算。观察继续运行的账户时可只按最后收盘估值。
    """
    if not forecasts or provenance not in ('causal', 'ex_post'):
        raise ValueError('预测或来源标记无效')
    if override_weights is not None and provenance != 'ex_post':
        raise ValueError('外部事后持仓必须明确标为 ex_post')
    if spec.get('ex_post', False) and provenance != 'ex_post':
        raise ValueError('事后同仓位规则不能标为可执行决策')
    weights = np.asarray(override_weights if override_weights is not None else policy_weights(forecasts, spec))
    if len(weights) != len(forecasts) or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1 + 1e-12):
        raise ValueError('目标仓位必须是0到1之间的有限数')
    weights = np.clip(weights, 0, 1)
    by_date = {str(r['date']): r for r in rows}
    days = [by_date[f['target_date']] for f in forecasts]
    account = market_data.LongOnlyT1Account(capital, [r['date'] for r in days], fee_rate)
    curve = [{'timestamp': f"{days[0]['date']} 09:29:00", 'equity': capital, 'cash': capital, 'units': 0, 'target_weight': 0, 'open_weight_after_trade': 0, 'close_weight': 0, 'daily_return': 0, 'drawdown': 0}]
    trades = []
    peak = previous = capital
    for i, (row, f, weight) in enumerate(zip(days, forecasts, weights)):
        on, price = (row['date'], row['open'])
        terminal = liquidate_at_end and i == len(days) - 1
        if not f['asof_date'] < str(on):
            raise ValueError('交易信号不能使用当天完整行情')
        if f.get('max_training_label_date') and (not f['max_training_label_date'] <= f['fit_asof'] <= f['asof_date']):
            raise ValueError('训练答案越过了预测时点')
        difference = (account.cash + account.units * price) * float(weight) / price - account.units
        if rebalance_on_weight_change_only:
            if i and weight == weights[i - 1]:
                difference = 0
            elif difference > 0:
                difference /= 1 + float(weight) * fee_rate
            elif difference < 0:
                difference /= 1 - float(weight) * fee_rate

        def record(side, q, p, clock, signal):
            trades.append({'date': str(on), 'time': clock, 'side': side, 'units': q, 'price': p, 'fee': q * p * fee_rate, 'signal_asof': signal, 'provenance': provenance})
        signal = f['asof_date'] if provenance == 'causal' else 'EX_POST_DIAGNOSTIC'
        if difference > 1e-08 and (not terminal):
            q = min(difference, account.cash / (price * (1 + fee_rate)))
            if q > 1e-08:
                account.buy(on, price, q)
                record('BUY', q, price, '09:30', signal)
        elif difference < -1e-08:
            q = min(-difference, account.sellable(on))
            if q > 1e-08:
                account.sell(on, price, q)
                record('SELL', q, price, '09:30', signal)
        open_equity = account.cash + account.units * price
        open_weight = account.units * price / open_equity
        if terminal and account.units > 1e-08:
            q = account.units
            account.sell(on, row['close'], q)
            record('SELL', q, row['close'], '15:00', 'PRECOMMITTED_END')
        equity = account.cash + account.units * row['close']
        peak = max(peak, equity)
        curve.append({'timestamp': f'{on} 15:00:00', 'equity': equity, 'cash': account.cash, 'units': account.units, 'target_weight': float(weight), 'open_weight_after_trade': open_weight, 'close_weight': account.units * row['close'] / equity, 'daily_return': equity / previous - 1, 'drawdown': equity / peak - 1})
        previous = equity
    daily = np.array([r['daily_return'] for r in curve[1:]])
    std = np.std(daily, ddof=1) if len(daily) > 1 else 0
    metrics = {'name': spec['name'], 'provenance': provenance, 'trading_days': len(days), 'fee_rate_each_side': fee_rate, 'final_equity': curve[-1]['equity'], 'total_return': curve[-1]['equity'] / capital - 1, 'annualized_return': (curve[-1]['equity'] / capital) ** (252 / len(days)) - 1, 'max_drawdown': -min((r['drawdown'] for r in curve)), 'sharpe_rf_zero': float(np.mean(daily) / std * np.sqrt(252)) if std > 1e-12 else 0.0, 'average_open_weight': float(np.mean([r['open_weight_after_trade'] for r in curve[1:]])), 'average_close_weight': float(np.mean([r['close_weight'] for r in curve[1:]])), 'order_count': len(trades), 'fees_paid': sum((t['fee'] for t in trades))}
    return {'curve': curve, 'trades': trades, 'metrics': metrics}

def verify_consensus_ledger(rows, decisions, run, liquidate_at_end=True):
    """Rebuild shares and cash from order records, independently of the account."""

    def check(condition, msg):
        if not condition:
            raise ValueError(f'Consensus replay audit: {msg}')

    def equal(a, b):
        return isclose(a, b, rel_tol=1e-10, abs_tol=1e-08)
    prices = {str(row['date']): row for row in rows}
    marks, trades, metrics = (run['curve'], run['trades'], run['metrics'])
    check(decisions and len(marks) == len(decisions) + 1, 'incorrect daily marks')
    fee_rate = metrics['fee_rate_each_side']
    check(0 <= fee_rate < 1, 'invalid fee rate')
    initial = marks[0]
    check(initial['timestamp'] == f"{decisions[0]['target_date']} 09:29:00" and initial['cash'] > 0 and equal(initial['cash'], initial['equity']) and equal(initial['units'], 0), 'account must start in cash')
    cash, units, fees = (initial['cash'], 0.0, 0.0)
    peak = previous_close = cash
    max_drawdown = previous_target = 0.0
    fills = defaultdict(list)
    previous_key = None
    for fill in trades:
        key = (fill['date'], fill['time'])
        check(previous_key is None or previous_key < key, 'orders must follow the trading clock')
        fills[fill['date']].append(fill)
        previous_key = key
    for index, (decision, mark) in enumerate(zip(decisions, marks[1:], strict=True)):
        day, target = (decision['target_date'], decision['strategy_weight'])
        row = prices[day]
        open_price, close_price = (row['open'], row['close'])
        final_day = liquidate_at_end and index == len(decisions) - 1
        day_fills = {fill['time']: fill for fill in fills.pop(day, [])}
        check(set(day_fills).issubset({'09:30', '15:00'}), f'{day}: invalid order time')
        sellable = units
        opening_weight_before = units * open_price / (cash + units * open_price)

        def apply_fill(fill, price):
            nonlocal cash, units, fees, sellable
            quantity, fee = (fill['units'], fill['fee'])
            check(fill['provenance'] == 'causal' and equal(fill['price'], price) and (quantity > 0) and equal(fee, quantity * price * fee_rate), f'{day}: invalid execution or fee')
            if fill['side'] == 'BUY':
                cash -= quantity * price + fee
                units += quantity
            else:
                check(fill['side'] == 'SELL' and quantity <= sellable + 1e-08, f'{day}: sale exceeds T+1 inventory')
                cash += quantity * price - fee
                units -= quantity
                sellable -= quantity
            fees += fee
            check(cash >= -1e-08 and units >= -1e-08, f'{day}: negative cash or shares')
        if '09:30' in day_fills:
            fill = day_fills['09:30']
            check(target != previous_target and fill['signal_asof'] == decision['asof_date'] < day and (not final_day or fill['side'] == 'SELL'), f'{day}: opening order violates the decision clock or target rule')
            apply_fill(fill, open_price)
        opening_weight = units * open_price / (cash + units * open_price)
        if target != previous_target and (not final_day or target <= opening_weight_before):
            check(equal(opening_weight, target), f'{day}: missed target on the transition')
        check(equal(mark['target_weight'], target) and equal(mark['open_weight_after_trade'], opening_weight), f'{day}: opening weight or target does not match ledger')
        if '15:00' in day_fills:
            fill = day_fills['15:00']
            check(final_day and fill['side'] == 'SELL' and equal(fill['units'], units) and (fill['signal_asof'] == 'PRECOMMITTED_END'), f'{day}: invalid liquidation')
            apply_fill(fill, close_price)
        check(not final_day or equal(units, 0), f'{day}: shares remain after final close')
        equity = cash + units * close_price
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        max_drawdown = max(-drawdown, max_drawdown)
        check(mark['timestamp'] == f'{day} 15:00:00' and equal(mark['equity'], equity) and equal(mark['cash'], cash) and equal(mark['units'], units) and equal(mark['close_weight'], units * close_price / equity) and equal(mark['daily_return'], equity / previous_close - 1) and equal(mark['drawdown'], drawdown), f'{day}: closing marks do not reconcile')
        previous_target, previous_close = (target, equity)
    check(not fills and len(trades) == metrics['order_count'] and equal(fees, metrics['fees_paid']) and equal(previous_close, metrics['final_equity']) and equal(previous_close / initial['cash'] - 1, metrics['total_return']) and equal(max_drawdown, metrics['max_drawdown']), 'account totals do not reconcile')

def consensus_decisions(tcn_weights, utility):
    """Match the binary signals by target and as-of; retain source weights for audit."""
    combined = []
    previous_target = None
    for tcn, util in zip(tcn_weights, utility, strict=True):
        asof, target = (str(tcn['asof_date']), str(tcn['target_date']))
        if (asof, target) != (str(util['asof_date']), str(util['target_date'])) or asof >= target or (previous_target is not None and target <= previous_target):
            raise ValueError('Consensus requires aligned, strictly ordered, prior-session signals')
        tcn_weight = _binary_weight(tcn, asof, 'TCN')
        utility_weight = _binary_weight(util, asof, 'Utility')
        fit_dates = [str(d['fit_asof']) for d in (tcn, util) if d.get('fit_asof')]
        label_dates = [str(d['max_training_label_date']) for d in (tcn, util) if d.get('max_training_label_date')]
        combined.append({'model': 'consensus', 'asof_date': asof, 'target_date': target, 'strategy_weight': (tcn_weight + utility_weight) / 2, 'tcn_holds': bool(tcn_weight), 'utility_holds': bool(utility_weight), 'fit_asof': max(fit_dates, default=''), 'max_training_label_date': max(label_dates, default='')})
        previous_target = target
    if not combined:
        raise ValueError('Consensus requires at least one paired trading session')
    return combined

def policy_weights(forecasts, spec):
    kind = spec.get('kind', 'ml')
    if kind == 'forecast_weight':
        return np.array([float(f['strategy_weight']) for f in forecasts])
    if kind == 'fixed':
        return np.full(len(forecasts), spec['weight'], dtype=float)
    if kind == 'ma20':
        return np.array([float(f['ma_gap_20'] > 0) for f in forecasts])
    weights = np.array([float(f['predicted_return'] > spec['threshold']) for f in forecasts])
    profile = spec.get('profile', 'raw')
    if profile not in PROFILES:
        raise ValueError('未知风控方案')
    if 'trend20' in profile:
        weights *= np.array([float(f['ma_gap_20'] > 0) for f in forecasts])
    if 'vol15' in profile:
        weights *= np.array([min(1.0, 0.15 / max(f['annual_volatility_20'], 1e-08)) for f in forecasts])
    return weights

def _binary_weight(decision, asof, name):
    if decision.get('provenance', 'causal') != 'causal':
        raise ValueError(f'{name} must contain causal decisions')
    fit_asof = decision.get('fit_asof')
    max_label = decision.get('max_training_label_date')
    if fit_asof and str(fit_asof) > asof or (max_label and (not fit_asof or str(max_label) > str(fit_asof))):
        raise ValueError(f'{name} training information exceeds its as-of date')
    try:
        weight = float(decision['strategy_weight'])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} requires binary target weights') from exc
    if weight not in (0.0, 1.0):
        raise ValueError(f'{name} requires binary target weights')
    return weight

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
