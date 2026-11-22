"""固定 TCN 预测，按截止日前的扣费收益选择换仓规则。"""
import hashlib
import json
import math
from collections import defaultdict, deque
from datetime import date

import step2_models as m
import step3_robustness as s3

OUT = m.ROOT / 'outputs/step7'
SOURCE = m.ROOT / 'outputs/step6'
ENTRY = 0.001
FEE = m.FEE_RATE
RULES = {
    'daily': '每日比较，预测超过 0.10% 持有',
    'buffer': '预测超过 0.10% 入场，降至 0% 或以下才退出',
    'mean3': '最近三次预测的平均值超过 0.10% 持有，前两个交易日观望',
    'review5': '从区间首日开始每五个交易日按 0.10% 门槛复查',
}
TRADE_FIELDS = ['date', 'time', 'side', 'units', 'price', 'fee', 'signal_asof', 'provenance']


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_forecasts(path, rows, first, last):
    dates = [str(row['date']) for row in rows]
    asof_by_target = dict(zip(dates[1:], dates))
    records = []
    for row in s3.read_csv(path):
        # 决策代码接收预测、交易日和拟合时点三类字段。
        known = {key: row[key] for key in ('asof_date', 'target_date', 'fit_asof', 'max_training_label_date')}
        known['predicted_return'] = float(row['predicted_return'])
        assert math.isfinite(known['predicted_return'])
        assert known['max_training_label_date'] <= known['fit_asof'] <= known['asof_date'] < known['target_date']
        assert asof_by_target[known['target_date']] == known['asof_date']
        records.append(known)
    assert [r['target_date'] for r in records] == [d for d in dates if str(first) <= d <= str(last)]
    return records


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


def replay(rows, forecasts, rule, fee=FEE):
    decisions = weighted_forecasts(forecasts, rule)
    return s3.replay(rows, decisions, {'name': rule, 'kind': 'forecast_weight'}, fee_rate=fee)


def choose(rows, forecasts, cutoff):
    prior_rows = [r for r in rows if r['date'] < cutoff]
    prior_forecasts = [r for r in forecasts if r['target_date'] < str(cutoff)]
    candidates = [replay(prior_rows, prior_forecasts, rule)['metrics'] for rule in RULES]
    winner = max(candidates, key=lambda r: (r['total_return'], -r['max_drawdown']))
    return winner['name'], candidates


def verify_ledger(rows, result):
    # 四个候选均只在满仓和空仓间切换，可用单一持仓独立复算。
    by_date = {str(r['date']): r for r in rows}
    trades_by_day = defaultdict(list)
    for trade in result['trades']:
        trades_by_day[trade['date']].append(trade)
    capital, units, peak, max_dd = 100_000.0, 0.0, 100_000.0, 0.0
    entry_date = None
    for day in result['curve'][1:]:
        when = day['timestamp'][:10]
        for trade in trades_by_day.pop(when, []):
            qty, price = trade['units'], trade['price']
            assert math.isclose(trade['fee'], qty * price * FEE, abs_tol=1e-8)
            if trade['time'] == '09:30':
                assert price == by_date[when]['open'] and trade['signal_asof'] < when
            else:
                assert when == str(m.END) and trade['side'] == 'SELL' and trade['time'] == '15:00'
                assert price == by_date[when]['close']
            if trade['side'] == 'BUY':
                assert units == 0 and when < str(m.END)
                capital -= qty * price + trade['fee']
                units, entry_date = qty, when
            else:
                assert trade['side'] == 'SELL' and entry_date < when
                assert math.isclose(qty, units, abs_tol=1e-8)
                capital += qty * price - trade['fee']
                units = 0.0
            assert capital >= -1e-7
        equity = capital + units * by_date[when]['close']
        assert math.isclose(equity, day['equity'], abs_tol=1e-7)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    assert not trades_by_day and units == 0
    assert math.isclose(capital, result['metrics']['final_equity'], abs_tol=1e-7)
    assert math.isclose(max_dd, result['metrics']['max_drawdown'], abs_tol=1e-9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, data_audit = m.read_prices(m.ROOT / 'data/raw/SZ#399006.txt')
    previous = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    assert previous['training_plan']['source_sha256'] == data_audit['sha256']
    assert previous['threshold'] == ENTRY and previous['vol_target'] is None
    assert previous['fee_rate_each_side'] == FEE
    history = [r for r in raw if r['date'] < m.START]
    val_path = SOURCE / 'validation_predictions.csv'
    plan = {'step6_protocol_id': previous['protocol_id'], 'rules': RULES, 'entry_threshold': ENTRY,
            'fee_each_side': FEE, 'cutoff': str(m.START), 'formal_period_previously_viewed': True,
            'selection': 'max cumulative net validation return, ties by lower drawdown',
            'history_prediction_sha256': hashlib.sha256(val_path.read_bytes()).hexdigest()}
    save_json('plan.json', plan)
    val = load_forecasts(val_path, history, m.VALID_START, history[-1]['date'])
    winner, scores = choose(history, val, m.START)
    protocol = {**plan, 'selected': winner, 'selection_last_target': val[-1]['target_date']}
    protocol['protocol_id'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    save_json('protocol.json', protocol)
    m.write_csv(OUT / 'validation_candidates.csv', scores)
    historical_checks = []
    for year in (2024, 2025):
        cutoff, end = date(year, 1, 1), date(year + 1, 1, 1)
        choice, _ = choose(history, val, cutoff)
        block = [r for r in val if str(cutoff) <= r['target_date'] < str(end)]
        for label, rule in [('selected_from_prior_history', choice), ('original', 'daily')]:
            result = replay(history, block, rule)['metrics']
            historical_checks.append({'year': year, 'role': label, **result})
    m.write_csv(OUT / 'historical_checks.csv', historical_checks)

    # 先落盘历史规则，再读取规定投资区间预测。
    rows = [r for r in raw if r['date'] <= m.END]
    formal = load_forecasts(SOURCE / 'test_predictions.csv', rows, m.START, m.END)
    assert formal[0]['max_training_label_date'] < str(m.START)
    results = {}
    for label, rule in [('selected', winner), ('original', 'daily')]:
        prefix = formal[:100]
        assert weighted_forecasts(formal, rule)[:100] == weighted_forecasts(prefix, rule)
        result = replay(rows, formal, rule)
        verify_ledger(rows, result)
        m.write_csv(OUT / f'{label}_daily.csv', result['curve'])
        m.write_csv(OUT / f'{label}_trades.csv', result['trades'], TRADE_FIELDS)
        gross = replay(rows, formal, rule, 0)['metrics']['total_return']
        net = result['metrics']
        # 满仓往返一次的费用因子为 (1-f)/(1+f)。二元仓位每次平仓后资金全部回到现金。
        trips = net['order_count'] // 2
        assert net['order_count'] == 2 * trips
        assert math.isclose(1 + net['total_return'], (1 + gross) * ((1 - FEE) / (1 + FEE)) ** trips, abs_tol=1e-10)
        factor = (1 + gross) ** (-1 / trips) if trips and gross > 0 else None
        results[label] = {**net, 'gross_return': gross, 'break_even_each_side_fee': (1 - factor) / (1 + factor) if factor else None}
    save_json('summary.json', {'protocol': protocol, 'historical_checks': historical_checks,
                              'validation': scores, 'formal': results, 'ledger_check': 'passed'})
    print(json.dumps({'selected': winner, 'validation': [(r['name'], r['total_return']) for r in scores],
                      'formal': results, 'ledger_check': 'passed'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
