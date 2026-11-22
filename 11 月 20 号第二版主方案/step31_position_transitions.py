"""Frozen entry, exit and distribution-day overlays on the published and dual-LSTM strategies."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from math import isclose, expm1, fsum, log1p
from itertools import groupby

import step19_breadth_overlay as s19
from step25_medium_transformer import m, ledger, seq, tcn, buffer
from step29_dual_lstm import PERIODS


ROOT = m.ROOT / 'outputs'
RULES = {
    'base': 'Unchanged raw target.',
    'stage_buy': 'On an increase of the raw target, go to midpoint of prior and new raw target; otherwise track raw.',
    'strong_exit': 'On a decrease, if above MA20 and close in upper third of daily range, retain half the target decrease for one session.',
    'supply': 'On down close in bottom third of daily range on above-median 20-session amount, cap at half; return to raw otherwise.',
    'paired': 'Use staged buy and strong exit, then apply the supply cap.',
}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def signals(rows, raw):
    indices = {str(row['date']): position for position,row in enumerate(rows)}
    result = {name:[] for name in RULES}
    previous = 0.
    last_target = None
    for record in raw:
        day, target = record['asof_date'], record['target_date']
        pos = indices[day]
        if (pos<20 or indices[target] != pos+1 or last_target is not None and last_target != day
                or not record['max_training_label_date'] <= record['fit_asof'] <= day < target):
            raise ValueError('Signal clock, order or maturity violation')
        today = rows[pos]
        baseline = float(record['strategy_weight'])
        if baseline not in (0., .5, 1.):
            raise ValueError('Unsupported baseline weight')
        location = ((today['close']-today['low'])/(today['high']-today['low'])
                    if today['high']>today['low'] else .5)
        trend = today['close'] > fsum(row['close'] for row in rows[pos-19:pos+1])/20
        supply = (location<=1/3 and today['close']<rows[pos-1]['close'] and
                  today['amount']>median(row['amount'] for row in rows[pos-20:pos]))
        buy = (baseline+previous)/2 if baseline>previous else baseline
        sell = (baseline+previous)/2 if baseline<previous and trend and location>=2/3 else baseline
        cap = .5 if supply else 1.
        weights = {'base':baseline,'stage_buy':buy,'strong_exit':sell,'supply':min(cap,baseline),
                   'paired':min(cap, buy if baseline>previous else sell)}
        for name, weight in weights.items():
            result[name].append({'asof_date':day, 'target_date':target,'fit_asof':record['fit_asof'],
                                 'max_training_label_date':record['max_training_label_date'],
                                 'original':baseline,'prior_original':previous, 'supply':supply,
                                 'above_ma20':trend, 'range_location':location,'strategy_weight':weight})
        previous, last_target = baseline,target
    return result


def sources(phase, rows):
    if phase=='course':
        repo=read(ROOT/'step19/breadth_half_signals.csv')
    else:
        first,last=PERIODS[phase]
        name='step13_history/predictions.csv' if phase=='early' else 'step6/validation_predictions.csv'
        forecasts=buffer.load_forecasts(ROOT/name,rows,first,rows[-1]['date'])
        from_day=first if phase=='early' else s19.s13.HISTORY_START
        direct=seq.utility_decisions(rows,from_day,rows[-1]['date'])
        breadth=s19.s9.load_breadth(m.START)
        repo=s19.decisions_for(rows,forecasts,direct,breadth,first,last)['breadth_half']
    return {'repo':repo, 'dual_lstm':read(ROOT/'step29_dual_lstm'/phase/'signals.csv')}


def evaluate(phase, all_rows, out):
    first,last=PERIODS[phase]
    rows=[row for row in all_rows if row['date']<=last]
    summary=[]
    for family, raw in sources(phase,rows).items():
        if [rec['target_date'] for rec in raw] != [str(row['date']) for row in rows if row['date']>=first]:
            raise ValueError('Incomplete phase')
        choices=signals(rows,raw)
        count=len(raw)//2
        prefix_end=raw[count-1]['target_date']
        truncated=signals([row for row in rows if str(row['date'])<=prefix_end],raw[:count])
        for name, decisions in choices.items():
            if truncated[name]!=decisions[:count]:
                raise ValueError('Future-dependent decisions')
            metrics,result=ledger(rows,name,decisions)
            if name=='base':
                if family=='dual_lstm':
                    ref=next(item for item in read(ROOT/'step29_dual_lstm'/phase/'metrics.csv') if item['name']=='dual')
                elif phase=='course':
                    ref=json.loads((m.ROOT/'Machine_learning_big_homework/results/metrics.json').read_text())
                elif phase=='history':
                    ref=json.loads((ROOT/'step19/history_protocol.json').read_text())['history']['breadth_half']
                else:
                    ref=None
                if ref and any(not isclose(float(ref[key]),metrics[key],abs_tol=1e-8) for key in
                               ('total_return','max_drawdown','order_count','fees_paid')):
                    raise ValueError(f'Mismatch with {family}/{phase} benchmark')
            stressed=seq.replay_engine.replay(rows,decisions,{'name':name,'kind':'forecast_weight'},
                                              fee_rate=2*m.FEE_RATE,rebalance_on_weight_change_only=True)
            seq.verify_consensus_ledger(rows,decisions,stressed)
            april=expm1(fsum(log1p(mark['daily_return']) for mark in result['curve'][1:] if mark['timestamp'][5:7]!='04'))
            metrics.update(phase=phase,family=family,first=raw[0]['target_date'],last=raw[-1]['target_date'],
                           double_fee_return=stressed['metrics']['total_return'],omit_april_contribution=april,
                           days_changed=sum(rec['strategy_weight']!=rec['original'] for rec in decisions))
            summary.append(metrics)
            base_path=out/f'{phase}_{family}_{name}'
            for field,recs in [('signals',decisions),('daily',result['curve']),('trades',result['trades'])]:
                m.write_csv(base_path.with_name(base_path.name+f'_{field}.csv'),recs)
        print(phase,family,[(row['name'],f"{row['total_return']:.2%}",f"{row['max_drawdown']:.2%}") for row in summary
                            if row['family']==family],flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'step31_position_transitions')
    out=parser.parse_args().out
    out.mkdir(parents=True,exist_ok=False)
    rows,source=m.read_prices(m.ROOT/'data/raw/SZ#399006.txt')
    if json.loads((ROOT/'step29_dual_lstm/protocol.json').read_text())['raw_sha256']!=source['sha256']:
        raise ValueError('Archived inputs changed')
    tcn.write_json(out/'protocol.json',{'rules':RULES,'price_sha256':source['sha256'],'primary_family':'repo',
        'additional_family':'dual_lstm','phase_start_state':'fresh cash and preceding raw target zero',
        'early_breadth_note':'Before the point-in-time breadth series begins in December 2022, the breadth override is inactive.',
        'qualification':'Higher net return and no higher drawdown than its own base in BOTH pre-course phases, '
                        'beats base at double fees in both, and >=2 consecutive full cash sessions in both. '
                        'If multiple pass, select highest 2023-cutoff return; otherwise retain base.',
        'formal_period_previously_viewed':True,'fee_each_side':m.FEE_RATE})
    metrics=evaluate('early',rows,out)+evaluate('history',rows,out)
    grid={(row['phase'],row['family'],row['name']):row for row in metrics}
    selection={}
    for family in ('repo','dual_lstm'):
        passed=[]
        for name in list(RULES)[1:]:
            ok=True
            for phase in ('early','history'):
                new,old=grid[phase,family,name],grid[phase,family,'base']
                ok &= (new['total_return']>old['total_return'] and new['max_drawdown']<=old['max_drawdown']+1e-10 and
                       new['double_fee_return']>old['double_fee_return'] and new['longest_flat_full_sessions']>=2)
            if ok: passed.append(name)
        selection[family]={'passed':passed,'selected':max(passed,key=lambda name:grid['history',family,name]['total_return'],default='base')}
    tcn.write_json(out/'selection_before_course.json',selection)
    metrics+=evaluate('course',rows,out)
    m.write_csv(out/'comparison.csv',metrics)
    tcn.write_json(out/'comparison.json',metrics)
    print(selection,flush=True)


if __name__=='__main__':
    main()
