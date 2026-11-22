"""Allocate to the consensus strategy on evidence of its own net profitability."""
import argparse
import json
from math import fsum, isclose, log1p, sqrt
from pathlib import Path
from statistics import stdev

from step38_history_repair import BASE, TOL, assess, ledger, m, panel, read, ROOT, seq, tcn
from threadpoolctl import threadpool_limits


# Warmup uses the independent policy if there are insufficient marks for the chosen window.
CONFIG = {
    "one_year": (252, 0, 1, 1.),
    "two_year": (504, 0, 1, 1.),
    "two_year_half_fallback": (504, 0, 1, .5),
    "two_year_evidence": (504, 1, 1, 1.),
    "two_year_quarterly": (504, 0, 3, 1.),
    "one_year_annual": (252, 0, 12, 1.),
}


def plans_for(rows, records, names, configs=None):
    if configs is None:
        configs=CONFIG
    base=[{**r,"strategy_weight":r[BASE]} for r in records]
    run=seq.replay_engine.replay(rows,base,{"name":BASE,"kind":"forecast_weight"},
                                 rebalance_on_weight_change_only=True,liquidate_at_end=False)
    seq.verify_consensus_ledger(rows,base,run,liquidate_at_end=False)
    log_returns=[log1p(r["daily_return"]) for r in run["curve"][1:]]
    output={name:[] for name in names}
    for name in names:
        blocked=False
        review=records[0]["asof_date"]
        score=None
        for i,r in enumerate(records):
            if i and r["asof_date"]!=records[i-1]["target_date"]:
                raise ValueError("Gap in the shadow account")
            if name in configs:
                window,buffer,months,allocation=configs[name]
                first_in_month=i==0 or records[i-1]["target_date"][:7]!=r["target_date"][:7]
                if i==0 or first_in_month and (int(r["target_date"][5:7])-1)%months==0:
                    review=r["asof_date"]
                    score=None
                    if i>=window:
                        observed=log_returns[i-window:i]
                        score=fsum(observed)-buffer*sqrt(window)*stdev(observed)
                    blocked=score is None or score<=0
                part=allocation if blocked else 0.
            else:
                part={BASE:0.,"ridge_only":1.,"fixed_mix":.5}[name]
            w=(1-part)*r[BASE]+part*r["ridge"]
            output[name].append({**r,"strategy_weight":w,"review_asof":review,"profit_score":score,
                                 "switched":bool(part),"ridge_allocation":part})
    return output


def run_study(out):
    prices,meta=m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    out.mkdir(parents=True,exist_ok=False)
    tcn.write_json(out/"protocol.json",{
        "source_sha256":meta["sha256"], "window_buffer_reviewmonths_fallbackweight":CONFIG,
        "scope":"Monthly review by default. Use full consensus if its own trailing realized net log growth is positive "
        "(evidence variant subtracts one daily standard deviation times sqrt(window)); otherwise use the independent "
        "open-only Ridge model. No comparison with recent Ridge profit. Fixed single-expert and equal mix controls.",
        "cuts":"2016-17 warmup; 2018-20 development, 2021-22 chronological check, 2023 to 2025-10-09 recent check; "
        "historical windows and course were previously viewed in other studies.",
        "selection":"Positive early and recent returns at normal and doubled fees; early drawdown strictly below base; "
        "positive recent return at least the same-period buy-and-hold return; at least 3/5 early calendar years "
        "ahead of baseline and >=2 consecutive flat sessions both periods. Rank by higher minimum annualized return "
        "of the two history windows then smaller early drawdown. Course replay only for locked winner and controls. "
        "If no winner, also replay closest historical candidate by count of checks then same ranking.",
        "formal_gate":"Net return >= exact consensus baseline, max drawdown <= exact baseline, 1e-10 tolerance.",
        "init":"Start the shadow from cash in 2016, never reset it. Each scored interval starts with 100000 cash. "
        "Use fixed, established segment-specific Ridge refit calendar, previously tested in step34.",
    })
    all_names=(BASE,"ridge_only","fixed_mix",*CONFIG)
    panel_history=[]
    scores={}
    years=[]
    def extend(phase,last):
        rows=[r for r in prices if str(r["date"])<=last]
        panel_history.extend(panel(rows,phase))
        return rows
    extend("warm","2017-12-31")
    rows=extend("early","2022-12-31")
    plans=plans_for(rows,panel_history,all_names)
    for p,a,b in (("development","2018-01-01","2020-12-31"),("later_early","2021-01-01","2022-12-31"),
                  ("early","2018-01-01","2022-12-31")):
        scores[p]=assess(rows,plans,a,b,p,years,out)
    rows=extend("history","2025-10-09")
    plans=plans_for(rows,panel_history,all_names)
    scores["history"]=assess(rows,plans,"2023-01-01","2025-10-09","history",years,out)
    early,recent=scores["early"],scores["history"]
    annual={(r["name"],r["year"]):r["total_return"] for r in years if r["phase"]=="early"}
    checks={}
    for n in CONFIG:
        checks[n]=dict(early_positive=early[n]["total_return"]>0,recent_positive=recent[n]["total_return"]>0,
            double_positive=min(early[n]["double_fee_return"],recent[n]["double_fee_return"])>0,
            early_drawdown=early[n]["max_drawdown"]<early[BASE]["max_drawdown"],
            recent_beats_index=recent[n]["total_return"]>=recent[n]["buy_hold_return"],
            early_years=sum(annual[n,str(y)]>annual[BASE,str(y)]+TOL for y in range(2018,2023))>=3,
            cash=min(early[n]["longest_flat_full_sessions"],recent[n]["longest_flat_full_sessions"])>=2)
    selected=max(CONFIG,key=lambda n:(sum(checks[n].values()),min(early[n]["annualized_return"],
                 recent[n]["annualized_return"]),-early[n]["max_drawdown"]))
    passed=all(checks[selected].values())
    tcn.write_json(out/"historical_choice.json",{"checks":checks,"locked":selected,"passed":passed})
    names=(BASE,"ridge_only","fixed_mix",selected)
    before={n:plans[n] for n in names}
    rows=extend("course","2026-10-23")
    plans=plans_for(rows,panel_history,names)
    if any(plans[n][:len(x)]!=x for n,x in before.items()):
        raise ValueError("Changed historical targets after extending prices")
    k=next(i for i,r in enumerate(panel_history) if r["target_date"].startswith("2026-04"))
    cut=[r for r in rows if str(r["date"])<=panel_history[k]["target_date"]]
    cut[-1]={**cut[-2],"date":cut[-1]["date"]}
    masked=plans_for(cut,panel_history[:k+1],names)
    if any(plans[n][:k+1]!=v for n,v in masked.items()):
        raise ValueError("Target day used to score the model")
    scores["course"]=assess(rows,plans,"2025-10-10","2026-10-23","course",years,out)
    refs={r["phase"]:r for r in read(ROOT/"step36_agreement_entry/sizing_check.csv") if r["name"]==BASE}
    for phase in ("early","history","course"):
        for key in ("total_return","max_drawdown","fees_paid","double_fee_return"):
            assert isclose(scores[phase][BASE][key],float(refs[phase][key]),abs_tol=1e-9)
    g=scores["course"]
    formal=(g[selected]["total_return"]+TOL>=g[BASE]["total_return"] and
            g[selected]["max_drawdown"]<=g[BASE]["max_drawdown"]+TOL and g[selected]["longest_flat_full_sessions"]>=2)
    tcn.write_json(out/"verdict.json",{"locked":selected,"history_passed":passed,"formal_passed":formal})
    m.write_csv(out/"comparison.csv",[r for group in scores.values() for r in group.values()])
    m.write_csv(out/"years.csv",years)
    for n in all_names:
        print(n,[(p,round(s[n]["total_return"]*100,2),round(s[n]["max_drawdown"]*100,2))
                 for p,s in scores.items() if n in s])
    print("locked",selected,"passed",passed,formal)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"step39_consensus_evidence")
    with threadpool_limits(limits=1):
        run_study(parser.parse_args().out)
