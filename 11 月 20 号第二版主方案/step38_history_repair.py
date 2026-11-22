"""Monthly, causal selection between the unanimous model and independent fallback policies."""
import argparse
import json
from math import expm1, fsum, isclose, log1p
from pathlib import Path
from statistics import fmean

from step34_session_policy import COST, forecasts
from step32_entry_votes import PERIODS, ROOT, ledger, m, read, seq, sources, tcn, vote
from step1_baseline import buy_and_hold


BASE = "both_100"
EXPERTS = ("ridge", "trend")
GATES = ("loss", "leader")
CONFIGS = {f"{expert}_{gate}_{size}": (expert, gate, size/100)
           for expert in EXPERTS for gate in GATES for size in (50, 100)}
CONFIGS["cash_loss"] = ("cash", "loss", 1.)
TOL = 1e-10


def panel(rows, phase, raw=None):
    if raw is None:
        if phase == "warm":
            raw = [r for r in read(ROOT/"step17_history/continuous_trend_vote_signals.csv") if r["target_date"] < "2018-01-01"]
        else:
            raw = sources(phase, rows)["repo"]
    first, last = raw[0]["target_date"], raw[-1]["target_date"]
    estimates = forecasts(rows, first, last)
    index = {str(r["date"]):i for i,r in enumerate(rows)}
    ridge = 0.
    records = []
    for r, pred in zip(raw, estimates, strict=True):
        day, target = r["asof_date"], r["target_date"]
        i = index[day]
        if (day, target) != (pred["asof_date"], pred["target_date"]) or index[target] != i+1:
            raise ValueError("Signals disagree on the next trading date")
        for signal in (r, pred):
            if not signal["max_training_label_date"] <= signal["fit_asof"] <= day < target:
                raise ValueError("Immature labels")
        gain = pred["intraday"] + pred["overnight"]
        if gain > COST:
            ridge = .95
        elif gain < 0:
            ridge = 0.
        records.append(dict(asof_date=day, target_date=target,
                            fit_asof=max(r["fit_asof"],pred["fit_asof"]),
                            max_training_label_date=max(r["max_training_label_date"],pred["max_training_label_date"]),
                            both_100=float(vote(r["tcn_holds"]) and vote(r["utility_holds"])), ridge=ridge,
                            trend=float(rows[i]["close"] > fmean(b["close"] for b in rows[i-199:i+1])), cash=0.))
    return records


def signals(rows, records):
    indices = {str(r["date"]):i for i,r in enumerate(rows)}
    for i,r in enumerate(records):
        if (indices[r["target_date"]] != indices[r["asof_date"]]+1 or
                i and records[i-1]["target_date"] != r["asof_date"]):
            raise ValueError("Nonconsecutive shadow ledger")
    curves = {}
    for name in (BASE,*EXPERTS):
        orders = [{**r,"strategy_weight":r[name]} for r in records]
        run = seq.replay_engine.replay(rows, orders, {"name":name,"kind":"forecast_weight"},
                                       rebalance_on_weight_change_only=True, liquidate_at_end=False)
        seq.verify_consensus_ledger(rows, orders, run, liquidate_at_end=False)
        curves[name] = [r["equity"] for r in run["curve"]]
    result = {n:[] for n in (BASE, *CONFIGS)}
    active = {}
    for i,r in enumerate(records):
        month = r["target_date"][:7]
        if i==0 or month != records[i-1]["target_date"][:7]:
            review = r["asof_date"]
            # The initial cash mark is index 0, mark i is the close BEFORE target i.
            for name,(expert,gate,_) in CONFIGS.items():
                switch = False
                if i>=252:
                    b=curves[BASE]
                    base_year = b[i]/b[i-252]-1
                    if expert=="cash":
                        switch = base_year<0
                    else:
                        alt = curves[expert]
                        year = alt[i]/alt[i-252]-1
                        if gate=="loss":
                            switch = base_year<0 and year>0
                        else:
                            switch = year>max(0.,base_year) and alt[i]/alt[i-63]>max(1.,b[i]/b[i-63])
                active[name] = switch
        for name, series in result.items():
            weight = r[BASE]
            switched = name!=BASE and active[name]
            if switched:
                expert,_,size=CONFIGS[name]
                weight = (1-size)*weight + size*r[expert]
            series.append({**r,"review_asof":review,"switched":switched,"strategy_weight":weight})
    return result


def assess(rows, plans, first, last, phase, year_rows, out):
    scores={}
    benchmark = buy_and_hold([r for r in rows if first<=str(r["date"])<=last], fee_rate=m.FEE_RATE)[1]
    for name, full in plans.items():
        orders=[r for r in full if first<=r["target_date"]<=last]
        score, run=ledger(rows,name,orders)
        double=seq.replay_engine.replay(rows,orders,{"name":name,"kind":"forecast_weight"},
                                        fee_rate=2*m.FEE_RATE,rebalance_on_weight_change_only=True)
        seq.verify_consensus_ledger(rows,orders,double)
        score.update(phase=phase, double_fee_return=double["metrics"]["total_return"],
                     buy_hold_return=benchmark["total_return"], buy_hold_drawdown=benchmark["max_drawdown"],
                     switched_sessions=sum(r["switched"] for r in orders),
                     changed_sessions=sum(r["strategy_weight"]!=r[BASE] for r in orders))
        scores[name]=score
        if phase in ("early","history","course","stress"):
            years=sorted({r["target_date"][:4] for r in orders})
            for year in years:
                marks=[p for p in run["curve"][1:] if p["timestamp"].startswith(year)]
                year_rows.append(dict(phase=phase, name=name, year=year,
                    total_return=expm1(fsum(log1p(p["daily_return"]) for p in marks))))
        if phase=="course":
            m.write_csv(out/f"{name}_signals.csv",orders)
            m.write_csv(out/f"{name}_daily.csv",run["curve"])
            m.write_csv(out/f"{name}_trades.csv",run["trades"])
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"step38_history_repair")
    out=parser.parse_args().out
    out.mkdir(parents=True,exist_ok=False)
    all_rows,meta=m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    tcn.write_json(out/"protocol.json",{
        "base":BASE,"sha256":meta["sha256"],"fee":m.FEE_RATE,"candidates":CONFIGS,
        "experts":"Previously studied two-target Ridge, 95% or cash; close above 200-session mean, 100% or cash.",
        "monthly_clock":"Review only for the first open of each month, at preceding close. Shadow accounts accrue actual fees "
                        "and are never reset across the continuous 2016-course ledger. Base is default.",
        "gates":"loss: last 252-session base net return negative AND alternative positive. leader: alternative net "
                "positive and ahead of baseline over both 252 and 63 sessions. Cash-only loss gate is a control.",
        "development":"Choose highest net return during 2018-2020 among candidates with positive net return, "
                      "better return and lower drawdown than base; use 2021-2022 as a separate check.",
        "history_screen":"Before course, require early return positive, both early and recent returns positive, "
                         "early drawdown smaller, recent drawdown no greater, both double-fee returns positive, "
                         "at least 3/5 early years beating base and consecutive flat sessions in both segments. "
                         "Among passers rank by smaller early drawdown then higher early return.",
        "formal_gate":"Return at least baseline, drawdown at most baseline, tolerance 1e-10. Formal history was seen in previous studies.",
        "annual_table":"Annual contribution to continuously marked segment equity. Fresh cash for each standalone segment."
    })
    tables, years, accumulated = {}, [], []
    def extend(phase, last):
        rows=[r for r in all_rows if str(r["date"])<=last]
        accumulated.extend(panel(rows,phase))
        return rows,signals(rows,accumulated)

    extend("warm","2017-12-31")
    rows,plans=extend("early","2022-12-31")
    tables["development"]=assess(rows,plans,"2018-01-01","2020-12-31","development",years,out)
    g=tables["development"]
    candidates=[n for n in CONFIGS if n!="cash_loss" and g[n]["total_return"]>max(0.,g[BASE]["total_return"])
                and g[n]["max_drawdown"]<g[BASE]["max_drawdown"]]
    dev=max(candidates,key=lambda n:g[n]["total_return"],default=BASE)
    tcn.write_json(out/"development_selection.json",{"selected":dev,"qualified":candidates})
    tables["later_early"]=assess(rows,plans,"2021-01-01","2022-12-31","later_early",years,out)
    tables["early"]=assess(rows,plans,"2018-01-01","2022-12-31","early",years,out)
    rows,plans=extend("history","2025-10-09")
    tables["history"]=assess(rows,plans,"2023-01-01","2025-10-09","history",years,out)
    annual={(r["name"],r["year"]):r["total_return"] for r in years if r["phase"]=="early"}
    early,recent=tables["early"],tables["history"]
    qualified=[]
    for n in CONFIGS:
        if n=="cash_loss":
            continue
        wins=sum(annual[n,str(y)]>annual[BASE,str(y)]+TOL for y in range(2018,2023))
        if (wins>=3 and early[n]["max_drawdown"]<early[BASE]["max_drawdown"] and
            recent[n]["max_drawdown"]<=recent[BASE]["max_drawdown"]+TOL and
            all(s[n]["total_return"]>0 and s[n]["double_fee_return"]>0 and s[n]["longest_flat_full_sessions"]>=2
                for s in (early,recent))):
            qualified.append(n)
    selected=min(qualified,key=lambda n:(early[n]["max_drawdown"],-early[n]["total_return"]),default=BASE)
    tcn.write_json(out/"history_selection.json",{"selected":selected,"qualified":qualified,"development_winner":dev})
    cutoff_plans=plans
    rows,plans=extend("course","2026-10-23")
    if any(s!=plans[n][:len(s)] for n,s in cutoff_plans.items()):
        raise ValueError("Future course prices changed historical plans")
    # A masked target day at an intra-course month boundary cannot affect that day's target.
    k=next(i for i,r in enumerate(accumulated) if r["target_date"].startswith("2026-04"))
    cut=[r for r in rows if str(r["date"])<=accumulated[k]["target_date"]]
    cut[-1]={**cut[-2],"date":cut[-1]["date"]}
    prefix=signals(cut,accumulated[:k+1])
    if any(s!=plans[n][:k+1] for n,s in prefix.items()):
        raise ValueError("A future fill affected the monthly choice")
    tables["course"]=assess(rows,plans,"2025-10-10","2026-10-23","course",years,out)
    refs={r["phase"]:r for r in read(ROOT/"step36_agreement_entry/sizing_check.csv") if r["name"]==BASE}
    for p in ("early","history","course"):
        for key in ("total_return","max_drawdown","double_fee_return","fees_paid"):
            if not isclose(tables[p][BASE][key],float(refs[p][key]),abs_tol=1e-9):
                raise ValueError(f"Archived base mismatch {p} {key}")
    g=tables["course"]
    passed=[n for n in qualified if g[n]["total_return"]+TOL>=g[BASE]["total_return"]
            and g[n]["max_drawdown"]<=g[BASE]["max_drawdown"]+TOL and g[n]["longest_flat_full_sessions"]>=2]
    tcn.write_json(out/"formal_checks.json",{"historical_winner":selected,"historical_winner_passed":selected in passed,
                                             "passes_both_screens":passed})
    m.write_csv(out/"comparison.csv",[s for g in tables.values() for s in g.values()])
    m.write_csv(out/"years.csv",years)
    print("development",dev,"history",selected,"qualified",qualified,"both",passed)
    for n in (BASE,*CONFIGS):
        print(n,[(p,round(100*tables[p][n]["total_return"],2),round(100*tables[p][n]["max_drawdown"],2))
                 for p in ("early","history","course")])


if __name__=="__main__":
    main()
