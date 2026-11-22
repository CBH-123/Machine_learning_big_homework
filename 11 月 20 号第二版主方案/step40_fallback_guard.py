"""Protect the fallback sleeve while retaining the existing monthly consensus gate."""
import argparse
from datetime import date
from math import expm1, fsum, isclose, log1p
from pathlib import Path

from run_adaptive_consensus import PRIMARY, BASE, rebuild
from step38_history_repair import m, ROOT, panel, ledger, seq, tcn, read
from step39_consensus_evidence import plans_for
from step33_exit_timing import forward
from threadpoolctl import threadpool_limits


ARCHIVE = ROOT / "adaptive_consensus_from_prices"
TRIGGERS = ("loss10", "loss20", "past_stress", "negative_year")
RULES = {f"{trigger}_{action}":(trigger,action) for trigger in TRIGGERS for action in ("blend","cash")}
PERIODS = {
    "early":("2018-01-01","2022-12-31"), "development":("2023-01-01","2024-12-31"),
    "check_2025":("2025-01-01","2025-10-09"), "history":("2023-01-01","2025-10-09"),
    "course":("2025-10-10","2026-10-23"), "forward":("2026-10-28","2026-11-17"),
}
EPS = 1e-10


def archive_signals(rows):
    warm = [r for r in rows if str(r["date"])<"2018-01-01"]
    records = plans_for(warm,panel(warm,"warm"),(PRIMARY,))[PRIMARY]
    for phase in ("early","history","course"):
        for rec in read(ARCHIVE/f"{phase}_{PRIMARY}_signals.csv"):
            records.append({**rec,BASE:float(rec[BASE]),"ridge":float(rec["ridge"]),
                            "switched":rec["switched"]=="True","strategy_weight":float(rec["strategy_weight"])})
    return records


def guard(rows, records, names, *, deep_loss=.20, review_offset=0):
    order_dates = [r["target_date"] for r in records]
    expected=[str(b["date"]) for b in rows if order_dates[0]<=str(b["date"])<=order_dates[-1]]
    if order_dates!=expected or any(a["target_date"]!=b["asof_date"] for a,b in zip(records,records[1:])):
        raise ValueError("Incomplete trading calendar")
    shadow_orders=[{**r,"strategy_weight":r["ridge"]} for r in records]
    run=seq.replay_engine.replay(rows,shadow_orders,{"name":"ridge","kind":"forecast_weight"},
                                 rebalance_on_weight_change_only=True,liquidate_at_end=False)
    seq.verify_consensus_ledger(rows,shadow_orders,run,liquidate_at_end=False)
    marks=run["curve"]
    states={t:False for t in TRIGGERS}
    rolling_drawdowns=[]
    equities=[s["equity"] for s in marks]
    output={n:[] for n in names}
    review=records[0]["asof_date"]
    for i,r in enumerate(records):
        # Index zero is initial cash. Mark i is already known at the close preceding target i.
        eq=equities[i]
        dd=1-eq/max(equities[max(0,i-252):i+1])
        rolling_drawdowns.append(dd)
        if i==0 or r["target_date"][:7]!=records[i-1]["target_date"][:7]:
            month_start=i
        if i-month_start==review_offset:
            review=r["asof_date"]
            for key,threshold in (("loss10",.10),("loss20",deep_loss)):
                states[key]=dd >= (threshold/2 if states[key] else threshold)
            states["negative_year"]=i>=252 and eq<equities[i-252]
            # Compare today's stress with the previous two years excluding the latest quarter.
            if i>=567:
                past=max(rolling_drawdowns[i-567:i-63])
                states["past_stress"]=dd >= (past/2 if states["past_stress"] else past) and past>0
        for n in names:
            active=False
            weight=r["strategy_weight"]
            if n!=PRIMARY:
                trigger,action=RULES[n]
                active=r["switched"] and states[trigger]
                if active:
                    weight=(weight+(r[BASE] if action=="blend" else 0.))/2
            output[n].append({**r,"strategy_weight":weight,"guard_active":active,
                              "guard_review":review,"prior_ridge_drawdown":dd})
    return output


def score(rows,plans,period,yearly):
    first,last=PERIODS[period]
    results={}
    for name,all_orders in plans.items():
        orders=[r for r in all_orders if first<=r["target_date"]<=last]
        s,run=ledger(rows,name,orders)
        double=seq.replay_engine.replay(rows,orders,{"name":name,"kind":"forecast_weight"},
                                        rebalance_on_weight_change_only=True,fee_rate=2*m.FEE_RATE)
        seq.verify_consensus_ledger(rows,orders,double)
        s.update(phase=period,double_fee_return=double["metrics"]["total_return"],
                 guarded=sum(r["guard_active"] for r in orders),
                 changed=sum(r["guard_active"] and r["strategy_weight"]!=r["ridge"] for r in orders))
        results[name]=s
        if period in ("history","early"):
            for year in sorted({r["target_date"][:4] for r in orders}):
                moves=[log1p(row["daily_return"]) for row in run["curve"][1:] if row["timestamp"].startswith(year)]
                yearly.append(dict(phase=period,name=name,year=year,total_return=expm1(fsum(moves))))
    return results


def extension(prices):
    last,=read(ROOT/"step17/new_session.csv")
    return prices+forward.read_external()+[{k:date.fromisoformat(v) if k=="date" else float(v) for k,v in last.items()}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"step40_fallback_guard")
    parser.add_argument("--retrain",action="store_true",help="Verify the locked rule from raw index prices, including the 15-session extension")
    args=parser.parse_args()
    out=args.out
    out.mkdir(exist_ok=True,parents=True)
    prices,meta=m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    if args.retrain:
        import json
        selected=json.loads((out/"choice_before_2025.json").read_text(encoding="utf-8"))["selected"]
        rows=extension(prices)
        tcn.torch.set_num_threads(2)
        tcn.torch.use_deterministic_algorithms(True)
        raw=rebuild(rows,rows[-1]["date"])[PRIMARY]
        plans=guard(rows,raw,(PRIMARY,selected))
        archived={(r["phase"],r["name"]):r for r in read(out/"comparison.csv")}
        reference={r["phase"]:r for r in read(ARCHIVE/"metrics.csv") if r["name"]==PRIMARY}
        results=[]
        for p,(first,last) in PERIODS.items():
            group=score(rows,plans,p,[])
            for n,s in group.items():
                for k in ("total_return","max_drawdown","fees_paid","order_count"):
                    if p!="forward" and not isclose(s[k],float(archived[p,n][k]),abs_tol=1e-8):
                        raise ValueError("Rebuilt model mismatch")
                    if n==PRIMARY and p in reference and not isclose(s[k],float(reference[p][k]),abs_tol=1e-8):
                        raise ValueError("Archived baseline mismatch")
                results.append(s)
            if p not in ("development","check_2025"):
                orders=[r for r in plans[selected] if first<=r["target_date"]<=last]
                _,run=ledger(rows,selected,orders)
                for suffix,table in (("signals",orders),("daily",run["curve"]),("trades",run["trades"])):
                    m.write_csv(out/f"{p}_{suffix}.csv",table)
                if p in ("early","course","forward") and group[selected]["changed"]==0:
                    assert group[selected]["changed"]==0
                    for suffix,new in (("daily",run["curve"]),("trades",run["trades"])):
                        assert read(out/f"{p}_{suffix}.csv")==read(ARCHIVE/f"{p}_{PRIMARY}_{suffix}.csv")
                if p in ("course","forward"):
                    assert group[selected]["changed"]==0
        m.write_csv(out/"retrained.csv",results)
        return
    names=(PRIMARY,*RULES)
    tcn.write_json(out/"protocol.json",{
        "baseline":PRIMARY,"source_sha256":meta["sha256"],"fees":m.FEE_RATE,"grid":RULES,
        "month_end":"Evaluate the previous close's net Ridge shadow equity against its trailing 252-session high. "
        "Trigger at 10%/20%, release at half. Alternatively breach the historical 504-session max of trailing "
        "drawdowns excluding the latest 63 sessions, or show a negative annual net return. Apply only during fallback.",
        "actions":"While active retain half the existing target and put the other half into cash or the consensus model.",
        "selection":"Require absolute early return difference <=2 percentage points and drawdown difference <=1 point, "
        "and higher net return and smaller drawdown on 2023-2024. Among passes rank by improvement in net return plus "
        "drawdown reduction. Keep the leading diagnostic by early-distance then development gain if none pass.",
        "validation":"Lock before evaluating 2025 (fresh capital) and the full recent period. The historical results, "
        "including the drawdown dates, and course results had already been viewed; this is exploratory.",
    })
    records=archive_signals(prices)
    cut="2024-12-31"
    early_records=[r for r in records if r["target_date"]<=cut]
    early_rows=[b for b in prices if str(b["date"])<=cut]
    pre=guard(early_rows,early_records,names)
    years=[]
    scores={p:score(early_rows,pre,p,years) for p in ("early","development")}
    early,dev=scores["early"],scores["development"]
    def rank(n):
        dr=abs(early[n]["total_return"]-early[PRIMARY]["total_return"])
        dd=abs(early[n]["max_drawdown"]-early[PRIMARY]["max_drawdown"])
        gain=dev[n]["total_return"]-dev[PRIMARY]["total_return"]
        reduction=dev[PRIMARY]["max_drawdown"]-dev[n]["max_drawdown"]
        return dr<=.02+EPS and dd<=.01+EPS and gain>EPS and reduction>EPS, -max(0,dr-.02)-max(0,dd-.01), gain+reduction
    selected=max(RULES,key=rank)
    tcn.write_json(out/"choice_before_2025.json",{"selected":selected,"development_pass":rank(selected)[0],
                                                 "passing":[n for n in RULES if rank(n)[0]]})
    plans=guard(prices,records,names)
    if any(plans[n][:len(seq0)]!=seq0 for n,seq0 in pre.items()):
        raise ValueError("Future observations changed earlier orders")
    for p in ("check_2025","history","course"):
        scores[p]=score(prices,plans,p,years)
    m.write_csv(out/"comparison.csv",[r for g in scores.values() for r in g.values()])
    m.write_csv(out/"years.csv",years)
    print("Selected",selected,"development pass",rank(selected)[0])
    for n in names:
        print(n,[(p,round(s[n]["total_return"]*100,2),round(s[n]["max_drawdown"]*100,2)) for p,s in scores.items()])


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
