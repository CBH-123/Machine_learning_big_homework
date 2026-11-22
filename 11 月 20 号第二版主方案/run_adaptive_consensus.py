"""从指数原始行情重训双模型和备用模型，重放月度切换策略。"""
import argparse
import csv
import os
from datetime import date
from math import expm1, fsum, log1p
from pathlib import Path

from step39_consensus_evidence import BASE, m, plans_for, ROOT, seq, tcn
from step38_history_repair import panel, ledger
from step14_consensus import consensus_decisions
from step7_net_return import weighted_forecasts
from step1_baseline import buy_and_hold
from threadpoolctl import threadpool_limits


PRIMARY = "two_year_evidence"
BALANCED = "balanced_fallback"


def rebuild(prices, end):
    blocks = [("warm",date(2016,1,1),date(2017,12,31)),("early",date(2018,1,1),date(2022,12,31)),
              ("history",m.VALID_START,date(2025,10,9)),("course",m.START,end)]
    accumulated = []
    for phase,first,last in blocks:
        rows = [r for r in prices if r["date"]<=last]
        forecasts,_,_ = tcn.rolling(rows, first, last, tcn.torch.device("cpu"))
        # Keep the archived rolling fit calendar for the 2023 validation block.
        utility_first = date(2019,1,1) if phase=="history" else first
        utility = [s for s in seq.utility_decisions(rows, utility_first, last) if s["target_date"]>=str(first)]
        raw = consensus_decisions(weighted_forecasts(forecasts,"buffer"),utility)
        records = panel(rows,phase,raw=raw)
        for rec,fc in zip(records,forecasts,strict=True):
            rec["predicted_close"]=fc["predicted_close"]
        accumulated.extend(records)
    plans = plans_for(prices,accumulated,(BASE,PRIMARY))
    plans[BALANCED]=[{**r,"strategy_weight":(r[BASE]+r["ridge"])/2 if r["switched"] else r[BASE],
                      "ridge_allocation":.5 if r["switched"] else 0.} for r in plans[PRIMARY]]
    return plans


def plot_history(out):
    """Three separate accounts, each rebased to the same initial capital."""
    os.environ.setdefault("MPLCONFIGDIR",str(m.ROOT/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import dates as md, font_manager, pyplot as plt
    from datetime import datetime
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,3,figsize=(11,3.9),sharey=True)
    curves=((PRIMARY,"切换至备用模型","#007969"),(BALANCED,"备用时各占一半","#907249"),(BASE,"原双模型","#5B7190"))
    for ax,(period,title) in zip(axes,(("early","2018–2022 年"),("history","2023 年至训练截止日"),("course","作业考核区间"))):
        ax.axhline(10,linewidth=.7,color="#ABB2BA")
        for name,label,color in curves:
            with (out/f"{period}_{name}_daily.csv").open(encoding="utf-8-sig") as f:
                marks=list(csv.DictReader(f))
            days=[datetime.fromisoformat(r["timestamp"]) for r in marks]
            values=[float(r["equity"])/10000 for r in marks]
            ax.plot(days,values,label=label,color=color,linewidth=1.9 if name==PRIMARY else 1.1,
                    linestyle="--" if name==BASE else "-",alpha=.95)
        ax.set_title(title,fontsize=11)
        tick=md.AutoDateLocator(minticks=3,maxticks=4)
        ax.xaxis.set_major_locator(tick)
        ax.xaxis.set_major_formatter(md.ConciseDateFormatter(tick))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("账户资产（万元）")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.91))
    fig.suptitle("同一切换规则在三个区间的扣费表现",y=.985,fontsize=15)
    fig.subplots_adjust(top=.73,bottom=.16,left=.07,right=.98,wspace=.14)
    fig.savefig(out/"history_comparison.png",dpi=160,facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"adaptive_consensus_from_prices")
    parser.add_argument("--include-extension",action="store_true",help="额外评估项目已存的考核后15个交易日")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    prices,meta = m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    periods=[("early","2018-01-01","2022-12-31"),("history","2023-01-01","2025-10-09"),
             ("course",str(m.START),str(m.END))]
    end = m.END
    if args.include_extension:
        from step33_exit_timing import forward,read
        last,=read(ROOT/"step17/new_session.csv")
        extra=forward.read_external()+[{k:date.fromisoformat(v) if k=="date" else float(v) for k,v in last.items()}]
        if extra[0]["date"]<=prices[-1]["date"]:
            raise ValueError("External price extension overlaps the teacher's file")
        prices=prices+extra
        end=prices[-1]["date"]
        periods.append(("forward",str(extra[0]["date"]),str(end)))
    prices=[r for r in prices if r["date"]<=end]
    tcn.torch.use_deterministic_algorithms(True)
    tcn.torch.set_num_threads(2)
    plans=rebuild(prices,end)
    stats=[]
    annual=[]
    for phase,first,last in periods:
        bars = [r for r in prices if first<=str(r["date"])<=last]
        benchmark = buy_and_hold(bars,fee_rate=m.FEE_RATE)[1]
        for name,all_signals in plans.items():
            orders=[r for r in all_signals if first<=r["target_date"]<=last]
            if [r["target_date"] for r in orders]!=[str(b["date"]) for b in bars]:
                raise ValueError("Incomplete target calendar")
            s,run=ledger(prices,name,orders)
            double=seq.replay_engine.replay(prices,orders,{"name":name,"kind":"forecast_weight"},
                                            fee_rate=2*m.FEE_RATE,rebalance_on_weight_change_only=True)
            seq.verify_consensus_ledger(prices,orders,double)
            s.update(phase=phase, double_fee_return=double["metrics"]["total_return"],
                     buy_hold_return=benchmark["total_return"],buy_hold_drawdown=benchmark["max_drawdown"],
                     switched_sessions=sum(r["switched"] for r in orders),
                     changed_sessions=sum(r["strategy_weight"]!=r[BASE] for r in orders))
            stats.append(s)
            for year in sorted({r["target_date"][:4] for r in orders}):
                marks=[r for r in run["curve"][1:] if r["timestamp"].startswith(year)]
                annual.append(dict(phase=phase,name=name,year=year,
                                   net_return=expm1(fsum(log1p(r["daily_return"]) for r in marks))))
            for suffix,table in (("daily",run["curve"]),("trades",run["trades"]),("signals",orders)):
                m.write_csv(args.out/f"{phase}_{name}_{suffix}.csv",table)
    m.write_csv(args.out/"metrics.csv",stats)
    m.write_csv(args.out/"annual_returns.csv",annual)
    plot_history(args.out)
    tcn.write_json(args.out/"protocol.json",{
        "input_sha256":meta["sha256"],"fee_each_side":m.FEE_RATE,"forward_included":args.include_extension,
        "primary":"Use consensus if its 504-session net log return exceeds one sample daily standard deviation "
                  "times sqrt(504); otherwise follow open-only Ridge. Review at prior close for first open of month.",
        "balanced":"Same gate; when evidence is insufficient, use equal target weights across the two strategies.",
        "note":"The balanced version is a follow-up allocation sensitivity. Historical and course periods previously examined."
    })
    for r in stats:
        print(r["phase"],r["name"],f"{r['total_return']:.2%}",f"{r['max_drawdown']:.2%}",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
