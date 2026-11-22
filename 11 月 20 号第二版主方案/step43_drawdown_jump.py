"""Dissect an observed equity jump and test pre-close, staged fallback risk limits."""
import argparse
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from math import expm1, fsum, isclose, log1p
import json
import os
from pathlib import Path
import statistics

from step40_fallback_guard import BASE, PRIMARY, ROOT, PERIODS, archive_signals, extension, ledger, m, read, score, seq, tcn
from run_adaptive_consensus import rebuild
from threadpoolctl import threadpool_limits


OLD = "old20"
ARCHIVE = ROOT/"step41_weekly_guard"
RULES = {
    OLD: (.20, .50, .50, 1.00),
    "earlier": (.18, .50, .50, 1.00),
    "early_cap": (.18, .50, .50, .75),
    "two_stage": (.18, .50, .75, 1.00),
    "two_stage_cap": (.18, .50, .75, .75),
    "handover": (.18, .50, 1.00, 1.00),
    "handover_cap": (.18, .50, 1.00, .75),
}

# After comparing full exposure and the 75% cap, test their midpoint.
MID = "handover_midcap"
RULES[MID] = (.18, .50, 1.00, .875)
for stem in ("earlier","two_stage"):
    RULES[stem+"_midcap"] = (*RULES[stem][:3], .875)
FOLLOWUPS = (MID, "earlier_midcap", "two_stage_midcap", "staged_entry", "volatility_cap")
for variant in ("staged_entry", "volatility_cap"):
    RULES[variant] = RULES["handover"]

def policies(rows, records, names):
    forecasts = [{**r, "strategy_weight":r["ridge"]} for r in records]
    shadow = seq.replay_engine.replay(rows, forecasts, {"name":"ridge","kind":"forecast_weight"},
                                      rebalance_on_weight_change_only=True, liquidate_at_end=False)
    seq.verify_consensus_ledger(rows, forecasts, shadow, liquidate_at_end=False)
    equity = [s["equity"] for s in shadow["curve"]]
    by_date = {str(bar["date"]):i for i,bar in enumerate(rows)}
    changes = [0.] + [b["close"]/a["close"]-1 for a,b in zip(rows,rows[1:])]
    result = {}
    for name in names:
        trigger, first, second, cap = RULES[name]
        active = deep = False
        result[name] = []
        for i, r in enumerate(records):
            # Mark i precedes target open i; include the initial cash mark for warmup.
            loss = 1-equity[i]/max(equity[max(0,i-252):i+1])
            active = loss >= (trigger/2 if active else trigger)
            deep = loss >= (.10 if deep else .20)
            enabled = active and r["switched"]
            w = r["strategy_weight"]
            share = 0.
            if enabled:
                share = second if deep else first
                w = min(cap, (1-share)*r["ridge"]+share*r[BASE])
                if name == "staged_entry":
                    prior = result[name][-1]["strategy_weight"] if result[name] else 0.
                    w = min(w, prior+.50)
                if name == "volatility_cap":
                    end = by_date[r["asof_date"]] + 1
                    if end > 252:
                        month = statistics.stdev(changes[end-20:end])
                        year = statistics.stdev(changes[end-252:end])
                        if month > year:
                            w = min(w, year/month)
            result[name].append({**r,"strategy_weight":w,"ridge_drawdown_asof":loss,
                                 "guard_active":enabled,"share_consensus":share})
    return result


def attribution(rows, orders, run):
    prices = {str(r["date"]):r for r in rows}
    fills = defaultdict(list)
    for t in run["trades"]:
        fills[t["date"]].append(t)
    output = []
    for s, prev, mark in zip(orders,run["curve"][:-1],run["curve"][1:],strict=True):
        day, asof = s["target_date"], s["asof_date"]
        p = prices[day]
        prev_close = prices[asof]["close"]
        old = prev["units"]
        after_open = old
        for t in fills[day]:
            if t["time"]=="09:30":
                assert t["signal_asof"]==asof and asof<day and t["price"]==p["open"]
                after_open += t["units"] if t["side"]=="BUY" else -t["units"]
        gap = old*(p["open"]-prev_close)
        intraday = after_open*(p["close"]-p["open"])
        fee = fsum(t["fee"] for t in fills[day])
        net = gap+intraday-fee
        if not isclose(net,mark["equity"]-prev["equity"],rel_tol=1e-10,abs_tol=1e-7):
            raise ValueError(f"Overnight + intraday - fees differs from net P&L on {day}")
        output.append(dict(date=day,signal_asof=asof,prev_close=prev_close,open=p["open"],close=p["close"],
                           index_return=p["close"]/prev_close-1,prior_close_weight=prev["close_weight"],
                           after_open_weight=mark["open_weight_after_trade"],overnight_pnl=gap,intraday_pnl=intraday,
                           fees=fee,net_pnl=net,account_return=mark["daily_return"],equity=mark["equity"],
                           drawdown=mark["drawdown"],guard_active=s["guard_active"]))
    return output


def inspect_run(rows, records, name, out):
    first,last=PERIODS["history"]
    orders=[r for r in records if first<=r["target_date"]<=last]
    s,run=ledger(rows,name,orders)
    data=attribution(rows,orders,run)
    m.write_csv(out/f"{name}_attribution.csv",data)
    if name!=OLD:
        m.write_csv(out/f"{name}_history_signals.csv",orders)
        m.write_csv(out/f"{name}_history_trades.csv",run["trades"])
    trough=min(range(len(run["curve"])),key=lambda i:run["curve"][i]["drawdown"])
    peak=max(range(trough+1),key=lambda i:run["curve"][i]["equity"])
    strip={"2024-09-30","2024-10-08"}
    return {**s,"drawdown_peak":run["curve"][peak],"drawdown_trough":run["curve"][trough],
            "net_return_omitting_two_recorded_days":expm1(fsum(log1p(r["account_return"]) for r in data if r["date"] not in strip)),
            "oct8":next(r for r in data if r["date"]=="2024-10-08"),
            "sep30":next(r for r in data if r["date"]=="2024-09-30"),
            "two_session_compound":expm1(fsum(log1p(r["account_return"]) for r in data if r["date"] in strip))}

def plot_audit(out, recommendation):
    os.environ.setdefault("MPLCONFIGDIR", str(m.ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import dates, font_manager, pyplot as plt
    from matplotlib.ticker import FuncFormatter

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False, "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.edgecolor":"#aab5bd"})
    fig, (depth, zoom) = plt.subplots(1, 2, figsize=(10.6, 4.1), layout="constrained")
    traces = [(OLD, "原保护规则", "#596c89", "--", "s"), (recommendation, "分档保护并限仓", "#007567", "-", "o")]
    for name, label, color, style, marker in traces:
        daily = read(out / f"{name}_attribution.csv")
        dates_all = [datetime.fromisoformat(row["date"]) for row in daily]
        losses = [100 * float(row["drawdown"]) for row in daily]
        depth.plot(dates_all, losses, label=label, color=color, ls=style, lw=1.3)
        trough = losses.index(min(losses))
        depth.scatter(dates_all[trough], losses[trough], color=color, marker=marker, s=23, zorder=4)
        right = name == OLD
        depth.annotate(f"{losses[trough]:.2f}%", (dates_all[trough], losses[trough]),
                       xytext=((26 if right else -27), (-7 if right else -12)),
                       textcoords="offset points", ha="left" if right else "right",
                       fontsize=9, color=color, arrowprops={"arrowstyle":"-", "color":color,"lw":.7},
                       bbox={"facecolor":"white","edgecolor":"none","pad":.4,"alpha":.90})
        event = [row for row in daily if "2024-09-20" <= row["date"] <= "2024-10-10"]
        x = range(len(event))
        if name == OLD:
            index = [float(row["close"])/float(event[0]["close"])*100 for row in event]
            zoom.plot(x, index, color="#b37422", marker="^", markersize=4, lw=1.5, label="创业板指")
        nav = [float(row["equity"])/float(event[0]["equity"])*100 for row in event]
        zoom.plot(x, nav, color=color, ls=style, marker=marker, markerfacecolor="white" if right else color,
                  markersize=3.7, lw=1.6, label=label)
        if name == recommendation:
            shade_label = "新版保护生效期"
            for active, group in groupby(daily, key=lambda row:row["guard_active"]=="True"):
                if active:
                    days = [datetime.fromisoformat(row["date"]) for row in group]
                    depth.axvspan(days[0], days[-1], color="#d8943a", alpha=.14, lw=0, label=shade_label)
                    shade_label = None

    depth.set(title="2023 年至训练截止日的每日回撤", ylabel="从账户此前最高收盘净值跌了多少")
    depth.set_ylim(-28, 1.5)
    depth.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    depth.xaxis.set_major_locator(dates.MonthLocator(bymonth=(1,7)))
    depth.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    zoom.set(title="放大 2024 年国庆前后的行情", ylabel="9 月 20 日收盘 = 100", xlabel="相邻两点相隔一个交易日")
    tick_indices = [i for i,r in enumerate(event) if r["date"][-2:] in ("20","24","26","30","08","10")]
    zoom.set_xticks(tick_indices, [event[i]["date"][5:].replace("-","/") for i in tick_indices])
    for ax in (depth, zoom):
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, fontsize=9, loc="lower left" if ax==depth else "upper left")
        ax.tick_params(labelsize=9)
    fig.savefig(out/"drawdown_and_rally.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"step43_drawdown_jump")
    parser.add_argument("--retrain",action="store_true")
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    prices,meta=m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    if args.retrain:
        locked=json.loads((out/"selection.json").read_text())
        selected=locked["selected"]
        recommended=locked["recommended"]
        rows=extension(prices)
        tcn.torch.set_num_threads(2)
        tcn.torch.use_deterministic_algorithms(True)
        records=rebuild(rows,rows[-1]["date"])[PRIMARY]
        names=tuple(dict.fromkeys((OLD,selected,recommended,*FOLLOWUPS)))
    else:
        tcn.write_json(out/"protocol.json",{
            "input_sha256":meta["sha256"],"baseline":OLD,"fees_each_side":m.FEE_RATE,"grid":RULES,
            "rules":"First threshold and half-threshold release, more protection at 20% until below 10%. "
                    "Numbers in grid are first trigger, first and deep consensus share, target cap while active. "
                    "Use previous-close Ridge shadow equity, during existing fallback only, next open.",
            "selection":"Lock the lowest 2023-24 drawdown among variants whose early daily equity and trades exactly "
                        "reproduce old20 and whose 2023-24 net and double-fee return are no lower than old20. "
                        "Tie: prefer higher net return. Verify 2025, combined period, coursework and forward after lock.",
            "midpoint_check":"The 87.5% cap is a follow-up sensitivity added after seeing the original cap endpoints. "
                             "It is excluded from the initial guard selection.",
            "followup":"Two more exploratory alternatives aim to reduce jump concentration: add at most 50 percentage "
                       "points to target exposure per session, or cap exposure at trailing 252-session volatility "
                       "divided by 20-session volatility. Apply only while protection is enabled.",
            "concentration_check":"Check the 87.5% midpoint across all three depths. Among all rules that match early "
                                  "weights, preserve 2023-24 returns at standard and double fees, and reduce drawdown, "
                                  "choose the smallest drawdown with no increase in maximum rolling two-session gain.",
            "post_hoc":"All windows previously viewed. The 18% threshold showed promise in the previous sensitivity run. "
                       "The return audit uses no alteration of actual returns. Removing extreme days is attribution only."})
        rows=prices
        records=archive_signals(prices)
        names=tuple(RULES)
    choices=policies(rows,records,names)
    cut="2024-12-31"
    prefix=[r for r in records if r["target_date"]<=cut]
    prefix_prices=[r for r in rows if str(r["date"])<=cut]
    prefix_prices[-1]={**prefix_prices[-2],"date":prefix_prices[-1]["date"]}
    prefix_orders=policies(prefix_prices,prefix,names)
    assert all(v==choices[n][:len(v)] for n,v in prefix_orders.items())
    scores={}
    yearly=[]
    for p in ("early","development","check_2025","history","course","forward"):
        if p=="forward" and not args.retrain:
            continue
        scores[p]=score(rows,choices,p,yearly)
        if p=="development" and not args.retrain:
            g=scores[p]
            early_ref=[r["strategy_weight"] for r in choices[OLD] if PERIODS["early"][0]<=r["target_date"]<=PERIODS["early"][1]]
            qualifying=[n for n in names if n!=OLD and
                    early_ref==[r["strategy_weight"] for r in choices[n] if PERIODS["early"][0]<=r["target_date"]<=PERIODS["early"][1]]
                    and g[n]["total_return"]>=g[OLD]["total_return"]-1e-10
                    and g[n]["double_fee_return"]>=g[OLD]["double_fee_return"]-1e-10
                    and g[n]["max_drawdown"]<g[OLD]["max_drawdown"]-1e-10]
            passed=[n for n in qualifying if n not in FOLLOWUPS]
            ranking=lambda n:(g[n]["max_drawdown"],-g[n]["total_return"])
            selected=min(passed,key=ranking,default=OLD)
            max_rise={}
            first,last=PERIODS["development"]
            for n in [OLD,*qualifying]:
                orders=[r for r in choices[n] if first<=r["target_date"]<=last]
                _,run=ledger(rows,n,orders)
                marks=run["curve"]
                max_rise[n]=max(b["equity"]/a["equity"]-1 for a,b in zip(marks,marks[2:]))
            smoother=[n for n in qualifying if max_rise[n]<=max_rise[OLD]+1e-10]
            recommended=min(smoother,key=ranking,default=OLD)
            tcn.write_json(out/"selection.json",{"selected":selected,"passed":passed,"recommended":recommended,
                                                 "max_rolling_two_day_gain":max_rise,"smoother":smoother})
    if args.retrain:
        old={(r["name"],r["phase"]):r for r in read(out/"comparison.csv")}
        for p,group in scores.items():
            for n,s in group.items():
                if p!="forward":
                    for k in ("total_return","max_drawdown","order_count","fees_paid"):
                        assert isclose(s[k],float(old[n,p][k]),abs_tol=1e-8)
                first,last=PERIODS[p]
                if p in ("early","course","forward") or p=="history" and n==OLD:
                    orders=[r for r in choices[n] if first<=r["target_date"]<=last]
                    _,run=ledger(rows,n,orders)
                    for field,suffix in (("curve","daily"),("trades","trades")):
                        actual=[{k:str(v) for k,v in row.items()} for row in run[field]]
                        assert actual==read(ARCHIVE/f"{p}_daily_loss_{suffix}.csv")
    diagnostics={n:inspect_run(rows,choices[n],n,out) for n in dict.fromkeys((OLD,selected,recommended,*FOLLOWUPS))}
    tcn.write_json(out/"jump_audit.json",diagnostics)
    plot_audit(out,recommended)
    m.write_csv(out/("retrained.csv" if args.retrain else "comparison.csv"),[s for g in scores.values() for s in g.values()])
    for n in names:
        print(n,[(p,round(100*g[n]["total_return"],3),round(100*g[n]["max_drawdown"],3)) for p,g in scores.items()])
    print("Initial guard:",selected,"/ With jump constraint:",recommended)


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
