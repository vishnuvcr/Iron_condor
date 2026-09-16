from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(Path("config.yaml").read_text())
CAPITAL = float(CFG["capital"])


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"): return 25
    if e < pd.Timestamp("2021-08-01"): return 75
    if e < pd.Timestamp("2024-05-02"): return 50
    if e < pd.Timestamp("2024-11-20"): return 25
    if e < pd.Timestamp("2026-01-06"): return 75
    return 65


class Market:
    def __init__(self, options, futures):
        r = options.copy()
        r["date"] = pd.to_datetime(r.date).dt.normalize(); r["expiry"] = pd.to_datetime(r.expiry).dt.normalize()
        r["option_type"] = r.option_type.astype(str).str.upper(); r = r.drop_duplicates(["date","expiry","strike","option_type"])
        f = futures.copy(); f["date"] = pd.to_datetime(f.date).dt.normalize(); f["expiry"] = pd.to_datetime(f.expiry).dt.normalize()
        # Underlying = nearest non-expired NIFTY index future by expiry.
        f = f.sort_values(["date","expiry"]).drop_duplicates(["date","expiry"])
        spot = f[f.expiry >= f.date].sort_values(["date","expiry"]).groupby("date", as_index=False).first()[["date","close"]]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(r.date) & set(spot.date)))
        self.expiries = {}
        self.strikes = {}
        self.price = {}
        self.high = {}
        self.low = {}
        for (d,e), g in r.groupby(["date","expiry"]):
            d,e = pd.Timestamp(d),pd.Timestamp(e)
            if d not in self.spot: continue
            self.expiries.setdefault(d, []).append(e)
            self.strikes[(d,e)] = np.asarray(sorted(g.strike.astype(float).unique()))
        for d,e,k,t,close,high,low in r[["date","expiry","strike","option_type","close","high","low"]].itertuples(index=False, name=None):
            key=(pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())
            if pd.notna(close): self.price[key]=float(close)
            if pd.notna(high): self.high[key]=float(high)
            if pd.notna(low): self.low[key]=float(low)
        for d in self.expiries: self.expiries[d]=tuple(sorted(self.expiries[d]))

    def p(self,d,e,k,t): return self.price.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t))
    def h(self,d,e,k,t): return self.high.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t))
    def l(self,d,e,k,t): return self.low.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t))


def choose(m,d,e,distance,width):
    spot=m.spot.get(pd.Timestamp(d)); strikes=m.strikes.get((pd.Timestamp(d),pd.Timestamp(e)))
    if spot is None or strikes is None or len(strikes)<8: return None
    puts=strikes[strikes <= spot*(1-distance)]; calls=strikes[strikes >= spot*(1+distance)]
    if not len(puts) or not len(calls): return None
    ps,cs=float(puts[-1]),float(calls[0])
    pls=strikes[strikes <= ps-width+1e-9]; cls=strikes[strikes >= cs+width-1e-9]
    if not len(pls) or not len(cls): return None
    return float(pls[-1]),ps,cs,float(cls[0])


def costs(entry, exit_, lot):
    # Same modeled India F&O cost convention used in the research pipeline.
    brokerage=80.0
    sell_value=(entry[1]+entry[2]+exit_[0]+exit_[3])*lot
    buy_value=(entry[0]+entry[3]+exit_[1]+exit_[2])*lot
    turnover=sum(abs(a)+abs(b) for a,b in zip(entry,exit_))*lot
    txn=turnover*0.00035; stt=max(sell_value,0)*0.001; stamp=max(buy_value,0)*0.00003; sebi=turnover*0.000001
    gst=(brokerage+txn+sebi)*0.18
    return brokerage+stt+stamp+sebi+txn+gst


def backtest(m, distance, width, tp, sl, start=None, end=None):
    rows=[]; target_weekday=int(CFG["entry_weekday"]); min_dte=int(CFG["min_days_to_expiry"]); max_dte=int(CFG["max_days_to_expiry"])
    dates=[d for d in m.dates if d.weekday()==target_weekday and (start is None or d>=start) and (end is None or d<=end)]
    for d in dates:
        exp=next((e for e in m.expiries.get(d,()) if min_dte <= (e-d).days <= max_dte),None)
        if exp is None: continue
        legs=choose(m,d,exp,distance,width)
        if legs is None: continue
        strikes=[(legs[0],"PE"),(legs[1],"PE"),(legs[2],"CE"),(legs[3],"CE")]
        en=[m.p(d,exp,k,t) for k,t in strikes]
        if any(x is None or x<=0 for x in en): continue
        credit=en[1]+en[2]-en[0]-en[3]
        if credit < float(CFG.get("minimum_credit_points",0)): continue
        lot=lot_size(exp); max_loss=max(legs[1]-legs[0],legs[3]-legs[2])-credit
        if credit<=0 or max_loss<=0: continue
        exit_date, reason = None, "expiry"
        mark=None
        after=[x for x in m.dates if d<x<=exp]
        for day in after:
            vals=[m.p(day,exp,k,t) for k,t in strikes]
            if any(x is None for x in vals): continue
            mark=vals[1]+vals[2]-vals[0]-vals[3]
            pnl_pts=credit+mark
            if pnl_pts >= tp*credit: exit_date,reason=day,"take_profit"; break
            if pnl_pts <= -sl*credit: exit_date,reason=day,"stop_loss"; break
        if exit_date is None: exit_date=exp
        vals=[m.p(exit_date,exp,k,t) for k,t in strikes]
        if any(x is None for x in vals): continue
        gross=(credit+vals[1]+vals[2]-vals[0]-vals[3])*lot
        c=costs(en,vals,lot); net=gross-c
        rows.append({"entry":d,"exit":exit_date,"net":net,"lot":lot,"credit":credit,"reason":reason,"max_loss":max_loss*lot})
    return pd.DataFrame(rows)


def score(t):
    if t.empty: return -1e9
    weekly=t.groupby(t.exit.dt.to_period("W")).net.sum()/CAPITAL
    win=float((t.net>0).mean()); worst=float(weekly.min());
    return 2*float(weekly.median()) + float(weekly.mean()) + 0.10*float((weekly>=0.05).mean()) + 0.25*float((weekly>=0).mean()) + 0.10*win + 0.50*worst


def select(m):
    results=[]
    for d,w,tp,sl in itertools.product(CFG["short_distance_pct"],CFG["wing_width_points"],CFG["take_profit_credit_pct"],CFG["stop_loss_credit_multiple"]):
        t=backtest(m,float(d),int(w),float(tp),float(sl))
        results.append({"distance":d,"width":w,"take_profit":tp,"stop_loss":sl,"trades":len(t),"score":score(t),
                        "win_rate":float((t.net>0).mean()) if not t.empty else 0.0,
                        "total_return":float(t.net.sum()/CAPITAL) if not t.empty else 0.0,
                        "median_trade":float(t.net.median()) if not t.empty else 0.0,
                        "worst_trade":float(t.net.min()) if not t.empty else 0.0})
    df=pd.DataFrame(results).sort_values(["score","total_return"],ascending=False).reset_index(drop=True)
    return df.iloc[0].to_dict(),df


def next_trading_date(d,m):
    return next((x for x in m.dates if x>d), None)


def generate_call(m, selected):
    latest=max(m.dates); weekday=int(CFG["entry_weekday"])
    signal_date=latest if latest.weekday()==weekday else next((x for x in m.dates if x>latest and x.weekday()==weekday),None)
    if signal_date is None:
        return {"status":"NO_SIGNAL","reason":"Latest completed session is not an entry signal session and the next signal session is not in the current dataset.","latest_data_date":str(latest.date())}
    exp=next((e for e in m.expiries.get(signal_date,()) if int(CFG["min_days_to_expiry"]) <= (e-signal_date).days <= int(CFG["max_days_to_expiry"])),None)
    if exp is None: return {"status":"NO_SIGNAL","signal_date":str(signal_date.date()),"reason":"No valid expiry in configured DTE window."}
    legs=choose(m,signal_date,exp,float(selected["distance"]),int(selected["width"]))
    if legs is None: return {"status":"NO_SIGNAL","signal_date":str(signal_date.date()),"reason":"Insufficient strikes to construct four legs."}
    names=["put_long","put_short","call_short","call_long"]; types=["PE","PE","CE","CE"]
    premiums=[m.p(signal_date,exp,k,t) for k,t in zip(legs,types)]
    if any(x is None or x<=0 for x in premiums): return {"status":"NO_SIGNAL","signal_date":str(signal_date.date()),"reason":"Missing EOD premium for one or more legs."}
    credit=premiums[1]+premiums[2]-premiums[0]-premiums[3]
    lot=lot_size(exp); max_loss_points=max(legs[1]-legs[0],legs[3]-legs[2])-credit; max_loss=max_loss_points*lot
    paper_lots=min(int(CFG.get("max_paper_lots",1)), max(1,int(math.floor(CAPITAL/max_loss))) if max_loss>0 else 0)
    if paper_lots<1: paper_lots=1
    target_debit=credit*(1-float(selected["take_profit"]))
    stop_debit=credit*(1+float(selected["stop_loss"]))
    slippage=float(CFG.get("slippage_per_leg",0.5)); effective_credit=credit-4*slippage
    proposed_lots=paper_lots
    # Costs shown on a paper position use the reference premiums; actual contract-note costs should override them.
    approx_cost=costs(premiums,premiums,lot*proposed_lots)
    target_pnl=(credit-target_debit)*lot*proposed_lots-approx_cost
    stop_pnl=(credit-stop_debit)*lot*proposed_lots-approx_cost
    entry_date=next_trading_date(signal_date,m)
    out={
        "status":"PENDING_ENTRY","signal_id":f"IC-{signal_date:%Y%m%d}-{exp:%Y%m%d}-{int(legs[1])}-{int(legs[2])}",
        "generated_utc":pd.Timestamp.utcnow().isoformat(),"signal_date":str(signal_date.date()),"planned_entry_date":str(entry_date.date()) if entry_date is not None else None,
        "expiry":str(exp.date()),"dte_at_signal":int((exp-signal_date).days),"nifty_reference":float(m.spot[signal_date]),
        "strategy":"Success V1 delayed-entry NIFTY weekly iron condor","entry_rule":"Signal after Tuesday close; paper entry on next eligible NSE session",
        "distance_pct":float(selected["distance"]),"wing_width_points":int(selected["width"]),"take_profit_credit_pct":float(selected["take_profit"]),"stop_loss_credit_multiple":float(selected["stop_loss"]),
        "legs": {n:{"strike":float(k),"type":t,"reference_eod_premium":float(p)} for n,k,t,p in zip(names,legs,types,premiums)},
        "reference_credit_points":float(credit),"reference_credit_value_per_lot":float(credit*lot),"effective_credit_after_configured_slippage":float(effective_credit),
        "breakeven_lower":float(legs[1]-credit),"breakeven_upper":float(legs[2]+credit),
        "lot_size":int(lot),"suggested_paper_lots":int(proposed_lots),"premium_cash_per_lot":float(credit*lot),"premium_cash_total":float(credit*lot*proposed_lots),
        "max_expiry_loss_points":float(max_loss_points),"max_expiry_loss_per_lot":float(max_loss),"max_expiry_loss_total":float(max_loss*proposed_lots),
        "take_profit_debit":float(target_debit),"stop_loss_debit":float(stop_debit),"target_pnl_total_reference":float(target_pnl),"stop_pnl_total_reference":float(stop_pnl),
        "modeled_round_trip_cost_reference":float(approx_cost),"configured_slippage_per_leg":slippage,
        "capital":CAPITAL,"capital_remaining_after_max_loss_reference":float(CAPITAL-max_loss*proposed_lots),
        "execution_note":"EOD premiums are reference values, not guaranteed fills. Record actual fills, timestamp, and broker contract-note costs in the paper ledger.",
    }
    return out


def main():
    options=pd.read_parquet("data/cache/nifty_options.parquet"); futures=pd.read_parquet("data/cache/nifty_futures.parquet")
    m=Market(options,futures)
    selected, leaderboard=select(m)
    Path("data/research").mkdir(parents=True,exist_ok=True); leaderboard.to_csv("data/research/parameter_leaderboard.csv",index=False)
    Path("data/research/selected_parameters.json").write_text(json.dumps(selected,indent=2,default=str))
    call=generate_call(m,selected); Path("signals").mkdir(exist_ok=True)
    Path("signals/latest_trade_call.json").write_text(json.dumps(call,indent=2,default=str))
    md=[f"# NIFTY Iron Condor — {call.get('status')}","",f"Generated: {call.get('generated_utc','')}",""]
    if call.get("status")!="PENDING_ENTRY":
        md += [f"**Reason:** {call.get('reason','')}"]
    else:
        md += [f"**Signal ID:** `{call['signal_id']}`",f"**Signal session:** {call['signal_date']}",f"**Planned paper entry:** {call['planned_entry_date']}",f"**Expiry:** {call['expiry']} ({call['dte_at_signal']} DTE)",f"**NIFTY reference:** {call['nifty_reference']:.2f}","", "## Spread"]
        for n,v in call["legs"].items(): md.append(f"- {n}: **{v['strike']:.0f} {v['type']}**, reference premium {v['reference_eod_premium']:.2f}")
        md += [f"- Credit: **{call['reference_credit_points']:.2f} points**",f"- Breakevens: **{call['breakeven_lower']:.2f} / {call['breakeven_upper']:.2f}**",f"- Lot size: **{call['lot_size']}**",f"- Suggested paper lots: **{call['suggested_paper_lots']}**",f"- Premium received (reference): **₹{call['premium_cash_total']:,.2f}**",f"- Maximum expiry loss (reference): **₹{call['max_expiry_loss_total']:,.2f}**",f"- Target debit: **{call['take_profit_debit']:.2f}**",f"- Stop debit: **{call['stop_loss_debit']:.2f}**",f"- Target P&L (reference): **₹{call['target_pnl_total_reference']:,.2f}**",f"- Stop P&L (reference): **₹{call['stop_pnl_total_reference']:,.2f}**",f"- Modeled costs: **₹{call['modeled_round_trip_cost_reference']:,.2f}**", "", "## Execution",f"Configured slippage stress: **₹{call['configured_slippage_per_leg']:.2f}/leg**", "Do not treat reference EOD premiums as guaranteed fills. Record actual paper fills and contract-note charges."]
    Path("signals/latest_trade_call.md").write_text("\n".join(md)+"\n")
    if call.get("status")=="PENDING_ENTRY": Path("signals/open_trade.json").write_text(json.dumps(call,indent=2,default=str))
    print(json.dumps(call,indent=2,default=str))

if __name__ == "__main__": main()
