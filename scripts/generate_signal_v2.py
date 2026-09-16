from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.v2.yaml").read_text())
CAPITAL = float(CFG["capital"])


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"): return 25
    if e < pd.Timestamp("2021-08-01"): return 75
    if e < pd.Timestamp("2024-05-02"): return 50
    if e < pd.Timestamp("2024-11-20"): return 25
    if e < pd.Timestamp("2026-01-06"): return 75
    return 65


def cost_model(entry, exit_px, lot):
    brokerage = 80.0
    sell_value = (entry[1] + entry[2] + exit_px[0] + exit_px[3]) * lot
    buy_value = (entry[0] + entry[3] + exit_px[1] + exit_px[2]) * lot
    turnover = sum(abs(a) + abs(b) for a, b in zip(entry, exit_px)) * lot
    txn = turnover * 0.00035; stt = max(sell_value, 0) * 0.0010; stamp = max(buy_value, 0) * 0.00003; sebi = turnover * 0.000001; gst = (brokerage + txn + sebi) * 0.18
    return brokerage + stt + stamp + sebi + txn + gst


class Market:
    def __init__(self, options, futures):
        o = options.copy(); o["date"] = pd.to_datetime(o.date).dt.normalize(); o["expiry"] = pd.to_datetime(o.expiry).dt.normalize(); o["option_type"] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(["date","expiry","strike","option_type"], keep="last")
        f = futures.copy(); f["date"] = pd.to_datetime(f.date).dt.normalize(); f["expiry"] = pd.to_datetime(f.expiry).dt.normalize(); f=f[f.expiry>=f.date].sort_values(["date","expiry"]).drop_duplicates(["date","expiry"],keep="last")
        spot=f.sort_values(["date","expiry"]).groupby("date",as_index=False).first()[["date","close"]]
        self.spot=dict(zip(spot.date,spot.close.astype(float))); self.dates=tuple(pd.Timestamp(x) for x in sorted(set(o.date)&set(spot.date)))
        self.expiries={}; self.strikes={}; self.px={}
        for (d,e),g in o.groupby(["date","expiry"],sort=False):
            d,e=pd.Timestamp(d),pd.Timestamp(e)
            if d in self.spot: self.expiries.setdefault(d,[]).append(e); self.strikes[(d,e)]=np.asarray(sorted(pd.to_numeric(g.strike,errors='coerce').dropna().unique()),dtype=float)
        for d in self.expiries: self.expiries[d]=tuple(sorted(self.expiries[d]))
        for d,e,k,t,c in o[["date","expiry","strike","option_type","close"]].itertuples(index=False,name=None): self.px[(pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())]=float(c) if pd.notna(c) else np.nan

    def price(self,d,e,k,t):
        v=self.px.get((pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())); return None if v is None or pd.isna(v) else float(v)
    def expiry_for(self,d):
        for e in self.expiries.get(pd.Timestamp(d),()):
            if int(CFG['min_days_to_expiry']) <= (e-pd.Timestamp(d)).days <= int(CFG['max_days_to_expiry']): return e
        return None
    def choose(self,d,e,distance,width):
        spot=self.spot.get(pd.Timestamp(d)); strikes=self.strikes.get((pd.Timestamp(d),pd.Timestamp(e)))
        if spot is None or strikes is None or len(strikes)<8: return None
        psx=strikes[strikes<=spot*(1-distance)]; csx=strikes[strikes>=spot*(1+distance)]
        if len(psx)==0 or len(csx)==0: return None
        ps,cs=float(psx[-1]),float(csx[0]); pls=strikes[strikes<=ps-width+1e-9]; cls=strikes[strikes>=cs+width-1e-9]
        if len(pls)==0 or len(cls)==0: return None
        return float(pls[-1]),ps,cs,float(cls[0])


def run_config(m,p,start,end,slip=None):
    slip=float(CFG['selection_slippage_per_leg'] if slip is None else slip); rows=[]; active_until=None
    dates=[d for d in m.dates if d.weekday()==int(p['entry_weekday']) and start<=d<=end]
    for signal_date in dates:
        if active_until is not None and signal_date<=active_until: continue
        exp=m.expiry_for(signal_date)
        if exp is None: continue
        legs=m.choose(signal_date,exp,float(p['distance']),int(p['width']))
        if legs is None: continue
        lk=[(legs[0],'PE'),(legs[1],'PE'),(legs[2],'CE'),(legs[3],'CE')]
        entry_date=next((d for d in m.dates if d>signal_date and d<=exp),None)
        if entry_date is None or entry_date>end: continue
        en=[m.price(entry_date,exp,k,t) for k,t in lk]
        if any(v is None or v<=0 for v in en): continue
        raw=en[1]+en[2]-en[0]-en[3]; eff=raw-4*slip; min_credit=float(p.get('minimum_credit',0))
        if raw<=0 or eff<min_credit: continue
        max_loss=max(legs[1]-legs[0],legs[3]-legs[2])-raw
        if max_loss<=0: continue
        exit_date=reason=exit_mark=None; exit_px=None
        for day in [d for d in m.dates if entry_date<d<=exp and d<=end]:
            vals=[m.price(day,exp,k,t) for k,t in lk]
            if any(v is None for v in vals): continue
            mark=vals[1]+vals[2]-vals[0]-vals[3]; debit=-mark+4*slip
            if debit <= (1-float(p['take_profit']))*eff: exit_date,reason,exit_mark,exit_px=day,'take_profit',mark,vals; break
            if debit >= (1+float(p['stop_loss']))*eff: exit_date,reason,exit_mark,exit_px=day,'stop_loss',mark,vals; break
        if exit_date is None:
            if exp>end: continue
            vals=[m.price(exp,exp,k,t) for k,t in lk]
            if any(v is None for v in vals): continue
            exit_date,reason,exit_mark,exit_px=exp,'expiry',vals[1]+vals[2]-vals[0]-vals[3],vals
        lot=lot_size(exp); gross=(eff+exit_mark-4*slip)*lot; charge=cost_model(en,exit_px,lot); net=gross-charge
        rows.append({'entry_date':entry_date,'exit_date':exit_date,'net_pnl':net,'reason':reason})
        active_until=pd.Timestamp(exit_date)
    return pd.DataFrame(rows)


def score(t):
    if t.empty: return -1e9
    w=t.groupby(t.exit_date.dt.to_period('W')).net_pnl.sum()/CAPITAL; eq=CAPITAL+t.net_pnl.cumsum(); dd=eq/eq.cummax()-1; win=float((t.net_pnl>0).mean())
    return 2*float(w.median())+float(w.mean())+0.10*float((w>=0.05).mean())+0.25*float((w>=0).mean())+0.10*win+0.50*float(w.min())+0.50*float(dd.min())


def select(m):
    base=[]
    for wd,d,w,tp,sl in itertools.product(CFG['entry_weekdays'],CFG['short_distance_pct'],CFG['wing_width_points'],CFG['take_profit_credit_pct'],CFG['stop_loss_credit_multiple']):
        p={'entry_weekday':wd,'distance':d,'width':w,'take_profit':tp,'stop_loss':sl,'minimum_credit':0.0}; t=run_config(m,p,m.dates[0],m.dates[-1])
        if len(t)>=40: base.append((score(t),p,t))
    base.sort(key=lambda x:x[0],reverse=True); top=base[:10]; cand=[]
    for _,p,_ in top:
        for mc in CFG['minimum_credit_points']:
            q={**p,'minimum_credit':mc}; t=run_config(m,q,m.dates[0],m.dates[-1])
            if len(t)>=40: cand.append((score(t),q,t))
    if not cand: raise RuntimeError('No V2 production configuration met the minimum training trade requirement')
    cand.sort(key=lambda x:x[0],reverse=True); return cand[0],base[:25]


def signal_from_selected(m,selected):
    latest=max(m.dates); expected_day=int(CFG['entry_weekdays'][0])
    # Never manufacture a future signal. Preserve the previous file on non-signal sessions.
    if latest.weekday()!=expected_day:
        return None
    p=selected; exp=m.expiry_for(latest)
    if exp is None: return None
    legs=m.choose(latest,exp,float(p['distance']),int(p['width']))
    if legs is None: return None
    types=['PE','PE','CE','CE']; names=['put_long','put_short','call_short','call_long']; prem=[m.price(latest,exp,k,t) for k,t in zip(legs,types)]
    if any(v is None or v<=0 for v in prem): return None
    credit=prem[1]+prem[2]-prem[0]-prem[3]; lot=lot_size(exp); max_loss=max(legs[1]-legs[0],legs[3]-legs[2])-credit
    if credit<=0 or max_loss<=0: return None
    entry_date=next((d for d in m.dates if d>latest and d<=exp),None); slippage=float(CFG['selection_slippage_per_leg']); eff=credit-4*slippage
    paper_lots=min(int(CFG['max_paper_lots']),max(1,int(math.floor(CAPITAL/(max_loss*lot)))))
    return {
      'status':'PENDING_ENTRY','signal_id':f"IC-V2-{latest:%Y%m%d}-{exp:%Y%m%d}-{int(legs[1])}-{int(legs[2])}",
      'strategy':'Iron Condor V2','generated_utc':pd.Timestamp.utcnow().isoformat(),'signal_date':str(latest.date()),'planned_entry_date':str(entry_date.date()) if entry_date is not None else None,'expiry':str(exp.date()),
      'dte_at_signal':int((exp-latest).days),'nifty_reference':float(m.spot[latest]),'entry_rule':'Thursday close signal; observe next-session EOD reference entry and accept only if effective credit meets the trained minimum-credit filter.',
      'entry_condition':f"Next-session EOD effective credit >= {p['minimum_credit']:.2f} points after {slippage:.2f}-point/leg modeled slippage.",
      'distance_pct':float(p['distance']),'wing_width_points':int(p['width']),'take_profit_credit_pct':float(p['take_profit']),'stop_loss_credit_multiple':float(p['stop_loss']),'minimum_credit_points':float(p['minimum_credit']),'configured_slippage_per_leg':slippage,
      'legs':{n:{'strike':float(k),'type':t,'reference_eod_premium':float(pr)} for n,k,t,pr in zip(names,legs,types,prem)},
      'reference_credit_points':float(credit),'effective_credit_if_same_reference':float(eff),'breakeven_lower':float(legs[1]-credit),'breakeven_upper':float(legs[2]+credit),
      'lot_size':int(lot),'suggested_paper_lots':int(paper_lots),'max_expiry_loss_points':float(max_loss),'max_expiry_loss_total':float(max_loss*lot*paper_lots),
      'take_profit_debit':float((1-float(p['take_profit']))*max(eff,0)),'stop_loss_debit':float((1+float(p['stop_loss']))*max(eff,0)),'capital':CAPITAL,
      'execution_note':'Reference prices are EOD observations. This call is a paper-research instruction, not a guaranteed executable fill.'
    }


def main():
    o=pd.read_parquet('data/cache/nifty_options.parquet'); f=pd.read_parquet('data/cache/nifty_futures.parquet'); m=Market(o,f)
    selected,top=select(m); score_,p,train=selected
    Path('v2/signal').mkdir(parents=True,exist_ok=True); Path('v2/data').mkdir(parents=True,exist_ok=True); Path('v2/research').mkdir(parents=True,exist_ok=True)
    model={'strategy':'Iron Condor V2','training_start':str(m.dates[0].date()),'training_end':str(m.dates[-1].date()),'selected_parameters':p,'selection_score':float(score_),'training_trades':int(len(train)),'training_win_rate':float((train.net_pnl>0).mean()),'training_net_pnl':float(train.net_pnl.sum()),'training_return':float(train.net_pnl.sum()/CAPITAL)}
    Path('v2/data/model_snapshot.json').write_text(json.dumps(model,indent=2,default=str)); pd.DataFrame(top,columns=['score','params','trades']).to_json('v2/data/training_top25.json',orient='records',date_format='iso')
    call=signal_from_selected(m,p)
    # Preserve the latest prospective call on non-signal days. Do not overwrite with NO_SIGNAL.
    sp=Path('v2/signal/latest_trade_call.json')
    if call is not None: sp.write_text(json.dumps(call,indent=2,default=str))
    elif not sp.exists(): sp.write_text(json.dumps({'status':'NO_SIGNAL','latest_data_date':str(max(m.dates).date()),'reason':'No Thursday signal session in current dataset.'},indent=2))
    if call is not None:
        md=[f"# Iron Condor V2 — {call['signal_id']}","",f"Signal session: {call['signal_date']}",f"Planned entry observation: {call['planned_entry_date']}",f"Expiry: {call['expiry']}",f"NIFTY reference: {call['nifty_reference']:.2f}",f"Minimum-credit condition: {call['entry_condition']}","","## Four-leg spread"]
        for n,v in call['legs'].items(): md.append(f"- {n}: **{v['strike']:.0f} {v['type']}**, EOD reference premium {v['reference_eod_premium']:.2f}")
        md += [f"- Reference credit: **{call['reference_credit_points']:.2f} points**",f"- Breakevens: **{call['breakeven_lower']:.2f} / {call['breakeven_upper']:.2f}**",f"- Lot size: **{call['lot_size']}**",f"- Suggested paper lots: **{call['suggested_paper_lots']}**",f"- Maximum expiry loss: **₹{call['max_expiry_loss_total']:,.2f}**"]
        Path('v2/signal/latest_trade_call.md').write_text('\n'.join(md)+'\n')
    pd.DataFrame([model]).to_json('v2/data/model_snapshot_table.json',orient='records',date_format='iso')

if __name__=='__main__': main()
