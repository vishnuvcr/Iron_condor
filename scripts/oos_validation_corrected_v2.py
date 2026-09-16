from __future__ import annotations

import itertools, json, math
from pathlib import Path
import numpy as np, pandas as pd, yaml

CFG=yaml.safe_load(Path('config.yaml').read_text()); CAPITAL=float(CFG['capital'])

def lot_size(expiry):
 e=pd.Timestamp(expiry).normalize()
 if e<pd.Timestamp('2015-10-30'): return 25
 if e<pd.Timestamp('2021-08-01'): return 75
 if e<pd.Timestamp('2024-05-02'): return 50
 if e<pd.Timestamp('2024-11-20'): return 25
 if e<pd.Timestamp('2026-01-06'): return 75
 return 65

def costs(entry,exit_,lot):
 brokerage=80.0
 sell=(entry[1]+entry[2]+exit_[0]+exit_[3])*lot; buy=(entry[0]+entry[3]+exit_[1]+exit_[2])*lot
 turnover=sum(abs(a)+abs(b) for a,b in zip(entry,exit_))*lot
 txn=turnover*0.00035; stt=max(sell,0)*0.001; stamp=max(buy,0)*0.00003; sebi=turnover*0.000001; gst=(brokerage+txn+sebi)*0.18
 return brokerage+stt+stamp+sebi+txn+gst

class Market:
 def __init__(self,o,f):
  o=o.copy(); o['date']=pd.to_datetime(o.date).dt.normalize(); o['expiry']=pd.to_datetime(o.expiry).dt.normalize(); o['option_type']=o.option_type.astype(str).str.upper(); o=o.drop_duplicates(['date','expiry','strike','option_type'])
  f=f.copy(); f['date']=pd.to_datetime(f.date).dt.normalize(); f['expiry']=pd.to_datetime(f.expiry).dt.normalize(); f=f[f.expiry>=f.date].sort_values(['date','expiry']).drop_duplicates(['date','expiry'])
  spot=f.groupby('date',as_index=False).first()[['date','close']]; self.spot=dict(zip(spot.date,spot.close.astype(float))); self.dates=tuple(pd.Timestamp(x) for x in sorted(set(o.date)&set(spot.date)))
  self.exp={}; self.strikes={}; self.px={}
  for (d,e),g in o.groupby(['date','expiry']):
   d,e=pd.Timestamp(d),pd.Timestamp(e); self.exp.setdefault(d,[]).append(e); self.strikes[(d,e)]=np.asarray(sorted(g.strike.astype(float).unique()))
  for d in self.exp:self.exp[d]=tuple(sorted(self.exp[d]))
  for d,e,k,t,c in o[['date','expiry','strike','option_type','close']].itertuples(index=False,name=None): self.px[(pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())]=float(c) if pd.notna(c) else np.nan
 def p(self,d,e,k,t):
  v=self.px.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t)); return None if v is None or pd.isna(v) else float(v)
 def expiry(self,d):
  for e in self.exp.get(pd.Timestamp(d),()):
   if int(CFG['min_days_to_expiry'])<=(e-pd.Timestamp(d)).days<=int(CFG['max_days_to_expiry']): return e
 def choose(self,d,e,distance,width):
  s=self.spot.get(pd.Timestamp(d)); ks=self.strikes.get((pd.Timestamp(d),pd.Timestamp(e)))
  if s is None or ks is None: return None
  ps=ks[ks<=s*(1-distance)]; cs=ks[ks>=s*(1+distance)]
  if not len(ps) or not len(cs): return None
  ps,cs=float(ps[-1]),float(cs[0]); pls=ks[ks<=ps-width+1e-9]; cls=ks[ks>=cs+width-1e-9]
  if not len(pls) or not len(cls): return None
  return float(pls[-1]),ps,cs,float(cls[0])

def backtest(m,distance,width,tp,sl,start,end,slip=0.5):
 rows=[]; dates=[d for d in m.dates if d.weekday()==int(CFG['entry_weekday']) and start<=d<=end]
 for d in dates:
  e=m.expiry(d); legs=m.choose(d,e,distance,width) if e is not None else None
  if legs is None: continue
  lk=[(legs[0],'PE'),(legs[1],'PE'),(legs[2],'CE'),(legs[3],'CE')]; en=[m.p(d,e,k,t) for k,t in lk]
  if any(v is None or v<=0 for v in en): continue
  credit=en[1]+en[2]-en[0]-en[3]
  if credit<float(CFG.get('minimum_credit_points',0)): continue
  lot=lot_size(e); max_loss=max(legs[1]-legs[0],legs[3]-legs[2])-credit
  if credit<=0 or max_loss<=0: continue
  exit_date,reason=None,'expiry'; exit_px=None; exit_mark=None
  for day in [x for x in m.dates if d<x<=e]:
   vals=[m.p(day,e,k,t) for k,t in lk]
   if any(v is None for v in vals): continue
   mark=vals[1]+vals[2]-vals[0]-vals[3]
   debit=mark+4*slip
   effective_credit=credit-4*slip
   if debit <= (1-tp)*effective_credit: exit_date,reason,exit_px,exit_mark=day,'take_profit',vals,mark; break
   if debit >= (1+sl)*effective_credit: exit_date,reason,exit_px,exit_mark=day,'stop_loss',vals,mark; break
  if exit_date is None:
   vals=[m.p(e,e,k,t) for k,t in lk]
   if any(v is None for v in vals): continue
   exit_date,reason,exit_px,exit_mark=e,'expiry',vals,vals[1]+vals[2]-vals[0]-vals[3]
  gross=(credit-4*slip-exit_mark-4*slip)*lot
  net=gross-costs(en,exit_px,lot)
  rows.append({'entry':d,'exit':exit_date,'net':net,'lot':lot,'credit':credit,'reason':reason,'max_loss':max_loss*lot})
 return pd.DataFrame(rows)

def score(t):
 if t.empty:return -1e9
 w=t.groupby(t.exit.dt.to_period('W')).net.sum()/CAPITAL; win=float((t.net>0).mean()); worst=float(w.min())
 return 2*float(w.median())+float(w.mean())+0.10*float((w>=0.05).mean())+0.25*float((w>=0).mean())+0.10*win+0.50*worst

def metrics(t):
 if t.empty:return {'trades':0,'win_rate':0,'total_return':0,'net_pnl':0}
 gains=float(t.loc[t.net>0,'net'].sum()); losses=float(-t.loc[t.net<0,'net'].sum())
 eq=CAPITAL+t.net.cumsum(); dd=float((eq/eq.cummax()-1).min())
 return {'trades':int(len(t)),'win_rate':float((t.net>0).mean()),'total_return':float(t.net.sum()/CAPITAL),'net_pnl':float(t.net.sum()),'median_trade':float(t.net.median()),'worst_trade':float(t.net.min()),'best_trade':float(t.net.max()),'profit_factor':float(gains/losses) if losses else math.inf,'max_drawdown':dd,'take_profit_trades':int((t.reason=='take_profit').sum()),'stop_loss_trades':int((t.reason=='stop_loss').sum()),'expiry_trades':int((t.reason=='expiry').sum())}

def main():
 m=Market(pd.read_parquet('data/cache/nifty_options.parquet'),pd.read_parquet('data/cache/nifty_futures.parquet')); dates=list(m.dates); split=int(len(dates)*0.70); ts,te=dates[0],dates[split-1]; os,oe=dates[split],dates[-1]
 res=[]
 for d,w,tp,sl in itertools.product(CFG['short_distance_pct'],CFG['wing_width_points'],CFG['take_profit_credit_pct'],CFG['stop_loss_credit_multiple']):
  t=backtest(m,float(d),int(w),float(tp),float(sl),ts,te); res.append({'distance':d,'width':w,'take_profit':tp,'stop_loss':sl,'trades':len(t),'score':score(t),'win_rate':float((t.net>0).mean()) if len(t) else 0,'total_return':float(t.net.sum()/CAPITAL) if len(t) else 0,'median_trade':float(t.net.median()) if len(t) else 0,'worst_trade':float(t.net.min()) if len(t) else 0})
 df=pd.DataFrame(res).sort_values(['score','total_return'],ascending=False).reset_index(drop=True); sel=df.iloc[0].to_dict()
 tr=backtest(m,float(sel['distance']),int(sel['width']),float(sel['take_profit']),float(sel['stop_loss']),ts,te); oos=backtest(m,float(sel['distance']),int(sel['width']),float(sel['take_profit']),float(sel['stop_loss']),os,oe)
 summary={'method':'Chronological 70/30 holdout, corrected condor accounting','dataset_start':str(ts.date()),'dataset_end':str(oe.date()),'train_start':str(ts.date()),'train_end':str(te.date()),'oos_start':str(os.date()),'oos_end':str(oe.date()),'selected_parameters':{k:sel[k] for k in ['distance','width','take_profit','stop_loss']},'train':metrics(tr),'oos':metrics(oos),'accounting_fix':'P&L = entry credit - exit condor mark - entry/exit modeled slippage - broker costs; TP/SL use closing debit mark, not the inverse sign.'}
 Path('data/research').mkdir(parents=True,exist_ok=True); Path('data/research/oos_validation_corrected_v2.json').write_text(json.dumps(summary,indent=2,allow_nan=True)); df.to_csv('data/research/oos_train_parameter_leaderboard_corrected_v2.csv',index=False); oos.to_csv('data/research/oos_trade_history_corrected_v2.csv',index=False)
 print(json.dumps(summary,indent=2,allow_nan=True))
if __name__=='__main__':main()
