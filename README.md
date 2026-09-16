# NIFTY Iron Condor — Prospective Paper-Trading Pipeline

This repository turns the currently validated **Success V1** NIFTY weekly iron-condor research configuration into a repeatable signal pipeline for prospective paper trading.

## What it does

1. Downloads the currently available NIFTY option and NIFTY index-futures daily EOD history from NSE archives.
2. Builds the NIFTY underlying reference from the nearest non-expired NIFTY future.
3. Trains/selects the iron-condor configuration on **all data available up to the latest completed session**.
4. Generates the next prospective signal using **delayed execution**: strikes are selected from the signal session and the paper entry is scheduled for the next eligible trading session.
5. Produces a detailed trade card containing:
   - signal date and planned entry date
   - expiry and DTE
   - NIFTY reference price
   - four strikes and each leg's EOD reference premium
   - net credit
   - breakevens
   - wing width
   - maximum theoretical expiry loss
   - lot size and suggested paper lots
   - total premium received / capital allocation
   - take-profit debit
   - stop-loss debit
   - target / stop P&L per lot and for the proposed position
   - modeled transaction costs
   - position status

## Important execution convention

The research result was not treated as executable from the same day's closing prices. The prospective pipeline therefore selects the structure on the **Tuesday signal session** and plans entry on the **next eligible NSE session**. The option premiums in the signal are reference EOD values only; actual paper fills should be recorded separately.

## Capital / lots

`capital` is the notional paper-trading bankroll. `max_paper_lots` defaults to 1 so the first live-paper phase does not overstate scalability. The pipeline also calculates how many lots fit inside the modeled maximum-loss budget, but it does **not** claim that this equals broker margin requirement.

## Run it manually

```bash
pip install -r requirements.txt
python scripts/download_nifty_history.py
python scripts/generate_signal.py
```

Outputs:

- `signals/latest_trade_call.md`
- `signals/latest_trade_call.json`
- `signals/open_trade.json`
- `signals/trade_history.csv`
- `data/research/selected_parameters.json`

## GitHub Actions

The scheduled workflow runs weekly after the Tuesday market close. It can also be started manually with `workflow_dispatch`.

This is a **paper-trading / research system**, not an automated broker execution system. Do not assume the EOD reference premiums are obtainable fills.
