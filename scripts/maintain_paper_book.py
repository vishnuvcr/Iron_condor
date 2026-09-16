from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lot_size(expiry: pd.Timestamp) -> int:
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"): return 25
    if e < pd.Timestamp("2021-08-01"): return 75
    if e < pd.Timestamp("2024-05-02"): return 50
    if e < pd.Timestamp("2024-11-20"): return 25
    if e < pd.Timestamp("2026-01-06"): return 75
    return 65


def costs(entry, exit_px, lot):
    brokerage = 80.0
    sell_value = (entry[1] + entry[2] + exit_px[0] + exit_px[3]) * lot
    buy_value = (entry[0] + entry[3] + exit_px[1] + exit_px[2]) * lot
    turnover = sum(abs(a) + abs(b) for a, b in zip(entry, exit_px)) * lot
    txn = turnover * 0.00035
    stt = max(sell_value, 0.0) * 0.0010
    stamp = max(buy_value, 0.0) * 0.00003
    sebi = turnover * 0.000001
    gst = (brokerage + txn + sebi) * 0.18
    return brokerage + stt + stamp + sebi + txn + gst


def load_events(path: Path) -> list[dict]:
    if not path.exists() or path.read_text().strip() == "": return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_event(path: Path, event_type: str, version: str, signal_id: str, payload: dict) -> None:
    event = {"event_id": f"{version}-{event_type}-{signal_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}", "event_type": event_type, "version": version, "signal_id": signal_id, "recorded_utc": iso_now(), **payload}
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")


def market_from_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    o = pd.read_parquet("data/cache/nifty_options.parquet").copy()
    f = pd.read_parquet("data/cache/nifty_futures.parquet").copy()
    for df in (o, f):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    o["option_type"] = o["option_type"].astype(str).str.upper()
    return o, f


def price(o, d, e, strike, typ):
    x = o[(o.date == d) & (o.expiry == e) & (o.strike == float(strike)) & (o.option_type == typ)]
    if x.empty: return None
    v = pd.to_numeric(x.close, errors="coerce").dropna()
    return float(v.iloc[-1]) if not v.empty else None


def mark(o, d, e, legs):
    vals = [price(o, d, e, x["strike"], x["type"]) for x in legs]
    if any(v is None for v in vals): return None, vals
    return vals[1] + vals[2] - vals[0] - vals[3], vals


def reconcile(version: str, signal_path: str, root_dir: str = "paper") -> None:
    signal = json.loads(Path(signal_path).read_text())
    vdir = Path(root_dir) / version
    vdir.mkdir(parents=True, exist_ok=True)
    events_path = vdir / "events.jsonl"
    book_path = vdir / "trade_book.json"
    events = load_events(events_path)
    signal_ids = {e["signal_id"] for e in events if e.get("event_type") == "SIGNAL_CREATED"}

    if signal.get("status") == "PENDING_ENTRY" and signal.get("signal_id") not in signal_ids:
        append_event(events_path, "SIGNAL_CREATED", version, signal["signal_id"], {"call_snapshot": signal})
        events = load_events(events_path)

    o, _ = market_from_files()
    outcomed = {e["signal_id"] for e in events if e.get("event_type") == "OUTCOME_RECORDED"}
    entered = {e["signal_id"] for e in events if e.get("event_type") == "ENTRY_OBSERVED"}
    created = [e for e in events if e.get("event_type") == "SIGNAL_CREATED"]

    for ev in created:
        sid = ev["signal_id"]
        if sid in outcomed: continue
        call = ev["call_snapshot"]
        if not call.get("planned_entry_date") or not call.get("expiry"): continue
        entry_date = pd.Timestamp(call["planned_entry_date"]).normalize()
        expiry = pd.Timestamp(call["expiry"]).normalize()
        if entry_date not in set(o.date.unique()): continue
        legs_dict = call.get("legs", {})
        names = ["put_long", "put_short", "call_short", "call_long"]
        legs = [{"strike": float(legs_dict[n]["strike"]), "type": str(legs_dict[n]["type"]).upper()} for n in names]
        entry = [price(o, entry_date, expiry, x["strike"], x["type"]) for x in legs]
        if any(v is None or v <= 0 for v in entry): continue
        if sid not in entered:
            raw_credit = entry[1] + entry[2] - entry[0] - entry[3]
            append_event(events_path, "ENTRY_OBSERVED", version, sid, {"entry_date": str(entry_date.date()), "entry_prices": entry, "raw_credit_points": raw_credit, "note": "EOD reference observation used for automated paper-book reconciliation; replace with actual fill event when available."})
            events = load_events(events_path)
        raw_credit = entry[1] + entry[2] - entry[0] - entry[3]
        slip = float(call.get("configured_slippage_per_leg", 0.0))
        effective_credit = raw_credit - 4 * slip
        min_credit = float(call.get("minimum_credit_points", 0.0))
        if effective_credit < min_credit or effective_credit <= 0: continue
        tp = float(call.get("take_profit_credit_pct", 0.0))
        sl = float(call.get("stop_loss_credit_multiple", 0.0))
        lot = int(call.get("lot_size") or lot_size(expiry)) * int(call.get("suggested_paper_lots", 1))
        exit_date = None; reason = None; exit_mark = None; exit_px = None
        dates = sorted(d for d in o.date.unique() if entry_date < d <= expiry)
        for day in dates:
            m, vals = mark(o, day, expiry, legs)
            if m is None: continue
            exit_debit = -m + 4 * slip
            if exit_debit <= (1 - tp) * effective_credit:
                exit_date, reason, exit_mark, exit_px = day, "take_profit", m, vals; break
            if exit_debit >= (1 + sl) * effective_credit:
                exit_date, reason, exit_mark, exit_px = day, "stop_loss", m, vals; break
        if exit_date is None:
            expiry_vals = mark(o, expiry, expiry, legs)
            if expiry_vals[0] is None: continue
            exit_date, reason, exit_mark, exit_px = expiry, "expiry", expiry_vals[0], expiry_vals[1]
        gross = (effective_credit + exit_mark - 4 * slip) * lot
        charge = costs(entry, exit_px, lot)
        net = gross - charge
        append_event(events_path, "OUTCOME_RECORDED", version, sid, {"entry_date": str(entry_date.date()), "exit_date": str(exit_date.date()), "expiry": str(expiry.date()), "entry_prices": entry, "exit_prices": exit_px, "entry_credit_points": raw_credit, "effective_credit_points": effective_credit, "gross_pnl": gross, "modeled_costs": charge, "net_pnl": net, "return_on_capital": net / float(call.get("capital", 100000.0)), "reason": reason, "outcome_basis": "Automated EOD-reference paper simulation; not a statement of actual fill execution."})

    events = load_events(events_path)
    latest_signal = {e["signal_id"]: e for e in events if e.get("event_type") == "SIGNAL_CREATED"}
    entry_events = {e["signal_id"]: e for e in events if e.get("event_type") == "ENTRY_OBSERVED"}
    outcome_events = {e["signal_id"]: e for e in events if e.get("event_type") == "OUTCOME_RECORDED"}
    trades = []
    for sid, ev in latest_signal.items():
        call = ev["call_snapshot"]
        item = {"signal_id": sid, "version": version, "call_snapshot": call, "entry_observation": entry_events.get(sid), "outcome": outcome_events.get(sid), "paper_status": "CLOSED" if sid in outcome_events else ("OPEN" if sid in entry_events else "PENDING_ENTRY")}
        trades.append(item)
    trades.sort(key=lambda x: x["call_snapshot"].get("signal_date", ""))
    book = {"version": version, "generated_utc": iso_now(), "prospective_only": True, "oos_trades_excluded": True, "trades": trades}
    book_path.write_text(json.dumps(book, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", required=True, choices=["v1", "v2"]); ap.add_argument("--signal", required=True)
    args = ap.parse_args(); reconcile(args.version, args.signal)


if __name__ == "__main__": main()
