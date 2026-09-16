from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

# Reuse the exact production research functions and cost model.
from generate_signal import CFG, CAPITAL, Market, backtest, score, costs, choose, lot_size


def window_select(m: Market, start: pd.Timestamp, end: pd.Timestamp):
    results = []
    for d, w, tp, sl in itertools.product(
        CFG["short_distance_pct"],
        CFG["wing_width_points"],
        CFG["take_profit_credit_pct"],
        CFG["stop_loss_credit_multiple"],
    ):
        t = backtest(m, float(d), int(w), float(tp), float(sl), start=start, end=end)
        results.append(
            {
                "distance": float(d),
                "width": int(w),
                "take_profit": float(tp),
                "stop_loss": float(sl),
                "trades": int(len(t)),
                "score": float(score(t)),
                "win_rate": float((t.net > 0).mean()) if not t.empty else 0.0,
                "total_return": float(t.net.sum() / CAPITAL) if not t.empty else 0.0,
                "median_trade": float(t.net.median()) if not t.empty else 0.0,
                "worst_trade": float(t.net.min()) if not t.empty else 0.0,
            }
        )
    df = pd.DataFrame(results).sort_values(["score", "total_return"], ascending=False).reset_index(drop=True)
    return df.iloc[0].to_dict(), df


def trade_metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "total_return": 0.0, "median_trade": 0.0, "worst_trade": 0.0, "best_trade": 0.0}
    gains = float(t.loc[t.net > 0, "net"].sum())
    losses = float(-t.loc[t.net < 0, "net"].sum())
    return {
        "trades": int(len(t)),
        "win_rate": float((t.net > 0).mean()),
        "total_return": float(t.net.sum() / CAPITAL),
        "net_pnl": float(t.net.sum()),
        "median_trade": float(t.net.median()),
        "worst_trade": float(t.net.min()),
        "best_trade": float(t.net.max()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "take_profit_trades": int((t.reason == "take_profit").sum()),
        "stop_loss_trades": int((t.reason == "stop_loss").sum()),
        "expiry_trades": int((t.reason == "expiry").sum()),
    }


def enrich_trade_ledger(m: Market, trades: pd.DataFrame, selected: dict) -> pd.DataFrame:
    """Reconstruct the complete four-leg trade card for every OOS trade.

    The production backtest stores the dates/credit/P&L. Here we deterministically
    reconstruct the exact strikes and reference EOD premiums from the same market
    snapshot used by the backtest so the Pages ledger exposes the full trade.
    """
    if trades.empty:
        return trades.copy()

    out = []
    for row in trades.itertuples(index=False):
        entry = pd.Timestamp(row.entry)
        exit_ = pd.Timestamp(row.exit)
        expiry = next((e for e in m.expiries.get(entry, ()) if int(CFG["min_days_to_expiry"]) <= (e - entry).days <= int(CFG["max_days_to_expiry"])), None)
        if expiry is None:
            continue
        legs = choose(m, entry, expiry, float(selected["distance"]), int(selected["width"]))
        if legs is None:
            continue
        strike_tuples = [(legs[0], "PE"), (legs[1], "PE"), (legs[2], "CE"), (legs[3], "CE")]
        entry_px = [m.p(entry, expiry, k, t) for k, t in strike_tuples]
        exit_px = [m.p(exit_, expiry, k, t) for k, t in strike_tuples]
        if any(x is None for x in entry_px + exit_px):
            continue

        credit = float(row.credit)
        lot = int(row.lot)
        gross = float((credit + exit_px[1] + exit_px[2] - exit_px[0] - exit_px[3]) * lot)
        modeled_cost = float(costs(entry_px, exit_px, lot))
        max_loss_points = float(row.max_loss) / lot if lot else None
        spread_width = max(float(legs[1] - legs[0]), float(legs[3] - legs[2]))
        lower_be = float(legs[1] - credit)
        upper_be = float(legs[2] + credit)
        tp_credit_pct = float(selected["take_profit"])
        sl_credit_multiple = float(selected["stop_loss"])
        target_debit = float(credit * (1.0 - tp_credit_pct))
        stop_debit = float(credit * (1.0 + sl_credit_multiple))
        entry_value = float(credit * lot)
        max_expiry_loss = float(max_loss_points * lot)

        detail = {
            "trade_id": f"OOS-{entry:%Y%m%d}-{expiry:%Y%m%d}-{int(legs[1])}-{int(legs[2])}",
            "entry_date": str(entry.date()),
            "exit_date": str(exit_.date()),
            "expiry": str(expiry.date()),
            "dte_at_entry": int((expiry - entry).days),
            "nifty_reference_entry": float(m.spot[entry]),
            "put_long_strike": float(legs[0]),
            "put_short_strike": float(legs[1]),
            "call_short_strike": float(legs[2]),
            "call_long_strike": float(legs[3]),
            "wing_width_points": float(spread_width),
            "put_long_entry_premium": float(entry_px[0]),
            "put_short_entry_premium": float(entry_px[1]),
            "call_short_entry_premium": float(entry_px[2]),
            "call_long_entry_premium": float(entry_px[3]),
            "put_long_exit_premium": float(exit_px[0]),
            "put_short_exit_premium": float(exit_px[1]),
            "call_short_exit_premium": float(exit_px[2]),
            "call_long_exit_premium": float(exit_px[3]),
            "entry_credit_points": float(credit),
            "entry_credit_value_per_lot": entry_value,
            "lower_breakeven": lower_be,
            "upper_breakeven": upper_be,
            "take_profit_credit_pct": tp_credit_pct,
            "stop_loss_credit_multiple": sl_credit_multiple,
            "take_profit_debit": target_debit,
            "stop_loss_debit": stop_debit,
            "lot_size": lot,
            "gross_pnl": gross,
            "modeled_costs": modeled_cost,
            "net_pnl": float(row.net),
            "return_on_capital": float(row.net / CAPITAL),
            "return_on_max_loss": float(row.net / max_expiry_loss) if max_expiry_loss else None,
            "max_expiry_loss_points": max_loss_points,
            "max_expiry_loss": max_expiry_loss,
            "exit_reason": str(row.reason),
        }
        out.append(detail)

    return pd.DataFrame(out)


def main():
    options = pd.read_parquet("data/cache/nifty_options.parquet")
    futures = pd.read_parquet("data/cache/nifty_futures.parquet")
    m = Market(options, futures)
    dates = list(m.dates)
    if len(dates) < 100:
        raise RuntimeError("Not enough data for OOS validation")

    split_idx = int(len(dates) * 0.70)
    train_start, train_end = dates[0], dates[split_idx - 1]
    oos_start, oos_end = dates[split_idx], dates[-1]

    selected, leaderboard = window_select(m, train_start, train_end)
    train_trades = backtest(m, float(selected["distance"]), int(selected["width"]), float(selected["take_profit"]), float(selected["stop_loss"]), start=train_start, end=train_end)
    oos_trades = backtest(m, float(selected["distance"]), int(selected["width"]), float(selected["take_profit"]), float(selected["stop_loss"]), start=oos_start, end=oos_end)
    detailed_oos = enrich_trade_ledger(m, oos_trades, selected)

    summary = {
        "method": "Chronological 70/30 holdout",
        "selection_rule": "512-configuration grid selected on training period only using the production research score",
        "production_training_note": "The live signal is trained on all available data separately; these OOS figures are a historical validation snapshot and are not the live model's in-sample result.",
        "dataset_start": str(train_start.date()),
        "dataset_end": str(oos_end.date()),
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "oos_start": str(oos_start.date()),
        "oos_end": str(oos_end.date()),
        "train_sessions": int((pd.Series(dates) <= train_end).sum()),
        "oos_sessions": int((pd.Series(dates) >= oos_start).sum()),
        "selected_parameters": {
            "distance": float(selected["distance"]),
            "width": int(selected["width"]),
            "take_profit": float(selected["take_profit"]),
            "stop_loss": float(selected["stop_loss"]),
        },
        "train": trade_metrics(train_trades),
        "oos": trade_metrics(oos_trades),
        "oos_trades_file": "data/research/oos_trade_history.csv",
        "oos_detailed_trades_file": "data/research/oos_trade_history_detailed.csv",
        "oos_detailed_trades_json": "data/research/oos_trade_history_detailed.json",
    }

    Path("data/research").mkdir(parents=True, exist_ok=True)
    Path("data/research/oos_validation.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    leaderboard.to_csv("data/research/oos_train_parameter_leaderboard.csv", index=False)
    oos_trades.to_csv("data/research/oos_trade_history.csv", index=False)
    detailed_oos.to_csv("data/research/oos_trade_history_detailed.csv", index=False)
    detailed_oos.to_json("data/research/oos_trade_history_detailed.json", orient="records", indent=2, date_format="iso")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
