from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

a = __import__('sys')

# Reuse the exact production research functions and cost model.
from generate_signal import CFG, CAPITAL, Market, backtest, score


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
    }

    Path("data/research").mkdir(parents=True, exist_ok=True)
    Path("data/research/oos_validation.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    leaderboard.to_csv("data/research/oos_train_parameter_leaderboard.csv", index=False)
    oos_trades.to_csv("data/research/oos_trade_history.csv", index=False)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
