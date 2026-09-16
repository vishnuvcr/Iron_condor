# Prospective Paper Trading Ledger

This directory contains immutable event logs and derived paper-trading books for each strategy version.

- `v1/events.jsonl` contains append-only V1 signal/outcome events.
- `v2/events.jsonl` contains append-only V2 signal/outcome events.
- `v1/trade_book.json` and `v2/trade_book.json` are derived current views.
- A signal's call snapshot is never rewritten. Later outcomes are recorded as separate immutable events keyed by `signal_id`.

The OOS/backtest trades are kept separate from these prospective books and are never inserted into them.