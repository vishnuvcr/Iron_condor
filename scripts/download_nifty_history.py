from __future__ import annotations

import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}


def _num(s, idx):
    return pd.to_numeric(s, errors="coerce") if s is not None else pd.Series(np.nan, index=idx)


def _pick(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _download(dt: date, retries=4):
    dd, mon, yyyy = dt.strftime("%d"), dt.strftime("%b").upper(), dt.strftime("%Y")
    if dt >= date(2024, 7, 8):
        urls = [
            f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
            f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
        ]
    else:
        urls = [
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
        ]
    for url in urls:
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
                if r.ok and r.content[:2] == b"PK":
                    return r.content
                if r.status_code == 404:
                    break
            except Exception:
                pass
            time.sleep(min(2 + 2 * attempt, 8))
    return None


def _read(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next((x for x in z.namelist() if x.lower().endswith((".csv", ".txt"))), None)
        if not name:
            return pd.DataFrame()
        with z.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _options(raw, dt):
    if {"SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "CLOSE"}.issubset(raw.columns):
        symbol = raw.SYMBOL.astype(str).str.upper().str.strip()
        typ = raw.OPTION_TYP.astype(str).str.upper().str.strip()
        expiry = pd.to_datetime(raw.EXPIRY_DT, errors="coerce").dt.normalize()
        strike = _num(raw.STRIKE_PR, raw.index)
        close = _num(raw.CLOSE, raw.index)
        open_ = _num(_pick(raw, "OPEN"), raw.index)
        high = _num(_pick(raw, "HIGH"), raw.index)
        low = _num(_pick(raw, "LOW"), raw.index)
        volume = _num(_pick(raw, "CONTRACTS"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OPEN_INT"), raw.index).fillna(0)
        mask = symbol.eq("NIFTY") & typ.isin(["CE", "PE"])
    elif {"TckrSymb", "XpryDt", "StrkPric", "OptnTp", "ClsPric"}.issubset(raw.columns):
        symbol = raw.TckrSymb.astype(str).str.upper().str.strip()
        typ = raw.OptnTp.astype(str).str.upper().str.strip()
        expiry = pd.to_datetime(raw.XpryDt, errors="coerce").dt.normalize()
        strike = _num(raw.StrkPric, raw.index)
        close = _num(raw.ClsPric, raw.index)
        open_ = _num(_pick(raw, "OpnPric"), raw.index)
        high = _num(_pick(raw, "HghPric"), raw.index)
        low = _num(_pick(raw, "LwPric"), raw.index)
        volume = _num(_pick(raw, "TtlTradgVol"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OpnIntrst"), raw.index).fillna(0)
        mask = symbol.eq("NIFTY") & typ.isin(["CE", "PE"])
    else:
        return pd.DataFrame()
    idx = raw.index[mask]
    if len(idx) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({"date": pd.Timestamp(dt), "expiry": expiry.loc[idx].values, "strike": strike.loc[idx].values,
                        "option_type": typ.loc[idx].values, "open": open_.loc[idx].values, "high": high.loc[idx].values,
                        "low": low.loc[idx].values, "close": close.loc[idx].values, "volume": volume.loc[idx].values,
                        "open_interest": oi.loc[idx].values})
    out = out.dropna(subset=["expiry", "strike", "close"])
    out = out[out.expiry >= out.date]
    return out.drop_duplicates(["date", "expiry", "strike", "option_type"])


def _futures(raw, dt):
    if {"SYMBOL", "EXPIRY_DT", "CLOSE", "INSTRUMENT"}.issubset(raw.columns):
        symbol = raw.SYMBOL.astype(str).str.upper().str.strip()
        inst = raw.INSTRUMENT.astype(str).str.upper().str.strip()
        expiry = pd.to_datetime(raw.EXPIRY_DT, errors="coerce").dt.normalize()
        close = _num(raw.CLOSE, raw.index)
        volume = _num(_pick(raw, "CONTRACTS"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OPEN_INT"), raw.index).fillna(0)
        mask = symbol.eq("NIFTY") & inst.str.contains("FUT")
    elif {"TckrSymb", "XpryDt", "ClsPric"}.issubset(raw.columns):
        symbol = raw.TckrSymb.astype(str).str.upper().str.strip()
        inst = raw.FinInstrmTp.astype(str).str.upper().str.strip() if "FinInstrmTp" in raw.columns else pd.Series("", index=raw.index)
        expiry = pd.to_datetime(raw.XpryDt, errors="coerce").dt.normalize()
        close = _num(raw.ClsPric, raw.index)
        volume = _num(_pick(raw, "TtlTradgVol"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OpnIntrst"), raw.index).fillna(0)
        mask = symbol.eq("NIFTY") & inst.str.contains("FUT")
        if not mask.any() and "OptnTp" in raw.columns:
            mask = symbol.eq("NIFTY") & raw.OptnTp.isna() & expiry.notna() & close.notna()
    else:
        return pd.DataFrame()
    idx = raw.index[mask]
    if len(idx) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({"date": pd.Timestamp(dt), "expiry": expiry.loc[idx].values, "close": close.loc[idx].values,
                        "volume": volume.loc[idx].values, "open_interest": oi.loc[idx].values})
    out = out.dropna(subset=["expiry", "close"])
    out = out[out.expiry >= out.date]
    return out.drop_duplicates(["date", "expiry"])


def main():
    start = date.fromisoformat(os.environ.get("NSE_HISTORY_START", "2015-01-01"))
    end = date.fromisoformat(os.environ.get("NSE_HISTORY_END", str(datetime.now().date())))
    workers = int(os.environ.get("NSE_HISTORY_WORKERS", "5"))
    opt_dir = Path("data/cache/option_parts"); fut_dir = Path("data/cache/future_parts")
    opt_dir.mkdir(parents=True, exist_ok=True); fut_dir.mkdir(parents=True, exist_ok=True)
    days = [x.date() for x in pd.date_range(start, end, freq="B")]

    def worker(dt):
        op = opt_dir / f"{dt:%Y%m%d}.parquet"; fp = fut_dir / f"{dt:%Y%m%d}.parquet"
        if op.exists(): return dt, True
        content = _download(dt)
        if content is None: return dt, False
        try:
            raw = _read(content); opt = _options(raw, dt); fut = _futures(raw, dt)
            if opt.empty: return dt, False
            opt.to_parquet(op, index=False)
            if not fut.empty: fut.to_parquet(fp, index=False)
            return dt, True
        except Exception:
            return dt, False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, d) for d in days]
        for i, f in enumerate(as_completed(futures), 1):
            _, good = f.result(); ok += int(good)
            if i % 100 == 0 or i == len(days): print(f"Downloaded {i}/{len(days)} days; usable={ok}", flush=True)

    opts = sorted(opt_dir.glob("*.parquet")); futs = sorted(fut_dir.glob("*.parquet"))
    if not opts: raise RuntimeError("No NIFTY option history downloaded")
    o = pd.concat((pd.read_parquet(p) for p in opts), ignore_index=True)
    o["date"] = pd.to_datetime(o.date).dt.normalize(); o["expiry"] = pd.to_datetime(o.expiry).dt.normalize()
    o = o.drop_duplicates(["date","expiry","strike","option_type"]).sort_values(["date","expiry","strike","option_type"])
    Path("data/cache").mkdir(parents=True, exist_ok=True); o.to_parquet("data/cache/nifty_options.parquet", index=False)
    if futs:
        f = pd.concat((pd.read_parquet(p) for p in futs), ignore_index=True)
        f["date"] = pd.to_datetime(f.date).dt.normalize(); f["expiry"] = pd.to_datetime(f.expiry).dt.normalize()
        f = f.drop_duplicates(["date","expiry"]).sort_values(["date","expiry"])
        f.to_parquet("data/cache/nifty_futures.parquet", index=False)
    manifest = {"start": str(o.date.min().date()), "end": str(o.date.max().date()), "rows": int(len(o)),
                "dates": int(o.date.nunique()), "expiries": int(o.expiry.nunique()), "usable_download_days": ok}
    Path("data/cache/manifest.json").write_text(__import__("json").dumps(manifest, indent=2))
    print(manifest)

if __name__ == "__main__": main()
