"""Kronos price forecast for every ticker in TICKERS (daily bars, CPU).

Env:
  TICKERS     comma list in STOCK_LIST format (hk00700, 688981, NVDA, ...)
  PRED_DAYS   trading days to forecast (default 5)
  KRONOS_DIR  path to the Kronos source checkout (default kronos_src)
  OUT_DIR     where to write results (default kronos_out)

Writes OUT_DIR/kronos_forecast.json and OUT_DIR/kronos_forecast.md.
A forecast is a model's guess, not a prediction you can trade on.
"""

import datetime as dt
import json
import os
import sys
import traceback

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tickers import parse_list  # noqa: E402

sys.path.insert(0, os.environ.get("KRONOS_DIR", "kronos_src"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

LOOKBACK = 400
PRED_DAYS = int(os.environ.get("PRED_DAYS", "5"))
OUT = os.environ.get("OUT_DIR", "kronos_out")


def bars(symbol):
    df = yf.Ticker(symbol).history(period="3y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("no price data from Yahoo")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["amount"] = df["close"] * df["volume"]
    if len(df) < 120:
        raise RuntimeError("only %d daily bars" % len(df))
    return df.tail(LOOKBACK)


def main():
    tickers = parse_list(os.environ.get("TICKERS", ""))
    if not tickers:
        sys.exit("TICKERS is empty")
    os.makedirs(OUT, exist_ok=True)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(model, tok, device="cpu", max_context=512)

    results = []
    for sym in tickers:
        try:
            df = bars(sym)
            last = df.index[-1]
            # Crypto trades every day; stocks only on business days.
            rng = pd.date_range if sym.endswith(("-USD", "-USDT")) else pd.bdate_range
            future = rng(last + pd.Timedelta(days=1), periods=PRED_DAYS)
            p = pred.predict(
                df=df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
                x_timestamp=pd.Series(df.index),
                y_timestamp=pd.Series(future),
                pred_len=PRED_DAYS, T=1.0, top_p=0.9, sample_count=5, verbose=False,
            )
            close = float(df["close"].iloc[-1])
            end = float(p["close"].iloc[-1])
            results.append({
                "symbol": sym, "last_date": last.date().isoformat(), "last_close": round(close, 4),
                "days": PRED_DAYS,
                "path": [{"date": d.date().isoformat(), "close": round(float(c), 4)}
                         for d, c in zip(future, p["close"])],
                "forecast_close": round(end, 4),
                "forecast_pct": round((end / close - 1) * 100, 2),
                "forecast_high": round(float(p["high"].max()), 4),
                "forecast_low": round(float(p["low"].min()), 4),
            })
            print("%-10s %+.2f%% over %d days" % (sym, results[-1]["forecast_pct"], PRED_DAYS))
        except Exception as exc:  # one bad ticker never stops the rest
            traceback.print_exc()
            results.append({"symbol": sym, "error": str(exc)})

    stamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(os.path.join(OUT, "kronos_forecast.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": stamp, "model": "NeoQuasar/Kronos-small", "results": results}, fh, indent=2)

    ok = sorted([r for r in results if "error" not in r], key=lambda r: r["forecast_pct"], reverse=True)
    lines = ["# Kronos %d-day forecast" % PRED_DAYS, "",
             "Generated %s with Kronos-small (5 samples averaged). A model's guess, not advice." % stamp, "",
             "| Symbol | Last close | Forecast close | Change | Forecast high | Forecast low |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in ok:
        lines.append("| %s | %.2f | %.2f | %+.2f%% | %.2f | %.2f |" % (
            r["symbol"], r["last_close"], r["forecast_close"], r["forecast_pct"],
            r["forecast_high"], r["forecast_low"]))
    for r in results:
        if "error" in r:
            lines.append("| %s | error: %s | | | | |" % (r["symbol"], r["error"]))
    with open(os.path.join(OUT, "kronos_forecast.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
