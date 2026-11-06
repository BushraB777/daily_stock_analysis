"""Run TradingAgents (multi-agent LLM debate) on a few tickers with Gemini.

Env:
  TICKERS          comma list in STOCK_LIST format; keep it to 1-3 on the free tier
  TRADE_DATE       YYYY-MM-DD (default: today, UTC)
  GOOGLE_API_KEY   Gemini key (mapped from the GEMINI_API_KEY secret)
  DEEP_MODEL       managers' model (default gemini-3.5-flash)
  QUICK_MODEL      analysts' model (default gemini-3.1-flash-lite)
  OUT_DIR          default ta_out

Writes OUT_DIR/<SYMBOL>/ (the full reports, incl. complete_report.html)
and OUT_DIR/tradingagents_summary.json / .md. Research output, not advice.
"""

import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tickers import parse_list  # noqa: E402

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "ta_out"))


def main():
    tickers = parse_list(os.environ.get("TICKERS", ""))
    if not tickers:
        sys.exit("TICKERS is empty")
    date = os.environ.get("TRADE_DATE") or dt.datetime.utcnow().date().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "llm_provider": "google",
        "deep_think_llm": os.environ.get("DEEP_MODEL") or "gemini-3.5-flash",
        "quick_think_llm": os.environ.get("QUICK_MODEL") or "gemini-3.1-flash-lite",
        "results_dir": str(OUT / "_logs"),
        "llm_max_retries": 6,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    })
    ta = TradingAgentsGraph(debug=False, config=cfg)

    summary = []
    for sym in tickers:
        try:
            state, signal = ta.propagate(sym, date)
            path = ta.save_reports(state, sym, save_path=OUT / sym)
            summary.append({"symbol": sym, "date": date, "rating": str(signal), "report_dir": str(path)})
            print("%-10s %s" % (sym, signal))
        except Exception as exc:  # one failure never stops the rest
            traceback.print_exc()
            summary.append({"symbol": sym, "date": date, "error": str(exc)[:500]})

    (OUT / "tradingagents_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# TradingAgents decisions (%s)" % date, "",
             "Models: deep=%s, quick=%s. Research output, not advice." % (cfg["deep_think_llm"], cfg["quick_think_llm"]), "",
             "| Symbol | Rating |", "|---|---|"]
    for s in summary:
        lines.append("| %s | %s |" % (s["symbol"], s.get("rating") or "error: " + s["error"][:120]))
    (OUT / "tradingagents_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
