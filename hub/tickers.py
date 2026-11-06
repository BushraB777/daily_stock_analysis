"""Convert daily_stock_analysis STOCK_LIST codes to Yahoo Finance symbols.

    hk00700   -> 0700.HK
    688981    -> 688981.SS   (Shanghai: 6xxxxx)
    002230    -> 002230.SZ   (Shenzhen: 0xxxxx / 3xxxxx)
    2330.TW, 005930.KS, 6758.T, NVDA -> unchanged
"""

import re


def to_yahoo(code: str) -> str:
    c = code.strip().upper()
    m = re.fullmatch(r"HK0*(\d{1,5})", c)
    if m:
        return m.group(1).zfill(4) + ".HK"
    if re.fullmatch(r"\d{6}", c):
        return c + (".SS" if c.startswith(("6", "9")) else ".SZ")
    return c


def parse_list(raw: str) -> list:
    return [to_yahoo(t) for t in re.split(r"[,\s]+", raw or "") if t.strip()]
