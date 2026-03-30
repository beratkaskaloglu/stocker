"""
scripts/collect_real_data.py
S&P 500 gercek veri toplama — OHLCV + haberler.

Kullanim:
    python scripts/collect_real_data.py --symbols AAPL,MSFT,GOOGL --start 2023-01-01
    python scripts/collect_real_data.py --top 50 --start 2022-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import finnhub
from loguru import logger

# ─── Setup ──────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("FINNHUB_API_KEY", "d73s199r01qjjol44ul0d73s199r01qjjol44ulg")
client = finnhub.Client(api_key=API_KEY)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ─── S&P 500 Top Symbols ────────────────────────────────────────────────────

TOP_SP500 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ABT", "CRM",
    "ACN", "DHR", "LIN", "TXN", "NEE", "ADBE", "CMCSA", "NKE", "PM", "ORCL",
    "RTX", "UPS", "HON", "LOW", "QCOM", "INTC", "INTU", "AMAT", "BA", "CAT",
]


def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """yfinance ile OHLCV cek (ucretsiz, kayit gereksiz)."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")

    if df.empty:
        logger.warning(f"  {symbol}: no OHLCV data")
        return pd.DataFrame()

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    df["symbol"] = symbol
    return df


def fetch_news(symbol: str, start: str, end: str) -> list[dict]:
    """Finnhub'dan haber cek (3 aylik parcalarda, rate limit'e uygun)."""
    all_news = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=90), end_dt)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        try:
            raw = client.company_news(symbol, _from=s, to=e)
            if raw:
                for item in raw:
                    all_news.append({
                        "headline": item.get("headline", ""),
                        "summary": item.get("summary", ""),
                        "source": item.get("source", ""),
                        "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                        "symbol": symbol,
                    })
        except Exception as ex:
            logger.warning(f"  {symbol} news {s}→{e}: {ex}")
            # Rate limit hit → back off and retry once
            if "429" in str(ex) or "Too Many" in str(ex):
                logger.info(f"  Rate limit hit, waiting 30s...")
                time.sleep(30)
                try:
                    raw = client.company_news(symbol, _from=s, to=e)
                    if raw:
                        for item in raw:
                            all_news.append({
                                "headline": item.get("headline", ""),
                                "summary": item.get("summary", ""),
                                "source": item.get("source", ""),
                                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                                "symbol": symbol,
                            })
                except Exception:
                    pass

        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)  # rate limit: 60 req/min → ~40 req/min safe

    return all_news


def collect(symbols: list[str], start: str, end: str) -> None:
    """Tum semboller icin OHLCV + haber topla."""
    logger.info(f"Collecting {len(symbols)} symbols | {start} → {end}")

    all_ohlcv = []
    all_news = []

    for i, sym in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] {sym}")

        # OHLCV
        df = fetch_ohlcv(sym, start, end)
        if len(df) > 0:
            all_ohlcv.append(df)
            logger.info(f"  OHLCV: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
        time.sleep(0.3)

        # News
        news = fetch_news(sym, start, end)
        all_news.extend(news)
        logger.info(f"  News: {len(news)} articles")
        time.sleep(0.3)

    # Save OHLCV
    if all_ohlcv:
        ohlcv_df = pd.concat(all_ohlcv)
        ohlcv_path = DATA_DIR / f"ohlcv_{start}_{end}.csv"
        ohlcv_df.to_csv(ohlcv_path)
        logger.info(f"OHLCV saved: {ohlcv_path} ({len(ohlcv_df)} rows)")

    # Save News
    if all_news:
        news_path = DATA_DIR / f"news_{start}_{end}.json"
        news_path.write_text(json.dumps(all_news, ensure_ascii=False, indent=2))
        logger.info(f"News saved: {news_path} ({len(all_news)} articles)")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"TOPLAM: {len(all_ohlcv)} hisse, {sum(len(d) for d in all_ohlcv)} bar, {len(all_news)} haber")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=None, help="Virgülle ayrılmış semboller: AAPL,MSFT")
    parser.add_argument("--top", type=int, default=10, help="S&P 500'den ilk N hisse")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",")]
    else:
        syms = TOP_SP500[:args.top]

    collect(syms, args.start, args.end)
