"""
core/data/sources/finnhub.py
US piyasası için Finnhub veri kaynağı.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import finnhub
from loguru import logger


class FinnhubSource:
    """Finnhub API for US markets (S&P 500)."""

    RESOLUTION_MAP = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "1d": "D", "1w": "W", "1M": "M",
    }

    def __init__(self):
        api_key = os.environ["FINNHUB_API_KEY"]
        self.client = finnhub.Client(api_key=api_key)

    def fetch(self, symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
        """
        Finnhub stock_candles API ile OHLCV verisi çeker.

        Parameters
        ----------
        symbol : str   – örn. "AAPL"
        start  : str   – "2024-01-01"
        end    : str   – "2024-12-31"
        timeframe : str – "1m","5m","15m","30m","1h","1d","1w","1M"

        Returns
        -------
        pd.DataFrame – columns: [open, high, low, close, volume], index: datetime
        """
        resolution = self.RESOLUTION_MAP.get(timeframe)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        ts_start = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        ts_end = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

        data = self.client.stock_candles(symbol, resolution, ts_start, ts_end)

        if data.get("s") != "ok" or not data.get("t"):
            logger.warning(f"Finnhub: no data for {symbol} ({timeframe})")
            return pd.DataFrame()

        df = pd.DataFrame({
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"],
        }, index=pd.to_datetime(data["t"], unit="s", utc=True))

        df.index.name = "datetime"
        df = df.sort_index()
        df["symbol"] = symbol
        return df

    def get_news(self, symbol: str, start: str, end: str) -> list[dict]:
        """
        Finnhub company_news API ile haber çeker.

        Returns
        -------
        list[dict] – her eleman: {headline, summary, source, url, datetime}
        """
        raw = self.client.company_news(symbol, _from=start, to=end)

        if not raw:
            logger.warning(f"Finnhub: no news for {symbol}")
            return []

        news = []
        for item in raw:
            news.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "symbol": symbol,
            })
        return news
