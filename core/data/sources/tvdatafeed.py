"""
core/data/sources/tvdatafeed.py
TradingView scraper — US ve BIST için fallback kaynak.
"""
from __future__ import annotations

import os

import pandas as pd
from loguru import logger


class TvDatafeedSource:
    """TradingView data feed — fallback for both US and BIST markets."""

    INTERVAL_MAP = {
        "1m": "in_1_minute",
        "5m": "in_5_minute",
        "15m": "in_15_minute",
        "30m": "in_30_minute",
        "1h": "in_1_hour",
        "4h": "in_4_hour",
        "1d": "in_daily",
        "1w": "in_weekly",
        "1M": "in_monthly",
    }

    EXCHANGE_MAP = {
        "US_NASDAQ": "NASDAQ",
        "US_NYSE": "NYSE",
        "BIST": "BIST",
    }

    def __init__(self):
        from tvDatafeed import TvDatafeed, Interval
        self._Interval = Interval
        username = os.environ.get("TV_USERNAME")
        password = os.environ.get("TV_PASSWORD")
        self.tv = TvDatafeed(username=username, password=password)

    def fetch(
        self, symbol: str, exchange: str, timeframe: str, n_bars: int = 5000
    ) -> pd.DataFrame:
        """
        TradingView'dan OHLCV verisi çeker.

        Parameters
        ----------
        symbol   : str – örn. "AAPL", "THYAO"
        exchange : str – "NASDAQ", "NYSE", "BIST" veya EXCHANGE_MAP key
        timeframe: str – "1m","5m","15m","30m","1h","4h","1d","1w","1M"
        n_bars   : int – çekilecek bar sayısı (max ~5000)

        Returns
        -------
        pd.DataFrame – columns: [open, high, low, close, volume], index: datetime
        """
        # Exchange mapping
        exchange = self.EXCHANGE_MAP.get(exchange, exchange)

        # Interval mapping
        interval_name = self.INTERVAL_MAP.get(timeframe)
        if interval_name is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        interval = getattr(self._Interval, interval_name)

        try:
            raw = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars,
            )
        except Exception as e:
            logger.warning(f"TvDatafeed: fetch failed for {symbol}@{exchange}: {e}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning(f"TvDatafeed: no data for {symbol}@{exchange}")
            return pd.DataFrame()

        df = raw.rename(columns=str.lower)
        # tvDatafeed returns: symbol, open, high, low, close, volume
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df.index.name = "datetime"
        df["symbol"] = symbol
        return df
