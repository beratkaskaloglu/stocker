"""
core/data/sources/tvdatafeed.py
TradingView scraper — US ve BIST için fallback kaynak.
"""
from __future__ import annotations

import pandas as pd


class TvDatafeedSource:
    """
    PSEUDO:
    1. TvDatafeed() client başlat (username/password opsiyonel)
    2. fetch(symbol, exchange, timeframe, n_bars=5000):
       a. tv.get_hist(symbol, exchange, interval, n_bars)
       b. DataFrame normalize et
    3. Exchange mapping:
       - US:   exchange="NASDAQ" veya "NYSE"
       - BIST: exchange="BIST"
    """

    INTERVAL_MAP = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "1H", "4h": "4H", "1d": "1D",
    }

    EXCHANGE_MAP = {
        "US_NASDAQ": "NASDAQ",
        "US_NYSE": "NYSE",
        "BIST": "BIST",
    }

    def __init__(self):
        # TODO: from tvdatafeed import TvDatafeed, Interval
        pass

    def fetch(self, symbol: str, exchange: str, timeframe: str, n_bars: int = 5000) -> pd.DataFrame:
        # TODO: implement
        raise NotImplementedError
