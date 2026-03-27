"""
core/data/sources/finnhub.py
US piyasası için Finnhub veri kaynağı.
"""
from __future__ import annotations

import os
import pandas as pd
import finnhub


class FinnhubSource:
    """
    PSEUDO:
    1. FINNHUB_API_KEY ile client başlat
    2. fetch(symbol, start, end, timeframe):
       a. finnhub_client.stock_candles(symbol, resolution, _from, _to)
       b. JSON → DataFrame dönüştür (o, h, l, c, v, t sütunları)
       c. timestamp → datetime index
       d. Hata varsa (boş data, rate limit) logla + raise
    3. get_news(symbol, start, end):
       a. finnhub_client.company_news(symbol, _from, _to)
       b. headline + summary + sentiment döndür
    """

    RESOLUTION_MAP = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "1d": "D", "1w": "W", "1M": "M",
    }

    def __init__(self):
        api_key = os.environ["FINNHUB_API_KEY"]
        self.client = finnhub.Client(api_key=api_key)

    def fetch(self, symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
        # TODO: implement
        raise NotImplementedError

    def get_news(self, symbol: str, start: str, end: str) -> list[dict]:
        # TODO: implement
        raise NotImplementedError
