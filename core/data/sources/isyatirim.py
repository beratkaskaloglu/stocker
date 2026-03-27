"""
core/data/sources/isyatirim.py
BIST piyasası için İş Yatırım veri kaynağı.
"""
from __future__ import annotations

import pandas as pd


class IsYatirimSource:
    """
    PSEUDO:
    1. isyatirimhisse kütüphanesini import et
    2. fetch(symbol, start, end, timeframe):
       a. hisse = Hisse(symbol)
       b. hisse.get_historical_data(start, end) → DataFrame
       c. Sütun adlarını normalize et (Open, High, Low, Close, Volume)
       d. timeframe resample (günlük → saatlik vs.)
    3. get_bist100_symbols():
       a. isyatirimhisse'den BIST100 listesi çek
       b. [THYAO, AKBNK, GARAN, ...] döndür
    """

    def __init__(self):
        # TODO: isyatirimhisse import
        pass

    def fetch(self, symbol: str, start: str, end: str, timeframe: str = "1d") -> pd.DataFrame:
        # TODO: implement
        raise NotImplementedError

    def get_bist100_symbols(self) -> list[str]:
        # TODO: implement
        raise NotImplementedError
