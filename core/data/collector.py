"""
core/data/collector.py
Veri toplama factory — market'e göre doğru source'u seçer.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from pathlib import Path


class DataCollector:
    """
    PSEUDO:
    1. market parametresine göre kaynak seç (finnhub / isyatirim / tvdatafeed)
    2. symbols listesini yükle (config'den veya index'ten)
    3. Her sembol için OHLCV çek
    4. Hatalı sembolleri logla, devam et
    5. DataFrame birleştir
    6. storage.py ile SQLite + CSV'ye kaydet
    7. outputs/data/latest.json yaz (agent iletişimi için)
    """

    def __init__(self, market: str, config: dict):
        self.market = market
        self.config = config
        self.source = self._init_source()

    def _init_source(self):
        # TODO: market'e göre import et
        # if self.market == "US": return FinnhubSource(...)
        # if self.market == "BIST": return IsYatirimSource(...)
        # fallback: return TvDatafeedSource(...)
        raise NotImplementedError

    def collect(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        PSEUDO:
        - her sembol için source.fetch(symbol, start, end, timeframe)
        - başarısız sembolleri atla
        - tüm veriyi birleştir → multi-index DataFrame
        - storage'a kaydet
        - outputs/data/latest.json güncelle
        """
        raise NotImplementedError

    def get_index_symbols(self) -> list[str]:
        """SP500 veya BIST100 sembol listesini döndür."""
        # TODO: config/us.yaml veya config/bist.yaml'dan oku
        raise NotImplementedError
