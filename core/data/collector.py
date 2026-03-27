"""
core/data/collector.py
Veri toplama factory — market'e göre doğru source'u seçer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from core.data.sources.finnhub import FinnhubSource
from core.data.sources.isyatirim import IsYatirimSource
from core.data.sources.tvdatafeed import TvDatafeedSource
from core.data.storage import Storage


class DataCollector:
    """
    Market-based data collection factory.
    1. market parametresine göre kaynak seç (finnhub / isyatirim / tvdatafeed)
    2. Her sembol için OHLCV çek, hatalıları logla
    3. DataFrame birleştir, storage'a kaydet
    4. outputs/data/latest.json yaz (agent iletişimi için)
    """

    def __init__(self, market: str, config: dict):
        self.market = market.upper()
        self.config = config
        self.source = self._init_source()
        self.storage = Storage()

    def _init_source(self):
        primary = self.config.get("primary_source", "tvdatafeed")

        if self.market == "US" and primary == "finnhub":
            try:
                return FinnhubSource()
            except Exception as e:
                logger.warning(f"FinnhubSource init failed: {e}, falling back to TvDatafeed")
                return TvDatafeedSource()

        if self.market == "BIST" and primary == "isyatirim":
            try:
                return IsYatirimSource()
            except Exception as e:
                logger.warning(f"IsYatirimSource init failed: {e}, falling back to TvDatafeed")
                return TvDatafeedSource()

        return TvDatafeedSource()

    def collect(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Tüm semboller için OHLCV verisi toplar.

        Returns
        -------
        pd.DataFrame – multi-symbol OHLCV, index: datetime, column: symbol included
        """
        frames: list[pd.DataFrame] = []
        failed: list[str] = []

        for symbol in symbols:
            try:
                if isinstance(self.source, TvDatafeedSource):
                    exchange = "BIST" if self.market == "BIST" else "NASDAQ"
                    df = self.source.fetch(symbol, exchange, timeframe)
                else:
                    df = self.source.fetch(symbol, start, end, timeframe)

                if not df.empty:
                    frames.append(df)
                    logger.debug(f"Collected {len(df)} bars for {symbol}")
                else:
                    failed.append(symbol)
            except Exception as e:
                logger.warning(f"Failed to collect {symbol}: {e}")
                failed.append(symbol)

        if failed:
            logger.warning(f"Failed symbols ({len(failed)}/{len(symbols)}): {failed[:10]}...")

        if not frames:
            logger.error("No data collected for any symbol")
            return pd.DataFrame()

        result = pd.concat(frames)

        # Storage'a kaydet
        self.storage.save_ohlcv(result, self.market, timeframe)

        # Agent iletişim dosyası
        out_dir = Path("outputs/data")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "agent": "data_collector",
            "market": self.market,
            "timeframe": timeframe,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "n_symbols": len(frames),
            "n_failed": len(failed),
            "total_bars": len(result),
        }
        with open(out_dir / "latest.json", "w") as f:
            json.dump(meta, f, indent=2)

        return result

    def get_index_symbols(self) -> list[str]:
        """SP500 veya BIST100 sembol listesini döndürür (config'den veya API'den)."""
        config_dir = Path("config")

        if self.market == "US":
            cfg_path = config_dir / "us.yaml"
        elif self.market == "BIST":
            cfg_path = config_dir / "bist.yaml"
        else:
            raise ValueError(f"Unknown market: {self.market}")

        # Config'de symbols listesi varsa onu kullan
        with open(cfg_path) as f:
            market_cfg = yaml.safe_load(f)

        if "symbols" in market_cfg:
            return market_cfg["symbols"]

        # Config'de yoksa API'den çek
        if self.market == "BIST" and isinstance(self.source, IsYatirimSource):
            return self.source.get_bist100_symbols()

        # US S&P500 — Finnhub indices endpoint
        if self.market == "US" and isinstance(self.source, FinnhubSource):
            try:
                constituents = self.source.client.indices_const(symbol="^GSPC")
                return sorted(constituents.get("constituents", []))
            except Exception as e:
                logger.warning(f"Failed to fetch S&P500 constituents: {e}")

        logger.warning(f"No symbol list available for {self.market}")
        return []
