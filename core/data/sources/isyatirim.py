"""
core/data/sources/isyatirim.py
BIST piyasası için İş Yatırım veri kaynağı.
"""
from __future__ import annotations

import requests
import pandas as pd
from loguru import logger


class IsYatirimSource:
    """İş Yatırım API for BIST markets (isyatirimhisse v5.x)."""

    _BIST100_URL = (
        "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/IndexConstituents"
    )

    def __init__(self):
        from isyatirimhisse import fetch_stock_data
        self._fetch_stock_data = fetch_stock_data

    def fetch(self, symbol: str, start: str, end: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        İş Yatırım'dan OHLCV verisi çeker.

        Parameters
        ----------
        symbol : str    – örn. "THYAO"
        start  : str    – "2024-01-01" (ISO format, internally converted to DD-MM-YYYY)
        end    : str    – "2024-12-31"
        timeframe : str – sadece "1d" desteklenir (İş Yatırım günlük veri sağlar)

        Returns
        -------
        pd.DataFrame – columns: [open, high, low, close, volume], index: datetime
        """
        if timeframe != "1d":
            logger.warning(
                f"IsYatirim only supports daily data; ignoring timeframe={timeframe}"
            )

        # İş Yatırım DD-MM-YYYY formatı bekler
        start_fmt = pd.Timestamp(start).strftime("%d-%m-%Y")
        end_fmt = pd.Timestamp(end).strftime("%d-%m-%Y")

        try:
            raw = self._fetch_stock_data(
                symbols=symbol,
                start_date=start_fmt,
                end_date=end_fmt,
            )
        except Exception as e:
            logger.warning(f"IsYatirim: fetch failed for {symbol}: {e}")
            return pd.DataFrame()

        if raw.empty:
            logger.warning(f"IsYatirim: no data for {symbol}")
            return pd.DataFrame()

        # Sütun normalizasyonu
        col_map = {
            "HGDG_TARIH": "datetime",
            "HGDG_KAPANIS": "close",
            "HGDG_MAX": "high",
            "HGDG_MIN": "low",
            "HGDG_AOF": "open",
            "HGDG_HACIM": "volume",
        }

        available = {k: v for k, v in col_map.items() if k in raw.columns}
        df = raw[list(available.keys())].rename(columns=available)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()

        df["symbol"] = symbol
        return df

    def get_bist100_symbols(self) -> list[str]:
        """BIST-100 endeks bileşenlerini İş Yatırım sitesinden çeker."""
        try:
            resp = requests.get(
                "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/IndexConstituents",
                params={"index": "XU100"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            symbols = [item["HISSE_KODU"] for item in data.get("value", [])]
            if symbols:
                return sorted(symbols)
        except Exception as e:
            logger.warning(f"IsYatirim: BIST100 symbol fetch failed: {e}")

        # Fallback: hardcoded top BIST100 symbols
        logger.info("Using fallback BIST100 symbol list")
        return [
            "AEFES", "AFYON", "AGESA", "AKBNK", "AKFGY", "AKFYE", "AKSA", "AKSEN",
            "ALARK", "ALFAS", "ARCLK", "ASELS", "ASUZU", "AYGAZ", "BAGFS", "BASGZ",
            "BERA", "BIMAS", "BRYAT", "BUCIM", "CCOLA", "CEMTS", "CIMSA", "DOAS",
            "DOHOL", "ECILC", "EGEEN", "EKGYO", "ENJSA", "ENKAI", "EREGL", "EUPWR",
            "FROTO", "GARAN", "GESAN", "GUBRF", "HALKB", "HEKTS", "ISCTR", "ISGYO",
            "KCHOL", "KMPUR", "KONTR", "KOZAA", "KOZAL", "KRDMD", "MGROS", "MPARK",
            "ODAS", "OTKAR", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE",
            "SKBNK", "SOKM", "TABGD", "TATGD", "TAVHL", "TCELL", "THYAO", "TKFEN",
            "TKNSA", "TMSN", "TOASO", "TRGYO", "TTKOM", "TTRAK", "TUPRS", "ULKER",
            "VAKBN", "VESBE", "VESTL", "YKBNK", "YATAS", "ZOREN",
        ]
