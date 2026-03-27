"""
core/data/storage.py
SQLite + CSV kayıt katmanı.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


class Storage:
    """SQLite + CSV persistence layer for OHLCV, news, and predictions."""

    def __init__(self, db_path: str = "db/stocker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol    TEXT    NOT NULL,
                    market    TEXT    NOT NULL,
                    timeframe TEXT    NOT NULL,
                    datetime  TEXT    NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    REAL,
                    PRIMARY KEY (symbol, market, timeframe, datetime)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    symbol          TEXT NOT NULL,
                    market          TEXT NOT NULL,
                    datetime        TEXT NOT NULL,
                    headline        TEXT,
                    sentiment_score REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    symbol       TEXT NOT NULL,
                    market       TEXT NOT NULL,
                    timeframe    TEXT NOT NULL,
                    datetime     TEXT NOT NULL,
                    direction    TEXT,
                    target_price REAL,
                    confidence   REAL
                )
            """)
            conn.commit()

    def save_ohlcv(self, df: pd.DataFrame, market: str, timeframe: str) -> None:
        """
        OHLCV verisini SQLite'a upsert + CSV'ye yazar.

        df must have columns: [symbol, open, high, low, close, volume]
        and a datetime index.
        """
        if df.empty:
            return

        records = df.reset_index()
        records["market"] = market
        records["timeframe"] = timeframe
        records["datetime"] = records["datetime"].astype(str)

        cols = ["symbol", "market", "timeframe", "datetime",
                "open", "high", "low", "close", "volume"]
        records = records[cols]

        with sqlite3.connect(self.db_path) as conn:
            for _, row in records.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO ohlcv
                       (symbol, market, timeframe, datetime, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row["symbol"], row["market"], row["timeframe"], row["datetime"],
                     row["open"], row["high"], row["low"], row["close"], row["volume"]),
                )
            conn.commit()

        # CSV backup
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        csv_dir = Path("outputs/data")
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{market}_{timeframe}_{date_str}.csv"
        records.to_csv(csv_path, index=False, mode="a",
                       header=not csv_path.exists())
        logger.info(f"Saved {len(records)} rows → {self.db_path} + {csv_path}")

    def load_ohlcv(
        self, symbol: str, market: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        """SQLite'dan OHLCV verisi yükler."""
        query = """
            SELECT datetime, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND market = ? AND timeframe = ?
              AND datetime >= ? AND datetime <= ?
            ORDER BY datetime
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, market, timeframe, start, end))

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df["symbol"] = symbol
        return df

    def write_agent_output(self, agent_name: str, payload: dict) -> None:
        out_dir = Path(f"outputs/{agent_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(out_dir / "latest.json", "w") as f:
            json.dump(payload, f, indent=2)

    def read_agent_output(self, agent_name: str) -> dict:
        path = Path(f"outputs/{agent_name}/latest.json")
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)
