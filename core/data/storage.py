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


class Storage:
    """
    PSEUDO:
    1. __init__: SQLite bağlantısı aç, tablolar yoksa oluştur
       Tables:
         - ohlcv (symbol, market, timeframe, datetime, open, high, low, close, volume)
         - news  (symbol, market, datetime, headline, sentiment_score)
         - predictions (symbol, market, timeframe, datetime, direction, target_price, confidence)
    2. save_ohlcv(df, market, timeframe):
       a. df → SQLite upsert (conflict: replace)
       b. df → CSV: outputs/data/{market}_{timeframe}_{date}.csv
    3. load_ohlcv(symbol, market, timeframe, start, end) → DataFrame
    4. write_agent_output(agent_name, payload: dict):
       a. outputs/{agent_name}/latest.json yaz
       b. payload: {agent, market, timeframe, timestamp, status, output_path, metadata}
    5. read_agent_output(agent_name) → dict
    """

    def __init__(self, db_path: str = "db/stocker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        # TODO: CREATE TABLE IF NOT EXISTS ...
        raise NotImplementedError

    def save_ohlcv(self, df: pd.DataFrame, market: str, timeframe: str) -> None:
        # TODO: implement
        raise NotImplementedError

    def load_ohlcv(
        self, symbol: str, market: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        # TODO: implement
        raise NotImplementedError

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
