"""
agents/base_agent.py
Ortak agent interface — tüm agentlar bunu extend eder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AgentOutput:
    symbol: str
    market: str
    timeframe: str
    vector: np.ndarray        # Model girdisine eklenecek feature vektörü
    metadata: dict            # Ek bilgi (sentiment label, indicator değerleri vs.)


class BaseAgent(ABC):
    """
    PSEUDO:
    1. __init__(market, timeframe, config)
    2. analyze(symbol, data: pd.DataFrame) → AgentOutput
       a. Girdi hazırla
       b. Model / kural çalıştır
       c. AgentOutput döndür
    3. batch_analyze(symbols, data_dict) → dict[str, AgentOutput]
       a. Paralel veya sıralı analyze() çağırır
    4. get_vector_dim() → int
       a. AgentOutput.vector boyutunu döndür (aggregator için)
    """

    def __init__(self, market: str, timeframe: str, config: dict):
        self.market = market
        self.timeframe = timeframe
        self.config = config

    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        raise NotImplementedError

    def batch_analyze(self, symbols: list[str], data_dict: dict) -> dict:
        return {s: self.analyze(s, data_dict[s]) for s in symbols}

    @abstractmethod
    def get_vector_dim(self) -> int:
        raise NotImplementedError
