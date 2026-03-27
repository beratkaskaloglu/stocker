"""
agents/news/finbert_tr_agent.py
Türkçe finansal haber sentiment analizi — BIST piyasası.
"""
from __future__ import annotations

import pandas as pd
from agents.base_agent import BaseAgent, AgentOutput


class FinBERTTRAgent(BaseAgent):
    """
    Model: savasy/bert-base-turkish-sentiment-cased  (HuggingFace)
    Alternatif: dbmdz/bert-base-turkish-cased + custom fine-tune

    PSEUDO:
    1. __init__:
       a. Türkçe BERT tokenizer + sentiment model yükle
       b. Türkçe finans haberleri için fine-tune edilmiş checkpoint varsa yükle
    2. analyze(symbol, news_list) → AgentOutput
       a. FinBERTAgent ile aynı akış, Türkçe model ile
       b. BIST'e özgü şirket adları için entity recognition ekle
          (THYAO = Türk Hava Yolları, AKBNK = Akbank vs.)
    3. collect_bist_news(symbol: str) → list[str]
       a. Web scraper ile Türkçe haber sitelerinden çek:
          - Borsa Istanbul resmi haberler
          - KAP (Kamuyu Aydınlatma Platformu) bildirimleri
          - Finans gündem, Bloomberg HT
       b. KAP bildirimleri yüksek öncelik (özel durum açıklamaları)
    """

    VECTOR_DIM = 4
    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

    def __init__(self, market: str = "BIST", timeframe: str = "1d", config: dict = None):
        super().__init__(market, timeframe, config or {})
        self.tokenizer = None
        self.model = None

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        # TODO: implement — Türkçe haber analizi
        raise NotImplementedError

    def collect_bist_news(self, symbol: str) -> list[str]:
        # TODO: KAP + finans haberleri scraper
        raise NotImplementedError

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
