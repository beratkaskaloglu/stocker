"""
agents/news/finbert_agent.py
FinBERT tabanlı haber sentiment analizi — US piyasası (İngilizce).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentOutput


class FinBERTAgent(BaseAgent):
    """
    Model: ProsusAI/finbert  (HuggingFace)
    Çıktı: 4 boyutlu sentiment vektörü

    PSEUDO:
    1. __init__:
       a. AutoTokenizer + AutoModelForSequenceClassification yükle
       b. Model: ProsusAI/finbert (positive/negative/neutral)
       c. Device: cuda veya cpu
    2. analyze(symbol, news_list: list[str]) → AgentOutput
       a. Son N haberi al (N=10, timeframe'e göre değişir)
       b. Her haber için tokenize + model forward
       c. Softmax → [p_positive, p_negative, p_neutral]
       d. Ağırlıklı ortalama (son haberler daha önemli)
       e. Sentiment vektörü: [p_pos, p_neg, p_neu, net_sentiment]
          net_sentiment = p_pos - p_neg  ∈ [-1, +1]
       f. AgentOutput(vector=4d, metadata={headlines, scores})
    3. fine_tune(labeled_data, market, timeframe):
       a. Etiketli haber → fiyat değişimi veriyle fine-tune
       b. Checkpoint: outputs/models/finbert_{market}_{timeframe}.pt
    """

    VECTOR_DIM = 4
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, market: str = "US", timeframe: str = "1d", config: dict = None):
        super().__init__(market, timeframe, config or {})
        self.tokenizer = None   # TODO: load AutoTokenizer
        self.model = None       # TODO: load AutoModel

    def _load_model(self):
        # TODO:
        # from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        # self.model.eval()
        raise NotImplementedError

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        # TODO: implement
        # data: DataFrame with 'headline', 'datetime' columns
        raise NotImplementedError

    def fine_tune(self, labeled_data: pd.DataFrame) -> None:
        # TODO: fine-tune on (headline, price_change) pairs
        raise NotImplementedError

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
