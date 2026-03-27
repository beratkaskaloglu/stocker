"""
agents/news/finbert_tr_agent.py
Türkçe finansal haber sentiment analizi — BIST piyasası.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agents.base_agent import AgentOutput, BaseAgent

logger = logging.getLogger(__name__)

# BIST ticker → şirket adı eşlemesi (en yaygın 30+ hisse)
BIST_ENTITY_MAP: dict[str, list[str]] = {
    "THYAO": ["türk hava yolları", "thy", "türk havayolları"],
    "AKBNK": ["akbank"],
    "GARAN": ["garanti", "garanti bankası", "garanti bbva"],
    "ISCTR": ["iş bankası", "işbank", "türkiye iş bankası"],
    "YKBNK": ["yapı kredi", "yapıkredi"],
    "HALKB": ["halkbank", "halk bankası"],
    "VAKBN": ["vakıfbank", "vakıf bankası"],
    "SAHOL": ["sabancı", "sabancı holding"],
    "KCHOL": ["koç", "koç holding"],
    "TUPRS": ["tüpraş"],
    "EREGL": ["ereğli", "ereğli demir çelik", "erdemir"],
    "ASELS": ["aselsan"],
    "BIMAS": ["bim", "bim mağazaları"],
    "SISE": ["şişecam", "şişe cam"],
    "TCELL": ["turkcell"],
    "TTKOM": ["türk telekom"],
    "PGSUS": ["pegasus", "pegasus havayolları"],
    "TAVHL": ["tav", "tav havalimanları"],
    "EKGYO": ["emlak konut"],
    "KOZAL": ["koza altın"],
    "KOZAA": ["koza anadolu"],
    "FROTO": ["ford otosan"],
    "TOASO": ["tofaş"],
    "ARCLK": ["arçelik"],
    "VESTL": ["vestel"],
    "PETKM": ["petkim"],
    "SASA": ["sasa polyester", "sasa"],
    "ENKAI": ["enka"],
    "MGROS": ["migros"],
    "SOKM": ["şok", "şok market"],
}


class FinBERTTRAgent(BaseAgent):
    """
    Türkçe finansal sentiment analizi — savasy/bert-base-turkish-sentiment-cased.

    Çıktı: 4 boyutlu sentiment vektörü (FinBERTAgent ile aynı format)
        [p_positive, p_negative, p_neutral, net_sentiment]
    """

    VECTOR_DIM = 4
    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
    # Bu model 'positive' ve 'negative' iki sınıf döndürür
    LABELS = ["positive", "negative"]

    def __init__(
        self,
        market: str = "BIST",
        timeframe: str = "1d",
        config: dict | None = None,
    ):
        super().__init__(market, timeframe, config or {})
        self.max_headlines: int = self.config.get("max_headlines", 10)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None

    # ── Model lifecycle ─────────────────────────────────────────────

    def _load_model(self) -> None:
        if self.model is not None:
            return
        logger.info("Loading Turkish BERT model: %s", self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )
        self.model.to(self.device)
        self.model.eval()

    # ── Core inference ──────────────────────────────────────────────

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        """
        Softmax olasılıkları döndürür.
        savasy model 2 sınıf (positive, negative) döndürür;
        neutral skoru 1 - max(pos, neg) olarak türetilir.

        Returns shape: (n_texts, 3) — [positive, negative, neutral]
        """
        self._load_model()
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # 2-class → 3-class: neutral = 1 - |pos - neg| (düşük polarite = nötr)
        n = raw_probs.shape[0]
        probs_3 = np.zeros((n, 3), dtype=np.float32)
        probs_3[:, 0] = raw_probs[:, 0]  # positive
        probs_3[:, 1] = raw_probs[:, 1]  # negative

        polarization = np.abs(raw_probs[:, 0] - raw_probs[:, 1])
        neutral_score = 1.0 - polarization
        # Renormalize so all 3 sum to 1
        total = probs_3[:, 0] + probs_3[:, 1] + neutral_score
        probs_3[:, 0] /= total
        probs_3[:, 1] /= total
        probs_3[:, 2] = neutral_score / total

        return probs_3

    # ── Entity matching ─────────────────────────────────────────────

    @staticmethod
    def is_relevant(headline: str, symbol: str) -> bool:
        """Haberin belirli bir BIST sembolüyle ilgili olup olmadığını kontrol eder."""
        text = headline.lower()

        # Direkt ticker eşleşmesi
        if symbol.lower() in text:
            return True

        # Şirket adı eşleşmesi
        aliases = BIST_ENTITY_MAP.get(symbol, [])
        return any(alias in text for alias in aliases)

    @staticmethod
    def enrich_headline(headline: str) -> str:
        """KAP bildirimlerindeki kısaltmaları açar."""
        replacements = {
            "ÖDA": "Özel Durum Açıklaması",
            "FR": "Finansal Rapor",
            "BPP": "Bedelli Pay Programı",
            "KAR DAĞITIM": "Kar Dağıtım Kararı",
        }
        for abbr, full in replacements.items():
            headline = re.sub(
                rf"\b{re.escape(abbr)}\b", f"{abbr} ({full})", headline
            )
        return headline

    # ── Public API ──────────────────────────────────────────────────

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        """
        Türkçe haber sentiment analizi.

        Parameters
        ----------
        data : DataFrame
            'headline' ve 'datetime' sütunları beklenir.
            Opsiyonel: 'source' (KAP haberlerine yüksek ağırlık verilir).
        """
        if data.empty or "headline" not in data.columns:
            return self._neutral_output(symbol)

        df = data.sort_values("datetime", ascending=False).head(
            self.max_headlines
        )
        headlines: list[str] = [
            self.enrich_headline(h) for h in df["headline"].tolist()
        ]

        probs = self._predict_batch(headlines)  # (N, 3)

        # Ağırlık: zaman (exp decay) + KAP bonus
        n = len(headlines)
        time_weights = np.exp(np.linspace(0, 1, n))

        source_weights = np.ones(n, dtype=np.float32)
        if "source" in df.columns:
            sources = df["source"].tolist()
            for i, src in enumerate(sources):
                if src and "kap" in str(src).lower():
                    source_weights[i] = 2.0  # KAP bildirimleri 2x ağırlık

        weights = time_weights * source_weights
        weights /= weights.sum()

        weighted_probs = (probs.T * weights).T
        avg = weighted_probs.sum(axis=0)

        p_pos, p_neg, p_neu = avg[0], avg[1], avg[2]
        net_sentiment = float(p_pos - p_neg)

        vector = np.array([p_pos, p_neg, p_neu, net_sentiment], dtype=np.float32)

        label_idx = int(np.argmax(avg[:3]))
        label_names = ["positive", "negative", "neutral"]

        per_headline = [
            {
                "headline": h,
                "positive": round(float(p[0]), 4),
                "negative": round(float(p[1]), 4),
                "neutral": round(float(p[2]), 4),
            }
            for h, p in zip(headlines, probs)
        ]

        return AgentOutput(
            symbol=symbol,
            market=self.market,
            timeframe=self.timeframe,
            vector=vector,
            metadata={
                "agent": "finbert_tr",
                "model": self.MODEL_NAME,
                "n_headlines": n,
                "net_sentiment": net_sentiment,
                "label": label_names[label_idx],
                "scores": per_headline,
            },
        )

    # ── BIST haber toplama ──────────────────────────────────────────

    def collect_bist_news(self, symbol: str) -> list[dict]:
        """
        BIST hissesi için haber toplar.

        Returns list of dicts: [{"headline": ..., "datetime": ..., "source": ...}, ...]

        Not: Gerçek scraping implementasyonu data/sources altında yaşamalı.
        Bu metot placeholder olarak yapıyı gösterir.
        """
        logger.warning(
            "collect_bist_news is a stub — integrate with actual news sources "
            "(KAP, Bloomberg HT, Finans Gündem) via core/data/sources."
        )
        return []

    # ── Helpers ─────────────────────────────────────────────────────

    def _neutral_output(self, symbol: str) -> AgentOutput:
        return AgentOutput(
            symbol=symbol,
            market=self.market,
            timeframe=self.timeframe,
            vector=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            metadata={"agent": "finbert_tr", "label": "neutral", "n_headlines": 0},
        )

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
