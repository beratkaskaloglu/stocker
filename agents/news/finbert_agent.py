"""
agents/news/finbert_agent.py
FinBERT tabanlı haber sentiment analizi — US piyasası (İngilizce).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agents.base_agent import AgentOutput, BaseAgent

logger = logging.getLogger(__name__)


class FinBERTAgent(BaseAgent):
    """
    ProsusAI/finbert ile İngilizce finansal haber sentiment analizi.

    Çıktı: 4 boyutlu sentiment vektörü
        [p_positive, p_negative, p_neutral, net_sentiment]
    """

    VECTOR_DIM = 4
    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        market: str = "US",
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
        """Lazy-load: ilk analyze() çağrısında yüklenir."""
        if self.model is not None:
            return
        logger.info("Loading FinBERT model: %s", self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )
        self.model.to(self.device)
        self.model.eval()

    # ── Core inference ──────────────────────────────────────────────

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        """
        Bir metin listesi için softmax olasılıkları döndürür.
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
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ── Public API ──────────────────────────────────────────────────

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        """
        Sentiment analizi çalıştırır.

        Parameters
        ----------
        data : DataFrame
            'headline' ve 'datetime' sütunları beklenir.
            Opsiyonel: 'source', 'url'.
        """
        if data.empty or "headline" not in data.columns:
            return self._neutral_output(symbol)

        # En güncel N haberi al
        df = data.sort_values("datetime", ascending=False).head(
            self.max_headlines
        )
        headlines: list[str] = df["headline"].tolist()

        probs = self._predict_batch(headlines)  # (N, 3)

        # Zaman ağırlığı: son haberler daha önemli (exponential decay)
        n = len(headlines)
        weights = np.exp(np.linspace(0, 1, n))  # artan ağırlık (en yeni = en yüksek)
        weights /= weights.sum()

        weighted_probs = (probs.T * weights).T  # broadcast
        avg = weighted_probs.sum(axis=0)         # [p_pos, p_neg, p_neu]

        p_pos, p_neg, p_neu = avg[0], avg[1], avg[2]
        net_sentiment = float(p_pos - p_neg)     # ∈ [-1, +1]

        vector = np.array([p_pos, p_neg, p_neu, net_sentiment], dtype=np.float32)

        # Per-headline skor metadata
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
                "agent": "finbert",
                "model": self.MODEL_NAME,
                "n_headlines": n,
                "net_sentiment": net_sentiment,
                "label": self.LABELS[int(np.argmax(avg[:3]))],
                "scores": per_headline,
            },
        )

    # ── Fine-tuning ─────────────────────────────────────────────────

    def fine_tune(
        self,
        labeled_data: pd.DataFrame,
        epochs: int = 3,
        lr: float = 2e-5,
        output_dir: str = "outputs/models",
    ) -> Path:
        """
        (headline, label) çiftleri üzerinde fine-tune.
        label: 0=positive, 1=negative, 2=neutral

        Returns checkpoint path.
        """
        from torch.utils.data import DataLoader, TensorDataset

        self._load_model()
        self.model.train()

        encodings = self.tokenizer(
            labeled_data["headline"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        labels = torch.tensor(labeled_data["label"].values, dtype=torch.long)
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for input_ids, attention_mask, batch_labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            logger.info(
                "FinBERT fine-tune epoch %d/%d — loss: %.4f",
                epoch + 1,
                epochs,
                total_loss / len(loader),
            )

        self.model.eval()

        ckpt_dir = Path(output_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"finbert_{self.market}_{self.timeframe}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)
        return ckpt_path

    # ── Helpers ─────────────────────────────────────────────────────

    def _neutral_output(self, symbol: str) -> AgentOutput:
        """Haber yoksa nötr vektör döndür."""
        return AgentOutput(
            symbol=symbol,
            market=self.market,
            timeframe=self.timeframe,
            vector=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            metadata={"agent": "finbert", "label": "neutral", "n_headlines": 0},
        )

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
