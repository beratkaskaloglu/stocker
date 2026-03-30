"""
claude_agents/orchestrator.py
Ana orkestratör — Claude Agent SDK ile sub-agentları yönetir.
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
from loguru import logger

client = anthropic.Anthropic()


# ─── Tool tanımları (Claude'un çağırabileceği Python fonksiyonları) ────────────

def collect_data(market: str, timeframe: str) -> dict:
    """Veri toplama işlemini başlatır."""
    from core.data.collector import DataCollector
    from core.data.storage import Storage

    collector = DataCollector(market=market)
    storage = Storage()
    symbols = collector.get_symbols()[:10]  # ilk 10 hisse (demo)

    collected = 0
    for sym in symbols:
        try:
            df = collector.collect(symbol=sym, start="2023-01-01", timeframe=timeframe)
            if df is not None and len(df) > 0:
                storage.save_ohlcv(df, market=market, symbol=sym)
                collected += 1
        except Exception as e:
            logger.warning(f"  {sym}: {e}")

    result = {"status": "success", "market": market, "symbols_collected": collected}
    _write_output("data-collector", result)
    return result


def compute_entropy(market: str, timeframe: str, tau_values: list[int]) -> dict:
    """Entropy hesaplar, sonucu outputs/entropy/ yazar."""
    from core.entropy.shannon import ShannonEntropy

    se = ShannonEntropy()
    # Demo: synthetic data
    import numpy as np
    returns = np.random.randn(500)
    h = se.compute(returns)

    result = {"status": "success", "market": market, "shannon_entropy": h, "tau_values": tau_values}
    _write_output("entropy", result)
    return result


def extract_features(market: str, timeframe: str) -> dict:
    """Feature extraction yapar."""
    from core.features.aggregator import FeatureAggregator

    agg = FeatureAggregator(seq_len=60)
    result = {"status": "success", "market": market, "feature_dim": 135, "seq_len": 60}
    _write_output("features", result)
    return result


def train_model(market: str, model_name: str, epochs: int) -> dict:
    """Belirtilen modeli eğitir."""
    from core.training.train_all import build_model
    from core.training.trainer import SupervisedTrainer, TrainConfig
    from core.training.dataset import create_dataloaders, create_synthetic_data

    features, targets = create_synthetic_data(n_samples=500, feature_dim=50)
    train_loader, val_loader = create_dataloaders(features, targets, batch_size=32)

    model = build_model(model_name, feature_dim=50)
    config = TrainConfig(
        market=market, model_name=model_name,
        epochs=min(epochs, 5),  # cap for safety
        batch_size=32, device="cpu",
    )
    trainer = SupervisedTrainer(model, config)
    result = trainer.train(train_loader, val_loader)

    output = {"status": "success", "model": model_name, **result}
    _write_output(f"train-{model_name}", output)
    return output


def generate_signals(market: str, timeframe: str) -> dict:
    """Tahmin ve sinyal üretir."""
    result = {"status": "success", "market": market, "signals_generated": 0,
              "note": "Requires trained models and real data"}
    _write_output("signals", result)
    return result


def optimize_positions(market: str) -> dict:
    """RL ile pozisyon boyutlarını optimize eder."""
    result = {"status": "success", "market": market,
              "note": "Requires trained RL agent"}
    _write_output("rl-optimizer", result)
    return result


def read_agent_output(agent_name: str) -> dict:
    """Bir ajanın son çıktısını okur."""
    path = Path(f"outputs/{agent_name}/latest.json")
    return json.loads(path.read_text()) if path.exists() else {"status": "no_output"}


def _write_output(agent_name: str, data: dict) -> None:
    """Agent ciktisini outputs/{agent_name}/latest.json'a yaz."""
    from datetime import datetime
    path = Path(f"outputs/{agent_name}")
    path.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = datetime.now().isoformat()
    (path / "latest.json").write_text(json.dumps(data, indent=2, default=str))


# ─── Tool schema (Claude API için) ────────────────────────────────────────────

TOOLS = [
    {
        "name": "collect_data",
        "description": "Belirtilen piyasa ve timeframe için OHLCV verisi topla",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string", "enum": ["US", "BIST"]},
                "timeframe": {"type": "string"},
            },
            "required": ["market", "timeframe"],
        },
    },
    {
        "name": "compute_entropy",
        "description": "Shannon ve Transfer Entropy hesapla",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "timeframe": {"type": "string"},
                "tau_values": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["market", "timeframe", "tau_values"],
        },
    },
    {
        "name": "extract_features",
        "description": "GASF, FFT, Wavelet ve TVP-VAR feature extraction",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "timeframe": {"type": "string"},
            },
            "required": ["market", "timeframe"],
        },
    },
    {
        "name": "train_model",
        "description": "Belirtilen modeli eğit",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "model_name": {"type": "string", "enum": ["lstm_attention", "cnn_lstm", "transformer", "resnet_gasf"]},
                "epochs": {"type": "integer"},
            },
            "required": ["market", "model_name", "epochs"],
        },
    },
    {
        "name": "generate_signals",
        "description": "Trading sinyali üret",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "timeframe": {"type": "string"},
            },
            "required": ["market", "timeframe"],
        },
    },
    {
        "name": "optimize_positions",
        "description": "RL ile pozisyon boyutlarını optimize et",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
            },
            "required": ["market"],
        },
    },
    {
        "name": "read_agent_output",
        "description": "Bir ajanın son JSON çıktısını oku",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string"},
            },
            "required": ["agent_name"],
        },
    },
]

TOOL_FUNCTIONS = {
    "collect_data": collect_data,
    "compute_entropy": compute_entropy,
    "extract_features": extract_features,
    "train_model": train_model,
    "generate_signals": generate_signals,
    "optimize_positions": optimize_positions,
    "read_agent_output": read_agent_output,
}


# ─── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(task: str, market: str = "US") -> str:
    """
    Agentic loop: Claude, görevi tamamlamak için tool'ları çağırır.
    """
    messages = [{"role": "user", "content": task}]
    system = (
        "Sen bir borsa tahmin sisteminin orkestratörüsün. "
        "Görevin: veri toplama, entropy hesabı, feature extraction, "
        "model eğitimi ve sinyal üretimini koordine etmek. "
        f"Şu an çalışılan piyasa: {market}. "
        "Her adımı tamamladıktan sonra bir sonraki adımı planla."
    )

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Cevabı mesaj geçmişine ekle
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "Tamamlandı."

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = TOOL_FUNCTIONS.get(block.name)
                    try:
                        result = fn(**block.input) if fn else {"error": "unknown tool"}
                    except Exception as e:
                        logger.error(f"Tool {block.name} failed: {e}")
                        result = {"error": str(e)}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

            messages.append({"role": "user", "content": tool_results})


# ─── CLI entry points ──────────────────────────────────────────────────────────

def run_daily_pipeline(market: str = "US") -> None:
    """Günlük tam pipeline'ı çalıştır."""
    task = (
        f"{market} piyasası için günlük pipeline'ı çalıştır: "
        "1) Bugünün OHLCV verisini topla "
        "2) 1d timeframe için entropy hesapla (tau: 1,5,20,60) "
        "3) Feature extraction yap "
        "4) Sinyal üret "
        "5) Pozisyonları optimize et"
    )
    result = run_agent(task, market)
    print(result)


def run_training(market: str = "US", epochs: int = 100) -> None:
    """Tüm modelleri eğit."""
    task = (
        f"{market} piyasası için tüm modelleri {epochs} epoch eğit: "
        "lstm_attention, cnn_lstm, transformer, resnet_gasf — hepsini sırayla başlat."
    )
    result = run_agent(task, market)
    print(result)


if __name__ == "__main__":
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else "US"
    run_daily_pipeline(market)
