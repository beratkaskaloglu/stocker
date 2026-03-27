"""
claude_agents/orchestrator.py
Ana orkestratör — Claude Agent SDK ile sub-agentları yönetir.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic

client = anthropic.Anthropic()


# ─── Tool tanımları (Claude'un çağırabileceği Python fonksiyonları) ────────────

def collect_data(market: str, timeframe: str) -> dict:
    """Veri toplama işlemini başlatır."""
    # TODO: core/data/collector.py çağır
    raise NotImplementedError


def compute_entropy(market: str, timeframe: str, tau_values: list[int]) -> dict:
    """Entropy hesaplar, sonucu outputs/entropy/ yazar."""
    # TODO: core/entropy/ çağır
    raise NotImplementedError


def extract_features(market: str, timeframe: str) -> dict:
    """Feature extraction yapar."""
    # TODO: core/features/ çağır
    raise NotImplementedError


def train_model(market: str, model_name: str, epochs: int) -> dict:
    """Belirtilen modeli eğitir."""
    # TODO: core/models/ çağır
    raise NotImplementedError


def generate_signals(market: str, timeframe: str) -> dict:
    """Tahmin ve sinyal üretir."""
    # TODO: signals/ çağır
    raise NotImplementedError


def optimize_positions(market: str) -> dict:
    """RL ile pozisyon boyutlarını optimize eder."""
    # TODO: rl/ çağır
    raise NotImplementedError


def read_agent_output(agent_name: str) -> dict:
    """Bir ajanın son çıktısını okur."""
    path = Path(f"outputs/{agent_name}/latest.json")
    return json.loads(path.read_text()) if path.exists() else {}


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
                "model_name": {"type": "string", "enum": ["lstm", "cnn_lstm", "transformer", "resnet"]},
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

    PSEUDO:
    1. System prompt: Stocker pipeline koordinatörü olduğunu söyle
    2. İlk mesaj: task
    3. Loop:
       a. API çağır (tools=TOOLS)
       b. stop_reason == "end_turn" → bitir
       c. stop_reason == "tool_use" → tool çağır → sonucu geri ver
       d. Tekrarla
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
            # Son text bloğunu döndür
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
                    except NotImplementedError:
                        result = {"status": "not_implemented_yet", "tool": block.name}
                    except Exception as e:
                        result = {"error": str(e)}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
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
    """Tüm modelleri paralel eğit."""
    task = (
        f"{market} piyasası için tüm modelleri {epochs} epoch eğit: "
        "lstm, cnn_lstm, transformer, resnet — hepsini sırayla başlat."
    )
    result = run_agent(task, market)
    print(result)


if __name__ == "__main__":
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else "US"
    run_daily_pipeline(market)
