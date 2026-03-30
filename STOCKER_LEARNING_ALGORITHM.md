# STOCKER — Öğrenme Algoritması, Modüller & Agent Mimarisi

> **Versiyon**: 1.0 | **Tarih**: Mart 2026
> **Stack**: Python 3.11 · PyTorch · Claude Agent SDK
> **Ortam**: Miniconda · GPU (eğitim) · Local (dev/inference)

---

## İçindekiler

1. [Ortam Kurulumu](#1-ortam-kurulumu)
2. [Repo Yapısı](#2-repo-yapısı)
3. [Veri Akış Diyagramı](#3-veri-akış-diyagramı)
4. [Öğrenme Algoritması](#4-öğrenme-algoritması)
5. [Modül Detayları](#5-modül-detayları)
6. [Agent Mimarisi](#6-agent-mimarisi)
7. [Claude Code Entegrasyonu](#7-claude-code-entegrasyonu)
8. [Worktree Stratejisi](#8-worktree-stratejisi)
9. [GPU Eğitim Workflow'u](#9-gpu-eğitim-workflowu)
10. [Veri Kaynakları](#10-veri-kaynakları)
11. [Pipeline Test Sonuçları](#11-pipeline-test-sonuçları-28-mart-2026)
12. [Kod Raporu](#12-kod-raporu) → `STOCKER_CODE_REPORT.md`
13. [GPU Eğitim Rehberi](#13-gpu-eğitim-rehberi)
14. [Gerçek Veri Toplama & Dataset](#14-gerçek-veri-toplama--dataset)

---

## 1. Ortam Kurulumu

### 1.1 Miniconda Environment

```bash
# Miniconda kur (yoksa)
# https://docs.conda.io/en/latest/miniconda.html

# Environment oluştur
conda create -n stocker python=3.11 -y
conda activate stocker

# CUDA destekli PyTorch (GPU için)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Temel bağımlılıklar
pip install -r requirements.txt
```

### 1.2 requirements.txt (kategorilere göre)

```
# ── Veri ────────────────────────────────────
yfinance>=0.2.36
tvdatafeed>=2.4.0
isyatirimhisse>=1.0.0
finnhub-python>=2.4.19
ccxt>=4.3.0
python-binance>=1.0.19
marketaux>=0.1.0

# ── Veri İşleme ─────────────────────────────
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.13.0
networkx>=3.3

# ── Deep Learning ───────────────────────────
torch>=2.3.0
torchvision>=0.18.0
transformers>=4.41.0        # FinBERT
scikit-learn>=1.5.0
xgboost>=2.0.3

# ── Reinforcement Learning ──────────────────
stable-baselines3>=2.3.0
gymnasium>=0.29.1

# ── Sinyal İşleme & Feature Engineering ─────
PyWavelets>=1.6.0
pyts>=0.13.0                # GASF encoding
ta-lib>=0.4.32
statsmodels>=0.14.2         # TVP-VAR

# ── Bilgi Teorisi ───────────────────────────
# Transfer Entropy KSG estimator — custom impl.
# scipy, networkx yeterli

# ── Altyapı ─────────────────────────────────
click>=8.1.7
loguru>=0.7.2
pyyaml>=6.0.1
python-dotenv>=1.0.1

# ── Claude Agent SDK ────────────────────────
anthropic>=0.30.0

# ── Test ────────────────────────────────────
pytest>=8.2.0
pytest-asyncio>=0.23.7
```

### 1.3 GitHub'a İlk Push

```bash
git init
git add requirements.txt
git commit -m "init: project scaffold + requirements"
git remote add origin https://github.com/<kullanici>/stocker.git
git push -u origin main
```

---

## 2. Repo Yapısı

```
stocker/
│
├── config/
│   ├── base.yaml               # Ortak config (timeframes, markets)
│   ├── us.yaml                 # US market config
│   ├── bist.yaml               # BIST market config
│   └── agents.yaml             # Agent parametreleri
│
├── core/
│   ├── data/
│   │   ├── collector.py        # Ana veri toplayıcı (factory pattern)
│   │   ├── sources/
│   │   │   ├── finnhub.py      # US stocks
│   │   │   ├── isyatirim.py    # BIST stocks
│   │   │   ├── tvdatafeed.py   # Fallback (US + BIST)
│   │   │   └── ccxt_source.py  # Crypto
│   │   └── storage.py          # SQLite + CSV writer
│   │
│   ├── entropy/
│   │   ├── shannon.py          # Shannon entropy per stock
│   │   ├── transfer.py         # Transfer entropy (KSG estimator)
│   │   └── graph.py            # Yönlü bilgi akış grafiği
│   │
│   ├── features/
│   │   ├── returns.py          # DR, DR² hesabı
│   │   ├── gasf.py             # GASF 2D image encoding
│   │   ├── frequency.py        # FFT + Wavelet
│   │   ├── tvpvar.py           # TVP-VAR
│   │   └── aggregator.py       # Tüm feature'ları birleştirir
│   │
│   └── models/
│       ├── lstm_attention.py   # LSTM + Multi-head Attention
│       ├── cnn_lstm.py         # CNN-LSTM hybrid
│       ├── transformer.py      # Transformer encoder
│       ├── resnet_gasf.py      # ResNet (GASF image input)
│       └── meta_learner.py     # Ensemble combiner
│
├── agents/
│   ├── base_agent.py           # Ortak agent interface
│   ├── news/
│   │   ├── finbert_agent.py    # FinBERT sentiment (EN)
│   │   ├── finbert_tr_agent.py # FinBERT-TR (BIST haberleri)
│   │   └── news_collector.py   # Finnhub / Marketaux
│   └── market/
│       ├── technical_agent.py  # Teknik analiz (ta-lib)
│       └── timeframe_agent.py  # Çoklu timeframe wrapper
│
├── rl/
│   ├── env.py                  # Custom Gymnasium trading env
│   ├── sac_agent.py            # Soft Actor-Critic (pozisyon boyutu)
│   ├── dqn_agent.py            # DQN (buy/sell/hold)
│   └── reward.py               # Pro Trader reward shaping
│
├── signals/
│   ├── generator.py            # Sinyal üretimi
│   └── error_module.py         # Predicted Error Module
│
├── backtest/
│   ├── engine.py               # Walk-forward backtest motoru
│   └── metrics.py              # Sharpe, Sortino, Max Drawdown
│
├── claude_agents/              # Claude Agent SDK ile çalışan agentlar
│   ├── orchestrator.py         # Ana orkestratör
│   ├── data_agent.py           # Veri toplama sub-agent
│   ├── train_agent.py          # Eğitim sub-agent
│   ├── predict_agent.py        # Tahmin sub-agent
│   └── rl_agent.py             # RL optimizasyon sub-agent
│
├── outputs/                    # Agent çıktıları (JSON/YAML)
│   ├── data/
│   ├── entropy/
│   ├── features/
│   ├── models/
│   ├── signals/
│   └── rl/
│
├── db/
│   └── stocker.db              # SQLite veritabanı
│
├── cli/
│   └── main.py                 # Click CLI entry point
│
├── tests/
│   ├── test_data.py
│   ├── test_entropy.py
│   ├── test_models.py
│   └── test_rl.py
│
├── .env.example                # API key template
├── requirements.txt
├── environment.yml             # Conda environment export
├── META_PLAN.md
└── STOCKER_LEARNING_ALGORITHM.md
```

---

## 3. Veri Akış Diyagramı

```mermaid
flowchart TD
    subgraph DATA["📥 Veri Toplama"]
        A1[Finnhub / yfinance\nUS OHLCV] --> DB
        A2[IsYatirim / tvDatafeed\nBIST OHLCV] --> DB
        A3[CCXT / Binance\nCrypto OHLCV] --> DB
        A4[Finnhub News\nMarketaux] --> DB
        A5[Türkçe Haber\nScraper] --> DB
        DB[(SQLite / CSV)]
    end

    subgraph ENTROPY["🔢 Bilgi Teorisi"]
        DB --> E1[DR = ln(P_t / P_t-1)\nGünlük Getiri]
        E1 --> E2[DR² = Volatilite Proxy]
        E1 --> E3[Shannon Entropy H(X)\nTahmin edilebilirlik skoru]
        E1 --> E4[Transfer Entropy TE(X→Y)\nKSG Estimator\nmulti-tau: 30m,1h,4h,1d,5d,20d,60d]
        E4 --> E5[Net Transfer Entropy\nNTE = TE(X→Y) - TE(Y→X)\nYönlü Etki Grafiği]
    end

    subgraph FEATURES["⚙️ Feature Engineering"]
        DB --> F1[GASF Encoding\n2D Image]
        DB --> F2[FFT + Wavelet\nFrekans Analizi]
        DB --> F3[TVP-VAR\nDinamik İlişkiler]
        E3 --> F4[Entropy Features]
        E5 --> F4
    end

    subgraph AGENTS["🤖 AI Agentlar"]
        DB --> AG1[FinBERT Agent EN\nUS Haber Sentiment]
        DB --> AG2[FinBERT-TR Agent\nBIST Haber Sentiment]
        DB --> AG3[Market Agent\nTA-Lib Teknik Analiz]
        AG1 --> AGV[Sentiment Vektörleri\nTimeframe-aware]
        AG2 --> AGV
        AG3 --> AGV
    end

    subgraph MODELS["🧠 Multi-Branch Ensemble"]
        F1 --> M4[ResNet\nGASF Image]
        F2 --> M2[CNN-LSTM\nFrekans Patternler]
        F3 --> M1[LSTM + Attention\nDinamik Zaman Serisi]
        F4 --> M1
        AGV --> M3[Transformer\nSentiment + Market]
        M1 & M2 & M3 & M4 --> ML[Meta-Learner\nEnsemble Combiner]
    end

    subgraph SIGNAL["📊 Sinyal Üretimi"]
        ML --> S1[Yön Tahmini\nUp/Down]
        ML --> S2[Hedef Fiyat]
        ML --> S3[Güven Skoru]
        S1 & S2 & S3 --> EM[Error Module\nTahmin Güvenilirliği]
        EM --> SIG[Ham Sinyal\nBuy/Sell/Hold + Confidence]
    end

    subgraph RL["🎯 RL Katmanı"]
        SIG --> RL1[DQN\nBuy/Sell/Hold Kararı]
        SIG --> RL2[SAC\nPozisyon Boyutu 0.0-1.0]
        RL1 & RL2 --> FINAL[Final Trade Kararı\nSinyal + Boyut + Stop/Take]
    end

    FINAL --> OUT[📤 Çıktı\nJSON: outputs/signals/latest.json]
```

---

## 4. Öğrenme Algoritması

Sistem **3 aşamalı öğrenme döngüsünde** çalışır:

### Aşama 1: Supervised Pre-Training

**Ne öğrenir?** Geçmiş fiyat ve haber verisinden gelecekteki fiyat yönünü.

```
Giriş:  [OHLCV features, entropy, GASF images, sentiment vectors]
        ↓ shape: (batch=64, seq_len=60, features=N)
Çıkış:  [direction: {-1,0,+1}, target_price: float, confidence: [0,1]]
Kayıp:  CrossEntropy (yön) + MSE (fiyat) + BCE (confidence)
```

**Walk-Forward Validasyon (zorunlu):**
```
Zaman: ────────────────────────────────────────>
Train: [────────][  val  ]
       [──────────────][  val  ]
       [────────────────────][  val  ]
       ...
```
- Window size: 252 gün (1 yıl)
- Step size: 21 gün (1 ay)
- Min 5 fold

**Overfitting önleme:**

| Teknik | Değer |
|--------|-------|
| Dropout | 0.3 – 0.5 |
| L2 Regularization | weight_decay=1e-4 |
| Early Stopping | patience=15 epoch |
| Data Augmentation | jittering, window slicing, magnitude warping |
| Ensemble Diversity | 4 farklı mimari |

---

### Aşama 2: Agent Fine-Tuning

**Ne öğrenir?** FinBERT agentı belirli piyasa + timeframe kombinasyonuna göre sentiment çıkarımını iyileştirir.

```
Giriş:  [Haber metni] → FinBERT tokenizer
        [Timeframe embedding: 30m, 1h, 1d, ...]
Çıkış:  [sentiment_score: -1.0 to +1.0, impact_magnitude: float]
Kayıp:  MSE(predicted_sentiment_impact, actual_price_change)
```

- News agent ve market agent **ayrı** fine-tune edilir
- Her timeframe için **ayrı** checkpoint kaydedilir

---

### Aşama 3: RL Online Learning

**Ne öğrenir?** Supervised sinyalleri gerçek trading kısıtlarına göre optimize eder.

#### DQN (Discrete Action Space)
```
State:   [fiyat features, sinyal, haber sentiment, portföy durumu]
Action:  {0: Hold, 1: Buy, 2: Sell}
Reward:  Pro Trader şekillendirmesi (aşağıya bkz.)
```

#### SAC (Continuous Action Space)
```
State:   [aynı]
Action:  position_size ∈ [-1.0, +1.0]  (-1=full short, +1=full long)
Reward:  aynı
```

#### Pro Trader Reward Fonksiyonu
```python
def reward(portfolio_return, trade_cost, drawdown, holding_days):
    r = portfolio_return
    r -= trade_cost * 0.001        # komisyon cezası
    r -= max(0, drawdown - 0.02)   # %2 üzeri drawdown cezası
    r += 0.0001 * holding_days     # uzun vadeli tutmaya bonus
    return r
```

**RL Eğitim Parametreleri:**
| Parametre | Değer |
|-----------|-------|
| Episodes | 5,000 – 10,000 |
| Replay Buffer | 100,000 |
| Batch Size | 256 |
| Learning Rate | 3e-4 |
| Gamma (discount) | 0.99 |

---

## 5. Modül Detayları

### 5.1 `core/data/` — Veri Toplama

```python
# Interface
class DataCollector:
    def collect(market: str, symbols: list[str],
                start: str, end: str,
                timeframe: str) -> pd.DataFrame
    # Çıktı: outputs/data/{market}_{timeframe}_{date}.csv
```

| Modül | Giriş | Çıkış |
|-------|-------|-------|
| `finnhub.py` | ticker, start, end, tf | OHLCV DataFrame |
| `isyatirim.py` | ticker (.IS), start, end | OHLCV DataFrame |
| `tvdatafeed.py` | exchange, ticker, tf | OHLCV DataFrame |
| `storage.py` | DataFrame | SQLite + CSV |

---

### 5.2 `core/entropy/` — Bilgi Teorisi

```python
# Interface
class ShannonEntropy:
    def compute(returns: np.ndarray, bins=50) -> float
    # H(X) = -Σ p(x_i) * ln(p(x_i))

class TransferEntropy:
    def compute(x: np.ndarray, y: np.ndarray,
                tau: int, k=1, l=1) -> float
    # KSG estimator — GPU paralel hesap mümkün
    # S&P 500: 124,750 çift × N tau değeri = yoğun hesap

class InfoGraph:
    def build(te_matrix: np.ndarray) -> nx.DiGraph
    # NTE(X→Y) = TE(X→Y) - TE(Y→X)
    # Pozitif kenar = X, Y'yi etkiliyor
```

**Hesaplama notu**: S&P 500 için 124,750 hisse çifti × 7 tau değeri = **~870,000 TE hesabı**. GPU vektörizasyonu zorunlu.

---

### 5.3 `core/features/` — Feature Engineering

```python
# Interface
class FeatureAggregator:
    def transform(ohlcv: pd.DataFrame,
                  entropy_features: dict,
                  sentiment_vectors: dict) -> np.ndarray
    # Çıktı: (seq_len, total_features) tensor
```

| Feature | Boyut | Açıklama |
|---------|-------|----------|
| DR, DR² | 2 | Günlük getiri + volatilite |
| Shannon Entropy | 1 | Tahmin edilebilirlik |
| TE Graph Features | 10 | In/out degree, centrality |
| GASF Image | 64×64 | ResNet için 2D encoding |
| FFT Spectrum | 32 | Frekans bileşenleri |
| Wavelet Coeffs | 48 | Çok ölçekli zaman-frekans |
| TVP-VAR | 20 | Dinamik regresyon katsayıları |
| TA-Lib Indicators | 30 | RSI, MACD, Bollinger vs. |
| Sentiment Vector | 4 | FinBERT çıktısı |
| **Toplam** | **~211** | Tam feature vektörü |

---

### 5.4 `core/models/` — Ensemble Modeller

```python
# Her model aynı interface'i uygular
class BaseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> ModelOutput
    # ModelOutput: direction_logits, target_price, confidence

class MetaLearner(nn.Module):
    """4 modelin çıktısını birleştiren hafif MLP"""
    def forward(self, outputs: list[ModelOutput]) -> FinalPrediction
```

| Model | Input | Özellik |
|-------|-------|---------|
| LSTM + Attention | (B, T, F) | Uzun dönem bağımlılık |
| CNN-LSTM | (B, T, F) | Yerel + global pattern |
| Transformer | (B, T, F) | Self-attention, paralel |
| ResNet-GASF | (B, 3, 64, 64) | Image-based pattern |

---

### 5.5 `rl/env.py` — Trading Environment

```python
class StockerTradingEnv(gymnasium.Env):
    observation_space: Box(shape=(state_dim,))
    action_space: Discrete(3)  # DQN için
    # veya Box(-1, 1)          # SAC için

    def step(self, action) -> (obs, reward, done, info)
    def reset() -> obs

    # State: [normalized_prices, signals, sentiment,
    #         portfolio_value, position, days_held]
```

---

### 5.6 `signals/error_module.py` — Hata Tahmini

```
Gereksinim: Minimum 6 ay geçmiş tahmin verisi (cold start var)
Çalışma: predicted_error = f(model_confidence, entropy, volatility)
Kullanım: final_signal = raw_signal × (1 - predicted_error)
```

---

## 6. Agent Mimarisi

### 6.1 Ana Agentlar (Claude Agent SDK)

```
Orchestrator Agent
│
├── DataCollectorAgent     — Veri çek, DB'ye yaz
├── EntropyAgent           — Entropy hesapla
├── FeatureAgent           — Feature extraction
├── TrainAgent             — Model eğitimi koordine et
│   ├── LSTMTrainerSubAgent    (sub)
│   ├── CNNLSTMTrainerSubAgent (sub)
│   ├── TransformerTrainerSubAgent (sub)
│   └── ResNetTrainerSubAgent  (sub)
├── PredictAgent           — Sinyal üret
│   ├── SignalGeneratorSubAgent (sub)
│   └── ErrorModuleSubAgent    (sub)
└── RLAgent                — Pozisyon optimize et
    ├── DQNSubAgent            (sub)
    └── SACSubAgent            (sub)
```

### 6.2 Agent İletişim Protokolü (JSON)

Her agent çalışmasını tamamlayınca şu yapıda çıktı yazar:

```json
// outputs/{agent_name}/latest.json
{
  "agent": "entropy-agent",
  "market": "US",
  "timeframe": "1d",
  "timestamp": "2026-03-28T18:00:00Z",
  "status": "success",
  "output_path": "outputs/entropy/US_1d_20260328.parquet",
  "metadata": {
    "n_stocks": 503,
    "n_pairs": 124750,
    "tau_values": [1, 5, 20, 60]
  }
}
```

Bağımlı agent bu dosyayı okuyup `status: success` kontrolü yaparak başlar.

### 6.3 `claude_agents/orchestrator.py` — Örnek Akış

```python
from anthropic import Anthropic

client = Anthropic()

async def run_daily_pipeline(market: str):
    """Her gün piyasa kapanışında çalışır"""

    # 1. Veri topla
    data_result = await run_subagent(
        name="data-collector",
        task=f"Collect {market} OHLCV data for today",
        tools=[collect_data_tool, save_to_db_tool]
    )

    # 2. Entropy hesapla (data hazır olduktan sonra)
    entropy_result = await run_subagent(
        name="entropy-agent",
        task=f"Compute Shannon and Transfer Entropy for {market}",
        tools=[load_data_tool, compute_entropy_tool, save_graph_tool]
    )

    # 3. Feature extraction + model inference paralel
    feature_result, news_result = await asyncio.gather(
        run_subagent("feature-agent", f"Extract features for {market}", ...),
        run_subagent("news-agent", f"Analyze news sentiment for {market}", ...)
    )

    # 4. Sinyal üret
    signal_result = await run_subagent(
        name="predict-agent",
        task=f"Generate trading signals for {market}",
        tools=[load_model_tool, predict_tool, error_module_tool]
    )

    # 5. RL optimizasyon
    rl_result = await run_subagent(
        name="rl-agent",
        task="Optimize position sizes based on signals",
        tools=[load_rl_model_tool, optimize_positions_tool]
    )

    return rl_result
```

### 6.4 Agent Yapılandırması (`config/agents.yaml`)

```yaml
agents:
  data_collector:
    model: claude-opus-4-6
    max_tokens: 4096
    schedule: "0 18 * * 1-5"   # Hafta içi 18:00
    markets: [US, BIST]
    timeframes: [30m, 1h, 1d]

  entropy:
    model: claude-opus-4-6
    depends_on: data_collector
    tau_values:
      30m: [1, 2, 4, 8]
      1h:  [1, 4, 8]
      1d:  [1, 5, 20, 60]

  train:
    model: claude-opus-4-6
    schedule: manual            # GPU gerektirir, manuel başlatılır
    sub_agents:
      - lstm_trainer
      - cnn_lstm_trainer
      - transformer_trainer
      - resnet_trainer
    parallel: true              # 4 model paralel eğitilir

  predict:
    model: claude-opus-4-6
    schedule: "0 9 * * 1-5"    # Hafta içi 09:00
    depends_on: [entropy, data_collector]

  rl:
    model: claude-opus-4-6
    schedule: manual            # Aylık re-training
    algorithm: [sac, dqn]
    episodes: 5000
```

---

## 7. Claude Code Entegrasyonu

### 7.1 CLI Komutları

```bash
# Veri toplama
python cli/main.py collect --market US --start 2020-01-01
python cli/main.py collect --market BIST --start 2015-01-01

# Entropy hesaplama
python cli/main.py entropy --market US --timeframe 1d --tau 1,5,20,60
python cli/main.py entropy --market BIST --timeframe 1d --tau 1,5,20,60

# Feature extraction
python cli/main.py features --market US --timeframe 1d

# Model eğitimi (GPU gerektirir)
python cli/main.py train --market US --epochs 100 --device cuda
python cli/main.py train --market BIST --epochs 100 --device cuda

# Agent fine-tuning
python cli/main.py finetune --agent news --market US --timeframe 1d
python cli/main.py finetune --agent news --market BIST --timeframe 1d

# RL eğitimi
python cli/main.py train-rl --market US --algo sac --episodes 5000
python cli/main.py train-rl --market BIST --algo dqn --episodes 5000

# Tahmin
python cli/main.py predict --symbol AAPL --timeframe 1d
python cli/main.py predict --symbol THYAO --market BIST --timeframe 1d

# Backtest
python cli/main.py backtest --market US --start 2023-01-01 --end 2024-01-01
python cli/main.py backtest --market BIST --start 2023-01-01

# Tam pipeline (orchestrator)
python cli/main.py run-pipeline --market US
python cli/main.py run-pipeline --market BIST
```

### 7.2 Claude Code Cron (RemoteTrigger)

```yaml
# .claude/triggers.yaml
triggers:
  daily_data_us:
    schedule: "0 23 * * 1-5"    # ABD kapanış (23:00 TR)
    command: python cli/main.py collect --market US

  daily_data_bist:
    schedule: "0 18 * * 1-5"    # BIST kapanış (18:00 TR)
    command: python cli/main.py collect --market BIST

  daily_predict_us:
    schedule: "30 14 * * 1-5"   # ABD açılış öncesi (14:30 TR)
    command: python cli/main.py predict-all --market US

  daily_predict_bist:
    schedule: "0 10 * * 1-5"    # BIST açılış (10:00 TR)
    command: python cli/main.py predict-all --market BIST

  weekly_backtest:
    schedule: "0 8 * * 6"       # Cumartesi sabah
    command: python cli/main.py backtest --market US --market BIST
```

---

## 8. Worktree Stratejisi

### 8.1 Neden Worktree?

Her modül bağımsız geliştirilebilir. Model eğitimi sürerken sinyal modülü ayrı branch'te ilerler. 4 model paralel geliştirilebilir.

### 8.2 Worktree Kurulumu

```bash
# Ana repo hazır olduktan sonra

# Core data + entropy
git worktree add ../stocker-core feat/core-data-entropy
# Core modeller (derin öğrenme)
git worktree add ../stocker-models feat/deep-learning-models
# AI Agentlar (FinBERT + market)
git worktree add ../stocker-agents feat/ai-agents
# RL katmanı
git worktree add ../stocker-rl feat/reinforcement-learning
# Sinyal + hata modülü
git worktree add ../stocker-signals feat/signal-generation
# Backtest motoru
git worktree add ../stocker-backtest feat/backtest-engine
# Claude agent entegrasyonu
git worktree add ../stocker-claude feat/claude-agents

# Tüm worktree'leri listele
git worktree list
```

### 8.3 Paralel Geliştirme Planı

```
Faz 1 (Paralel):  stocker-core  ──────────────>|
                  stocker-models ─────────────>|
                                               |
Faz 2 (Paralel):                    stocker-agents ──>|
                                    stocker-rl ───────>|
                                                       |
Faz 3 (Sıralı):                                  stocker-signals ──>
                                                  stocker-backtest ─>
                                                  stocker-claude ───>
                                                                     |
                                                                 MERGE → main
```

### 8.4 Worktree'de Claude Agents Kullanımı

```bash
# Her worktree'de ayrı Claude Code session açabilirsin
cd ../stocker-models
claude  # Sadece model eğitimiyle ilgili kodlara odaklanır

cd ../stocker-rl
claude  # Sadece RL kodlarına odaklanır
```

---

## 9. GPU Eğitim Workflow'u

### 9.1 GPU Seçenekleri

| Servis | GPU | Fiyat | Not |
|--------|-----|-------|-----|
| RunPod | A100 40GB | ~$1.5/saat | En ucuz |
| Vast.ai | A100/H100 | ~$1-3/saat | Spot pricing |
| Lambda Labs | A100 80GB | ~$1.9/saat | Sabit fiyat |
| Google Colab Pro+ | A100 | ~$50/ay | Kolay kurulum |

### 9.2 GPU'ya Kod Taşıma

```bash
# requirements.txt ve kod GPU'ya kopyala
rsync -avz ./stocker/ user@gpu-server:/workspace/stocker/

# Conda env kur
conda create -n stocker python=3.11 -y
conda activate stocker
pip install -r requirements.txt

# Eğitimi başlat (tmux içinde)
tmux new -s training
python cli/main.py train --market US --epochs 100 --device cuda
python cli/main.py train --market BIST --epochs 100 --device cuda
python cli/main.py train-rl --market US --algo sac --episodes 5000

# Modeli geri al
rsync -avz user@gpu-server:/workspace/stocker/outputs/models/ ./outputs/models/
```

### 9.3 Eğitim Sonrası Local'e Dönüş

```bash
# Eğitilmiş modeller local'e alındıktan sonra
# Tahmin ve live trading local'de çalışır (CPU yeterli)
python cli/main.py predict --symbol AAPL --timeframe 1d --device cpu
```

---

## 10. Veri Kaynakları

### 10.1 US (S&P 500)

| Kaynak | Paket | Ücretsiz Limit | Kullanım |
|--------|-------|----------------|----------|
| **Finnhub** | `finnhub-python` | 60 req/min, real-time | Ana kaynak |
| **tvDatafeed** | `tvdatafeed` | Sınırsız (TradingView scraper) | Fallback + GASF verisi |
| yfinance | `yfinance` | Sınırsız (unstable) | Son fallback |
| EODHD | `eodhd` | 20 call/gün | Fundamentals ($60/ay) |

### 10.2 BIST

| Kaynak | Paket | Ücretsiz Limit | Kullanım |
|--------|-------|----------------|----------|
| **IsYatirim** | `isyatirimhisse` | Sınırsız | Ana kaynak |
| **tvDatafeed** | `tvdatafeed` | Sınırsız | Fallback (BIST.IS) |
| Yahoo Finance | `yfinance` | Sınırsız | THYAO.IS, AKBNK.IS vs. |

### 10.3 Kripto

| Kaynak | Paket | Ücretsiz Limit | Kullanım |
|--------|-------|----------------|----------|
| **CCXT** | `ccxt` | Rate limit per exchange | Ana kaynak (107 exchange) |
| **Binance** | `python-binance` | 1200 req/min | Real-time |

### 10.4 Haber / Sentiment

| Kaynak | Dil | Paket | Kullanım |
|--------|-----|-------|----------|
| **Finnhub News** | EN | `finnhub-python` | US hisselerine özel |
| **Marketaux** | EN | REST API | Genel finansal haber |
| **FinBERT-TR** | TR | `transformers` | BIST haber scraping |

### 10.5 `.env.example`

```bash
# API Keys
FINNHUB_API_KEY=your_key_here
MARKETAUX_API_KEY=your_key_here
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_key_here
EODHD_API_KEY=your_key_here           # Opsiyonel (fundamentals)

# DB
SQLITE_DB_PATH=db/stocker.db

# Model
DEVICE=cuda                            # cuda veya cpu
MODEL_CHECKPOINT_DIR=outputs/models/

# Claude API (sub-agents için)
ANTHROPIC_API_KEY=your_key_here
```

---

## 11. Pipeline Test Sonuçları (28 Mart 2026)

### 11.1 Test Durumu: 17/17 PASSED

```
tests/test_train_pipeline.py — 7.35s
├── test_returns_calculator        ✅ Log returns DR, DR² hesabı
├── test_shannon_entropy           ✅ H_norm ∈ [0,1] doğrulandı
├── test_transfer_entropy          ✅ KSG estimator — bağımsız serilerde TE ≈ 0
├── test_info_flow_graph           ✅ NetworkX DiGraph + 5 centrality feature
├── test_gasf_encoder              ✅ 64×64 GASF image, cos(phi_i + phi_j)
├── test_lstm_attention            ✅ Forward pass — 3 head output
├── test_cnn_lstm                  ✅ Forward pass — Conv1d×3 + LSTM
├── test_transformer               ✅ Forward pass — positional encoding
├── test_resnet_gasf               ✅ Forward pass — ResNet-18 modified
├── test_meta_learner              ✅ Stacking + weighted mode
├── test_full_model_pipeline       ✅ 4 model → meta-learner zinciri
├── test_signal_generator          ✅ Error-adjusted confidence filtering
├── test_error_module              ✅ XGBoost cold start + auto-retrain
├── test_trading_env               ✅ DQN + SAC env step/reset
├── test_rl_training_short         ✅ DQN loss=0.0013, SAC actor_loss=-1.97
├── test_backtest_walkforward      ✅ 12 fold walk-forward split
└── test_training_loop             ✅ Supervised loss: 1.38 → 1.17 (10 step)
```

### 11.2 Düzeltilen Buglar

| Bug | Dosya | Sorun | Çözüm |
|-----|-------|-------|-------|
| **GASF NotImplemented** | `core/features/gasf.py` | Worktree agent encode() yazmamıştı | `arccos(x_norm) → cos(phi_i + phi_j)` tam implementasyon |
| **TE KSG Yanlış Formül** | `core/entropy/transfer.py` | Bağımsız serilerde TE ≈ 3.09 çıkıyordu | 4-space KSG: `ψ(k) - <ψ(n_xz+1)> - <ψ(n_marginal+1)> + <ψ(n_z+1)>` |
| **SAC Env Step** | `rl/env.py:99` | `float(np.clip(action,...))` array'de hata | `float(np.clip(np.asarray(action).flatten()[0], -1, 1))` |
| **SAC Predict** | `rl/sac_agent.py:81` | Aynı array-to-float bug | `float(np.asarray(action).flatten()[0])` |
| **XGBoost libomp** | macOS dependency | `libomp` eksikti | `brew install libomp` |

### 11.3 Henüz Implement Edilmemiş Modüller

| Modül | Durum | Öncelik |
|-------|--------|---------|
| `core/features/frequency.py` | NotImplementedError — FFT + Wavelet | Orta |
| `core/features/aggregator.py` | NotImplementedError — Feature birleştirme | Yüksek |
| `core/features/tvpvar.py` | Dosya yok — TVP-VAR | Düşük |

---

## 12. Eğitim Altyapısı & Monitoring

### 12.1 Yeni Modüller

| Dosya | Açıklama |
|-------|----------|
| `core/training/trainer.py` | SupervisedTrainer — TensorBoard + wandb + early stopping + grad clip + checkpoint |
| `core/training/dataset.py` | StockerDataset + DataLoader factory + synthetic data generator |
| `core/training/train_all.py` | 4 model + ResNet GASF pipeline — tek komutla eğitim |

### 12.2 Monitoring Araçları

**TensorBoard (her zaman aktif):**
```bash
# Eğitim sırasında veya sonrasında:
tensorboard --logdir outputs/logs/US
# Tarayıcıda: http://localhost:6006
```

İzlenebilen metrikler:
- `Loss/train`, `Loss/val` — toplam kayıp
- `Loss/train_direction`, `Loss/val_direction` — yön tahmini kaybı
- `Loss/train_price`, `Loss/val_price` — fiyat kaybı
- `Loss/train_confidence`, `Loss/val_confidence` — güven kaybı
- `Accuracy/train`, `Accuracy/val` — yön doğruluğu
- `Training/lr` — learning rate (ReduceLROnPlateau)
- `Training/grad_norm` — gradient norm

**Weights & Biases (opsiyonel, remote monitoring):**
```bash
pip install wandb && wandb login
python -m core.training.train_all --market US --epochs 100 --wandb
# wandb.ai dashboard'dan izle (telefon/tablet dahil)
```

### 12.3 Eğitim Komutları

```bash
# Local test (CPU, synthetic data)
python -m core.training.train_all --market US --epochs 3 --device cpu

# GPU eğitimi (gerçek data ile)
python -m core.training.train_all --market US --epochs 100 --device cuda --data data/US_features.npz

# wandb ile remote monitoring
python -m core.training.train_all --market US --epochs 100 --device cuda --wandb

# BIST eğitimi
python -m core.training.train_all --market BIST --epochs 100 --device cuda --data data/BIST_features.npz
```

### 12.4 Çıktı Yapısı

```
outputs/
├── models/
│   └── US/
│       ├── lstm_attention/
��       │   ├── best.pt          ← en iyi val_loss checkpoint
│       │   ├── final.pt         ← son epoch
│       │   └── epoch_010.pt     ← periyodik checkpoint
��       ├── cnn_lstm/best.pt
│       ├── transformer/best.pt
│       ├── resnet_gasf/best.pt
│       └── training_summary.json  ← tüm modellerin özet metrikleri
└── logs/
    └── US/
        ├── lstm_attention/
        │   ├── events.out.tfevents.*  ← TensorBoard log
        │   └── training_history.json  ← epoch-by-epoch JSON
        ├── cnn_lstm/
        ├── transformer/
        └── resnet_gasf/
```

### 12.5 CPU Test Sonucu (3 epoch, synthetic data)

```
Model                val_loss   Süre
─────────────────────────────────────
lstm_attention       1.4557     27.7s
cnn_lstm             1.4554     12.0s
transformer          1.4500     21.4s
resnet_gasf          1.4627    282.4s (GASF encode dahil)
─────────────────────────────────────
TOPLAM                          5.7 dk
```

---

## 13. GPU Eğitim Rehberi

### 13.1 Nerede Başlatılır?

| Servis | GPU | Fiyat | Tahmini Süre (100 epoch) | Öneri |
|--------|-----|-------|--------------------------|-------|
| **RunPod** | A100 40GB | ~$0.5-1.5/saat | ~1 saat | En ucuz, community image |
| **Vast.ai** | A100/H100 | ~$0.5-3/saat | ~1 saat | Spot pricing, esnek |
| **Lambda Labs** | A100 80GB | ~$1.9/saat | ~1 saat | Sabit fiyat, güvenilir |
| **Colab Pro+** | A100 | ~$50/ay | ~1-2 saat | Kolay kurulum |

### 13.2 GPU'ya Deploy

```bash
# 1. Kodu kopyala
rsync -avz --exclude='outputs/' --exclude='.git/' --exclude='__pycache__/' \
  ./stocker/ user@gpu-server:/workspace/stocker/

# 2. Environment kur
ssh user@gpu-server
conda create -n stocker python=3.11 -y && conda activate stocker
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install tensorboard wandb

# 3. tmux içinde eğitimi başlat
tmux new -s training
python -m core.training.train_all --market US --epochs 100 --device cuda --wandb

# 4. TensorBoard'u tunnel ile izle
# Local terminalde:
ssh -L 6006:localhost:6006 user@gpu-server
tensorboard --logdir /workspace/stocker/outputs/logs/US --port 6006

# 5. Modeli geri al
rsync -avz user@gpu-server:/workspace/stocker/outputs/models/ ./outputs/models/
```

---

## 14. Gerçek Veri Toplama & Dataset (28 Mart 2026)

### 14.1 Veri Kaynakları

| Kaynak | Veri | Maliyet | Rate Limit |
|--------|------|---------|------------|
| **yfinance** | OHLCV (günlük) | Ücretsiz | Yok |
| **Finnhub** | Şirket haberleri | Ücretsiz (60 req/min) | 1.5s/istek |

### 14.2 Toplanan Veri (İlk Batch)

```
Tarih aralığı: 2024-01-01 → 2026-03-28
Semboller: S&P 500 top 50
OHLCV: 2,805 satır (data/ohlcv_2024-01-01_2026-03-28.csv)
Haberler: 4,817 makale (data/news_2024-01-01_2026-03-28.json)
```

### 14.3 Dataset Pipeline (`scripts/build_dataset.py`)

```
Ham Veri → Feature Engineering → Normalization → .npz

Adımlar:
1. OHLCV yükle, sembol bazında ayır
2. Her sembol için 125 feature hesapla:
   - DR, DR², Rolling Stats (6)
   - Rolling Shannon Entropy (1)
   - FFT top-32 components (64)
   - Wavelet decomposition (20)
   - Teknik indikatörler — ta kütüphanesi (30)
   - FinBERT sentiment (4)
3. Label üret: close pct_change > ±1% → Up/Down, else Hold
4. Sliding window: (N, 60, 125)
5. IQR-based robust normalization
6. Kaydet: data/US_dataset.npz
```

### 14.4 Kullanım

```bash
# FinBERT ile (MPS/CPU, ~10-15 dk)
conda activate stocker
python scripts/build_dataset.py \
  --ohlcv data/ohlcv_2024-01-01_2026-03-28.csv \
  --news data/news_2024-01-01_2026-03-28.json \
  --out data/US_dataset.npz

# FinBERT olmadan (hızlı test, keyword sentiment)
python scripts/build_dataset.py \
  --ohlcv data/ohlcv_2024-01-01_2026-03-28.csv \
  --news data/news_2024-01-01_2026-03-28.json \
  --out data/US_dataset.npz --no-finbert
```

---

## Sonraki Adımlar

```
[x] 1. Git repo init + GitHub remote
[x] 2. Miniconda env + requirements.txt commit
[x] 3. Worktree'leri oluştur (git worktree add ...)
[x] 4. core/data modülünü yaz — Finnhub + IsYatirim collector
[x] 5. core/entropy modülünü yaz — KSG estimator
[x] 6. core/features modülünü yaz — GASF ✅ | FFT ✅ | Wavelet ✅ | Aggregator ✅
[x] 7. core/models — 4 mimari + meta-learner
[x] 8. agents/ — FinBERT + FinBERT-TR + market agent
[x] 9. rl/ — SAC + DQN + trading env
[x] 10. signals/ — generator + error module
[x] 11. backtest/ — walk-forward engine
[x] 12. claude_agents/ — orchestrator + sub-agents
[x] 13. Pipeline testi — 17/17 PASSED ✅
[x] 14. Training altyapısı — TensorBoard + wandb + early stopping + checkpoint
[x] 15. 4 model eğitim testi — CPU'da 3 epoch başarılı (5.7 dk)
[x] 16. Gerçek veri toplama — yfinance OHLCV + Finnhub news (S&P 500 top 50)
[x] 17. build_dataset.py — OHLCV+News → .npz (FinBERT sentiment + 125 feature)
[ ] 18. FinBERT ile sentiment hesapla + dataset oluştur
[ ] 19. GPU'da tam eğitim (100 epoch + 5000 episode RL, gerçek veri)
[ ] 20. Backtest + değerlendirme (gerçek veriyle)
[ ] 21. Live trading bağlantısı
```
