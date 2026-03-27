# META PLAN — Stocker Öğrenme Algoritması MD'si Nasıl Oluşturulacak

> Bu dosya, `STOCKER_LEARNING_ALGORITHM.md` adlı ana belgenin **nasıl inşa edileceğini** tanımlar.
> Ana belgeyi yazmadan önce bu plana göre alignment sağlanacak.

---

## 1. Ana Belgenin Amacı

`STOCKER_LEARNING_ALGORITHM.md` şunları kapsayacak:

| Bölüm | İçerik |
|-------|--------|
| **A. Öğrenme Algoritması** | Sistemin nasıl öğrendiği — veri → feature → model → sinyal zinciri |
| **B. Kod Şeması & Modüller** | Klasör yapısı, modül sınırları, veri akış diyagramı |
| **C. Agent Mimarisi** | Ana agentlar, sub-agentlar, aralarındaki mesaj akışı |
| **D. Claude Code Entegrasyonu** | Agentları Claude Code'dan nasıl çalıştıracağız |
| **E. Worktree Stratejisi** | Her modül/agent için ayrı worktree planı |

---

## 2. Bölüm A — Öğrenme Algoritması Yapısı

### 2.1 Ne Anlatılacak
- Sistemin 3 aşamalı öğrenme döngüsü:
  1. **Supervised Pre-training** (LSTM/CNN/Transformer/ResNet ensemble)
  2. **Agent Fine-tuning** (FinBERT news agent + market agent)
  3. **RL Online Learning** (SAC/DQN üzerinden pozisyon optimizasyonu)

### 2.2 Nasıl Yazılacak
- Her aşama için:
  - Giriş (input) nedir
  - Çıkış (output) nedir
  - Kayıp fonksiyonu nedir
  - Hangi validasyon stratejisi kullanılır (walk-forward)
- Matematiksel formüller referans olarak eklenecek (spesifikasyondan alınacak)
- Overfitting önleme stratejileri tablolaştırılacak

---

## 3. Bölüm B — Kod Şeması & Modüller

### 3.1 Modül Sınırları (Taslak)

```
stocker/
├── core/
│   ├── data/          # Veri toplama (yfinance, ccxt)
│   ├── entropy/       # Shannon + Transfer Entropy hesabı
│   ├── features/      # GASF, FFT, Wavelet, TVP-VAR
│   └── models/        # LSTM, CNN-LSTM, Transformer, ResNet
├── agents/
│   ├── news/          # FinBERT tabanlı haber analizi
│   └── market/        # Teknik analiz agenti
├── rl/
│   ├── sac/           # Soft Actor-Critic
│   └── dqn/           # Deep Q-Network
├── signals/           # Sinyal üretimi + hata modülü
├── backtest/          # Walk-forward backtest
├── cli/               # Click tabanlı CLI
└── config/            # YAML konfigürasyonlar
```

### 3.2 Veri Akış Diyagramı (Mermaid)
Ana belgede tam Mermaid diyagramı çizilecek:

```
RAW OHLCV → DR/DR² → Shannon Entropy
                    → Transfer Entropy (multi-tau)
                    → GASF Encoding → ResNet
                    → FFT/Wavelet → CNN-LSTM
                    → TVP-VAR → LSTM+Attention
                    → News → FinBERT Agent → Transformer
                              ↓
                         Meta-Learner
                              ↓
                    Signal + Confidence + Error Prediction
                              ↓
                         RL Layer (SAC/DQN)
                              ↓
                       Final Trade Decision
```

### 3.3 Ne Anlatılacak
- Her modülün interface'i (giriş/çıkış tipleri)
- Modüller arası bağımlılık grafiği
- Config ile hangi modüllerin açılıp kapanabileceği

---

## 4. Bölüm C — Agent Mimarisi

### 4.1 Ana Agentlar (Claude Code'dan yönetilen)

| Agent | Sorumluluk | Trigger |
|-------|-----------|---------|
| `data-collector-agent` | Piyasadan veri çek, DB'ye yaz | Schedule / manuel |
| `entropy-agent` | Entropy hesapla, graph güncelle | data-collector biter |
| `feature-agent` | Feature extraction (GASF, FFT...) | entropy-agent biter |
| `train-agent` | Model eğit / fine-tune | feature-agent biter |
| `predict-agent` | Sinyal üret | Canlı veri gelince |
| `rl-agent` | Pozisyon optimize et | predict-agent çıktısı |
| `backtest-agent` | Backtest çalıştır | Manuel / CI |

### 4.2 Sub-Agentlar
Her ana agent kendi içinde sub-agent'ları yönetir:

```
train-agent
  ├── lstm-trainer      (sub-agent)
  ├── cnn-lstm-trainer  (sub-agent)
  ├── transformer-trainer (sub-agent)
  └── resnet-trainer    (sub-agent)
        ↓ tamamlar
  meta-learner-combiner (sub-agent)
```

### 4.3 Agent İletişimi
- Agent'lar arası iletişim: **SQLite event log tablosu** veya **YAML dosyaları**
- Her agent çıktısını `outputs/{agent_name}/latest.json` dosyasına yazar
- Bağımlı agent bu dosyayı okuyarak başlar

---

## 5. Bölüm D — Claude Code Entegrasyonu

### 5.1 Claude Code'dan Agent Çalıştırma
Her agent bir Claude Code slash command veya RemoteTrigger olarak tanımlanacak:

```bash
# Örnek kullanım
/run-agent data-collector --market US --start 2020-01-01
/run-agent entropy --market US --timeframes 30m,1h,1d
/run-agent train --market US --epochs 100
/run-agent predict --symbol AAPL --timeframe 1d
```

### 5.2 Cron / Schedule Entegrasyonu
- `data-collector-agent`: Her gün 18:00'de (piyasa kapanışı)
- `predict-agent`: Her gün 09:00'da (açılış öncesi)
- `backtest-agent`: Haftalık

### 5.3 Agent SDK Kullanımı
Ana belge, `claude_agent_sdk` ile nasıl sub-agent spawn edileceğini gösterecek:

```python
# Örnek: train-agent içinde sub-agent spawn
agent = ClaudeAgent(
    name="lstm-trainer",
    tools=[train_lstm_tool, save_model_tool],
    model="claude-opus-4-6"
)
result = await agent.run(task="Train LSTM on US market data")
```

---

## 6. Bölüm E — Worktree Stratejisi

### 6.1 Neden Worktree?
- Her modül/agent bağımsız geliştirilebilir
- Paralel development mümkün
- Test izolasyonu sağlanır

### 6.2 Worktree Planı

| Worktree | Branch | Kapsam |
|----------|--------|--------|
| `stocker-core` | `feat/core-data` | data + entropy modülleri |
| `stocker-models` | `feat/models` | LSTM, CNN, Transformer, ResNet |
| `stocker-agents` | `feat/agents` | news + market agentları |
| `stocker-rl` | `feat/rl` | SAC + DQN |
| `stocker-signals` | `feat/signals` | Sinyal + hata modülü |
| `stocker-backtest` | `feat/backtest` | Backtest motoru |

### 6.3 Worktree Oluşturma Adımları
Ana belgede her worktree için:
```bash
git worktree add ../stocker-core feat/core-data
git worktree add ../stocker-models feat/models
# ...
```

---

## 7. Ana Belge Yazım Sırası

```
[ ] 1. Repo init + git setup (main branch)
[ ] 2. Bölüm B: Klasör yapısı + modül sınırları
[ ] 3. Bölüm A: Öğrenme algoritması akışı
[ ] 4. Bölüm B: Veri akış diyagramı (Mermaid)
[ ] 5. Bölüm C: Agent mimarisi tabloları
[ ] 6. Bölüm D: Claude Code komutları
[ ] 7. Bölüm E: Worktree kurulum scriptleri
[ ] 8. Son review + gap check
```

---

## 8. Kararlar (Tamamlandı)

| Soru | Karar |
|------|-------|
| Piyasa | US (S&P 500) + BIST — ayrı pipeline'lar |
| Deployment | Local dev → GPU kiralama (eğitim) → live trading |
| Repo | Tek monorepo, modüler yapı |
| Agent iletişimi | JSON / YAML dosyaları |
| RL katmanı | İlk sürümden itibaren dahil |
| Live trading | Evet — hem uzun vadeli hem kısa vadeli |
| Environment | Miniconda env + requirements.txt + GitHub |
| Test | pytest |

## 9. Veri Kaynakları (Araştırıldı)

### BIST
| Kaynak | Paket | Not |
|--------|-------|-----|
| İş Yatırım (IsYatirim) | `isyatirimhisse` | Ücretsiz, unofficial wrapper |
| Yahoo Finance (.IS suffix) | `yfinance` | THYAO.IS, AKBNK.IS vb. — yavaş ama ücretsiz |
| TvDataFeed | `tvdatafeed` | TradingView scraper, reliable |
| Türkçe Haber NLP | custom FinBERT-TR | GitHub'da hazır model var |

### US (S&P 500)
| Kaynak | Paket | Not |
|--------|-------|-----|
| Finnhub | `finnhub-python` | **En iyi ücretsiz** — 60 req/min, real-time, news sentiment |
| yfinance | `yfinance` | Fallback, ücretsiz ama unreliable |
| tvDatafeed | `tvdatafeed` | TradingView scraper, çok güvenilir |
| EODHD | `eodhd` | Fundamentals için ($60/ay) |

### Crypto
| Kaynak | Paket | Not |
|--------|-------|-----|
| CCXT | `ccxt` | 107+ exchange, ücretsiz, unified API |
| Binance | `python-binance` | Direkt exchange, en güvenilir |

### Haber / Sentiment
| Kaynak | Dil | Not |
|--------|-----|-----|
| Finnhub News | İngilizce | US hisseleri için dahili sentiment |
| Marketaux | İngilizce | Ücretsiz financial news API |
| FinBERT-TR | Türkçe | BIST haberleri için fine-tuned BERT |
