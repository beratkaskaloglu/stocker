# STOCKER — Kod Raporu & Pseudo-Code Ozeti

> **Tarih**: 28 Mart 2026 | **Test**: 17/17 PASSED | **Toplam Dosya**: 34 Python modulu

---

## 1. Genel Durum

| Kategori | Sayi | Oran |
|----------|------|------|
| Tam Implement | 27 | %79 |
| Kismi Implement | 3 | %9 |
| Stub (NotImplementedError) | 4 | %12 |

---

## 2. Tum Dosyalar ve Durumlari

### 2.1 core/data/ — Veri Toplama

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `core/data/collector.py` | :white_check_mark: Tam | 151 | Factory pattern, market'e gore kaynak secer |
| `core/data/sources/finnhub.py` | :white_check_mark: Tam | 93 | US OHLCV + haber, Finnhub API |
| `core/data/sources/isyatirim.py` | :white_check_mark: Tam | 111 | BIST veri, isyatirimhisse + BIST-100 fallback |
| `core/data/sources/tvdatafeed.py` | :white_check_mark: Tam | 89 | TradingView scraper, US+BIST fallback |
| `core/data/storage.py` | :white_check_mark: Tam | 136 | SQLite (ohlcv, news, predictions) + CSV export |

**Pseudo-code — Veri Akisi:**
```
DataCollector.collect(market="US", symbols, start, end, timeframe):
    1. market == "US" → FinnhubSource kullan
       market == "BIST" → IsYatirimSource kullan
       hata → TvDatafeedSource fallback
    2. source.fetch_ohlcv(symbol, start, end, timeframe) → pd.DataFrame
    3. Storage.save_ohlcv(df) → SQLite'a yaz
    4. return df  # columns: [date, open, high, low, close, volume, symbol]
```

---

### 2.2 core/entropy/ — Bilgi Teorisi

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `core/entropy/shannon.py` | :white_check_mark: Tam | 76 | H_norm = H(X) / log(bins), ∈ [0,1] |
| `core/entropy/transfer.py` | :white_check_mark: Tam | 172 | KSG estimator, GPU matrix + CPU fallback |
| `core/entropy/graph.py` | :white_check_mark: Tam | 95 | NTE → DiGraph, 5 centrality feature |

**Pseudo-code — Entropy Pipeline:**
```
ShannonEntropy.compute(returns, bins=50):
    1. histogram(returns, bins) → olasılık dağılımı p(x)
    2. H = -Σ p(x) * ln(p(x))
    3. H_norm = H / ln(bins)  # normalize [0,1]
    4. return H_norm  # 1'e yakın = rastgele, 0'a yakın = öngörülebilir

TransferEntropy.compute(x, y, tau, k=1, l=1):
    1. Embedding: y_future, y_past, x_past vektörlerini oluştur
    2. Joint space: (y_future, y_past, x_past) → cKDTree
    3. k-NN mesafesi bul (Chebyshev, k=5)
    4. 3 marginal space: (y_future, y_past), (y_past, x_past), (y_past)
    5. Her nokta için epsilon içindeki komşu sayısını say
    6. TE = ψ(k) - <ψ(n_xz+1)> - <ψ(n_marginal+1)> + <ψ(n_z+1)>
    7. return max(TE, 0.0)

InfoFlowGraph.build(nte_matrix, threshold):
    1. NTE[i,j] > threshold → kenar ekle (i → j etkileniyor)
    2. 5 centrality hesapla: in_degree, out_degree, pagerank, clustering, betweenness
    3. return graph, features_dict
```

---

### 2.3 core/features/ — Feature Engineering

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `core/features/returns.py` | :warning: Kismi | 44 | DR, DR² calisiyor; rolling_stats stub |
| `core/features/gasf.py` | :white_check_mark: Tam | 67 | Zaman serisi → 64x64 GASF image |
| `core/features/frequency.py` | :x: Stub | 45 | FFT + Wavelet — NotImplementedError |
| `core/features/aggregator.py` | :x: Stub | 66 | Feature birlestirme — NotImplementedError |

**Pseudo-code — Feature Pipeline:**
```
ReturnsCalculator:
    daily_return(prices):  DR(t) = ln(P(t) / P(t-1))
    volatility_proxy(prices):  DR²(t) = DR(t)²

GASFEncoder.encode(series, image_size=64):
    1. Normalize: x_norm = 2*(x - min)/(max - min) - 1  → [-1, 1]
    2. Resample: len != image_size → linspace index ile 64 noktaya indir
    3. Açı: phi = arccos(x_norm)
    4. GASF: G[i,j] = cos(phi_i + phi_j)  → (64, 64) float32
    5. to_rgb(): (64,64) → (3, 64, 64)  # ResNet girişi

FrequencyFeatures (TODO):
    fft(series):      FFT → top-32 frekans bileşeni
    wavelet(series):  Wavelet decomposition → 48 katsayı

FeatureAggregator (TODO):
    transform(ohlcv, entropy, sentiment):
        1. Her feature türünü hesapla
        2. Normalize (Z-score / MinMax)
        3. Concat → (seq_len, ~211 feature) tensor
```

---

### 2.4 core/models/ — Deep Learning Ensemble

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `core/models/lstm_attention.py` | :white_check_mark: Tam | 82 | Bidirectional LSTM + MultiheadAttention |
| `core/models/cnn_lstm.py` | :white_check_mark: Tam | 69 | Conv1d×3 + LSTM hybrid |
| `core/models/transformer.py` | :white_check_mark: Tam | 88 | TransformerEncoder + PositionalEncoding |
| `core/models/resnet_gasf.py` | :white_check_mark: Tam | 54 | ResNet-18 (64×64 GASF input) |
| `core/models/meta_learner.py` | :white_check_mark: Tam | 81 | Stacking (MLP) + Weighted (softmax) mode |

**Pseudo-code — Model Pipeline:**
```
Her model aynı interface:
    input:  (batch, seq_len, features) veya (batch, 3, 64, 64)
    output: (direction_logits[3], target_price[1], confidence[1])

LSTMAttentionModel.forward(x):            # x: (B, T, F)
    1. LSTM(bidirectional) → (B, T, 2*hidden)
    2. MultiheadAttention(Q=K=V=lstm_out) → (B, T, 2*hidden)
    3. Son timestep al → (B, 2*hidden)
    4. 3 head: direction(3), price(1), confidence(1)

CNNLSTMModel.forward(x):                  # x: (B, T, F)
    1. x.transpose → (B, F, T)
    2. Conv1d(F→64) → ReLU → Conv1d(64→128) → ReLU → Conv1d(128→64)
    3. transpose → LSTM → son timestep
    4. 3 head: direction, price, confidence

TransformerModel.forward(x):              # x: (B, T, F)
    1. Linear(F → d_model) + PositionalEncoding(sinusoidal)
    2. TransformerEncoder(nhead=4, layers=3)
    3. Mean pooling → (B, d_model)
    4. 3 head: direction, price, confidence

ResNetGASFModel.forward(x):               # x: (B, 3, 64, 64)
    1. ResNet-18 (pretrained=False, modified: avgpool→AdaptiveAvgPool)
    2. resnet.fc kaldırıldı → features (B, 512)
    3. 3 head: direction, price, confidence

MetaLearner.forward(outputs: list[4]):
    mode == "stacking":
        1. Concat 4 model çıktısı → (B, 4*5=20)
        2. MLP(20 → 64 → direction, price, confidence)
    mode == "weighted":
        1. Learned weights w = softmax(raw_weights)  # (4,)
        2. Weighted sum: Σ w_i * output_i
```

---

### 2.5 agents/ — AI Agentlar

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `agents/base_agent.py` | :white_check_mark: Tam | 52 | ABC + AgentOutput dataclass |
| `agents/news/finbert_agent.py` | :white_check_mark: Tam | 228 | ProsusAI/finbert, US haber sentiment |
| `agents/news/finbert_tr_agent.py` | :white_check_mark: Tam | 267 | Turkce BERT, BIST entity map + KAP |
| `agents/market/technical_agent.py` | :white_check_mark: Tam | 269 | 30 indikatör (ta library) |

**Pseudo-code — Agent Pipeline:**
```
FinBERTAgent.analyze(symbol, news_list):
    1. model = AutoModelForSequenceClassification("ProsusAI/finbert")
    2. Her haber için:
       a. tokenize(text, max_length=512)
       b. logits = model(input_ids) → softmax → [pos, neg, neutral]
       c. sentiment_score = pos - neg  # [-1, +1]
    3. Time-weighted average (yeni haberler 2× ağırlık)
    4. return AgentOutput(vector=[avg_sentiment, pos_ratio, neg_ratio, news_count])

FinBERTTRAgent.analyze(symbol, news_list):
    1. entity_map: {"THYAO": "Türk Hava Yolları", "AKBNK": "Akbank", ...}
    2. Her haber: KAP haberi mi? → 2× ağırlık
    3. model("dbmdz/bert-base-turkish-cased") → sentiment
    4. return AgentOutput(vector=[...])

TechnicalAgent.analyze(symbol, ohlcv):
    1. 30 indikatör hesapla (ta library):
       - Trend: SMA(20,50), EMA(12,26), MACD, ADX, Ichimoku, Parabolic SAR
       - Momentum: RSI, Stochastic, Williams%R, CCI, ROC, MFI
       - Volatility: BB(upper,lower,width), ATR, Keltner, Donchian
       - Volume: OBV, VWAP, AD, CMF
       - Pattern: golden_cross, death_cross, bb_squeeze
    2. Normalize by type:
       - Fiyat bazlı → % değişim
       - Osilatörler → /100
       - Volume → log1p
    3. return AgentOutput(vector=[30 normalized values])
```

---

### 2.6 rl/ — Reinforcement Learning

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `rl/env.py` | :white_check_mark: Tam | 261 | Gymnasium env, DQN + SAC mode |
| `rl/dqn_agent.py` | :white_check_mark: Tam | 92 | stable-baselines3 DQN wrapper |
| `rl/sac_agent.py` | :white_check_mark: Tam | 90 | stable-baselines3 SAC wrapper |

**Pseudo-code — RL Pipeline:**
```
StockerTradingEnv(gym.Env):
    State:  [price_features, signal_confidence, sentiment,
             portfolio_norm, position_sign, days_held_norm,
             unrealized_pnl, drawdown]

    DQN mode:
        action_space = Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        Buy:  cash → hisse al (tüm cash ile)
        Sell: hisse → cash'e çevir
        Hold: bir şey yapma

    SAC mode:
        action_space = Box(-1, 1)   # pozisyon oranı
        target_frac = action[0]     # portföyün %kaçı hissede olsun
        delta_shares hesapla → al/sat

    Reward (Pro Trader):
        r = portfolio_return
        r -= trade_cost / portfolio_value    # komisyon cezası
        r -= max(0, drawdown - 0.02)         # %2 üzeri drawdown ceza
        r += 0.0001 * days_held              # tutma bonusu
        if position_flipped: r -= 0.001      # flip cezası

    Termination:
        portfolio_value <= %50 initial → oyun bitti
        step >= 252 → truncated

DQNTradingAgent:
    train(env, timesteps=100_000):  stable_baselines3.DQN(MlpPolicy)
    predict(state) → int  # 0, 1, veya 2

SACTradingAgent:
    train(env, timesteps=100_000):  stable_baselines3.SAC(MlpPolicy)
    predict(state) → float  # [-1, +1] pozisyon oranı
```

---

### 2.7 signals/ — Sinyal Uretimi

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `signals/generator.py` | :white_check_mark: Tam | 112 | TradingSignal + error-adjusted confidence |
| `signals/error_module.py` | :white_check_mark: Tam | 135 | XGBoost kalibrasyon, cold start, auto-retrain |

**Pseudo-code — Sinyal Pipeline:**
```
SignalGenerator.generate(model_output, error_module):
    1. direction = argmax(direction_logits)  # -1, 0, +1
    2. raw_confidence = sigmoid(confidence_logit)
    3. predicted_error = error_module.predict(features)
    4. adjusted_confidence = raw_confidence * (1 - predicted_error)
    5. if adjusted_confidence < threshold → direction = 0 (Hold)
    6. return TradingSignal(symbol, direction, target_price, adjusted_confidence)

PredictedErrorModule:
    predict(features):
        if sample_count < 126:  # 6 ay cold start
            return 0.0
        return xgboost_model.predict(features)  # [0, 1]

    update(predicted, actual):
        buffer.append(error)
        if len(buffer) % 7 == 0:  # her 7 günde retrain
            xgb.fit(feature_buffer, error_buffer)
```

---

### 2.8 backtest/ — Walk-Forward Backtest

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `backtest/engine.py` | :white_check_mark: Tam | 442 | Walk-forward engine, Position/Portfolio |
| `backtest/metrics.py` | :white_check_mark: Tam | 41 | Sharpe, Sortino, MDD, win_rate |

**Pseudo-code — Backtest Pipeline:**
```
BacktestEngine.run(market, ohlcv, strategy, window=252, step=21):
    1. walk_forward_splits(dates, window, step):
       for each fold:
           train_start ──[252 gün]── train_end | val_start ──[21 gün]── val_end
           yield (train_start, train_end, val_start, val_end)

    2. Her fold için:
       a. strategy.on_train(train_data)  # modeli eğit
       b. val_data üzerinde gün gün:
          - _check_exits(): stop-loss / take-profit kontrol
          - strategy.predict(row) → signal
          - _apply_trade(): pozisyon aç/kapa
          - equity kaydet

    3. Metrikler:
       Sharpe   = mean(returns) / std(returns) * √252
       Sortino  = mean(returns) / std(negative_returns) * √252
       MDD      = max(peak - trough) / peak
       Win Rate = profitable_trades / total_trades

    4. return BacktestResult(total_return, sharpe, sortino, mdd, equity_curve, trades)
```

---

### 2.9 claude_agents/ — Orkestrasyon

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `claude_agents/orchestrator.py` | :warning: Kismi | 254 | Agent loop calisiyor, tool'lar stub |
| `claude_agents/train_agent.py` | :warning: Kismi | 112 | Tool tanımlari tamam, execution stub |

**Pseudo-code — Orchestrator:**
```
run_agent(task, market):
    1. System prompt: "Sen orkestratörsün, pipeline'ı koordine et"
    2. messages = [user: task]
    3. LOOP:
       a. response = claude.messages.create(tools=TOOLS)
       b. if stop_reason == "end_turn" → return response.text
       c. if stop_reason == "tool_use":
          - tool_name, args çıkar
          - TOOL_FUNCTIONS[tool_name](**args) çağır
          - sonucu tool_result olarak geri ver
       d. GOTO 3a

    TOOLS (7 adet):
    ├── collect_data(market, tf)        → core/data/collector  [STUB]
    ├── compute_entropy(market, tf, tau) → core/entropy/       [STUB]
    ├── extract_features(market, tf)    → core/features/       [STUB]
    ├── train_model(market, model, ep)  → core/models/         [STUB]
    ├── generate_signals(market, tf)    → signals/             [STUB]
    ├── optimize_positions(market)      → rl/                  [STUB]
    └── read_agent_output(agent_name)   → outputs/latest.json  [OK]
```

---

### 2.10 cli/ — Komut Satiri

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `cli/main.py` | :x: Stub | 116 | 8 Click komutu, hepsi NotImplementedError |

**Pseudo-code — CLI:**
```
cli komutlari (hepsi TODO):
    collect   --market --start --end --timeframe
    entropy   --market --timeframe --tau
    features  --market --timeframe
    train     --market --epochs --device
    train-rl  --market --algo --episodes
    predict   --symbol --market --timeframe
    backtest  --market --start --end
    run-pipeline --market
```

---

### 2.11 core/training/ — Egitim Altyapisi (YENi)

| Dosya | Durum | Satir | Aciklama |
|-------|-------|-------|----------|
| `core/training/trainer.py` | :white_check_mark: Tam | 355 | SupervisedTrainer — TensorBoard + wandb + early stop + checkpoint |
| `core/training/dataset.py` | :white_check_mark: Tam | 85 | StockerDataset + DataLoader + synthetic data |
| `core/training/train_all.py` | :white_check_mark: Tam | 220 | 4 model pipeline — tek komutla egitim |

**Pseudo-code — Training Pipeline:**
```
python -m core.training.train_all --market US --epochs 100 --device cuda --wandb

SupervisedTrainer.train(train_loader, val_loader):
    for epoch in range(epochs):
        1. TRAIN:
           for batch in train_loader:
               forward → multi_loss(direction + price + confidence)
               backward → grad_clip(1.0) → optimizer.step()
           → log: train_loss, accuracy, grad_norm, lr

        2. VALIDATE:
           for batch in val_loader:
               forward → compute_loss (no grad)
           → log: val_loss, val_accuracy

        3. LOG:
           → TensorBoard: writer.add_scalar(Loss/train, Loss/val, Accuracy/*)
           → wandb: wandb.log({train/*, val/*})
           → Console: loguru formatted output
           → JSON: training_history.json

        4. CHECKPOINT:
           val_loss < best → save best.pt
           epoch % 10 == 0 → save epoch_010.pt
           patience exceeded → early stop

    Izleme:
        tensorboard --logdir outputs/logs/US    → http://localhost:6006
        wandb.ai dashboard                      → remote (telefondan bile)

train_all_models(market, epochs, device):
    1. lstm_attention  → SupervisedTrainer → outputs/models/US/lstm_attention/best.pt
    2. cnn_lstm        → SupervisedTrainer → outputs/models/US/cnn_lstm/best.pt
    3. transformer     → SupervisedTrainer → outputs/models/US/transformer/best.pt
    4. resnet_gasf     → GASFEncoder → SupervisedTrainer → outputs/models/US/resnet_gasf/best.pt
    5. training_summary.json yazılır
```

---

## 3. Test Durumu (17/17 PASSED)

```
pytest tests/test_train_pipeline.py — 7.35s

Veri & Entropy:     test_returns_calculator     ✅
                    test_shannon_entropy        ✅
                    test_transfer_entropy       ✅ (KSG fix sonrası)
                    test_info_flow_graph        ✅
Features:           test_gasf_encoder           ✅ (implement sonrası)
Modeller:           test_lstm_attention         ✅
                    test_cnn_lstm               ✅
                    test_transformer            ✅
                    test_resnet_gasf            ✅
                    test_meta_learner           ✅
                    test_full_model_pipeline    ✅
Sinyaller:          test_signal_generator       ✅
                    test_error_module           ✅
RL:                 test_trading_env            ✅
                    test_rl_training_short      ✅ (SAC fix sonrası)
Backtest:           test_backtest_walkforward   ✅
Egitim:             test_training_loop          ✅ (loss: 1.38 → 1.17)
```

---

## 4. Duzeltilen Buglar (28 Mart 2026)

| # | Dosya | Bug | Fix |
|---|-------|-----|-----|
| 1 | `core/features/gasf.py` | NotImplementedError | arccos → cos(phi_i + phi_j) tam impl. |
| 2 | `core/entropy/transfer.py` | KSG formulu yanlis (TE≈3.09 bagimsiz seri) | 4-space decomposition: `ψ(k)-<ψ(n_xz+1)>-<ψ(n_marginal+1)>+<ψ(n_z+1)>` |
| 3 | `rl/env.py:99` | `float(action)` array'de hata | `float(np.asarray(action).flatten()[0])` |
| 4 | `rl/sac_agent.py:81` | Ayni array-to-float | `float(np.asarray(action).flatten()[0])` + numpy import |
| 5 | macOS | XGBoost libomp eksik | `brew install libomp` |

---

## 5. Kalan Isler

### Yuksek Oncelik
- [ ] `core/features/aggregator.py` — Feature birlestirme (modeller bu veriyi bekliyor)
- [ ] `claude_agents/orchestrator.py` — Tool fonksiyonlarini gercek modullere bagla
- [ ] `cli/main.py` — CLI komutlarini implement et

### Orta Oncelik
- [ ] `core/features/frequency.py` — FFT + Wavelet
- [ ] `core/features/returns.py` — rolling_stats tamamla
- [ ] `claude_agents/train_agent.py` — Tool execution bagla

### Tamamlandi
- [x] `core/training/trainer.py` — TensorBoard + wandb + early stopping + checkpoint
- [x] `core/training/dataset.py` — Dataset + DataLoader factory
- [x] `core/training/train_all.py` — 4 model pipeline (CPU test: 5.7 dk)

### Dusuk Oncelik
- [ ] `core/features/tvpvar.py` — TVP-VAR (dosya henuz yok)
- [ ] GPU'da tam egitim (100 epoch + 5000 episode RL)
- [ ] Gercek veri ile backtest
- [ ] Live trading entegrasyonu
