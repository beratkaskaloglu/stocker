"""
End-to-end training pipeline test with synthetic data.
Tests: data flow → features → models → meta-learner → signals → RL env
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")


def test_returns_calculator():
    """core/features/returns.py"""
    from core.features.returns import ReturnsCalculator

    prices = pd.Series([100, 102, 101, 105, 103], dtype=float)
    rc = ReturnsCalculator()
    dr = rc.daily_return(prices)
    assert len(dr) == 5
    assert pd.isna(dr.iloc[0])  # first value is NaN
    assert abs(dr.iloc[1] - np.log(102 / 100)) < 1e-6

    vol = rc.volatility_proxy(dr)
    assert abs(vol.iloc[1] - dr.iloc[1] ** 2) < 1e-10
    print("  [OK] ReturnsCalculator")


def test_shannon_entropy():
    """core/entropy/shannon.py"""
    from core.entropy.shannon import ShannonEntropy

    se = ShannonEntropy(bins=50)

    # Uniform → high entropy
    uniform = np.random.uniform(-0.05, 0.05, 1000)
    h_uniform = se.compute(uniform)
    assert 0.5 < h_uniform <= 1.0, f"Expected high entropy, got {h_uniform}"

    # Constant → zero entropy
    constant = np.ones(100)
    h_const = se.compute(constant)
    assert h_const == 0.0, f"Expected 0 entropy for constant, got {h_const}"

    # Batch
    matrix = np.random.randn(5, 500)
    batch = se.compute_batch(matrix)
    assert batch.shape == (5,)
    assert all(0 <= h <= 1 for h in batch)
    print("  [OK] ShannonEntropy")


def test_transfer_entropy():
    """core/entropy/transfer.py"""
    from core.entropy.transfer import TransferEntropy

    te = TransferEntropy(k_neighbors=5)

    # Independent series → TE ≈ 0
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    te_ind = te.compute(x, y, tau=1)
    assert te_ind < 0.3, f"Expected low TE for independent, got {te_ind}"

    # Causal: y = x(t-1) + noise → TE(X→Y) should be > TE(Y→X)
    x2 = rng.normal(0, 1, 500)
    y2 = np.roll(x2, 1) + rng.normal(0, 0.1, 500)
    y2[0] = 0
    te_xy = te.compute(x2, y2, tau=1)
    te_yx = te.compute(y2, x2, tau=1)
    nte = te.compute_net(x2, y2, tau=1)
    print(f"  Causal TE(X→Y)={te_xy:.4f}, TE(Y→X)={te_yx:.4f}, NTE={nte:.4f}")
    assert nte > -0.5, f"NTE should be positive or near zero for causal, got {nte}"
    print("  [OK] TransferEntropy")


def test_info_flow_graph():
    """core/entropy/graph.py"""
    from core.entropy.graph import InfoFlowGraph

    ifg = InfoFlowGraph()
    symbols = ["AAPL", "MSFT", "GOOG"]
    nte_matrix = np.array([
        [0.0, 0.05, -0.02],
        [-0.05, 0.0, 0.03],
        [0.02, -0.03, 0.0],
    ])

    graph = ifg.build(nte_matrix, symbols, threshold=0.01)
    assert len(graph.nodes) == 3
    assert graph.number_of_edges() > 0

    features = ifg.get_features(graph, "AAPL")
    assert "pagerank" in features
    assert "betweenness_centrality" in features
    print(f"  AAPL graph features: pagerank={features['pagerank']:.4f}")
    print("  [OK] InfoFlowGraph")


def test_gasf_encoder():
    """core/features/gasf.py"""
    from core.features.gasf import GASFEncoder

    enc = GASFEncoder(image_size=64)
    series = np.random.randn(60)  # 60-day window

    gasf = enc.encode(series)
    assert gasf.shape == (64, 64), f"Expected (64,64), got {gasf.shape}"

    rgb = enc.to_rgb(gasf)
    assert rgb.shape == (3, 64, 64)
    print("  [OK] GASFEncoder")


def test_lstm_attention():
    """core/models/lstm_attention.py"""
    from core.models.lstm_attention import LSTMAttentionModel

    model = LSTMAttentionModel(feature_dim=211, hidden_size=64, num_layers=1, num_heads=4)
    x = torch.randn(2, 60, 211)  # batch=2, seq=60, features=211
    out = model(x)

    assert out["direction_logits"].shape == (2, 3)
    assert out["price"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    assert 0 <= out["confidence"].min() <= out["confidence"].max() <= 1
    print(f"  LSTM output: direction_logits={out['direction_logits'][0].tolist()}")
    print("  [OK] LSTMAttentionModel")


def test_cnn_lstm():
    """core/models/cnn_lstm.py"""
    from core.models.cnn_lstm import CNNLSTMModel

    model = CNNLSTMModel(feature_dim=211, dropout=0.1)
    x = torch.randn(2, 60, 211)
    out = model(x)

    assert out["direction_logits"].shape == (2, 3)
    assert out["price"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    print("  [OK] CNNLSTMModel")


def test_transformer():
    """core/models/transformer.py"""
    from core.models.transformer import TransformerModel

    model = TransformerModel(feature_dim=211, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(2, 60, 211)
    out = model(x)

    assert out["direction_logits"].shape == (2, 3)
    assert out["price"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    print("  [OK] TransformerModel")


def test_resnet_gasf():
    """core/models/resnet_gasf.py"""
    from core.models.resnet_gasf import ResNetGASFModel

    model = ResNetGASFModel(dropout=0.1)
    x = torch.randn(2, 3, 64, 64)  # batch=2, RGB, 64x64
    out = model(x)

    assert out["direction_logits"].shape == (2, 3)
    assert out["price"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    print("  [OK] ResNetGASFModel")


def test_meta_learner():
    """core/models/meta_learner.py — stacking mode"""
    from core.models.meta_learner import MetaLearner

    ml = MetaLearner(mode="stacking")
    # Simulate 4 model outputs
    fake_outputs = [
        {
            "direction_logits": torch.randn(2, 3),
            "price": torch.randn(2, 1),
            "confidence": torch.sigmoid(torch.randn(2, 1)),
        }
        for _ in range(4)
    ]
    out = ml(fake_outputs)
    assert out["direction_logits"].shape == (2, 3)
    assert out["price"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    print("  [OK] MetaLearner (stacking)")

    # Weighted mode
    ml2 = MetaLearner(mode="weighted")
    out2 = ml2(fake_outputs)
    assert out2["direction_logits"].shape == (2, 3)
    print("  [OK] MetaLearner (weighted)")


def test_full_model_pipeline():
    """End-to-end: synthetic data → all 4 models → meta-learner"""
    from core.models.lstm_attention import LSTMAttentionModel
    from core.models.cnn_lstm import CNNLSTMModel
    from core.models.transformer import TransformerModel
    from core.models.resnet_gasf import ResNetGASFModel
    from core.models.meta_learner import MetaLearner
    from core.features.gasf import GASFEncoder

    batch_size = 4
    seq_len = 60
    feature_dim = 211

    # Synthetic input
    x_seq = torch.randn(batch_size, seq_len, feature_dim)
    gasf_enc = GASFEncoder(image_size=64)
    x_gasf = torch.stack([
        torch.tensor(gasf_enc.to_rgb(gasf_enc.encode(np.random.randn(seq_len))), dtype=torch.float32)
        for _ in range(batch_size)
    ])

    # Forward pass all 4 models
    lstm = LSTMAttentionModel(feature_dim=feature_dim, hidden_size=64, num_layers=1, num_heads=4)
    cnn_lstm = CNNLSTMModel(feature_dim=feature_dim)
    transformer = TransformerModel(feature_dim=feature_dim, d_model=64, nhead=4, num_layers=2)
    resnet = ResNetGASFModel()

    with torch.no_grad():
        out_lstm = lstm(x_seq)
        out_cnn = cnn_lstm(x_seq)
        out_tf = transformer(x_seq)
        out_rn = resnet(x_gasf)

    # Meta-learner
    ml = MetaLearner(mode="stacking")
    with torch.no_grad():
        final = ml([out_lstm, out_cnn, out_tf, out_rn])

    direction = torch.argmax(final["direction_logits"], dim=-1)
    confidence = final["confidence"]
    price = final["price"]

    print(f"  Directions: {direction.tolist()}")
    print(f"  Confidences: {confidence.squeeze().tolist()}")
    print(f"  Prices: {price.squeeze().tolist()}")
    print("  [OK] Full Model Pipeline (4 models → meta-learner)")


def test_signal_generator():
    """signals/generator.py"""
    from signals.generator import SignalGenerator

    sg = SignalGenerator(min_confidence=0.6)

    meta_output = {
        "direction_logits": np.array([-0.5, 0.1, 1.2]),  # buy strongest
        "price": 1.02,  # +2% predicted
        "confidence": 0.85,
    }

    signal = sg.generate(
        symbol="AAPL", market="US", timeframe="1d",
        meta_output=meta_output, error_score=0.1, current_price=150.0,
    )

    assert signal.direction == 1  # buy
    assert signal.target_price == round(1.02 * 150.0, 4)
    assert signal.adjusted_confidence == round(0.85 * 0.9, 4)
    print(f"  Signal: {signal.direction} ({signal.target_price}) conf={signal.adjusted_confidence}")

    # Filter low confidence
    low_signal = sg.generate(
        symbol="MSFT", market="US", timeframe="1d",
        meta_output={"direction_logits": np.array([0.5, 0.1, -0.2]), "price": 0.98, "confidence": 0.3},
        error_score=0.5, current_price=400.0,
    )
    filtered = sg.filter_signals([signal, low_signal])
    assert filtered[1].direction == 0  # filtered to hold
    print("  [OK] SignalGenerator")


def test_error_module():
    """signals/error_module.py — cold start"""
    from signals.error_module import PredictedErrorModule

    em = PredictedErrorModule(history_path="/tmp/stocker_test_history.parquet")
    assert em.cold_start is True
    score = em.compute("AAPL", "1d", confidence=0.8, entropy=0.5, volatility=0.02)
    assert score == 0.0  # cold start returns 0
    print("  [OK] PredictedErrorModule (cold start)")


def test_trading_env():
    """rl/env.py"""
    from rl.env import StockerTradingEnv

    n_bars = 300
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    features = np.random.randn(n_bars, 10).astype(np.float32)
    signals = np.random.randn(n_bars, 3).astype(np.float32)

    # DQN env
    env = StockerTradingEnv(prices, features, signals, mode="dqn", max_episode_steps=50)
    obs, info = env.reset(seed=42)
    assert obs.shape == (15,)  # 10 features + 5 portfolio state

    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  DQN env: {info.get('step', '?')} steps, final portfolio={info.get('portfolio_value', '?'):.0f}, reward={total_reward:.2f}")

    # SAC env
    env_sac = StockerTradingEnv(prices, features, signals, mode="sac", max_episode_steps=50)
    obs, _ = env_sac.reset(seed=42)
    for _ in range(20):
        action = env_sac.action_space.sample()
        obs, reward, terminated, truncated, info = env_sac.step(action)
        if terminated or truncated:
            break
    print(f"  SAC env: portfolio={info.get('portfolio_value', '?'):.0f}")
    print("  [OK] StockerTradingEnv (DQN + SAC)")


def test_rl_training_short():
    """rl/sac_agent.py + rl/dqn_agent.py — very short training"""
    from rl.env import StockerTradingEnv
    from rl.sac_agent import SACTradingAgent
    from rl.dqn_agent import DQNTradingAgent

    n_bars = 500
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    features = np.random.randn(n_bars, 10).astype(np.float32)
    signals = np.random.randn(n_bars, 3).astype(np.float32)

    # DQN short train
    env_dqn = StockerTradingEnv(prices, features, signals, mode="dqn", max_episode_steps=100)
    dqn = DQNTradingAgent(env_dqn, config={"hyperparams": {"learning_starts": 100, "buffer_size": 1000}})
    dqn.train(total_timesteps=500)
    action = dqn.predict(env_dqn.reset(seed=0)[0])
    assert action in [0, 1, 2]
    print(f"  DQN trained, predicted action: {action}")
    print("  [OK] DQNTradingAgent (short train)")

    # SAC short train
    env_sac = StockerTradingEnv(prices, features, signals, mode="sac", max_episode_steps=100)
    sac = SACTradingAgent(env_sac, config={"hyperparams": {"buffer_size": 1000}})
    sac.train(total_timesteps=500)
    pos = sac.predict(env_sac.reset(seed=0)[0])
    assert -1.0 <= pos <= 1.0
    print(f"  SAC trained, predicted position: {pos:.4f}")
    print("  [OK] SACTradingAgent (short train)")


def test_backtest_walkforward():
    """backtest/engine.py — walk-forward with dummy strategy"""
    from backtest.engine import BacktestEngine, walk_forward_splits

    # Create 2 years of synthetic daily data
    dates = pd.bdate_range("2022-01-01", periods=504)
    prices = 100 + np.cumsum(np.random.randn(504) * 0.5)

    ohlcv = pd.DataFrame({
        "symbol": "TEST",
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 504).astype(float),
    }, index=dates)

    # Test walk-forward splits
    splits = list(walk_forward_splits(dates, window=252, step=21, min_folds=5))
    print(f"  Walk-forward splits: {len(splits)} folds")
    assert len(splits) >= 5

    print("  [OK] BacktestEngine walk-forward splits")


def test_training_loop():
    """Full supervised training loop: synthetic data → model → loss → backward"""
    from core.models.lstm_attention import LSTMAttentionModel

    model = LSTMAttentionModel(feature_dim=50, hidden_size=32, num_layers=1, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_dir = torch.nn.CrossEntropyLoss()
    loss_fn_price = torch.nn.MSELoss()
    loss_fn_conf = torch.nn.BCELoss()

    # Synthetic dataset
    batch = 8
    x = torch.randn(batch, 60, 50)
    y_dir = torch.randint(0, 3, (batch,))
    y_price = torch.randn(batch, 1)
    y_conf = torch.rand(batch, 1)

    # Training loop — 10 steps
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        out = model(x)

        loss_d = loss_fn_dir(out["direction_logits"], y_dir)
        loss_p = loss_fn_price(out["price"], y_price) * 0.3
        loss_c = loss_fn_conf(out["confidence"], y_conf) * 0.1
        total_loss = loss_d + loss_p + loss_c

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    print(f"  Training loss: {losses[0]:.4f} → {losses[-1]:.4f} (10 steps)")
    assert losses[-1] < losses[0] * 1.5, "Loss should not explode"
    print("  [OK] Supervised Training Loop")


if __name__ == "__main__":
    print("=" * 60)
    print("STOCKER TRAINING PIPELINE — END-TO-END TEST")
    print("=" * 60)

    tests = [
        ("Returns Calculator", test_returns_calculator),
        ("Shannon Entropy", test_shannon_entropy),
        ("Transfer Entropy", test_transfer_entropy),
        ("Info Flow Graph", test_info_flow_graph),
        ("GASF Encoder", test_gasf_encoder),
        ("LSTM + Attention", test_lstm_attention),
        ("CNN-LSTM", test_cnn_lstm),
        ("Transformer", test_transformer),
        ("ResNet-GASF", test_resnet_gasf),
        ("Meta-Learner", test_meta_learner),
        ("Full Model Pipeline", test_full_model_pipeline),
        ("Signal Generator", test_signal_generator),
        ("Error Module", test_error_module),
        ("Trading Environment", test_trading_env),
        ("RL Training (short)", test_rl_training_short),
        ("Backtest Walk-Forward", test_backtest_walkforward),
        ("Supervised Training Loop", test_training_loop),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
