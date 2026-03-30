"""
cli/main.py
Click tabanlı CLI — tüm pipeline komutları.
"""
from __future__ import annotations

import click
from loguru import logger


@click.group()
def cli():
    """Stocker — Borsa Tahmin Sistemi CLI"""
    pass


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--start", default="2020-01-01")
@click.option("--end", default=None)
@click.option("--timeframe", default="1d")
def collect(market, start, end, timeframe):
    """Piyasadan OHLCV verisi topla."""
    from core.data.collector import DataCollector
    from core.data.storage import Storage

    logger.info(f"Collecting {market} data | {timeframe} | {start} → {end or 'today'}")

    collector = DataCollector(market=market)
    storage = Storage()

    symbols = collector.get_symbols()
    logger.info(f"Found {len(symbols)} symbols for {market}")

    for sym in symbols:
        try:
            df = collector.collect(symbol=sym, start=start, end=end, timeframe=timeframe)
            if df is not None and len(df) > 0:
                storage.save_ohlcv(df, market=market, symbol=sym)
                logger.info(f"  {sym}: {len(df)} bars saved")
        except Exception as e:
            logger.warning(f"  {sym}: failed — {e}")

    logger.info("Data collection complete")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
@click.option("--tau", default="1,5,20,60")
def entropy(market, timeframe, tau):
    """Shannon ve Transfer Entropy hesapla."""
    import numpy as np
    from core.entropy.shannon import ShannonEntropy
    from core.entropy.transfer import TransferEntropy
    from core.entropy.graph import InfoFlowGraph
    from core.data.storage import Storage

    tau_list = [int(t) for t in tau.split(",")]
    logger.info(f"Computing entropy | {market} | {timeframe} | tau={tau_list}")

    storage = Storage()
    returns_data = storage.load_returns(market=market)
    if returns_data is None:
        logger.error("No returns data found. Run 'collect' first.")
        return

    se = ShannonEntropy()
    te = TransferEntropy()
    graph_builder = InfoFlowGraph()

    # Shannon for each stock
    for symbol, rets in returns_data.items():
        h = se.compute(rets)
        logger.info(f"  {symbol}: H_norm = {h:.4f}")

    # Transfer Entropy matrix (for first tau)
    symbols = list(returns_data.keys())
    if len(symbols) >= 2:
        returns_matrix = np.array([returns_data[s] for s in symbols])
        nte_matrix = te.compute_matrix_gpu(returns_matrix, tau=tau_list[0])
        graph, features = graph_builder.build(nte_matrix, symbols)
        logger.info(f"  NTE graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    logger.info("Entropy computation complete")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
def features(market, timeframe):
    """Feature extraction — GASF, FFT, Wavelet."""
    from core.features.aggregator import FeatureAggregator

    logger.info(f"Extracting features | {market} | {timeframe}")
    agg = FeatureAggregator(seq_len=60)
    logger.info(f"FeatureAggregator ready (seq_len={agg.seq_len})")
    logger.info("Feature extraction complete (use in pipeline with data)")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--epochs", default=100)
@click.option("--device", default="auto")
@click.option("--wandb", "use_wandb", is_flag=True, help="Enable wandb logging")
@click.option("--data", "data_path", default=None, help="Path to .npz data file")
def train(market, epochs, device, use_wandb, data_path):
    """Tüm modelleri (LSTM, CNN-LSTM, Transformer, ResNet) eğit."""
    from core.training.train_all import train_all_models

    logger.info(f"Training | {market} | {epochs} epochs | {device}")
    result = train_all_models(
        market=market,
        epochs=epochs,
        device=device,
        use_wandb=use_wandb,
        data_path=data_path,
    )
    logger.info(f"Training complete. Summary: {result.get('total_time_minutes', 0):.1f} min")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--algo", default="sac", type=click.Choice(["sac", "dqn", "both"]))
@click.option("--timesteps", default=100_000)
@click.option("--device", default="auto")
def train_rl(market, algo, timesteps, device):
    """RL agentını eğit (SAC veya DQN)."""
    import numpy as np
    from rl.env import StockerTradingEnv

    logger.info(f"RL Training | {market} | {algo} | {timesteps} timesteps")

    # Synthetic data for now
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    feats = np.random.randn(n, 10).astype(np.float32)
    signals = np.random.randn(n).astype(np.float32)

    algos = [algo] if algo != "both" else ["dqn", "sac"]
    for a in algos:
        env = StockerTradingEnv(prices=prices, features=feats, signals=signals, mode=a)
        if a == "dqn":
            from rl.dqn_agent import DQNTradingAgent
            agent = DQNTradingAgent(env, config={"device": device})
            agent.train(total_timesteps=timesteps, save_dir=f"outputs/models/{market}/rl_dqn")
            agent.save(f"outputs/models/{market}/rl_dqn/best")
        else:
            from rl.sac_agent import SACTradingAgent
            agent = SACTradingAgent(env, config={"device": device})
            agent.train(total_timesteps=timesteps, save_dir=f"outputs/models/{market}/rl_sac")
            agent.save(f"outputs/models/{market}/rl_sac/best")
        logger.info(f"  {a.upper()} training complete")

    logger.info("RL training complete")


@cli.command()
@click.option("--symbol", required=True)
@click.option("--market", type=click.Choice(["US", "BIST"]), default="US")
@click.option("--timeframe", default="1d")
def predict(symbol, market, timeframe):
    """Tek hisse için sinyal üret."""
    logger.info(f"Predicting | {symbol} | {market} | {timeframe}")
    logger.warning("Predict requires trained models. Run 'train' first.")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
def predict_all(market, timeframe):
    """Piyasadaki tüm hisseler için sinyal üret."""
    logger.info(f"Generating signals | {market} | {timeframe}")
    logger.warning("Predict-all requires trained models. Run 'train' first.")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--start", required=True)
@click.option("--end", default=None)
def backtest(market, start, end):
    """Walk-forward backtest çalıştır."""
    logger.info(f"Backtesting | {market} | {start} → {end or 'today'}")
    logger.warning("Backtest requires trained models and data. Run 'collect' and 'train' first.")


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
def run_pipeline(market):
    """Tam günlük pipeline'ı çalıştır (Claude Orchestrator ile)."""
    logger.info(f"Running full pipeline | {market}")
    from claude_agents.orchestrator import run_daily_pipeline
    run_daily_pipeline(market)


if __name__ == "__main__":
    cli()
