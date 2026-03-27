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
    logger.info(f"Collecting {market} data | {timeframe} | {start} → {end or 'today'}")
    # TODO: DataCollector çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
@click.option("--tau", default="1,5,20,60")
def entropy(market, timeframe, tau):
    """Shannon ve Transfer Entropy hesapla."""
    tau_list = [int(t) for t in tau.split(",")]
    logger.info(f"Computing entropy | {market} | {timeframe} | tau={tau_list}")
    # TODO: ShannonEntropy + TransferEntropy çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
def features(market, timeframe):
    """Feature extraction — GASF, FFT, Wavelet, TVP-VAR."""
    logger.info(f"Extracting features | {market} | {timeframe}")
    # TODO: FeatureAggregator çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--epochs", default=100)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
def train(market, epochs, device):
    """Tüm modelleri (LSTM, CNN-LSTM, Transformer, ResNet) eğit."""
    logger.info(f"Training | {market} | {epochs} epochs | {device}")
    # TODO: claude_agents/train_agent.py çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--algo", default="sac", type=click.Choice(["sac", "dqn", "both"]))
@click.option("--episodes", default=5000)
def train_rl(market, algo, episodes):
    """RL agentını eğit (SAC veya DQN)."""
    logger.info(f"RL Training | {market} | {algo} | {episodes} episodes")
    # TODO: rl/sac_agent.py veya rl/dqn_agent.py çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--timeframe", default="1d")
def predict_all(market, timeframe):
    """Piyasadaki tüm hisseler için sinyal üret."""
    logger.info(f"Generating signals | {market} | {timeframe}")
    # TODO: SignalGenerator çağır
    raise NotImplementedError


@cli.command()
@click.option("--symbol", required=True)
@click.option("--market", type=click.Choice(["US", "BIST"]), default="US")
@click.option("--timeframe", default="1d")
def predict(symbol, market, timeframe):
    """Tek hisse için sinyal üret."""
    logger.info(f"Predicting | {symbol} | {market} | {timeframe}")
    # TODO: implement
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
@click.option("--start", required=True)
@click.option("--end", default=None)
def backtest(market, start, end):
    """Walk-forward backtest çalıştır."""
    logger.info(f"Backtesting | {market} | {start} → {end or 'today'}")
    # TODO: BacktestEngine çağır
    raise NotImplementedError


@cli.command()
@click.option("--market", type=click.Choice(["US", "BIST"]), required=True)
def run_pipeline(market):
    """Tam günlük pipeline'ı çalıştır (Claude Orchestrator ile)."""
    logger.info(f"Running full pipeline | {market}")
    from claude_agents.orchestrator import run_daily_pipeline
    run_daily_pipeline(market)


if __name__ == "__main__":
    cli()
