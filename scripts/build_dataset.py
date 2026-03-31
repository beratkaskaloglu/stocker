"""
scripts/build_dataset.py
Ham OHLCV + haber verisini eğitime hazır .npz dataset'e dönüştürür.

Kullanım:
    python scripts/build_dataset.py --ohlcv data/ohlcv_2024-01-01_2026-03-28.csv \
                                     --news data/news_2024-01-01_2026-03-28.json \
                                     --out data/US_dataset.npz \
                                     --seq-len 60 --threshold 0.01

Pipeline:
    1. OHLCV CSV yükle, sembol bazında ayır
    2. Her sembol için: DR, DR², rolling stats, FFT, Wavelet, teknik indikatörler
    3. Haberlere FinBERT sentiment uygula → sembol/gün bazında sentiment vektörü
    4. Fiyat değişiminden label üret (Up/Hold/Down)
    5. Sliding window ile sequence oluştur
    6. .npz olarak kaydet
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path (needed when running as script)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Technical Indicators ────────────────────────────────────────────────────

def compute_technical_indicators(df: pd.DataFrame) -> np.ndarray:
    """
    OHLCV'den 30 teknik indikatör hesapla.
    ta kütüphanesi kullanır (pip install ta).

    Returns: ndarray (n_rows, 30)
    """
    import ta

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    indicators = pd.DataFrame(index=df.index)

    # Trend (10)
    indicators["sma_10"] = ta.trend.sma_indicator(close, window=10)
    indicators["sma_20"] = ta.trend.sma_indicator(close, window=20)
    indicators["sma_50"] = ta.trend.sma_indicator(close, window=50)
    indicators["ema_10"] = ta.trend.ema_indicator(close, window=10)
    indicators["ema_20"] = ta.trend.ema_indicator(close, window=20)
    indicators["macd"] = ta.trend.macd(close)
    indicators["macd_signal"] = ta.trend.macd_signal(close)
    indicators["macd_diff"] = ta.trend.macd_diff(close)
    indicators["adx"] = ta.trend.adx(high, low, close, window=14)
    indicators["cci"] = ta.trend.cci(high, low, close, window=20)

    # Momentum (8)
    indicators["rsi_14"] = ta.momentum.rsi(close, window=14)
    indicators["stoch_k"] = ta.momentum.stoch(high, low, close, window=14)
    indicators["stoch_d"] = ta.momentum.stoch_signal(high, low, close, window=14)
    indicators["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
    indicators["roc_10"] = ta.momentum.roc(close, window=10)
    indicators["mfi"] = ta.volume.money_flow_index(high, low, close, volume, window=14)
    indicators["tsi"] = ta.momentum.tsi(close)
    indicators["uo"] = ta.momentum.ultimate_oscillator(high, low, close)

    # Volatility (7)
    indicators["bb_upper"] = ta.volatility.bollinger_hband(close, window=20)
    indicators["bb_lower"] = ta.volatility.bollinger_lband(close, window=20)
    indicators["bb_width"] = ta.volatility.bollinger_wband(close, window=20)
    indicators["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    indicators["kc_upper"] = ta.volatility.keltner_channel_hband(high, low, close, window=20)
    indicators["kc_lower"] = ta.volatility.keltner_channel_lband(high, low, close, window=20)
    indicators["dc_upper"] = ta.volatility.donchian_channel_hband(high, low, close, window=20)

    # Volume (5)
    indicators["obv"] = ta.volume.on_balance_volume(close, volume)
    indicators["vwap"] = ta.volume.volume_weighted_average_price(high, low, close, volume)
    indicators["cmf"] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    indicators["fi"] = ta.volume.force_index(close, volume, window=13)
    indicators["eom"] = ta.volume.ease_of_movement(high, low, volume, window=14)

    # Forward-fill NaN, then zero-fill remaining
    indicators = indicators.ffill().fillna(0.0)

    # Normalize: each column → z-score
    for col in indicators.columns:
        std = indicators[col].std()
        if std > 1e-8:
            indicators[col] = (indicators[col] - indicators[col].mean()) / std

    return indicators.values.astype(np.float32)


# ─── FinBERT Sentiment ──────────────────────────────────────────────────────

def compute_sentiment_vectors(
    news_list: list[dict],
    use_finbert: bool = True,
    batch_size: int = 32,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Haberlere sentiment uygula → {symbol: {date_str: sentiment_vector(4)}}.

    Sentiment vector: [positive, negative, neutral, compound]
    compound = positive - negative

    FinBERT yoksa basit keyword-based fallback kullanır.
    """
    if not news_list:
        return {}

    # Group news by (symbol, date)
    from collections import defaultdict
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in news_list:
        sym = item.get("symbol", "")
        dt_str = item.get("datetime", "")[:10]  # YYYY-MM-DD
        text = f"{item.get('headline', '')}. {item.get('summary', '')}"
        if text.strip(". "):
            grouped[sym][dt_str].append(text)

    result: dict[str, dict[str, np.ndarray]] = {}

    if use_finbert:
        try:
            return _finbert_sentiment(grouped, batch_size)
        except Exception as e:
            logger.warning(f"FinBERT failed, falling back to keyword sentiment: {e}")

    # Keyword-based fallback
    return _keyword_sentiment(grouped)


def _finbert_sentiment(
    grouped: dict, batch_size: int = 32
) -> dict[str, dict[str, np.ndarray]]:
    """FinBERT ile sentiment hesapla."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    logger.info("Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    logger.info(f"FinBERT loaded on {device}")

    result: dict[str, dict[str, np.ndarray]] = {}

    # Collect all texts for batch processing
    all_items = []  # (symbol, date, text)
    for sym, dates in grouped.items():
        for dt, texts in dates.items():
            for text in texts:
                all_items.append((sym, dt, text[:512]))  # FinBERT max 512 tokens

    logger.info(f"Processing {len(all_items)} news articles with FinBERT...")

    # Process in batches
    sentiments: dict[str, dict[str, list]] = {}
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]
        texts = [item[2] for item in batch]

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            # FinBERT labels: positive, negative, neutral

        for j, (sym, dt, _) in enumerate(batch):
            if sym not in sentiments:
                sentiments[sym] = {}
            if dt not in sentiments[sym]:
                sentiments[sym][dt] = []
            sentiments[sym][dt].append(probs[j])

        if (i // batch_size) % 10 == 0:
            logger.info(f"  FinBERT progress: {i + len(batch)}/{len(all_items)}")

    # Aggregate: mean of all news per (symbol, date)
    for sym, dates in sentiments.items():
        result[sym] = {}
        for dt, prob_list in dates.items():
            mean_probs = np.mean(prob_list, axis=0)  # [pos, neg, neutral]
            compound = float(mean_probs[0] - mean_probs[1])
            result[sym][dt] = np.array([
                mean_probs[0], mean_probs[1], mean_probs[2], compound
            ], dtype=np.float32)

    return result


def _keyword_sentiment(grouped: dict) -> dict[str, dict[str, np.ndarray]]:
    """Basit keyword-based sentiment (fallback)."""
    positive_words = {"up", "rise", "gain", "bull", "profit", "growth", "beat", "surge",
                      "rally", "strong", "high", "positive", "outperform", "upgrade"}
    negative_words = {"down", "fall", "loss", "bear", "drop", "decline", "miss", "crash",
                      "weak", "low", "negative", "underperform", "downgrade", "cut"}

    result: dict[str, dict[str, np.ndarray]] = {}
    for sym, dates in grouped.items():
        result[sym] = {}
        for dt, texts in dates.items():
            pos_count, neg_count, total = 0, 0, 0
            for text in texts:
                words = set(text.lower().split())
                pos_count += len(words & positive_words)
                neg_count += len(words & negative_words)
                total += len(words)
            total = max(total, 1)
            pos = pos_count / total
            neg = neg_count / total
            neu = 1.0 - pos - neg
            result[sym][dt] = np.array([pos, neg, max(neu, 0), pos - neg], dtype=np.float32)
    return result


# ─── Label Generation ────────────────────────────────────────────────────────

def generate_labels(
    close_prices: pd.Series,
    threshold: float = 0.01,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fiyat değişiminden label üret.

    Returns:
        direction: ndarray int64 (0=Down, 1=Hold, 2=Up)
        target_return: ndarray float32 (actual return value for regression)
    """
    future_return = close_prices.pct_change(periods=horizon).shift(-horizon)

    direction = np.full(len(close_prices), 1, dtype=np.int64)  # default Hold
    direction[future_return > threshold] = 2   # Up
    direction[future_return < -threshold] = 0  # Down

    target_return = future_return.fillna(0.0).values.astype(np.float32)

    # Son 'horizon' satır geçersiz (gelecek yok)
    direction[-horizon:] = 1
    target_return[-horizon:] = 0.0

    return direction, target_return


# Multi-horizon label generation
DAILY_HORIZONS = {"1d": 1, "15d": 15, "1m": 21, "3m": 63, "6m": 126, "1y": 252}
INTRADAY_HORIZONS = {"1h": 1, "4h": 4}


def generate_multi_horizon_labels(
    close_prices: pd.Series,
    threshold: float = 0.01,
    horizons: dict[str, int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Birden fazla horizon için label üret.

    Returns:
        {horizon_name: (direction, target_return)}
    """
    if horizons is None:
        horizons = DAILY_HORIZONS

    result = {}
    for name, periods in horizons.items():
        # Uzun horizon'lar için threshold'u ölçekle
        scaled_threshold = threshold * (periods ** 0.5)  # sqrt scaling
        direction, target_return = generate_labels(close_prices, scaled_threshold, periods)
        result[name] = (direction, target_return)

    return result


# ─── Feature Pipeline ────────────────────────────────────────────────────────

def build_features_for_symbol(
    sym_df: pd.DataFrame,
    sentiment_map: dict[str, np.ndarray] | None = None,
    seq_len: int = 60,
) -> np.ndarray:
    """
    Tek sembol için tüm feature'ları hesapla.

    Returns: ndarray (n_rows, feature_dim)
    Feature dim = 6 (returns) + 1 (entropy) + 64 (FFT) + 20 (Wavelet) + 30 (TA) + 4 (sentiment) = 125
    Not: Graph features (10) çıkarıldı çünkü sembol bazında graph yok.
    """
    from core.features.returns import ReturnsCalculator
    from core.features.frequency import FrequencyFeatures

    n = len(sym_df)
    rc = ReturnsCalculator()
    ff = FrequencyFeatures(n_fft_components=32, wavelet_level=4)

    # 1. Returns: DR, DR², rolling stats (6 features)
    df = rc.compute_all(sym_df)
    returns_features = df[["dr", "dr2", "rolling_mean", "rolling_std",
                           "rolling_skew", "rolling_kurtosis"]].values.astype(np.float32)

    # 2. Shannon Entropy — rolling window (1 feature)
    dr = df["dr"].values
    entropy_vals = np.zeros(n, dtype=np.float32)
    for i in range(20, n):
        window = dr[i - 20:i]
        hist, _ = np.histogram(window, bins=10, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            p = hist / hist.sum()
            entropy_vals[i] = float(-np.sum(p * np.log2(p + 1e-10)))

    # 3. FFT + Wavelet (64 + 20 = 84 features)
    close = df["close"].values.astype(np.float64)
    freq_features = np.zeros((n, 84), dtype=np.float32)
    for i in range(seq_len, n):
        window = close[i - seq_len:i]
        try:
            freq_features[i] = ff.compute_all(window)
        except Exception:
            pass

    # 4. Technical Indicators (30 features)
    tech = compute_technical_indicators(sym_df)

    # 5. Sentiment (4 features)
    sentiment = np.zeros((n, 4), dtype=np.float32)
    if sentiment_map:
        dates = pd.to_datetime(sym_df.index).strftime("%Y-%m-%d")
        for i, dt in enumerate(dates):
            if dt in sentiment_map:
                sentiment[i] = sentiment_map[dt]
            elif i > 0:
                sentiment[i] = sentiment[i - 1]  # carry forward

    # Stack: (n, 6) + (n, 1) + (n, 84) + (n, 30) + (n, 4) = (n, 125)
    all_features = np.hstack([
        returns_features,               # 6
        entropy_vals.reshape(-1, 1),     # 1
        freq_features,                   # 84
        tech,                            # 30
        sentiment,                       # 4
    ])

    return all_features.astype(np.float32)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def build_dataset(
    ohlcv_path: str,
    news_path: str | None,
    output_path: str,
    seq_len: int = 60,
    threshold: float = 0.01,
    use_finbert: bool = True,
) -> dict:
    """Ana pipeline: ham veri → .npz dataset."""

    logger.info("=" * 60)
    logger.info("BUILD DATASET PIPELINE")
    logger.info("=" * 60)

    # 1. Load OHLCV
    logger.info(f"Loading OHLCV from {ohlcv_path}")
    ohlcv_df = pd.read_csv(ohlcv_path, parse_dates=["date"], index_col="date")
    symbols = ohlcv_df["symbol"].unique()
    logger.info(f"  {len(symbols)} symbols, {len(ohlcv_df)} total rows")

    # 2. Load & process news
    sentiment_by_symbol: dict[str, dict[str, np.ndarray]] = {}
    if news_path and Path(news_path).exists():
        logger.info(f"Loading news from {news_path}")
        news_list = json.loads(Path(news_path).read_text())
        logger.info(f"  {len(news_list)} articles")

        logger.info("Computing sentiment vectors...")
        sentiment_by_symbol = compute_sentiment_vectors(
            news_list, use_finbert=use_finbert, batch_size=32
        )
        logger.info(f"  Sentiment computed for {len(sentiment_by_symbol)} symbols")
    else:
        logger.warning("No news file — sentiment features will be zero")

    # 3. First pass: count total windows for memmap pre-allocation
    horizon_names = list(DAILY_HORIZONS.keys())  # ["1d", "15d", "1m"]
    stats = {}
    symbol_data = []  # (sym, sym_df) pairs that passed filtering

    logger.info("Pass 1: counting windows per symbol...")
    total_windows = 0
    feature_dim = None

    for i, sym in enumerate(symbols):
        sym_df = ohlcv_df[ohlcv_df["symbol"] == sym].copy()
        sym_df = sym_df.drop(columns=["symbol"])
        sym_df = sym_df.sort_index()

        if len(sym_df) < seq_len + 10:
            continue

        n_windows = len(sym_df) - seq_len
        total_windows += n_windows
        symbol_data.append((sym, sym_df))

        # Detect feature_dim from first symbol
        if feature_dim is None:
            sentiment_map = sentiment_by_symbol.get(sym, None)
            sample_features = build_features_for_symbol(sym_df.iloc[:seq_len + 5], sentiment_map, seq_len)
            feature_dim = sample_features.shape[1]

    logger.info(f"  {len(symbol_data)} symbols, {total_windows} total windows, feature_dim={feature_dim}")
    if total_windows == 0 or feature_dim is None:
        logger.error("No data produced!")
        return {"status": "error", "reason": "no data"}

    # 4. Second pass: write directly to memmap (disk-backed, no RAM explosion)
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Pass 2: building features → memmap ({total_windows} × {seq_len} × {feature_dim})...")

    X_mmap = np.memmap(tmp_dir / "X.dat", dtype=np.float32, mode="w+",
                        shape=(total_windows, seq_len, feature_dim))
    labels_mmap = {}
    for h in horizon_names:
        labels_mmap[f"dir_{h}"] = np.memmap(tmp_dir / f"dir_{h}.dat", dtype=np.int64,
                                              mode="w+", shape=(total_windows,))
        labels_mmap[f"ret_{h}"] = np.memmap(tmp_dir / f"ret_{h}.dat", dtype=np.float32,
                                              mode="w+", shape=(total_windows,))

    write_idx = 0
    all_symbols_out = []
    all_dates = []

    for i, (sym, sym_df) in enumerate(symbol_data):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"[{i + 1}/{len(symbol_data)}] Processing {sym}...")

        sentiment_map = sentiment_by_symbol.get(sym, None)
        features = build_features_for_symbol(sym_df, sentiment_map, seq_len)
        multi_labels = generate_multi_horizon_labels(sym_df["close"], threshold, DAILY_HORIZONS)

        n_win = len(sym_df) - seq_len
        for j in range(seq_len, len(sym_df)):
            w = j - seq_len
            X_mmap[write_idx] = features[j - seq_len:j]
            for h in horizon_names:
                direction, target_return = multi_labels[h]
                labels_mmap[f"dir_{h}"][write_idx] = direction[j]
                labels_mmap[f"ret_{h}"][write_idx] = target_return[j]
            all_symbols_out.append(sym)
            all_dates.append(str(sym_df.index[j].date()))
            write_idx += 1

        dir_1d = multi_labels["1d"][0][seq_len:]
        n_up = int((dir_1d == 2).sum())
        n_down = int((dir_1d == 0).sum())
        n_hold = int((dir_1d == 1).sum())
        stats[sym] = {"rows": len(sym_df), "windows": n_win,
                       "up": n_up, "down": n_down, "hold": n_hold}

    X_mmap.flush()
    for m in labels_mmap.values():
        m.flush()

    logger.info(f"\nTotal: {write_idx} windows written")

    # 5. Normalize using streaming stats (no full copy in RAM)
    logger.info("Computing normalization stats (streaming)...")
    # Sample 100K random rows for stats (good enough, saves RAM)
    sample_size = min(100_000, write_idx)
    sample_idx = np.random.choice(write_idx, sample_size, replace=False)
    sample_flat = X_mmap[sample_idx].reshape(-1, feature_dim)

    q25 = np.percentile(sample_flat, 25, axis=0)
    q75 = np.percentile(sample_flat, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-8] = 1.0
    median = np.median(sample_flat, axis=0)
    del sample_flat  # free RAM

    # Normalize in-place, chunk by chunk
    logger.info("Normalizing in chunks...")
    chunk_size = 10000
    for start in range(0, write_idx, chunk_size):
        end = min(start + chunk_size, write_idx)
        chunk = X_mmap[start:end].reshape(-1, feature_dim)
        chunk = (chunk - median) / iqr
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=3.0, neginf=-3.0)
        X_mmap[start:end] = chunk.reshape(end - start, seq_len, feature_dim).astype(np.float32)
    X_mmap.flush()

    # 6. Save as directory of .npy files (no RAM spike)
    output = Path(output_path).with_suffix("")  # e.g. data/sp500_dataset/
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output}/ (directory format, RAM-safe)...")

    # Move memmap files to final location
    import shutil
    shutil.move(str(tmp_dir / "X.dat"), str(output / "X.npy.dat"))

    # Save X shape info
    np.save(output / "X_shape.npy", np.array([write_idx, seq_len, feature_dim]))

    # Copy the memmap as proper .npy by re-opening
    # X is already on disk — just rename
    # For labels: small enough to fit in RAM
    label_dist = {}
    for h in horizon_names:
        y_dir = np.array(labels_mmap[f"dir_{h}"][:write_idx])
        y_ret = np.array(labels_mmap[f"ret_{h}"][:write_idx])
        np.save(output / f"y_direction_{h}.npy", y_dir)
        np.save(output / f"y_return_{h}.npy", y_ret)
        label_dist[h] = {
            "up": int((y_dir == 2).sum()),
            "hold": int((y_dir == 1).sum()),
            "down": int((y_dir == 0).sum()),
        }
        logger.info(f"  {h}: Up={label_dist[h]['up']} Hold={label_dist[h]['hold']} Down={label_dist[h]['down']}")
        del y_dir, y_ret

    # Save metadata
    np.save(output / "normalization_params.npy", np.stack([median, iqr]))
    np.save(output / "horizon_names.npy", np.array(horizon_names))

    # Symbols & dates as text (small)
    with open(output / "symbols.txt", "w") as f:
        f.write("\n".join(all_symbols_out))
    with open(output / "dates.txt", "w") as f:
        f.write("\n".join(all_dates))

    # Cleanup remaining temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_size_mb = sum(f.stat().st_size for f in output.glob("*")) / 1e6
    logger.info(f"\nDataset saved: {output}/ ({total_size_mb:.1f} MB)")
    logger.info(f"  Shape: ({write_idx}, {seq_len}, {feature_dim})")
    logger.info(f"  Horizons: {horizon_names}")

    summary = {
        "status": "success",
        "output_path": str(output),
        "n_samples": write_idx,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "n_symbols": len(stats),
        "horizons": horizon_names,
        "label_dist": label_dist,
        "size_mb": round(total_size_mb, 1),
    }

    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Summary saved: {summary_path}")

    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset from raw data")
    parser.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--news", default=None, help="Path to news JSON")
    parser.add_argument("--out", default="data/US_dataset.npz", help="Output .npz path")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length (days)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Up/Down threshold (default 1%)")
    parser.add_argument("--no-finbert", action="store_true", help="Skip FinBERT, use keyword sentiment")
    args = parser.parse_args()

    build_dataset(
        ohlcv_path=args.ohlcv,
        news_path=args.news,
        output_path=args.out,
        seq_len=args.seq_len,
        threshold=args.threshold,
        use_finbert=not args.no_finbert,
    )
