"""
scripts/run_full_test.py
S&P 500 tam pipeline testi — CPU'da ~1 saat.

Adımlar:
    1. OHLCV indir (503 sembol, 2018-2026) — ~15-20 dk
    2. Dataset oluştur (FinBERT yok, keyword sentiment) — ~10-15 dk
    3. 4 model eğit (3 epoch her biri, CPU) — ~20-30 dk
    4. Sonuçları raporla

Kullanım:
    conda activate stocker
    python scripts/run_full_test.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

def step1_collect_ohlcv() -> str:
    """OHLCV topla (yfinance batch, ~15-20 dk)."""
    from scripts.collect_sp500_full import fetch_all_ohlcv, SP500_ALL

    start = "2018-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    ohlcv_path = DATA_DIR / f"sp500_ohlcv_{start}_{end}.csv"

    if ohlcv_path.exists():
        logger.info(f"OHLCV zaten var: {ohlcv_path}")
        return str(ohlcv_path)

    DATA_DIR.mkdir(exist_ok=True)
    ohlcv_df = fetch_all_ohlcv(SP500_ALL, start, end)

    if ohlcv_df.empty:
        logger.error("OHLCV toplama başarısız!")
        sys.exit(1)

    ohlcv_df.to_csv(ohlcv_path)
    logger.info(f"OHLCV kaydedildi: {ohlcv_path} ({len(ohlcv_df)} satır, "
                f"{ohlcv_df['symbol'].nunique()} sembol)")
    return str(ohlcv_path)


def step2_build_dataset(ohlcv_path: str) -> str:
    """Dataset oluştur (keyword sentiment, FinBERT yok — hızlı)."""
    from scripts.build_dataset import build_dataset

    # Mevcut haber dosyasını bul (varsa)
    news_files = sorted(DATA_DIR.glob("*news*.json"))
    news_path = str(news_files[0]) if news_files else None

    out_path = "data/sp500_dataset.npz"

    logger.info(f"Dataset oluşturuluyor...")
    logger.info(f"  OHLCV: {ohlcv_path}")
    logger.info(f"  News: {news_path or 'YOK — sentiment sıfır olacak'}")

    summary = build_dataset(
        ohlcv_path=ohlcv_path,
        news_path=news_path,
        output_path=out_path,
        seq_len=60,
        threshold=0.01,
        use_finbert=False,  # CPU test → keyword sentiment (hızlı)
    )

    logger.info(f"Dataset: {summary.get('n_samples', 0)} sample, "
                f"feature_dim={summary.get('feature_dim', 0)}")
    return out_path


def step3_train_models(dataset_path: str) -> dict:
    """4 modeli 3 epoch eğit (CPU test) — multi-horizon."""
    logger.info("Modeller eğitiliyor (3 epoch, CPU, multi-horizon)...")

    from core.training.train_all import train_all_models

    results = train_all_models(
        market="US",
        epochs=3,
        batch_size=64,
        device="cpu",
        data_path=dataset_path,
    )

    return results.get("models", {})


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("STOCKER FULL PIPELINE TEST (CPU)")
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Step 1: Collect
    logger.info("\n📊 STEP 1/3: OHLCV Collection")
    t1 = time.time()
    ohlcv_path = step1_collect_ohlcv()
    logger.info(f"  ⏱ {(time.time() - t1) / 60:.1f} dk")

    # Step 2: Build dataset
    logger.info("\n🔧 STEP 2/3: Dataset Build")
    t2 = time.time()
    dataset_path = step2_build_dataset(ohlcv_path)
    logger.info(f"  ⏱ {(time.time() - t2) / 60:.1f} dk")

    # Step 3: Train
    logger.info("\n🧠 STEP 3/3: Model Training (3 epoch)")
    t3 = time.time()
    results = step3_train_models(dataset_path)
    logger.info(f"  ⏱ {(time.time() - t3) / 60:.1f} dk")

    # Summary
    total_min = (time.time() - t0) / 60
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PIPELINE TAMAMLANDI — {total_min:.1f} dakika")
    logger.info(f"{'=' * 60}")

    for model, res in results.items():
        status = "✓" if res.get("status") == "success" else "✗"
        loss = res.get("best_val_loss", "N/A")
        logger.info(f"  {status} {model}: {loss}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_minutes": round(total_min, 1),
        "models": results,
        "dataset_path": dataset_path,
        "ohlcv_path": ohlcv_path,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "full_test_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
