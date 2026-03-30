"""
scripts/collect_sp500_full.py
S&P 500 tam liste — OHLCV (2018+) + Haberler.

yfinance OHLCV: ~500 hisse × 7 yıl = ~15-20 dk (batch download)
Finnhub news: ~500 hisse × rate limit = çok uzun → sadece top-100 için haber

Kullanım:
    conda activate stocker
    python scripts/collect_sp500_full.py --start 2018-01-01
    python scripts/collect_sp500_full.py --start 2018-01-01 --news-top 100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
from loguru import logger

# ─── Setup ──────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("FINNHUB_API_KEY", "d73s199r01qjjol44ul0d73s199r01qjjol44ulg")
news_client = finnhub.Client(api_key=API_KEY)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─── S&P 500 Full List (503 symbols, Wikipedia March 2026) ──────────────────

SP500_ALL = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW",
    "CHTR", "CI", "CIEN", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COHR", "COIN", "COO", "COP", "COR", "COST",
    "CPAY", "CPB", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP",
    "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL",
    "DASH", "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR",
    "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK",
    "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX",
    "EL", "ELV", "EME", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE",
    "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE",
    "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO",
    "FIS", "FISV", "FITB", "FIX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL",
    "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP",
    "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
    "JCI", "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR",
    "KLAC", "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LII", "LIN", "LITE", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU",
    "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM",
    "MNST", "MO", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
    "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON",
    "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM",
    "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSKY",
    "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM", "RCL", "REG", "REGN", "RF",
    "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SATS", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNDK",
    "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS",
    "TPL", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT",
    "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC",
    "VRSK", "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB",
    "WSM", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM",
    "ZBH", "ZBRA", "ZTS",
]


# ─── OHLCV: yfinance batch download ─────────────────────────────────────────

def fetch_all_ohlcv(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    yfinance batch download — 500 hisseyi ~15-20 dk'da indirir.
    Tek tek değil toplu çeker (çok daha hızlı).
    """
    logger.info(f"Downloading OHLCV for {len(symbols)} symbols ({start} → {end})...")

    all_dfs = []
    # yfinance batch: max ~50-100 ticker per call for stability
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        ticker_str = " ".join(batch)
        logger.info(f"  Batch {i // batch_size + 1}/{(len(symbols) - 1) // batch_size + 1}: "
                     f"{batch[0]}...{batch[-1]} ({len(batch)} symbols)")

        try:
            data = yf.download(
                ticker_str,
                start=start, end=end,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            if data.empty:
                logger.warning(f"  Empty batch!")
                continue

            # Parse multi-ticker response
            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = data.copy()
                    else:
                        sym_df = data[sym].copy()

                    sym_df = sym_df.dropna(subset=["Close"])
                    if sym_df.empty:
                        continue

                    sym_df = sym_df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume"
                    })
                    sym_df = sym_df[["open", "high", "low", "close", "volume"]]
                    sym_df.index.name = "date"
                    sym_df["symbol"] = sym
                    all_dfs.append(sym_df)
                except Exception:
                    pass  # bazı semboller bulunamayabilir

        except Exception as e:
            logger.error(f"  Batch download failed: {e}")

        time.sleep(1)  # polite delay

    if all_dfs:
        result = pd.concat(all_dfs)
        logger.info(f"OHLCV total: {len(result)} rows, {result['symbol'].nunique()} symbols")
        return result
    return pd.DataFrame()


# ─── News: Finnhub (rate-limited) ───────────────────────────────────────────

def fetch_news_for_symbols(
    symbols: list[str], start: str, end: str
) -> list[dict]:
    """
    Finnhub haberler — rate limit: 60 req/min.
    500 sembol × 28 çeyrek = ~14,000 istek → ~6 saat
    Top 100 sembol × 28 çeyrek = ~2,800 istek → ~1.5 saat
    """
    all_news = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    total_symbols = len(symbols)
    for idx, sym in enumerate(symbols):
        logger.info(f"[{idx + 1}/{total_symbols}] News: {sym}")
        sym_news = 0
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(days=90), end_dt)
            s = current.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            try:
                raw = news_client.company_news(sym, _from=s, to=e)
                if raw:
                    for item in raw:
                        all_news.append({
                            "headline": item.get("headline", ""),
                            "summary": item.get("summary", ""),
                            "source": item.get("source", ""),
                            "datetime": datetime.fromtimestamp(
                                item.get("datetime", 0)
                            ).strftime("%Y-%m-%d %H:%M"),
                            "symbol": sym,
                        })
                        sym_news += 1
            except Exception as ex:
                if "429" in str(ex) or "Too Many" in str(ex):
                    logger.warning(f"  Rate limit hit, waiting 30s...")
                    time.sleep(30)
                    # Retry once
                    try:
                        raw = news_client.company_news(sym, _from=s, to=e)
                        if raw:
                            for item in raw:
                                all_news.append({
                                    "headline": item.get("headline", ""),
                                    "summary": item.get("summary", ""),
                                    "source": item.get("source", ""),
                                    "datetime": datetime.fromtimestamp(
                                        item.get("datetime", 0)
                                    ).strftime("%Y-%m-%d %H:%M"),
                                    "symbol": sym,
                                })
                                sym_news += 1
                    except Exception:
                        pass
                else:
                    logger.warning(f"  {sym} news {s}→{e}: {ex}")

            current = chunk_end + timedelta(days=1)
            time.sleep(1.5)  # safe rate limit

        logger.info(f"  → {sym_news} articles")

    logger.info(f"Total news collected: {len(all_news)}")
    return all_news


# ─── Main ────────────────────────────────────────────────────────────────────

# ─── Intraday OHLCV (1h bars, son 730 gün) ──────────────────────────────────

def fetch_intraday_ohlcv(symbols: list[str], interval: str = "1h") -> pd.DataFrame:
    """
    yfinance intraday data — son 730 gün, 1h veya 4h barlar.
    1h ve 4h tahminler için kullanılır.
    """
    logger.info(f"Downloading {interval} intraday for {len(symbols)} symbols...")

    all_dfs = []
    batch_size = 20  # intraday daha hassas, küçük batch

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        ticker_str = " ".join(batch)
        logger.info(f"  Intraday batch {i // batch_size + 1}: {batch[0]}...{batch[-1]}")

        try:
            data = yf.download(
                ticker_str, period="730d", interval=interval,
                group_by="ticker", auto_adjust=True, threads=True, progress=False,
            )
            if data.empty:
                continue

            for sym in batch:
                try:
                    sym_df = data[sym].copy() if len(batch) > 1 else data.copy()
                    sym_df = sym_df.dropna(subset=["Close"])
                    if sym_df.empty:
                        continue
                    sym_df = sym_df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume"
                    })
                    sym_df = sym_df[["open", "high", "low", "close", "volume"]]
                    sym_df.index.name = "date"
                    sym_df["symbol"] = sym
                    all_dfs.append(sym_df)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"  Intraday batch failed: {e}")

        time.sleep(2)

    if all_dfs:
        result = pd.concat(all_dfs)
        logger.info(f"Intraday {interval} total: {len(result)} rows, {result['symbol'].nunique()} symbols")
        return result
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Collect S&P 500 full data")
    parser.add_argument("--start", default="2018-01-01", help="Start date")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--news-top", type=int, default=0,
                        help="Kaç sembol için haber topla (0=haber toplama, 100=top 100)")
    parser.add_argument("--ohlcv-only", action="store_true", help="Sadece OHLCV topla")
    parser.add_argument("--intraday", action="store_true", help="1h intraday veri de topla (son 2 yıl)")
    args = parser.parse_args()

    start, end = args.start, args.end
    symbols = SP500_ALL

    logger.info(f"{'=' * 60}")
    logger.info(f"S&P 500 DATA COLLECTION")
    logger.info(f"Symbols: {len(symbols)} | Period: {start} → {end}")
    logger.info(f"{'=' * 60}")

    # 1. OHLCV (fast: batch download)
    ohlcv_path = DATA_DIR / f"sp500_ohlcv_{start}_{end}.csv"
    if ohlcv_path.exists():
        logger.info(f"OHLCV already exists: {ohlcv_path}")
        ohlcv_df = pd.read_csv(ohlcv_path)
    else:
        ohlcv_df = fetch_all_ohlcv(symbols, start, end)
        if not ohlcv_df.empty:
            ohlcv_df.to_csv(ohlcv_path)
            logger.info(f"OHLCV saved: {ohlcv_path} ({len(ohlcv_df)} rows)")

    # 1b. Intraday (optional)
    if args.intraday:
        intraday_path = DATA_DIR / f"sp500_intraday_1h.csv"
        if intraday_path.exists():
            logger.info(f"Intraday already exists: {intraday_path}")
        else:
            intraday_df = fetch_intraday_ohlcv(symbols, interval="1h")
            if not intraday_df.empty:
                intraday_df.to_csv(intraday_path)
                logger.info(f"Intraday saved: {intraday_path} ({len(intraday_df)} rows)")

    if args.ohlcv_only:
        logger.info("OHLCV only mode — done.")
        return

    # 2. News (slow: rate limited)
    if args.news_top > 0:
        news_symbols = symbols[:args.news_top]
        news_path = DATA_DIR / f"sp500_news_top{args.news_top}_{start}_{end}.json"

        if news_path.exists():
            logger.info(f"News already exists: {news_path}")
        else:
            logger.info(f"\nCollecting news for top {args.news_top} symbols...")
            all_news = fetch_news_for_symbols(news_symbols, start, end)
            if all_news:
                news_path.write_text(json.dumps(all_news, ensure_ascii=False, indent=2))
                logger.info(f"News saved: {news_path} ({len(all_news)} articles)")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"DONE!")
    if not ohlcv_df.empty:
        logger.info(f"OHLCV: {ohlcv_df['symbol'].nunique()} symbols, {len(ohlcv_df)} rows")
    logger.info(f"Files in {DATA_DIR}/:")
    for f in sorted(DATA_DIR.glob("sp500_*")):
        size_mb = f.stat().st_size / 1e6
        logger.info(f"  {f.name} ({size_mb:.1f} MB)")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
