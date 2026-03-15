"""
Fetch multi-pair OHLCV data via CCXT for AlphaGPT research.
Replaces the PostgreSQL dependency with local CSV files.

Usage:
    python data/fetch_data.py                    # default 20 pairs, 1h, 2000 bars
    python data/fetch_data.py --pairs 50 --tf 4h # 50 pairs, 4h timeframe
"""
import argparse
import os
import time

import ccxt
import pandas as pd

# Top crypto pairs by volume (USDT quoted)
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT",
    "INJ/USDT", "TIA/USDT", "SEI/USDT", "FET/USDT", "RENDER/USDT",
    "STX/USDT", "IMX/USDT", "AAVE/USDT", "MKR/USDT", "PEPE/USDT",
    "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "SHIB/USDT", "ORDI/USDT",
    "JUP/USDT", "WLD/USDT", "PYTH/USDT", "TRX/USDT", "TON/USDT",
    "ALGO/USDT", "FTM/USDT", "RUNE/USDT", "ENS/USDT", "CRV/USDT",
    "DYDX/USDT", "GMX/USDT", "SNX/USDT", "COMP/USDT", "LDO/USDT",
]


def fetch_ohlcv(exchange, symbol, timeframe="1h", limit=2000):
    """Fetch OHLCV for a single pair."""
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol
        return df
    except Exception as e:
        print(f"  SKIP {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for AlphaGPT")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs to fetch")
    parser.add_argument("--tf", type=str, default="1h", help="Timeframe (1h, 4h, 1d)")
    parser.add_argument("--limit", type=int, default=2000, help="Max candles per pair")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID")
    args = parser.parse_args()

    symbols = DEFAULT_SYMBOLS[: args.pairs]
    print(f"Fetching {len(symbols)} pairs from {args.exchange} ({args.tf}, limit={args.limit})")

    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    exchange.load_markets()

    # Filter to symbols actually available
    available = [s for s in symbols if s in exchange.markets]
    skipped = [s for s in symbols if s not in exchange.markets]
    if skipped:
        print(f"  Not available: {skipped}")

    all_dfs = []
    for i, sym in enumerate(available):
        print(f"  [{i+1}/{len(available)}] {sym}...", end=" ", flush=True)
        df = fetch_ohlcv(exchange, sym, timeframe=args.tf, limit=args.limit)
        if df is not None:
            all_dfs.append(df)
            print(f"{len(df)} bars")
        time.sleep(0.2)

    if not all_dfs:
        print("No data fetched!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"ohlcv_{args.tf}.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows ({len(all_dfs)} pairs) to {out_path}")


if __name__ == "__main__":
    main()
