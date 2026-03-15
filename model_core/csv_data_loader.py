"""
CSV-based data loader for AlphaGPT.
Replaces CryptoDataLoader (PostgreSQL) with local CSV files.
"""
import os
import torch
import pandas as pd
from .config import ModelConfig
from .factors import FeatureEngineer


class CsvDataLoader:
    """Load multi-pair OHLCV from CSV, pivot to tensor format."""

    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "ohlcv_1h.csv")
        self.csv_path = os.path.abspath(csv_path)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None

    def load_data(self, limit_tokens=None):
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])

        symbols = df["symbol"].unique()
        if limit_tokens and limit_tokens < len(symbols):
            symbols = symbols[:limit_tokens]
            df = df[df["symbol"].isin(symbols)]

        print(f"  {len(symbols)} pairs, {len(df)} total rows")

        def to_tensor(col):
            pivot = df.pivot(index="timestamp", columns="symbol", values=col)
            pivot = pivot.ffill().fillna(0.0)
            # Shape: [num_tokens, num_timesteps]
            return torch.tensor(
                pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE
            )

        close_t = to_tensor("close")
        volume_t = to_tensor("volume")

        self.raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": close_t,
            "volume": volume_t,
            # No liquidity/fdv in CEX data — use volume as proxy
            "liquidity": volume_t * close_t,  # notional volume ≈ liquidity
            "fdv": close_t * 1e6,  # placeholder (large constant × price)
        }

        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # Target: 2-period-ahead log return
        op = self.raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_ret = torch.clamp(self.target_ret, -0.5, 0.5)
        self.target_ret[:, -2:] = 0.0

        print(f"  Features shape: {self.feat_tensor.shape}")
        print(f"  Device: {ModelConfig.DEVICE}")
