"""
data_pipeline.py
----------------
Data loading and feature engineering utilities for NetSense AI.

This module provides:
1. Unified data loading (router logs, topology, etc.).
2. Basic preprocessing (timestamps, sorting).
3. Feature engineering (temporal + utilization metrics).

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Global data directory reference
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_and_preprocess_data() -> Optional[pd.DataFrame]:
    """
    Load raw CSVs (logs, topology, etc.) and return a merged DataFrame.

    Returns
    -------
    pd.DataFrame or None
        Cleaned and merged DataFrame, or None if no real data is available.
    """
    print("[INFO] Loading data from:", DATA_DIR)

    try:
        # Example placeholder files
        logs_path = DATA_DIR / "router_logs_15_days.csv"
        topo_path = DATA_DIR / "network_topology.csv"

        if not logs_path.exists() or not topo_path.exists():
            print("[WARN] CSVs not found. Returning None (dummy path).")
            return None

        # Load datasets
        logs = pd.read_csv(logs_path, parse_dates=["timestamp"])
        topology = pd.read_csv(topo_path)

        # Merge on device_name (adjust as per schema)
        df = logs.merge(topology, on="device_name", how="left")

        # Sort for time-series operations
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"[INFO] Data loaded successfully. Shape = {df.shape}")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def engineer_features(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Add temporal and utilization-based features to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw/merged DataFrame.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with engineered features, or None if input was None.
    """
    if df is None:
        print("[WARN] No DataFrame provided. Skipping feature engineering.")
        return None

    print("[INFO] Engineering features...")

    try:
        # Temporal features
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Utilization ratio
        df["bandwidth_utilization"] = (
            df["bandwidth_used"] / df["bandwidth_allocated"].replace(0, np.nan)
        )

        # Rolling averages per device (1-hour window, requires fixed freq index)
        df = df.set_index("timestamp")
        df["utilization_rolling_avg_1h"] = (
            df.groupby("device_name")["bandwidth_utilization"]
            .transform(lambda x: x.rolling("1H").mean())
        )
        df = df.reset_index()

        print("[INFO] Feature engineering complete. Shape =", df.shape)
        return df

    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        return df


if __name__ == "__main__":
    # Run pipeline demo
    data = load_and_preprocess_data()
    features = engineer_features(data)
    print("[INFO] Pipeline executed successfully.")
