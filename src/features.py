from __future__ import annotations
import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.clip(lower=1e-12)).diff().dropna()


def rolling_zscore(x: pd.Series, window: int = 60) -> pd.Series:
    m = x.rolling(window, min_periods=max(10, window // 5)).mean()
    s = x.rolling(window, min_periods=max(10, window // 5)).std(ddof=1).replace(0, np.nan)
    return (x - m) / s


def cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    def _z(row: pd.Series) -> pd.Series:
        mu = row.mean()
        sd = row.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return row * 0.0
        return (row - mu) / sd
    z = df.apply(_z, axis=1).clip(-clip, clip)
    return z.fillna(0.0)
