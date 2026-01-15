from __future__ import annotations
import pandas as pd
from .features import cs_zscore


def signal_momentum(rets: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    return rets.rolling(lookback).sum()


def signal_mean_reversion(rets: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    # MR on cumulative return => -zscore of cumret (approx via rolling sum)
    cum = rets.cumsum()
    mr = -cum.rolling(window).apply(lambda x: (x[-1] - x.mean()) / (x.std(ddof=1) + 1e-12), raw=False)
    return mr


def signal_breakout(prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    # breakout score: price vs rolling max/min band
    hi = prices.rolling(window).max()
    lo = prices.rolling(window).min()
    score = (prices - (hi + lo) / 2.0) / ((hi - lo).replace(0, 1e-12))
    return score


def build_score(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    s_mom = cs_zscore(signal_momentum(rets, 20))
    s_mr  = cs_zscore(signal_mean_reversion(rets, 60))
    s_brk = cs_zscore(signal_breakout(prices, 50))
    # Blend (simple, explainable)
    return 0.45 * s_mom + 0.35 * s_mr + 0.20 * s_brk
