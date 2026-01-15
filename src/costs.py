from __future__ import annotations
import pandas as pd


def transaction_costs(turnover: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    """
    Linear costs: cost_bps * turnover
    turnover is sum(|Δw|) per day
    """
    return (cost_bps * 1e-4) * turnover
