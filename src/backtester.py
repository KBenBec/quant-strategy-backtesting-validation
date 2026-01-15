from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .costs import transaction_costs


@dataclass(frozen=True)
class BacktestConfig:
    vol_target_ann: float = 0.10
    max_gross: float = 2.0
    cost_bps: float = 1.0
    vol_lookback: int = 60
    ann_days: int = 252


def vol_target_weights(scores: pd.DataFrame, rets: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    w = scores.copy()
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)

    pnl_proxy = (w.shift(1) * rets).sum(axis=1).fillna(0.0)
    vol = pnl_proxy.rolling(cfg.vol_lookback).std(ddof=1).replace(0, np.nan).bfill()

    daily_target = cfg.vol_target_ann / np.sqrt(cfg.ann_days)
    scale = (daily_target / vol).clip(0.0, 10.0)
    w = w.mul(scale, axis=0)

    gross2 = w.abs().sum(axis=1)
    cap = (cfg.max_gross / gross2).clip(upper=1.0)
    w = w.mul(cap, axis=0)
    return w


def run_backtest(rets: pd.DataFrame, scores: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rets = rets.loc[scores.index].copy()
    w = vol_target_weights(scores, rets, cfg)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = transaction_costs(turnover, cfg.cost_bps)

    pnl_gross = (w.shift(1) * rets).sum(axis=1).fillna(0.0)
    pnl_net = pnl_gross - costs
    equity = (1.0 + pnl_net).cumprod()

    bt = pd.DataFrame(
        {"pnl_gross": pnl_gross, "pnl_net": pnl_net, "turnover": turnover, "costs": costs, "equity": equity},
        index=rets.index,
    )
    return bt, w
