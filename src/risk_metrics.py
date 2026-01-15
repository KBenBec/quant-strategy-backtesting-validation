from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe(pnl: pd.Series, ann_days: int = 252) -> float:
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    return float(np.sqrt(ann_days) * mu / (sd + 1e-12))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def var_es_hist(pnl: pd.Series, alpha: float = 0.01) -> tuple[float, float]:
    loss = -pnl.dropna().values
    var = float(np.quantile(loss, alpha))
    tail = loss[loss >= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def summary(bt: pd.DataFrame) -> dict:
    pnl = bt["pnl_net"]
    eq = bt["equity"]
    v, e = var_es_hist(pnl, 0.01)
    return {
        "Sharpe": sharpe(pnl),
        "MaxDrawdown": max_drawdown(eq),
        "TotalReturn": float(eq.iloc[-1] - 1.0),
        "VaR_1%": v,
        "ES_1%": e,
        "AvgTurnover": float(bt["turnover"].mean()),
    }
