from __future__ import annotations
import numpy as np
import pandas as pd


def walk_forward_splits(index: pd.DatetimeIndex, train_days: int = 252*2, test_days: int = 63, step_days: int = 21):
    """
    Rolling windows: train -> test, shifting by step_days.
    """
    dates = pd.Series(index=index, data=np.arange(len(index)))
    n = len(index)
    start = 0
    while True:
        tr_end = start + train_days
        te_end = tr_end + test_days
        if te_end > n:
            break
        yield index[start:tr_end], index[tr_end:te_end]
        start += step_days


def permutation_test_sharpe(pnl: pd.Series, n_perm: int = 2000, seed: int = 0) -> float:
    """
    Permutation test for Sharpe:
    shuffle pnl order => destroy time structure; p-value = P(Sharpe_perm >= Sharpe_obs).
    """
    rng = np.random.default_rng(seed)
    x = pnl.dropna().values
    if len(x) < 30:
        return 1.0
    obs = x.mean() / (x.std(ddof=1) + 1e-12)

    cnt = 0
    for _ in range(n_perm):
        xp = rng.permutation(x)
        sp = xp.mean() / (xp.std(ddof=1) + 1e-12)
        if sp >= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def deflated_sharpe_proxy(pnl: pd.Series, n_trials: int = 10) -> float:
    """
    Very light proxy: penalize Sharpe by multiple testing trials (n_trials).
    Returns a 'deflated' Sharpe-like score.
    """
    s = pnl.mean() / (pnl.std(ddof=1) + 1e-12)
    return float(s / np.sqrt(max(n_trials, 1)))
