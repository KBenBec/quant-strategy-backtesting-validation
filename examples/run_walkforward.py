import numpy as np
import pandas as pd

from src.data_io import load_prices
from src.features import log_returns
from src.strategies import build_score
from src.backtester import BacktestConfig, run_backtest
from src.validation import walk_forward_splits
from src.risk_metrics import sharpe


def main():
    prices = load_prices("data/sample_prices.csv")
    rets = log_returns(prices)

    cfg = BacktestConfig(vol_target_ann=0.10, cost_bps=1.0, max_gross=2.0)

    sharpes = []
    for train_idx, test_idx in walk_forward_splits(rets.index, train_days=504, test_days=63, step_days=21):
        # fit-free demo: scores use only past windows via rolling functions inside build_score
        test_rets = rets.loc[test_idx]
        test_prices = prices.loc[test_idx]

        scores = build_score(test_prices, test_rets)
        bt, _ = run_backtest(test_rets, scores, cfg)
        sharpes.append(sharpe(bt["pnl_net"]))

    print("WF Sharpe (mean):", float(np.mean(sharpes)))
    print("WF Sharpe (median):", float(np.median(sharpes)))
    print("WF Sharpe (min/max):", float(np.min(sharpes)), float(np.max(sharpes)))


if __name__ == "__main__":
    main()
