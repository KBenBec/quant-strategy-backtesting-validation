import numpy as np
import pandas as pd

from src.data_io import load_prices
from src.features import log_returns
from src.strategies import build_score
from src.backtester import BacktestConfig, run_backtest
from src.risk_metrics import summary


def main():
    prices = load_prices("data/sample_prices.csv")
    rets = log_returns(prices)

    scores = build_score(prices.loc[rets.index], rets)
    cfg = BacktestConfig(vol_target_ann=0.10, cost_bps=1.0, max_gross=2.0)

    bt, w = run_backtest(rets, scores, cfg)
    print("=== Metrics ===")
    print(summary(bt))
    print(bt.tail())


if __name__ == "__main__":
    main()
