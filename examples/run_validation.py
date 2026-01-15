import pandas as pd

from src.data_io import load_prices
from src.features import log_returns
from src.strategies import build_score
from src.backtester import BacktestConfig, run_backtest
from src.validation import permutation_test_sharpe, deflated_sharpe_proxy
from src.risk_metrics import summary


def main():
    prices = load_prices("data/sample_prices.csv")
    rets = log_returns(prices)

    scores = build_score(prices.loc[rets.index], rets)
    cfg = BacktestConfig(vol_target_ann=0.10, cost_bps=1.0, max_gross=2.0)

    bt, _ = run_backtest(rets, scores, cfg)

    print("=== Metrics ===")
    print(summary(bt))

    pval = permutation_test_sharpe(bt["pnl_net"], n_perm=2000, seed=0)
    dsh = deflated_sharpe_proxy(bt["pnl_net"], n_trials=10)

    print("\n=== Statistical validation ===")
    print("Permutation p-value (Sharpe):", pval)
    print("Deflated Sharpe proxy:", dsh)


if __name__ == "__main__":
    main()
