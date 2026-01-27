# baseline_buy_and_hold.py
"""
Buy-and-Hold baseline on the same test split as PPO.
"""

import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL

from configs.config import CONFIG
from utils.metrics import sharpe_ratio, max_drawdown


def main():
    train_ratio = CONFIG.get("train_ratio", 0.7)

    df_full = STOCKS_GOOGL.copy().reset_index(drop=True)
    n = len(df_full)
    split_idx = int(n * train_ratio)
    df_test = df_full.iloc[split_idx:].reset_index(drop=True)

    close = df_test["Close"].to_numpy(dtype=float)

    if close.size < 2:
        print("[ERROR] Not enough data in test split for baseline.")
        return

    # 买入持有：一直满仓做多
    returns = close[1:] / close[:-1] - 1.0

    equity = np.empty(returns.size + 1, dtype=float)
    equity[0] = 1.0
    for t, r in enumerate(returns):
        equity[t + 1] = equity[t] * (1.0 + r)

    T = returns.size
    sr = sharpe_ratio(returns)
    mdd = max_drawdown(equity)

    if T > 0:
        ann_ret = equity[-1] ** (252.0 / T) - 1.0
        ann_vol = returns.std(ddof=1) * np.sqrt(252.0)
    else:
        ann_ret = 0.0
        ann_vol = 0.0

    print(f"[BASELINE] Steps (test period): {T}")
    print(f"[BASELINE] Final equity: {equity[-1]:.4f}")
    print(f"[BASELINE] Total net return: {equity[-1] - 1.0:.6f}")
    print(f"[BASELINE] Annualized return: {ann_ret:.2%}")
    print(f"[BASELINE] Annualized volatility: {ann_vol:.2%}")
    print(f"[BASELINE] Sharpe ratio: {sr:.6f}")
    print(f"[BASELINE] Max drawdown: {mdd:.4%}")


if __name__ == "__main__":
    main()
