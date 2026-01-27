# utils/metrics.py

import numpy as np


def sharpe_ratio(returns, risk_free: float = 0.0, periods: int = 252) -> float:
    """
    年化 Sharpe 比率:
        SR = (E[R] - R_f) / sigma  （这里 R_f 设为 0 可以简化）
    returns: 每期收益率（比如日收益率）
    risk_free: 年化无风险利率（这里默认 0）
    periods: 一年多少期（股票日频一般 252）
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0

    # 简化：risk_free 默认为 0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0

    ann_ret = mean * periods
    ann_vol = std * np.sqrt(periods)
    return float(ann_ret / ann_vol)


def max_drawdown(equity) -> float:
    """
    最大回撤 (max drawdown)，返回一个<=0的值（比如 -0.25 代表 -25%）。
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())
