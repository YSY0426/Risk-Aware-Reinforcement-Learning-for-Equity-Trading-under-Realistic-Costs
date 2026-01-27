# utils/metrics.py

import numpy as np


def sharpe_ratio(returns, risk_free: float = 0.0, periods_per_year: int = 252):
    """
    Compute an approximate annualized Sharpe ratio from periodic returns.
    returns: list or np.ndarray of per-step returns (e.g. daily).
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free

    if excess.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(equity_curve):
    """
    Compute maximum drawdown given an equity curve.
    equity_curve: array-like of cumulative equity values (start from 1.0).
    Return is a negative number, e.g. -0.25 for -25%.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min())
