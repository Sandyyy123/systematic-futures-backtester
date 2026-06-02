"""
Monte Carlo simulation via trade-shuffle method.
Randomizes trade order to estimate drawdown distribution and ruin probability.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def monte_carlo_simulation(trades_pnl: pd.Series,
                            n_sims: int = 10000,
                            ruin_threshold: float = -0.50) -> dict:
    """
    Trade-shuffle Monte Carlo simulation.

    Parameters
    ----------
    trades_pnl : Series of per-trade net P&L values
    n_sims : number of simulations (default 10,000)
    ruin_threshold : drawdown level considered ruin (default -50%)

    Returns
    -------
    dict with drawdown distribution, return distribution, ruin probability
    """
    pnl_array = trades_pnl.dropna().values
    n_trades = len(pnl_array)
    if n_trades < 10:
        raise ValueError(f'Need at least 10 trades for MC simulation, got {n_trades}')

    max_dds = np.empty(n_sims)
    final_returns = np.empty(n_sims)
    ruin_count = 0

    rng = np.random.default_rng(seed=42)

    for i in range(n_sims):
        shuffled = rng.permutation(pnl_array)
        equity = np.cumsum(shuffled)
        equity_curve = 1.0 + equity / abs(pnl_array.mean() * n_trades) * 0.5

        roll_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = drawdown.min()

        max_dds[i] = max_dd
        final_returns[i] = equity_curve[-1] - 1.0

        if max_dd < ruin_threshold:
            ruin_count += 1

    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]

    return {
        'n_simulations': n_sims,
        'n_trades': n_trades,
        'max_drawdown': {
            'distribution': max_dds.tolist(),
            'percentiles': {p: float(np.percentile(max_dds, p)) for p in percentiles},
            'median': float(np.median(max_dds)),
            'mean': float(np.mean(max_dds))
        },
        'annual_return': {
            'distribution': final_returns.tolist(),
            'percentiles': {p: float(np.percentile(final_returns, p)) for p in percentiles},
            'median': float(np.median(final_returns)),
            'mean': float(np.mean(final_returns))
        },
        'ruin_probability': ruin_count / n_sims,
        'profitable_sims': float((final_returns > 0).mean()),
        'sharpe_5th_pct': float(np.percentile(final_returns, 5) / (np.std(final_returns) + 1e-9))
    }


def position_size_kelly(win_rate: float, avg_win: float, avg_loss: float,
                         fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion for position sizing.
    fraction=0.25 = quarter-Kelly (conservative, recommended).
    """
    if avg_loss == 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    p = win_rate
    kelly = (b * p - (1 - p)) / b
    return max(0.0, kelly * fraction)
