"""
Event-driven backtesting engine.
Converts a signal series into a trade log and performance metrics.
"""
import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, signals: pd.Series, stop_atr: pd.Series = None,
                 cost_per_side: float = 0.75) -> dict:
    """
    Vectorized backtest.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex
    signals : 1 (long) / -1 (short) / 0 (flat) Series, already shifted for lookahead avoidance
    stop_atr : optional ATR stop distance series
    cost_per_side : points per trade side (commission + slippage)

    Returns
    -------
    dict with equity_curve, trades, metrics
    """
    position = signals.copy()
    raw_returns = df['Close'].diff() * position.shift(1)
    transaction_costs = position.diff().abs() * cost_per_side
    net_returns = raw_returns - transaction_costs

    equity = (1 + net_returns / df['Close'].iloc[0]).cumprod()

    trades = []
    in_trade = 0
    entry_price = None
    entry_idx = None

    for i, (idx, pos) in enumerate(position.items()):
        if in_trade == 0 and pos != 0:
            in_trade = pos
            entry_price = df.loc[idx, 'Close']
            entry_idx = idx
        elif in_trade != 0 and (pos == 0 or pos != in_trade):
            exit_price = df.loc[idx, 'Close']
            gross_pnl = (exit_price - entry_price) * in_trade
            net_pnl = gross_pnl - 2 * cost_per_side
            trades.append({'entry': entry_idx, 'exit': idx,
                           'direction': 'long' if in_trade == 1 else 'short',
                           'gross_pnl': gross_pnl, 'net_pnl': net_pnl})
            in_trade = pos if pos != 0 else 0
            entry_price = df.loc[idx, 'Close'] if pos != 0 else None
            entry_idx = idx if pos != 0 else None

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = compute_metrics(equity, trades_df, net_returns)

    return {'equity_curve': equity, 'trades': trades_df, 'metrics': metrics,
            'net_returns': net_returns}


def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    returns: pd.Series) -> dict:
    """Compute standard performance metrics from equity curve and trade log."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return {}

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_years = len(equity) / (252 * 78)  # 78 x 5-min bars per day approx
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_ret = returns.mean() * 252 * 78
    ann_vol = returns.std() * np.sqrt(252 * 78)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252 * 78)
    sortino = ann_ret / downside if downside > 0 else 0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    m = {'total_return': total_return, 'cagr': cagr, 'sharpe': sharpe,
         'sortino': sortino, 'max_drawdown': max_dd, 'calmar': calmar,
         'n_trades': len(trades)}

    if not trades.empty and 'net_pnl' in trades.columns:
        wins = trades[trades['net_pnl'] > 0]
        m['win_rate'] = len(wins) / len(trades)
        gross_wins = wins['net_pnl'].sum()
        gross_losses = abs(trades[trades['net_pnl'] <= 0]['net_pnl'].sum())
        m['profit_factor'] = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        m['avg_win'] = wins['net_pnl'].mean() if len(wins) else 0
        losses = trades[trades['net_pnl'] <= 0]
        m['avg_loss'] = losses['net_pnl'].mean() if len(losses) else 0

    return m
