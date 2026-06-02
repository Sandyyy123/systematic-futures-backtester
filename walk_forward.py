"""
Walk-forward optimization engine.
Runs rolling IS/OOS windows and computes WFO efficiency score.
"""
import itertools
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any


def walk_forward_test(df: pd.DataFrame,
                      strategy_cls,
                      param_grid: dict,
                      is_months: int = 12,
                      oos_months: int = 3,
                      cost_per_side: float = 0.75,
                      optimize_metric: str = 'sharpe') -> Dict:
    """
    Rolling walk-forward optimization.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex
    strategy_cls : strategy class with generate_signals() method
    param_grid : dict mapping param name to list of values
    is_months : in-sample window in months
    oos_months : out-of-sample window in months
    cost_per_side : points per trade side
    optimize_metric : metric to maximize in IS (sharpe, calmar, profit_factor)
    """
    from backtest_engine import run_backtest

    all_params = [dict(zip(param_grid.keys(), v))
                  for v in itertools.product(*param_grid.values())]

    dates = df.index
    start = dates[0]
    end = dates[-1]

    is_window = pd.DateOffset(months=is_months)
    oos_window = pd.DateOffset(months=oos_months)

    windows = []
    cur = start + is_window
    while cur + oos_window <= end:
        windows.append({
            'is_start': cur - is_window,
            'is_end': cur,
            'oos_start': cur,
            'oos_end': cur + oos_window
        })
        cur += oos_window

    results = []
    for w in windows:
        is_df = df[w['is_start']:w['is_end']]
        oos_df = df[w['oos_start']:w['oos_end']]
        if len(is_df) < 100 or len(oos_df) < 20:
            continue

        best_params, best_score = None, -np.inf
        for params in all_params:
            strat = strategy_cls(**params, commission_pts=cost_per_side, slippage_pts=0)
            sig = strat.generate_signals(is_df)
            res = run_backtest(is_df, sig, cost_per_side=cost_per_side)
            score = res['metrics'].get(optimize_metric, -np.inf)
            if score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            continue

        strat = strategy_cls(**best_params, commission_pts=cost_per_side, slippage_pts=0)
        oos_sig = strat.generate_signals(oos_df)
        oos_res = run_backtest(oos_df, oos_sig, cost_per_side=cost_per_side)

        results.append({
            'is_start': w['is_start'], 'is_end': w['is_end'],
            'oos_start': w['oos_start'], 'oos_end': w['oos_end'],
            'best_params': best_params,
            'is_sharpe': best_score,
            'oos_sharpe': oos_res['metrics'].get('sharpe', 0),
            'oos_calmar': oos_res['metrics'].get('calmar', 0),
            'oos_win_rate': oos_res['metrics'].get('win_rate', 0),
            'oos_max_dd': oos_res['metrics'].get('max_drawdown', 0),
            'n_oos_trades': len(oos_res.get('trades', []))
        })

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return {'windows': df_results, 'wfo_efficiency': 0, 'pct_profitable': 0}

    is_mean = df_results['is_sharpe'].mean()
    oos_mean = df_results['oos_sharpe'].mean()
    wfo_efficiency = oos_mean / is_mean if is_mean > 0 else 0
    pct_profitable = (df_results['oos_sharpe'] > 0).mean()

    return {
        'windows': df_results,
        'wfo_efficiency': wfo_efficiency,
        'pct_profitable': pct_profitable,
        'is_sharpe_mean': is_mean,
        'oos_sharpe_mean': oos_mean,
        'oos_calmar_mean': df_results['oos_calmar'].mean(),
        'n_windows': len(df_results)
    }
