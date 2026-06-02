"""
Base strategy class and sample mean-reversion / momentum implementations.
"""
import numpy as np
import pandas as pd


class BaseStrategy:
    def __init__(self, commission_pts=0.75, slippage_pts=0.75):
        self.commission = commission_pts
        self.slippage = slippage_pts
        self.cost_per_trade = commission_pts + slippage_pts

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def net_return_per_trade(self, raw_return: float) -> float:
        return raw_return - self.cost_per_trade


class MACrossStrategy(BaseStrategy):
    """
    Dual moving-average crossover with ATR-based stop.
    
    Parameters
    ----------
    fast : int  -- fast MA period (default 10)
    slow : int  -- slow MA period (default 40)
    atr_period : int  -- ATR period for stop calculation (default 14)
    atr_mult : float  -- ATR multiplier for stop distance (default 2.0)
    """

    def __init__(self, fast=10, slow=40, atr_period=14, atr_mult=2.0, **kwargs):
        super().__init__(**kwargs)
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of 1 (long), -1 (short), 0 (flat)."""
        fast_ma = df['Close'].rolling(self.fast).mean()
        slow_ma = df['Close'].rolling(self.slow).mean()

        signal = pd.Series(0, index=df.index)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        return signal.shift(1)  # enter on next bar

    def stops(self, df: pd.DataFrame) -> pd.Series:
        return self._atr(df) * self.atr_mult


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion with momentum confirmation.

    Parameters
    ----------
    rsi_period : int  -- RSI period (default 14)
    oversold : float  -- entry threshold for long (default 30)
    overbought : float  -- entry threshold for short (default 70)
    atr_period : int  -- ATR period (default 14)
    """

    def __init__(self, rsi_period=14, oversold=30, overbought=70, atr_period=14, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(df['Close'])
        signal = pd.Series(0, index=df.index)
        signal[rsi < self.oversold] = 1
        signal[rsi > self.overbought] = -1
        return signal.shift(1)
