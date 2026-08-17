"""
Calibration and implied volatility.

  * calibrate_gbm  — estimate GBM drift mu and volatility sigma from a price
                     series (Ch. 2: parameter calibration from historical data).
  * implied_vol    — invert the Black-Scholes formula for sigma given a market
                     option price (Ch. 7 formula, solved by root-finding).
  * vol_smile      — implied vol across strikes for an option chain -> the smile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .black_scholes import bs_price


def calibrate_gbm(prices, dt=1.0 / 252.0):
    """
    Estimate annualized GBM parameters from a price series.

    Returns {'mu', 'sigma', 'n_returns'}. `dt` is the spacing between
    observations in years (1/252 for daily closes).
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 3:
        raise ValueError("Need at least 3 prices to estimate returns.")
    log_ret = np.diff(np.log(prices))
    sigma = log_ret.std(ddof=1) / np.sqrt(dt)
    mu = log_ret.mean() / dt + 0.5 * sigma ** 2      # GBM drift of S (not of log S)
    return {"mu": float(mu), "sigma": float(sigma), "n_returns": int(len(log_ret))}


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                lo=1e-6, hi=5.0):
    """Black-Scholes implied volatility via Brent's method; NaN if no root
    exists in [lo, hi] (e.g. an arbitrageable quote)."""
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - price

    try:
        return float(brentq(objective, lo, hi, maxiter=200, xtol=1e-8))
    except ValueError:
        return float("nan")


def vol_smile(chain, S, T, r, q=0.0):
    """
    Compute the implied-volatility smile for an option chain.

    `chain` is a DataFrame with columns ['strike', 'price', 'type'] where type
    is 'call' or 'put'. Returns a DataFrame with an added 'implied_vol' column
    and 'moneyness' = strike / S, sorted by strike.
    """
    df = chain.copy()
    df["implied_vol"] = [
        implied_vol(row["price"], S, row["strike"], T, r, row["type"], q)
        for _, row in df.iterrows()
    ]
    df["moneyness"] = df["strike"] / S
    return df.sort_values("strike").reset_index(drop=True)
