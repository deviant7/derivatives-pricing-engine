"""
Closed-form Black-Scholes pricing and analytic Greeks for European options.

Convention: all rates and volatilities are annualized; T is in years.
An optional continuous dividend yield `q` is supported (default 0).
`S` and `K` may be scalars or numpy arrays; the remaining inputs are scalars.

Reference: SI 527, Ch. 7 (Black-Scholes formula) and Ch. 8 (the Greeks).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")
    S = np.asarray(S, dtype=float)
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc = np.exp(-r * T)
    div = np.exp(-q * T)
    S = np.asarray(S, dtype=float)
    if option_type == "call":
        price = S * div * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option_type == "put":
        price = K * disc * norm.cdf(-d2) - S * div * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return float(price) if np.ndim(price) == 0 else price


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Analytic Black-Scholes delta (used by the hedging simulation)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    div = np.exp(-q * T)
    if option_type == "call":
        delta = div * norm.cdf(d1)
    else:
        delta = -div * norm.cdf(-d1)
    return float(delta) if np.ndim(delta) == 0 else delta


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Return a dict of the five analytic Greeks.

    Conventions:
      - delta  = dV/dS
      - gamma  = d2V/dS2
      - vega   = dV/dsigma        (per 1.0 change in sigma; divide by 100 for per-1%)
      - theta  = dV/dt            (per year; typically negative for long options)
      - rho    = dV/dr            (per 1.0 change in r; divide by 100 for per-1%)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc = np.exp(-r * T)
    div = np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)
    S = np.asarray(S, dtype=float)

    gamma = div * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * div * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = div * norm.cdf(d1)
        theta = (-S * div * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * disc * norm.cdf(d2)
                 + q * S * div * norm.cdf(d1))
        rho = K * T * disc * norm.cdf(d2)
    elif option_type == "put":
        delta = -div * norm.cdf(-d1)
        theta = (-S * div * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * disc * norm.cdf(-d2)
                 - q * S * div * norm.cdf(-d1))
        rho = -K * T * disc * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    out = {"delta": delta, "gamma": gamma, "vega": vega,
           "theta": theta, "rho": rho}
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}
