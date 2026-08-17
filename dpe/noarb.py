"""
No-arbitrage sanity checks used to validate the pricers' outputs.

  * put_call_parity_gap  — C - P should equal S e^{-qT} - K e^{-rT} (Ch. 4).
  * european_premium_bounds — model-free lower/upper bounds on a European
                     option premium (Ch. 4), derived from arbitrage arguments.
"""
from __future__ import annotations

import numpy as np


def put_call_parity_gap(call_price, put_price, S, K, T, r, q=0.0):
    """Return (C - P) - (S e^{-qT} - K e^{-rT}); should be ~0 in an
    arbitrage-free market."""
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return (call_price - put_price) - parity_rhs


def european_premium_bounds(S, K, T, r, option_type="call", q=0.0):
    """Model-free no-arbitrage (lower, upper) bounds on a European premium."""
    disc = np.exp(-r * T)
    div = np.exp(-q * T)
    if option_type == "call":
        lower = max(S * div - K * disc, 0.0)
        upper = S * div
    elif option_type == "put":
        lower = max(K * disc - S * div, 0.0)
        upper = K * disc
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return lower, upper
