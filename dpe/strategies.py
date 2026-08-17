"""
Net-gain (profit-at-expiry) functions for standard option strategies.

Each function takes an array of terminal underlying prices `ST` (and the strikes
/ premiums of the legs) and returns the net profit per unit, i.e. the expiry
payoff minus the net premium paid. These reproduce the gain functions in
SI 527, Ch. 8 (protective put, covered call, bull/bear/butterfly spreads,
straddle, strangle).
"""
from __future__ import annotations

import numpy as np


def _call(ST, K):
    return np.maximum(ST - K, 0.0)


def _put(ST, K):
    return np.maximum(K - ST, 0.0)


def bull_call_spread(ST, K1, K2, c1, c2):
    """Long K1 call (c1), short K2 call (c2), with K1 < K2. Net debit c1 - c2."""
    return _call(ST, K1) - _call(ST, K2) - (c1 - c2)


def bear_put_spread(ST, K1, K2, p1, p2):
    """Long K2 put (p2), short K1 put (p1), with K1 < K2. Net debit p2 - p1."""
    return _put(ST, K2) - _put(ST, K1) - (p2 - p1)


def butterfly_call(ST, K1, K2, K3, c1, c2, c3):
    """Long K1 & K3 calls, short 2 x K2 calls, with K1 < K2 < K3."""
    return _call(ST, K1) - 2 * _call(ST, K2) + _call(ST, K3) - (c1 - 2 * c2 + c3)


def straddle(ST, K, c, p):
    """Long call + long put at the same strike K."""
    return _call(ST, K) + _put(ST, K) - (c + p)


def strangle(ST, K1, K2, p, c):
    """Long K1 put + long K2 call, with K1 < K2 (cheaper OTM volatility play)."""
    return _put(ST, K1) + _call(ST, K2) - (p + c)


def protective_put(ST, S0, K, p):
    """Long stock (bought at S0) + long K put (premium p)."""
    return (ST - S0) + _put(ST, K) - p


def covered_call(ST, S0, K, c):
    """Long stock (bought at S0) + short K call (premium c received)."""
    return (ST - S0) - _call(ST, K) + c
