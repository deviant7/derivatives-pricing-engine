"""
Cox-Ross-Rubinstein (CRR) binomial tree pricing.

Handles European and American exercise for vanilla payoffs. As the number of
steps N grows, the CRR price converges to Black-Scholes at rate O(1/N).

Reference: SI 527, Ch. 5 (Discrete-time models, binomial model, American
early exercise). CRR calibration: u = exp(sigma*sqrt(dt)), d = 1/u,
risk-neutral probability p* = (exp((r-q)*dt) - d) / (u - d).
"""
from __future__ import annotations

import numpy as np


def _payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type must be 'call' or 'put'.")


def binomial_price(S, K, T, r, sigma, N=500, option_type="call",
                   exercise="european", q=0.0):
    """
    Price a vanilla option on a CRR binomial lattice.

    Parameters
    ----------
    N : number of time steps.
    exercise : 'european' or 'american'.
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p*={p:.4f} is outside (0,1); "
            "reduce dt (increase N) or check parameters."
        )

    # Terminal asset prices: S * u^j * d^(N-j), j = 0..N
    j = np.arange(N + 1)
    ST = S * u ** j * d ** (N - j)
    values = _payoff(ST, K, option_type)

    american = exercise == "american"
    for i in range(N - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            Si = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            values = np.maximum(values, _payoff(Si, K, option_type))

    return float(values[0])
