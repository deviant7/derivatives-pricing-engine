"""
Discrete-time delta-hedging simulation.

We SELL one option, collect the Black-Scholes premium, and delta-hedge it along
simulated GBM paths, rebalancing at `n_rebalance` equally spaced dates. If
replication were perfect the terminal hedging P&L would be exactly zero; in
discrete time it is a mean-zero random variable whose standard deviation shrinks
like O(1/sqrt(n_rebalance)). Hedging with a wrong volatility (`sigma_hedge` !=
realised `sigma`) introduces a systematic bias.

Reference: SI 527, Ch. 8 (dynamic delta-hedging, rebalancing, hedging error).
"""
from __future__ import annotations

import numpy as np

from .black_scholes import bs_price, bs_delta
from .monte_carlo import simulate_gbm_paths


def simulate_delta_hedge(S0, K, T, r, sigma, option_type="call",
                         n_rebalance=52, n_paths=20_000, sigma_hedge=None,
                         q=0.0, seed=None):
    """
    Returns the array of terminal hedging P&L across paths (one number per path).

    `sigma` is the realised volatility used to generate the paths; `sigma_hedge`
    (default = sigma) is the volatility used to price and to compute the hedge
    delta.
    """
    sigma_hedge = sigma if sigma_hedge is None else sigma_hedge
    dt = T / n_rebalance
    paths = simulate_gbm_paths(S0, r, sigma, T, n_rebalance, n_paths, q,
                               antithetic=True, seed=seed)
    n = paths.shape[0]

    premium = bs_price(S0, K, T, r, sigma_hedge, option_type, q)
    delta = bs_delta(S0, K, T, r, sigma_hedge, option_type, q)
    cash = premium - delta * S0                 # sold option, bought delta shares
    delta_prev = np.full(n, delta)

    for k in range(1, n_rebalance):             # last rebalance at k = n_rebalance-1
        cash = cash * np.exp(r * dt)
        Sk = paths[:, k]
        tau = T - k * dt                        # strictly positive (= dt at the last step)
        delta_k = bs_delta(Sk, K, tau, r, sigma_hedge, option_type, q)
        cash = cash - (delta_k - delta_prev) * Sk
        delta_prev = delta_k

    cash = cash * np.exp(r * dt)                 # accrue over the final sub-period
    ST = paths[:, -1]
    payoff = (np.maximum(ST - K, 0.0) if option_type == "call"
              else np.maximum(K - ST, 0.0))
    pnl = cash + delta_prev * ST - payoff
    return pnl
