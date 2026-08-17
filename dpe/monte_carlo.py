"""
Monte-Carlo pricing under the risk-neutral GBM measure.

Prices European (terminal payoff) and arithmetic-average Asian (path-dependent)
options, returning both the price estimate and its standard error. The MC error
shrinks as O(1/sqrt(M)) in the number of paths M. Antithetic variates are used
for variance reduction; a geometric-Asian control variate is available.

Reference: SI 527, Ch. 2 (GBM simulation) and Ch. 7 (risk-neutral pricing).
Under the risk-neutral measure: dS = (r - q) S dt + sigma S dW, so
S_{t+dt} = S_t * exp((r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from .black_scholes import bs_price

MCResult = namedtuple("MCResult", ["price", "stderr"])


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0,
                       antithetic=True, seed=None):
    """
    Simulate risk-neutral GBM paths.

    Returns an array of shape (n_paths, n_steps + 1) including the initial spot
    in column 0. With `antithetic=True`, n_paths is rounded down to an even
    number and paths come in +Z / -Z pairs.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + vol * Z
    log_paths = np.concatenate(
        [np.zeros((Z.shape[0], 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def _discounted_payoff_stats(payoff, r, T):
    disc = np.exp(-r * T)
    values = disc * payoff
    price = values.mean()
    stderr = values.std(ddof=1) / np.sqrt(len(values))
    return MCResult(float(price), float(stderr))


def mc_price(S, K, T, r, sigma, option_type="call", style="european",
             n_paths=100_000, n_steps=1, q=0.0, antithetic=True,
             control_variate=True, seed=None):
    """
    Monte-Carlo price.

    style : 'european'  -> uses terminal price only (n_steps forced to 1);
            'asian'      -> arithmetic average of the monitored path.
    """
    if style == "european":
        paths = simulate_gbm_paths(S, r, sigma, T, 1, n_paths, q,
                                   antithetic, seed)
        ST = paths[:, -1]
        payoff = (np.maximum(ST - K, 0.0) if option_type == "call"
                  else np.maximum(K - ST, 0.0))
        return _discounted_payoff_stats(payoff, r, T)

    if style == "asian":
        paths = simulate_gbm_paths(S, r, sigma, T, n_steps, n_paths, q,
                                   antithetic, seed)
        avg = paths[:, 1:].mean(axis=1)  # arithmetic average over monitoring dates
        payoff = (np.maximum(avg - K, 0.0) if option_type == "call"
                  else np.maximum(K - avg, 0.0))
        disc = np.exp(-r * T)
        values = disc * payoff

        if control_variate:
            # Geometric-average Asian has a closed form -> use as a control.
            geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
            geo_payoff = (np.maximum(geo - K, 0.0) if option_type == "call"
                          else np.maximum(K - geo, 0.0))
            geo_values = disc * geo_payoff
            geo_closed = _geometric_asian_closed_form(
                S, K, T, r, sigma, option_type, n_steps, q)
            cov = np.cov(values, geo_values, ddof=1)
            beta = cov[0, 1] / cov[1, 1]
            adjusted = values - beta * (geo_values - geo_closed)
            price = adjusted.mean()
            stderr = adjusted.std(ddof=1) / np.sqrt(len(adjusted))
            return MCResult(float(price), float(stderr))

        return _discounted_payoff_stats(payoff, r, T)

    raise ValueError("style must be 'european' or 'asian'.")


def _geometric_asian_closed_form(S, K, T, r, sigma, option_type, n_steps, q=0.0):
    """Closed-form geometric-average Asian (discrete monitoring) via a
    Black-Scholes call with adjusted volatility and dividend yield."""
    n = n_steps
    dt = T / n
    # Effective vol and drift for the geometric average of GBM samples.
    sigma_g = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n ** 2))
    mu_g = (r - q - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_g ** 2
    # Price as a BS option with dividend yield q_g s.t. (r - q_g) = mu_g.
    q_g = r - mu_g
    return bs_price(S, K, T, r, sigma_g, option_type, q=q_g)
