"""
Finite-difference ("bump-and-reprice") Greeks that work with *any* pricer,
plus a helper that checks the analytic Black-Scholes Greeks against them.

`price_fn` must be callable as
    price_fn(S=, K=, T=, r=, sigma=, option_type=, q=)
so wrap a multi-argument pricer with functools.partial / a lambda, e.g.
    fd_greeks(lambda **kw: binomial_price(N=2000, **kw), S, K, T, r, sigma)

Reference: SI 527, Ch. 8 (the Greeks). Central differences are used
throughout; theta = dV/dt = -dV/dT.
"""
from __future__ import annotations

import numpy as np

from .black_scholes import bs_greeks, bs_price


def fd_greeks(price_fn, S, K, T, r, sigma, option_type="call", q=0.0,
              hS=None, hsig=1e-4, hT=1e-5, hr=1e-4):
    hS = hS if hS is not None else 1e-3 * S

    def P(**overrides):
        kw = dict(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
        kw.update(overrides)
        return price_fn(**kw)

    base = P()
    delta = (P(S=S + hS) - P(S=S - hS)) / (2 * hS)
    gamma = (P(S=S + hS) - 2 * base + P(S=S - hS)) / hS ** 2
    vega = (P(sigma=sigma + hsig) - P(sigma=sigma - hsig)) / (2 * hsig)
    theta = -(P(T=T + hT) - P(T=T - hT)) / (2 * hT)   # dV/dt = -dV/dT
    rho = (P(r=r + hr) - P(r=r - hr)) / (2 * hr)
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


def verify_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Compare analytic Black-Scholes Greeks with finite-difference Greeks of the
    Black-Scholes price. Returns a dict {greek: (analytic, fd, abs_error)}.
    """
    analytic = bs_greeks(S, K, T, r, sigma, option_type, q)
    fd = fd_greeks(bs_price, S, K, T, r, sigma, option_type, q)
    return {g: (analytic[g], fd[g], abs(analytic[g] - fd[g])) for g in analytic}
