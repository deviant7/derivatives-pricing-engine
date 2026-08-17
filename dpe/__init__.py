"""
dpe — a small derivatives pricing & hedging engine.

Prices European / American / Asian options four independent ways
(closed-form Black-Scholes, CRR binomial tree, Monte-Carlo GBM, and a
Crank-Nicolson PDE solver), computes and verifies the Greeks, simulates
delta-hedging, calibrates GBM to data, and inverts Black-Scholes for the
implied-volatility smile.

Built on the SI 527 "Introduction to Derivative Pricing" syllabus
(Prof. S. Baskar, Dept. of Mathematics, IIT Bombay).
"""
from .black_scholes import bs_price, bs_greeks, bs_delta
from .binomial import binomial_price
from .monte_carlo import mc_price, simulate_gbm_paths
from .pde import pde_price
from .greeks import fd_greeks, verify_greeks
from .hedging import simulate_delta_hedge
from .calibration import calibrate_gbm, implied_vol, vol_smile
from .noarb import put_call_parity_gap, european_premium_bounds

__all__ = [
    "bs_price", "bs_greeks", "bs_delta",
    "binomial_price",
    "mc_price", "simulate_gbm_paths",
    "pde_price",
    "fd_greeks", "verify_greeks",
    "simulate_delta_hedge",
    "calibrate_gbm", "implied_vol", "vol_smile",
    "put_call_parity_gap", "european_premium_bounds",
]
