"""
Cross-validation test suite: the four pricers must agree, the Greeks must match
finite differences, no-arbitrage relations must hold, and implied vol must
round-trip. Run:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from dpe import (bs_price, bs_greeks, binomial_price, mc_price, pde_price,
                 fd_greeks, simulate_delta_hedge, calibrate_gbm, implied_vol,
                 vol_smile, put_call_parity_gap, european_premium_bounds)

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.06, 0.20


@pytest.mark.parametrize("opt", ["call", "put"])
def test_binomial_converges_to_bs(opt):
    bs = bs_price(S, K, T, r, sigma, opt)
    tree = binomial_price(S, K, T, r, sigma, N=3000, option_type=opt)
    assert abs(tree - bs) < 0.02


@pytest.mark.parametrize("opt", ["call", "put"])
def test_pde_matches_bs(opt):
    bs = bs_price(S, K, T, r, sigma, opt)
    pde = pde_price(S, K, T, r, sigma, opt, M=600, N=600)
    assert abs(pde - bs) < 0.05


@pytest.mark.parametrize("opt", ["call", "put"])
def test_mc_matches_bs_within_error(opt):
    bs = bs_price(S, K, T, r, sigma, opt)
    res = mc_price(S, K, T, r, sigma, opt, n_paths=400_000, seed=0)
    assert abs(res.price - bs) < 4 * res.stderr


def test_put_call_parity():
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    assert abs(put_call_parity_gap(c, p, S, K, T, r)) < 1e-9


def test_prices_within_noarb_bounds():
    for opt in ("call", "put"):
        price = bs_price(S, K, T, r, sigma, opt)
        lo, hi = european_premium_bounds(S, K, T, r, opt)
        assert lo - 1e-9 <= price <= hi + 1e-9


def test_american_call_equals_european_no_dividend():
    # With no dividends an American call should not be exercised early.
    eu = binomial_price(S, K, T, r, sigma, N=1500, option_type="call",
                        exercise="european")
    am = binomial_price(S, K, T, r, sigma, N=1500, option_type="call",
                        exercise="american")
    assert abs(am - eu) < 1e-6


def test_american_put_premium_over_european():
    eu = binomial_price(S, K, T, r, sigma, N=1500, option_type="put",
                        exercise="european")
    am = binomial_price(S, K, T, r, sigma, N=1500, option_type="put",
                        exercise="american")
    assert am > eu + 1e-3  # early-exercise premium is strictly positive


def test_american_put_pde_matches_binomial():
    tree = binomial_price(S, K, T, r, sigma, N=2000, option_type="put",
                          exercise="american")
    pde = pde_price(S, K, T, r, sigma, "put", exercise="american", M=800, N=800)
    assert abs(pde - tree) < 0.05


def test_asian_cheaper_than_vanilla():
    vanilla = bs_price(S, K, T, r, sigma, "call")
    asian = mc_price(S, K, T, r, sigma, "call", style="asian",
                     n_paths=200_000, n_steps=50, seed=0)
    assert 0 < asian.price < vanilla  # averaging lowers volatility -> cheaper


def test_greeks_match_finite_difference():
    analytic = bs_greeks(S, K, T, r, sigma, "call")
    fd = fd_greeks(bs_price, S, K, T, r, sigma, "call")
    for g in analytic:
        assert abs(analytic[g] - fd[g]) < 1e-3


def test_delta_hedge_pnl_is_mean_zero():
    pnl = simulate_delta_hedge(S, K, T, r, sigma, "call", n_rebalance=252,
                               n_paths=40_000, seed=1)
    # mean P&L within a few standard errors of zero
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl))


def test_calibration_recovers_sigma():
    rng = np.random.default_rng(0)
    dt, true_sigma, n = 1 / 252, 0.20, 4000
    z = rng.standard_normal(n)
    logret = (0.10 - 0.5 * true_sigma ** 2) * dt + true_sigma * np.sqrt(dt) * z
    prices = 100 * np.exp(np.cumsum(logret))
    est = calibrate_gbm(prices, dt=dt)
    assert abs(est["sigma"] - true_sigma) < 0.02


def test_implied_vol_round_trip():
    price = bs_price(S, K, T, r, 0.27, "call")
    iv = implied_vol(price, S, K, T, r, "call")
    assert abs(iv - 0.27) < 1e-4


def test_vol_smile_recovers_input():
    chain = pd.DataFrame({
        "strike": [90, 100, 110],
        "type": ["put", "call", "call"],
        "price": [bs_price(S, 90, 0.25, r, 0.24, "put"),
                  bs_price(S, 100, 0.25, r, 0.22, "call"),
                  bs_price(S, 110, 0.25, r, 0.23, "call")],
    })
    out = vol_smile(chain, S, 0.25, r)
    assert np.allclose(out["implied_vol"].values, [0.24, 0.22, 0.23], atol=1e-4)
