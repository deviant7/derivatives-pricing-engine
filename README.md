# Derivatives Pricing & Hedging Engine

A compact, from-scratch options-pricing engine in Python. It prices European /
American / Asian options **four independent ways**, computes and verifies the
Greeks, simulates dynamic delta-hedging, calibrates a GBM to data, and inverts
Black–Scholes for the implied-volatility smile.

## What's inside

| Module | Does | Course chapter |
|---|---|---|
| `dpe/black_scholes.py` | Closed-form price + analytic Greeks (Δ Γ Θ ν ρ) | 7, 8 |
| `dpe/binomial.py` | CRR tree — European & American | 5 |
| `dpe/monte_carlo.py` | GBM Monte-Carlo — European & (control-variate) Asian | 2, 7 |
| `dpe/pde.py` | Crank–Nicolson Black–Scholes PDE — European & American | 7 |
| `dpe/greeks.py` | Finite-difference Greeks for *any* pricer + verifier | 8 |
| `dpe/hedging.py` | Discrete delta-hedging simulation | 8 |
| `dpe/calibration.py` | GBM calibration, implied vol, vol smile | 2, 7 |
| `dpe/strategies.py` | Spread / straddle / strangle payoff functions | 8 |
| `dpe/noarb.py` | Put–call parity & premium-bound checks | 3, 4 |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                       # cross-validation suite (four pricers must agree)
python scripts/01_convergence.py     # four methods → one price, + convergence plots
python scripts/02_greeks.py          # analytic vs finite-difference Greeks
python scripts/03_delta_hedging.py   # hedging P&L → 0, error ~ O(1/√N)
python scripts/04_vol_smile.py       # implied-vol smile (synthetic or real chain)
python scripts/05_strategies.py      # strategy payoff diagrams
```

Plots are written to `outputs/`.

## The headline result

For a European call (S=K=100, T=1, r=6%, σ=20%) the four methods agree to
< 0.5%. The binomial error decays as `O(1/N)` in tree steps and the Monte-Carlo
error as `O(1/√M)` in paths; the delta-hedging P&L is mean-zero with standard
deviation shrinking as `O(1/√N)` in rebalancing frequency.

## Using real market data

`scripts/04_vol_smile.py` runs on a synthetic chain out of the box. To use a
real one (e.g. a Nifty option chain), drop a CSV at `data/nifty_chain.csv` with
columns `strike, price, type` and set `S, T, r` at the top of the script.

## A one-line pricing example

```python
from dpe import bs_price, binomial_price, mc_price, pde_price
bs_price(100, 100, 1, 0.06, 0.20, "call")                 # 10.99…
binomial_price(100, 100, 1, 0.06, 0.20, N=2000, option_type="call")
mc_price(100, 100, 1, 0.06, 0.20, "call", n_paths=1_000_000).price
pde_price(100, 100, 1, 0.06, 0.20, "call", M=600, N=600)
```
