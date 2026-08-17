"""
Demo 5 — option strategy payoff diagrams.

Plots the net profit-at-expiry of the classic strategies from SI 527 Ch. 8,
with each leg's premium priced consistently by Black-Scholes.

Run:  python scripts/05_strategies.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpe import bs_price
from dpe.strategies import (bull_call_spread, bear_put_spread, butterfly_call,
                            straddle, strangle)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

S, T, r, sigma = 100.0, 0.5, 0.06, 0.25
ST = np.linspace(60, 140, 400)


def C(K):
    return bs_price(S, K, T, r, sigma, "call")


def P(K):
    return bs_price(S, K, T, r, sigma, "put")


panels = [
    ("Bull call spread (95/105)",
     bull_call_spread(ST, 95, 105, C(95), C(105))),
    ("Bear put spread (95/105)",
     bear_put_spread(ST, 95, 105, P(95), P(105))),
    ("Butterfly call (90/100/110)",
     butterfly_call(ST, 90, 100, 110, C(90), C(100), C(110))),
    ("Straddle (K=100)",
     straddle(ST, 100, C(100), P(100))),
    ("Strangle (95 put / 105 call)",
     strangle(ST, 95, 105, P(95), C(105))),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (title, gain) in zip(axes.ravel(), panels):
    ax.plot(ST, gain, lw=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(S, color="grey", ls=":", lw=1)
    ax.fill_between(ST, gain, 0, where=gain >= 0, color="C2", alpha=0.25)
    ax.fill_between(ST, gain, 0, where=gain < 0, color="C3", alpha=0.25)
    ax.set_title(title); ax.set_xlabel("underlying at expiry  S_T")
    ax.set_ylabel("net profit"); ax.grid(alpha=0.3)
axes.ravel()[-1].axis("off")

fig.suptitle("Option strategy net-profit diagrams (premiums via Black-Scholes)", y=1.02)
fig.tight_layout()
path = os.path.join(OUT, "05_strategies.png")
fig.savefig(path, dpi=130, bbox_inches="tight")
print(f"saved  {path}")
