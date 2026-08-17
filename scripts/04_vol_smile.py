"""
Demo 4 — the implied-volatility smile.

Real option markets do NOT quote a single flat Black-Scholes volatility: implied
vol varies with strike (the "smile"/"skew"). This script builds a synthetic
option chain from a known smile, inverts Black-Scholes strike-by-strike to
recover the implied vols, and plots the smile — recovering the input curve and
exposing the limitation of the constant-sigma Black-Scholes assumption.

To use REAL data, drop a CSV at data/nifty_chain.csv with columns
[strike, price, type] and set S, T, r below; the same code path handles it.

Run:  python scripts/04_vol_smile.py
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
import pandas as pd

from dpe import bs_price, vol_smile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)

S, T, r = 100.0, 0.25, 0.06
csv_path = os.path.join(DATA, "synthetic_chain.csv")


def true_smile(moneyness):
    """A downward-sloping, convex skew, typical of equity-index options."""
    x = moneyness - 1.0
    return 0.22 - 0.18 * x + 0.30 * x ** 2


# Build a synthetic chain (OTM calls above spot, OTM puts below) unless a real
# chain is provided.
real = os.path.join(DATA, "nifty_chain.csv")
if os.path.exists(real):
    chain = pd.read_csv(real)
    print(f"using real chain: {real}")
else:
    strikes = np.arange(80, 121, 2.5)
    rows = []
    for Kk in strikes:
        typ = "call" if Kk >= S else "put"
        iv = true_smile(Kk / S)
        rows.append({"strike": Kk, "type": typ,
                     "price": bs_price(S, Kk, T, r, iv, typ)})
    chain = pd.DataFrame(rows)
    chain.to_csv(csv_path, index=False)
    print(f"wrote synthetic chain: {csv_path}")

smile = vol_smile(chain, S, T, r)
print(smile[["strike", "type", "price", "implied_vol", "moneyness"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(smile["moneyness"], 100 * smile["implied_vol"], "o-", label="recovered implied vol")
mm = np.linspace(smile["moneyness"].min(), smile["moneyness"].max(), 200)
ax.plot(mm, 100 * true_smile(mm), "k--", lw=1, label="input smile")
ax.axhline(100 * 0.22, color="grey", ls=":", lw=1, label="flat BS assumption")
ax.axvline(1.0, color="grey", lw=0.8, alpha=0.6)
ax.set_xlabel("moneyness  K / S"); ax.set_ylabel("implied volatility (%)")
ax.set_title("Implied-volatility smile / skew")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT, "04_vol_smile.png")
fig.savefig(path, dpi=130)
print(f"\nsaved  {path}")
