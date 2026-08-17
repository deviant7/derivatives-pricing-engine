"""
Demo 2 — the Greeks, verified.

Prints the five analytic Black-Scholes Greeks alongside finite-difference
("bump-and-reprice") Greeks of the same pricer, and plots Delta and Gamma
across spot.

Run:  python scripts/02_greeks.py
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

from dpe import bs_greeks, verify_greeks

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

S, K, T, r, sigma, opt = 100.0, 100.0, 0.5, 0.06, 0.25, "call"

print(f"Greeks - European {opt}  S={S} K={K} T={T} r={r} sigma={sigma}")
print("-" * 60)
print(f"{'greek':<8}{'analytic':>14}{'finite-diff':>16}{'|error|':>14}")
for g, (an, fd, err) in verify_greeks(S, K, T, r, sigma, opt).items():
    print(f"{g:<8}{an:>14.6f}{fd:>16.6f}{err:>14.2e}")

# Delta / Gamma across spot
spots = np.linspace(60, 140, 200)
gk = bs_greeks(spots, K, T, r, sigma, opt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(spots, gk["delta"], lw=2)
ax1.axvline(K, color="grey", ls=":", lw=1)
ax1.set_xlabel("spot  S"); ax1.set_ylabel("Delta")
ax1.set_title("Call Delta vs spot"); ax1.grid(alpha=0.3)

ax2.plot(spots, gk["gamma"], lw=2, color="C3")
ax2.axvline(K, color="grey", ls=":", lw=1)
ax2.set_xlabel("spot  S"); ax2.set_ylabel("Gamma")
ax2.set_title("Call Gamma vs spot (peaks near the strike)"); ax2.grid(alpha=0.3)

fig.tight_layout()
path = os.path.join(OUT, "02_greeks.png")
fig.savefig(path, dpi=130)
print(f"\nsaved  {path}")
