"""
Demo 3 — dynamic delta-hedging.

Sells a European call and delta-hedges it along simulated GBM paths. Shows (a)
the terminal hedging-P&L distribution collapsing toward zero as rebalancing
gets finer, and (b) the hedging-error standard deviation shrinking like
O(1/sqrt(N_rebalance)). Also demonstrates the bias from hedging at the wrong vol.

Run:  python scripts/03_delta_hedging.py
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

from dpe import simulate_delta_hedge

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

S0, K, T, r, sigma, opt = 100.0, 100.0, 1.0, 0.06, 0.20, "call"

freqs = [4, 12, 26, 52, 104, 252]
stds = []
print("Delta-hedging a short European call (hedge vol = realised vol)")
print("-" * 58)
print(f"{'rebalances/yr':>14}{'mean P&L':>12}{'std P&L':>12}")
for nf in freqs:
    pnl = simulate_delta_hedge(S0, K, T, r, sigma, opt, n_rebalance=nf,
                               n_paths=40_000, seed=7)
    stds.append(pnl.std())
    print(f"{nf:>14}{pnl.mean():>12.4f}{pnl.std():>12.4f}")
stds = np.array(stds)

# Wrong-vol bias: hedge at 15% while the market realises 25%.
pnl_biased = simulate_delta_hedge(S0, K, T, r, sigma=0.25, option_type=opt,
                                  n_rebalance=52, n_paths=40_000,
                                  sigma_hedge=0.15, seed=7)
print(f"\nHedging at 15% vol while realised vol is 25% "
      f"-> mean P&L = {pnl_biased.mean():.3f} (systematic loss for the seller)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for nf in (4, 52, 252):
    pnl = simulate_delta_hedge(S0, K, T, r, sigma, opt, n_rebalance=nf,
                               n_paths=40_000, seed=7)
    ax1.hist(pnl, bins=60, density=True, alpha=0.5, label=f"{nf} rebalances/yr")
ax1.set_xlabel("terminal hedging P&L"); ax1.set_ylabel("density")
ax1.set_title("Hedging P&L tightens around 0 as rebalancing gets finer")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.loglog(freqs, stds, "o-", label="std(P&L)")
ax2.loglog(freqs, stds[0] * np.sqrt(freqs[0] / np.array(freqs)), "k--", lw=1,
           label="O(1/√N) reference")
ax2.set_xlabel("rebalances per year  N"); ax2.set_ylabel("std of hedging P&L")
ax2.set_title("Hedging error ~ O(1/√N)")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
path = os.path.join(OUT, "03_delta_hedging.png")
fig.savefig(path, dpi=130)
print(f"\nsaved  {path}")
