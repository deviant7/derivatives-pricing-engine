"""
Demo 1 — four methods, one price.

Prices a European call four independent ways (Black-Scholes closed form, CRR
binomial tree, Monte-Carlo GBM, Crank-Nicolson PDE) and shows they agree, then
plots the convergence of the binomial tree (O(1/N)) and Monte-Carlo (O(1/sqrt(M)))
to the closed-form value.

Run:  python scripts/01_convergence.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")   # so plots' console echo never crashes
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpe import bs_price, binomial_price, mc_price, pde_price

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

# --- Contract ---------------------------------------------------------------
S, K, T, r, sigma, opt = 100.0, 100.0, 1.0, 0.06, 0.20, "call"

bs = bs_price(S, K, T, r, sigma, opt)
tree = binomial_price(S, K, T, r, sigma, N=2000, option_type=opt)
mc = mc_price(S, K, T, r, sigma, opt, n_paths=1_000_000, seed=0)
pde = pde_price(S, K, T, r, sigma, opt, M=600, N=600)

print("European call  S=100 K=100 T=1 r=6% sigma=20%")
print("-" * 52)
print(f"  Black-Scholes (closed form) : {bs:10.5f}")
print(f"  CRR binomial   (N=2000)     : {tree:10.5f}   |err|={abs(tree-bs):.2e}")
print(f"  Monte-Carlo    (1e6 paths)  : {mc.price:10.5f}   |err|={abs(mc.price-bs):.2e}  (se={mc.stderr:.2e})")
print(f"  Crank-Nicolson PDE (600x600): {pde:10.5f}   |err|={abs(pde-bs):.2e}")
print(f"  max pairwise disagreement   : {max(abs(tree-bs), abs(mc.price-bs), abs(pde-bs)):.2e}")

# --- Binomial convergence ---------------------------------------------------
Ns = np.unique(np.round(np.logspace(1, 3.2, 30)).astype(int))
tree_err = np.array([abs(binomial_price(S, K, T, r, sigma, N=n, option_type=opt) - bs)
                     for n in Ns])

# --- Monte-Carlo convergence ------------------------------------------------
Ms = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
mc_err, mc_se = [], []
for m in Ms:
    res = mc_price(S, K, T, r, sigma, opt, n_paths=int(m), seed=1)
    mc_err.append(abs(res.price - bs))
    mc_se.append(res.stderr)
mc_err, mc_se = np.array(mc_err), np.array(mc_se)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.loglog(Ns, tree_err, "o-", ms=3, label="binomial |error|")
ax1.loglog(Ns, tree_err[0] * Ns[0] / Ns, "k--", lw=1, label="O(1/N) reference")
ax1.set_xlabel("tree steps  N"); ax1.set_ylabel("|price − Black-Scholes|")
ax1.set_title("CRR binomial → Black-Scholes  (O(1/N), with even/odd oscillation)")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

ax2.loglog(Ms, mc_err, "o-", ms=4, label="MC |error|")
ax2.loglog(Ms, mc_se, "s--", ms=3, label="MC standard error")
ax2.loglog(Ms, mc_se[0] * np.sqrt(Ms[0] / Ms), "k:", lw=1, label="O(1/√M) reference")
ax2.set_xlabel("paths  M"); ax2.set_ylabel("error")
ax2.set_title("Monte-Carlo → Black-Scholes  (O(1/√M))")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
path = os.path.join(OUT, "01_convergence.png")
fig.savefig(path, dpi=130)
print(f"\nsaved  {path}")
