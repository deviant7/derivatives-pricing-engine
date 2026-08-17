"""
Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

Prices European and American vanilla options by numerically solving
    dV/dt + 0.5 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0
backward in time on a uniform price grid. For a European option this must
agree with the closed-form Black-Scholes price; American exercise is handled
by projecting V := max(V, payoff) after each time step (Brennan-Schwartz style).

Reference: SI 527, Ch. 7 (the Black-Scholes differential equation).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def pde_price(S, K, T, r, sigma, option_type="call", exercise="european",
              q=0.0, M=400, N=400, Smax_mult=4.0):
    """
    Parameters
    ----------
    M : number of price steps.
    N : number of time steps.
    Smax_mult : far boundary Smax = Smax_mult * max(S, K).
    """
    Smax = Smax_mult * max(S, K)
    dS = Smax / M
    dt = T / N
    S_grid = np.linspace(0.0, Smax, M + 1)

    if option_type == "call":
        payoff = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    V = payoff.copy()

    # Crank-Nicolson coefficients for interior nodes i = 1 .. M-1.
    ii = np.arange(1, M)
    a = 0.25 * dt * (sigma ** 2 * ii ** 2 - (r - q) * ii)
    b = -0.5 * dt * (sigma ** 2 * ii ** 2 + r)
    c = 0.25 * dt * (sigma ** 2 * ii ** 2 + (r - q) * ii)

    # Banded left-hand matrix (implicit half-step): tridiag(-a, 1 - b, -c).
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -c[:-1]     # super-diagonal
    ab[1, :] = 1.0 - b      # main diagonal
    ab[2, :-1] = -a[1:]     # sub-diagonal

    american = exercise == "american"

    def boundaries(tau):
        if option_type == "call":
            return 0.0, Smax * np.exp(-q * tau) - K * np.exp(-r * tau)
        return K * np.exp(-r * tau), 0.0

    for n in range(N, 0, -1):
        tau_prev = n * dt          # explicit (later) level, already known
        tau_now = (n - 1) * dt     # implicit (earlier) level, being solved
        V0_prev, VM_prev = boundaries(tau_prev)
        V0_now, VM_now = boundaries(tau_now)

        Vint = V[1:M]
        left = np.concatenate([[V0_prev], Vint[:-1]])
        right = np.concatenate([Vint[1:], [VM_prev]])
        rhs = a * left + (1.0 + b) * Vint + c * right

        # Move known boundary values at the implicit level to the RHS.
        rhs[0] += a[0] * V0_now
        rhs[-1] += c[-1] * VM_now

        Vint_now = solve_banded((1, 1), ab, rhs)
        V = np.concatenate([[V0_now], Vint_now, [VM_now]])

        if american:
            V = np.maximum(V, payoff)

    return float(np.interp(S, S_grid, V))
