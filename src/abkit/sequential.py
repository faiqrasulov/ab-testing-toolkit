"""Valid sequential testing: group-sequential boundaries and always-valid p-values.

Two methods are implemented from scratch:

1. **Group-sequential boundaries via Lan-DeMets alpha spending** (Lan & DeMets 1983).
   The z-statistic at information fraction ``t_k`` is ``Z_k = S(t_k) / sqrt(t_k)`` where ``S``
   is Brownian motion with drift. Boundaries ``c_k`` are found look by look so that the
   probability of first crossing ``|Z_k| >= c_k`` at look ``k`` equals the alpha spent in
   ``(t_{k-1}, t_k]``. The sub-density of ``S`` on the continuation region is propagated
   with the Armitage-McPherson-Rowe recursion, integrated on a grid with Simpson's rule
   (Jennison & Turnbull 2000, ch. 19); exit probabilities are exact normal tail integrals.

2. **Mixture SPRT** (Robbins 1970; Johari, Koomen, Pekelis & Walsh 2017). For an estimate
   ``D ~ N(theta, V)`` and a ``N(0, tau^2)`` mixing distribution over ``theta`` the mixture
   likelihood ratio against ``theta = 0`` is

       Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 D^2 / (2 V (V + tau^2)))

   which is a nonnegative martingale under H0, so by Ville's inequality
   ``P(sup_n Lambda_n >= 1/alpha) <= alpha``: the running minimum of ``1/Lambda`` is a
   p-value that stays valid under continuous monitoring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

SpendingFunction = Callable[[np.ndarray, float], np.ndarray]

_MAX_GRID = 2001


def obrien_fleming_spending(t: np.ndarray, alpha: float) -> np.ndarray:
    """Lan-DeMets O'Brien-Fleming-type spending for one-sided level ``alpha``.

    ``alpha(t) = 2 - 2 Phi(z_{alpha/2} / sqrt(t))``: almost nothing is spent early, so early
    stops need overwhelming evidence and the final boundary stays close to the fixed test.
    """
    t = np.asarray(t, dtype=float)
    return 2 * stats.norm.sf(stats.norm.isf(alpha / 2) / np.sqrt(t))


def pocock_spending(t: np.ndarray, alpha: float) -> np.ndarray:
    """Lan-DeMets Pocock-type spending: ``alpha * log(1 + (e - 1) t)``."""
    return alpha * np.log1p((np.e - 1) * np.asarray(t, dtype=float))


_SPENDING: dict[str, SpendingFunction] = {
    "obf": obrien_fleming_spending,
    "pocock": pocock_spending,
}


@dataclass(frozen=True)
class GroupSequentialDesign:
    """Two-sided symmetric efficacy boundaries on the z scale."""

    info_fractions: np.ndarray
    boundaries: np.ndarray
    cumulative_alpha: np.ndarray
    alpha: float
    spending: str

    @property
    def nominal_alpha(self) -> np.ndarray:
        """Per-look two-sided p-value threshold equivalent to each boundary."""
        return 2 * stats.norm.sf(self.boundaries)


def _info(info_fractions: np.ndarray) -> np.ndarray:
    t = np.asarray(info_fractions, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("info_fractions must be strictly increasing and positive")
    return t


def _grid_step(t: np.ndarray) -> float:
    return float(np.sqrt(np.min(np.diff(np.r_[0.0, t]))) / 10)


def _grid(b: float, t_k: float, drift: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Simpson grid on the continuation region (-b, b) of S(t_k), truncated where negligible."""
    b = min(b, abs(drift) * t_k + 12 * np.sqrt(t_k))
    m = int(min(_MAX_GRID, max(101, np.ceil(2 * b / h) + 1)))
    m += 1 - m % 2  # Simpson needs an odd number of points
    s = np.linspace(-b, b, m)
    w = np.ones(m)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return s, w * (s[1] - s[0]) / 3


def _exit_prob(s: np.ndarray, wf: np.ndarray, dt: float, drift: float, b: float) -> float:
    """P(continue to the previous look, then |S| >= b at the next look)."""
    mu = s + drift * dt
    sd = np.sqrt(dt)
    tail = stats.norm.sf((b - mu) / sd) + stats.norm.cdf((-b - mu) / sd)
    return float(wf @ tail)


def _propagate(
    s: np.ndarray, wf: np.ndarray, s_new: np.ndarray, dt: float, drift: float
) -> np.ndarray:
    """Convolve the continuation sub-density with the N(drift*dt, dt) increment."""
    sd = np.sqrt(dt)
    kernel = stats.norm.pdf((s_new[:, None] - s[None, :] - drift * dt) / sd) / sd
    return kernel @ wf


def _first_look(c: float, t1: float, drift: float, h: float):
    sd, mu = np.sqrt(t1), drift * t1
    b = c * sd
    p_exit = float(stats.norm.sf((b - mu) / sd) + stats.norm.cdf((-b - mu) / sd))
    s, w = _grid(b, t1, drift, h)
    return p_exit, s, w * stats.norm.pdf(s, mu, sd)


def _solve_boundary(
    s: np.ndarray, wf: np.ndarray, dt: float, sqrt_t: float, target: float
) -> float:
    """z-boundary whose first-crossing probability at this look equals ``target``."""
    if target <= 0:
        return np.inf
    return float(
        optimize.brentq(
            lambda ck: _exit_prob(s, wf, dt, 0.0, ck * sqrt_t) - target,
            1e-6,
            60.0,
            xtol=1e-10,
            rtol=1e-12,
        )
    )


def crossing_probabilities(
    boundaries: np.ndarray, info_fractions: np.ndarray, drift: float = 0.0
) -> np.ndarray:
    """Probability of *first* crossing ``|Z_k| >= c_k`` at each look.

    ``drift`` is the expected z-statistic at information fraction 1 (0 under H0). The sum
    is the overall rejection probability: type-I error under H0, power under H1.
    """
    t = _info(info_fractions)
    c = np.broadcast_to(np.asarray(boundaries, dtype=float), t.shape)
    h = _grid_step(t)
    probs = np.empty(t.size)
    probs[0], s, wf = _first_look(c[0], t[0], drift, h)
    for k in range(1, t.size):
        dt = t[k] - t[k - 1]
        b = c[k] * np.sqrt(t[k])
        probs[k] = _exit_prob(s, wf, dt, drift, b)
        if k < t.size - 1:
            s_new, w_new = _grid(b, t[k], drift, h)
            wf = w_new * _propagate(s, wf, s_new, dt, drift)
            s = s_new
    return probs


def expected_sample_fraction(
    boundaries: np.ndarray, info_fractions: np.ndarray, drift: float = 0.0
) -> float:
    """Expected fraction of the maximum sample size used before stopping."""
    t = _info(info_fractions)
    p = crossing_probabilities(boundaries, t, drift)
    return float(np.sum(t[:-1] * p[:-1]) + t[-1] * (1 - p[:-1].sum()))


def group_sequential_boundaries(
    n_looks: int | None = None,
    info_fractions: np.ndarray | None = None,
    alpha: float = 0.05,
    spending: str | SpendingFunction = "obf",
) -> GroupSequentialDesign:
    """Two-sided symmetric efficacy boundaries for a Lan-DeMets alpha-spending design.

    Give either ``n_looks`` (equally spaced looks) or explicit ``info_fractions`` ending at 1.
    Spending functions take a one-sided level; the overall two-sided type-I error is ``alpha``.
    """
    if info_fractions is None:
        if not n_looks or n_looks < 1:
            raise ValueError("give n_looks >= 1 or info_fractions")
        info_fractions = np.arange(1, n_looks + 1) / n_looks
    t = _info(info_fractions)
    if not np.isclose(t[-1], 1.0):
        raise ValueError("the last information fraction must be 1")
    fn = _SPENDING[spending] if isinstance(spending, str) else spending
    name = spending if isinstance(spending, str) else getattr(fn, "__name__", "custom")
    # Symmetric two-sided design: each side spends fn(t, alpha / 2) (gsDesign/ldbounds convention).
    cum = np.minimum(np.maximum.accumulate(2 * fn(t, alpha / 2)), alpha)
    cum[-1] = alpha
    increments = np.diff(np.r_[0.0, cum])

    h = _grid_step(t)
    c = np.empty(t.size)
    c[0] = stats.norm.isf(increments[0] / 2) if increments[0] > 0 else np.inf
    _, s, wf = _first_look(c[0], t[0], 0.0, h)
    for k in range(1, t.size):
        dt, sqrt_t = t[k] - t[k - 1], np.sqrt(t[k])
        c[k] = _solve_boundary(s, wf, dt, sqrt_t, increments[k])
        if k < t.size - 1:
            s_new, w_new = _grid(c[k] * sqrt_t, t[k], 0.0, h)
            wf = w_new * _propagate(s, wf, s_new, dt, 0.0)
            s = s_new
    return GroupSequentialDesign(t, c, cum, alpha, name)


def first_crossing(z: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Index of the first look with ``|z| >= boundary`` along the last axis (-1 if none)."""
    hit = np.abs(z) >= np.asarray(boundaries)
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1), -1)


def msprt_log_likelihood_ratio(
    estimate: np.ndarray, variance: np.ndarray, tau: float, theta0: float = 0.0
) -> np.ndarray:
    """Log mixture likelihood ratio for ``estimate ~ N(theta, variance)``, prior N(theta0, tau^2).

    Returns 0 (no evidence) where ``variance <= 0``, e.g. before any conversions are seen.
    """
    est = np.asarray(estimate, dtype=float)
    v = np.asarray(variance, dtype=float)
    t2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * np.log(v / (v + t2)) + t2 * (est - theta0) ** 2 / (2 * v * (v + t2))
    return np.where(v > 0, out, 0.0)


def always_valid_pvalues(
    estimate: np.ndarray, variance: np.ndarray, tau: float, theta0: float = 0.0
) -> np.ndarray:
    """Always-valid p-values ``p_n = min(1, min_{m<=n} 1/Lambda_m)`` along the last axis."""
    p = np.minimum(1.0, np.exp(-msprt_log_likelihood_ratio(estimate, variance, tau, theta0)))
    return np.minimum.accumulate(p, axis=-1)


def msprt_z_boundary(variance: np.ndarray, tau: float, alpha: float = 0.05) -> np.ndarray:
    """|z| threshold at which the mSPRT rejects (``Lambda >= 1/alpha``) for a given variance.

    Solving ``Lambda = 1/alpha`` for ``z = D / sqrt(V)`` gives
    ``z^2 = (1 + V/tau^2) * (2 log(1/alpha) + log(1 + tau^2/V))``: very strict while the
    estimate is noisy, then growing only like ``sqrt(log n)`` as data accumulate.
    """
    v = np.asarray(variance, dtype=float)
    t2 = tau**2
    return np.sqrt((1 + v / t2) * (2 * np.log(1 / alpha) + np.log1p(t2 / v)))
