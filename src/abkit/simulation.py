"""Vectorised Monte Carlo building blocks used by the validation studies and the tests.

Every function takes an explicit ``numpy.random.Generator`` so results are reproducible,
and simulates many experiments at once (rows = replications). Conversions are drawn as
binomial counts per look, which is exactly equivalent to simulating individual users.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import stats

from abkit.cuped import cuped
from abkit.frequentist import (
    LiftResult,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_stats,
)
from abkit.sequential import GroupSequentialDesign, always_valid_pvalues, first_crossing


def binary_paths(
    rng: np.random.Generator, reps: int, n_cum: np.ndarray, p_a: float, p_b: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative conversions per arm at each look, each of shape ``(reps, n_looks)``."""
    inc = np.diff(np.r_[0, np.asarray(n_cum)])
    x_a = rng.binomial(inc, p_a, size=(reps, inc.size)).cumsum(axis=1)
    x_b = rng.binomial(inc, p_b, size=(reps, inc.size)).cumsum(axis=1)
    return x_a, x_b


@dataclass(frozen=True)
class SequentialOutcome:
    """Per-replication decision and the fraction of the maximum sample used."""

    rejected: np.ndarray
    sample_fraction: np.ndarray


def _outcome(first: np.ndarray, frac: np.ndarray) -> SequentialOutcome:
    return SequentialOutcome(first >= 0, np.where(first >= 0, frac[np.maximum(first, 0)], 1.0))


def sequential_outcomes(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_cum: np.ndarray,
    alpha: float,
    design: GroupSequentialDesign | None = None,
    tau: float | None = None,
) -> dict[str, SequentialOutcome]:
    """Decisions of each monitoring strategy on the same simulated paths (equal arms).

    ``fixed``: one z-test at the final look. ``naive``: z-test at every look, stop at the
    first p < alpha. ``group_sequential``: Lan-DeMets boundaries from ``design``.
    ``msprt``: stop when the always-valid p-value drops below alpha.
    """
    n_cum = np.asarray(n_cum, dtype=float)
    frac = n_cum / n_cum[-1]
    z = two_proportion_ztest(x_a, n_cum, x_b, n_cum).statistic
    crit = stats.norm.isf(alpha / 2)
    out = {
        "fixed": SequentialOutcome(np.abs(z[:, -1]) >= crit, np.ones(len(z))),
        "naive": _outcome(first_crossing(z, crit), frac),
    }
    if design is not None:
        if design.boundaries.size != n_cum.size:
            raise ValueError("design has a different number of looks than the paths")
        out["group_sequential"] = _outcome(first_crossing(z, design.boundaries), frac)
    if tau is not None:
        p_pool = (x_a + x_b) / (2 * n_cum)
        var0 = p_pool * (1 - p_pool) * 2 / n_cum
        p_av = always_valid_pvalues(x_b / n_cum - x_a / n_cum, var0, tau)
        hit = p_av <= alpha
        out["msprt"] = _outcome(np.where(hit.any(axis=1), hit.argmax(axis=1), -1), frac)
    return out


def lognormal_correlation(r: float, sigma: float) -> float:
    """Pearson correlation of ``exp(sigma U)`` and ``exp(sigma V)`` when corr(U, V) = r."""
    return float(np.expm1(sigma**2 * r) / np.expm1(sigma**2))


def cuped_trials(
    rng: np.random.Generator,
    reps: int,
    n: int,
    r: float,
    effect: float,
    distribution: str = "normal",
    sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate ``reps`` experiments with a pre-period covariate and analyse each with CUPED.

    The pre-period metric ``x`` and outcome ``y`` share a latent correlation ``r``; for
    ``distribution="lognormal"`` both are ``exp(sigma * latent)`` (skewed, revenue-like) and
    the Pearson correlation is :func:`lognormal_correlation`. Treatment adds ``effect`` to y.
    """
    u = rng.standard_normal((reps, 2 * n))
    v = r * u + np.sqrt(1 - r**2) * rng.standard_normal((reps, 2 * n))
    if distribution == "lognormal":
        u, v = np.exp(sigma * u), np.exp(sigma * v)
    elif distribution != "normal":
        raise ValueError("distribution must be 'normal' or 'lognormal'")
    v[:, n:] += effect
    res = cuped(v[:, :n], u[:, :n], v[:, n:], u[:, n:])
    return {
        "plain": res.unadjusted.estimate,
        "adjusted": res.adjusted.estimate,
        "adjusted_ci_low": res.adjusted.ci_low,
        "adjusted_ci_high": res.adjusted.ci_high,
        "reported_variance_reduction": res.variance_reduction,
    }


def variance_ratio(plain: np.ndarray, adjusted: np.ndarray) -> tuple[float, float]:
    """``Var(adjusted) / Var(plain)`` over replications, with a delta-method standard error."""
    a = (plain - plain.mean()) ** 2
    b = (adjusted - adjusted.mean()) ** 2
    ma, mb = a.mean(), b.mean()
    cov = np.cov(a, b)
    var = (cov[1, 1] / ma**2 + mb**2 * cov[0, 0] / ma**4 - 2 * mb * cov[0, 1] / ma**3) / a.size
    return float(mb / ma), float(np.sqrt(var))


def binary_coverage(
    rng: np.random.Generator, reps: int, n: int, p_a: float, rel_lift: float, alpha: float = 0.05
) -> dict[str, float]:
    """Coverage of the absolute (Wald) and relative (delta, log) CIs for conversion rates."""
    p_b = p_a * (1 + rel_lift)
    x_a = rng.binomial(n, p_a, reps)
    x_b = rng.binomial(n, p_b, reps)
    hits = _ci_hits(
        partial(two_proportion_ztest, x_a, n, x_b, n, alpha, "two-sided"), p_b - p_a, rel_lift
    )
    return _coverage_summary(hits)


def lognormal_coverage(
    rng: np.random.Generator,
    reps: int,
    n: int,
    sigma: float,
    rel_lift: float,
    alpha: float = 0.05,
    chunk: int = 2000,
) -> dict[str, float]:
    """Coverage of Welch and relative-lift CIs on skewed lognormal (revenue-like) data."""
    true_abs = np.exp(sigma**2 / 2) * rel_lift
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        a = rng.lognormal(0.0, sigma, (size, n))
        b = rng.lognormal(0.0, sigma, (size, n)) * (1 + rel_lift)
        parts.append(_ci_hits(partial(welch_ttest, a, b, alpha, "two-sided"), true_abs, rel_lift))
    return _coverage_summary({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def _ci_hits(
    analyse: Callable[[str], LiftResult], true_abs: float, true_rel: float
) -> dict[str, np.ndarray]:
    """Per-replication indicators that each CI covers the truth (and where it is undefined)."""
    delta, log = analyse("delta"), analyse("log")
    return {
        "absolute": (delta.ci_low <= true_abs) & (true_abs <= delta.ci_high),
        "relative_delta": (delta.rel_ci_low <= true_rel) & (true_rel <= delta.rel_ci_high),
        "relative_log": (log.rel_ci_low <= true_rel) & (true_rel <= log.rel_ci_high),
        "relative_undefined": ~(np.isfinite(delta.rel_ci_low) & np.isfinite(delta.rel_ci_high)),
    }


def _coverage_summary(hits: dict[str, np.ndarray]) -> dict[str, float]:
    """Coverage rates; relative CIs are scored only where defined (control mean > 0)."""
    defined = ~hits["relative_undefined"]
    out = {"absolute": float(hits["absolute"].mean())}
    for k in ("relative_delta", "relative_log"):
        out[k] = float(hits[k][defined].mean()) if defined.any() else float("nan")
    out["relative_undefined"] = float((~defined).mean())
    return out


def binary_power(
    rng: np.random.Generator, reps: int, n: int, p_a: float, p_b: float, alpha: float = 0.05
) -> dict[str, float]:
    """Empirical power of the z-test plus the exaggeration ratio of significant estimates.

    ``exaggeration`` (Gelman & Carlin's type-M error) is the mean significant positive
    estimate divided by the true effect: how much a "winning" result overstates the lift.
    """
    x_a = rng.binomial(n, p_a, reps)
    x_b = rng.binomial(n, p_b, reps)
    res = two_proportion_ztest(x_a, n, x_b, n, alpha)
    return _power_summary(res.estimate, res.p_value < alpha, p_b - p_a)


def normal_means_power(
    rng: np.random.Generator,
    reps: int,
    n: int,
    delta: float,
    sd: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Empirical power of Welch's test on normal data via exact sufficient statistics."""
    mean_a = rng.normal(0.0, sd / np.sqrt(n), reps)
    mean_b = rng.normal(delta, sd / np.sqrt(n), reps)
    var_a = sd**2 * rng.chisquare(n - 1, reps) / (n - 1)
    var_b = sd**2 * rng.chisquare(n - 1, reps) / (n - 1)
    res = welch_ttest_from_stats(mean_a, var_a, n, mean_b, var_b, n, alpha)
    return _power_summary(res.estimate, res.p_value < alpha, delta)


def _power_summary(estimate: np.ndarray, significant: np.ndarray, true_effect: float) -> dict:
    out = {"power": float(significant.mean())}
    win = significant & (estimate > 0)
    if true_effect > 0 and win.any():
        out["exaggeration"] = float(estimate[win].mean() / true_effect)
        out["wrong_sign_share"] = float((significant & (estimate < 0)).sum() / significant.sum())
    return out
