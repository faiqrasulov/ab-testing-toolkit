"""Fixed-horizon frequentist tests with confidence intervals for absolute and relative lift.

All functions broadcast over numpy arrays, so the same code analyses one experiment in the
CLI and hundreds of thousands of simulated experiments in the validation studies.
Confidence intervals are always two-sided at level ``1 - alpha``; ``alternative`` only
changes the p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from abkit.power import Alternative, check_alternative

ArrayLike = float | np.ndarray
RelativeMethod = Literal["delta", "log"]


@dataclass(frozen=True)
class LiftResult:
    """Result of a two-sample comparison of B (treatment) against A (control)."""

    mean_a: ArrayLike
    mean_b: ArrayLike
    estimate: ArrayLike  # absolute lift, mean_b - mean_a
    se: ArrayLike
    ci_low: ArrayLike
    ci_high: ArrayLike
    relative: ArrayLike  # mean_b / mean_a - 1
    rel_ci_low: ArrayLike
    rel_ci_high: ArrayLike
    statistic: ArrayLike
    p_value: ArrayLike
    alpha: float
    method: str


def _p_value(stat: ArrayLike, alternative: str, dist=stats.norm, **kw) -> ArrayLike:
    check_alternative(alternative)
    if alternative == "two-sided":
        return 2 * dist.sf(np.abs(stat), **kw)
    if alternative == "larger":
        return dist.sf(stat, **kw)
    return dist.cdf(stat, **kw)


def relative_lift_ci(
    mean_a: ArrayLike,
    var_mean_a: ArrayLike,
    mean_b: ArrayLike,
    var_mean_b: ArrayLike,
    alpha: float = 0.05,
    method: RelativeMethod = "delta",
    crit: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """CI for the relative lift ``mean_b / mean_a - 1`` from two independent means.

    ``var_mean_*`` are the variances of the *means* (e.g. ``s^2 / n``).
    ``method="delta"``: first-order Taylor expansion of the ratio,
    ``Var(B/A) ~ Var(B)/A^2 + B^2 Var(A)/A^4``.
    ``method="log"``: delta method on ``log(B/A)`` and exponentiate (asymmetric interval,
    always above -100%). ``crit`` overrides the normal critical value (e.g. a t quantile).
    Returns ``(relative_lift, low, high)``; NaN where ``mean_a == 0``.
    """
    z = stats.norm.isf(alpha / 2) if crit is None else crit
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean_b / mean_a
        if method == "delta":
            se = np.sqrt(var_mean_b / mean_a**2 + mean_b**2 * var_mean_a / mean_a**4)
            return ratio - 1, ratio - z * se - 1, ratio + z * se - 1
        if method == "log":
            se = np.sqrt(var_mean_b / mean_b**2 + var_mean_a / mean_a**2)
            log_r = np.log(ratio)
            return ratio - 1, np.exp(log_r - z * se) - 1, np.exp(log_r + z * se) - 1
    raise ValueError("method must be 'delta' or 'log'")


def two_proportion_ztest(
    x_a: ArrayLike,
    n_a: ArrayLike,
    x_b: ArrayLike,
    n_b: ArrayLike,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    relative_method: RelativeMethod = "delta",
) -> LiftResult:
    """Two-proportion z-test (pooled SE for the test, unpooled Wald SE for the CIs)."""
    x_a, n_a, x_b, n_b = (np.asarray(v, dtype=float) for v in (x_a, n_a, x_b, n_b))
    p_a, p_b = x_a / n_a, x_b / n_b
    var_a, var_b = p_a * (1 - p_a) / n_a, p_b * (1 - p_b) / n_b
    diff = p_b - p_a
    se = np.sqrt(var_a + var_b)
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_pool > 0, diff / se_pool, 0.0)
    crit = stats.norm.isf(alpha / 2)
    rel, rel_lo, rel_hi = relative_lift_ci(p_a, var_a, p_b, var_b, alpha, relative_method)
    return LiftResult(
        mean_a=p_a,
        mean_b=p_b,
        estimate=diff,
        se=se,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        relative=rel,
        rel_ci_low=rel_lo,
        rel_ci_high=rel_hi,
        statistic=z,
        p_value=_p_value(z, alternative),
        alpha=alpha,
        method="two-proportion z-test",
    )


def welch_ttest_from_stats(
    mean_a: ArrayLike,
    var_a: ArrayLike,
    n_a: ArrayLike,
    mean_b: ArrayLike,
    var_b: ArrayLike,
    n_b: ArrayLike,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    relative_method: RelativeMethod = "delta",
) -> LiftResult:
    """Welch's unequal-variance t-test from summary statistics (``var_*`` use ddof=1)."""
    mean_a, var_a, n_a, mean_b, var_b, n_b = (
        np.asarray(v, dtype=float) for v in (mean_a, var_a, n_a, mean_b, var_b, n_b)
    )
    vm_a, vm_b = var_a / n_a, var_b / n_b
    se = np.sqrt(vm_a + vm_b)
    diff = mean_b - mean_a
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vm_a + vm_b) ** 2 / (vm_a**2 / (n_a - 1) + vm_b**2 / (n_b - 1))
        t = diff / se
    crit = stats.t.isf(alpha / 2, df)
    rel, rel_lo, rel_hi = relative_lift_ci(
        mean_a, vm_a, mean_b, vm_b, alpha, relative_method, crit=crit
    )
    return LiftResult(
        mean_a=mean_a,
        mean_b=mean_b,
        estimate=diff,
        se=se,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        relative=rel,
        rel_ci_low=rel_lo,
        rel_ci_high=rel_hi,
        statistic=t,
        p_value=_p_value(t, alternative, stats.t, df=df),
        alpha=alpha,
        method="Welch t-test",
    )


def welch_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    relative_method: RelativeMethod = "delta",
) -> LiftResult:
    """Welch's t-test on raw observations; the last axis indexes units (batched if 2-D)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return welch_ttest_from_stats(
        a.mean(axis=-1),
        a.var(axis=-1, ddof=1),
        a.shape[-1],
        b.mean(axis=-1),
        b.var(axis=-1, ddof=1),
        b.shape[-1],
        alpha,
        alternative,
        relative_method,
    )
