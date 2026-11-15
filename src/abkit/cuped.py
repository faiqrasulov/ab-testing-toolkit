"""CUPED variance reduction (Deng, Xu, Kohavi & Walker, WSDM 2013).

Each unit's metric ``y`` is adjusted with a pre-experiment covariate ``x``:

    y_adj = y - theta * (x - mean(x)),   theta = Cov(x, y) / Var(x)

``theta`` is estimated on both groups pooled. Because ``x`` is measured before
randomisation it is independent of assignment, so the adjusted difference in means is
still unbiased, while its variance shrinks by the factor ``1 - rho^2`` (rho = corr(x, y)).
Functions operate along the last axis, so a batch of simulated experiments can be passed
as 2-D arrays of shape ``(reps, n)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abkit.frequentist import ArrayLike, LiftResult, RelativeMethod, welch_ttest
from abkit.power import Alternative


@dataclass(frozen=True)
class CupedResult:
    theta: ArrayLike
    correlation: ArrayLike
    variance_reduction: ArrayLike  # 1 - Var(adjusted estimate) / Var(unadjusted estimate)
    adjusted: LiftResult
    unadjusted: LiftResult


def cuped_theta(y: np.ndarray, x: np.ndarray) -> tuple[ArrayLike, ArrayLike]:
    """OLS slope of ``y`` on ``x`` and their Pearson correlation, along the last axis."""
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    sxy = (xc * yc).sum(axis=-1)
    sxx = (xc**2).sum(axis=-1)
    syy = (yc**2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(sxx > 0, sxy / sxx, 0.0)
        rho = np.where((sxx > 0) & (syy > 0), sxy / np.sqrt(sxx * syy), 0.0)
    return theta, rho


def cuped(
    y_a: np.ndarray,
    x_a: np.ndarray,
    y_b: np.ndarray,
    x_b: np.ndarray,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    relative_method: RelativeMethod = "delta",
) -> CupedResult:
    """CUPED-adjusted Welch comparison of B vs A using pre-experiment covariate ``x``."""
    y_a, x_a, y_b, x_b = (np.asarray(v, dtype=float) for v in (y_a, x_a, y_b, x_b))
    y = np.concatenate([y_a, y_b], axis=-1)
    x = np.concatenate([x_a, x_b], axis=-1)
    theta, rho = cuped_theta(y, x)
    t = np.expand_dims(theta, -1)
    x_mean = x.mean(axis=-1, keepdims=True)
    adjusted = welch_ttest(
        y_a - t * (x_a - x_mean), y_b - t * (x_b - x_mean), alpha, alternative, relative_method
    )
    unadjusted = welch_ttest(y_a, y_b, alpha, alternative, relative_method)
    variance_reduction = 1 - (adjusted.se / unadjusted.se) ** 2
    return CupedResult(theta, rho, variance_reduction, adjusted, unadjusted)
