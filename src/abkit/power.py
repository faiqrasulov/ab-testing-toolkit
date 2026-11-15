"""Power, sample size and minimum detectable effect (MDE) for two-sample A/B tests.

Conventions used throughout ``abkit``: group A is control, group B is treatment,
``n_a`` is the control sample size and ``ratio = n_b / n_a`` (as in statsmodels).
Power always includes both rejection tails for two-sided tests, so the functions are
exact inverses of each other up to the integer rounding of sample sizes.
"""

from __future__ import annotations

import math
from typing import Literal

from scipy import optimize, stats

Alternative = Literal["two-sided", "larger", "smaller"]
ProportionMethod = Literal["pooled", "arcsine"]
MeansMethod = Literal["t", "z"]

_N_MAX = 1e10


def check_alternative(alternative: str) -> None:
    if alternative not in ("two-sided", "larger", "smaller"):
        raise ValueError("alternative must be 'two-sided', 'larger' or 'smaller'")


def _normal_power(
    delta: float, sd_null: float, sd_alt: float, alpha: float, alternative: str
) -> float:
    """Power when the test rejects for |estimate| > z * sd_null and estimate ~ N(delta, sd_alt)."""
    check_alternative(alternative)
    a = alpha / 2 if alternative == "two-sided" else alpha
    z = stats.norm.isf(a)
    power = 0.0
    if alternative in ("two-sided", "larger"):
        power += stats.norm.sf((z * sd_null - delta) / sd_alt)
    if alternative in ("two-sided", "smaller"):
        power += stats.norm.cdf((-z * sd_null - delta) / sd_alt)
    return float(power)


def power_two_proportions(
    p_a: float,
    p_b: float,
    n_a: float,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    method: ProportionMethod = "pooled",
) -> float:
    """Power of the two-proportion z-test.

    ``method="pooled"`` is the classic normal approximation used by the z-test in this
    package (pooled variance under H0, unpooled under H1; matches
    ``statsmodels.stats.proportion.power_proportions_2indep``). ``method="arcsine"`` uses
    Cohen's h with the variance-stabilising transform (matches ``NormalIndPower``).
    """
    n_b = ratio * n_a
    if method == "pooled":
        p_bar = (n_a * p_a + n_b * p_b) / (n_a + n_b)
        sd_null = math.sqrt(p_bar * (1 - p_bar) * (1 / n_a + 1 / n_b))
        sd_alt = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        return _normal_power(p_b - p_a, sd_null, sd_alt, alpha, alternative)
    if method == "arcsine":
        h = 2 * math.asin(math.sqrt(p_b)) - 2 * math.asin(math.sqrt(p_a))
        sd = math.sqrt(1 / n_a + 1 / n_b)
        return _normal_power(h, sd, sd, alpha, alternative)
    raise ValueError("method must be 'pooled' or 'arcsine'")


def power_two_means(
    delta: float,
    sd: float,
    n_a: float,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    method: MeansMethod = "t",
) -> float:
    """Power of the two-sample test for a difference in means ``delta`` with common ``sd``.

    ``method="t"`` uses the noncentral t distribution (exact for normal data with equal
    variances; matches ``statsmodels.stats.power.TTestIndPower``); ``"z"`` uses the
    normal approximation.
    """
    check_alternative(alternative)
    n_b = ratio * n_a
    n_eff = 1 / (1 / n_a + 1 / n_b)
    if method == "z":
        se = sd / math.sqrt(n_eff)
        return _normal_power(delta, se, se, alpha, alternative)
    if method != "t":
        raise ValueError("method must be 't' or 'z'")
    df = n_a + n_b - 2
    nc = delta / sd * math.sqrt(n_eff)
    a = alpha / 2 if alternative == "two-sided" else alpha
    crit = stats.t.isf(a, df)
    power = 0.0
    if alternative in ("two-sided", "larger"):
        power += _nct_sf(crit, df, nc)
    if alternative in ("two-sided", "smaller"):
        power += _nct_sf(crit, df, -nc)  # P(T_nc < -crit) = P(T_{-nc} > crit) by symmetry
    return float(power)


def _nct_sf(x: float, df: float, nc: float) -> float:
    """Upper tail P(T > x) of the noncentral t distribution, guarded against NaN.

    Lower tails are computed by reflection because ``scipy.stats.nct.cdf`` (and
    ``scipy.special.nctdtr``) return NaN deep in the far tail for some (df, nc) in
    scipy >= 1.18, while ``sf`` stays finite. If a NaN still appears, the normal
    approximation is used; it only happens where the probability is negligible.
    """
    p = float(stats.nct.sf(x, df, nc))
    return p if math.isfinite(p) else float(stats.norm.sf(x - nc))


def _solve_n(power_fn, target: float, n_min: float) -> int:
    """Smallest integer n >= n_min with power_fn(n) >= target (power increases with n)."""
    if not 0 < target < 1:
        raise ValueError("power must be in (0, 1)")
    if power_fn(_N_MAX) < target:
        raise ValueError("target power is not reachable for this effect size")
    if power_fn(n_min) >= target:
        return math.ceil(n_min)
    n = optimize.brentq(lambda x: power_fn(x) - target, n_min, _N_MAX, xtol=1e-6, rtol=1e-12)
    n_int = math.ceil(n - 1e-9)
    while power_fn(n_int) < target:  # guard against root-finding tolerance at the boundary
        n_int += 1
    return n_int


def sample_size_two_proportions(
    p_a: float,
    p_b: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
    method: ProportionMethod = "pooled",
) -> int:
    """Control-group size ``n_a`` needed to detect ``p_a -> p_b`` (treatment: ``ratio * n_a``)."""
    if p_a == p_b:
        raise ValueError("p_a and p_b must differ")
    return _solve_n(
        lambda n: power_two_proportions(p_a, p_b, n, ratio, alpha, alternative, method),
        power,
        n_min=2.0,
    )


def sample_size_two_means(
    delta: float,
    sd: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
    method: MeansMethod = "t",
) -> int:
    """Control-group size ``n_a`` needed to detect a mean difference ``delta``."""
    if delta == 0:
        raise ValueError("delta must be non-zero")
    return _solve_n(
        lambda n: power_two_means(delta, sd, n, ratio, alpha, alternative, method),
        power,
        n_min=2.0,
    )


def mde_two_proportions(
    p_a: float,
    n_a: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
    method: ProportionMethod = "pooled",
) -> float:
    """Minimum detectable absolute effect ``p_b - p_a`` for a given sample size.

    Searches upward from ``p_a`` (downward for ``alternative="smaller"``, which returns a
    negative number). Relative MDE is ``mde / p_a``.
    """
    if not 0 < p_a < 1:
        raise ValueError("p_a must be in (0, 1)")
    sign = -1.0 if alternative == "smaller" else 1.0
    edge = 1e-12 if sign < 0 else 1 - 1e-12

    def gap(p_b: float) -> float:
        return power_two_proportions(p_a, p_b, n_a, ratio, alpha, alternative, method) - power

    if gap(edge) < 0:
        raise ValueError("no effect size reaches the target power with this sample size")
    lo, hi = (edge, p_a - 1e-12) if sign < 0 else (p_a + 1e-12, edge)
    p_b = optimize.brentq(gap, lo, hi, xtol=1e-12)
    return float(p_b - p_a)


def mde_two_means(
    sd: float,
    n_a: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
    method: MeansMethod = "t",
) -> float:
    """Minimum detectable absolute difference in means for a given sample size."""
    sign = -1.0 if alternative == "smaller" else 1.0

    def gap(d: float) -> float:
        return power_two_means(sign * d, sd, n_a, ratio, alpha, alternative, method) - power

    hi = sd
    while gap(hi) < 0:
        hi *= 2
    return float(sign * optimize.brentq(gap, 0.0, hi, xtol=1e-12 * sd))

