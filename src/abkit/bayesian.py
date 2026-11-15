"""Bayesian A/B analysis: Beta-Binomial for conversions, normal approximation for means.

``P(B > A)`` is computed two ways: by one-dimensional numerical integration (accurate to
~1e-10, used as the reference) and by Monte Carlo with a fixed seed (needed anyway for
expected loss and the credible interval of the relative lift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        lo, hi = stats.beta.interval(level, self.alpha, self.beta)
        return float(lo), float(hi)


@dataclass(frozen=True)
class BayesResult:
    posterior_a: BetaPosterior
    posterior_b: BetaPosterior
    prob_b_better: float  # numerical integration
    prob_b_better_mc: float  # Monte Carlo estimate (same draws as the quantities below)
    expected_loss_b: float  # E[max(p_A - p_B, 0)]: expected regret if we ship B
    expected_loss_a: float  # E[max(p_B - p_A, 0)]: expected regret if we keep A
    abs_lift_interval: tuple[float, float]
    rel_lift_mean: float
    rel_lift_interval: tuple[float, float]
    level: float
    n_draws: int
    seed: int


def beta_posterior(
    successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> BetaPosterior:
    """Conjugate update of a Beta(prior_alpha, prior_beta) prior with binomial data."""
    if not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials")
    return BetaPosterior(prior_alpha + successes, prior_beta + trials - successes)


def prob_b_beats_a(post_a: BetaPosterior, post_b: BetaPosterior) -> float:
    """P(p_B > p_A) = integral of f_B(x) F_A(x) dx, by adaptive quadrature."""
    lo = stats.beta.ppf(1e-14, post_b.alpha, post_b.beta)
    hi = stats.beta.isf(1e-14, post_b.alpha, post_b.beta)
    points = None
    if post_b.alpha > 1 and post_b.beta > 1:  # help quad find the peak of a narrow posterior
        points = [(post_b.alpha - 1) / (post_b.alpha + post_b.beta - 2)]
    value, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, post_b.alpha, post_b.beta)
        * stats.beta.cdf(x, post_a.alpha, post_a.beta),
        lo,
        hi,
        points=points,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=200,
    )
    return float(min(max(value, 0.0), 1.0))


def beta_binomial_ab(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
    n_draws: int = 200_000,
    seed: int = 0,
) -> BayesResult:
    """Full Beta-Binomial comparison of conversion rates with a shared Beta prior."""
    post_a = beta_posterior(x_a, n_a, prior_alpha, prior_beta)
    post_b = beta_posterior(x_b, n_b, prior_alpha, prior_beta)
    rng = np.random.default_rng(seed)
    pa = rng.beta(post_a.alpha, post_a.beta, n_draws)
    pb = rng.beta(post_b.alpha, post_b.beta, n_draws)
    diff = pb - pa
    rel = pb / pa - 1
    q = [(1 - level) / 2, (1 + level) / 2]
    abs_lo, abs_hi = np.quantile(diff, q)
    rel_lo, rel_hi = np.quantile(rel, q)
    return BayesResult(
        posterior_a=post_a,
        posterior_b=post_b,
        prob_b_better=prob_b_beats_a(post_a, post_b),
        prob_b_better_mc=float(np.mean(diff > 0)),
        expected_loss_b=float(np.mean(np.maximum(-diff, 0))),
        expected_loss_a=float(np.mean(np.maximum(diff, 0))),
        abs_lift_interval=(float(abs_lo), float(abs_hi)),
        rel_lift_mean=float(rel.mean()),
        rel_lift_interval=(float(rel_lo), float(rel_hi)),
        level=level,
        n_draws=n_draws,
        seed=seed,
    )


def normal_prob_b_better(estimate: float, se: float) -> float:
    """P(B > A) under a flat prior and normal likelihood for the difference: Phi(est / se)."""
    return float(stats.norm.cdf(estimate / se))


def normal_expected_loss(estimate: float, se: float) -> tuple[float, float]:
    """Expected loss of shipping B and of keeping A when D ~ N(estimate, se^2).

    Closed form: E[max(-D, 0)] = se * phi(m/se) - m * Phi(-m/se), and symmetrically.
    """
    z = estimate / se
    loss_b = se * stats.norm.pdf(z) - estimate * stats.norm.cdf(-z)
    loss_a = se * stats.norm.pdf(z) + estimate * stats.norm.cdf(z)
    return float(loss_b), float(loss_a)
