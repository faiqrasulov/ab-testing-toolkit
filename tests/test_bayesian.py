import numpy as np
import pytest
from scipy.special import betaln

from abkit.bayesian import (
    beta_binomial_ab,
    beta_posterior,
    normal_expected_loss,
    prob_b_beats_a,
)


def exact_prob_b_beats_a(a_a, b_a, a_b, b_b):
    """Closed form for integer alpha_B (Evan Miller, 'Formulas for Bayesian A/B testing')."""
    i = np.arange(int(a_b))
    terms = (
        betaln(a_a + i, b_a + b_b) - np.log(b_b + i) - betaln(1 + i, b_b) - betaln(a_a, b_a)
    )
    return float(np.exp(terms).sum())


@pytest.mark.parametrize(
    "x_a,n_a,x_b,n_b", [(3, 20, 7, 20), (100, 1000, 120, 1000), (40, 500, 38, 480), (0, 10, 0, 10)]
)
def test_quadrature_matches_closed_form(x_a, n_a, x_b, n_b):
    post_a, post_b = beta_posterior(x_a, n_a), beta_posterior(x_b, n_b)
    exact = exact_prob_b_beats_a(post_a.alpha, post_a.beta, post_b.alpha, post_b.beta)
    assert prob_b_beats_a(post_a, post_b) == pytest.approx(exact, abs=1e-8)


def test_monte_carlo_agrees_with_exact_within_mc_error():
    res = beta_binomial_ab(1000, 10_000, 1060, 10_000, n_draws=200_000, seed=7)
    se = np.sqrt(res.prob_b_better * (1 - res.prob_b_better) / res.n_draws)
    assert abs(res.prob_b_better_mc - res.prob_b_better) < 4 * se


def test_identical_data_is_a_coin_flip():
    res = beta_binomial_ab(50, 500, 50, 500, seed=1)
    assert res.prob_b_better == pytest.approx(0.5, abs=1e-9)
    assert res.expected_loss_a == pytest.approx(res.expected_loss_b, rel=0.05)


def test_posterior_known_answer_and_reproducibility():
    post = beta_posterior(30, 100, prior_alpha=2, prior_beta=3)
    assert (post.alpha, post.beta) == (32, 73)
    assert post.mean == pytest.approx(32 / 105)
    a = beta_binomial_ab(30, 100, 40, 100, seed=3)
    b = beta_binomial_ab(30, 100, 40, 100, seed=3)
    assert a == b


def test_normal_expected_loss_matches_monte_carlo():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.2, 2_000_000)
    loss_b, loss_a = normal_expected_loss(0.3, 1.2)
    assert loss_b == pytest.approx(np.maximum(-d, 0).mean(), rel=5e-3)
    assert loss_a == pytest.approx(np.maximum(d, 0).mean(), rel=5e-3)
    assert loss_a - loss_b == pytest.approx(0.3)
