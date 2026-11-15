import numpy as np
import pytest
from scipy import integrate, stats

from abkit.sequential import (
    always_valid_pvalues,
    crossing_probabilities,
    expected_sample_fraction,
    group_sequential_boundaries,
    msprt_log_likelihood_ratio,
    msprt_z_boundary,
    obrien_fleming_spending,
    pocock_spending,
)


def test_obf_boundaries_match_published_table():
    # Lan-DeMets O'Brien-Fleming, 5 equally spaced looks, two-sided alpha = 0.05
    # (Jennison & Turnbull 2000; gsDesign sfLDOF): 4.877, 3.357, 2.680, 2.290, 2.031.
    design = group_sequential_boundaries(5, alpha=0.05)
    assert design.boundaries == pytest.approx([4.8769, 3.3569, 2.6803, 2.2898, 2.0310], abs=1e-3)


def test_single_look_is_the_fixed_test():
    assert group_sequential_boundaries(1).boundaries[0] == pytest.approx(stats.norm.isf(0.025))


@pytest.mark.parametrize("spending", ["obf", "pocock"])
def test_cumulative_crossing_equals_alpha_spent(spending):
    t = np.array([0.15, 0.4, 0.55, 0.8, 1.0])
    design = group_sequential_boundaries(info_fractions=t, spending=spending)
    cum = np.cumsum(crossing_probabilities(design.boundaries, t))
    fn = obrien_fleming_spending if spending == "obf" else pocock_spending
    assert cum == pytest.approx(2 * fn(t, 0.025), abs=1e-7)
    assert cum[-1] == pytest.approx(0.05, abs=1e-8)


def test_naive_repeated_testing_matches_armitage():
    # Armitage, McPherson & Rowe (1969): P(any |Z_k| > 1.96) over K equally spaced looks.
    published = {2: 0.083, 3: 0.107, 5: 0.142, 10: 0.193, 20: 0.248}
    for k, p in published.items():
        t = np.arange(1, k + 1) / k
        assert crossing_probabilities(np.full(k, 1.959964), t).sum() == pytest.approx(p, abs=1e-3)


def test_power_and_sample_size_match_brownian_simulation():
    design = group_sequential_boundaries(4)
    t = design.info_fractions
    drift = 2.8
    rng = np.random.default_rng(0)
    inc = rng.normal(drift * np.diff(np.r_[0, t]), np.sqrt(np.diff(np.r_[0, t])), (200_000, 4))
    z = inc.cumsum(axis=1) / np.sqrt(t)
    hit = np.abs(z) >= design.boundaries
    stopped = hit.any(axis=1)
    first = np.where(stopped, hit.argmax(axis=1), 3)
    assert crossing_probabilities(design.boundaries, t, drift).sum() == pytest.approx(
        stopped.mean(), abs=0.004
    )
    assert expected_sample_fraction(design.boundaries, t, drift) == pytest.approx(
        t[first].mean(), abs=0.004
    )


def test_msprt_likelihood_ratio_matches_numerical_mixture():
    est, var, tau = 0.012, 2e-5, 0.01

    def integrand(theta):
        lr = stats.norm.pdf(est, theta, np.sqrt(var)) / stats.norm.pdf(est, 0, np.sqrt(var))
        return lr * stats.norm.pdf(theta, 0, tau)

    ref, _ = integrate.quad(integrand, -0.2, 0.2, points=[0, est], epsabs=1e-12)
    assert np.exp(msprt_log_likelihood_ratio(est, var, tau)) == pytest.approx(ref, rel=1e-8)


def test_z_boundary_is_where_always_valid_p_hits_alpha():
    var, tau = np.array([1e-4, 1e-5, 1e-6]), 0.01
    z = msprt_z_boundary(var, tau, alpha=0.05)
    p = np.exp(-msprt_log_likelihood_ratio(z * np.sqrt(var), var, tau))
    assert p == pytest.approx(0.05, rel=1e-10)


def test_always_valid_pvalues_are_monotone_and_valid_under_continuous_monitoring():
    rng = np.random.default_rng(1)
    reps, looks = 4000, 300
    n = np.arange(1, looks + 1) * 20.0
    diff = rng.normal(0, np.sqrt(2 / 20), (reps, looks)).cumsum(axis=1) / (n / 20)
    p = always_valid_pvalues(diff, 2 / n, tau=0.1)
    assert np.all(np.diff(p, axis=1) <= 0)
    fpr = (p[:, -1] <= 0.05).mean()
    assert fpr <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / reps)


def test_zero_variance_gives_no_evidence():
    assert always_valid_pvalues(np.array([0.0, 0.1]), np.array([0.0, 0.0]), 0.1).tolist() == [1, 1]
