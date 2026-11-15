import math

import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import power_proportions_2indep, proportion_effectsize

from abkit.power import (
    mde_two_means,
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)


@pytest.mark.parametrize("p_a,p_b,n_a,ratio", [(0.10, 0.11, 14_000, 1.0), (0.05, 0.04, 3_000, 2.0),
                                               (0.30, 0.36, 500, 0.5)])
@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_pooled_power_matches_statsmodels(p_a, p_b, n_a, ratio, alternative):
    # statsmodels tests sample 1 vs sample 2: map treatment B -> sample 1, control A -> sample 2.
    n_b = ratio * n_a
    ref = power_proportions_2indep(p_b - p_a, p_a, n_b, ratio=n_a / n_b,
                                   alternative=alternative, return_results=False)
    ours = power_two_proportions(p_a, p_b, n_a, ratio, alternative=alternative)
    assert ours == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize("p_a,p_b,ratio", [(0.10, 0.11, 1.0), (0.02, 0.025, 1.0), (0.5, 0.45, 3.0)])
def test_arcsine_sample_size_matches_statsmodels(p_a, p_b, ratio):
    h = proportion_effectsize(p_b, p_a)
    ref = NormalIndPower().solve_power(abs(h), power=0.8, alpha=0.05, ratio=ratio)
    assert sample_size_two_proportions(p_a, p_b, ratio=ratio, method="arcsine") == math.ceil(ref)


def test_pooled_sample_size_matches_textbook_formula():
    # Fleiss (no continuity correction): ignoring the far rejection tail changes n by < 1.
    p1, p2 = 0.10, 0.12
    za, zb = stats.norm.isf(0.025), stats.norm.isf(0.2)
    p_bar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * p_bar * (1 - p_bar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    n = n**2 / (p2 - p1) ** 2
    assert sample_size_two_proportions(p1, p2) == math.ceil(n)


@pytest.mark.parametrize("d,n,ratio", [(0.2, 200, 1.0), (0.5, 30, 1.0), (0.3, 80, 2.5)])
@pytest.mark.parametrize("alternative", ["two-sided", "larger"])
def test_t_power_matches_statsmodels(d, n, ratio, alternative):
    ref = TTestIndPower().power(d, n, 0.05, ratio=ratio, alternative=alternative)
    # scipy.stats.nct vs statsmodels' scipy.special.nctdtr differ at the ~1e-9 level.
    assert power_two_means(d * 2.0, 2.0, n, ratio, alternative=alternative) == pytest.approx(
        ref, abs=1e-7
    )


def test_z_power_matches_statsmodels_normal():
    ref = NormalIndPower().power(0.25, 120, 0.05, ratio=1.5)
    assert power_two_means(0.25, 1.0, 120, 1.5, method="z") == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("d", [0.1, 0.2, 0.5])
def test_t_sample_size_matches_statsmodels(d):
    ref = TTestIndPower().solve_power(d, power=0.8, alpha=0.05)
    assert sample_size_two_means(d, 1.0) == math.ceil(ref)


def test_mde_means_matches_statsmodels_and_round_trips():
    ref = TTestIndPower().solve_power(None, nobs1=394, power=0.8, alpha=0.05)
    mde = mde_two_means(1.0, 394)
    assert mde == pytest.approx(ref, rel=1e-4)  # statsmodels' root finder is looser than ours
    assert power_two_means(mde, 1.0, 394) == pytest.approx(0.8, abs=1e-9)


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_mde_proportions_round_trip(alternative):
    mde = mde_two_proportions(0.10, 14_751, alternative=alternative)
    assert (mde < 0) == (alternative == "smaller")
    power = power_two_proportions(0.10, 0.10 + mde, 14_751, alternative=alternative)
    assert power == pytest.approx(0.8, abs=1e-9)


def test_sample_size_is_smallest_sufficient_integer():
    n = sample_size_two_proportions(0.10, 0.11)
    assert power_two_proportions(0.10, 0.11, n) >= 0.8 > power_two_proportions(0.10, 0.11, n - 1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.1, 0.1)
    with pytest.raises(ValueError):
        power_two_proportions(0.1, 0.2, 100, alternative="bigger")
    with pytest.raises(ValueError):
        mde_two_proportions(0.5, 3, power=0.999999)
