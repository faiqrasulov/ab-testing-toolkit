import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from abkit.frequentist import relative_lift_ci, two_proportion_ztest, welch_ttest


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_ztest_matches_statsmodels(alternative):
    res = two_proportion_ztest(1000, 10_000, 1100, 10_000, alternative=alternative)
    z, p = proportions_ztest([1100, 1000], [10_000, 10_000], alternative=alternative)
    assert float(res.statistic) == pytest.approx(z, rel=1e-12)
    assert float(res.p_value) == pytest.approx(p, rel=1e-10)


def test_absolute_ci_matches_statsmodels_wald():
    res = two_proportion_ztest(230, 2000, 270, 2100)
    lo, hi = confint_proportions_2indep(270, 2100, 230, 2000, method="wald", compare="diff")
    assert (float(res.ci_low), float(res.ci_high)) == pytest.approx((lo, hi), rel=1e-12)


def test_log_ratio_ci_matches_statsmodels():
    res = two_proportion_ztest(230, 2000, 270, 2100, relative_method="log")
    lo, hi = confint_proportions_2indep(270, 2100, 230, 2000, method="log", compare="ratio")
    assert float(res.rel_ci_low) == pytest.approx(lo - 1, rel=1e-9)
    assert float(res.rel_ci_high) == pytest.approx(hi - 1, rel=1e-9)


def test_delta_method_relative_ci_known_answer():
    # B/A = 1.1; Var(ratio) = 0.04^2/10^2 + 11^2 * 0.03^2 / 10^4 = 2.689e-5
    r, lo, hi = relative_lift_ci(10.0, 0.03**2, 11.0, 0.04**2, alpha=0.05)
    se = np.sqrt(0.04**2 / 100 + 121 * 0.03**2 / 10_000)
    z = stats.norm.isf(0.025)
    assert r == pytest.approx(0.1)
    assert (lo, hi) == pytest.approx((0.1 - z * se, 0.1 + z * se))


def test_welch_matches_scipy_and_statsmodels():
    rng = np.random.default_rng(0)
    a = rng.lognormal(0, 1, 300)
    b = rng.lognormal(0.1, 1.2, 250)
    res = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert float(res.statistic) == pytest.approx(ref.statistic, rel=1e-12)
    assert float(res.p_value) == pytest.approx(ref.pvalue, rel=1e-10)
    lo, hi = CompareMeans(DescrStatsW(b), DescrStatsW(a)).tconfint_diff(usevar="unequal")
    assert (float(res.ci_low), float(res.ci_high)) == pytest.approx((lo, hi), rel=1e-10)


def test_one_sided_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 40), rng.normal(0.3, 2, 60)
    res = welch_ttest(a, b, alternative="larger")
    ref = stats.ttest_ind(b, a, equal_var=False, alternative="greater")
    assert float(res.p_value) == pytest.approx(ref.pvalue, rel=1e-10)


def test_batched_welch_equals_row_by_row():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(5, 50)), rng.normal(0.2, 1, size=(5, 60))
    batch = welch_ttest(a, b)
    for i in range(5):
        single = welch_ttest(a[i], b[i])
        assert batch.p_value[i] == pytest.approx(float(single.p_value))
        assert batch.rel_ci_high[i] == pytest.approx(float(single.rel_ci_high))


def test_zero_control_rate_gives_nan_relative_ci():
    res = two_proportion_ztest(0, 100, 3, 100)
    assert np.isnan(res.rel_ci_low) or np.isinf(res.rel_ci_low)
    assert np.isfinite(res.p_value)
