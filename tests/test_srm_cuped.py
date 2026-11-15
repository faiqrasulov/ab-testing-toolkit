import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from abkit.cuped import cuped, cuped_theta
from abkit.srm import SampleRatioMismatchWarning, srm_check


def test_srm_matches_scipy_chisquare():
    res = srm_check([5100, 4900], warn=False)
    ref = stats.chisquare([5100, 4900])
    assert res.chi2 == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    assert not res.mismatch


def test_srm_with_unequal_planned_split():
    res = srm_check([2000, 8000], expected_shares=[0.2, 0.8], warn=False)
    assert res.chi2 == pytest.approx(0.0)
    ref = stats.chisquare([2100, 7900], f_exp=[2000, 8000])
    assert srm_check([2100, 7900], [0.2, 0.8], warn=False).p_value == pytest.approx(ref.pvalue)


def test_srm_warns_on_mismatch_only():
    with pytest.warns(SampleRatioMismatchWarning, match="SAMPLE RATIO MISMATCH"):
        res = srm_check([5300, 4700])
    assert res.mismatch and res.p_value < 0.001
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "No evidence of SRM" in srm_check([5010, 4990]).message()


def test_srm_rejects_bad_input():
    with pytest.raises(ValueError):
        srm_check([100])
    with pytest.raises(ValueError):
        srm_check([100, 100], expected_shares=[1.0])


def _experiment(rng, n, rho, effect):
    x = rng.standard_normal(2 * n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(2 * n)
    y[n:] += effect
    return y[:n], x[:n], y[n:], x[n:]


def test_theta_is_ols_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.5 * x + rng.normal(size=500)
    theta, rho = cuped_theta(y, x)
    assert theta == pytest.approx(np.polyfit(x, y, 1)[0])
    assert rho == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_cuped_matches_ancova_regression():
    rng = np.random.default_rng(1)
    y_a, x_a, y_b, x_b = _experiment(rng, 5000, 0.7, 0.05)
    res = cuped(y_a, x_a, y_b, x_b)
    treat = np.r_[np.zeros(5000), np.ones(5000)]
    design = sm.add_constant(np.column_stack([treat, np.r_[x_a, x_b]]))
    ols = sm.OLS(np.r_[y_a, y_b], design).fit()
    # Pooled-theta CUPED and ANCOVA agree up to O(1/n) differences in the slope.
    assert float(res.adjusted.estimate) == pytest.approx(ols.params[1], abs=2e-3)
    assert float(res.adjusted.se) == pytest.approx(ols.bse[1], rel=0.01)


def test_cuped_variance_reduction_close_to_rho_squared():
    rng = np.random.default_rng(2)
    res = cuped(*_experiment(rng, 20_000, 0.6, 0.0))
    assert float(res.variance_reduction) == pytest.approx(0.36, abs=0.02)
    res0 = cuped(*_experiment(rng, 20_000, 0.0, 0.0))
    assert abs(float(res0.variance_reduction)) < 0.01


def test_cuped_batched_shapes():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(4, 200))
    x = rng.normal(size=(4, 200))
    res = cuped(y[:, :100], x[:, :100], y[:, 100:], x[:, 100:])
    assert np.shape(res.adjusted.estimate) == (4,)
    assert np.shape(res.theta) == (4,)
