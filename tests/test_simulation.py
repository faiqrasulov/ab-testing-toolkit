import json

import numpy as np
import pytest

from abkit.power import power_two_proportions
from abkit.sequential import group_sequential_boundaries
from abkit.simulation import (
    binary_coverage,
    binary_paths,
    binary_power,
    cuped_trials,
    lognormal_correlation,
    sequential_outcomes,
    variance_ratio,
)
from abkit.studies import run_all

TOL = 3 * np.sqrt(0.05 * 0.95 / 6000)


def test_binary_paths_are_cumulative_and_bounded():
    rng = np.random.default_rng(0)
    n_cum = np.array([100, 250, 400])
    x_a, x_b = binary_paths(rng, 50, n_cum, 0.1, 0.2)
    assert x_a.shape == (50, 3)
    assert np.all(np.diff(x_a, axis=1) >= 0) and np.all(x_b <= n_cum)


def test_peeking_inflates_and_sequential_methods_control_type_one_error():
    rng = np.random.default_rng(1)
    n_cum = np.arange(1, 11) * 1000
    design = group_sequential_boundaries(10)
    x_a, x_b = binary_paths(rng, 6000, n_cum, 0.1, 0.1)
    out = sequential_outcomes(x_a, x_b, n_cum, 0.05, design, tau=0.02)
    assert abs(out["fixed"].rejected.mean() - 0.05) < TOL
    assert out["naive"].rejected.mean() > 0.15  # exact value for 10 looks: 0.193
    assert abs(out["group_sequential"].rejected.mean() - 0.05) < TOL
    assert out["msprt"].rejected.mean() <= 0.05 + TOL
    assert np.all(out["naive"].sample_fraction <= 1)


def test_cuped_trials_match_theory():
    rng = np.random.default_rng(2)
    res = cuped_trials(rng, 800, 300, 0.7, 0.1)
    ratio, se = variance_ratio(res["plain"], res["adjusted"])
    assert abs(ratio - (1 - 0.7**2)) < 4 * se


def test_lognormal_correlation_formula():
    rng = np.random.default_rng(3)
    u = rng.standard_normal(1_000_000)
    v = 0.6 * u + 0.8 * rng.standard_normal(1_000_000)
    emp = np.corrcoef(np.exp(u), np.exp(v))[0, 1]
    assert lognormal_correlation(0.6, 1.0) == pytest.approx(emp, abs=0.01)


def test_binary_coverage_and_power_helpers():
    rng = np.random.default_rng(4)
    cov = binary_coverage(rng, 6000, 5000, 0.1, 0.1)
    assert abs(cov["absolute"] - 0.95) < 0.015
    assert abs(cov["relative_delta"] - 0.95) < 0.015
    pw = binary_power(rng, 6000, 5000, 0.10, 0.115)
    analytic = power_two_proportions(0.10, 0.115, 5000)
    assert abs(pw["power"] - analytic) < 4 * np.sqrt(analytic * (1 - analytic) / 6000)
    assert pw["exaggeration"] > 1


def test_run_all_smoke(tmp_path):
    metrics = run_all(tmp_path, scale=0.002)
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert set(on_disk) >= {"peeking", "cuped", "coverage", "power"}
    assert metrics["peeking"]["null"][-1]["looks"] == 100
    figures = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert figures == [
        "ci_coverage.png",
        "cuped_variance_reduction.png",
        "peeking_false_positives.png",
        "power_curves.png",
    ]
