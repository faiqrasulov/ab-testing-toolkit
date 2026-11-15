"""abkit: A/B-test analysis whose statistical guarantees are verified by simulation."""

from abkit.bayesian import beta_binomial_ab, beta_posterior, prob_b_beats_a
from abkit.cuped import cuped
from abkit.frequentist import LiftResult, two_proportion_ztest, welch_ttest
from abkit.power import (
    mde_two_means,
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)
from abkit.sequential import (
    always_valid_pvalues,
    crossing_probabilities,
    group_sequential_boundaries,
)
from abkit.srm import SampleRatioMismatchWarning, srm_check

__version__ = "0.1.0"

__all__ = [
    "LiftResult",
    "SampleRatioMismatchWarning",
    "always_valid_pvalues",
    "beta_binomial_ab",
    "beta_posterior",
    "crossing_probabilities",
    "cuped",
    "group_sequential_boundaries",
    "mde_two_means",
    "mde_two_proportions",
    "power_two_means",
    "power_two_proportions",
    "prob_b_beats_a",
    "sample_size_two_means",
    "sample_size_two_proportions",
    "srm_check",
    "two_proportion_ztest",
    "welch_ttest",
]
