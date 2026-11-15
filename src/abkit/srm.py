"""Sample Ratio Mismatch (SRM) check: chi-square goodness-of-fit on assignment counts."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


class SampleRatioMismatchWarning(UserWarning):
    """Raised when observed group sizes are inconsistent with the planned split."""


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    threshold: float
    mismatch: bool

    def message(self) -> str:
        shares = ", ".join(f"{o / sum(self.observed):.2%}" for o in self.observed)
        if self.mismatch:
            return (
                f"SAMPLE RATIO MISMATCH: observed shares [{shares}] are inconsistent with the "
                f"planned split (chi2 = {self.chi2:.2f}, p = {self.p_value:.2g} < "
                f"{self.threshold:g}). Assignment or logging is broken; do not trust the lift "
                "estimates until the cause is found."
            )
        return (
            f"No evidence of SRM: observed shares [{shares}] (chi2 = {self.chi2:.2f}, "
            f"p = {self.p_value:.3f} >= {self.threshold:g})."
        )


def srm_check(
    counts: Sequence[int],
    expected_shares: Sequence[float] | None = None,
    threshold: float = 0.001,
    warn: bool = True,
) -> SRMResult:
    """Chi-square goodness-of-fit test of group sizes against the planned allocation.

    The conventional SRM threshold is p < 0.001: with many experiments a stricter cut-off
    avoids false alarms, and genuine SRMs (bot filtering, redirects, logging loss) usually
    produce far smaller p-values.
    """
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2 or np.any(obs < 0):
        raise ValueError("counts must be a 1-D sequence of at least two non-negative integers")
    shares = (
        np.full(obs.size, 1 / obs.size)
        if expected_shares is None
        else np.asarray(expected_shares, dtype=float)
    )
    if shares.shape != obs.shape or np.any(shares <= 0):
        raise ValueError("expected_shares must be positive and match counts in length")
    exp = shares / shares.sum() * obs.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=obs.size - 1))
    result = SRMResult(
        observed=tuple(int(c) for c in obs),
        expected=tuple(float(e) for e in exp),
        chi2=chi2,
        p_value=p,
        threshold=threshold,
        mismatch=p < threshold,
    )
    if warn and result.mismatch:
        warnings.warn(result.message(), SampleRatioMismatchWarning, stacklevel=2)
    return result
