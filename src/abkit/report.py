"""End-to-end analysis of one experiment table and its markdown report."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from abkit.bayesian import (
    BayesResult,
    beta_binomial_ab,
    normal_expected_loss,
    normal_prob_b_better,
)
from abkit.cuped import CupedResult, cuped
from abkit.frequentist import LiftResult, two_proportion_ztest, welch_ttest
from abkit.power import mde_two_means, mde_two_proportions
from abkit.srm import SampleRatioMismatchWarning, SRMResult, srm_check

CONTROL_NAMES = ("control", "ctrl", "a", "baseline")


@dataclass(frozen=True)
class GroupSummary:
    name: str
    n: int
    mean: float
    sd: float


@dataclass(frozen=True)
class Analysis:
    source: str
    group_col: str
    metric: str
    pre_metric: str | None
    metric_type: str
    alpha: float
    power: float
    n_dropped: int
    control: GroupSummary
    treatment: GroupSummary
    srm: SRMResult
    primary: LiftResult
    cuped: CupedResult | None
    bayes: BayesResult | None
    prob_b_better: float
    expected_loss_b: float
    expected_loss_a: float
    mde: float
    mde_cuped: float | None

    @property
    def headline(self) -> LiftResult:
        """CUPED-adjusted result when a pre-metric is given, else the plain test."""
        return self.cuped.adjusted if self.cuped is not None else self.primary


def _pick_groups(groups: list[str], control: str | None) -> tuple[str, str]:
    if len(groups) != 2:
        raise ValueError(f"expected exactly 2 groups, found {len(groups)}: {groups}")
    if control is None:
        named = [g for g in groups if g.lower() in CONTROL_NAMES]
        control = named[0] if named else sorted(groups)[0]
    if control not in groups:
        raise ValueError(f"control group {control!r} not found in {groups}")
    return control, next(g for g in groups if g != control)


def analyze(
    df: pd.DataFrame,
    group: str,
    metric: str,
    pre_metric: str | None = None,
    control: str | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    control_share: float = 0.5,
    metric_type: str = "auto",
    source: str = "<dataframe>",
    seed: int = 0,
) -> Analysis:
    """Run SRM check, frequentist test, optional CUPED, Bayesian view and an MDE check."""
    cols = [group, metric] + ([pre_metric] if pre_metric else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}; available: {list(df.columns)}")
    data = df[cols].dropna()
    n_dropped = len(df) - len(data)
    data = data.assign(**{group: data[group].astype(str)})
    ctrl, treat = _pick_groups(sorted(data[group].unique()), control)
    a = data.loc[data[group] == ctrl]
    b = data.loc[data[group] == treat]
    y_a, y_b = a[metric].to_numpy(float), b[metric].to_numpy(float)

    if metric_type == "auto":
        values = np.unique(np.concatenate([y_a, y_b]))
        metric_type = "binary" if np.isin(values, [0.0, 1.0]).all() else "continuous"
    if metric_type not in ("binary", "continuous"):
        raise ValueError("metric_type must be 'auto', 'binary' or 'continuous'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SampleRatioMismatchWarning)  # reported in the text
        srm = srm_check([len(a), len(b)], [control_share, 1 - control_share])

    n_a, n_b = len(a), len(b)
    if metric_type == "binary":
        x_a, x_b = int(y_a.sum()), int(y_b.sum())
        primary = two_proportion_ztest(x_a, n_a, x_b, n_b, alpha)
        mde = mde_two_proportions(y_a.mean(), n_a, power, alpha, ratio=n_b / n_a)
    else:
        primary = welch_ttest(y_a, y_b, alpha)
        mde = mde_two_means(np.sqrt((y_a.var(ddof=1) + y_b.var(ddof=1)) / 2), n_a, power,
                            alpha, ratio=n_b / n_a)

    cuped_res, mde_cuped = None, None
    if pre_metric:
        cuped_res = cuped(y_a, a[pre_metric].to_numpy(float), y_b, b[pre_metric].to_numpy(float),
                          alpha)
        mde_cuped = mde * float(np.sqrt(1 - cuped_res.variance_reduction))

    bayes = None
    if metric_type == "binary" and cuped_res is None:
        bayes = beta_binomial_ab(int(y_a.sum()), n_a, int(y_b.sum()), n_b, seed=seed)
        prob, loss_b, loss_a = bayes.prob_b_better, bayes.expected_loss_b, bayes.expected_loss_a
    else:
        head = cuped_res.adjusted if cuped_res is not None else primary
        prob = normal_prob_b_better(float(head.estimate), float(head.se))
        loss_b, loss_a = normal_expected_loss(float(head.estimate), float(head.se))

    def summary(name: str, y: np.ndarray) -> GroupSummary:
        return GroupSummary(name, len(y), float(y.mean()), float(y.std(ddof=1)))

    return Analysis(
        source=source,
        group_col=group,
        metric=metric,
        pre_metric=pre_metric,
        metric_type=metric_type,
        alpha=alpha,
        power=power,
        n_dropped=n_dropped,
        control=summary(ctrl, y_a),
        treatment=summary(treat, y_b),
        srm=srm,
        primary=primary,
        cuped=cuped_res,
        bayes=bayes,
        prob_b_better=prob,
        expected_loss_b=loss_b,
        expected_loss_a=loss_a,
        mde=mde,
        mde_cuped=mde_cuped,
    )


def _fmt_abs(x: float, binary: bool) -> str:
    return f"{x * 100:+.2f} pp" if binary else f"{x:+.4g}"


def _fmt_level(x: float, binary: bool) -> str:
    return f"{x:.2%}" if binary else f"{x:.4g}"


def _fmt_loss(x: float, binary: bool) -> str:
    return f"{x * 100:.3f} pp" if binary else f"{x:.4g}"


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _result_rows(r: LiftResult, binary: bool) -> list[str]:
    level = f"{1 - r.alpha:.0%}"
    return [
        f"| Absolute lift (B - A) | {_fmt_abs(float(r.estimate), binary)} "
        f"| {level} CI [{_fmt_abs(float(r.ci_low), binary)}, "
        f"{_fmt_abs(float(r.ci_high), binary)}] |",
        f"| Relative lift (B / A - 1) | {float(r.relative):+.2%} "
        f"| {level} CI [{float(r.rel_ci_low):+.2%}, {float(r.rel_ci_high):+.2%}] (delta method) |",
        f"| p-value ({r.method}) | {_fmt_p(float(r.p_value))} | alpha = {r.alpha:g} |",
    ]


def decision_guidance(a: Analysis) -> list[str]:
    """Plain-English recommendation that respects SRM, significance and power."""
    binary = a.metric_type == "binary"
    r = a.headline
    est, p = float(r.estimate), float(r.p_value)
    lines = []
    if a.srm.mismatch:
        lines.append(
            "**Do not make a decision yet.** The group sizes do not match the planned split "
            "(SRM). Typical causes: bot filtering applied to one arm, redirects that lose "
            "users, broken logging, or assignment bugs. Find the cause and re-run; lift "
            "estimates from a mismatched experiment are biased in unknown direction."
        )
    elif p < a.alpha and est > 0:
        lines.append(
            f"**Treatment wins.** Lift {_fmt_abs(est, binary)} ({float(r.relative):+.2%}), "
            f"p = {_fmt_p(p)}. P(B > A) = {a.prob_b_better:.1%}; expected loss if you ship B: "
            f"{_fmt_loss(a.expected_loss_b, binary)}."
        )
    elif p < a.alpha:
        lines.append(
            f"**Treatment loses.** Lift {_fmt_abs(est, binary)} ({float(r.relative):+.2%}), "
            f"p = {_fmt_p(p)}. Do not ship B."
        )
    else:
        mde = a.mde_cuped if a.mde_cuped is not None else a.mde
        lines.append(
            f"**Inconclusive.** The {1 - a.alpha:.0%} CI "
            f"[{_fmt_abs(float(r.ci_low), binary)}, {_fmt_abs(float(r.ci_high), binary)}] "
            f"includes zero. With these sample sizes the test had {a.power:.0%} power only for "
            f"effects of about {_fmt_abs(mde, binary)} or larger, so smaller effects are not "
            "ruled out. Absence of evidence is not evidence of no effect."
        )
    lines.append(
        "These p-values are fixed-horizon: valid only if the sample size was fixed in advance "
        "and the data analysed once. If results were checked repeatedly, use a sequential "
        "design (`abkit.sequential`)."
    )
    return lines


def render_markdown(a: Analysis) -> str:
    binary = a.metric_type == "binary"
    c, t = a.control, a.treatment
    out = [
        f"# A/B test report: `{a.metric}` by `{a.group_col}`",
        "",
        f"Source: `{a.source}` | control (A) = `{c.name}`, treatment (B) = `{t.name}` | "
        f"metric type: {a.metric_type} | alpha = {a.alpha:g}",
    ]
    if a.n_dropped:
        out.append(f"\nDropped {a.n_dropped:,} rows with missing values.")
    stat = "conversion rate" if binary else "mean"
    out += [
        "",
        "## 1. Sample ratio mismatch (SRM) check",
        "",
        "| group | users | share | expected share |",
        "|---|---:|---:|---:|",
    ]
    total = c.n + t.n
    for g, e in zip((c, t), a.srm.expected, strict=True):
        out.append(f"| {g.name} | {g.n:,} | {g.n / total:.2%} | {e / total:.2%} |")
    flag = "**WARNING**" if a.srm.mismatch else "OK"
    out += ["", f"{flag}: {a.srm.message()}", "", "## 2. Result (fixed-horizon test)", ""]
    out += [
        f"| group | n | {stat} | std |",
        "|---|---:|---:|---:|",
        f"| {c.name} (A) | {c.n:,} | {_fmt_level(c.mean, binary)} | {c.sd:.4g} |",
        f"| {t.name} (B) | {t.n:,} | {_fmt_level(t.mean, binary)} | {t.sd:.4g} |",
        "",
        "| quantity | estimate | interval |",
        "|---|---:|---|",
        *_result_rows(a.primary, binary),
    ]
    if a.cuped is not None:
        cu = a.cuped
        out += [
            "",
            f"## 3. CUPED adjustment (pre-experiment covariate `{a.pre_metric}`)",
            "",
            f"theta = {float(cu.theta):.4g}, corr(pre, metric) = {float(cu.correlation):.3f}, "
            f"variance of the lift estimate reduced by {float(cu.variance_reduction):.1%} "
            f"(equivalent to {1 / (1 - float(cu.variance_reduction)):.2f}x the sample size).",
            "",
            "| quantity | estimate | interval |",
            "|---|---:|---|",
            *_result_rows(cu.adjusted, binary),
        ]
    out += ["", f"## {4 if a.cuped is not None else 3}. Bayesian view", ""]
    if a.bayes is not None:
        bz = a.bayes
        out += [
            f"Beta-Binomial model with a uniform Beta(1, 1) prior; Monte Carlo with "
            f"{bz.n_draws:,} draws (seed {bz.seed}).",
            "",
            f"- P(B > A) = **{bz.prob_b_better:.1%}** (numerical integration; MC: "
            f"{bz.prob_b_better_mc:.1%})",
            f"- Expected loss if shipping B: {_fmt_loss(bz.expected_loss_b, True)}; if keeping "
            f"A: {_fmt_loss(bz.expected_loss_a, True)}",
            f"- {bz.level:.0%} credible interval of relative lift: "
            f"[{bz.rel_lift_interval[0]:+.2%}, {bz.rel_lift_interval[1]:+.2%}]",
        ]
    else:
        out += [
            "Normal approximation with a flat prior on the "
            f"{'CUPED-adjusted ' if a.cuped is not None else ''}difference in means.",
            "",
            f"- P(B > A) = **{a.prob_b_better:.1%}**",
            f"- Expected loss if shipping B: {_fmt_loss(a.expected_loss_b, binary)}; if keeping "
            f"A: {_fmt_loss(a.expected_loss_a, binary)} (per user)",
        ]
    mde_rel = a.mde / c.mean if c.mean else float("nan")
    out += [
        "",
        f"## {5 if a.cuped is not None else 4}. Power check",
        "",
        f"With n = {c.n:,} / {t.n:,}, the minimum detectable effect at {a.power:.0%} power is "
        f"{_fmt_abs(a.mde, binary)} ({mde_rel:+.1%} relative)"
        + (
            f"; with CUPED about {_fmt_abs(a.mde_cuped, binary)} "
            f"({a.mde_cuped / c.mean:+.1%} relative)."
            if a.mde_cuped is not None and c.mean
            else "."
        ),
        "",
        "## Decision guidance",
        "",
        *[f"- {line}" for line in decision_guidance(a)],
        "",
    ]
    return "\n".join(out)
