"""Figures for the simulation studies (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from abkit.sequential import msprt_z_boundary  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2,
    "lines.markersize": 6.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _percent_axis(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_peeking(m: dict, path: Path) -> None:
    rows = m["null"]
    looks = np.array([r["looks"] for r in rows])
    design = m["design"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax.plot(looks, [r["naive_exact"] for r in rows], color=ORANGE, alpha=0.45, lw=2)
    ax.plot(looks, [r["naive"] for r in rows], "o", color=ORANGE,
            label="Naive peeking: z-test at every look")
    ax.plot(looks, [r["group_sequential_obf"] for r in rows], "s-", color=BLUE,
            label="Group sequential (Lan-DeMets OBF)")
    ax.plot(looks, [r["msprt"] for r in rows], "^-", color=AQUA,
            label="mSPRT always-valid p-value")
    ax.axhline(design["alpha"], color=INK2, ls="--", lw=1.2)
    ax.text(looks[-1], design["alpha"] - 0.013, "nominal 5%", color=INK2, fontsize=9, ha="right")
    last = rows[-1]
    ax.annotate(f"{last['naive']:.1%}", (looks[-1], last["naive"]), xytext=(-8, 8),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(looks, [str(k) for k in looks])
    ax.minorticks_off()
    ax.set_ylim(0, max(r["naive"] for r in rows) * 1.12)
    _percent_axis(ax)
    ax.set_xlabel("Number of interim looks (equally spaced)")
    ax.set_ylabel("False-positive rate in A/A tests")
    ax.set_title("Peeking inflates false positives; sequential methods do not")
    ax.legend(loc="upper left")
    ax.text(0.02, 0.66, f"{design['reps_per_look_setting']:,} simulated A/A tests per point\n"
            "faint line = exact naive rate (numerical integration)", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="top")

    bounds = np.array(design["obf_10_boundaries"], dtype=float)
    t10 = np.arange(1, bounds.size + 1) / bounds.size
    n_max = design["n_per_arm_max"]
    p0 = design["baseline"]
    t = np.linspace(0.02, 1, 300)
    z_msprt = msprt_z_boundary(2 * p0 * (1 - p0) / (t * n_max), design["msprt_tau"],
                               design["alpha"])
    ax2.plot(t, z_msprt, color=AQUA, label="mSPRT (any time, continuous)")
    ax2.plot(t10, bounds, "s-", color=BLUE, label="Lan-DeMets OBF (10 looks)")
    ax2.plot(t10, np.full(t10.size, 1.96), "o-", color=ORANGE, label="Naive: 1.96 every look")
    ax2.set_ylim(0, 8)
    ax2.set_xlim(0, 1.03)
    _percent_axis(ax2, "x")
    ax2.set_xlabel("Share of the planned sample collected")
    ax2.set_ylabel("|z| needed to stop and declare a winner")
    ax2.set_title("Why: valid methods demand more evidence early")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def fig_cuped(m: dict, path: Path) -> None:
    rows = m["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rho = np.linspace(0, 1, 200)
    ax.plot(rho, 1 - rho**2, "--", color=INK2, lw=1.5, label=r"Theory: $1-\rho^2$")
    for dist, marker, color, label in (
        ("normal", "o", BLUE, "Simulated, normal metric"),
        ("lognormal", "D", ORANGE, "Simulated, skewed lognormal metric"),
    ):
        sel = [r for r in rows if r["distribution"] == dist]
        ax.errorbar([r["rho"] for r in sel], [r["variance_ratio"] for r in sel],
                    yerr=[1.96 * r["variance_ratio_se"] for r in sel], fmt=marker,
                    color=color, ms=7, elinewidth=1.2, capsize=0, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1)
    hi = min((r for r in rows if r["distribution"] == "normal"), key=lambda r: abs(r["rho"] - 0.9))
    ax.annotate(
        f"$\\rho$ = {hi['rho']:.1f}: variance x {hi['variance_ratio']:.2f}\n"
        f"= {hi['effective_sample_size_multiplier']:.1f}x effective sample size",
        (hi["rho"], hi["variance_ratio"]), xytext=(-150, -6), textcoords="offset points",
        fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"Correlation $\rho$ between pre-period covariate and metric")
    ax.set_ylabel("Var(CUPED estimate) / Var(plain estimate)")
    ax.set_title("CUPED variance reduction matches theory")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.97, f"{m['reps']:,} simulated experiments per point, n = "
            f"{m['n_per_arm']:,} per arm\n95% Monte Carlo intervals are about marker size",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, path)


def fig_power(m: dict, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    b, mm = m["binary"], m["means"]
    for axis, rows, xkey, xscale, xlabel, title in (
        (ax, b["rows"], "relative_lift", 100.0, "True relative lift (%) on a 10% baseline",
         f"Conversion rate, z-test, n = {b['n_per_arm']:,} per arm"),
        (ax2, mm["rows"], "cohens_d", 1.0, "True effect size (Cohen's d)",
         f"Continuous metric, Welch t-test, n = {mm['n_per_arm']:,} per arm"),
    ):
        x = np.array([r[xkey] for r in rows]) * xscale
        axis.plot(x, [r["analytic"] for r in rows], color=BLUE, label="Analytic power")
        axis.plot(x, [r["power"] for r in rows], "o", color=ORANGE,
                  markeredgecolor=SURFACE, label=f"Simulated ({m['reps']:,} tests/point)")
        axis.axhline(0.8, color=INK2, ls=":", lw=1.2)
        axis.text(x[0], 0.82, "80% target", color=INK2, fontsize=9)
        axis.set_ylim(0, 1.03)
        _percent_axis(axis)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.legend(loc="lower right")
    ax.set_ylabel("Power (share of tests with p < 0.05)")
    fig.tight_layout()
    _save(fig, path)


def fig_coverage(m: dict, path: Path) -> None:
    rows = m["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(rows))[::-1]
    nominal = m["nominal"]
    for key, marker, color, off, label in (
        ("absolute", "o", BLUE, 0.2, "Absolute lift (Wald / Welch)"),
        ("relative_delta", "s", ORANGE, 0.0, "Relative lift, delta method"),
        ("relative_log", "^", AQUA, -0.2, "Relative lift, log-ratio delta method"),
    ):
        ax.plot([r[key] for r in rows], y + off, marker, color=color, ms=7, ls="none",
                label=label, markeredgecolor=SURFACE)
    ax.axvline(nominal, color=INK2, ls="--", lw=1.2)
    ax.set_yticks(y, [f"{r['label']}" for r in rows])
    lo = min(min(r[k] for k in ("absolute", "relative_delta", "relative_log")) for r in rows)
    x0 = min(0.9, np.floor(lo * 50) / 50 - 0.01)
    ax.set_xlim(x0, 1.0)
    for yi, r in zip(y, rows, strict=True):
        if r["relative_undefined"] > 0.001:
            ax.text(x0 + 0.035, yi - 0.12, f"relative CI undefined in {r['relative_undefined']:.1%}"
                    " of runs\n(zero control conversions)", fontsize=8, color=MUTED, va="top")
    reps = sorted({r["reps"] for r in rows}, reverse=True)
    ax.text(0.01, 0.99, "simulated experiments per row: " + " / ".join(f"{n:,}" for n in reps),
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    _percent_axis(ax, "x")
    ax.set_xlabel("Empirical coverage of the nominal 95% confidence interval")
    ax.set_title("CI coverage: reliable at realistic sizes, not with tiny counts or heavy skew")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def make_figures(metrics: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_peeking(metrics["peeking"], out / "peeking_false_positives.png")
        fig_cuped(metrics["cuped"], out / "cuped_variance_reduction.png")
        fig_power(metrics["power"], out / "power_curves.png")
        fig_coverage(metrics["coverage"], out / "ci_coverage.png")
