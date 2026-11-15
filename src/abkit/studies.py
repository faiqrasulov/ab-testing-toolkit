"""Simulation studies that verify abkit's statistical guarantees.

Run ``python -m abkit.studies --out reports`` to regenerate ``reports/metrics.json`` and
``reports/figures/*.png``. All data are simulated so the ground truth is known; every
study uses its own fixed seed, so results are reproducible bit for bit.

(a) peeking:  type-I error of fixed-horizon vs naive peeking vs valid sequential methods
(b) cuped:    variance of the CUPED estimate vs theory ``1 - rho^2``
(c) coverage: CI coverage for absolute and relative lift
(d) power:    empirical vs analytic power, and the exaggeration of significant estimates
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from abkit.power import (
    power_two_means,
    power_two_proportions,
    sample_size_two_proportions,
)
from abkit.sequential import (
    crossing_probabilities,
    expected_sample_fraction,
    group_sequential_boundaries,
)
from abkit.simulation import (
    binary_coverage,
    binary_paths,
    binary_power,
    cuped_trials,
    lognormal_correlation,
    lognormal_coverage,
    normal_means_power,
    sequential_outcomes,
    variance_ratio,
)

ALPHA = 0.05
BASELINE = 0.10  # control conversion rate used in the conversion studies
DESIGN_LIFT = 0.10  # relative lift the fixed-horizon test is powered for (80%)
LOOKS = (1, 2, 3, 5, 10, 20, 50, 100)
CONTINUOUS_LOOKS = 1000
SEEDS = {"peeking": 101, "cuped": 202, "coverage": 303, "power": 404}


def _reps(base: int, scale: float) -> int:
    return max(200, int(round(base * scale)))


def _chunks(total: int, size: int):
    for start in range(0, total, size):
        yield min(size, total - start)


def _mc_se(p: float, reps: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / reps))


def _look_sizes(n_max: int, n_looks: int) -> np.ndarray:
    return np.round(n_max * np.arange(1, n_looks + 1) / n_looks).astype(int)


def study_peeking(scale: float = 1.0, seed: int = SEEDS["peeking"]) -> dict:
    """(a) A/A false-positive rates under repeated looks, plus power/ESS under H1."""
    rng = np.random.default_rng(seed)
    p_b_design = BASELINE * (1 + DESIGN_LIFT)
    delta = p_b_design - BASELINE
    n_max = sample_size_two_proportions(BASELINE, p_b_design, power=0.8, alpha=ALPHA)
    tau = delta  # mSPRT mixing scale = the effect size the test is designed to detect
    crit = stats.norm.isf(ALPHA / 2)

    reps = _reps(100_000, scale)
    null_rows = []
    for k in LOOKS:
        n_cum = _look_sizes(n_max, k)
        design = group_sequential_boundaries(k, alpha=ALPHA)
        counts = {"fixed": 0, "naive": 0, "group_sequential": 0, "msprt": 0}
        for size in _chunks(reps, 10_000):
            x_a, x_b = binary_paths(rng, size, n_cum, BASELINE, BASELINE)
            out = sequential_outcomes(x_a, x_b, n_cum, ALPHA, design, tau)
            for m in counts:
                counts[m] += int(out[m].rejected.sum())
        rates = {m: c / reps for m, c in counts.items()}
        t = n_cum / n_max
        null_rows.append(
            {
                "looks": k,
                "naive": rates["naive"],
                "naive_exact": float(crossing_probabilities(np.full(k, crit), t).sum()),
                "group_sequential_obf": rates["group_sequential"],
                "msprt": rates["msprt"],
                "fixed_horizon": rates["fixed"],
                "mc_se_at_5pct": _mc_se(ALPHA, reps),
                "final_obf_boundary": float(design.boundaries[-1]),
            }
        )

    # Continuous monitoring (1,000 looks) under H0 and H1 on the same kind of paths.
    n_cum = _look_sizes(n_max, CONTINUOUS_LOOKS)
    step = CONTINUOUS_LOOKS // 10
    sub = np.arange(step - 1, CONTINUOUS_LOOKS, step)  # the 10 equally spaced looks
    design10 = group_sequential_boundaries(10, alpha=ALPHA)
    reps_cont = _reps(50_000, scale)
    agg: dict[str, dict[str, list]] = {}
    for label, p_b in (("null", BASELINE), ("alternative", p_b_design)):
        acc = {m: [0, 0.0] for m in ("fixed", "naive_10", "group_sequential_10", "msprt")}
        for size in _chunks(reps_cont, 2_000):
            x_a, x_b = binary_paths(rng, size, n_cum, BASELINE, p_b)
            cont = sequential_outcomes(x_a, x_b, n_cum, ALPHA, tau=tau)
            ten = sequential_outcomes(x_a[:, sub], x_b[:, sub], n_cum[sub], ALPHA, design10)
            for m, o in (
                ("fixed", ten["fixed"]),
                ("naive_10", ten["naive"]),
                ("group_sequential_10", ten["group_sequential"]),
                ("msprt", cont["msprt"]),
            ):
                acc[m][0] += int(o.rejected.sum())
                acc[m][1] += float(o.sample_fraction.sum())
        agg[label] = {
            m: {"rejection_rate": r / reps_cont, "mean_sample_fraction": f / reps_cont}
            for m, (r, f) in acc.items()
        }

    se_final = np.sqrt((BASELINE * (1 - BASELINE) + p_b_design * (1 - p_b_design)) / n_max)
    drift = delta / se_final
    t10 = design10.info_fractions
    return {
        "design": {
            "baseline": BASELINE,
            "design_relative_lift": DESIGN_LIFT,
            "n_per_arm_max": n_max,
            "alpha": ALPHA,
            "msprt_tau": tau,
            "reps_per_look_setting": reps,
            "reps_continuous": reps_cont,
            "continuous_looks": CONTINUOUS_LOOKS,
            "obf_10_boundaries": design10.boundaries.round(4).tolist(),
        },
        "null": null_rows,
        "continuous": agg,
        "obf_10_analytic": {
            "power": float(crossing_probabilities(design10.boundaries, t10, drift).sum()),
            "mean_sample_fraction": expected_sample_fraction(design10.boundaries, t10, drift),
        },
        "fixed_analytic_power": power_two_proportions(BASELINE, p_b_design, n_max),
    }


def study_cuped(scale: float = 1.0, seed: int = SEEDS["cuped"]) -> dict:
    """(b) Empirical variance ratio Var(CUPED)/Var(plain) vs the theoretical 1 - rho^2."""
    rng = np.random.default_rng(seed)
    n, effect, reps = 1_000, 0.1, _reps(10_000, scale)
    settings = [("normal", r, r) for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
    settings += [("lognormal", r, lognormal_correlation(r, 1.0)) for r in (0.3, 0.6, 0.8, 0.95)]
    rows = []
    for dist, r, rho in settings:
        parts = [cuped_trials(rng, size, n, r, effect, dist) for size in _chunks(reps, 1_000)]
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        ratio, ratio_se = variance_ratio(res["plain"], res["adjusted"])
        theory = 1 - rho**2
        covered = (res["adjusted_ci_low"] <= effect) & (effect <= res["adjusted_ci_high"])
        rows.append(
            {
                "distribution": dist,
                "rho": rho,
                "variance_ratio": ratio,
                "variance_ratio_se": ratio_se,
                "theory": theory,
                "relative_error": ratio / theory - 1,
                "reported_variance_reduction": float(res["reported_variance_reduction"].mean()),
                "effective_sample_size_multiplier": 1 / ratio,
                "bias_adjusted": float(res["adjusted"].mean() - effect),
                "ci_coverage_adjusted": float(covered.mean()),
            }
        )
    z = [abs(r["variance_ratio"] - r["theory"]) / r["variance_ratio_se"] for r in rows]
    return {
        "n_per_arm": n,
        "effect": effect,
        "reps": reps,
        "rows": rows,
        "max_abs_z_vs_theory": float(max(z)),
    }


COVERAGE_BINARY = [
    {"label": "p=10%, no lift, n=10,000", "p_a": 0.10, "lift": 0.0, "n": 10_000},
    {"label": "p=10%, +10%, n=10,000", "p_a": 0.10, "lift": 0.10, "n": 10_000},
    {"label": "p=2%, +20%, n=2,000", "p_a": 0.02, "lift": 0.20, "n": 2_000},
    {"label": "p=1%, +20%, n=300", "p_a": 0.01, "lift": 0.20, "n": 300},
]
COVERAGE_LOGNORMAL = [
    {"label": "revenue, +5%, n=2,000", "sigma": 1.5, "lift": 0.05, "n": 2_000},
    {"label": "revenue, +5%, n=100", "sigma": 1.5, "lift": 0.05, "n": 100},
]


def study_coverage(scale: float = 1.0, seed: int = SEEDS["coverage"]) -> dict:
    """(c) Coverage of nominal 95% CIs for absolute and relative lift."""
    rng = np.random.default_rng(seed)
    reps_b, reps_l = _reps(100_000, scale), _reps(20_000, scale)
    rows = []
    for s in COVERAGE_BINARY:
        cov = binary_coverage(rng, reps_b, s["n"], s["p_a"], s["lift"], ALPHA)
        rows.append({**s, "metric": "conversion", "reps": reps_b, **cov})
    for s in COVERAGE_LOGNORMAL:
        cov = lognormal_coverage(rng, reps_l, s["n"], s["sigma"], s["lift"], ALPHA)
        rows.append({**s, "metric": "revenue (lognormal)", "reps": reps_l, **cov})
    return {"nominal": 1 - ALPHA, "rows": rows}


def study_power(scale: float = 1.0, seed: int = SEEDS["power"]) -> dict:
    """(d) Empirical vs analytic power curves and the exaggeration ratio of 'winners'."""
    rng = np.random.default_rng(seed)
    reps = _reps(50_000, scale)
    n_bin = sample_size_two_proportions(BASELINE, BASELINE * (1 + DESIGN_LIFT), alpha=ALPHA)
    binary = []
    for lift in np.round(np.arange(0, 0.161, 0.02), 3):
        p_b = BASELINE * (1 + lift)
        sim = binary_power(rng, reps, n_bin, BASELINE, p_b, ALPHA)
        analytic = power_two_proportions(BASELINE, p_b, n_bin, alpha=ALPHA)
        binary.append({"relative_lift": float(lift), "analytic": analytic, **sim})
    n_means = 200
    means = []
    for d in np.round(np.arange(0, 0.401, 0.05), 3):
        sim = normal_means_power(rng, reps, n_means, float(d), 1.0, ALPHA)
        analytic = power_two_means(float(d), 1.0, n_means, alpha=ALPHA)
        means.append({"cohens_d": float(d), "analytic": analytic, **sim})

    def max_dev(rows: list[dict]) -> float:
        return float(max(abs(r["power"] - r["analytic"]) for r in rows))

    sample_sizes = [
        {
            "relative_mde": rel,
            "n_per_arm": sample_size_two_proportions(BASELINE, BASELINE * (1 + rel), alpha=ALPHA),
        }
        for rel in (0.01, 0.02, 0.05, 0.10, 0.20)
    ]
    return {
        "reps": reps,
        "binary": {"baseline": BASELINE, "n_per_arm": n_bin, "rows": binary},
        "means": {"sd": 1.0, "n_per_arm": n_means, "rows": means},
        "max_abs_deviation_binary": max_dev(binary),
        "max_abs_deviation_means": max_dev(means),
        "mc_se_at_50pct": _mc_se(0.5, reps),
        "sample_size_table": {"baseline": BASELINE, "power": 0.8, "rows": sample_sizes},
    }


def _round(obj, digits: int = 6):
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def run_all(out: Path, scale: float = 1.0, figures: bool = True) -> dict:
    """Run the four studies, write ``metrics.json`` and the figures into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"scale": scale, "seeds": SEEDS}
    for name, fn in (
        ("peeking", study_peeking),
        ("cuped", study_cuped),
        ("coverage", study_coverage),
        ("power", study_power),
    ):
        start = time.perf_counter()
        metrics[name] = fn(scale)
        print(f"[abkit] study {name:<8} done in {time.perf_counter() - start:6.1f} s")
    metrics = _round(metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if figures:
        from abkit.figures import make_figures  # matplotlib is only needed for figures

        make_figures(metrics, out / "figures")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run abkit's simulation studies.")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument(
        "--scale", type=float, default=1.0, help="multiply all replication counts (e.g. 0.05)"
    )
    args = parser.parse_args(argv)
    start = time.perf_counter()
    run_all(args.out, args.scale)
    print(f"[abkit] wrote {args.out / 'metrics.json'} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
