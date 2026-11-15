"""Command-line interface: ``abkit analyze`` and ``abkit sample-size``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from abkit.power import sample_size_two_means, sample_size_two_proportions
from abkit.report import analyze, render_markdown


def _cmd_analyze(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    analysis = analyze(
        df,
        group=args.group,
        metric=args.metric,
        pre_metric=args.pre_metric,
        control=args.control,
        alpha=args.alpha,
        power=args.power,
        control_share=args.control_share,
        metric_type=args.metric_type,
        source=str(args.csv),
    )
    text = render_markdown(analysis)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


def _cmd_sample_size(args: argparse.Namespace) -> int:
    if args.baseline is not None:
        if args.relative_mde is None:
            raise SystemExit("--baseline needs --relative-mde")
        p_b = args.baseline * (1 + args.relative_mde)
        n_a = sample_size_two_proportions(args.baseline, p_b, args.power, args.alpha, args.ratio)
        effect = f"{args.baseline:.2%} -> {p_b:.2%} ({args.relative_mde:+.1%} relative)"
    else:
        if args.sd is None or args.mde is None:
            raise SystemExit("give --baseline/--relative-mde (conversion) or --sd/--mde (means)")
        n_a = sample_size_two_means(args.mde, args.sd, args.power, args.alpha, args.ratio)
        effect = f"difference in means {args.mde:g} (sd {args.sd:g})"
    n_b = math.ceil(n_a * args.ratio - 1e-9)
    print(
        f"Detect {effect} with {args.power:.0%} power at alpha = {args.alpha:g} (two-sided):\n"
        f"  control n = {n_a:,}, treatment n = {n_b:,}, total = {n_a + n_b:,}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abkit", description="Rigorous A/B-test analysis with simulation-verified methods."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyze", help="analyse an experiment CSV and print a markdown report")
    an.add_argument("csv", type=Path, help="one row per user")
    an.add_argument("--group", required=True, help="column with the variant label")
    an.add_argument("--metric", required=True, help="outcome column (0/1 or continuous)")
    an.add_argument("--pre-metric", help="pre-experiment covariate column for CUPED")
    an.add_argument("--control", help="name of the control group (default: auto-detect)")
    an.add_argument("--alpha", type=float, default=0.05)
    an.add_argument("--power", type=float, default=0.8, help="power for the MDE check")
    an.add_argument("--control-share", type=float, default=0.5, help="planned control share")
    an.add_argument("--metric-type", choices=("auto", "binary", "continuous"), default="auto")
    an.add_argument("--output", help="also write the report to this file")
    an.set_defaults(func=_cmd_analyze)

    ss = sub.add_parser("sample-size", help="users per arm needed to detect an effect")
    ss.add_argument("--baseline", type=float, help="control conversion rate, e.g. 0.10")
    ss.add_argument("--relative-mde", type=float, help="relative lift to detect, e.g. 0.05")
    ss.add_argument("--sd", type=float, help="standard deviation of a continuous metric")
    ss.add_argument("--mde", type=float, help="absolute difference in means to detect")
    ss.add_argument("--alpha", type=float, default=0.05)
    ss.add_argument("--power", type=float, default=0.8)
    ss.add_argument("--ratio", type=float, default=1.0, help="treatment n / control n")
    ss.set_defaults(func=_cmd_sample_size)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"abkit: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
