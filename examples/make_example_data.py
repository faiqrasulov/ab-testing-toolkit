"""Generate the SIMULATED example experiments used in the README and `make run`.

The data are synthetic on purpose: the true effects are known, so the CLI report can be
checked against the ground truth. Nothing here is real user data.

simulated_ab_test.csv      8,000 users, 50/50 split. Treatment truly raises the expected
                           number of orders per user by 8%. Columns: user_id, variant,
                           converted (any order in the test window), revenue (test window)
                           and pre_revenue (same user, 4 weeks before the test; for CUPED).
simulated_ab_test_srm.csv  8,000 users with NO true effect, but a simulated logging bug drops
                           25% of treatment users who never ordered. The naive analysis shows
                           a "significant" lift; the SRM check flags the experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_USERS = 8_000
TRUE_LIFT = 0.08  # relative increase in expected orders per user (main file)
DROP_RATE = 0.25  # share of non-ordering treatment users lost by the logging bug (SRM file)


def simulate_users(rng: np.random.Generator, n: int, lift: float) -> pd.DataFrame:
    """Existing customers with persistent engagement, so pre- and in-test revenue correlate.

    With these parameters about 44% of users order in a 4-week window and the population
    correlation between pre-period and in-test revenue is about 0.55.
    """
    engagement = rng.standard_normal(n)
    price_level = 3.2 + 0.3 * rng.standard_normal(n)  # log of a user's typical order value
    rate = np.exp(-0.7 + 1.0 * engagement)  # expected orders per 4 weeks (existing customers)
    treated = rng.random(n) < 0.5

    def revenue(orders: np.ndarray) -> np.ndarray:
        noise = np.exp(0.25 * rng.standard_normal(n))
        return np.round(orders * np.exp(price_level) * noise, 2)

    orders_pre = rng.poisson(rate)
    orders = rng.poisson(rate * np.where(treated, 1 + lift, 1.0))
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": np.where(treated, "treatment", "control"),
            "converted": (orders > 0).astype(int),
            "revenue": revenue(orders),
            "pre_revenue": revenue(orders_pre),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("examples"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    main_df = simulate_users(rng, N_USERS, TRUE_LIFT)
    main_df.to_csv(args.out / "simulated_ab_test.csv", index=False)

    srm_df = simulate_users(rng, N_USERS, 0.0)
    lost = (
        (srm_df["variant"] == "treatment")
        & (srm_df["converted"] == 0)
        & (rng.random(len(srm_df)) < DROP_RATE)
    )
    srm_df.loc[~lost, ["user_id", "variant", "converted"]].to_csv(
        args.out / "simulated_ab_test_srm.csv", index=False
    )
    print(
        f"wrote {args.out / 'simulated_ab_test.csv'} ({len(main_df):,} rows) and "
        f"{args.out / 'simulated_ab_test_srm.csv'} ({int((~lost).sum()):,} rows)"
    )


if __name__ == "__main__":
    main()
