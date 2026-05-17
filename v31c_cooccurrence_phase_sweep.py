import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [3, 5, 7, 10]
MIN_HITS = [1, 2, 3, 4, 5]

SEEDS = 30
AGENTS = 10000
STEPS = 2000
NOISE_LEVEL = 0.50

FALSE_NEGATIVE_RATE = 0.20
INTERVENTION_DELAY = 5
REGRET_NOISE = 0.20
DRIFT_SCALE = 1.50
ADVERSARIAL_SHOCK_PROB = 0.02

OUTPUT_DIR = Path("cooccurrence_phase_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_simulation(
    seed,
    cooccurrence_window,
    cooccurrence_min_hits,
):

    rng = np.random.default_rng(seed)

    states = rng.normal(0, 0.05, AGENTS)
    alive = np.ones(AGENTS, dtype=bool)

    correction_budget = 0
    correction_history = []

    start_time = time.time()

    for step in range(STEPS):

        drift = rng.normal(0, 0.01 * DRIFT_SCALE, AGENTS)
        states += drift

        shock_mask = (
            rng.random(AGENTS)
            < ADVERSARIAL_SHOCK_PROB
        )
        states[shock_mask] += rng.normal(
            0,
            0.20,
            shock_mask.sum()
        )

        risky = np.abs(states) > 0.50

        false_positive = (
            rng.random(AGENTS)
            < NOISE_LEVEL
        )

        false_negative = (
            rng.random(AGENTS)
            < FALSE_NEGATIVE_RATE
        )

        signal = (
            (risky | false_positive)
            & (~false_negative)
        )

        correction_history.append(signal)

        if len(correction_history) > cooccurrence_window:
            correction_history.pop(0)

        stacked = np.stack(correction_history)

        cooccurring = (
            stacked.sum(axis=0)
            >= cooccurrence_min_hits
        )

        correction_mask = (
            cooccurring
            & alive
        )

        delayed_mask = (
            rng.random(AGENTS)
            > (INTERVENTION_DELAY / 10)
        )

        correction_mask &= delayed_mask

        correction_strength = (
            0.10
            + rng.normal(
                0,
                REGRET_NOISE,
                AGENTS
            )
        )

        states[correction_mask] -= (
            np.sign(
                states[correction_mask]
            )
            * correction_strength[
                correction_mask
            ]
        )

        correction_budget += (
            correction_mask.sum()
        )

        dead = (
            np.abs(states)
            > 1.50
        )

        alive[dead] = False

    latency_ms = (
        time.time()
        - start_time
    ) * 1000

    failure_rate = (
        (~alive).sum()
        / AGENTS
    )

    return {
        "failure_rate":
            failure_rate,
        "correction_budget":
            correction_budget,
        "final_instability":
            np.mean(np.abs(states)),
        "latency_ms":
            latency_ms,
    }


results = []

print(
    "Starting Co-occurrence "
    "Phase Sweep"
)
print()

for window, min_hits in itertools.product(
    WINDOWS,
    MIN_HITS,
):

    print(
        f"window={window}, "
        f"min_hits={min_hits}"
    )

    for seed in range(SEEDS):

        metrics = run_simulation(
            seed=seed,
            cooccurrence_window=window,
            cooccurrence_min_hits=min_hits,
        )

        metrics["window"] = window
        metrics["min_hits"] = min_hits
        metrics["seed"] = seed

        results.append(metrics)

        if (seed + 1) % 10 == 0:
            print(
                f"  seed "
                f"{seed+1}/{SEEDS}"
            )

df = pd.DataFrame(results)

raw_path = (
    OUTPUT_DIR
    / "cooccurrence_phase_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "cooccurrence_phase_summary.csv"
)

df.to_csv(
    raw_path,
    index=False,
)

summary = (
    df.groupby(
        ["window", "min_hits"]
    )[
        [
            "failure_rate",
            "correction_budget",
            "final_instability",
            "latency_ms",
        ]
    ]
    .mean()
    .round(4)
)

summary.to_csv(summary_path)

print()
print("Saved results.")
print(summary)
print()
print("Done.")
