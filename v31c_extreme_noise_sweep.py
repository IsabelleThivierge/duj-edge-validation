import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# ======================
# CONFIG
# ======================

AGENTS = 10000
STEPS = 2000
SEEDS = 50

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95]

MODES = [
    "control",
    "continuous_supervision",
    "event_gated_no_cooccurrence",
    "event_gated_duj"
]

OUTDIR = "extreme_noise_sweep_results"

os.makedirs(OUTDIR, exist_ok=True)

# ======================
# EXTREME STRESS CONFIG
# ======================

DRIFT_SIGMA = 0.08       # was 0.035
SHOCK_PROB = 0.03        # was 0.015
SHOCK_SCALE = 0.80       # was 0.35

FALSE_NEGATIVE_RATE = 0.35   # was 0.10
DELAY_PROB = 0.40            # was 0.10
REGRET_NOISE = 0.25          # was 0.08

FAILURE_THRESHOLD = 3.0
CORRECTION_GAIN = 0.35
EVENT_THRESHOLD = 0.75

CO_WINDOW = 5
CO_MIN_HITS = 2


# ======================
# FUNCTIONS
# ======================

def adversarial_signal(rng, true_risk, false_positive_rate):
    signal = true_risk.copy()

    safe = ~true_risk
    risky = true_risk

    false_pos = safe & (
        rng.random(AGENTS)
        < false_positive_rate
    )

    false_neg = risky & (
        rng.random(AGENTS)
        < FALSE_NEGATIVE_RATE
    )

    signal[false_pos] = True
    signal[false_neg] = False

    return signal


def apply_correction(state, mask, rng):

    corrected = state.copy()

    noise = rng.normal(
        0,
        REGRET_NOISE,
        size=AGENTS
    )

    corrected[mask] = (
        corrected[mask]
        - CORRECTION_GAIN
        * corrected[mask]
        + noise[mask]
    )

    return corrected


def run_single(mode, seed, fp_rate):

    rng = np.random.default_rng(seed)

    state = rng.normal(
        0,
        0.05,
        size=AGENTS
    )

    correction_budget = 0

    failures_seen = np.zeros(
        AGENTS,
        dtype=bool
    )

    recent_hits = np.zeros(
        (CO_WINDOW, AGENTS),
        dtype=bool
    )

    delayed_mask = np.zeros(
        AGENTS,
        dtype=bool
    )

    start = time.perf_counter()

    for t in range(STEPS):

        # heavy drift
        state += rng.normal(
            0,
            DRIFT_SIGMA,
            size=AGENTS
        )

        # stronger adversarial shocks
        shocks = (
            rng.random(AGENTS)
            < SHOCK_PROB
        )

        state[shocks] += rng.normal(
            0,
            SHOCK_SCALE,
            size=np.sum(shocks)
        )

        true_risk = (
            np.abs(state)
            > EVENT_THRESHOLD
        )

        noisy_signal = adversarial_signal(
            rng,
            true_risk,
            fp_rate
        )

        # delayed interventions
        if rng.random() < DELAY_PROB:

            current_signal = delayed_mask.copy()

            delayed_mask = noisy_signal.copy()

        else:
            current_signal = noisy_signal

        # ======================
        # CORRECTION MODES
        # ======================

        if mode == "control":

            correction_mask = np.zeros(
                AGENTS,
                dtype=bool
            )

        elif mode == "continuous_supervision":

            correction_mask = (
                current_signal
                | (
                    np.abs(state)
                    > EVENT_THRESHOLD * 0.5
                )
            )

        elif mode == "event_gated_no_cooccurrence":

            correction_mask = current_signal

        elif mode == "event_gated_duj":

            recent_hits[
                t % CO_WINDOW
            ] = current_signal

            hit_count = recent_hits.sum(
                axis=0
            )

            correction_mask = (
                current_signal
                & (
                    hit_count
                    >= CO_MIN_HITS
                )
                & (
                    np.abs(state)
                    > EVENT_THRESHOLD
                )
            )

        # apply corrections
        if np.any(correction_mask):

            state = apply_correction(
                state,
                correction_mask,
                rng
            )

            correction_budget += int(
                np.sum(correction_mask)
            )

        failures_now = (
            np.abs(state)
            > FAILURE_THRESHOLD
        )

        failures_seen |= failures_now

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    failures = int(
        np.sum(failures_seen)
    )

    failure_rate = (
        failures / AGENTS
    )

    final_instability = float(
        np.mean(
            np.abs(state)
            > EVENT_THRESHOLD
        )
    )

    return {
        "mode": mode,
        "seed": seed,
        "noise": fp_rate,
        "failures": failures,
        "failure_rate": failure_rate,
        "correction_budget": correction_budget,
        "latency_ms": latency_ms,
        "final_instability": final_instability
    }


# ======================
# RUN EXPERIMENT
# ======================

results = []

print("Starting EXTREME Noise Sweep")

for noise in NOISE_LEVELS:

    print(f"\nNoise={noise}")

    for mode in MODES:

        print(f"  {mode}")

        for seed in range(SEEDS):

            result = run_single(
                mode,
                seed,
                noise
            )

            results.append(result)

            if (
                (seed + 1)
                % 10
                == 0
            ):
                print(
                    f"    seed {seed+1}/{SEEDS}"
                )

df = pd.DataFrame(results)

df.to_csv(
    f"{OUTDIR}/extreme_noise_raw.csv",
    index=False
)

summary = (
    df.groupby(
        ["noise","mode"]
    )
    .mean(numeric_only=True)
)

summary.to_csv(
    f"{OUTDIR}/extreme_noise_summary.csv"
)

print("\nSaved results.")

print(summary[
    [
        "failure_rate",
        "correction_budget",
        "final_instability",
        "latency_ms"
    ]
])

print("\nDone.")
