#!/usr/bin/env python3
"""
V31C-style DUJ reproducibility sweep
- Multi-seed adversarial stress test
- Compares:
  1. control
  2. continuous_supervision
  3. event_gated_duj
  4. event_gated_no_cooccurrence
- Outputs:
  results/v31c_raw.csv
  results/v31c_summary.csv
  results/v31c_summary.txt
  results/v31c_failure_plot.png
  results/v31c_budget_plot.png
  results/v31c_latency_plot.png
"""

import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    agents: int = 5000
    steps: int = 1000
    drift_sigma: float = 0.035
    shock_prob: float = 0.015
    shock_scale: float = 0.35
    false_positive_rate: float = 0.30
    false_negative_rate: float = 0.10
    delay_prob: float = 0.10
    regret_noise: float = 0.08
    failure_threshold: float = 3.0
    correction_gain: float = 0.35
    event_threshold: float = 0.75
    cooccurrence_window: int = 5
    cooccurrence_min_hits: int = 2


@dataclass
class RunResult:
    mode: str
    seed: int
    agents: int
    steps: int
    failures: int
    failure_rate: float
    max_abs_state: float
    mean_abs_state: float
    correction_budget: float
    correction_events: int
    latency_ms: float
    final_instability: float


def adversarial_signal(
    rng: np.random.Generator,
    true_risk: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Produces corrupted risk signal:
    - false positives: flags safe agents
    - false negatives: hides risky agents
    """
    signal = true_risk.copy()

    safe = ~true_risk
    risky = true_risk

    false_pos = safe & (rng.random(cfg.agents) < cfg.false_positive_rate)
    false_neg = risky & (rng.random(cfg.agents) < cfg.false_negative_rate)

    signal[false_pos] = True
    signal[false_neg] = False

    return signal


def apply_correction(
    state: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pulls selected agents back toward bounded state.
    Regret noise simulates imperfect correction.
    """
    corrected = state.copy()
    noise = rng.normal(0, cfg.regret_noise, size=cfg.agents)

    corrected[mask] = (
        corrected[mask]
        - cfg.correction_gain * corrected[mask]
        + noise[mask]
    )

    return corrected


def run_single(mode: str, seed: int, cfg: Config) -> RunResult:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    state = rng.normal(0, 0.05, size=cfg.agents)

    correction_budget = 0.0
    correction_events = 0
    failures_seen = np.zeros(cfg.agents, dtype=bool)

    recent_hits = np.zeros((cfg.cooccurrence_window, cfg.agents), dtype=bool)

    delayed_mask = np.zeros(cfg.agents, dtype=bool)

    for t in range(cfg.steps):
        # baseline drift
        state += rng.normal(0, cfg.drift_sigma, size=cfg.agents)

        # rare adversarial shocks
        shocks = rng.random(cfg.agents) < cfg.shock_prob
        state[shocks] += rng.normal(0, cfg.shock_scale, size=np.sum(shocks))

        true_risk = np.abs(state) > cfg.event_threshold
        noisy_signal = adversarial_signal(rng, true_risk, cfg)

        # delayed signal injection
        use_delay = rng.random() < cfg.delay_prob
        if use_delay:
            current_signal = delayed_mask.copy()
            delayed_mask = noisy_signal.copy()
        else:
            current_signal = noisy_signal

        if mode == "control":
            correction_mask = np.zeros(cfg.agents, dtype=bool)

        elif mode == "continuous_supervision":
            correction_mask = current_signal | (np.abs(state) > cfg.event_threshold * 0.5)

        elif mode == "event_gated_no_cooccurrence":
            correction_mask = current_signal

        elif mode == "event_gated_duj":
            recent_hits[t % cfg.cooccurrence_window] = current_signal
            hit_count = recent_hits.sum(axis=0)

            correction_mask = (
                current_signal
                & (hit_count >= cfg.cooccurrence_min_hits)
                & (np.abs(state) > cfg.event_threshold)
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if np.any(correction_mask):
            state = apply_correction(state, correction_mask, cfg, rng)
            correction_budget += float(np.sum(correction_mask))
            correction_events += 1

        failed_now = np.abs(state) > cfg.failure_threshold
        failures_seen |= failed_now

    latency_ms = (time.perf_counter() - start) * 1000.0

    failures = int(np.sum(failures_seen))
    failure_rate = failures / cfg.agents

    return RunResult(
        mode=mode,
        seed=seed,
        agents=cfg.agents,
        steps=cfg.steps,
        failures=failures,
        failure_rate=failure_rate,
        max_abs_state=float(np.max(np.abs(state))),
        mean_abs_state=float(np.mean(np.abs(state))),
        correction_budget=float(correction_budget),
        correction_events=int(correction_events),
        latency_ms=float(latency_ms),
        final_instability=float(np.mean(np.abs(state) > cfg.event_threshold)),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "failures",
        "failure_rate",
        "max_abs_state",
        "mean_abs_state",
        "correction_budget",
        "correction_events",
        "latency_ms",
        "final_instability",
    ]

    rows = []

    for mode, group in df.groupby("mode"):
        row = {"mode": mode, "n": len(group)}
        for m in metrics:
            values = group[m].to_numpy()
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0

            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mode")


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, outpath: str):
    x = summary["mode"]
    y = summary[f"{metric}_mean"]
    err = summary[f"{metric}_ci95"]

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, yerr=err, capsize=5)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by mode, mean ± 95% CI")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def write_summary_txt(summary: pd.DataFrame, outpath: str):
    lines = []
    lines.append("V31C Reproducibility Sweep Summary")
    lines.append("=" * 40)
    lines.append("")

    for _, row in summary.iterrows():
        mode = row["mode"]
        lines.append(f"Mode: {mode}")
        lines.append(f"  seeds: {int(row['n'])}")
        lines.append(
            f"  failures: {row['failures_mean']:.2f} ± {row['failures_ci95']:.2f}"
        )
        lines.append(
            f"  failure_rate: {row['failure_rate_mean']:.6f} ± {row['failure_rate_ci95']:.6f}"
        )
        lines.append(
            f"  correction_budget: {row['correction_budget_mean']:.2f} ± {row['correction_budget_ci95']:.2f}"
        )
        lines.append(
            f"  latency_ms: {row['latency_ms_mean']:.2f} ± {row['latency_ms_ci95']:.2f}"
        )
        lines.append(
            f"  final_instability: {row['final_instability_mean']:.6f} ± {row['final_instability_ci95']:.6f}"
        )
        lines.append("")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--agents", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config(
        agents=args.agents,
        steps=args.steps,
    )

    modes = [
        "control",
        "continuous_supervision",
        "event_gated_no_cooccurrence",
        "event_gated_duj",
    ]

    results: List[Dict] = []

    print("Starting V31C reproducibility sweep")
    print(f"agents={cfg.agents}, steps={cfg.steps}, seeds={args.seeds}")
    print("")

    for mode in modes:
        print(f"Running mode: {mode}")
        for seed in range(args.seeds):
            result = run_single(mode=mode, seed=seed, cfg=cfg)
            results.append(asdict(result))

            if (seed + 1) % 10 == 0 or seed == args.seeds - 1:
                print(f"  completed seed {seed + 1}/{args.seeds}")

    raw = pd.DataFrame(results)
    summary = summarize(raw)

    raw_path = os.path.join(args.outdir, "v31c_raw.csv")
    summary_path = os.path.join(args.outdir, "v31c_summary.csv")
    txt_path = os.path.join(args.outdir, "v31c_summary.txt")

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_summary_txt(summary, txt_path)

    plot_metric(
        summary,
        metric="failures",
        ylabel="Failures",
        outpath=os.path.join(args.outdir, "v31c_failure_plot.png"),
    )

    plot_metric(
        summary,
        metric="correction_budget",
        ylabel="Correction Budget",
        outpath=os.path.join(args.outdir, "v31c_budget_plot.png"),
    )

    plot_metric(
        summary,
        metric="latency_ms",
        ylabel="Latency (ms)",
        outpath=os.path.join(args.outdir, "v31c_latency_plot.png"),
    )

    print("")
    print("Done.")
    print(f"Raw results: {raw_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary TXT: {txt_path}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
