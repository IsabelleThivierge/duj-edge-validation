# DUJ Edge Validation — v1 Flagship Experiment

## Problem

Continuous correction can maintain stability in large-scale multi-agent coordination, but it can become expensive as intervention frequency grows.

## Question

Can sparse event-gated invariant steering maintain bounded behavior with substantially lower correction cost?

## Result

Under tested conditions on Jetson Orin Nano, DUJ-style event-gated steering maintained zero observed failures across 50 seeds with 5,000 agents and 1,000 steps, while reducing correction budget from 1,539,312.84 to 35,162.52 versus continuous supervision.

This corresponds to an approximately 97.7% correction budget reduction.

## Experimental Configuration

- Hardware: NVIDIA Jetson Orin Nano
- Agents: 5,000
- Steps: 1,000
- Seeds: 50
- Configuration: centralized edge validation experiment

## Core Hypothesis

Coordination overhead can become a tighter scaling bottleneck than raw compute in large multi-agent systems.

## Summary

| Mode | Failure Rate | Correction Budget | Latency Mean |
|---|---:|---:|---:|
| control | 15.7% | 0.00 | 358.93 ms |
| continuous_supervision | 0.0% | 1,539,312.84 | 638.51 ms |
| event_gated_no_cooccurrence | 0.0% | 1,501,954.34 | 624.10 ms |
| event_gated_duj | 0.0% | 35,162.52 | 564.73 ms |

## Key Observation

This suggests that co-occurrence filtering is the primary driver of the observed sparsity in event-gated invariant steering under these tested conditions.

## Reproduce

Clone the repository:

```bash
git clone https://github.com/IsabelleThivierge/duj-edge-validation.git
cd duj-edge-validation
```

Install dependencies:

```bash
pip install -r v1_flagship/requirements.txt
```

Run the flagship experiment:

```bash
python3 v1_flagship/run_v1_flagship.py \
    --agents 5000 \
    --steps 1000 \
    --seeds 50 \
    --outdir v1_flagship/results
```

Generate figures:

```bash
python3 v1_flagship/plot_flagship.py
```

## Output

The run generates:

- `results/v31c_raw.csv`
- `results/v31c_summary.csv`
- `results/v31c_summary.txt`

## Scope and Limitations

This experiment uses 5,000 simulated agents executed in a centralized simulation on Jetson Orin Nano. The agents are not 5,000 physical robots, distributed devices, or independent concurrent OS processes.

The result demonstrates centralized edge execution and empirical coordination behavior under the tested simulation configuration. It does not yet demonstrate fully distributed hardware coordination.

The result should be interpreted as an empirical stress test under the tested configuration, not as a formal proof of stability.
