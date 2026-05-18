# DUJ Edge Validation — v1 Flagship Experiment

## Problem

Continuous correction can maintain stability in large-scale multi-agent coordination, but it can become expensive as intervention frequency grows.

## Question

Can sparse event-gated invariant steering maintain bounded behavior with substantially lower correction cost?

## Result

Under tested conditions on Jetson Orin Nano, DUJ-style event-gated steering maintained zero observed failures across 50 seeds with 5,000 agents and 1,000 steps, while reducing correction budget from 1,539,312.84 to 35,162.52 versus continuous supervision.

This corresponds to an approximately 97.7% correction budget reduction.

## Summary

| Mode | Failure Rate | Correction Budget | Latency Mean |
|---|---:|---:|---:|
| control | 15.7% | 0.00 | 358.93 ms |
| continuous_supervision | 0.0% | 1,539,312.84 | 638.51 ms |
| event_gated_no_cooccurrence | 0.0% | 1,501,954.34 | 624.10 ms |
| event_gated_duj | 0.0% | 35,162.52 | 564.73 ms |

## Key Observation

Simple event gating without co-occurrence filtering produced almost no correction-budget savings versus continuous supervision. The large reduction appears only in the DUJ event-gated condition, suggesting that co-occurrence filtering is the main driver of sparsity in this experiment.

## Reproduce

```bash
pip install -r requirements.txt
python3 run_v1_flagship.py --agents 5000 --steps 1000 --seeds 50 --outdir results
```

## Output

The run generates:

- `results/v31c_raw.csv`
- `results/v31c_summary.csv`
- `results/v31c_summary.txt`

## Scope and Limitations

This is a centralized edge validation experiment executed on Jetson Orin Nano. It does not yet demonstrate fully distributed hardware coordination.

The result should be interpreted as an empirical stress test under the tested configuration, not as a formal proof of stability.
