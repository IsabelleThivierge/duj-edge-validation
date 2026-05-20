# DUJ Edge Validation

## DUJ Living Swarm Visualization

A real-time visualization of DUJ swarm dynamics running locally on the NVIDIA Jetson Orin Nano.

<p align="center">
  <img src="./docs/assets/duj_swarm.gif" width="700">
</p>

## Overview

This repository explores **event-triggered invariant steering** for maintaining bounded multi-agent dynamics under adversarial conditions on **edge hardware**.

The core hypothesis:

> **Coordination mechanism design may matter nearly as much as compute scaling for stable multi-agent systems.**

Experiments evaluate whether sparse, event-gated intervention with **DUJ co-occurrence filtering** can maintain viability while dramatically reducing correction cost compared to continuous supervision.

All experiments are run on **Jetson Orin Nano Super** (JetPack 6.2.1, CUDA).

---
## Quick Start

### Install dependencies

```bash
pip install numpy pandas matplotlib
```

### Run reproducibility sweep

```bash
python3 v31c_repro_sweep.py
```

### Run adversarial noise sweep

```bash
python3 v31c_noise_sweep.py
```

### Run co-occurrence phase sweep

```bash
python3 v31c_cooccurrence_phase_sweep.py
```

### Run sweet spot robustness sweep

```bash
python3 v31c_sweetspot_noise_sweep.py
```

## Experimental Sweeps

This repository currently includes:

- Reproducibility sweep (50 seeds)
- Adversarial noise sweep
- Co-occurrence phase sweep
- Sweet spot robustness sweep
- Extreme stress testing

## Latest Result — Adversarial Multi-Agent Stress Test (v0.4)

Early results suggest that coordination mechanism may matter as much as compute substrate for multi-agent viability.

In adversarial multi-agent stress tests on **Jetson Orin Nano Super** (JetPack 6.2.1, CUDA), we observed:

- **Event-gated coordination maintained zero agent deaths**
- Graceful degradation under adversarial signal noise up to **50% false-positive rate**
- Successful adaptation across **7-phase non-monotonic regime schedules**
- Continuous supervision repeatedly collapsed under the same conditions
- Coordination achieved at substantially lower intervention cost

This is still early work. The emerging hypothesis is that **compute scaling and coordination scaling may be orthogonal problems**.

## Resources

- 📄 [Experiment note](docs/v0_4_adversarial_multi_agent_stress_test.md)
- 📊 [Raw results](results/v0_4_adversarial/v31c_adversarial_results.txt)

Event-triggered invariant steering maintains bounded multi-agent dynamics on edge hardware with reduced correction cost.
---

## Key Result

- Uncontrolled system diverges: `maxabs ≈ 5.60`
- Event-triggered constraint remains bounded:`≈ 2.0 (range: 1.95–2.29)
- Uses as few as **4 corrections over 120 steps**
- Reduces latency from ~390ms → ~207ms  
---

## Summary

This project demonstrates that **selective, threshold-based corrections** can maintain stability in multi-agent systems without continuous enforcement.

In a 10,000-agent system on edge hardware (Jetson Orin Nano):

- Event-triggered correction used **~27.9% of the continuous correction budget**
- Maintained **bounded behavior comparable to continuous enforcement**
- Reduced latency by **~40%**
- Prevented multi-invariant collapse observed in uncontrolled dynamics

---

## Approach

Instead of enforcing constraints at every step, the system:

- Monitors invariant violations
- Applies corrections **only when thresholds are exceeded**
- Maintains global stability with significantly fewer interventions

---

## Results

Latest:

- [v0.4 adversarial multi-agent stress test](docs/v0_4_adversarial_multi_agent_stress_test.md)
- [Raw v31c adversarial results](results/v0_4_adversarial/v31c_adversarial_results.txt)

Earlier validation:

- [Edge validation results summary](docs/results_summary.md)
---
## Limitations

- Single noise profile tested; behavior under different perturbation distributions unknown
- Results from a single seed; cross-seed variance not yet characterized
- Simulation environment; real multi-agent deployment not yet validated

---

## Status

Early validation. Ongoing work focuses on reproducibility, scaling, and formalization.

# V31C Experimental Results

## Overview

We evaluated **event-gated invariant steering with co-occurrence filtering (DUJ)** under adversarial multi-agent conditions on **Jetson Orin Nano Super (JetPack 6.x, CUDA)**.

The goal was to test whether sparse, event-triggered coordination could preserve bounded behavior more efficiently than continuous supervision while remaining robust under adversarial signal corruption.

We compared four intervention modes:

| Mode | Description |
|-------|-------------|
| `control` | No intervention |
| `continuous_supervision` | Continuous correction whenever instability signals emerge |
| `event_gated_no_cooccurrence` | Event-triggered correction without co-occurrence filtering |
| `event_gated_duj` | Event-triggered correction with DUJ co-occurrence filtering |

---

## V31C Reproducibility Sweep (50 Seeds)

### Configuration

- **Agents:** 5,000  
- **Steps:** 1,000  
- **Seeds:** 50  
- **Adversarial signal noise:** enabled  
- **Hardware:** NVIDIA Jetson Orin Nano Super

### Summary Results

| Mode | Failure Rate | Correction Budget | Final Instability | Latency (ms) |
|------|---------------|------------------|-------------------|---------------|
| continuous_supervision | 0.0000 | 1,539,312 | 0.000044 | 638.87 |
| control | 0.1570 | 0 | 0.666980 | 359.62 |
| event_gated_duj | 0.0000 | 35,163 | 0.004044 | 566.65 |
| event_gated_no_cooccurrence | 0.0000 | 1,501,954 | 0.000352 | 625.10 |

### Key Observation

**Event-gated DUJ maintained zero failures while reducing correction budget by ~97.7% relative to continuous supervision.**

The strongest mechanistic signal came from the **co-occurrence filter**:

- `event_gated_no_cooccurrence` required ~1.50M corrections
- `event_gated_duj` required ~35k corrections

This suggests co-occurrence filtering substantially suppresses unnecessary intervention churn while preserving stability.

---

## Adversarial Noise Sweep

We evaluated robustness under progressively increasing **false-positive signal corruption**:

```text
0%, 10%, 30%, 50%, 70%, 85%, 95%
```

Metrics tracked:

- Failure rate
- Correction budget
- Final instability
- Latency

Generated outputs:

- `noise_sweep_results/failure_rate.png`
- `noise_sweep_results/correction_budget.png`
- `noise_sweep_results/final_instability.png`
- `noise_sweep_results/latency_ms.png`

### Main Result

Under moderate adversarial corruption, **DUJ remained stable with dramatically lower intervention cost than continuous supervision**, while maintaining bounded behavior.

---

## Extreme Stress Regime

We intentionally stress-tested the system under substantially harsher conditions:

### Stressors

- Increased stochastic drift
- Larger adversarial shocks
- Higher false-negative rate
- Delayed interventions
- Regret noise
- High false-positive corruption

### Main Finding

Under extreme conditions, **continuous supervision achieved the strongest suppression**, but at very high correction cost.

DUJ no longer preserved perfect zero-failure stability, but **degraded gracefully**:

- substantially lower failure rate than no intervention
- ~20x lower correction budget than continuous supervision at high noise
- bounded degradation rather than collapse

Example at **95% false-positive noise**:

| Mode | Failure Rate | Correction Budget |
|------|---------------|------------------|
| continuous_supervision | 0.0001 | 19.24M |
| event_gated_duj | 0.0168 | 0.89M |
| control | 0.9962 | 0 |

This suggests a meaningful engineering tradeoff between:

**maximum stability** ↔ **intervention efficiency**

---

## Current Interpretation

These experiments provide early evidence that:

> **coordination mechanism design may matter nearly as much as compute substrate for multi-agent stability under adversarial conditions.**

Sparse, event-gated intervention appears capable of preserving bounded behavior while dramatically reducing corrective overhead relative to continuous supervision.

This work remains **early-stage experimental research** and has important limitations:

- centralized simulation
- limited adversarial profiles
- no distributed networked agents yet
- invariants are still being formalized

Future work includes:

- distributed multi-node testing
- larger-scale agent sweeps
- stronger control-theory baselines
- formal invariant analysis
- edge latency/coordination studies
