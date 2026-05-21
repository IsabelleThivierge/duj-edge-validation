# DUJ Edge Validation

## DUJ Living Swarm Visualization

A real-time visualization of DUJ swarm dynamics running locally on the NVIDIA Jetson Orin Nano.

**Note:** The visualization shows a representative **10K-agent swarm for interpretability**, while large-scale resident-style experiments executed **up to 1 billion logical DUJ agents** on a single Jetson Orin Nano Super (8GB).

<p align="center">
  <img src="./docs/assets/duj_swarm.gif" width="700">
</p>

<p align="center">
  <em>Green = stable agents • Yellow = correction events</em>
</p>

## Key Results

- **1B resident-style logical DUJ agents** executed on a single Jetson Orin Nano Super (8GB)
- **Zero violations through steps 0–7**
- **Sparse corrections only emerged late in the run**
- **No crash, no OOM, no thermal instability**
- **~3.36 s/step post-warmup at 1B resident-style scale**

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

## Resources

- 📄 [Experiment note](docs/v0_4_adversarial_multi_agent_stress_test.md)
- 📊 [Raw results](results/v0_4_adversarial/v31c_adversarial_results.txt)
- 📄 [V31C experimental results](docs/v31c_experimental_results.md)

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

