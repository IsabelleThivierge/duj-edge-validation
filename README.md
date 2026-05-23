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
## Experimental Progression

The repository evolved through progressively harder coordination tests:

- **V31C** — baseline adversarial coordination under signal corruption
- **V39** — layer ablation sweep to isolate micro, meso, and macro contributions
- **V40** — delayed micro propagation test showing meso activation under slower local intervention
- **V41** — macro activation test demonstrating that higher-layer intervention becomes necessary under stronger shocks and delayed lower layers

Emerging experimental pattern:

```text
Mild regime → micro sufficient
Delayed local correction → meso becomes useful
Harsh propagation regime → macro becomes necessary
```

## Resources

- 📄 [Experiment note](docs/v0_4_adversarial_multi_agent_stress_test.md)
- 📊 [Raw results](results/v0_4_adversarial/v31c_adversarial_results.txt)
- 📄 [V31C experimental results](docs/v31c_experimental_results.md)
- 📄 [V41 macro activation test](docs/v41_macro_activation_test.md)

