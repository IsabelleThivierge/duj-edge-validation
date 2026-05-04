# DUJ Edge Validation — Results Summary

Event-triggered invariant steering maintains bounded multi-agent dynamics while using ~28% of the correction budget of continuous enforcement on edge hardware.

## Overview

This repository documents early validation results for an event-triggered invariant steering mechanism (DUJ / FEITH-R) evaluated on edge hardware.

The goal is to test whether selective, threshold-driven corrections can maintain bounded multi-agent dynamics without continuous enforcement.

---

## Experimental Setup

- Device: NVIDIA Jetson Orin Nano
- Agents: 10,000
- Steps: 120
- Invariants:
  - Balance (global conservation)
  - Variance (dispersion)
  - Energy (state magnitude)
  - Center (global drift)

Three modes were evaluated:

- NULL (no correction)
- CONTINUOUS (full correction every step)
- EVENT (selective correction when thresholds are exceeded)

---

## Key Results

### NULL (no correction)

- System diverges rapidly
- maxabs ≈ 4471
- multi-invariant failure in 114 / 120 steps
- all invariants active simultaneously in 109 / 120 steps

---

### CONTINUOUS (full enforcement)

- Fully stable
- maxabs ≈ 2.29
- All invariants strictly bounded
- 480 / 480 corrections (100% budget)
- Latency ≈ 778 ms

---

### EVENT (selective enforcement)

- Stable, bounded system
- maxabs ≈ 2.27 (comparable to continuous)
- 134 / 480 corrections (~27.9% of full budget)
- Latency ≈ 466 ms (~40% reduction vs continuous)
- Multi-invariant triggers present but controlled (51 / 120 steps)
- No full 4-invariant simultaneous failure observed

---

## Interpretation

Event-triggered invariant steering:

- Prevents divergence observed in uncontrolled systems
- Achieves stability comparable to continuous enforcement
- Reduces correction budget by ~72%
- Reduces latency significantly
- Maintains bounded system behavior under multi-invariant pressure

---

## Status

These results demonstrate the feasibility of selective invariant enforcement on constrained edge hardware.

Further work focuses on:
- Reproducibility
- Scaling behavior
- Formalization of the coordination mechanism
