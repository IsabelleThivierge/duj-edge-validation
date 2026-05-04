# D2 — Threshold Sweep (Single Invariant)
## Overview
This experiment evaluates how an event-triggered invariant constraint behaves as the activation threshold is varied.
The goal is to measure whether bounded system behavior can be maintained while reducing correction frequency.
---
## Setup
- Device: NVIDIA Jetson Orin Nano
- Agents: 10,000
- Steps: 120
- Mode: Event-triggered invariant enforcement
- Variable: Threshold (trigger level for correction)
---
## Raw Output
```

Device: cuda
Agents: 10000
Steps: 120

NULL       | inv_int=22245934.47 | inv_max=367651.00 | maxabs=5.60 | lat=405.8ms

Threshold sweep:

thr=5000   | inv_int=183853.84 | inv_max=3113.10 | proj=60/120 | maxabs=1.95 | lat=390.1ms
thr=10000  | inv_int=551568.79 | inv_max=9258.75 | proj=30/120 | maxabs=1.95 | lat=280.6ms
thr=20000  | inv_int=1097592.49 | inv_max=18464.07 | proj=17/120 | maxabs=1.95 | lat=250.5ms
thr=40000  | inv_int=2341059.23 | inv_max=39949.29 | proj=8/120 | maxabs=2.16 | lat=217.1ms
thr=80000  | inv_int=4541430.65 | inv_max=79786.64 | proj=4/120 | maxabs=2.29 | lat=207.5ms

```
Done.

---
## Observations
- The NULL system diverges (maxabs ≈ 5.60)
- All thresholded configurations remain bounded (~1.95–2.29)
- Correction frequency decreases significantly as threshold increases:
  - 60/120 → 4/120
- Latency decreases with fewer corrections
---
## Interpretation
This experiment shows that:
- Stability does not require continuous enforcement  
- Event-triggered corrections maintain bounded behavior  
- The threshold acts as a tunable tradeoff between:
  - correction frequency  
  - computational cost  
  - tightness of control  
---
## Key Result
Event-triggered invariant enforcement maintains bounded system behavior while reducing correction frequency by up to ~85% compared to stricter configurations.
