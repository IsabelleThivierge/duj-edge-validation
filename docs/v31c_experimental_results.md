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
