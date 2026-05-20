# DUJ Scaling Experiments — Jetson Orin Nano

## 1B Resident-Style Result

Configuration:

- Parents: 200,000,000
- Children: 800,000,000
- Total agents: 1,000,000,000
- Steps: 10

Observed:

- Zero violations from steps 0–7
- 2 corrections at step 8
- 12 corrections at step 9
- Stable post-warmup latency: ~3362 ms/step
- No crash, no OOM, no thermal instability
- Sparse corrections only emerged late in the run

Interpretation:

This test demonstrates **1 billion resident-style logical DUJ agents** executing stably on a single Jetson Orin Nano Super (8GB). Unlike the streamed 1B test, this configuration keeps the resident-style hierarchical DUJ structure active across the run.

The separate 1T streamed test is a throughput experiment (batched logical processing), not a resident-state result.

## Environment

- Device: NVIDIA Jetson Orin Nano Super (8GB)
- CUDA execution via Numba
- Parent/child hierarchical DUJ agents
- Sparse event-triggered correction
- Lightweight scalar state (not LLM agents)

---

## Scaling Results

| Experiment | Total Agents | Mode | Result |
|---|---:|---|---|
| DUJ Resident | 40K | Resident | ~0.49 ms/step |
| DUJ Resident | 1M | Resident | ~6.56 ms/step |
| DUJ Resident | 500M | Resident-style | ~1.68 s/step |
| DUJ Resident | 995M | Resident-style | PASS |
| DUJ Resident | 1B | Resident-style | PASS (~3.36 s/step post-warmup) |
| DUJ Streamed | 1B | Streamed (1M batches) | ~5.1 s/pass |

---

## 1B Resident-Style Result

Configuration:

- Parents: 200,000,000
- Children: 800,000,000
- Total agents: 1,000,000,000
- Steps: 10

Observed:

- Stable post-warmup latency: ~3362 ms/step
- Near-zero violations for most of the run
- Sparse corrections emerged only at later steps

Example:

```text
step=008 viol=2 corr=2 energy=0.318
step=009 viol=12 corr=12 energy=1.920
```

---

## Important Caveats

These are **lightweight scalar DUJ agents**, not LLM-based autonomous agents.

The **streamed 1B test** processes logical agents in CUDA batches of 1M.

The **resident-style 1B test** maintains parent/child state arrays in device-accessible memory and executes updates directly on the Jetson GPU.

The purpose of these experiments is to evaluate **scaling behavior, sparse correction dynamics, and edge hardware feasibility under constrained compute**.
