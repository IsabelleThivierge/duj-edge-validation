# DUJ Edge Validation

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

See full breakdown:

→ [`docs/results_summary.md`](docs/results_summary.md)

---

## Status

Early validation. Ongoing work focuses on reproducibility, scaling, and formalization.
