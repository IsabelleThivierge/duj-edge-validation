# duj-edge-validation

Edge validation of event-triggered invariant steering (DUJ / FEITH-R).

---

## Summary

This project explores whether **selective, threshold-based corrections** can maintain stability in multi-agent systems without continuous enforcement.

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
