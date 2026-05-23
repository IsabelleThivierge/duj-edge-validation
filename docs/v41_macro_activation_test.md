# V41 Macro Activation Test

## Purpose

This experiment tested whether the macro layer becomes necessary when lower DUJ coordination layers are slowed under stronger shock conditions.

Earlier tests showed:

- Under mild stress, the micro layer was sufficient.
- When micro intervention was delayed, the meso layer became active.
- This test increases shock severity and delays both micro and meso intervention to evaluate whether macro-level correction becomes necessary.

## Configuration

- **Agents:** 5,000
- **Steps:** 120
- **Seeds:** 10
- **Shock probability:** 0.10
- **Shock magnitude:** 0.75
- **Micro cooldown:** 5 steps
- **Meso cooldown:** 3 steps
- **Failure threshold:** 1.0
- **Hardware:** NVIDIA Jetson Orin Nano Super

## Modes Tested

| Mode | Description |
|---|---|
| `full_duj` | Micro + meso + macro layers active |
| `no_macro` | Micro + meso active, macro disabled |
| `no_meso` | Micro + macro active, meso disabled |
| `no_micro` | Meso + macro active, micro disabled |
| `micro_only` | Only micro active |
| `meso_only` | Only meso active |
| `macro_only` | Only macro active |
| `control` | No intervention |

## Summary Results

| Mode | Mean Deaths | Max Deaths | Mean Corrections | Avg Instability | Max Instability | Micro Triggers | Meso Triggers | Macro Triggers | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_duj` | 0.00 | 0 | 76,416 | 0.1812 | 0.4993 | 47,208 | 16,384 | 12,823 | 1.56 |
| `no_macro` | 414.50 | 456 | 80,995 | 0.2241 | 0.9932 | 53,394 | 27,601 | 0 | 1.24 |
| `no_meso` | 0.00 | 0 | 67,872 | 0.1896 | 0.4990 | 48,618 | 0 | 19,254 | 1.29 |
| `no_micro` | 0.00 | 0 | 36,643 | 0.2061 | 0.4992 | 0 | 21,756 | 14,887 | 1.33 |
| `micro_only` | 2,140.50 | 2,204 | 51,515 | 0.2986 | 0.9965 | 51,515 | 0 | 0 | 0.80 |
| `meso_only` | 870.40 | 915 | 36,523 | 0.2559 | 0.9941 | 0 | 36,523 | 0 | 0.97 |
| `macro_only` | 0.00 | 0 | 23,179 | 0.2245 | 0.4993 | 0 | 0 | 23,179 | 1.04 |
| `control` | 4,063.90 | 4,103 | 0 | 0.4118 | 0.9926 | 0 | 0 | 0 | 0.43 |

## Key Finding

When both micro and meso intervention were delayed under stronger shocks, the macro layer became meaningfully active and contributed to viability.

The clearest comparison:

| Mode | Mean Deaths | Macro Triggers |
|---|---:|---:|
| `full_duj` | 0.00 | 12,823 |
| `no_macro` | 414.50 | 0 |
| `macro_only` | 0.00 | 23,179 |
| `control` | 4,063.90 | 0 |

This indicates that macro-level correction is not decorative. Under this stress regime, it functions as an emergency stabilizer.

## Interpretation

The experiment supports a layered coordination story:

1. Under mild stress, the micro layer is sufficient.
2. When micro intervention is delayed, meso-level correction becomes useful.
3. When both micro and meso intervention are slowed under stronger shocks, macro-level correction becomes necessary for zero-death stability.

In this regime, `full_duj` achieved zero deaths, while `no_macro` produced an average of 414.5 deaths. This suggests that higher-level intervention can prevent failures when local and mid-level regulation are delayed or insufficient.

## Scientist Interpretation

- Macro became meaningfully active under delayed micro/meso intervention.
- Full DUJ outperformed no-macro, suggesting macro contributed to viability.
- Macro-only improved survival relative to control, suggesting emergency correction has standalone value.
- Control collapsed more severely than full DUJ, confirming intervention improved viability.
- This regime tests whether global correction becomes necessary when lower layers are slowed.

## Limitations

- This is still a simulation, not a deployed multi-agent robotic or networked system.
- The stress profile is synthetic and should be tested across additional perturbation distributions.
- The macro layer currently acts immediately, while micro and meso are delayed by fixed cooldowns.
- Correction strengths and thresholds are hand-set rather than learned or formally optimized.
- Results should be repeated with larger seed counts and larger agent populations.

## Next Experiments

- Sweep shock magnitude to find the macro activation threshold.
- Sweep micro and meso cooldown values.
- Compare full DUJ against macro-only under multiple stress regimes.
- Measure correction efficiency versus survival more formally.
- Run the same experiment at larger agent counts.
- Add visual plots for deaths, correction budget, and layer activation share.

## Conclusion

V41 provides evidence that DUJ-style layered coordination exhibits regime-dependent behavior.

The macro layer was unnecessary in mild conditions, but became important when lower layers were delayed under stronger shocks. This supports the hypothesis that layered coordination can provide different stabilization roles across stress regimes rather than merely adding redundant correction overhead.
