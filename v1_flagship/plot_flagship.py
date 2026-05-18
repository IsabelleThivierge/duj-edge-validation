import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("v1_flagship/results/v31c_summary.csv")

# Clean names for plotting
label_map = {
    "control": "Control",
    "continuous_supervision": "Continuous",
    "event_gated_no_cooccurrence": "Event Gated\n(No Co-occurrence)",
    "event_gated_duj": "DUJ",
}

df["label"] = df["mode"].map(label_map)

# ---------- Figure 1: Correction Budget ----------
plt.figure(figsize=(8, 5))
plt.bar(df["label"], df["correction_budget_mean"])
plt.ylabel("Correction Budget")
plt.title("Correction Budget by Mode")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("v1_flagship/figures/flagship_budget.png")
plt.close()

# ---------- Figure 2: Failure Rate ----------
plt.figure(figsize=(8, 5))
plt.bar(df["label"], df["failure_rate_mean"])
plt.ylabel("Failure Rate")
plt.title("Failure Rate by Mode")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("v1_flagship/figures/flagship_failure_rate.png")
plt.close()

# ---------- Figure 3: Latency ----------
plt.figure(figsize=(8, 5))
plt.bar(df["label"], df["latency_ms_mean"])
plt.ylabel("Latency (ms)")
plt.title("Latency by Mode")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("v1_flagship/figures/flagship_latency.png")
plt.close()

print("Done.")
print("Saved:")
print(" - v1_flagship/figures/flagship_budget.png")
print(" - v1_flagship/figures/flagship_failure_rate.png")
print(" - v1_flagship/figures/flagship_latency.png")
