import random
import statistics
import time

NUM_AGENTS = 5000
NUM_STEPS = 120
SEEDS = 10

BASE_DRIFT = 0.03
SHOCK_PROBABILITY = 0.10
SHOCK_MAGNITUDE = 0.75

FAILURE_THRESHOLD = 1.0

MICRO_COOLDOWN_STEPS = 5
MESO_COOLDOWN_STEPS = 3


class Agent:
    def __init__(self):
        self.state = random.uniform(-0.1, 0.1)
        self.alive = True
        self.corrections = 0
        self.micro_cooldown = 0
        self.meso_cooldown = 0

    def step(self):
        drift = random.uniform(-BASE_DRIFT, BASE_DRIFT)

        if random.random() < SHOCK_PROBABILITY:
            drift += random.uniform(-SHOCK_MAGNITUDE, SHOCK_MAGNITUDE)

        self.state += drift

        if self.micro_cooldown > 0:
            self.micro_cooldown -= 1

        if self.meso_cooldown > 0:
            self.meso_cooldown -= 1


class DUJLayer:
    def __init__(self, name, threshold, correction_strength):
        self.name = name
        self.threshold = threshold
        self.correction_strength = correction_strength
        self.triggers = 0

    def intervene(self, agent):
        if abs(agent.state) > self.threshold:
            self.triggers += 1
            agent.corrections += 1
            agent.state *= self.correction_strength


def make_layers(mode):
    all_layers = {
        "micro": DUJLayer("micro", 0.20, 0.75),
        "meso": DUJLayer("meso", 0.35, 0.55),
        "macro": DUJLayer("macro", 0.50, 0.30),
    }

    if mode == "full_duj":
        active = ["micro", "meso", "macro"]
    elif mode == "micro_only":
        active = ["micro"]
    elif mode == "meso_only":
        active = ["meso"]
    elif mode == "macro_only":
        active = ["macro"]
    elif mode == "no_macro":
        active = ["micro", "meso"]
    elif mode == "no_meso":
        active = ["micro", "macro"]
    elif mode == "no_micro":
        active = ["meso", "macro"]
    elif mode == "control":
        active = []
    else:
        raise ValueError(mode)

    return [all_layers[name] for name in active], all_layers


def run_trial(mode, seed):
    random.seed(seed)

    layers, all_layers = make_layers(mode)
    agents = [Agent() for _ in range(NUM_AGENTS)]

    start = time.time()

    for _ in range(NUM_STEPS):
        for agent in agents:
            if not agent.alive:
                continue

            agent.step()

            for layer in layers:
                if layer.name == "micro":
                    if agent.micro_cooldown == 0:
                        layer.intervene(agent)
                        agent.micro_cooldown = MICRO_COOLDOWN_STEPS

                elif layer.name == "meso":
                    if agent.meso_cooldown == 0:
                        layer.intervene(agent)
                        agent.meso_cooldown = MESO_COOLDOWN_STEPS

                else:
                    # macro acts immediately
                    layer.intervene(agent)

            if abs(agent.state) > FAILURE_THRESHOLD:
                agent.alive = False

    runtime = time.time() - start

    alive = sum(a.alive for a in agents)
    deaths = NUM_AGENTS - alive
    corrections = sum(a.corrections for a in agents)

    alive_states = [abs(a.state) for a in agents if a.alive]

    avg_instability = statistics.mean(alive_states) if alive_states else FAILURE_THRESHOLD
    max_instability = max(alive_states) if alive_states else FAILURE_THRESHOLD

    return {
        "deaths": deaths,
        "corrections": corrections,
        "avg_instability": avg_instability,
        "max_instability": max_instability,
        "runtime": runtime,
        "micro": all_layers["micro"].triggers,
        "meso": all_layers["meso"].triggers,
        "macro": all_layers["macro"].triggers,
    }


def summarize(results):
    def mean(key):
        return statistics.mean(r[key] for r in results)

    def maxv(key):
        return max(r[key] for r in results)

    return {
        "death_mean": mean("deaths"),
        "death_max": maxv("deaths"),
        "correction_mean": mean("corrections"),
        "avg_instability": mean("avg_instability"),
        "max_instability": mean("max_instability"),
        "runtime": mean("runtime"),
        "micro": mean("micro"),
        "meso": mean("meso"),
        "macro": mean("macro"),
    }


def print_summary(all_results):
    print("=" * 118)
    print("SUMMARY")
    print("=" * 118)

    header = (
        f"{'Mode':<14}"
        f"{'Deaths':>10}"
        f"{'MaxD':>8}"
        f"{'Corr':>12}"
        f"{'AvgInst':>12}"
        f"{'MaxInst':>12}"
        f"{'Micro':>10}"
        f"{'Meso':>10}"
        f"{'Macro':>10}"
        f"{'Runtime':>10}"
    )

    print(header)
    print("-" * 118)

    for mode, s in all_results.items():
        print(
            f"{mode:<14}"
            f"{s['death_mean']:>10.2f}"
            f"{s['death_max']:>8.0f}"
            f"{s['correction_mean']:>12.0f}"
            f"{s['avg_instability']:>12.4f}"
            f"{s['max_instability']:>12.4f}"
            f"{s['micro']:>10.0f}"
            f"{s['meso']:>10.0f}"
            f"{s['macro']:>10.0f}"
            f"{s['runtime']:>10.2f}"
        )


def scientist_interpretation(all_results):
    full = all_results["full_duj"]
    no_macro = all_results["no_macro"]
    macro_only = all_results["macro_only"]
    control = all_results["control"]

    print("\n" + "=" * 118)
    print("SCIENTIST INTERPRETATION")
    print("=" * 118)

    if full["macro"] > 100:
        print("- Macro became meaningfully active under delayed micro/meso intervention.")

    if full["death_mean"] < no_macro["death_mean"]:
        print("- Full DUJ outperformed no-macro, suggesting macro contributed to viability.")

    if macro_only["death_mean"] < control["death_mean"]:
        print("- Macro-only improved survival relative to control, suggesting emergency correction has standalone value.")

    if control["death_mean"] > full["death_mean"]:
        print("- Control collapsed more severely than full DUJ, confirming intervention improved viability.")

    print("- This regime tests whether global correction becomes necessary when lower layers are slowed.")


def main():
    modes = [
        "full_duj",
        "no_macro",
        "no_meso",
        "no_micro",
        "micro_only",
        "meso_only",
        "macro_only",
        "control",
    ]

    print("\nRunning macro activation test...\n")
    print(f"Agents: {NUM_AGENTS}")
    print(f"Steps: {NUM_STEPS}")
    print(f"Seeds: {SEEDS}")
    print(f"Shock probability: {SHOCK_PROBABILITY}")
    print(f"Shock magnitude: {SHOCK_MAGNITUDE}")
    print(f"Micro cooldown: {MICRO_COOLDOWN_STEPS} steps")
    print(f"Meso cooldown: {MESO_COOLDOWN_STEPS} steps\n")

    all_results = {}

    for mode in modes:
        mode_results = []

        for seed in range(SEEDS):
            mode_results.append(run_trial(mode, seed))

        all_results[mode] = summarize(mode_results)

    print_summary(all_results)
    scientist_interpretation(all_results)


if __name__ == "__main__":
    main()
