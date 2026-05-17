# duj_bridge_v31c_adversarial_signal_noise.py
# DUJ V31C — Adversarial Signal Noise
#
# Stronger than V31B:
#   Noise attacks regime inference directly.
#
# Adds:
#   - false regime probability
#   - sensing delay
#   - regret estimate noise
#
# Variants:
#   FIELD_DUJ
#   FULLER_DUJ
#   PRIORITY_DUJ
#   CONTINUOUS

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


SIGNAL_LEVELS = (
    {"false_prob": 0.00, "delay": 0, "regret_noise": 0.00},
    {"false_prob": 0.10, "delay": 2, "regret_noise": 0.03},
    {"false_prob": 0.20, "delay": 5, "regret_noise": 0.06},
    {"false_prob": 0.35, "delay": 8, "regret_noise": 0.10},
    {"false_prob": 0.50, "delay": 12, "regret_noise": 0.15},
)


@dataclass
class Config:
    seed: int = 0
    variant: str = "FIELD_DUJ"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    base_agents: int = 256
    reserve_capacity: int = 128
    steps: int = 520
    noise: float = 0.12

    false_signal_prob: float = 0.0
    sensing_delay: int = 0
    regret_noise: float = 0.0

    damage_step: int = 50
    damage_fraction: float = 0.42

    min_segment_len: int = 18
    max_segment_len: int = 55

    experience_start: int = 36
    temporal_drift_threshold: int = 4
    morph_error_threshold: int = 18
    stress_threshold: float = 0.72
    regret_threshold: float = 0.08
    viability_threshold: float = 0.10

    new_arm_step: int = 4
    repair_step: int = 4

    confidence_decay: float = 0.018
    drift_growth: float = 0.035

    event_correction_strength: float = 0.62
    continuous_correction_strength: float = 0.075

    low_conf_threshold: float = 0.18
    high_drift_threshold: float = 0.78
    death_conf_threshold: float = 0.08
    death_drift_threshold: float = 1.25

    max_growth_per_event: int = 12
    max_conversion_per_event: int = 12
    max_prune_per_event: int = 12

    confidence_commit_threshold: float = 0.58

    old_target: Dict[str, int] = field(default_factory=lambda: {
        "vision": 32,
        "navigation": 32,
        "planner": 32,
        "manipulator": 32,
        "sensor": 32,
        "new_arm": 64,
        "repair": 32,
    })

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass
class AgentState:
    role: List[Optional[str]]
    alive: torch.Tensor
    confidence: torch.Tensor
    drift: torch.Tensor
    slot_used: torch.Tensor
    generation: torch.Tensor


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_from_values(cfg: Config, new_arm: int, repair: int) -> Dict[str, int]:
    t = dict(cfg.old_target)
    t["new_arm"] = new_arm
    t["repair"] = repair
    return t


def phase_target(cfg: Config, phase: str) -> Dict[str, int]:
    if phase == "A_PERFORMANCE":
        return target_from_values(cfg, 80, 24)
    if phase == "B_RESILIENCE":
        return target_from_values(cfg, 56, 40)
    return target_from_values(cfg, 68, 32)


def make_hidden_schedule(cfg: Config, seed: int) -> Tuple[Tuple[int, int, str], ...]:
    rng = random.Random(10_000 + seed)
    phases = ["A_PERFORMANCE", "B_RESILIENCE", "C_COMPROMISE"]
    schedule = []
    step = 0
    last = None

    while step < cfg.steps:
        choices = [p for p in phases if p != last]
        phase = rng.choice(choices)
        dur = rng.randint(cfg.min_segment_len, cfg.max_segment_len)
        end = min(cfg.steps, step + dur)
        schedule.append((step, end, phase))
        step = end
        last = phase

    return tuple(schedule)


def role_counts(state: AgentState, roles: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in roles}
    for i, r in enumerate(state.role):
        if r is not None and bool(state.alive[i]):
            counts[r] += 1
    return counts


def morph_error(counts: Dict[str, int], target: Dict[str, int]) -> int:
    return int(sum(abs(counts.get(k, 0) - v) for k, v in target.items()))


def counts_distance(counts: Dict[str, int], target: Dict[str, int]) -> float:
    return float(sum(abs(counts.get(k, 0) - v) for k, v in target.items()))


def target_distance(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a.keys()).union(set(b.keys()))
    return float(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))


def alive_indices(state: AgentState) -> List[int]:
    return [i for i in range(len(state.role)) if bool(state.alive[i])]


def alive_indices_for_role(state: AgentState, role: str) -> List[int]:
    return [i for i, r in enumerate(state.role) if r == role and bool(state.alive[i])]


def first_free_slots(state: AgentState, n: int) -> List[int]:
    out = []
    for i in range(len(state.role)):
        if not bool(state.slot_used[i]):
            out.append(i)
            if len(out) >= n:
                break
    return out


class DUJV31C:
    def __init__(self, cfg: Config, schedule: Tuple[Tuple[int, int, str], ...]):
        self.cfg = cfg
        self.variant = cfg.variant
        self.schedule = schedule
        self.mode = "continuous" if cfg.variant == "CONTINUOUS" else "event"
        self.device = torch.device(cfg.device)

        set_seed(cfg.seed)

        self.rng = random.Random(50_000 + cfg.seed)

        self.roles = list(cfg.old_target.keys())
        self.total_slots = cfg.base_agents + cfg.reserve_capacity
        self.state = self._init_state()
        self.current_target = dict(cfg.old_target)

        self.regime_conf = {
            "A_PERFORMANCE": 0.34,
            "B_RESILIENCE": 0.33,
            "C_COMPROMISE": 0.33,
        }

        self.last_phase = self.true_phase(0)
        self.steps_since_phase_change = 0

        self.micro_drift_ema = 0.0
        self.meso_drift_ema = 0.0
        self.macro_drift_ema = 0.0

        self.history: List[Dict] = []
        self.edit_history: List[Dict] = []

        self.corrections = 0
        self.deaths = 0
        self.spawned = 0
        self.pruned = 0

        self.viability_events = 0
        self.temporal_events = 0
        self.morph_events = 0
        self.co_occurrence_events = 0
        self.false_signal_events = 0
        self.hold_events = 0

        self.edits_up = 0
        self.edits_down = 0
        self.edits_hold = 0
        self.reconciliations = 0

    def _init_state(self) -> AgentState:
        role: List[Optional[str]] = [None for _ in range(self.total_slots)]

        idx = 0
        for r, n in self.cfg.old_target.items():
            for _ in range(n):
                role[idx] = r
                idx += 1

        alive = torch.zeros(self.total_slots, dtype=torch.bool, device=self.device)
        alive[:self.cfg.base_agents] = True

        slot_used = torch.zeros(self.total_slots, dtype=torch.bool, device=self.device)
        slot_used[:self.cfg.base_agents] = True

        confidence = torch.zeros(self.total_slots, device=self.device)
        drift = torch.zeros(self.total_slots, device=self.device)

        confidence[:self.cfg.base_agents] = 0.82 + 0.12 * torch.rand(
            self.cfg.base_agents, device=self.device
        )
        drift[:self.cfg.base_agents] = 0.08 + 0.05 * torch.rand(
            self.cfg.base_agents, device=self.device
        )

        generation = torch.zeros(self.total_slots, dtype=torch.long, device=self.device)
        return AgentState(role, alive, confidence, drift, slot_used, generation)

    def true_phase(self, step: int) -> str:
        for start, end, name in self.schedule:
            if start <= step < end:
                return name
        return self.schedule[-1][2]

    def perceived_phase(self, step: int) -> str:
        delayed_step = max(0, step - self.cfg.sensing_delay)
        true_delayed = self.true_phase(delayed_step)

        if self.mode == "continuous":
            return true_delayed

        if self.rng.random() < self.cfg.false_signal_prob:
            self.false_signal_events += 1
            phases = ["A_PERFORMANCE", "B_RESILIENCE", "C_COMPROMISE"]
            choices = [p for p in phases if p != true_delayed]
            return self.rng.choice(choices)

        return true_delayed

    def active_optimum(self, step: int) -> Dict[str, int]:
        # Real scoring uses true phase.
        return phase_target(self.cfg, self.true_phase(step))

    def perceived_optimum(self, step: int) -> Dict[str, int]:
        # Runtime adaptation uses noisy/stale perceived phase.
        return phase_target(self.cfg, self.perceived_phase(step))

    def transition_steps(self) -> List[int]:
        return [start for start, _, _ in self.schedule if start > 0]

    def is_fuller(self) -> bool:
        return self.variant == "FULLER_DUJ"

    def update_regime_confidence(self, step: int):
        if not self.is_fuller():
            return

        current = self.perceived_phase(step)

        if current != self.last_phase:
            self.steps_since_phase_change = 0
            self.last_phase = current
        else:
            self.steps_since_phase_change += 1

        rise = 0.045
        if self.steps_since_phase_change > 18:
            rise = 0.070
        if self.steps_since_phase_change > 36:
            rise = 0.095

        decay = 0.030

        for k in self.regime_conf:
            if k == current:
                self.regime_conf[k] += rise * (1.0 - self.regime_conf[k])
            else:
                self.regime_conf[k] *= (1.0 - decay)

        total = sum(self.regime_conf.values())
        for k in self.regime_conf:
            self.regime_conf[k] /= max(total, 1e-9)

    def interpreted_optimum(self, step: int) -> Dict[str, int]:
        if not self.is_fuller():
            return self.perceived_optimum(step)

        current = self.perceived_phase(step)
        current_conf = self.regime_conf[current]
        opt = self.perceived_optimum(step)

        if current_conf >= self.cfg.confidence_commit_threshold:
            return opt

        blend = current_conf / self.cfg.confidence_commit_threshold
        target = dict(self.cfg.old_target)
        target["new_arm"] = int(round(
            (blend * opt["new_arm"] + (1.0 - blend) * self.current_target["new_arm"]) / 4.0
        ) * 4)
        target["repair"] = int(round(
            (blend * opt["repair"] + (1.0 - blend) * self.current_target["repair"]) / 4.0
        ) * 4)
        return target

    def update_polytemporal_drift(self, step: int):
        counts = role_counts(self.state, self.roles)
        perceived = self.perceived_optimum(step)
        interpreted = self.interpreted_optimum(step)

        micro = counts_distance(counts, self.current_target)
        meso = counts_distance(counts, perceived)
        macro = target_distance(self.current_target, interpreted)

        self.micro_drift_ema = 0.65 * self.micro_drift_ema + 0.35 * micro
        self.meso_drift_ema = 0.82 * self.meso_drift_ema + 0.18 * meso
        self.macro_drift_ema = 0.92 * self.macro_drift_ema + 0.08 * macro

    def temporal_drift(self, step: int) -> float:
        if self.variant == "CONTINUOUS":
            return 0.0
        if self.is_fuller():
            return 0.35 * self.meso_drift_ema + 0.65 * self.macro_drift_ema
        return target_distance(self.current_target, self.perceived_optimum(step))

    def compute_utility_for_target(self, counts: Dict[str, int], opt: Dict[str, int], phase: str) -> float:
        new_arm = counts.get("new_arm", 0)
        repair = counts.get("repair", 0)
        planner = counts.get("planner", 0)
        sensor = counts.get("sensor", 0)
        manipulator = counts.get("manipulator", 0)
        navigation = counts.get("navigation", 0)

        support_score = min((planner + sensor + manipulator) / 96.0, 1.0)
        navigation_score = min(navigation / 32.0, 1.0)

        arm_score = max(0.0, 1.0 - abs(new_arm - opt["new_arm"]) / 48.0)
        repair_score = max(0.0, 1.0 - abs(repair - opt["repair"]) / 40.0)

        excess_arm = max(0, new_arm - opt["new_arm"])
        repair_deficit = max(0, opt["repair"] - repair)

        excess_arm_penalty = min(excess_arm / 40.0, 1.0)
        repair_deficit_penalty = min(repair_deficit / 32.0, 1.0)

        if phase == "A_PERFORMANCE":
            utility = (
                0.54 * arm_score
                + 0.20 * repair_score
                + 0.22 * support_score
                + 0.04 * navigation_score
                - 0.12 * excess_arm_penalty
            )
        elif phase == "B_RESILIENCE":
            utility = (
                0.30 * arm_score
                + 0.42 * repair_score
                + 0.16 * support_score
                + 0.12 * navigation_score
                - 0.20 * excess_arm_penalty
                - 0.20 * repair_deficit_penalty
            )
        else:
            utility = (
                0.40 * arm_score
                + 0.32 * repair_score
                + 0.20 * support_score
                + 0.08 * navigation_score
                - 0.12 * excess_arm_penalty
                - 0.08 * repair_deficit_penalty
            )

        return float(max(0.0, min(1.0, utility)))

    def compute_true_utility(self, counts: Dict[str, int], step: int) -> float:
        return self.compute_utility_for_target(
            counts, self.active_optimum(step), self.true_phase(step)
        )

    def compute_true_regret(self, counts: Dict[str, int], step: int) -> float:
        actual = self.compute_true_utility(counts, step)
        opt = self.active_optimum(step)

        opt_counts = dict(counts)
        opt_counts["new_arm"] = opt["new_arm"]
        opt_counts["repair"] = opt["repair"]

        best = self.compute_utility_for_target(opt_counts, opt, self.true_phase(step))
        return float(max(0.0, best - actual))

    def estimate_regret(self, counts: Dict[str, int], step: int) -> float:
        perceived = self.perceived_optimum(step)
        phase = self.perceived_phase(step)

        actual = self.compute_utility_for_target(counts, perceived, phase)

        opt_counts = dict(counts)
        opt_counts["new_arm"] = perceived["new_arm"]
        opt_counts["repair"] = perceived["repair"]

        best = self.compute_utility_for_target(opt_counts, perceived, phase)
        regret = max(0.0, best - actual)

        if self.cfg.regret_noise > 0:
            regret += self.rng.gauss(0.0, self.cfg.regret_noise)

        return float(max(0.0, min(1.0, regret)))

    def compute_stress(self, counts: Dict[str, int], step: int) -> float:
        s = self.state
        alive = s.alive
        alive_n = int(alive.sum().item())
        if alive_n == 0:
            return 1.0

        avg_low_conf = float((1.0 - s.confidence[alive].mean()).item())
        avg_drift = float(s.drift[alive].mean().item())
        err = morph_error(counts, self.current_target)

        perceived = self.perceived_optimum(step)
        excess_arm = max(0, counts.get("new_arm", 0) - perceived["new_arm"])
        repair_deficit = max(0, perceived["repair"] - counts.get("repair", 0))
        temporal = self.temporal_drift(step)

        stress = (
            0.20 * avg_low_conf
            + 0.24 * min(avg_drift, 1.0)
            + 0.17 * min(err / 96.0, 1.0)
            + 0.13 * min(excess_arm / 40.0, 1.0)
            + 0.12 * min(repair_deficit / 32.0, 1.0)
            + 0.14 * min(temporal / 32.0, 1.0)
        )
        return float(max(0.0, min(1.0, stress)))

    def viability_risk(self) -> float:
        s = self.state
        alive = s.alive

        if int(alive.sum().item()) == 0:
            return 1.0

        near_death = alive & (
            (s.confidence < self.cfg.low_conf_threshold)
            | (s.drift > self.cfg.high_drift_threshold)
        )

        frac = float(near_death.sum().item()) / max(int(alive.sum().item()), 1)
        avg_low_conf = float((1.0 - s.confidence[alive].mean()).item())
        avg_drift = float(s.drift[alive].mean().item())

        return max(frac, 0.45 * avg_low_conf + 0.55 * min(avg_drift, 1.0))

    def passive_noise(self, step: int):
        s = self.state
        alive = s.alive
        n_alive = int(alive.sum().item())

        if n_alive == 0:
            return

        counts = role_counts(s, self.roles)
        perceived = self.perceived_optimum(step)

        excess_arm = max(0, counts.get("new_arm", 0) - perceived["new_arm"])
        repair_deficit = max(0, perceived["repair"] - counts.get("repair", 0))

        instability = 1.0
        instability += 0.60 * min(excess_arm / 40.0, 1.0)
        instability += 0.55 * min(repair_deficit / 32.0, 1.0)

        s.confidence[alive] -= self.cfg.confidence_decay * torch.rand(
            n_alive, device=self.device
        )
        s.drift[alive] += self.cfg.drift_growth * instability * torch.rand(
            n_alive, device=self.device
        )

        s.confidence[alive] += 0.015 * self.cfg.noise * 8.0 * torch.randn(
            n_alive, device=self.device
        )
        s.drift[alive] += self.cfg.noise * 0.035 * instability * torch.randn(
            n_alive, device=self.device
        )

        s.confidence.clamp_(0.0, 1.0)
        s.drift.clamp_(0.0, 2.0)

    def apply_initial_damage(self):
        s = self.state
        alive = alive_indices(s)
        k = int(len(alive) * self.cfg.damage_fraction)
        damaged = random.sample(alive, k)

        for i in damaged:
            s.confidence[i] *= 0.28
            s.drift[i] += 0.72

    def apply_deaths(self):
        s = self.state
        alive = s.alive

        death_mask = (
            alive
            & (s.confidence < self.cfg.death_conf_threshold)
            & (s.drift > self.cfg.death_drift_threshold)
        )

        count = int(death_mask.sum().item())
        if count > 0:
            s.alive[death_mask] = False
            self.deaths += count

    def continuous_correction(self):
        s = self.state
        alive = s.alive
        n = int(alive.sum().item())

        if n == 0:
            return

        strength = self.cfg.continuous_correction_strength
        s.confidence[alive] += strength * (1.0 - s.confidence[alive])
        s.drift[alive] *= (1.0 - strength)
        self.corrections += n

    def correct_viability(self):
        s = self.state
        alive = s.alive

        stressed = alive & (
            (s.confidence < self.cfg.low_conf_threshold)
            | (s.drift > self.cfg.high_drift_threshold)
        )

        n = int(stressed.sum().item())
        if n == 0:
            return False

        strength = self.cfg.event_correction_strength
        s.confidence[stressed] += strength * (1.0 - s.confidence[stressed])
        s.drift[stressed] *= (1.0 - strength)

        self.corrections += n
        self.viability_events += 1
        return True

    def reconcile_target_toward(self, opt: Dict[str, int], step: int, reason: str):
        old = dict(self.current_target)
        changed = False
        direction = "HOLD"

        ca = self.current_target["new_arm"]
        cr = self.current_target["repair"]
        ta = opt["new_arm"]
        tr = opt["repair"]

        if ca < ta:
            self.current_target["new_arm"] = min(ca + self.cfg.new_arm_step, ta)
            if cr > tr:
                self.current_target["repair"] = max(cr - self.cfg.repair_step, tr)
            direction = "UP"
            self.edits_up += 1
            changed = True

        elif ca > ta:
            self.current_target["new_arm"] = max(ca - self.cfg.new_arm_step, ta)
            if cr < tr:
                self.current_target["repair"] = min(cr + self.cfg.repair_step, tr)
            direction = "DOWN"
            self.edits_down += 1
            changed = True

        else:
            if cr < tr:
                self.current_target["repair"] = min(cr + self.cfg.repair_step, tr)
                direction = "REPAIR_UP"
                changed = True
            elif cr > tr:
                self.current_target["repair"] = max(cr - self.cfg.repair_step, tr)
                direction = "REPAIR_DOWN"
                changed = True

        if not changed:
            self.edits_hold += 1
            return False

        self.reconciliations += 1

        if reason == "temporal":
            self.temporal_events += 1
        elif reason == "morph":
            self.morph_events += 1

        counts = role_counts(self.state, self.roles)
        self.edit_history.append({
            "step": step,
            "true_phase": self.true_phase(step),
            "perceived_phase": self.perceived_phase(step),
            "reason": reason,
            "direction": direction,
            "old_target": old,
            "new_target": dict(self.current_target),
            "optimum": dict(opt),
            "true_utility": self.compute_true_utility(counts, step),
            "true_regret": self.compute_true_regret(counts, step),
        })

        return True

    def priority_runtime(self, step: int):
        counts = role_counts(self.state, self.roles)
        v = self.viability_risk()
        t = self.temporal_drift(step)
        m = morph_error(counts, self.current_target)
        regret_est = self.estimate_regret(counts, step)
        stress = self.compute_stress(counts, step)

        if v > self.cfg.viability_threshold or stress > self.cfg.stress_threshold:
            self.correct_viability()
            self.express_structure()
            return

        if step >= self.cfg.experience_start and (
            t >= self.cfg.temporal_drift_threshold
            or regret_est >= self.cfg.regret_threshold
        ):
            self.reconcile_target_toward(self.perceived_optimum(step), step, "temporal")
            self.express_structure()
            return

        if m >= self.cfg.morph_error_threshold:
            self.reconcile_target_toward(self.current_target, step, "morph")
            self.express_structure()
            return

        self.hold_events += 1

    def field_runtime(self, step: int):
        counts = role_counts(self.state, self.roles)
        v = self.viability_risk()
        t = self.temporal_drift(step)
        regret_est = self.estimate_regret(counts, step)
        stress = self.compute_stress(counts, step)

        fired = 0

        if v > self.cfg.viability_threshold or stress > self.cfg.stress_threshold:
            if self.correct_viability():
                fired += 1

        if step >= self.cfg.experience_start and (
            t >= self.cfg.temporal_drift_threshold
            or regret_est >= self.cfg.regret_threshold
        ):
            if self.reconcile_target_toward(self.perceived_optimum(step), step, "temporal"):
                fired += 1

        self.express_structure()

        counts_after = role_counts(self.state, self.roles)
        if morph_error(counts_after, self.current_target) >= self.cfg.morph_error_threshold:
            if self.reconcile_target_toward(self.current_target, step, "morph"):
                fired += 1
            self.express_structure()

        if fired >= 2:
            self.co_occurrence_events += 1
        elif fired == 0:
            self.hold_events += 1

    def fuller_runtime(self, step: int):
        counts = role_counts(self.state, self.roles)
        err = morph_error(counts, self.current_target)
        stress = self.compute_stress(counts, step)
        regret_est = self.estimate_regret(counts, step)
        temporal = self.temporal_drift(step)

        s = self.state
        alive = s.alive

        low_conf = bool((s.confidence[alive] < self.cfg.low_conf_threshold).any().item())
        high_drift = bool((s.drift[alive] > self.cfg.high_drift_threshold).any().item())

        trigger = (
            low_conf
            or high_drift
            or err > self.cfg.morph_error_threshold
            or stress > self.cfg.stress_threshold
            or regret_est > self.cfg.regret_threshold
            or temporal >= self.cfg.temporal_drift_threshold
        )

        if trigger:
            self.correct_viability()
            self.express_structure()

        if step >= self.cfg.experience_start and step % 18 == 0:
            opt = self.interpreted_optimum(step)
            drift = target_distance(self.current_target, opt)

            if drift >= self.cfg.temporal_drift_threshold or regret_est >= self.cfg.regret_threshold:
                self.reconcile_target_toward(opt, step, "temporal")
                self.express_structure()

    def express_structure(self):
        self.repair_or_convert()
        self.grow_missing_roles()
        self.prune_surplus_roles()

    def repair_or_convert(self):
        s = self.state
        counts = role_counts(s, self.roles)
        target = self.current_target

        surplus_roles = []
        missing_roles = []

        for r in self.roles:
            delta = counts[r] - target[r]
            if delta > 0:
                surplus_roles.append((r, delta))
            elif delta < 0:
                missing_roles.append((r, -delta))

        if not surplus_roles or not missing_roles:
            return

        conversions = 0

        for missing_role, missing_n in missing_roles:
            if conversions >= self.cfg.max_conversion_per_event:
                break

            for surplus_role, surplus_n in list(surplus_roles):
                if conversions >= self.cfg.max_conversion_per_event:
                    break

                if surplus_n <= 0:
                    continue

                candidates = alive_indices_for_role(s, surplus_role)
                if not candidates:
                    continue

                convert_n = min(
                    missing_n,
                    surplus_n,
                    len(candidates),
                    self.cfg.max_conversion_per_event - conversions,
                )

                chosen = random.sample(candidates, convert_n)

                for idx in chosen:
                    s.role[idx] = missing_role
                    s.confidence[idx] = max(float(s.confidence[idx].item()), 0.52)
                    s.drift[idx] *= 0.55

                conversions += convert_n
                missing_n -= convert_n
                surplus_n -= convert_n

                if missing_n <= 0:
                    break

    def grow_missing_roles(self):
        s = self.state
        counts = role_counts(s, self.roles)
        target = self.current_target

        missing = []
        for r in self.roles:
            delta = target[r] - counts[r]
            if delta > 0:
                missing.append((r, delta))

        if not missing:
            return

        budget = self.cfg.max_growth_per_event

        for role_name, need in missing:
            if budget <= 0:
                break

            n = min(need, budget)
            slots = first_free_slots(s, n)

            if not slots:
                return

            for idx in slots:
                s.role[idx] = role_name
                s.alive[idx] = True
                s.slot_used[idx] = True
                s.confidence[idx] = 0.58 + 0.12 * random.random()
                s.drift[idx] = 0.22 + 0.08 * random.random()
                s.generation[idx] += 1

            self.spawned += len(slots)
            budget -= len(slots)

    def prune_surplus_roles(self):
        s = self.state
        counts = role_counts(s, self.roles)
        target = self.current_target

        for role_name in ("new_arm", "repair"):
            surplus = counts.get(role_name, 0) - target.get(role_name, 0)

            if surplus <= 0:
                continue

            candidates = alive_indices_for_role(s, role_name)
            if not candidates:
                continue

            candidates_sorted = sorted(
                candidates,
                key=lambda idx: float(s.drift[idx].item()),
                reverse=True,
            )
            prune_n = min(surplus, self.cfg.max_prune_per_event)

            for idx in candidates_sorted[:prune_n]:
                s.alive[idx] = False
                s.confidence[idx] = 0.0
                s.drift[idx] = 1.0
                self.pruned += 1

    def log(self, step: int):
        counts = role_counts(self.state, self.roles)
        true_opt = self.active_optimum(step)
        perceived_opt = self.perceived_optimum(step)

        self.history.append({
            "step": step,
            "true_phase": self.true_phase(step),
            "perceived_phase": self.perceived_phase(step),
            "true_utility": self.compute_true_utility(counts, step),
            "true_regret": self.compute_true_regret(counts, step),
            "regret_est": self.estimate_regret(counts, step),
            "stress": self.compute_stress(counts, step),
            "target_new_arm": self.current_target["new_arm"],
            "target_repair": self.current_target["repair"],
            "count_new_arm": counts.get("new_arm", 0),
            "count_repair": counts.get("repair", 0),
            "true_opt_new_arm": true_opt["new_arm"],
            "true_opt_repair": true_opt["repair"],
            "perceived_opt_new_arm": perceived_opt["new_arm"],
            "perceived_opt_repair": perceived_opt["repair"],
            "distance_to_true_optimum": counts_distance(counts, true_opt),
            "morph_error": morph_error(counts, self.current_target),
            "alive": int(self.state.alive.sum().item()),
        })

    def step(self, step: int):
        if step == self.cfg.damage_step:
            self.apply_initial_damage()

        self.update_regime_confidence(step)
        self.passive_noise(step)
        self.update_polytemporal_drift(step)

        if self.mode == "continuous":
            self.continuous_correction()
        elif self.variant == "FIELD_DUJ":
            self.field_runtime(step)
        elif self.variant == "PRIORITY_DUJ":
            self.priority_runtime(step)
        elif self.variant == "FULLER_DUJ":
            self.fuller_runtime(step)
        else:
            self.field_runtime(step)

        self.apply_deaths()
        self.log(step)

    def run(self):
        for step in range(self.cfg.steps):
            self.step(step)
        return self.summary()

    def phase_segment_success(self) -> Tuple[int, int]:
        successes = 0
        total = 0

        for start, end, phase_name in self.schedule:
            segment = [h for h in self.history if start <= h["step"] < end]
            if not segment:
                continue

            final = segment[-1]
            expected = phase_target(self.cfg, phase_name)

            ok = (
                final["target_new_arm"] == expected["new_arm"]
                and final["target_repair"] == expected["repair"]
                and final["count_new_arm"] == expected["new_arm"]
                and final["count_repair"] == expected["repair"]
            )

            successes += int(ok)
            total += 1

        return successes, total

    def recovery_latency_metrics(self) -> Dict[str, float]:
        latencies = []

        for t in self.transition_steps():
            recovered = None

            for h in self.history:
                if h["step"] < t:
                    continue

                if h["true_utility"] >= 0.95 and h["true_regret"] <= 0.05:
                    recovered = h["step"] - t
                    break

            latencies.append(recovered if recovered is not None else float("inf"))

        finite = [x for x in latencies if x != float("inf")]
        return {
            "successes": len(finite),
            "total": len(latencies),
            "avg_latency": sum(finite) / len(finite) if finite else float("inf"),
        }

    def overshoot_score(self) -> float:
        total = 0.0

        for h in self.history:
            arm = max(0, abs(h["count_new_arm"] - h["true_opt_new_arm"]) - 4)
            repair = max(0, abs(h["count_repair"] - h["true_opt_repair"]) - 4)
            total += arm + repair

        return total / max(len(self.history), 1)

    def regret_auc(self) -> float:
        return sum(h["true_regret"] for h in self.history)

    def avg_d_opt(self) -> float:
        return sum(h["distance_to_true_optimum"] for h in self.history) / max(len(self.history), 1)

    def thrash_score(self) -> int:
        dirs = [
            e["direction"]
            for e in self.edit_history
            if e["direction"] in ("UP", "DOWN")
        ]
        return sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)

    def summary(self) -> Dict:
        counts = role_counts(self.state, self.roles)
        final_step = self.cfg.steps - 1
        final_opt = self.active_optimum(final_step)

        segment_successes, segment_total = self.phase_segment_success()
        rec = self.recovery_latency_metrics()

        utility = self.compute_true_utility(counts, final_step)
        regret = self.compute_true_regret(counts, final_step)
        d_final = counts_distance(counts, final_opt)
        err = morph_error(counts, self.current_target)

        return {
            "variant": self.variant,
            "seed": self.cfg.seed,
            "false_prob": self.cfg.false_signal_prob,
            "delay": self.cfg.sensing_delay,
            "regret_noise": self.cfg.regret_noise,
            "segments": segment_successes,
            "segment_total": segment_total,
            "seg_rate": segment_successes / max(segment_total, 1),
            "recovery": rec["successes"],
            "recovery_total": rec["total"],
            "latency": rec["avg_latency"],
            "final_success": d_final == 0 and utility >= 0.98 and regret <= 0.02,
            "stable": err == 0 and self.deaths == 0,
            "alive": int(self.state.alive.sum().item()),
            "d_final": d_final,
            "utility": utility,
            "regret": regret,
            "regret_auc": self.regret_auc(),
            "avg_d_opt": self.avg_d_opt(),
            "overshoot": self.overshoot_score(),
            "thrash": self.thrash_score(),
            "corr": self.corrections,
            "deaths": self.deaths,
            "spawned": self.spawned,
            "pruned": self.pruned,
            "V": self.viability_events,
            "T": self.temporal_events,
            "M": self.morph_events,
            "CO": self.co_occurrence_events,
            "false_events": self.false_signal_events,
        }


def safe_fmt(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def aggregate(level_idx: int, level: Dict, variant: str, results: List[Dict]) -> Dict:
    def avg(k):
        vals = [float(r[k]) for r in results]
        finite = [v for v in vals if v != float("inf")]
        return sum(finite) / len(finite) if finite else float("inf")

    out = {
        "level": level_idx,
        "false_prob": level["false_prob"],
        "delay": level["delay"],
        "regret_noise": level["regret_noise"],
        "variant": variant,
        "segments": sum(r["segments"] for r in results),
        "segment_total": sum(r["segment_total"] for r in results),
        "recovery": sum(r["recovery"] for r in results),
        "recovery_total": sum(r["recovery_total"] for r in results),
        "final": sum(1 for r in results if r["final_success"]),
        "stable": sum(1 for r in results if r["stable"]),
        "seg_rate": avg("seg_rate"),
        "latency": avg("latency"),
        "overshoot": avg("overshoot"),
        "regret_auc": avg("regret_auc"),
        "avg_d_opt": avg("avg_d_opt"),
        "d_final": avg("d_final"),
        "utility": avg("utility"),
        "regret": avg("regret"),
        "corr": avg("corr"),
        "deaths": avg("deaths"),
        "V": avg("V"),
        "T": avg("T"),
        "M": avg("M"),
        "CO": avg("CO"),
        "false_events": avg("false_events"),
    }

    print(
        f"SUMMARY level={level_idx} variant={variant} | "
        f"false={level['false_prob']:.2f} delay={level['delay']} rnoise={level['regret_noise']:.2f} | "
        f"segments={out['segments']}/{out['segment_total']} | "
        f"seg_rate={out['seg_rate']:.3f} | "
        f"recovery={out['recovery']}/{out['recovery_total']} | "
        f"lat={safe_fmt(out['latency'])} | "
        f"final={out['final']}/5 | "
        f"stable={out['stable']}/5 | "
        f"overshoot={out['overshoot']:.2f} | "
        f"regret_auc={out['regret_auc']:.2f} | "
        f"avg_d={out['avg_d_opt']:.2f} | "
        f"final_d={out['d_final']:.2f} | "
        f"utility={out['utility']:.3f} | "
        f"regret={out['regret']:.3f} | "
        f"corr={out['corr']:.1f} | "
        f"deaths={out['deaths']:.1f} | "
        f"CO={out['CO']:.1f} | "
        f"false_events={out['false_events']:.1f}"
    )

    return out


def main():
    base = Config()
    variants = ["FIELD_DUJ", "FULLER_DUJ", "PRIORITY_DUJ", "CONTINUOUS"]

    print("\n===== DUJ V31C ADVERSARIAL SIGNAL NOISE =====")
    print(f"device: {base.device}")
    print(f"steps: {base.steps}")
    print(f"agent_noise: {base.noise}")
    print(f"segment length range: {base.min_segment_len}-{base.max_segment_len}")
    print(f"signal_levels: {list(SIGNAL_LEVELS)}")
    print(f"seeds: {list(base.seeds)}")
    print(f"variants: {variants}")

    matrix = []

    for idx, level in enumerate(SIGNAL_LEVELS):
        print("\n" + "#" * 92)
        print(
            f"SIGNAL LEVEL {idx}: "
            f"false_prob={level['false_prob']:.2f}, "
            f"delay={level['delay']}, "
            f"regret_noise={level['regret_noise']:.2f}"
        )

        for variant in variants:
            print("\n" + "=" * 72)
            print(f"{variant} @ signal_level={idx}")

            results = []

            for seed in base.seeds:
                cfg = Config(
                    seed=seed,
                    variant=variant,
                    false_signal_prob=level["false_prob"],
                    sensing_delay=level["delay"],
                    regret_noise=level["regret_noise"],
                )
                schedule = make_hidden_schedule(cfg, seed)
                model = DUJV31C(cfg, schedule)
                r = model.run()
                results.append(r)

                print(
                    f"seed={seed} | segments={r['segments']}/{r['segment_total']} | "
                    f"recover={r['recovery']}/{r['recovery_total']} | "
                    f"lat={safe_fmt(r['latency'])} | "
                    f"final={r['final_success']} | stable={r['stable']} | "
                    f"overshoot={r['overshoot']:.2f} | regret_auc={r['regret_auc']:.2f} | "
                    f"avg_d={r['avg_d_opt']:.2f} | d_final={r['d_final']:.1f} | "
                    f"utility={r['utility']:.3f} | regret={r['regret']:.3f} | "
                    f"corr={r['corr']} | deaths={r['deaths']} | "
                    f"V={r['V']} | T={r['T']} | M={r['M']} | CO={r['CO']} | "
                    f"false_events={r['false_events']}"
                )

            matrix.append(aggregate(idx, level, variant, results))

    print("\n" + "=" * 116)
    print("V31C ADVERSARIAL SIGNAL MATRIX")
    print(
        "level | false | delay | rnoise | variant | segments | recovery | final | stable | "
        "seg_rate | latency | overshoot | regret_auc | avg_d | final_d | utility | regret | "
        "corr | deaths | V | T | M | CO | false_events"
    )

    for r in matrix:
        print(
            f"{r['level']} | "
            f"{r['false_prob']:.2f} | "
            f"{r['delay']} | "
            f"{r['regret_noise']:.2f} | "
            f"{r['variant']} | "
            f"{r['segments']}/{r['segment_total']} | "
            f"{r['recovery']}/{r['recovery_total']} | "
            f"{r['final']}/5 | "
            f"{r['stable']}/5 | "
            f"{r['seg_rate']:.3f} | "
            f"{safe_fmt(r['latency'])} | "
            f"{r['overshoot']:.2f} | "
            f"{r['regret_auc']:.2f} | "
            f"{r['avg_d_opt']:.2f} | "
            f"{r['d_final']:.2f} | "
            f"{r['utility']:.3f} | "
            f"{r['regret']:.3f} | "
            f"{r['corr']:.1f} | "
            f"{r['deaths']:.1f} | "
            f"{r['V']:.1f} | "
            f"{r['T']:.1f} | "
            f"{r['M']:.1f} | "
            f"{r['CO']:.1f} | "
            f"{r['false_events']:.1f}"
        )

    print("\nFIELD_DUJ ADVERSARIAL SIGNAL CURVE")
    print("level | false | delay | rnoise | seg_rate | latency | regret_auc | utility | final | deaths | CO | false_events")
    for r in matrix:
        if r["variant"] == "FIELD_DUJ":
            print(
                f"{r['level']} | "
                f"{r['false_prob']:.2f} | "
                f"{r['delay']} | "
                f"{r['regret_noise']:.2f} | "
                f"{r['seg_rate']:.3f} | "
                f"{safe_fmt(r['latency'])} | "
                f"{r['regret_auc']:.2f} | "
                f"{r['utility']:.3f} | "
                f"{r['final']}/5 | "
                f"{r['deaths']:.1f} | "
                f"{r['CO']:.1f} | "
                f"{r['false_events']:.1f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
