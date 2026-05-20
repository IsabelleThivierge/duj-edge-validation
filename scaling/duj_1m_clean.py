import argparse
import time
import csv
import numpy as np
from numba import cuda


@cuda.jit(device=True)
def noise_hash(i, step, seed):
    x = i * 747796405 + step * 2891336453 + seed * 277803737
    x = (x ^ (x >> 16)) * 2246822519
    x = (x ^ (x >> 13)) * 3266489917
    x = x ^ (x >> 16)
    return ((x & 0xFFFF) / 32767.5) - 1.0


@cuda.jit
def duj_1m_kernel(
    state,
    parent_state,
    parent_ids,
    counters,
    step,
    seed,
    drift_strength,
    coupling_strength,
    threshold,
    gain,
):
    i = cuda.grid(1)

    if i >= state.size:
        return

    parent_id = parent_ids[i]

    x = state[i]
    p = parent_state[parent_id]

    n = noise_hash(i, step, seed)

    # Drift + weak parent coupling
    x = x + drift_strength * n + coupling_strength * (p - x)

    residual = abs(x - p)

    if residual > threshold:
        delta = gain * (p - x)
        x = x + delta

        cuda.atomic.add(counters, 0, 1.0)          # corrections
        cuda.atomic.add(counters, 1, abs(delta))   # correction energy
        cuda.atomic.add(counters, 2, 1.0)          # violations

    cuda.atomic.add(counters, 3, residual)         # residual sum

    # Approximate max residual using scaled int atomic
    scaled = int(residual * 1000000.0)
    cuda.atomic.max(counters, 4, float(scaled))

    state[i] = x


@cuda.jit
def parent_kernel(parent_state, counters, step, seed, drift_strength, parent_limit, parent_gain):
    i = cuda.grid(1)

    if i >= parent_state.size:
        return

    p = parent_state[i]
    n = noise_hash(i, step, seed)

    p = p + drift_strength * 0.1 * n

    residual = abs(p)

    if residual > parent_limit:
        delta = -parent_gain * p
        p = p + delta

        cuda.atomic.add(counters, 0, 1.0)
        cuda.atomic.add(counters, 1, abs(delta))
        cuda.atomic.add(counters, 2, 1.0)

    cuda.atomic.add(counters, 3, residual)

    scaled = int(residual * 1000000.0)
    cuda.atomic.max(counters, 4, float(scaled))

    parent_state[i] = p


def run(args):
    parents = args.parents
    children_per_parent = args.children_per_parent
    children = parents * children_per_parent
    total_agents = parents + children

    rng = np.random.default_rng(args.seed)

    parent_state = rng.normal(0.0, 0.05, size=parents).astype(np.float32)
    child_state = rng.normal(0.0, 0.05, size=children).astype(np.float32)

    parent_ids = (np.arange(children, dtype=np.int32) // children_per_parent).astype(np.int32)

    d_parent_state = cuda.to_device(parent_state)
    d_child_state = cuda.to_device(child_state)
    d_parent_ids = cuda.to_device(parent_ids)

    d_counters = cuda.to_device(np.zeros(5, dtype=np.float32))

    threads = args.threads
    parent_blocks = (parents + threads - 1) // threads
    child_blocks = (children + threads - 1) // threads

    print("")
    print("DUJ 1M-agent clean stress test")
    print(f"parents              : {parents}")
    print(f"children/parent      : {children_per_parent}")
    print(f"children             : {children}")
    print(f"total agents         : {total_agents}")
    print(f"threads/block        : {threads}")
    print(f"parent blocks        : {parent_blocks}")
    print(f"child blocks         : {child_blocks}")
    print(f"steps                : {args.steps}")
    print(f"seed                 : {args.seed}")
    print(f"output               : {args.output}")
    print("")

    rows = []
    zero = np.zeros(5, dtype=np.float32)

    # Warmup / compile
    d_counters.copy_to_device(zero)

    parent_kernel[parent_blocks, threads](
        d_parent_state,
        d_counters,
        0,
        args.seed,
        args.drift_strength,
        args.parent_limit,
        args.parent_gain,
    )

    duj_1m_kernel[child_blocks, threads](
        d_child_state,
        d_parent_state,
        d_parent_ids,
        d_counters,
        0,
        args.seed,
        args.drift_strength,
        args.coupling_strength,
        args.child_threshold,
        args.child_gain,
    )

    cuda.synchronize()

    total_start = time.perf_counter()

    for step in range(args.steps):
        d_counters.copy_to_device(zero)

        start = time.perf_counter()

        parent_kernel[parent_blocks, threads](
            d_parent_state,
            d_counters,
            step,
            args.seed,
            args.drift_strength,
            args.parent_limit,
            args.parent_gain,
        )

        duj_1m_kernel[child_blocks, threads](
            d_child_state,
            d_parent_state,
            d_parent_ids,
            d_counters,
            step,
            args.seed,
            args.drift_strength,
            args.coupling_strength,
            args.child_threshold,
            args.child_gain,
        )

        cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        counters = d_counters.copy_to_host()

        corrections = int(counters[0])
        correction_energy = float(counters[1])
        violations = int(counters[2])
        residual_sum = float(counters[3])
        max_residual = float(counters[4]) / 1000000.0

        mean_residual = residual_sum / total_agents

        row = {
            "step": step,
            "parents": parents,
            "children": children,
            "total_agents": total_agents,
            "violations": violations,
            "corrections": corrections,
            "correction_energy": correction_energy,
            "mean_residual": mean_residual,
            "max_residual": max_residual,
            "latency_ms": latency_ms,
        }

        rows.append(row)

        if step % args.print_every == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} "
                f"viol={violations:08d} "
                f"corr={corrections:08d} "
                f"energy={correction_energy:.3f} "
                f"mean_res={mean_residual:.6f} "
                f"max_res={max_residual:.6f} "
                f"lat={latency_ms:.3f}ms"
            )

    total_time = time.perf_counter() - total_start

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
    avg_corr = sum(r["corrections"] for r in rows) / len(rows)
    avg_energy = sum(r["correction_energy"] for r in rows) / len(rows)
    avg_mean_res = sum(r["mean_residual"] for r in rows) / len(rows)

    print("")
    print("Saved:", args.output)
    print("")
    print("Summary")
    print(f"total runtime sec        : {total_time:.3f}")
    print(f"avg latency ms/step      : {avg_latency:.6f}")
    print(f"avg corrections/step     : {avg_corr:.3f}")
    print(f"avg correction energy    : {avg_energy:.6f}")
    print(f"avg mean residual        : {avg_mean_res:.6f}")
    print(f"final max residual       : {rows[-1]['max_residual']:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--parents", type=int, default=200000)
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=256)

    parser.add_argument("--drift-strength", type=float, default=0.04)
    parser.add_argument("--coupling-strength", type=float, default=0.03)

    parser.add_argument("--parent-limit", type=float, default=1.0)
    parser.add_argument("--parent-gain", type=float, default=0.35)

    parser.add_argument("--child-threshold", type=float, default=0.35)
    parser.add_argument("--child-gain", type=float, default=0.45)

    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output", type=str, default="duj_1m_clean_results.csv")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
