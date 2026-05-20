import argparse, time, csv, math
import numpy as np
from numba import cuda, float64


@cuda.jit(device=True)
def noise_hash(i, step, seed):
    x = i * 747796405 + step * 2891336453 + seed * 277803737
    x = (x ^ (x >> 16)) * 2246822519
    x = (x ^ (x >> 13)) * 3266489917
    x = x ^ (x >> 16)
    return ((x & 0xFFFF) / 32767.5) - 1.0


@cuda.jit
def init_parent(parent_state, seed):
    i = cuda.grid(1)
    if i < parent_state.size:
        parent_state[i] = 0.05 * noise_hash(i, 0, seed)


@cuda.jit
def init_child(child_state, seed):
    i = cuda.grid(1)
    if i < child_state.size:
        child_state[i] = 0.05 * noise_hash(i + 1234567, 0, seed)


@cuda.jit
def parent_step(parent_state, block_metrics, step, seed, drift, limit, gain):
    sm_viol = cuda.shared.array(256, float64)
    sm_corr = cuda.shared.array(256, float64)
    sm_energy = cuda.shared.array(256, float64)
    sm_res = cuda.shared.array(256, float64)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    viol = 0.0
    corr = 0.0
    energy = 0.0
    res = 0.0

    if i < parent_state.size:
        p = parent_state[i]
        p = p + drift * 0.1 * noise_hash(i, step, seed)
        residual = abs(p)

        if residual > limit:
            delta = -gain * p
            p = p + delta
            viol = 1.0
            corr = 1.0
            energy = abs(delta)

        parent_state[i] = p
        res = residual

    sm_viol[tid] = viol
    sm_corr[tid] = corr
    sm_energy[tid] = energy
    sm_res[tid] = res
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sm_viol[tid] += sm_viol[tid + stride]
            sm_corr[tid] += sm_corr[tid + stride]
            sm_energy[tid] += sm_energy[tid + stride]
            sm_res[tid] += sm_res[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        b = cuda.blockIdx.x
        block_metrics[b, 0] = sm_viol[0]
        block_metrics[b, 1] = sm_corr[0]
        block_metrics[b, 2] = sm_energy[0]
        block_metrics[b, 3] = sm_res[0]


@cuda.jit
def child_step(child_state, parent_state, child_metrics, step, seed,
               cpp, drift, coupling, threshold, gain):
    sm_viol = cuda.shared.array(256, float64)
    sm_corr = cuda.shared.array(256, float64)
    sm_energy = cuda.shared.array(256, float64)
    sm_res = cuda.shared.array(256, float64)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    viol = 0.0
    corr = 0.0
    energy = 0.0
    res = 0.0

    if i < child_state.size:
        parent_id = i // cpp
        c = child_state[i]
        p = parent_state[parent_id]

        c = c + drift * noise_hash(i, step, seed) + coupling * (p - c)
        residual = abs(c - p)

        if residual > threshold:
            delta = gain * (p - c)
            c = c + delta
            viol = 1.0
            corr = 1.0
            energy = abs(delta)

        child_state[i] = c
        res = residual

    sm_viol[tid] = viol
    sm_corr[tid] = corr
    sm_energy[tid] = energy
    sm_res[tid] = res
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sm_viol[tid] += sm_viol[tid + stride]
            sm_corr[tid] += sm_corr[tid + stride]
            sm_energy[tid] += sm_energy[tid + stride]
            sm_res[tid] += sm_res[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        b = cuda.blockIdx.x
        child_metrics[b, 0] = sm_viol[0]
        child_metrics[b, 1] = sm_corr[0]
        child_metrics[b, 2] = sm_energy[0]
        child_metrics[b, 3] = sm_res[0]


def run(args):
    parents = args.parents
    cpp = args.children_per_parent
    children = parents * cpp
    total = parents + children
    threads = args.threads

    parent_blocks = math.ceil(parents / threads)
    child_blocks = math.ceil(children / threads)

    print("\nDUJ 1B resident-style stress test")
    print(f"parents         : {parents:,}")
    print(f"children        : {children:,}")
    print(f"total agents    : {total:,}")
    print(f"parent blocks   : {parent_blocks:,}")
    print(f"child blocks    : {child_blocks:,}")
    print(f"steps           : {args.steps}")
    print("")

    parent_state = cuda.device_array(parents, dtype=np.float32)
    child_state = cuda.device_array(children, dtype=np.float32)

    parent_metrics = cuda.device_array((parent_blocks, 4), dtype=np.float64)
    child_metrics = cuda.device_array((child_blocks, 4), dtype=np.float64)

    init_parent[parent_blocks, threads](parent_state, args.seed)
    init_child[child_blocks, threads](child_state, args.seed)
    cuda.synchronize()

    rows = []
    total_start = time.perf_counter()

    for step in range(args.steps):
        start = time.perf_counter()

        parent_step[parent_blocks, threads](
            parent_state, parent_metrics, step, args.seed,
            args.drift_strength, args.parent_limit, args.parent_gain
        )

        child_step[child_blocks, threads](
            child_state, parent_state, child_metrics, step, args.seed,
            cpp, args.drift_strength, args.coupling_strength,
            args.child_threshold, args.child_gain
        )

        cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000.0

        pm = parent_metrics.copy_to_host()
        cm = child_metrics.copy_to_host()

        violations = int(pm[:, 0].sum() + cm[:, 0].sum())
        corrections = int(pm[:, 1].sum() + cm[:, 1].sum())
        energy = float(pm[:, 2].sum() + cm[:, 2].sum())
        residual_sum = float(pm[:, 3].sum() + cm[:, 3].sum())
        mean_residual = residual_sum / total

        row = {
            "step": step,
            "parents": parents,
            "children": children,
            "total_agents": total,
            "violations": violations,
            "corrections": corrections,
            "correction_energy": energy,
            "mean_residual": mean_residual,
            "latency_ms": latency,
        }
        rows.append(row)

        print(
            f"step={step:03d} "
            f"viol={violations:,} "
            f"corr={corrections:,} "
            f"energy={energy:.3f} "
            f"mean_res={mean_residual:.8f} "
            f"lat={latency:.2f}ms"
        )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", args.output)
    print(f"total runtime sec: {time.perf_counter() - total_start:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parents", type=int, default=200_000_000)
    p.add_argument("--children-per-parent", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=256)

    p.add_argument("--drift-strength", type=float, default=0.04)
    p.add_argument("--coupling-strength", type=float, default=0.03)
    p.add_argument("--parent-limit", type=float, default=1.0)
    p.add_argument("--parent-gain", type=float, default=0.35)
    p.add_argument("--child-threshold", type=float, default=0.35)
    p.add_argument("--child-gain", type=float, default=0.45)

    p.add_argument("--output", default="duj_1b_resident_results.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
