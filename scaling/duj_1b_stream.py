import argparse, time, csv
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
def stream_kernel(counters, global_offset, batch_size, step, seed,
                  drift_strength, threshold, gain):
    i = cuda.grid(1)
    if i >= batch_size:
        return

    agent_id = global_offset + i

    # Stateless deterministic logical state
    base = 0.05 * noise_hash(agent_id, 0, seed)
    drift = drift_strength * noise_hash(agent_id, step, seed)
    x = base + drift

    residual = abs(x)

    if residual > threshold:
        delta = -gain * x
        x = x + delta
        cuda.atomic.add(counters, 0, 1.0)          # corrections
        cuda.atomic.add(counters, 1, abs(delta))   # energy
        cuda.atomic.add(counters, 2, 1.0)          # violations

    cuda.atomic.add(counters, 3, residual)         # residual sum


def run(args):
    total_agents = args.total_agents
    batch_size = args.batch_size
    batches = (total_agents + batch_size - 1) // batch_size

    threads = args.threads
    blocks = (batch_size + threads - 1) // threads

    d_counters = cuda.to_device(np.zeros(4, dtype=np.float32))
    zero = np.zeros(4, dtype=np.float32)

    print("\nDUJ 1B logical-agent streamed stress test")
    print(f"total agents   : {total_agents}")
    print(f"batch size     : {batch_size}")
    print(f"batches        : {batches}")
    print(f"threads/block  : {threads}")
    print(f"blocks/batch   : {blocks}")
    print(f"steps          : {args.steps}")
    print(f"output         : {args.output}\n")

    rows = []
    total_start = time.perf_counter()

    # warmup
    stream_kernel[blocks, threads](d_counters, 0, batch_size, 0, args.seed,
                                   args.drift_strength, args.threshold, args.gain)
    cuda.synchronize()

    for step in range(args.steps):
        d_counters.copy_to_device(zero)
        step_start = time.perf_counter()

        for b in range(batches):
            offset = b * batch_size
            current_batch = min(batch_size, total_agents - offset)

            stream_kernel[blocks, threads](
                d_counters,
                offset,
                current_batch,
                step,
                args.seed,
                args.drift_strength,
                args.threshold,
                args.gain,
            )

        cuda.synchronize()
        latency = (time.perf_counter() - step_start) * 1000.0

        counters = d_counters.copy_to_host()

        corrections = int(counters[0])
        energy = float(counters[1])
        violations = int(counters[2])
        mean_residual = float(counters[3]) / total_agents

        row = {
            "step": step,
            "total_agents": total_agents,
            "batch_size": batch_size,
            "batches": batches,
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

    total_time = time.perf_counter() - total_start

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", args.output)
    print(f"total runtime sec   : {total_time:.3f}")
    print(f"avg latency ms/step : {sum(r['latency_ms'] for r in rows)/len(rows):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-agents", type=int, default=1_000_000_000)
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--drift-strength", type=float, default=0.04)
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--gain", type=float, default=0.45)
    p.add_argument("--output", default="duj_1b_stream_results.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
