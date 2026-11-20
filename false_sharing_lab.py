import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "a1zzyBd"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
MASK64 = (1 << 64) - 1


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def cpu_model_string() -> str:
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # Deterministic 64-bit seed: first 16 hex digits of SHA-256(git_user || cpu_sig)
    digest = hashlib.sha256(f"{git_user}::{cpu_sig}::{cpu_model_string()}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # Core affinity (Linux only: os.sched_setaffinity)
    affinity_supported = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")
    core_ids = []
    if affinity_supported:
        allowed = sorted(os.sched_getaffinity(0))
        core_ids = allowed[:NUM_THREADS]
        if len(core_ids) < NUM_THREADS:
            affinity_supported = False
            core_ids = []

    return {
        "git_username": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model_string(),
        "cpu_sig": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "core_ids": core_ids,
    }


def aligned_buffer(n_elems: int, align: int = 64) -> np.ndarray:
    """Zeroed uint64 array whose data pointer is `align`-byte aligned."""
    raw = np.zeros(n_elems + align // 8, dtype=np.uint64)
    off = ((-raw.ctypes.data) % align) // 8
    buf = raw[off:off + n_elems]
    assert buf.ctypes.data % align == 0
    return buf


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    # `m` is 0 at runtime (slot is always in range) but LLVM cannot prove it.
    # The address of every access depends on the value just loaded, so LLVM
    # cannot promote buffer[slot] to a register: each iteration is a real
    # load + store to memory.
    n = buffer.shape[0]
    m = np.uint64(0) if slot < n else np.uint64(1)
    idx = slot
    for i in range(iters):
        v = buffer[idx]
        buffer[idx] = v + (np.uint64(i) ^ seed)
        idx = slot + np.int64(m & v)


# ==============================================================================
# TASK 4: Thread-local privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    acc = np.uint64(0)
    for i in range(iters):
        acc += np.uint64(i) ^ seed
    buffer[slot] = acc  # single shared write


# ==============================================================================
# TASK 3: Multi-threaded stride sweep
# ==============================================================================
def thread_seeds(instance_seed: int):
    return [np.uint64((instance_seed ^ ((t + 1) * 0x9E3779B97F4A7C15)) & MASK64)
            for t in range(NUM_THREADS)]


def run_stride_benchmark(stride: int, kernel, ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    buf = aligned_buffer(NUM_THREADS * stride)
    seeds = thread_seeds(ctx["instance_seed"])
    barrier = threading.Barrier(NUM_THREADS + 1)
    cores = ctx["core_ids"]

    def runner(t):
        if ctx["affinity_supported"]:
            try:
                os.sched_setaffinity(0, {cores[t]})  # pid 0 = calling thread on Linux
            except OSError:
                pass
        barrier.wait()
        kernel(buf, t * stride, iters, seeds[t])

    threads = [threading.Thread(target=runner, args=(t,)) for t in range(NUM_THREADS)]
    for th in threads:
        th.start()

    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None

    total_updates = NUM_THREADS * iters
    return {
        "stride": stride,
        "elapsed_ns": elapsed,
        "ns_per_update_per_thread": elapsed / iters,
        "ns_per_update_aggregate": elapsed / total_updates,
        "voluntary_ctx_switches": (ru1.ru_nvcsw - ru0.ru_nvcsw) if resource else None,
        "involuntary_ctx_switches": (ru1.ru_nivcsw - ru0.ru_nivcsw) if resource else None,
        "final_values": [int(buf[t * stride]) for t in range(NUM_THREADS)],
    }


def expected_final(seed: int, iters: int) -> int:
    """sum_{i<iters} (i XOR seed) mod 2^64, computed independently with NumPy."""
    total = 0
    chunk = 5_000_000
    s = np.uint64(seed)
    for start in range(0, iters, chunk):
        arr = np.arange(start, min(start + chunk, iters), dtype=np.uint64) ^ s
        total = (total + int(arr.sum(dtype=np.uint64))) & MASK64
    return total


# ==============================================================================
# TASK 5: Parity + LLVM IR audit + manifest
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, results: dict, private: dict) -> dict:
    # Bit-exact parity across all layouts and the privatized baseline
    reference = private["final_values"]
    parity_ok = all(results[s]["final_values"] == reference for s in STRIDES)

    expected = [expected_final(int(s), ITERATIONS_PER_THREAD) for s in thread_seeds(ctx["instance_seed"])]
    expected_ok = reference == expected

    # LLVM IR inspection
    ir_dict = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_dict.values())
    has_i64_store = "store i64" in ir
    py_fallback = any(tok in ir for tok in ("PyObject", "PyEval", "Py_IncRef", "PyArg_"))

    store_lines = [l.strip() for l in ir.splitlines() if "store i64" in l]
    load_lines = [l.strip() for l in ir.splitlines() if "load i64" in l]
    print("\n--- LLVM IR: lines with stores in worker_strided_accumulate ---")
    for line in store_lines:
        print(line)

    return {
        "llvm_store_i64_lines": store_lines[:6],
        "llvm_load_i64_lines": load_lines[:6],
        "parity_across_layouts": parity_ok,
        "matches_independent_numpy_sum": expected_ok,
        "llvm_has_store_i64": has_i64_store,
        "llvm_python_object_fallback": py_fallback,
        "verification_passed": bool(parity_ok and expected_ok and has_i64_store and not py_fallback),
    }


def main():
    ctx = init_system_context(GIT_USERNAME)
    print("Context:", json.dumps({k: v for k, v in ctx.items() if k != "instance_seed"}, indent=2))
    print("INSTANCE_SEED:", ctx["instance_seed"])

    # Warm-up: compile both kernels outside the timed region
    warm = aligned_buffer(NUM_THREADS * 16)
    worker_strided_accumulate(warm, 0, 10, np.uint64(1))
    worker_private_accumulate(warm, 0, 10, np.uint64(1))

    results = {}
    for stride in STRIDES:
        results[stride] = run_stride_benchmark(stride, worker_strided_accumulate, ctx)
        r = results[stride]
        print(f"stride={stride:>2}: {r['elapsed_ns']:>15,} ns | "
              f"{r['ns_per_update_per_thread']:.3f} ns/update/thread | "
              f"vcsw={r['voluntary_ctx_switches']} ivcsw={r['involuntary_ctx_switches']}")

    private = run_stride_benchmark(8, worker_private_accumulate, ctx)
    print(f"private  : {private['elapsed_ns']:>15,} ns | "
          f"{private['ns_per_update_per_thread']:.3f} ns/update/thread")

    slowdown_1_vs_8 = results[1]["elapsed_ns"] / results[8]["elapsed_ns"]
    slowdown_8_vs_16 = results[8]["elapsed_ns"] / results[16]["elapsed_ns"]
    print(f"slowdown stride1/stride8 = {slowdown_1_vs_8:.2f}x")
    print(f"stride8/stride16         = {slowdown_8_vs_16:.2f}x")
    print(f"stride8/private          = {results[8]['elapsed_ns'] / private['elapsed_ns']:.2f}x")

    audit = audit_llvm_and_parity(ctx, results, private)

    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "affinity_supported": ctx["affinity_supported"],
        "core_ids": ctx["core_ids"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "timings": {f"stride_{s}": {k: v for k, v in results[s].items() if k != "final_values"}
                    for s in STRIDES},
        "private_baseline": {k: v for k, v in private.items() if k != "final_values"},
        "final_values_reference": [str(v) for v in private["final_values"]],
        "slowdown_stride1_vs_stride8": slowdown_1_vs_8,
        "ratio_stride8_vs_stride16": slowdown_8_vs_16,
        **audit,
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nverification_passed:", audit["verification_passed"])
    print("Wrote cache_lab_manifest.json")


if __name__ == "__main__":
    main()
