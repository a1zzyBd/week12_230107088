import os
import sys
import re
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2          # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_N = 256                      # interior points per rank per axis
PAD = LOCAL_N + 2                  # 258 (ghost layers at index 0 and 257)
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15                 # strength of the absorption term  S(x,y)*u
CFL_SAFETY = 0.9                   # dt = CFL_SAFETY * dt_max

# message tags (named by the direction the message travels)
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 10, 11, 12, 13


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")
    ).digest()
    instance_seed = int.from_bytes(digest[:8], "little")   # deterministic 64-bit

    # --- non-periodic 2x2 Cartesian communicator -------------------------------
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)                # (row, col)
    up, down = cart_comm.Shift(0, 1)                         # (source, dest)
    left, right = cart_comm.Shift(1, 1)
    nbrs = {"up": up, "down": down, "left": left, "right": right}

    # --- node-local MPI-3 shared-memory window (64 uint64 slots, 64B aligned) --
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node_comm.Get_size() != 4:
        raise RuntimeError("All 4 ranks must share one node for Tier 4.")
    slots, spare = 64, 8                                     # spare words to realign
    nbytes = (slots + spare) * 8 if node_comm.Get_rank() == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, MPI.INFO_NULL, node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.frombuffer(buf, dtype=np.uint64)
    off = ((-raw.ctypes.data) % 64) // 8
    shm_view = raw[off:off + slots]
    assert shm_view.ctypes.data % 64 == 0, "shared window is not 64-byte aligned"

    return {
        "comm": comm, "rank": rank, "cart_comm": cart_comm, "cart_rank": cart_rank,
        "coords": coords, "nbrs": nbrs, "node_comm": node_comm, "win": win,
        "shm_view": shm_view, "gpu_uuid": gpu_uuid, "cpu_sig": cpu_sig,
        "cpu_line_bytes": line_bytes, "seed": instance_seed, "github_user": github_user,
    }


# ==============================================================================
# TASK 2: Rank-local fields + halo pack / unpack kernels
# ==============================================================================
def initialize_fields(ctx: dict):
    """Build (258,258) float64 arrays from GLOBAL coordinates (no global grid)."""
    cy, cx = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    idx = np.arange(PAD, dtype=np.float64)
    y = (cy * LOCAL_N + idx) * dx            # ghost row 0 of rank row 0 -> y = 0
    x = (cx * LOCAL_N + idx) * dx
    Y, X = np.meshgrid(y, x, indexing="ij")

    seed = ctx["seed"]
    x0 = 0.5 + 0.10 * (((seed >> 8) & 0xFFFF) / 65535.0 - 0.5)    # seed-bound bump
    y0 = 0.5 + 0.10 * (((seed >> 24) & 0xFFFF) / 65535.0 - 0.5)
    bump = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * 0.08 ** 2))

    u = np.zeros((PAD, PAD), dtype=np.float64)       # ghosts stay 0 (Dirichlet)
    u[1:-1, 1:-1] = bump[1:-1, 1:-1]
    kappa = 0.2 + 0.8 * 0.5 * (1.0 + np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y))
    S = 0.5 * (1.0 + np.cos(2 * np.pi * X))          # absorption profile in [0,1]
    return u, np.ascontiguousarray(kappa), np.ascontiguousarray(S)


@cuda.jit
def pack_halos_kernel(u, sbuf):
    """sbuf layout: [top row | bottom row | left col | right col], 256 each."""
    t = cuda.grid(1)
    if t < 4 * LOCAL_N:
        side = t // LOCAL_N
        k = t - side * LOCAL_N
        if side == 0:
            v = u[1, 1 + k]
        elif side == 1:
            v = u[LOCAL_N, 1 + k]
        elif side == 2:
            v = u[1 + k, 1]
        else:
            v = u[1 + k, LOCAL_N]
        sbuf[t] = v


@cuda.jit
def unpack_halos_kernel(u, rbuf, flags):
    """rbuf layout: [from up -> row 0 | from down -> row 257 |
                     from left -> col 0 | from right -> col 257]."""
    t = cuda.grid(1)
    if t < 4 * LOCAL_N:
        side = t // LOCAL_N
        k = t - side * LOCAL_N
        if flags[side] != 0:                          # skip physical boundaries
            v = rbuf[t]
            if side == 0:
                u[0, 1 + k] = v
            elif side == 1:
                u[PAD - 1, 1 + k] = v
            elif side == 2:
                u[1 + k, 0] = v
            else:
                u[1 + k, PAD - 1] = v


# ==============================================================================
# TASK 3: Shared-memory tiled stencil + dual-stream pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, unew, kap, sink, c, lam_dt, i0, j0, ni, nj):
    """Updates region [i0,i0+ni) x [j0,j0+nj) using an 18x18 shared tile."""
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = j0 + cuda.blockIdx.x * TILE_DIM + tx
    i = i0 + cuda.blockIdx.y * TILE_DIM + ty
    inb = (i >= 0) and (i < PAD) and (j >= 0) and (j < PAD)

    if inb:
        tile[ty + 1, tx + 1] = u[i, j]
        if ty == 0 and i >= 1:
            tile[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < PAD:
            tile[SHARED_DIM - 1, tx + 1] = u[i + 1, j]
        if tx == 0 and j >= 1:
            tile[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < PAD:
            tile[ty + 1, SHARED_DIM - 1] = u[i, j + 1]
    cuda.syncthreads()                                 # all threads reach this

    if i < i0 + ni and j < j0 + nj:
        uc = tile[ty + 1, tx + 1]
        kc = kap[i, j]
        kE = 0.5 * (kc + kap[i, j + 1])
        kW = 0.5 * (kc + kap[i, j - 1])
        kN = 0.5 * (kc + kap[i - 1, j])
        kS = 0.5 * (kc + kap[i + 1, j])
        lap = (kE * (tile[ty + 1, tx + 2] - uc) - kW * (uc - tile[ty + 1, tx])
               + kS * (tile[ty + 2, tx + 1] - uc) - kN * (uc - tile[ty, tx + 1]))
        unew[i, j] = uc + c * lap - lam_dt * sink[i, j] * uc


def launch_region(st, stream, i0, j0, ni, nj):
    grid = ((nj + TILE_DIM - 1) // TILE_DIM, (ni + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        st["d_u"], st["d_unew"], st["d_kap"], st["d_sink"],
        st["c"], st["lam_dt"], i0, j0, ni, nj)


def make_state(ctx, fields, c, lam_dt):
    u, kappa, S = fields
    return {
        "d_u": cuda.to_device(u), "d_unew": cuda.to_device(u),
        "d_kap": cuda.to_device(kappa), "d_sink": cuda.to_device(S),
        "d_sbuf": cuda.device_array(4 * LOCAL_N, dtype=np.float64),
        "d_rbuf": cuda.device_array(4 * LOCAL_N, dtype=np.float64),
        "d_flags": cuda.to_device(np.array(
            [ctx["nbrs"][k] != MPI.PROC_NULL for k in ("up", "down", "left", "right")],
            dtype=np.int32)),
        "h_send": cuda.pinned_array(4 * LOCAL_N, dtype=np.float64),
        "h_recv": cuda.pinned_array(4 * LOCAL_N, dtype=np.float64),
        "stream_interior": cuda.stream(), "stream_boundary": cuda.stream(),
        "c": float(c), "lam_dt": float(lam_dt),
    }


def post_halo_requests(ctx, st):
    cc, nb, n = ctx["cart_comm"], ctx["nbrs"], LOCAL_N
    s, r = st["h_send"], st["h_recv"]
    return [
        cc.Irecv(r[0:n],       source=nb["up"],    tag=TAG_DOWN),   # their bottom row
        cc.Irecv(r[n:2 * n],   source=nb["down"],  tag=TAG_UP),     # their top row
        cc.Irecv(r[2 * n:3 * n], source=nb["left"],  tag=TAG_RIGHT),
        cc.Irecv(r[3 * n:4 * n], source=nb["right"], tag=TAG_LEFT),
        cc.Isend(s[0:n],       dest=nb["up"],      tag=TAG_UP),
        cc.Isend(s[n:2 * n],   dest=nb["down"],    tag=TAG_DOWN),
        cc.Isend(s[2 * n:3 * n], dest=nb["left"],  tag=TAG_LEFT),
        cc.Isend(s[3 * n:4 * n], dest=nb["right"], tag=TAG_RIGHT),
    ]


def step_serialized(ctx, st):
    """Reference: exchange fully, then update the whole interior on the default stream."""
    pack_halos_kernel[4, LOCAL_N](st["d_u"], st["d_sbuf"])
    st["d_sbuf"].copy_to_host(st["h_send"])
    MPI.Request.Waitall(post_halo_requests(ctx, st))
    st["d_rbuf"].copy_to_device(st["h_recv"])
    unpack_halos_kernel[4, LOCAL_N](st["d_u"], st["d_rbuf"], st["d_flags"])
    launch_region(st, 0, 1, 1, LOCAL_N, LOCAL_N)
    cuda.synchronize()
    st["d_u"], st["d_unew"] = st["d_unew"], st["d_u"]


def step_overlapped(ctx, st):
    """Deep interior on stream_interior while halos move on stream_boundary."""
    si, sb = st["stream_interior"], st["stream_boundary"]
    pack_halos_kernel[4, LOCAL_N, sb](st["d_u"], st["d_sbuf"])
    st["d_sbuf"].copy_to_host(st["h_send"], stream=sb)
    sb.synchronize()
    reqs = post_halo_requests(ctx, st)                       # non-blocking MPI
    launch_region(st, si, 2, 2, LOCAL_N - 2, LOCAL_N - 2)    # deep interior [2..255]^2
    MPI.Request.Waitall(reqs)
    st["d_rbuf"].copy_to_device(st["h_recv"], stream=sb)
    unpack_halos_kernel[4, LOCAL_N, sb](st["d_u"], st["d_rbuf"], st["d_flags"])
    launch_region(st, sb, 1, 1, 1, LOCAL_N)                  # top strip
    launch_region(st, sb, LOCAL_N, 1, 1, LOCAL_N)            # bottom strip
    launch_region(st, sb, 2, 1, LOCAL_N - 2, 1)              # left strip
    launch_region(st, sb, 2, LOCAL_N, LOCAL_N - 2, 1)        # right strip
    sb.synchronize()
    si.synchronize()
    st["d_u"], st["d_unew"] = st["d_unew"], st["d_u"]


def global_energy(ctx, st, dx):
    u = st["d_u"].copy_to_host()
    local = float(np.sum(u[1:-1, 1:-1] ** 2)) * dx * dx
    return ctx["cart_comm"].allreduce(local, op=MPI.SUM)


def run_simulation(ctx, fields, c, lam_dt, dx, mode, steps):
    st = make_state(ctx, fields, c, lam_dt)
    step_fn = step_overlapped if mode == "overlapped" else step_serialized
    energies = {0: global_energy(ctx, st, dx)}
    elapsed, done = 0.0, 0
    while done < steps:
        chunk = min(100, steps - done)
        cuda.synchronize()
        ctx["cart_comm"].Barrier()
        t0 = time.perf_counter()
        for _ in range(chunk):
            step_fn(ctx, st)
        cuda.synchronize()
        ctx["cart_comm"].Barrier()
        elapsed += time.perf_counter() - t0
        done += chunk
        energies[done] = global_energy(ctx, st, dx)
    u_final = st["d_u"].copy_to_host()[1:-1, 1:-1].copy()
    return u_final, elapsed * 1000.0, energies


# ==============================================================================
# TASK 4: Host cache-line false-sharing sweep on the MPI-3 shared window
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(a, b, idx, iters):
    """a and b are two views of the SAME shared buffer. Reading through `a` and
    writing through `b` forces LLVM to reload/store every iteration (it cannot
    prove they don't alias), so the counter is never hoisted into a register
    and the loop is never collapsed to `+= iters`."""
    for _ in range(iters):
        b[idx] = a[idx] + np.uint64(1)


def benchmark_shm_strides(ctx, iters=SHM_ITERS, repeats=3):
    node_comm, shm = ctx["node_comm"], ctx["shm_view"]
    shm_b = shm.view()
    nr = node_comm.Get_rank()
    host_shm_accumulate(shm, shm_b, nr, 1)                # JIT warm-up
    results = {}
    for stride in (1, 8, 16):
        best = float("inf")
        for _ in range(repeats):
            node_comm.Barrier()
            if nr == 0:
                shm[:] = 0
            node_comm.Barrier()
            t0 = time.perf_counter()
            host_shm_accumulate(shm, shm_b, nr * stride, iters)
            t = time.perf_counter() - t0
            node_comm.Barrier()
            t_max = node_comm.allreduce(t, op=MPI.MAX)
            assert int(shm[nr * stride]) == iters, "lost updates on private slot"
            best = min(best, t_max)
        results[stride] = best * 1000.0
    return results


def audit_cpu_llvm():
    ir = list(host_shm_accumulate.inspect_llvm().values())[0]
    counts = {
        "load_i64": len(re.findall(r"\bload i64\b", ir)),
        "store_i64": len(re.findall(r"\bstore i64\b", ir)),
        "phi": len(re.findall(r"\bphi\b", ir)),
        "br": len(re.findall(r"\bbr\b", ir)),
    }
    assert counts["load_i64"] >= 1 and counts["store_i64"] >= 1, "update was hoisted!"
    return counts


def audit_gpu_ptx():
    ptx = list(tiled_stencil_kernel.inspect_asm().values())[0]
    counts = {
        "st.shared.f64": len(re.findall(r"st\.shared\.f64", ptx)),
        "ld.shared.f64": len(re.findall(r"ld\.shared\.f64", ptx)),
        "bar.sync": len(re.findall(r"bar\.sync", ptx)),
    }
    assert all(v >= 1 for v in counts.values()), f"PTX audit failed: {counts}"
    return counts


# ==============================================================================
# TASK 5: Verification + manifest
# ==============================================================================
def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: hybrid_mpi_cuda_cache.py <github_username>")
    user = sys.argv[1]
    ctx = init_hybrid_context(user)
    rank, cc = ctx["rank"], ctx["cart_comm"]
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    dx = 1.0 / (GLOBAL_NX + 1)
    fields = initialize_fields(ctx)
    kappa_max = cc.allreduce(float(fields[1].max()), op=MPI.MAX)
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max
    assert dt <= dt_max
    c = dt / (dx * dx)
    lam_dt = LAMBDA_DAMP * dt

    say(f"[Task1] user={user} seed=0x{ctx['seed']:016x} gpu={ctx['gpu_uuid']}")
    say(f"[Task1] cpu={ctx['cpu_sig']} line={ctx['cpu_line_bytes']}B "
        f"shm 64B-aligned, ranks={ctx['comm'].Get_size()}")
    say(f"[CFL] kappa_max={kappa_max:.6f} dx={dx:.8e} dt_max={dt_max:.6e} dt={dt:.6e}")

    # warm-up (JIT compile both paths) -- results discarded
    run_simulation(ctx, fields, c, lam_dt, dx, "serialized", 1)
    run_simulation(ctx, fields, c, lam_dt, dx, "overlapped", 1)

    u_ser, t_ser, e_ser = run_simulation(ctx, fields, c, lam_dt, dx, "serialized", STENCIL_STEPS)
    u_ovl, t_ovl, e_ovl = run_simulation(ctx, fields, c, lam_dt, dx, "overlapped", STENCIL_STEPS)

    max_diff = cc.allreduce(float(np.max(np.abs(u_ser - u_ovl))), op=MPI.MAX)
    t_ser = cc.allreduce(t_ser, op=MPI.MAX)
    t_ovl = cc.allreduce(t_ovl, op=MPI.MAX)
    say(f"[Task3] gpu_serialized_ms={t_ser:.3f} gpu_overlapped_ms={t_ovl:.3f} "
        f"speedup={t_ser / t_ovl:.3f}x")
    say(f"[Task3] parity max_diff={max_diff:.3e}")
    assert max_diff < 1e-12, "serialized and overlapped fields differ"

    for s in (0, 100, 200, 300):
        say(f"[Energy] step {s:3d}: global E = {e_ovl[s]:.10e}")
    seq = [e_ovl[s] for s in (0, 100, 200, 300)]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(seq, seq[1:])), "energy grew"

    shm_ms = benchmark_shm_strides(ctx)
    ratio_fs = shm_ms[1] / shm_ms[8]
    ratio_pad = shm_ms[16] / shm_ms[8]
    say(f"[Task4] stride1(8B)={shm_ms[1]:.2f} ms  stride8(64B)={shm_ms[8]:.2f} ms  "
        f"stride16(128B)={shm_ms[16]:.2f} ms")
    say(f"[Task4] false-sharing slowdown (s1/s8) = {ratio_fs:.2f}x ; s16/s8 = {ratio_pad:.2f}x")

    ptx = audit_gpu_ptx()
    llvm = audit_cpu_llvm()
    say(f"[Audit] PTX  {ptx}")
    say(f"[Audit] LLVM {llvm}")

    if rank == 0:
        manifest = {
            "github_username": user,
            "instance_seed_hex": f"0x{ctx['seed']:016x}",
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_signature": ctx["cpu_sig"],
            "cpu_line_bytes": ctx["cpu_line_bytes"],
            "grid": {"global": [GLOBAL_NX, GLOBAL_NY], "local": [LOCAL_N, LOCAL_N],
                     "padded": [PAD, PAD], "steps": STENCIL_STEPS},
            "kappa_max": kappa_max, "dx": dx, "dt_max": dt_max, "dt": dt,
            "cfl_ok": bool(dt <= dt_max),
            "gpu_serialized_ms": t_ser, "gpu_overlapped_ms": t_ovl,
            "overlap_speedup": t_ser / t_ovl, "parity_max_diff": max_diff,
            "energy": {str(s): e_ovl[s] for s in (0, 100, 200, 300)},
            "shm_ms": {"stride_1_8B": shm_ms[1], "stride_8_64B": shm_ms[8],
                       "stride_16_128B": shm_ms[16]},
            "false_sharing_ratio_s1_over_s8": ratio_fs,
            "padding_ratio_s16_over_s8": ratio_pad,
            "ptx_audit": ptx, "llvm_audit": llvm,
            "column_read_bytes": {"useful": LOCAL_N * 8,
                                  "gpu_32B_sectors_moved": LOCAL_N * 32,
                                  "cpu_64B_lines_moved": LOCAL_N * 64,
                                  "row_stride_bytes": PAD * 8},
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        say("[Task5] wrote hybrid_lab_manifest.json")

    cc.Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    main()
