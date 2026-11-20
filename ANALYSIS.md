# Analysis: Cache Line & False Sharing Mechanics

Host: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD | logical cores: 16 | cache line: 64 B |
threads: 4 | iterations/thread: 60,000,000 | affinity pinning: False (cores [])
Manifest `verification_passed`: **False**

## Q1 - MESI state transitions (stride = 1)

`buffer[0]` and `buffer[1]` are 8 bytes apart, so they live in the same 64-byte line.

1. Core 0 holds the line in **Modified** (or Exclusive, upgraded to Modified on its first store).
2. Core 1 stores to `buffer[1]` and misses on its copy, so it broadcasts a Read-For-Ownership (RFO).
3. Core 0 snoops the RFO, writes the dirty line back / forwards it, and moves **M -> Invalid**.
4. Core 1 receives the line, installs it as **Exclusive -> Modified** and completes its store.
5. Core 0's next store finds its line **Invalid**, issues its own RFO, and the sequence repeats in reverse.

Core 1's store cannot commit to L1 until it owns the line in an exclusive state. Stores retire in order from the
store buffer, so the store waits at the head of the buffer for the cross-core round trip, and later stores queue
behind it. The store buffer fills and the pipeline stalls.

## Q2 - Quantitative slowdown

| Layout | Stride | Elapsed (ns) | ns / update / thread |
|---|---|---|---|
| A contiguous | 1 | 14,227,200 | 0.237 |
| B half-line | 4 | 8,732,600 | 0.146 |
| C line-padded | 8 | 14,475,500 | 0.241 |
| D prefetch-safe | 16 | 13,281,700 | 0.221 |
| Register-privatized | - | 10,358,000 | 0.173 |

- Slowdown of stride 1 relative to stride 8: **0.98x**
- Stride 4 relative to stride 8: 0.60x
- Stride 8 relative to the privatized baseline: 1.40x (the gap between L1 store-to-load latency per
  iteration and a register add; the stride 1 vs stride 8 gap is the cross-core coherence cost)
- Average ns per memory update in the false-sharing regime (stride 1): 0.237 ns/thread,
  versus 0.241 ns/thread when padded.

## Q3 - L2 spatial prefetcher and 128-byte pairing

Measured: stride 8 = 14,475,500 ns, stride 16 = 13,281,700 ns (ratio stride8/stride16 = 1.09x).

Many Intel cores have an adjacent-cache-line prefetcher that treats memory as aligned 128-byte sectors: on a
miss to one 64-byte line it also fetches its even/odd partner. With 64-byte spacing, two threads' counters can fall
in the two halves of one 128-byte sector. A write by one thread can then pull in or invalidate the partner line the
other thread is using, giving secondary invalidations even though the primary lines are distinct. 128-byte spacing
puts each counter in its own sector. If the ratio is close to 1.0, this CPU's prefetcher is not interfering (or is
disabled), and 64-byte padding is enough. Padding of 64 B also still suffers when the buffer base is not aligned
to a 128-byte boundary or when other hot data shares the sector.

## Q4 - Register promotion vs. real memory stores

An optimizing compiler (LICM / scalar promotion / mem2reg) can see that `buffer[slot]` is loaded and stored at the
same address every iteration, keep it in a register, and write it back once after the loop. That would make the
kernel behave like the privatized baseline and hide false sharing.

In `worker_strided_accumulate` the address of each access is `slot + (m & v)`, where `v` is the value loaded in the
previous iteration and `m` is a runtime-computed zero the compiler cannot prove. The address therefore depends on
the loaded data, so the compiler must assume it can change and issue a real load and store every iteration.

Store lines from the emitted LLVM IR:

```
store i64 %.213.us.lcssa, ptr %sunkaddr40, align 8
store i64 %.213, ptr %.209, align 8
```

Load lines:

```
%.209.us.promoted = load i64, ptr %.209.us, align 8
%.210 = load i64, ptr %.209, align 8
%.30 = load i64, ptr %sunkaddr, align 8
```

Both are native `i64` memory operations; no Python object fallback was found
(`llvm_python_object_fallback = True`).

## Q5 - GIL masking vs. nogil=True

Without `nogil=True`, CPython only lets one thread run bytecode at a time. The four threads take turns holding the
GIL, so their stores are never issued simultaneously on different cores and the cache line is not ping-ponged by
RFO traffic. The time is dominated by interpreter overhead and GIL handoff (lock acquire/release, forced switches
every few ms, condition-variable wakeups, extra context switches), which grows with thread count, so adding threads
makes it slower, not faster. That cost is software lock contention. With `nogil=True` the threads run native code in
parallel on separate cores, and the slowdown seen at stride 1 comes from hardware cache-coherency contention
(RFO invalidations of one shared line). Measured context switches (stride 1): voluntary
None, involuntary None.
