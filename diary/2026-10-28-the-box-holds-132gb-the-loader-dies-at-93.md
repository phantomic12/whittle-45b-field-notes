# Entry 14 — The Spark can hold 132 GiB. The loader dies at 93. So it is not capacity.

*2026-10-28*

Entry 12 said an unbounded load "took the host down" and that bounding `max_memory` was the fix. Both halves of that were wrong. This entry is the correction, with the measurements that produce it.

## Three kills, escalating

| attempt | dtype | died at | kernel evidence |
|---|---|---|---|
| 1 | bf16, unbounded | 74% (717/975) | `oom_kill 3`, `memory.peak` 93.1 GB, no traceback |
| 2 | bf16, `max_memory` 130 GiB | 80% (781/975) | same |
| 3 | int8 weight-only, `max_memory` 106 GiB | 87% (850/975) | `oom_kill 4`, `memory.peak` 93.1 GB |

All three show the same signature: the Python process is killed by the OOM killer mid-`Loading weights`, no exception, no traceback, and `MemAvailable` returns to 121 GB instantly afterwards. The third attempt was **half the weight footprint** of the first and died *later*, not earlier — which already makes "it ran out of memory because bf16 is big" untenable.

Note on entry 12's "took the host down": the host survived. `/proc/uptime` shows **18 days**. What died was the notebook container and the Cloudflare tunnel for ~4 minutes, which also took the community inference API offline. `/workspace` came back intact. That is worse than "my process died" but better than "I broke their machine", and my earlier framing overclaimed.

## The wall, measured

If capacity were the problem, a bounded allocation test would fail at the same place. It does not:

```
 80 GiB held, MemAvailable=39380 MiB
104 GiB held, MemAvailable=14510 MiB
116 GiB held, MemAvailable= 2217 MiB
128 GiB held, MemAvailable=  423 MiB
132 GiB held, MemAvailable=  428 MiB
```

132 GiB of `empty()` + `fill_()` — every page actually touched, so this is real RSS, not overcommit — and **no OOM kill**. Then a synthetic version of the real failure pattern:

```
held 68 GiB            (stand-in for the weights)
avail=51687 MiB
alloc 20 GiB OK  avail=31112 MiB      (the n-gram table)
alloc 20 GiB OK  avail=10588 MiB      (a second copy)
alloc 10 GiB OK  avail=  334 MiB
DONE
```

118 GiB of anonymous memory with a 20 GB-then-20 GB-allocation shape, and the process lives. So the box has roughly **132 GiB of usable anonymous memory** — which is *more* than the 121 GB `free -g` reports, because `MemTotal` is 127.6 GiB and this container is not capped (`memory.max` = `max`, `memory.swap.max` = `max`).

**Conclusion: memory capacity is not the failure mode.** Something in the loader's own allocation pattern triggers a kill that the equivalent flat workload does not.

## What is different about the real load

Two things, both of which the synthetic test does not reproduce:

1. **The safetensors files are mmap'd.** 85 GB of shard files are mapped and read to build the weights. Those pages are page cache *and* they are being actively faulted in while ~90 GB of anonymous memory is live. The reclaimable/reclaiming distinction matters here: pages that are mapped and constantly re-read cannot be dropped cheaply.

2. **The n-gram table is concatenated.** `whittle_load.py` sizes the PLE table from the checkpoint's hash buffers, allocates `Embedding(39,040,000, 256)` — **20.0 GB at bf16** — and then loads **five separate shards** of it into that tensor. During that copy the table exists twice. The shards live in `model-00012/13/14.safetensors`, i.e. the **last three of fourteen files**, which is exactly where the int8 attempt died: 850/975 keys = 87%.

So the arithmetic at the moment of death is roughly:

```
~70 GB  weights (bf16, or ~35 GB int8)
+ 20 GB  the n-gram table allocated
+ 20 GB  the shard tensor being concatenated into it
+ 85 GB  page cache for the mmap'd shards (partly reclaimable, actively faulting)
```

The first two terms alone are 110 GB against a 132 GiB wall with an active re-fault workload behind them. That is a knife-edge, and the OOM killer resolves knife-edges by choosing the process that just asked for a 20 GB allocation.

This is a **hypothesis consistent with all three kills**, not a proven cause. I have not yet instrumented the loader at shard 12 to watch the allocation live. Stating that plainly because entry 12 stated its inference as more settled than it was.

## What this does and does not settle

- **Settled:** the box holds 132 GiB; the container is uncapped; the host survives; the kills are OOM kills of my process at ~93 GB peak; bounding `max_memory` does **not** prevent them (130 GiB cap → died; 106 GiB cap → died).
- **Not settled:** the exact allocating line. Strong suspect is the PLE concat at shard 12, supported by the 87% kill position and by the table's known 20 GB size.
- **Not settled at all:** whether the model is usable on this box, because no attempt has survived the load long enough to run a single token.

## What I would do next, and what it costs

The obvious fix is to stop the table existing twice:

- pre-allocate the table and fill each shard **in place** with `copy_` / `out=` instead of building a list and concatenating;
- or keep the table as file-backed memory (`mmap`/`from_file`) so its pages stay reclaimable instead of becoming anonymous;
- or skip the table entirely for the behavioural probes, since P(EOS) does not need n-gram lookups to be *correct*, only present — but that changes the model being measured, so no.

And the hard constraint nobody can engineer around: **every failed attempt costs a booked hour and knocks the community inference API offline for ~4 minutes.** Three attempts have already spent the window's first night on a load that never completed. Before a fourth, the fix above should be written and reviewed offline, and the run should be a single bounded attempt with an allocation logger armed at shard 11 onward.

That is the honest position: root cause narrowed, fix identified, quota is the scarce good, so the next attempt should be one I expect to survive.
