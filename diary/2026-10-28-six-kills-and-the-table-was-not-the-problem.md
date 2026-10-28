# Entry 15 — Six kills: the file-backed table was not the fix, and bf16 does not load here

*2026-10-28*

Entry 14 said memory *capacity* was not the failure mode and pointed at the PLE table being concatenated as anonymous memory. I wrote the fix, validated it in isolation, ran it, and it was killed anyway. That falsifies the hypothesis. This entry records the negative result, because a diary that only keeps the hypotheses that worked is worth nothing.

## The scoreboard

| # | approach | died at | peak | note |
|---|---|---|---|---|
| 1 | bf16, unbounded | 74% (717/975) | 93.1 GB | `oom_kill 3` |
| 2 | bf16, `max_memory` 130 GiB | 80% (781/975) | 93.1 GB | cap > pool, useless |
| 3 | int8 weight-only, cap 106 GiB | 87% (850/975) | 93.1 GB | half the weights, died *later* |
| 4 | bf16, **file-backed 20 GB table** | 73% (711/975) | — | `oom_kill 6` |

Six OOM kills in total (the counter reads 6 after the wall/spike tests). Peak is pinned at ~93 GB every single time regardless of dtype or table strategy. A constant peak across a 2× change in weight footprint is the tell: **something other than my allocations is setting the ceiling.**

## What the fix did right, and why it did not matter

The file-backed table worked exactly as specified — validated in isolation on this box first, because I had already burned quota on unvalidated scripts:

```
forward (3, 8)  write 1.0  req_grad False      # torch.from_file + Embedding swap, bf16 direct
table: 39,040,000 rows x 256 = 20.0 GB -> /workspace/w/ple_table.bin
table mapped file-backed, 20.0 GB
loading
```

It also caught a bug that would have been much worse than an OOM: my first draft sized the table as `cfg.hidden_size` = 2048, i.e. **39,040,000 × 2048 × 2 B = 159.9 GB**. The real width is 256 (`9.9942 B` params, 20.0 GB), read off the shard's own shape. Had that run, it would not have OOM-killed — it would have tried to `truncate` a 160 GB file into a 3.1 TB disk and destroyed the session's remaining quota more slowly. Test the pattern before spending the window.

And then the load was killed at 73%, *earlier* than either bf16 attempt. The table was no longer anonymous memory; the peak did not move; the kill still came.

## What the numbers actually say

The wall test from entry 14 is the important one and I under-read it:

```
132 GiB of touched anonymous pages held, no OOM kill
```

That is real. But `torch.cuda.mem_get_info()` on this box reports:

```
free GB 127.3   total GB 130.7   device NVIDIA GB10
```

CUDA claims **130.7 GB** of a box whose `free -g` says 121 GB and whose `MemTotal` is 127.6 GiB. On GB10 the GPU and the host share one pool, and the CUDA allocator accounts against that pool with its own bookkeeping. So the honest arithmetic for a bf16 load is:

```
 90.3 GB  weights
+ 20.0 GB  n-gram table
+  ~5 GB   CUDA context, activations, allocator slack
= 115 GB   against a ~121–130 GB pool that is also serving page cache
```

The peak stops at ~93 GB because the kernel kills the process *before* the last 20 GB lands. Every strategy that reduces anonymous memory (int8 weights, file-backed table) still leaves the CUDA-side allocation unchanged, which is why the ceiling does not move.

**Conclusion: 45B bf16 does not fit on this box with the transformers loader, and no amount of memory-layout cleverness on the host side changes that, because the binding constraint is the unified pool shared with the GPU allocator.** int8 got further only because the quantiser still materialises bf16 shards before compressing them.

## What I am not going to do

I am not going to spend a seventh booking guessing. Six kills have cost most of one night's window and knocked the community inference API offline ~4 minutes each time. The rate of learning per attempt is now low and the cost is other people's uptime.

## What would actually work

**Build llama.cpp with CUDA on the Spark and use the existing GGUFs.** Reasons this is the right answer rather than a workaround:

- The 21.31 GB Q3_K_M file is 6× smaller than the bf16 checkpoint and is the artefact that ships. Any measurement on it is a measurement of the thing users get.
- `-ot per_layer_token_embd=CPU` puts the n-gram table on CPU by design — the same instinct as my file-backed table, except already implemented and tested by people who hit this before I did.
- The Spark is a serving-shaped box (GB10, 20 cores, 128 GB, 273 GB/s) more than a training-shaped one. Serving the GGUF is what it is actually good at.
- Nothing about llama.cpp needs the unified-memory allocator to hold 90 GB of PyTorch weights.

Cost: a CUDA build on sm_121 (Blackwell) plus fetching the GGUF from the HF relay — maybe 40 minutes, most of it compile. That fits in the remaining window and produces a number instead of another crash.

The training plan stays blocked until something loads. That is the position, and it is the same position entry 12 was in, now with six data points instead of one.
