---
date: 2026-10-28
title: "The table is live, the idle was me: 54% of accelerator time was wall clock"
budget: "$0 spent; Spark quota dry, so everything ran on the Arc B60 and remediate-2"
---

# The table is live, the idle was me: 54% of accelerator time was wall clock

*2026-10-28 — $0 spent, no accelerator booked. Spark quota is dry; everything below ran on hardware we already own.*

Entry 17 has the full five-point capacity curve for `per_layer_token_embd`. This entry is two
things that entry deliberately leaves out: proof that the table is *actually read*, and an
accounting of where accelerator time went. The accounting is the embarrassing one.

## The table is not dead weight — I checked

A flat perplexity curve across a memory-reduction axis has an obvious dangerous reading:
"the table doesn't matter, delete it". So I ran the sanity cell that settles it — zero **100%**
of the table and compare.

| | PPL |
|---|---|
| intact | **14.8310** |
| table fully zeroed | **43,348.2086** |

Three orders of magnitude. The tensor is read on every forward pass and is completely
load-bearing. Combined with entry 17's curve, the correct conclusion is narrow and specific:
the table is **fully live but roughly half-redundant**. About 2.15 GB of the 4.29 GB in this
quant can go without moving average-text perplexity.

The caveat that keeps this honest: **perplexity is blind to long-tail n-gram recall.** A table
that's silently shrunk doesn't fail on ordinary prose — it fails on rare tokens, names, and
exact continuations. A flat PPL sweep is a *sizing* result, not a *usage* result.

(Round-trip verified: the file restored to `3544d100…824`.)

## Where the GPU time actually went

Asked to keep accelerator idle near zero, I assumed there'd be a technical culprit. There were
two candidates, and both were innocent.

**Model reload.** A 24-chunk grid cell on the Arc B60, timed end to end: **wall 574 s**, of
which **load 26 s = 4.5%** and **compute 547 s = 95.2%**. On CPU the load is ~5 s of a 3880 s
cell — 0.13%. Reloading the model is not a meaningful cost at either end.

**File mutation.** Zeroing 2.15 GB of GGUF takes **2.8 s**; restoring takes **1.0 s**. That's
**3.8 s of accelerator idle per cell out of 574 s** — 0.7%. (`cp --reflink` is unsupported on
the B60's ext4, but at this size it doesn't matter.)

**So I totalled the actual runs.**

| run | span | wall |
|---|---|---|
| abl_recon (24-chunk Q3_K_M) | 21:45 → 22:48 | 1.1 h |
| q4k_chunks2 | 22:48 → 23:58 | 1.2 h |
| q4k24 | 00:05 → 01:23 | 1.3 h |
| ple_sweep (5 cells) | 10:41 → 15:06 | 4.4 h |

Span **17.4 h, busy 7.9 h = 46% utilisation**. One **9.3-hour gap** sits between `q4k24`
finishing at 01:23 and the sweep launching at 10:41. Nothing was broken, no job crashed, no
queue was stuck. The box was simply idle waiting for a human turn to say "go".

That's the whole answer. The fix isn't a cleverer harness — it's **queue the entire grid as one
detached job**, all cells up front, restoring between them, instead of launching one cell per
conversation turn.

## A 31% tax nobody noticed

Smoking the corpus turned up a separate waste: llama.cpp starts with **`n_threads = 8` on a
20-thread i7-12700T**.

| threads | s/pass (`-c 512`, 1 chunk) |
|---|---|
| default (8) | 27.28 |
| `-t 20` | **18.80** |

**31% faster.** It doesn't corrupt any ratio — every cell in the grid paid it equally — but
every CPU wall-clock estimate I've given was about a third too pessimistic. Pass `-t $(nproc)`
explicitly and verify the `system_info: n_threads =` line in the log.

## The B60 is a legitimate substitute box

With the Spark gone I needed somewhere to run GPU cells. The Arc Pro B60, same corpus, same
`--chunks 24`:

| | B60 (Vulkan, `-ngl 99 -fa on`) | remediate-2 (CPU, `ngl=0`) |
|---|---|---|
| cell wall | **574 s** | 3880 s |
| PPL | **8.5109** | 8.5104 |

**6.75× faster, PPL matching to 0.006%.** That's tighter agreement than CUDA-vs-CPU managed
(1.37% apart on absolutes), so the B60 can carry whole sweeps: 5 cells in ~48 min instead of
~5.4 h. For PPL work it replaces the Spark entirely.

## What I predicted vs what I got

I expected the table to be mostly dead weight, and I expected GPU waste to come from reload
overhead. Both wrong. The table is fully live but half-redundant, and the waste was scheduling,
not engineering. The 54% figure is not a hardware problem I can optimise away — it's a
turnaround-time problem.

## What this does not show

- Whether a *genuinely* smaller table (rehashed, collision-heavy) keeps quality. Not tested.
- Whether the free half survives NIAH or rare-token recall. Not tested, and PPL can't see it.
- Which rows are actually rare. I zero a suffix of every head, which is arbitrary with respect
  to n-gram frequency.

## Reproduce

```bash
# is the tensor live? zero 100% and compare PPL
python3 ple_used_test.py

# accelerator idle from file rewriting (this is ~0.7%, so don't bother optimising it)
python3 io_gap.py
```

## Next: batch the queue

Every remaining cell goes out as **one detached script** — all of them queued up front,
`-t $(nproc)` on every CPU leg, no per-cell human turn. That's the only lever that touches the
54%.
