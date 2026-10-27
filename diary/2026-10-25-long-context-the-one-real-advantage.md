# Day 3 — Long context is the one real advantage

*2026-10-25 22:46 first 45B-only curve, 23:43 the matched cross-model run, 2026-10-26 00:29
complete curves to 32K.*

## Why this was the right place to look next

After entry 2 the scoreboard read: better than its parent at prose PPL, tied on name recall,
worse on arithmetic, 6 GB larger. That is a model with no reason to exist *unless* one of
its two structural oddities pays for itself. There are exactly two: the 20 KB/token KV
cache, and the 30-layer SSM stack that carries state without a growing cache. Both are
long-context arguments. So long context was not a nice-to-have experiment; it was the last
honest reason to keep the model.

Also, I had written in the previous day's notes that "if the 45B holds up better at 64K+
than the 35B, that is a real, defensible reason to run it". This is that test, minus the
64K (see below).

## Method, and why the text matters

Two concatenated public-domain novels, 600 KB, non-repetitive. Repetitive text collapses
perplexity and would hide exactly the long-range effect I was trying to measure — a model
that can copy the previous paragraph scores well at any length. Fixed settings:
`-ngl 0 -b 512 -ub 128`, 20 threads, CPU-only.

CPU-only for the cross-model comparison, because the two models live on different boxes with
different GPUs. Running both on CPU is slower and it is the only way the comparison is about
the models.

## First pass: 45B alone, and a flat-to-improving curve

| ctx | chunks | PPL | vs 4K |
|---|---|---|---|
| 4,096 | 8 | 7.1832 | — |
| 8,192 | 4 | 7.2753 | +0.09 |
| 16,384 | 2 | 6.8991 | −0.28 |
| **24,576** | 1 | **6.6741** | **−0.51** |
| 32,768 | 1 | 7.1262 | −0.06 |

Perplexity does not degrade with context. A dense model typically shows rising PPL as
context grows — more distant tokens, more chances to be distracted. This one is
flat-to-better out to 32K.

**Two caveats I wrote down that same evening, and they matter:**

- **`--chunks > 1` splits the text into independent passes.** Only the `chunks=1` rows
  (24,576 and 32,768) are true single-pass long-range measurements. The 4K/8K/16K rows are
  not "context length" experiments, they are *different text*. The apparent dip at 16K and
  24K could easily be the chunking, not the context.
- **32K is the ceiling of the tool, not the model.** 64K died with `std::bad_alloc`:
  `llama-perplexity` allocates `ctx × vocab × 4` bytes for logits, and
  65,536 × 248,320 × 4 = **65 GB**. 49,152 would need 49 GB. So the architecture's
  262,144 context is untestable with this binary on any box I have.

Which meant the first curve answered nothing about the model, only about my harness. The
comparison had to be at matched chunking, against the 35B.

## Matched cross-model run: the gap appears

Same text, same settings, both models CPU-only. 16K here is a single pass, so it is a
genuine long-range measurement.

| ctx | 45B | 35B | 45B gain |
|---|---|---|---|
| 4,096 | 7.4340 | 8.5582 | **13.1%** |
| 16,384 | **6.2103** | 8.3267 | **25.4%** |

And the mechanism, which is the part that makes this a finding rather than a number — each
model's own improvement from 4K to 16K:

| model | 4K | 16K | improvement |
|---|---|---|---|
| 45B | 7.4340 | 6.2103 | **−16.5%** |
| 35B | 8.5582 | 8.3267 | **−2.7%** |

**Given more context, the 45B gets 16.5% better and the 35B gets 2.7% better.** The gap more
than doubles. The 45B is not merely better at long text — it is better *because of* the
text, at six times the rate. That is the signature of the 30 SSM layers actually carrying
long-range state while the 10 sparse-attention layers keep the cache small.

This is also the measurement that withdrew the "strictly dominated" verdict from the
previous evening. It arrived 15 minutes after I wrote the retraction, and it made the
retraction look conservative.

## Complete curves: the 35B degrades, the 45B does not

| ctx | 45B | 35B | 45B gain |
|---|---|---|---|
| 4,096 | 7.4340 | 8.5582 | **13.1%** |
| 16,384 | **6.2103** | 8.3267 | **25.4%** |
| 32,768 | **7.1506** | 9.0503 | **21.0%** |

Own trajectory relative to each model's own 4K baseline:

| model | 4K | 16K | 32K |
|---|---|---|---|
| **45B** | 7.4340 | **−16.5%** | **−3.8%** |
| 35B | 8.5582 | −2.7% | **+5.8%** |

The 35B gets **worse** with long context: +5.8% PPL at 32K. The 45B never does — its worst
long-context point is still 3.8% better than its own 4K baseline, and its best is at 16K.
The 45B wins at every length, and the advantage is largest exactly where long context
matters.

**Honest limits on this.** The 16K dip followed by a 32K rise on the 45B is not monotonic,
so treat the precise shape of the curve as noise-limited. The sign is consistent and the
35B's degradation points the opposite way, which is the part I trust. It is a single
document: one data point per length, not a curve with error bars. And it is perplexity —
strong evidence the SSM layers carry long-range state, but a retrieval task is the
confirming check, not another PPL point. (That check is entry 6, and it passes 8/8.)

## The economics that make it matter

This is the part that turns a perplexity win into a deployment argument.

| context | f16 cache | q8 cache | as % of 27.1 GB of weights |
|---|---|---|---|
| 16,384 | 0.33 GB | 0.16 GB | 1.2% |
| 32,768 | 0.67 GB | 0.34 GB | 2.5% |
| 131,072 | 2.68 GB | 1.34 GB | 9.9% |
| 262,144 | 5.37 GB | 2.68 GB | 19.8% |

20.0 KB/token, because only 10 of 40 layers have a growing cache (2 KV heads × 256 dim ×
10 layers × 2 bytes). A dense 27B with conventional attention — 48 layers, 8 KV heads, 128
head dim, f16 — costs **98 KB/token**, which is 3.2 GB at 32K *before a single weight is
loaded*. The 45B's cache is ~5× smaller.

At 16K the cache is 1.2% of the weight budget. At 128K it is still under 10%. **Context
length is nearly free on this model, which is not true of a dense 27B** — and that is
architectural, not a quantisation trick, so it survives every quant in entry 5.

## Deployment verdict as it stood

| use case | recommended | why |
|---|---|---|
| long documents, RAG, agent loops | **45B** | 21% better at 32K, and it does not degrade |
| quick chat, arithmetic, tight RAM | 35B | 21.3 GB, 44 vs 42 on GSM-50, same quality at 4K within 13% |
| 16 GB box | 45B at Q2_K (16.4 GB) | *this row was wrong — see entry 5, Q2_K costs +40% PPL* |

The third row was a guess dressed as a recommendation: at that point I had no Q2_K
perplexity measurement, only the assumption that a model with 21% inert experts would
quantise gracefully. It does not. I am leaving the row in because I wrote it, and because
"it should quantise well given how much of it is dead" was a reasonable-sounding inference
that the next day's controlled test destroyed.

## What this changed

The model stopped being a failed splice and became a **long-context model with an
inefficient expert bank**. That reframing is what made the quantisation work worth doing:
if the only axis the 45B wins is long context, then the deployment question is "what is the
smallest file that keeps the long-context advantage", not "what is the smallest file". Those
have different answers, and entry 5 measures both.
