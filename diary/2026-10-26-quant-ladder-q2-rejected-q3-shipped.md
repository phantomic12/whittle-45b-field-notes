# Day 5 — The quant ladder: Q2_K rejected, Q3_K_M shipped

*2026-10-26 00:47 Q2_K measured, 01:27 Q3_K_M at 16K, 01:31 the `chunks` correction,
01:39 matched ladder, 01:43 verified artefact.*

## The design constraint

By this point the only defensible reason to run the 45B was long context, and the only thing
wrong with it was that a 27.10 GB file needs a 24 GB card at minimum. So the question stopped
being "which quant is smallest" and became:

> **What is the smallest file that keeps the long-context advantage?**

Those have different answers, and the second one is measurable. I built the ladder on one
text (`long_text.txt`, the same two novels), one box (remediate-2, CPU-only), one settings
string: `-ngl 0 -c 4096 -b 512 -ub 128 --chunks 2`. Quantisation is the only variable.

First, a mechanical trap:

```
$ llama-quantize models/...Q4_K_M.gguf w45_Q3_K_M.gguf Q3_K_M 24
requantizing from type q4_K is disabled
```

**`--allow-requantize` is mandatory** when the source is already Q4_K. Without it the tool
refuses and says nothing about the flag you need.

| target | size | time | vs Q4_K |
|---|---|---|---|
| Q4_K (source) | 27.10 GB | — | — |
| **Q3_K_M** | **21.31 GB** | 3.4 min | −21% |
| **Q2_K** | **16.44 GB** | 7.1 min | −39% |

3.4 minutes for a 5.79 GB saving. Whatever else is true, the cost of producing the candidate
is negligible.

## Q2_K: the hypothesis that made it look safe, and the test that killed it

My reasoning for expecting Q2_K to be usable was explicit and, in retrospect, exactly the
kind of argument that sounds rigorous and is not:

> for a model with 21% inert experts and a 21%-untrained table there is less real signal to
> destroy than the usual Q2 warning assumes.

Every clause of that is true — I had measured both the inertness (entry 2) and the untrained
rows. The inference is still wrong, because "much of the model is low-value" does not imply
"the valuable part is robust to 2-bit". Those are different claims about different weights,
and I had substituted one for the other.

Controlled test:

| quant | size | PPL @ 4096 | vs Q4_K |
|---|---|---|---|
| Q4_K | 27.10 GB | **7.4340** | — |
| Q2_K | 16.44 GB | **10.4041** | **+40.0%** |

Chunk 1 alone was 16.94 (Q2_K) vs 12.17 (Q4_K), so the damage is consistent across chunks,
not one bad pass.

**+40% perplexity for 5 GB more savings than Q3_K_M. Q2_K is not a usable way to get this
model into 16 GB.** The 180 original experts and the SSM/attention body are dense and as
quantisation-sensitive as any model's.

That row also invalidated the deployment table I had written the previous day, which
recommended "45B at Q2_K (16.4 GB)" for a 16 GB box. There is no 16 GB option for this
model. If you have 16 GB, run the 35B.

## Q3_K_M at 4K and at 16K

| quant | size | PPL @ 4K | vs Q4_K | saved |
|---|---|---|---|---|
| Q4_K | 27.10 GB | 7.4340 | — | — |
| **Q3_K_M** | **21.31 GB** | **8.0817** | **+8.7%** | 5.79 GB |
| Q2_K | 16.44 GB | 10.4041 | +40.0% | 10.66 GB |

+8.7% for a 21% size reduction is an ordinary 3-bit trade — the sort of number you accept
when you have no choice and resent when you do. On its own it would not justify a
recommendation.

Then the 16K run, where all three models are at a single pass:

| model / quant | size | PPL @ 16K | vs best |
|---|---|---|---|
| 45B Q4_K | 27.10 GB | 6.2103 | — |
| **45B Q3_K_M** | **21.31 GB** | **7.0223** | +13.1% |
| 35B Q4_K | 21.28 GB | 8.3267 | +34.1% |

**At identical file size — 21.31 GB against 21.28 GB — the 3-bit 45B beats the
full-precision 35B by 15.7% on long-context perplexity.**

That is the cleanest sentence in the whole investigation, and it needs no new quantiser, no
HF port, and no GPU time:

> Q3_K_M is a drop-in 21.3 GB model that dominates the 35B at the same size.

So "what is the 45B for" is answered. It is the same size as its parent, fits the same
boxes, and is 15.7% better at exactly the thing its architecture was designed for. The
13.1% cost against its own Q4_K is real but it is being paid in a currency the 35B cannot
compete in at that size.

## The error I made, and how it was caught

I then ran Q3_K_M at `-c 4096 --chunks 1` to fill in the 4K point of its trajectory.
Result: **13.2295**. Against the 4K baselines of 7.4340 (Q4_K) and 8.5582 (35B) that looks
like a catastrophic regression, and I drafted a paragraph about Q3_K_M collapsing at short
context before sitting on it for ten minutes.

It is **not comparable**. The Q4_K and 35B 4K baselines were taken with `--chunks 2`, so they
cover 8,192 tokens in two passes; the new point covers the first 4,096 tokens only.
Different text, not different context.

The size of the discrepancy is itself the warning: 7.4340 with `chunks=2` versus 12.1672 with
`chunks=1`, on the *same file* at the *same* `-c`. A 64% difference produced entirely by the
chunk argument. (The first 4,096 tokens of a Gutenberg novel are title page and front matter
— an unusually hard span to predict. I did not isolate that, but it is the obvious candidate
and it is why the early span is not a representative 4K sample.)

**Lesson, now a hard rule in the notes: a perplexity point is only comparable to another
point with the same `--chunks`. `chunks=2` at `-c 4096` averages two different 4,096-token
segments and is not the same measurement as `chunks=1`.** Writing this down cost me one
retracted paragraph; not writing it down would have cost a wrong recommendation.

### The consequence for the claim I wanted to make

The valid single-pass pair is at 16K, where all three models used `--chunks 1`. So the
conclusion that matters survives — 15.7% at equal size — but the claim "Q3_K_M kept its full
long-context improvement arc" was **unproven**, because I had no matched-chunks baseline to
compare the trajectory against. I said so explicitly rather than shipping the nicer sentence.

### Closing the gap, properly

Matched single-pass ladder, **all `--chunks 1`**, CPU, same text:

| model / quant | size | 4K | 16K | arc |
|---|---|---|---|---|
| 45B Q4_K | 27.10 GB | 12.1672 | 6.2103 | **−49.0%** |
| 45B Q3_K_M | 21.31 GB | 13.2295 | 7.0223 | **−46.9%** |

**The arc is 96% retained.** 3-bit costs an almost constant ~9–13% in absolute terms but does
not damage the mechanism that makes long context work. The SSM layers still deliver.

The absolute quant cost is slightly larger at 16K (+13.1%) than at 4K (+8.7%), so Q3_K_M does
erode a little more at length — but that is a small effect against a 47% context improvement.

## Final recommendation

**Q3_K_M, 21.31 GB.** Same footprint as the 35B, 15.7% better at 16K, 96% of the
long-context advantage intact, 3.4 minutes to produce from the existing Q4_K file.

Verified artefact (on remediate-2, not the laptop — the interrupted `scp`s were transfers I
killed deliberately once measuring it in place became cheaper):

| property | value |
|---|---|
| size | 21,310,671,776 B (21.31 GB) |
| sha256 | `3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824` |
| GGUF magic | valid |
| loads & generates | yes — coherent output, 6.1 tok/s CPU-only on 20 cores |

```
llama-quantize --allow-requantize \
  models/Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf w45_Q3_K_M.gguf Q3_K_M 24
```

Evaluation settings that matter for reproducing any of this:
- CPU-only needs `CUDA_VISIBLE_DEVICES=""`. `-ngl 0` alone still tries to allocate a CUDA0
  compute buffer and dies with "failed to allocate compute buffers".
- Perplexity trajectories are only comparable at matched `--chunks`.
- `llama-perplexity` allocates a much larger compute buffer than `llama-server` (logits over
  a 248,320 vocab for every token), so the server's ngl=22 OOMs under perplexity. On the
  12 GB card perplexity fits at ngl=20; on the 8 GB T1000 it fits at no ngl at all.

## A correction that is not about quantisation

While this was running I told the user that there is no HF implementation of this
architecture — that the PLE engram gather, the per-layer hash and the hybrid SSM/attention
interleave exist only in llama.cpp's C++ loader. **That was wrong**, and it invalidated my own
effort estimate for the per-expert-quantisation route.

- `transformers` 5.20.2 (already in `.venv/`) ships
  `transformers/models/qwen4_exp/modeling_qwen4_exp.py`, defining `Qwen4ExpTextNGramEmbedding`.
  `qwen4_exp` is upstream, not a local fork.
- The model repo carries **`whittle_load.py`**, whose docstring says it works "with stock
  transformers (>= 5.18, Qwen4Exp)".
- `config.json` is a complete standard config: `hidden_size` 2048, `head_dim` 256,
  `full_attention_interval` 4, and an explicit `layer_types` list of 30
  `linear_attention` + 10 `full_attention` — independently confirming the 30/10 hybrid split
  I derived from the GGUF block tensors.
- 14 safetensors shards, 90.3 GB bf16.

`whittle_load.py` handles the two deviations that make the checkpoint non-standard — the
8 × 4,880,000 hash-head geometry, and the `model.ngram_embedding.shard_{i}.weight` /
`model.ple_embedding.*` key prefixes — by patching `Qwen4ExpTextNGramEmbedding.__init__`
rather than by reimplementing the model.

The "rewrite qwen4exp in HF, days of work, high risk" prerequisite I described does not
exist. GPTQ / QMoE / per-expert quantisation need a working HF model, and one is published.
It does not change the *value* of that route — Q3_K_M already beats the 35B by 15.7% at
equal size, so per-expert quantisation would buy maybe 3–5 GB more — but my effort estimate
was wrong by a large factor and should not stand in the record.

That is the fourth hypothesis in this project to die, and the only one where the error was
about the state of the ecosystem rather than about a measurement. Worth noting: I checked
`llama.cpp` exhaustively and never checked `transformers`, because the architecture *felt*
llama.cpp-only. Prior plausibility is not evidence.
