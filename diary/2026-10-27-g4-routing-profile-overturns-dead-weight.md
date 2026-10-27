# Day 8 — The G4 routing profile that overturned "dead weight"

*2026-10-27 17:46. 3.14 CU of a 51 CU budget, one G4 session, ~21 minutes.*

## The last hypothesis standing

By the end of the falsification sweep (entry 2) exactly one claim about the spliced experts
was still untested: that they matter for **instruction-following and reasoning** rather than
for next-token prediction. Everything negative I had was perplexity, and perplexity is a
text-modelling metric. A component can be worth +0.27 nats to a language model's average
log-loss and still be the thing that makes arithmetic work.

That is a real gap, not a rhetorical one, so it justified GPU time — but only for
*profiling*, not training. Budgeted 16 CU; ran **3.14 CU (31 cents)** in ~21 minutes,
because the download hit 1.5 GB/s and load took 0.2 min.

## Method

`routing_profile.py` from the heal kit, unchanged, on the real 45B loaded bf16 entirely on
the GPU: 45.12 B params, 0 on CPU, **90.3 GB of VRAM of 102 GB available**. 32 forward passes
total — 8 GSM prompts, 8 name-recall prompts, one pass each. For every layer, count what
fraction of the selected experts fall in 180–255 (the restored set).

The reference point matters: with 76 of 256 experts restored, a router that ignored the
distinction would send **0.2969** of its selections there. "Share" below is measured against
that, and **skew** is share ÷ 0.2969, so skew 1.0 means "no preference".

### The bug I fixed mid-run

The first attempt pooled both prompt families into one `store` dict. `summarise()` keys
results by *layer name*, not by family, so pooling **destroys the comparison** — the output
is a perfectly plausible per-layer table with no gsm-vs-names contrast in it at all. Correct
form is one `store` and one hooks install **per family**, then compare the two summaries.

Recorded because the failure is silent and the output looks like a result. If I had not
noticed, this entry would have been "the router distributes experts evenly" with a table to
back it up.

## Result

| family | mean restored share | mean skew | median skew |
|---|---|---|---|
| **GSM (reasoning)** | **0.2686** | **0.905** | **0.948** |
| names (recall) | 0.1958 | 0.659 | 0.703 |
| *uniform expectation* | *0.2969* | *1.000* | *1.000* |

Difference: **+0.0728 absolute, +37% relative**. GSM prompts select the untrained experts
materially more than name-recall prompts do.

Per-layer, GSM skew exceeds names skew in **33 of 40 layers** (mean per-layer difference
+0.245). This is not one odd layer dragging an average upward; it is the consistent
direction of the model.

### The effect is concentrated in the middle

| layers | GSM share | names share | diff |
|---|---|---|---|
| 0–12 (early) | 0.247 | 0.182 | +0.065 |
| **13–26 (middle)** | **0.357** | **0.230** | **+0.127** |
| 27–39 (late) | 0.206 | 0.177 | +0.030 |

The middle band carries twice the differential of the early band and four times the late
band. The individual peaks are layer 28 (GSM skew 1.604 vs names 0.763), layer 19
(1.572 vs 1.259), layer 16 (1.317 vs 0.588), layer 23 (1.314 vs 0.707) and layer 26
(1.285 vs 0.834).

Layers 34–39 show low share in *both* families — the late layers barely route to restored
experts at all, for anything. So the late-layer flatness is not evidence about reasoning; it
is evidence that the last six layers ignore this expert bank entirely.

## What this does and does not establish

**Established:** the restored experts are preferentially recruited by reasoning-style
prompts, most strongly in the middle layers. They are not uniformly inert. The earlier
"30× inert on prose" result was a *text-modelling* measurement; this is a *routing*
measurement, and the two disagree. Both are true measurements of different things, which is
exactly why I ran this one.

**Not established:** that the recruitment *helps*. Being selected more often is not
contributing useful computation. These experts were spliced at original 35B weights scaled
by 1.009 and never trained in this hidden space, so they may be selected precisely *because*
they are ambiguous — a near-orthogonal projection produces a middling logit, which is what
gets picked when nothing is confident.

Two pieces of evidence for that reading, both from earlier measurements:

- The router-rescaling experiment showed the loss is minimised at the current weighting in
  both directions (0.75× → 10.38, 1.5× → 7.21, 3.0× → 7.24, baseline 6.9466). If they were
  being starved of a useful contribution, favouring them would have helped. It did not.
- GSM still routes **fewer** experts to the restored set than uniform: 0.269 against 0.297.
  The restored experts are under-used by *everything*; reasoning just under-uses them less.

So the honest sentence is: **the restored experts are under-used for everything, and
reasoning uses them 37% more than recall does.** That is a routing signal, not a quality
win, and the difference between those two claims is 15.5 CU of training.

## Behavioural probes: coherent, and chronically failing to stop

8 prompts, greedy, 160 tokens, on the same bf16 model.

| metric | value |
|---|---|
| content correct | **8/8** |
| emitted EOS | **2/8 (25%)** |
| mean P(EOS) at final position | **0.1972** |
| mean output length | 61.8 words |

The content is right. 12 apples → 13. 17 × 23 → 391. "the thing didnt work so we binned it"
→ "The solution was not effective, so we discarded it." Even the "answer in one word" probe
got Canberra. Arithmetic and instruction-following both work.

But **6 of 8 runs never emit EOS** and hit the 160-token cap, and the mean terminal P(EOS)
is **0.197**. That is the empty-content / reasoning-channel bleed from entry 4, now
quantified rather than described: the model does not know when to stop. A 0.20 terminal stop
probability is roughly a **5× longer expected generation** than intended, which is why the
earlier symptom was "burns the entire token budget and returns nothing".

This is a **serving** defect, not a knowledge defect, and it is independent of the expert
question. `--jinja` plus `enable_thinking: false` in the GGUF path already works around it;
the raw HF forward is where it shows. Anyone evaluating this model through transformers
without that flag will conclude it is broken.

## Revised recommendation

- The "publish the negative result" verdict **needs a correction**: this is not a pure
  negative. The experts carry measurable reasoning-domain routing signal in the middle
  layers, and the code-domain ablation was 7× less inert than prose. Three independent
  measurements now say "inefficient", and none says "dead".
- **Still do not train.** The heal premise was "these experts are dead and need reviving".
  The routing evidence says they are already being reached for. Training would optimise an
  assumption the measurement no longer supports, for 15.5 CU.
- **The interesting free experiment is ablation, not training**: zero experts 180–255 *only
  in layers 13–26*, then re-measure GSM. The existing ablation zeroed them everywhere, which
  averages a concentrated effect across 40 layers and would hide exactly this. That is a
  ~1 hour CPU job on the GGUF, not GPU time.

That last one is the experiment I would run next, and it is the one measurement that would
separate "selected because useful" from "selected because ambiguous". It is also the correct
response to a routing result: the layers tell you where to intervene.

## Two Colab facts that cost time and are not guessable

- **The root disk is 71 GB and the checkpoint is 90.3 GB.** `/dev/shm` is 93.4 GB of
  RAM-backed space on a G4 and is the correct destination. Set `HF_HOME=/dev/shm/hf`.
- **`colab upload` puts files at `/`, not `/content` or `~`.** `sys.path.insert(0, "/")`
  before importing the kit, or the import fails with the file visibly present.

## Closing position

Eight entries and four days, and the model landed somewhere I did not expect:

- worse than its parent at the two things its card sells (long-tail recall tied, GSM-50
  44 vs 42);
- better at the thing nobody advertised, increasingly so with context (13.1% → 25.4% →
  21.0%), until a 3-bit copy at the parent's exact file size is 15.7% ahead;
- carrying 5.79 GB of experts worth +0.27 nats individually and a measurable routing signal
  collectively, which is a strange and specific kind of asset;
- and shipped as a 21.31 GB file with 8/8 retrieval at 16K on a 21.4 GB card at 54 s/case.

The verdict I would defend: **Q3_K_M is worth running as a long-context model, the heal is
not worth running at all, and the spliced experts are inefficient rather than dead.** The
last clause took three corrections to get right, which is the argument for writing the
corrections down.
