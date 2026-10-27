# Day 2 — First perplexity, and the "strictly dominated" claim I had to withdraw

*2026-10-25. Ablation at 19:42, harness validation 19:47, code-domain rerun and router
falsification 20:33, k sweep 21:16, and the correction at 23:28.*

## The first number that changed my mind about the whole project

Everything up to this point had been capability probes — 20-item name cloze (18/20), hard
GSM (13/15), GSM-50. Those are noisy at small n and they invite motivated reading. I
wanted one continuous metric I could run in ten seconds on a CPU and trust.

Prose perplexity, base Q4_K_M, `-ngl 20`, 24 chunks of 512 tokens, fixed seed:
**6.9466**. Error bar on this text ±0.69 (measured by repeat runs). Every number in this
entry is the same text, same seed, same settings, so differences are the ablation and
nothing else.

## Ablation by zeroing bytes, not by inspecting routers

I could not load 90 GB of bf16 to delete modules, so I did not delete modules. In a K-quant,
a super-block whose scale `d` is 0 dequantises to exactly zero. An ablation is therefore a
byte write at a computed offset, and it is fully reversible from a saved backup.

This matters methodologically: it measures whether weights **matter**, which a routing
histogram cannot. An expert can be selected on every token and still contribute noise.

| condition | PPL | vs base | what was removed |
|---|---|---|---|
| base | 6.9466 | — | nothing |
| `experts:180-256` | 7.2144 | **+0.27** | 5.79 GB, the 76 spliced experts |
| `experts:0-8` | 7.3015 | +0.35 | 0.61 GB, 8 original experts |
| `experts:0-76` | 15.0565 | +8.11 | 76 **original** experts |
| `experts:76-152` | 14.9547 | +8.01 | 76 **original** experts |
| `experts:104-180` | 16.0702 | +9.12 | 76 **original** experts |
| `experts:0-180` | 383.19 | +376 | all 180 original |
| `per_layer_token_embd` (whole) | 584413 | — | the whole 10 B table |
| base after all restores | 6.9466 | **0.0000** | — |

The row that settles it is the same-mass comparison. 76 original experts cost **+8.0 to
+9.1 nats**. 76 spliced experts cost **+0.27**. A cost ratio of **31×**. And demolishing
**eight** original experts (+0.35) hurts *more* than demolishing all **seventy-six** spliced
ones (+0.27).

### Two things I checked before believing it

An ablation that finds "no effect" is exactly the shape of a broken harness, so I tested
the harness against a case where the answer is known.

1. **Zero the `hc_*` neck adapters** (0.28 GB, 323 tensors). Result: PPL **248320** —
   precisely the vocabulary size, i.e. the model predicts a uniform distribution over all
   tokens. That is total collapse and it is the *correct* answer, because hyper-connection
   necks carry the residual stream. Zeroing 323 tensors does annihilate the model, so the
   mechanism genuinely takes effect. The spliced experts' +0.27 is not a broken-harness
   artifact.
2. **Check 584413 is not information loss.** No loss of information can push perplexity
   above the uniform baseline — uniform is maximum entropy, and it costs exactly
   `log(248320) ≈ 12.42` nats → PPL 248,320. 584,413 is **2.35× worse than knowing
   nothing**, so those log-probs are not a valid distribution. The PLE path runs
   `grouped_norm(gated)` into a gate that clamps `|s|` to `1e-6` and takes a signed square
   root before a sigmoid; an exactly-constant all-zero stream drives it to a singularity.

I nearly published 584,413 as "the engram is load-bearing". It is a division by a clamped
zero. The graded sweep is what proves it — real degradation is monotonic, not
flat-then-cliff.

### The engram, graded

| fraction of table zeroed | PPL |
|---|---|
| 0% | 6.9466 |
| 10% | 7.0378 |
| 25% | 6.9331 |
| 50% | 7.0036 |

Zeroing half of it (2.81 GB) moves PPL by **+0.06**, inside the ±0.69 bar. (The 25% point
is *below* baseline, which tells you the bar is real and this curve is noise around a flat
line.)

### 21% of the table was never trained

Q4_K `d == 0` means all 256 weights in that super-block are exactly zero, which cannot
survive one gradient step. Control against a table that was certainly trained:

| table | rows | sampled blocks | zero-`d` | rate |
|---|---|---|---|---|
| `token_embd` | 248,320 | 20,000 | 0 | **0.00%** |
| `per_layer_token_embd` | 39,040,000 | 20,000 | 4,193 | **20.96%** |

Consistent with 25,178 training rows against 39 M addressable rows. Real text lands on
zero rows **14.6%** of the time (8,332 distinct rows addressed by a 1,110-token document) —
better than the table-wide 21%, same order.

### The router prefers the spliced experts, and it does not help

`ffn_gate_inp` row norms: original mean **0.7286**, spliced mean **0.8332** (1.144×
larger). By norm rank, spliced experts occupy top-8 slots at **8.0%** of (layer, expert)
pairs versus **1.06%** for originals.

So the router does not avoid them, it slightly prefers them — and the model still does not
care when they vanish. Selected confidently, contributing noise, removable at no cost. That
combination is the signature of weights in the wrong hidden space.

## The code rerun: my prose number was not the whole story

Same design, code corpus (34.9 KB of C, Python, Go, SQL plus a code-review prompt) instead
of prose. Base 5.5028.

| condition | PPL | delta | ×base |
|---|---|---|---|
| 76 spliced (180–256) | 6.1496 | **+0.647** | 1.12× |
| 76 original (0–76) | 8.2589 | +2.756 | 1.50× |
| 76 original (76–152) | 8.3486 | +2.846 | 1.52× |

| domain | original delta | spliced delta | factor |
|---|---|---|---|
| prose | +8.06 | +0.268 | **30.1×** |
| code | +2.80 | +0.647 | **4.3×** |

The factor fell from 30× to 4.3× and the honest reading is not the one I expected. Code is
just an easier domain for this model (base 5.50 vs 6.95), so *every* ablation costs less
there, originals included. The spliced set is still the cheaper one to remove, but it is
**7× less inert on code than on prose** — i.e. code uses them meaningfully more.

That is the first evidence these experts carry anything domain-specific, and it weakens
"pure dead weight". I recorded it and then, two hours later, over-read the prose number
anyway when writing the summary. The correction is at the bottom of this entry.

## Router rescaling: falsified, and it killed my own recommendation

A turn earlier I had recommended a learned additive router bias for experts 180–255, and
booked GPU time in my head for it. Before spending it, I simulated the bias by rescaling the
F32 router rows. Under softmax that is the same functional form — monotone in the logit,
reorders top-k identically.

| scale on experts 180–256 | PPL |
|---|---|
| 1.00 (baseline) | **6.9466** |
| 0.75 | 10.3797 |
| 1.50 | 7.2087 |
| 3.00 | 7.2397 |
| restored | 6.9466 |

**Every direction makes it worse.** Suppressing them is catastrophic (0.75 → 10.38);
favouring them is also worse (1.5 → 7.21, 3.0 → 7.24). The current weighting is already a
local optimum, and the loss is increasing in both directions.

So the cheap fix I proposed exists only if some scale beats 1.0. None does. Do not book the
A100. This is the first falsification in the project and it saved 15.5 CU.

*Caveat on one row:* `scale 0.00 → PPL 632` is **not** a valid "never route to them"
measurement. Scaling a row to zero sets logits to exactly 0, not −∞, so those experts still
compete and win on tie-break at arbitrary points. Use the ablation (+0.27) for the "routed
but useless" claim. The 0.5–3.0 sweep is valid because those are non-degenerate rescalings.

## k is a live knob, and raising it makes everything worse

`k` is not baked into weights. It is a GGUF metadata key:

```
--override-kv qwen4exp.expert_used_count=int:16
```

confirmed in the load log as `n_expert_used = 16`. No retraining, no new file. That makes
k the one cheap lever that would let the extra 76 experts participate more often, at k=8 of
256 each expert is drawn ~3% of the time.

| k | prose PPL | code PPL | gen t/s (ngl=20) |
|---|---|---|---|
| **8** | **6.9466** | **5.5028** | **25.4** |
| 12 | 7.2076 | 5.5736 | 23.6 |
| 16 | 7.7266 | 5.7534 | 21.4 |
| 24 | 8.9169 | 6.1334 | 18.4 |

Monotonically worse on both corpora, at a linear cost in expert traffic (k=16 ≈ 16% of gen
speed, k=24 ≈ 28%). Raising k adds noise, not capability.

### And k does not buy recall either

Earlier, 40 paired long-tail cloze items had hinted k=16 > k=8 (33/40 → 35/40), with the
flips looking exactly like restored long-tail recall (Sistine Chapel: "A. Raphael" →
"Michelangelo"). But 4 discordant items, McNemar **p = 0.625**. So I ran 200.

| k | cloze score | gen t/s |
|---|---|---|
| 8 | 179/200 = 89.5% | 25.4 |
| 16 | 182/200 = 91.0% | 21.4 |

Paired, same 200 items, greedy: **8 gained, 5 lost**, exact McNemar **p = 0.581**,
difference +1.5% with 95% CI [-2.0%, +5.0%] — spans zero.

k=16 is not worth 16% of generation speed. Ship k=8.

The failure *modes* do differ, and that is the useful part: **k=8 fails by emitting a fact
about the answer** ("awarded the Nobel Prize in 1945" for Fleming, "often considered the
greatest novelist of the twentieth century" for Proust) — it knows the neighbourhood, not
the entity. **k=16 fails by confidently substituting a wrong entity** (Caucasus for the
Europe/Asia range, Bach for Vivaldi, "Nay Pyi Taung" for Naypyidaw). So k is a *confidence*
dial, not a knowledge dial. That also explains the one paired GSM loss at k=16: "20% off $45
then 10% off" answered `32` instead of `32.4` — it rounded to the dollar. Commitment without
extra precision.

## Where I over-reached: "strictly dominated"

With cloze tied (180/180), GSM-50 favouring the 35B (44 vs 42, and my 42 vs the card's 38
at p ≈ 0.32), and every k setting worse than k=8, I wrote this:

> The 45B is **strictly dominated by the 35B** on everything measurable. Same table, same k,
> same active params, 6 GB larger file, no measured gain on any axis. The honest
> recommendation is unchanged and now has no remaining escape hatch.

That was wrong. Not the arithmetic — the *inference*. I had two axes, both narrow
(item-level recall and grade-school arithmetic), and I generalised to "any axis". The
sentence "no measured gain on any axis" was doing work that two probes cannot support, and I
had already written down the counter-evidence (code inertness 4.3× not 30×) that same
evening.

**The measurement that killed it, 23:28.** Matched settings, CPU-only on both boxes
(`ngl=0`, `-b 512 -ub 128`, 20 threads, same non-repetitive two-novel text). CPU
deliberately, because the two models live on different boxes with different GPUs — a GPU
comparison would confound model quality with hardware.

| model | PPL @ ctx 4096 | size |
|---|---|---|
| **45B** | **7.4340** | 27.10 GB |
| 35B | 8.5582 | 21.28 GB |

**13.1% better at general prose modelling.** 27% larger file, 13% better PPL.

### Revised picture

| axis | winner | margin |
|---|---|---|
| general prose PPL | **45B** | 13.1% lower |
| long-tail name recall (cloze 200) | tie | 180 / 180 |
| GSM-50 arithmetic | 35B | 44 / 42 |
| file size | 35B | 21.3 vs 27.1 GB |
| expert efficiency | 35B | 180/180 used vs 180/256 |

The extra capacity buys **general language modelling** — not the long-tail recall the card
claims, and not arithmetic. That story is coherent with the ablation: the spliced experts
are low-value *individually* (+0.27 for all 76), but collectively they still add 13%
modelling quality. Both results are true. I over-read the ablation.

**Lesson, and it is the one I keep re-learning: a low marginal value per byte is not zero
value.** The correct claim was "these experts are inefficient", not "these experts are
useless". An ablation measures the cost of removing a component from a model that has
adapted around it — it does not measure what the component contributes in a model trained
without it. Deleting 5.79 GB and calling the remainder equal is exactly the mistake that
argument structure invites.

One more correction from this evening, unrelated but worth recording: `ngl=20` was not the
GPU-layer ceiling. That number came from `llama-bench`; through `llama-server` with a real
HTTP request, ngl=22 is faster (27.41 t/s) and still fits. `llama-bench` excludes
chat-template and tokenisation overhead. Tune ngl through the server, because the server is
what users feel.
