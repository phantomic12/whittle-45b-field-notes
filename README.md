# Whittle-45B field notes

Field notes from four days (2026-10-24 to 2026-10-27) of evaluating, quantising and
trying to repair
[`logic65/Whittle-Qwen-3.8-45B-A3B`](https://huggingface.co/logic65) — a 45.1 B-parameter
mixture-of-experts that is, structurally, a Qwen3.6-35B-A3B with **76 extra experts per
layer spliced in from different weights and never trained**.

I went in expecting to find a battered but salvageable model and a heal worth paying GPU
time for. What I found instead was a model that is genuinely worse than its parent on the
two axes its own model card sells (long-tail recall, arithmetic), genuinely better on one
axis nobody advertised (long-context modelling), and carrying ~5.8 GB of experts whose
marginal value is an order of magnitude lower per byte than the ones it shipped with — but not zero, which
took me three attempts and one wrong verdict to establish.

Every number below is measured on this hardware. Where a result was later corrected, the
correction is in the same entry and dated. Where a hypothesis died, it has its own section,
because the falsifications turned out to be the most useful artefacts this work produced.

## The model, as measured rather than as described

| property | value | how |
|---|---|---|
| total parameters | **45.1168 B** | summed from GGUF tensor headers |
| routed experts | 32.2123 B (71.4%) | |
| n-gram table (`per_layer_token_embd`) | 9.9942 B (22.2%) | 5 shards, 39,040,000 rows = 8 heads × 4,880,000 × 256 |
| always-on weights | 2.910 B | |
| layers | 40 — **30 linear/SSM + 10 full attention**, interleaved every 4th | `layer_types` in config.json, cross-checked against block tensors |
| experts | 256 per layer, k=8 active | |
| active params at k=8 | 3.917 B (**14.1%** of a dense 27B) | |
| KV cache | **20.0 KB/token** f16 (10.0 KB/token at q8) | 2 KV heads × 256 dim × 10 layers × 2 B |
| spliced expert scope | 76/256 × 40 layers = **9.56 B** base params (21%) | |
| bf16 checkpoint | 14 safetensors shards, 90.3 GB | |

The splice: experts at indices 180–255 were copied from Qwen3.6-35B-A3B at original
weights, scaled by a median norm ratio of **1.009**, and never trained. Their SwiGLU
content sits in the wrong hidden space for this model.

## Headline results

| finding | measurement |
|---|---|
| **Deployment quant: Q3_K_M, 21.31 GB** | +8.7% PPL vs Q4_K at 4K — and **15.7% better than the full-precision 35B at identical file size** (21.31 vs 21.28 GB) at 16K |
| Long-context advantage grows | 45B is 13.1% better than 35B at 4K, **25.4% at 16K**, 21.0% at 32K; the 35B *degrades* +5.8% by 32K, the 45B never does |
| Q2_K rejected | **+40.0% PPL** (7.4340 → 10.4041); 16.44 GB is not a usable size for this model |
| Needle-in-haystack, 16K | **8/8 exact** at all four depths, on CPU *and* on an Intel Arc Pro B60 |
| Arc Pro B60 throughput | 54.0 s/case vs 555 s CPU-only = **10.3×**; 21.31 GB model in 21.4 GB VRAM via partial offload (~17.8 GB resident, ~3.5 GB in RAM) |
| Spliced experts, marginal cost | ablating all 76 costs **+0.268 PPL** (+0.038 nats); 76 *original* experts cost **+8.11 PPL** (+0.77 nats) — an order of magnitude per byte, and the damage is **4.9× worse on Q3_K_M than Q4_K at matched flags** |
| …but they are not dead | GSM prompts route to them **+37% relative** more than name-recall prompts, concentrated in **layers 13–26** |
| Behavioural probes | 8/8 content correct, but only **2/8 emit EOS**, mean terminal P(EOS) **0.1972** — a serving defect, not a knowledge defect |

## What did not survive testing

These are the results I would want if I were the author. Each one is a plan I had, a
prediction I made, and the measurement that killed it.

1. **"There is less signal to destroy, so this model should quantise better than most."**
   False. Q2_K costs +40.0% perplexity. The 21% untrained table and the low-value experts
   do not make the parts that matter any less sensitive.
2. **"A learned router bias could recover the spliced experts."** Falsified. Every
   rescaling of the 180–255 router rows — 0.75×, 1.5×, 3.0× — is *worse* than baseline
   (10.38, 7.21, 7.24 vs 6.9466). The current weighting is already a local optimum. I had
   recommended booking an A100 for this; the test says do not.
3. **"The 45B is strictly dominated by the 35B."** Wrong, and I committed to it in
   writing. It was based on two narrow probes (name cloze, GSM-50). At matched settings
   the 45B is 13.1% better at prose perplexity at 4K, and the margin *grows* with context.
   A low marginal value per byte is not zero value.
4. **"The spliced experts are pure dead weight."** Too strong. The 30× inertness factor is
   a prose number; on code it is 4.3×. And a routing profile on a G4 shows they are
   preferentially recruited by reasoning prompts. Selected is not the same as load-bearing —
   but "dead" was the wrong word.
5. **"The engram table is load-bearing."** Unsupported. Zeroing 50% of it moves PPL by
   +0.06, inside a ±0.69 error bar. The dramatic 584,413 figure for zeroing 100% is a
   numerical singularity in the PLE gate, not information loss — proven by an ablation of
   the `hc_*` necks that returns PPL = 248,320, exactly the vocabulary size, i.e. a uniform
   distribution over all tokens.
6. **"k=16 buys long-tail recall."** Not at n=200. 8 gained / 5 lost, McNemar p=0.581,
   95% CI [-2.0%, +5.0%]. k is a confidence dial, not a knowledge dial. Ship k=8.
7. **"ngl=20 is the GPU-layer ceiling."** Wrong — that came from `llama-bench`, which
   excludes chat-template and tokenisation overhead. Through `llama-server` with a real
   HTTP request, ngl=22 is faster (27.41 t/s) and still fits.
8. **"There is no HF implementation of this architecture, so per-expert quantisation needs
   a port."** Wrong by a large factor. `transformers` 5.20.2 ships
   `models/qwen4_exp/modeling_qwen4_exp.py`, and the model repo ships `whittle_load.py`.
   The port I described as "days of work, high risk" does not exist as a prerequisite.

## Diary index

Entries are ordered by the commit timestamps of the measurements they document, not by the
order I originally planned to write them. That plan had the quant ladder before the
long-context work; the ladder needs a matched single-pass baseline, so it came second, and
the index reflects what actually happened.

| # | date | entry | what it settles |
|---|---|---|---|
| 1 | 2026-10-24 | [Anatomy: a 35B wearing 76 borrowed experts](diary/2026-10-24-anatomy-35b-plus-76-spliced-experts.md) | the splice, the parameter split, and four loader failures that all look like data problems |
| 2 | 2026-10-25 | [First perplexity, and the "strictly dominated" claim I had to withdraw](diary/2026-10-25-perplexity-and-the-strictly-dominated-mistake.md) | ablation harness (+0.268 vs +8.11 PPL), router rescaling falsified, k sweep, then the 4K measurement that killed my own verdict |
| 3 | 2026-10-25 | [Long context is the one real advantage](diary/2026-10-25-long-context-the-one-real-advantage.md) | 13.1% → 25.4% → 21.0% gain from 4K to 32K, 35B degrades, 20 KB/token vs 98 KB for a dense 27B |
| 4 | 2026-10-25 | [Serving bugs and a legitimate zero](diary/2026-10-25-serving-bugs-and-a-legitimate-zero.md) | empty-content bug on 3/6 creative prompts, GSM8K 1198/1319, MiMo 0/22 measured on real tests |
| 5 | 2026-10-26 | [The quant ladder: Q2_K rejected, Q3_K_M shipped](diary/2026-10-26-quant-ladder-q2-rejected-q3-shipped.md) | +40% vs +8.7%, the 15.7% equal-size win at 16K, and the `chunks=1` vs `chunks=2` error that invalidated one point |
| 6 | 2026-10-26 | [Needle in a haystack: 8/8, after three harness bugs produced a false negative](diary/2026-10-26-needle-in-haystack-and-three-harness-bugs.md) | retrieval survives 3-bit at 16K and is depth-independent; the raw-prompt bug that made the model continue the document |
| 7 | 2026-10-26 | [Arc Pro B60: 10.3×, after four separate blockers](diary/2026-10-26-arc-pro-b60-four-blockers-ten-x.md) | qwen4exp is not upstream, SYCL aborts, Vulkan works, 21.31 GB fits 21.4 GB by partial offload |
| 8 | 2026-10-27 | [The G4 routing profile that overturned "dead weight"](diary/2026-10-27-g4-routing-profile-overturns-dead-weight.md) | +37% relative recruitment for reasoning, layers 13–26, and probes showing 2/8 EOS |
| 9 | 2026-10-27 | [Ablation: the experts are load-bearing, and the damage is superlinear](diary/2026-10-27-ablation-experts-are-load-bearing.md) | +2.48% PPL from ablating 14 middle layers, +15.42% from all 40 — 2.2x the proportional prediction |
| 10 | 2026-10-27 | [The harness was worth 5.3%, and the rest of the gap is real](diary/2026-10-27-harness-cost-5-percent-and-the-nats-label.md) | chunk count alone moves baseline +5.30%; at matched flags Q3_K_M damage is 4.9× Q4_K; and the "+0.27 nats" headline was PPL points — the nats ratio is 20.5×, not 31× |

## Provenance

Every figure here comes from measurements logged in a private lab notebook (~1,371 lines)
and the JSON artefacts of the runs themselves. Nothing is estimated from a model card, and
nothing is extrapolated without saying so. Where a number in an early entry was later shown
to be invalid, the entry says which one and why rather than being quietly edited — the error
rate of my own measurement pipeline is part of the record.

**Read the `--chunks` count before comparing any two perplexity numbers in this repo.**
Averaging 2 windows instead of 24 moves the baseline by +5.3% on the same file, and moves an
ablated model by +9.0%, so a mismatch between harnesses inflates a delta by ~1.26×. Every
table states its flags; entry 10 is the measurement of what happens when they do not match.

Hardware referenced throughout:

| box | CPU | GPU | role |
|---|---|---|---|
| laptop (Zephyrus) | Ryzen 9 7945HX, 32T | RTX 4080 Laptop 12 GB, PCIe 4.0 **x8** | 45B Q4 at 27.4 t/s, evaluation |
| remediate | i7-12700T, 20T | T1000 8 GB | 35B at 7.2 t/s, Docker grading |
| remediate-2 | i7-12700T, 20T | T1000 8 GB | CPU-only perplexity and NIAH, 20 threads |
| garlic-clove | — | **Intel Arc Pro B60** (BMG G21, 21414 MiB) | Vulkan inference, 10.3× the CPU rate |
| Colab G4 | — | 96 GB, 102 GB available | bf16 routing profile, 3.14 CU |

## Status

Notes, not a release. The Q3_K_M artefact discussed in entries 5–7 is 21,310,671,776 bytes,
sha256 `3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824`, reproducible from
the published Q4_K_M in 3.4 minutes with `llama-quantize --allow-requantize`.
