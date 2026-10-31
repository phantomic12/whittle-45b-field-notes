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
| Spliced experts, marginal cost | at matched settings on one box, ablating all 76 costs **+23.1% PPL on Q4_K** and **+19.4% on Q3_K_M** (`--chunks 24`, CPU, one corpus) — the quant barely changes the damage (ratio 1.19×). At `--chunks 2` the same two cells read +17.4% and +15.4%: the harness, not the quant, is what moves this number. 76 *original* experts cost **+8.11 PPL** on the earlier laptop run (see entry 11: the absolute figures from that run do not reproduce on CPU) |
| …but they are not dead | GSM prompts route to them **+37% relative** more than name-recall prompts, concentrated in **layers 13–26** |
| Behavioural probes | 8/8 content correct, but only **2/8 emit EOS**, mean terminal P(EOS) **0.1972** — a serving defect, not a knowledge defect |
| Engram table capacity, 4K | it is **4.29 GB in the shipped Q3_K_M**; keeping only **50% of the rows costs −0.36% PPL on prose** (z = −0.25) but **+28.73% on a rare-token corpus** (z = +11.86, entry 20) — five times the +5.48% entry 19 first measured, because 19 ran on a local requantisation and 20 ran on the hub file. Row capacity is not load-bearing on ordinary text and *is* load-bearing on the long-tail inputs the model card advertises — flat at 62.5% and 56.25% kept, then a cliff, so the threshold sits in (56.25%, 50%]. The damage is concentrated: **two of 24 chunks account for most of it** (5.94× and 4.49×), 19 of 24 are within 1.23× |
| Per-tensor SSM protection | **not worth its bytes.** Pinning `ssm_out`/`attn_gate` to q8_0 and `ssm_alpha`/`ssm_beta` to f16 costs **+245.4 MiB (+0.95%)** and moves PPL **+0.254%, z = +0.18** — indistinguishable from zero, in the wrong direction (entry 21). The control build is byte-identical to its source, so there is no requantisation confound |
| UD-style recipe | **dominated.** 19 `--tensor-type` overrides that demote the engram table to Q3_K, split the expert ladder into Q3_K/Q4_K/Q6_K and pin the always-on set to q8_0 give a file **1.12% smaller and 1.71% worse** (7.6821 → 7.8133, entry 24) — 3.7× the ladder's own rate for that saving, worse in **24 of 24 chunks**. A transferred imatrix changes **+0.003%, z = 0.00** |
| 20 GB-card recipe | **beats the shipped file where it counts.** Pin the engram table at **Q6_K** (7.64 GiB, charged to RAM) and cut the body to **17.04 GiB** and the file reads **7.9581 ±0.08082** — **+3.593% vs Q4_K_M** but **−6.490% vs the shipped Q3_K_M (8.5104)** for only **+7.50% body bytes**, i.e. **1.66× the ladder's efficiency per on-card byte** and **2.1× per body byte**, while being **3.23× worse per total file byte** (entry 25). It is a bigger file (24.68 vs 19.85 GiB); the whole value depends on the table living in RAM. **Where to put the cut was the next question and I had it backwards:** cutting `ffn_down_exps` instead of gate/up is strictly dominated — **320 MiB more body and +4.389% worse**, 24/24 chunks (entry 26). Cutting down to Q2_K costs **1.053 %PPL per %body-byte**, 2.0× the ladder's own rate. The better build if the card has 18.4 GiB of body to give is `w45_v5B` (gate/up Q4_K, down Q3_K, body 18,805.09 MiB) at **7.8502 = −7.758% vs the shipped Q3_K_M**. Neither has been loaded on a 20 GB card |
| Throughput on a 21.42 GiB file is set by RAM, not VRAM | two boxes with the *same* i9-12900H, RTX 3060 6 GB, driver and binary, differing only in RAM (31 vs 15 GB), read **195.2 vs 22.5 t/s** pp256 = **8.7×**. The 15 GB box holds **63%** of the file in page cache and re-reads **1.08–1.24× the whole model off NVMe per 2048-token prefill** at **0.30 GiB/s**; the 31 GB box reads **0 MiB**. Forcing eviction on the fast box with a 14 GiB anonymous hold reproduces **20.21 t/s** (entry 22) |

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
4. **"The spliced experts are pure dead weight."** Too strong, and the size of the error was
   only found three entries later. The "+0.27 PPL, removable at no cost" figure was measured
   at `ngl=20` on the laptop; at `ngl=0` on CPU the same ablation costs **+1.30 PPL** and
   **+17.4%**. The 30× inertness factor was a GPU-baseline artefact. On a CPU baseline with
   matched settings the spliced experts cost **+16%** perplexity to remove, on both 4-bit and
   3-bit. They are inefficient, not inert.
5. **"The engram table is load-bearing."** Unsupported *on prose*, now across five fractions rather than one.
   Zeroing 50% of it moves PPL by +0.06, inside a ±0.69 error bar (entry 5); the full sweep in
   entry 17 finds **−0.39% (z = −0.27) at 50% kept** and nothing measurable down to 2.15 GB, with a
   mild trend toward *less* table that I do not defend as a gain. The dramatic 584,413 figure for
   zeroing 100% is a numerical singularity in the PLE gate, not information loss — proven by an
   ablation of the `hc_*` necks that returns PPL = 248,320, exactly the vocabulary size, i.e. a
   uniform distribution over all tokens. **Partly reinstated by entries 19–20:** on a corpus built
   to fire the table, and on the file that is actually shipped, it is load-bearing at +28.73%. The
   correction to this item is that the flat prose sweep was measuring the corpus, not the table.
6. **"k=16 buys long-tail recall."** Not at n=200. 8 gained / 5 lost, McNemar p=0.581,
   95% CI [-2.0%, +5.0%]. k is a confidence dial, not a knowledge dial. Ship k=8.
7. **"ngl=20 is the GPU-layer ceiling."** Wrong — that came from `llama-bench`, which
   excludes chat-template and tokenisation overhead. Through `llama-server` with a real
   HTTP request, ngl=22 is faster (27.41 t/s) and still fits.
8. **"There is no HF implementation of this architecture, so per-expert quantisation needs
   a port."** Wrong by a large factor. `transformers` 5.20.2 ships
   `models/qwen4_exp/modeling_qwen4_exp.py`, and the model repo ships `whittle_load.py`.
   The port I described as "days of work, high risk" does not exist as a prerequisite.
9. **"A few fragile tensors are where a quant loses quality, so protect them."** Not here.
   Pinning `ssm_out`/`attn_gate` to q8_0 and `ssm_alpha`/`ssm_beta` to f16 costs **+245.4 MiB**
   and moves perplexity **+0.254% at z = +0.18** — inside the error bar and in the wrong
   direction (entry 21). The fragility on this model sits in 9.56 B of spliced experts and a
   9.99 B table, which together are 43% of the file; 245 MiB of small always-on tensors was
   never the lever.
10. **"Deleting table rows should cost damage in proportion to how many are deleted."** It
    does not. A `frac^8` survival model predicted +0.39%, +1.35% and +3.98% at 37.5%, 43.75%
    and 50% zeroed; measured **−0.46%, −0.44% and +28.73%** (entry 20). Monotone in structure,
    wrong in sign at two points and 7.2× low at the third, because the real behaviour is a
    threshold and the per-chunk vector shows the loss landing in **two chunks out of 24**.
11. **"Spend bytes on the experts and the table, give them back on the small tensors, and the
    file gets better for the same or less size."** Falsified as built. A 19-override UD-style
    recipe produces a file **1.12% smaller and 1.71% worse** (7.6821 → 7.8133, entry 24) —
    **3.7× the quant ladder's own exchange rate** for that saving, and worse in **24 of 24
    chunks** rather than in a couple. The transferred imatrix I expected to carry the result
    moves PPL **+0.003%, z = 0.00**. The mistake underneath it: I cited entry 17's finding that
    *half the table's rows can be deleted* as evidence that the table can be *stored at 3 bits
    instead of 4* — row capacity and per-row precision are different operations on the same
    tensor, and only the first was ever measured.
12. **"The down projection is the fat, redundant expert tensor, so make it the one that pays."**
    Backwards. `ffn_down_exps` at **Q2_K** with gate/up kept at Q4_K produces a body **320.0 MiB
    larger** than entry 25's and **+4.389%** worse (8.3074 vs 7.9581, 24/24 chunks, t = 17.9) —
    dominated, not merely inefficient output. The clean single-variable cell puts that one step
    at **+5.824% for −1,040 MiB = 1.053 %PPL per %body-byte, 2.0× the ladder's 0.520** (entry
    26). The error underneath it was arithmetic I never did: gate/up at Q4_K instead of Q3_K
    *adds* 2,720 MiB, which is more than the 2,400 MiB the down demotion saves.

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
| 10 | 2026-10-27 | [The harness was worth 5.3%, and the rest of the gap is real](diary/2026-10-27-harness-cost-5-percent-and-the-nats-label.md) | chunk count alone moves baseline +5.30% — but its headline "4.9× worse on Q3_K_M" compared different *hardware*, not matched flags. **Superseded by entry 11 within the hour.** |
| 11 | 2026-10-27 | [Q4_K is more damaged than Q3_K_M, not 4.9× less](diary/2026-10-27-q4k-is-more-damaged-than-q3-not-less.md) | at matched settings **Q4_K +17.44%** vs **Q3_K_M +15.42%** — quant level barely matters. Also: the original "+0.27 PPL, removable at no cost" was an `ngl=20` GPU number that does not reproduce on CPU (+1.30 PPL), so the "dead weight" verdict was wrong by ~5× |
| 12 | 2026-10-28 | [The Spark host died on an unbounded load](diary/2026-10-28-the-spark-host-died-on-an-unbounded-load.md) | a booked training window is not a running process: the 45B bf16 load took the shared DGX Spark's Cloudflare tunnel down (1033/530 for both notebook *and* community inference) and the remaining booked hours burned at 4 W |
| 13 | 2026-10-28 | [The 2×2 is complete: `--chunks 2` understates damage on both quants](diary/2026-10-28-the-2x2-is-complete-and-chunks-2-always-understates.md) | Q4_K at 24 chunks: **7.6957 → 9.4738 = +23.11%** (+0.208 nats, 13.9σ), not +17.44%. Both quants read low at `--chunks 2` (understating by 1.26× and 1.33×), so entry 10's "the harness inflated it" was true only for the file it tested. Quant-vs-quant ratio stays 1.13–1.19×: quant level barely matters |
| 14 | 2026-10-28 | [The Spark can hold 132 GiB. The loader dies at 93.](diary/2026-10-28-the-box-holds-132gb-the-loader-dies-at-93.md) | capacity ruled out by measurement: 132 GiB of touched pages holds, 118 GiB synthetic alloc sequence holds, host uptime 18 d. Blames the 20 GB PLE concat. **Hypothesis falsified by entry 15.** |
| 15 | 2026-10-28 | [Six kills: the file-backed table was not the fix](diary/2026-10-28-six-kills-and-the-table-was-not-the-problem.md) | file-backed 20 GB table loads correctly and still OOM-kills at 73%; peak pinned at ~93 GB across bf16/int8/any table strategy. **45B bf16 does not fit** — CUDA claims 130.7 GB of the same pool. Next step is llama.cpp CUDA + the GGUF, not more loader surgery |
| 16 | 2026-10-28 | [CUDA on the Spark reproduces the grid to 0.06%, at 37×](diary/2026-10-28-cuda-on-the-spark-reproduces-the-grid.md) | upstream llama.cpp **already supports `qwen4exp`** (no fork needed); GGUF on GB10 with table on CPU = **4.76 s/pass vs ~175 on CPU**. Same flags, different silicon: **+19.42%** vs **+19.41%**, both z=11.8 — the headline delta is a property of the model, not the harness. Absolutes still differ **1.37%** (CUDA reads low), so ship deltas across machines, never absolutes |
| 17 | 2026-10-28 | [Zeroing half the engram table costs nothing measurable](diary/2026-10-28-zeroing-half-the-engram-table-costs-nothing.md) | five-point capacity sweep at `--chunks 24`, CPU: keeping **50% of the 4.29 GB `per_layer_token_embd` reads 8.4769 ±0.08861 vs 8.5104 ±0.08909** baseline = **−0.39%, z = −0.27**. Nothing anywhere down to 2.15 GB, with a mild monotone trend toward *less* table that is not defensible as a gain. Zeroed rows keep addressing intact, so this bounds row capacity, **not** what a genuinely smaller (collision-heavy) table would score |
| 18 | 2026-10-28 | [The table is live, the idle was me: 54% of accelerator time was wall clock](diary/2026-10-28-the-table-is-live-the-idle-was-me.md) | liveness check: zeroing **100%** of the table gives **PPL 43,348 vs 14.83**, so it is fully read and entry 17's flatness means "half-redundant", not "dead weight". Idle audit: a B60 grid cell is **574 s = load 26 s (4.5%) + compute 547 s**; GGUF mutation is **3.8 s/cell (0.7%)**; yet across four runs utilisation was **46%** with a single **9.3 h** gap — the waste is turnaround, not engineering. Also llama.cpp defaulted to **8 threads on 20**: `-t 20` is **31% faster**. B60 replaces the Spark for PPL: **6.75× CPU, matching to 0.006%** |
| 19 | 2026-10-28 | [Rare tokens are where the engram table earns its keep](diary/2026-10-28-rare-tokens-are-where-the-engram-table-earns-its-keep.md) | entry 17's null was **the corpus, not the table**: same 2.15 GB deleted, **−0.36% on prose vs +5.48% on a rare-token corpus** (z = +3.43) — a 15× magnitude difference and a sign flip, so the table's rows carry real information exactly where the model card advertises it. Also: the harness is **bit-deterministic** (two intact and two 50% runs identical to 4 dp), so a rerun can never arbitrate a null — only a new corpus can; the curve is flat at 75% kept then jumps, so the threshold is in (75%, 50%] and unlocated. **And the provenance correction:** our measurement artefact matches *neither* hub revision (pre-heal `a679542` has the same byte count, different sha), so entries 5–18 measured a local requantisation — entry 16's "same size, so it came from the hub" was not a valid inference |
| 20 | 2026-10-29 | [The knee is at 56.25%, and the shipped file costs 5.24× what entry 19 measured](diary/2026-10-29-the-knee-is-at-5625-and-the-shipped-file-costs-5x-more.md) | run on the **hub file `9f034b43`**, not our local requant: intact **10.2570 ±0.10294**, flat at 62.5% (−0.46%) and 56.25% (−0.44%) kept, then **13.2038 = +28.73%, z = +11.86** at 50%. So entry 19's +5.48% **understates the cost 5.24×** and the knee is in (56.25%, 50%], one step above where 19 left it. The `frac^8` knee model is falsified at all three points (−0.46/−0.44 vs +0.39/+1.35 predicted, then 7.2× too low). **Per-chunk vector: the damage is two chunks of 24** — chunks 1–7 read +0.99%, chunks 8–24 +42.3%, with chunk 8 at **5.94×** and chunk 18 at **4.49×** and 19 of 24 chunks ≤1.23×. Occupancy counted on both files and **ruled out as the mechanism**: every half of every head matches to 0.01 pp, and in both files the tail is as dense as the head (head 3 19.7%, head 4 99.3% — a 5× spread across heads) |
| 21 | 2026-10-29 | [SSM and gate protection at q8_0 buys +0.25%, inside noise](diary/2026-10-29-ssm-protection-buys-nothing-measurable.md) | protecting `ssm_out`/`attn_gate` at q8_0 and `ssm_alpha`/`ssm_beta` at f16 costs **+245.4 MiB (+0.95%)** and moves PPL **7.6821 ±0.07722 → 7.7016 ±0.07745 = +0.254%, z = +0.18** — no effect detected, in the wrong direction. The control is **byte-identical to its source** (`f483b9dd`), so there is no requant confound. The lever that matters is still the experts: entry 13 measured **+23.11%** from ablating them at the same flags |
| 22 | 2026-10-30 | [RAM, not the GPU, sets throughput on this file](diary/2026-10-30-ram-not-the-gpu-sets-throughput.md) | five-host ladder on Clover-35B-A3B Q4_K_M at `--n-cpu-moe 99`: 4080 **384.91** > 3060/31 GB **318.17** > T1000 **51.94** pp2048, and one 3060/15 GB at **28.52**. The 3060 pair is the same laptop (same CPU, GPU, driver, binary) differing only in RAM: **195.2 vs 22.5 t/s** pp256 = **8.7×**, because 31 GB caches the whole 21.42 GiB file (**0 MiB read from disk**) and 15 GB holds **63%** and re-reads **1.08–1.24× the model per 2048-token prefill** at **0.30 GiB/s** (7.4× slower than the 1.9 GB/s sequential rate). Causality by eviction: holding 14 GiB of touched anonymous memory on the *fast* box drops it to **20.21 t/s**. **The dose–response curve is not reproducible** — two reps at the same hold differ up to **3.8×**, a log-linear fit gives R²=0.61, so the cliff is located only to (63%, 91%] resident and the fit is not published |

| 23 | 2026-10-30 | [First matched head-to-head: Whittle gives up 32 HumanEval items to buy 36 cloze items](diary/2026-10-30-first-head-to-head-whittle-loses-reasoning-wins-recall.md) | the comparison every note until now only inferred: **474 paired items, one box, one build, one sampler**, Whittle Q3_K_M vs Qwen3.8-27B Q4_K_M. Whittle wins long-tail cloze **191/200 vs 155/200** (+36, McNemar **p=1.7e-08**) and loses GSM-50 **37/50 vs 46/50** (+9 to Qwen, p=0.004), MATH-60 **52/60 vs 60/60** (+8, p=0.008) and HumanEval **123/164 vs 155/164** (+32, p=1.9e-07). Also measures the quant penalty the card's numbers do not: same thinking-off greedy protocol as the published **133/164**, Q3_K_M gets **123/164** — **−6.1 points**. Harness bug found and fixed: the first HumanEval pass ran thinking ON at a 1024 cap, truncating **41%** of Whittle's completions (26% of Qwen's) and reading 51.8%. Throughput on the same box: **33.1 vs 10.1 t/s** tg64, 4-slot aggregate **17.9** |

| 24 | 2026-10-31 | [A UD-style recipe is dominated: 1.12% fewer bytes, 1.71% worse](diary/2026-10-31-ud-recipe-is-dominated-and-the-imatrix-is-a-zero.md) | 19 hand-written `--tensor-type` overrides (engram table → Q3_K, expert ladder split up/gate/down as Q3_K/Q4_K/Q6_K, q8_0 always-on set) produce a file **290.5 MiB (−1.124%) smaller** and **1.708% worse** — control **7.6821 ±0.07722 reproduces entry 21's control to every printed digit**, prototype **7.8133 ±0.07890**. Against the ladder's own rate (+8.7% PPL per −21% bytes, entry 5) the saving should have cost ~+0.47%, so the recipe charges **3.7×** the going rate. Damage is **uniform: 24 of 24 chunks worse**, mean +1.54%, sd 0.27% (t = 14.4 on paired per-chunk nats) — the opposite shape from entry 20's two-chunk cliff, which is a broad requantisation tax, not one gutted component. The transferred `imat_qwen36` imatrix changes **+0.003%, z = 0.00** |

| 25 | 2026-10-31 | [The 20 GB build beats the shipped file by 6.49%](diary/2026-10-31-the-20gb-build-beats-the-shipped-file-by-6.5.md) | the recipe the deployment constraint actually asks for: pin the engram table at **Q6_K (7,818.60 MiB, +45.83%, charged to RAM)** and cut the body to **17,445.09 MiB (−14.78%)**. Result **PPL 7.9581 ±0.08082 = +3.593% vs control 7.6821** (z = +2.47, but t = 22.3 on 24 paired chunks, **24/24 worse**), which is **−6.490% against the shipped Q3_K_M (8.5104 ±0.08909, z = −4.59) for +7.50% body bytes**. The two exchange rates disagree and that is the finding: **3.23× worse** than the ladder charged on *total* file bytes, **2.1× better** charged on *on-card body* bytes. File is **567.4 MiB (−2.196%) smaller** than Q4_K_M and 4.83 GiB *larger* than Q3_K_M. Not decomposed: table promotion and expert demotion moved together. Not loaded on a 20 GB card yet |

| 26 | 2026-10-31 | [Sacrificing the down experts is strictly dominated](diary/2026-10-31-sacrificing-the-down-experts-is-strictly-dominated.md) | my hypothesis that `ffn_down_exps` is the cheaper expert class to compress is dead: putting the cut on down (**Q2_K**) while keeping gate/up at Q4_K gives **PPL 8.3074 ±0.08504** on a body **320.0 MiB *larger*** than entry 25's 17,445.09 MiB — **+4.389%** worse for more bytes, **24/24 chunks**, t = 17.9. Strictly dominated at any price. The clean single-variable cell (v5A vs v5B, down Q2_K vs Q3_K, everything else byte-identical in recipe) prices that step at **+5.824% for −1,040 MiB body = 1.053 %PPL per %body-byte, 2.0× the quant ladder's 0.520**. By-product is the best-quality build in the sequence: **`w45_v5B` — gate/up Q4_K, down Q3_K, body 18,805.09 MiB — reads 7.8502 ±0.07929 = +2.188% vs control (z = +1.52) and −7.758% vs the shipped Q3_K_M (z = −5.54)** for **+15.95% body bytes against Q3_K_M's body**; it is 3.07% *larger* than Q4_K_M in total bytes because 2,457.27 MiB of table moved to RAM. Gate/up was never isolated; the table-only cell is still unrun |

## Provenance

Every figure here comes from measurements logged in a private lab notebook (~1,371 lines)
and the JSON artefacts of the runs themselves. Nothing is estimated from a model card, and
nothing is extrapolated without saying so. Where a number in an early entry was later shown
to be invalid, the entry says which one and why rather than being quietly edited — the error
rate of my own measurement pipeline is part of the record.

**Read the `--chunks` count AND the `ngl` before comparing any two perplexity numbers in this
repo.** Two effects, both measured on the same file:

- `--chunks 2` vs `24` moves the baseline by +5.30% (Q3_K_M) and +3.52% (Q4_K) — **both read
  low** — and an ablated model by more than that, so the shipped harness *understates* an
  ablation effect by 1.26× on Q3_K_M and 1.33× on Q4_K. Entry 10 claimed the opposite
  direction from a single file; entry 13 is the 2×2 that corrects it.
- `ngl=20` (GPU) vs `ngl=0` (CPU) moves the Q4_K baseline **6.6%** — larger than the ablation
  effect being measured. Entry 10 compared a GPU number to a CPU number and drew a 4.9×
  conclusion from it; entry 11 is the correction.

**And for any *throughput* number, read the host's RAM before the GPU.** Every bench above ran
with the routed experts in system RAM (`--n-cpu-moe`), which makes residency of the `.gguf` in
page cache a first-class variable and not a detail: entry 22 measures **8.7×** on pp256 between
two identical laptops that differ only in 31 vs 15 GB of RAM. The tell is in the log's own error
bars — a cell whose ± is 50% or more is not slow, it is swapping. `fincore --bytes` the model
before quoting any t/s.

Every table states its flags; a ratio is only as matched as its worst-matched factor.

Hardware referenced throughout:

| box | CPU | GPU | role |
|---|---|---|---|
| laptop (Zephyrus) | Ryzen 9 7945HX, 32T | RTX 4080 Laptop 12 GB, PCIe 4.0 **x8** | 45B Q4 at 27.4 t/s, evaluation |
| remediate | i7-12700T, 20T | T1000 8 GB | 35B at 7.2 t/s, Docker grading |
| localhost (50.77) | i9-12900H, 20T | RTX 3060 Laptop 6 GB | Clover bench: 31 GB RAM caches the whole GGUF |
| DESKTOP-JKJQB5R (50.119) | i9-12900H, 20T | RTX 3060 Laptop 6 GB | same laptop as 50.77 but **15 GB RAM** — read the RAM before any number from this box |
| remediate-2 | i7-12700T, 20T | T1000 8 GB | CPU-only perplexity and NIAH, 20 threads |
| garlic-clove | — | **Intel Arc Pro B60** (BMG G21, 21414 MiB) | Vulkan inference, 10.3× the CPU rate |
| Colab G4 | — | 96 GB, 102 GB available | bf16 routing profile, 3.14 CU |
| DGX Spark | 20 cores, GB10 | **121 GB unified** (118 available) | bf16 baseline + expert heal; 12 h/week, nightly window, `/workspace` persists |

## Status

Notes, not a release. The Q3_K_M artefact discussed in entries 5–19 is 21,310,671,776 bytes,
sha256 `3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824`, reproducible from
the published Q4_K_M in 3.4 minutes with `llama-quantize --allow-requantize`.

**It is a local requantisation, not the hub's Q3_K_M file.** Entry 19 checked: the hub's
pre-heal revision `a679542` has the same 21,310,671,776 bytes but sha `d86cec54…43cc`, and
HEAD (after the 11:02 UTC 2026-10-28 heal commit) is 21,310,673,216 bytes, sha `9f034b43…b731`.
So every absolute perplexity in entries 5–19 describes a file that nobody downloads, and the
internal deltas — which compare that file against a mutated copy of itself — are unaffected.
Deployment claims still need re-measuring against the shipped file.

**Entry 20 is the first entry measured on the shipped file** (`9f034b43…b731`) rather than on
the local requant, and it is a warning about how far that gap reaches: the same 2.15 GB table
deletion costs **+5.48%** on our file and **+28.73%** on the hub's, a factor of 5.24, with the
intact baselines only 2.59% apart. Byte count and architecture do not bound a behavioural
difference. Treat every remaining figure here as a property of `3544d100…824` until it is
re-run, and note that entry 20's per-chunk result suggests the *shape* of a damage curve
matters as much as its mean.
