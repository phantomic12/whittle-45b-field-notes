# Day 1 — Anatomy: a 35B wearing 76 borrowed experts

*2026-10-24. First commit to the heal kit at 06:20, placement maths at 06:32.*

## Why I started here

The model card calls this a 45B and sells "restored experts". Before writing a single
training script I wanted to know three things that the card does not state plainly:

1. What is actually 45B in here, and how much of it runs per token?
2. Where does the extra 6 GB over its parent sit?
3. If I attach an adapter, does anything receive a gradient?

Question 3 turned out to be the important one, and the answer was no — four different ways.

## The parameter split, from headers

I read tensor shapes out of the GGUF header rather than downloading 90 GB. Range requests
against the shard index give exact shapes; the arithmetic is then not an estimate.

| component | parameters | share |
|---|---|---|
| routed experts (256/layer × 40) | 32.2123 B | 71.4% |
| n-gram table (`per_layer_token_embd`) | 9.9942 B | 22.2% |
| always-on (attention, SSM, norms, necks) | 2.910 B | 6.5% |
| **total** | **45.1168 B** | |

45.1168 B is a suspiciously exact match for the name. That is the first useful signal: the
name is honest about the total. It is not honest about what *runs*.

At k=8 of 256 experts the active parameter count is **3.917 B — 14.1% of a dense 27B**.
At k=16 it is 4.924 B, at k=24 it is 5.930 B. So the model is a 45B object that computes
like a 4B one, which is the entire pitch of MoE and also the reason the 76 extra experts
are interesting: at k=8 of 256, an individual expert is drawn roughly 3% of the time.

The n-gram table deserves its own line. It is 9.9942 B across 5 shards,
39,040,000 rows = 8 heads × 4,880,000 × 256 dims. That is 22% of the file spent on a hash
lookup, and no dense model has an equivalent line item. Any claim about this model's
"efficiency" has to survive that.

### The splice, stated precisely

Experts at indices **180–255** (76 of them, per layer, 40 layers) were copied from
Qwen3.6-35B-A3B at original weights, scaled by a **median norm ratio of 1.009**, and never
trained in this model. Their SwiGLU content is in the wrong hidden space. Scope:
76/256 × 40 layers = **9.56 B base parameters, 21% of the model**.

That is the whole modification. It is a splice, not a distillation, not a continued
pretrain. Which means the model's quality is a question about *interference* — what
happens when you add 76 experts' worth of foreign projections to a router that has to
learn to use them — and not a question about training data.

### Layer composition, cross-checked twice

30 linear/SSM layers (`ssm_out`, `attn_qkv`) carry constant state; 10 full-attention layers
(`attn_q`, `indexer`) carry a growing cache. They interleave every 4th layer: attention at
3, 7, 11, … 39.

I derived that from the GGUF block tensors first. Later, `config.json` confirmed it
independently — `full_attention_interval: 4` and an explicit `layer_types` list of 30
`linear_attention` + 10 `full_attention`. Two sources agreeing matters here because the
KV-cache claim depends entirely on it:

```
2 KV heads × 256 head dim × 10 layers × 2 bytes (f16) = 20.0 KB/token
```

20.0 KB/token, or 10.0 KB/token with a q8 cache. At 32K that is 0.67 GB; at 131K, 2.68 GB;
at 262K, 5.37 GB. A dense 27B with conventional attention (48 layers, 8 KV heads, 128 head
dim, f16) costs **98 KB/token**, i.e. 0.10 GB per 1K tokens, 3.2 GB at 32K before a single
weight is loaded.

The 45B's cache is ~5× smaller than a dense 27B's. This is architectural and it is the one
thing about the model that is unambiguously good. I did not know yet whether it *mattered*
— that is entry 3.

## Four loader failures, all of which look like data problems

This is the part of the project I would hand to anyone else first. Each of these fails
silently or misleadingly, and each one will convince you that your dataset is broken.

### 1. PEFT attaches to nothing

The experts are fused 3-D `nn.Parameter`s — `gate_up_proj [256, 1024, 2048]`,
`down_proj [256, 2048, 512]` — not `nn.Linear`. So:

```python
target_modules=["gate_proj"]   # matches 0 modules
```

PEFT accepts it, prints a trainable-parameter count, and **the loss still falls** (because
the head or a norm is still training). Nothing raises. You can burn a whole session
fine-tuning a model where the intended 21% of parameters received no gradient at all.

### 2. Adapters that exist but are never called

The stock forward reads the raw parameter, not the module. So a zero-init side adapter
contributes nothing to the autograd graph, and `backward()` raises:

```
element 0 of tensors does not require grad
```

This one at least fails loudly — but only after you have built the adapter, merged
nothing, and started to suspect your loss scaling.

### 3. A wrong `key_mapping` does not throw

It silently produces a **random table**. No shape error, no warning, no missing-key
report. The model loads, generates fluent text, and is simply not the model you asked
for. This is the worst of the four because the output looks plausible.

### 4. `model_type` mismatch

The checkpoint says `qwen4_exp_text`. Using `qwen4_exp` routes through the multimodal
wrapper and **every sub-config attribute comes back `None`** — `hidden_size is None`,
`num_hidden_layers is None`. The traceback points at whatever you touched first, not at
the model type.

The kit's answer to all four: never `from_pretrained`. Use the author's `whittle_load.py`
unmodified, and gate everything behind a round-trip test that asserts the loaded table is
**bit-identical** to the checkpoint.

```
$ ./run_phase0.sh
... 12 PASS lines, exit 0
```

The gate is deliberately boring: build a tiny Whittle-format checkpoint, load it, compare
bytes; attach → backward → merge → forward; install router hooks and assert the counts are
exact. Non-zero exit means do not spend GPU time. That is the only reason I trusted any
later ablation number — an ablation is only as good as the loader underneath it.

## What the box can and cannot do

The Zephyrus is an RTX 4080 Laptop 12 GB, Ryzen 9 7945HX, 30 GB DDR5, and — relevant
later — **PCIe 4.0 x8**, i.e. half the lanes, ~13 GB/s real.

**Inference fits, training does not.** Concretely: GPU holds the always-on weights plus
embed/head (~1.6 GB) plus ~9 GB of hot experts; RAM holds the expert banks (18.0 GB at Q4)
plus the n-gram table (5.6 GB) = 23.6 GB of the ~28 usable. Decode at k=8 moves
~564 MB of expert traffic per token → ~106 t/s if the banks stay RAM-resident, and only
~23 t/s if they have to cross PCIe. So: keep the banks in RAM, do not stream them.

Training is short by 48 GB: 90.2 GB bf16 against 42 GB total. LoRA makes the *adapter*
1.2 GB but the base body is still resident for backward. Layer streaming means each layer
is visited 3× → ~90 s/step of I/O, 500 steps ≈ **13 h**, against 4–6 h on a box with
everything resident.

**bitsandbytes QLoRA is a trap on this architecture.** `replace_with_bnb_linear` only
converts modules where `type(module) is nn.Linear`. The fused expert bank (32.2 B) and the
n-gram table (10.0 B) are skipped, so bnb quantises **2.91 B of 45.1 B** and leaves 84 GB
in bf16. Every "QLoRA on a 24 GB card" plan dies here and the error message will not tell
you why.

Verdict that day: this box profiles and evaluates; a Spark (or a G4) trains. LoRA r=8 with
the base frozen would train 75.5 M parameters, which is affordable — the *base residency*
is what is not.

## What I had wrong on day 1

I assumed "restored experts" meant the author had trained something, and that my job would
be to continue that training. The headers say otherwise: the weights are at original
values times a scalar. There is no restoration to continue. That reframes the project from
"finish the author's work" to "measure whether the splice works", which is a much cheaper
and much more honest question — and it is the question the rest of these notes answer.

I also assumed the 45B would be strictly better than its parent, because more experts is
more capacity. That assumption survives until entry 2 and then dies there, twice.
