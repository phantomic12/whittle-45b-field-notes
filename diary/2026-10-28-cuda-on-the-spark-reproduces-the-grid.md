# Entry 16 — CUDA on the Spark reproduces the CPU grid to 0.06%, at 37×

*2026-10-28*

Entries 12–15 spent the Spark session trying to make the bf16 checkpoint load. It does not load:
peak pins at ~93 GB and OOM-kills regardless of loader strategy, while CUDA on the same box
demonstrably claims 130.7 GB of the same 121 GB pool. Conclusion at the time was "next step is
llama.cpp CUDA + the GGUF, not more loader surgery."

That next step is done, and it works. This entry is the first measurement that came out of it.

## The setup that finally worked

Upstream `ggml-org/llama.cpp` at `8ca4060` **already knows `LLM_ARCH_QWEN4EXP`** — no fork
needed, contrary to what earlier notes in this project said (I had been carrying "upstream does
not know qwen4exp, build lendome's fork" since the Arc B60 work). Confirmed by grepping the
checkout: `src/llama-arch.cpp`, `llama-model.cpp`, `llama-hparams.h`, `llama-context.cpp` all
mention it, and `LLM_ARCH_QWEN4EXP` is a first-class enum value.

```
$ ./llama.cpp/build/bin/llama-perplexity -m gguf/Whittle-Qwen-3.8-45B-A3B-Q3_K_M.gguf \
    -f long_text.txt -c 4096 -b 512 -ub 128 --chunks 24 \
    -ngl 99 -ot 'per_layer_token_embd=CPU' -fa on
system_info: CUDA : ARCHS = 1210 | USE_GRAPHS = 1 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | BLACKWELL_NATIVE_FP4 = 1
perplexity: 4.76 seconds per pass
```

- `-ngl 99` puts the transformer on the GB10.
- `-ot 'per_layer_token_embd=CPU'` keeps the ~10 B n-gram table off the GPU — the exact
  deployment shape the whole project is aiming at, and it works unmodified.
- `-fa on` flash attention, 20 threads.
- **4.76 s/pass.** remediate-2's CPU does ~175 s/pass. That is **37×**, and it turns a 70-minute
  grid cell into a 2-minute one.

The GGUF came from the public repo `logic65/Whittle-Qwen-3.8-45B-A3B-GGUF` (Q3_K_M, 21,310,671,776
bytes — same size as our artefact), so no private relay was needed.

## The cross-check

Same file, same corpus, same flags, same `--chunks 24`, only the compute device differs:

| device | baseline | ablated (experts 180–255, all 40 layers) | Δ PPL | Δ nats | z |
|---|---|---|---|---|---|
| remediate-2 CPU, `ngl=0` | 8.5104 ±0.08909 | 10.1622 ±0.11046 | **+19.41%** | +0.17739 | 11.8 |
| Spark GB10 CUDA, `ngl=99` + table on CPU | 8.3934 ±0.08749 | 10.0235 ±0.10827 | **+19.42%** | +0.17749 | 11.8 |

**The deltas agree to 0.01 PPL points — 0.06% relative on nats — and both land at z = 11.8.**

That is the result I wanted and did not quite expect. It means the +19.41% figure from entry 13
is a property of the *model and the ablation*, not of my CPU harness, my thread count, or that
particular box. The single most load-bearing number in this project now has an independent
confirmation on different silicon, a different backend, and a different OS build.

## The 1.37% offset is real and does not cancel

The absolute perplexities differ: CUDA reads **1.37% lower** on the baseline and **1.36%** lower
on the ablated model. Almost exactly the same offset in both conditions, which is why the ratio
survives untouched.

This is worth recording because it is a *fourth* harness axis with a measurable price, and it is
the one most likely to be silently abused:

- `--chunks 2` vs `24`: +3.5–5.3% on the baseline (entries 10, 13)
- `ngl=20` GPU vs `ngl=0` CPU, same box: 6.6% (entry 11)
- **CUDA vs CPU, same flags, different machine: 1.37%**

CPU and CUDA do not produce bit-identical logits. Different reduction orders in the matmuls,
different flash-attention kernels, different `ub` behaviour. 1.37% is small enough to look like
noise and large enough to flip a comparison between two close quants — which is precisely the
kind of error entries 10 and 11 were made of. **The rule stands and gets a third clause: a delta
survives a change of hardware; an absolute perplexity does not.** Report deltas across machines,
never absolutes.

## What this changes for the plan

The Spark is now the measurement box for this project, not a training box. That is a demotion
for the heal and a promotion for everything else:

- **The full grid is now ~10 minutes of work**, not a night. Any ablation hypothesis can be
  tested in the time it takes to make coffee. The "prefer `--chunks 24`" rule from entry 13 was
  written as a wall-clock tradeoff; on this box there is no tradeoff to make.
- **Training is still off the table here.** 45B bf16 does not fit (entries 14, 15), and the
  quantised GGUF is not trainable in llama.cpp without a LoRA path for `qwen4exp` that does not
  exist. The heal needs real VRAM or a different model size — Colab's 102 GB RTX PRO 6000
  already loaded the 90.3 GB bf16 checkpoint, and that is where it has to happen.
- **Cheap experiments are now unblocked.** The n-gram table sizing question (`per_layer_token_embd`
  is 9.99 B of 45 B and has never been tested at the sizes that matter) becomes a 10-minute
  ablation per k value on this box instead of a night each.

## Reproduction

```bash
# on the Spark, from /workspace
cmake --build llama.cpp/build --target llama-perplexity llama-quantize -j 12
python3 -c "from huggingface_hub import hf_hub_download as d; \
  d('logic65/Whittle-Qwen-3.8-45B-A3B-GGUF','Whittle-Qwen-3.8-45B-A3B-Q3_K_M.gguf', \
    local_dir='/workspace/gguf')"
# long_text.txt must be byte-identical to the CPU grid: 607408 B, sha256 acdb05ba…863a
./llama.cpp/build/bin/llama-perplexity -m gguf/*Q3_K_M.gguf -f long_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -ot 'per_layer_token_embd=CPU' -fa on
# 8.3934 +/- 0.08749, then ablate experts:180-256 and re-run for 10.0235 +/- 0.10827
```

Note on the corpus: my first attempt used a different Gutenberg edition (772,386 B) and gave
8.4184 at `--chunks 2`. Numbers are only comparable within one corpus file. The 607,408 B file
is the one every entry from 5 onward used, and its sha256 is the thing to match.
