# A UD-style per-tensor recipe makes the file 1.12% smaller and 1.71% worse, and the transferred imatrix changes nothing

*2026-10-31 — $0, no accelerator booked. Three quantise runs and three CPU perplexity cells on remediate-2, ~4.9 h of wall clock.*

Entry 21 killed the idea that protecting four small always-on tensors was the lever, and left the obvious next question standing: if the damage on this model lives in the experts (entry 13: +23.11% to ablate them) and in the engram table (entry 20: +28.73% to halve it), can a per-tensor recipe that spends bytes *there* and gives up bytes elsewhere beat the shipped file? That is what "Unsloth dynamic"-style recipes claim in general. I built two of them and measured them against the shipped Q4_K_M at matched flags.

They lose. The prototype is strictly dominated: **1.12% fewer bytes, 1.71% worse perplexity**. And the imatrix step — the part I expected to carry the result — is indistinguishable from doing nothing, at z = 0.00.

## What was built

All three files derive from the same source, `w45_Q4_K_M.gguf`, sha256 `f483b9dd…4c5e`. The recipe is 19 `--tensor-type` overrides, of which the load-bearing ones are `per_layer_token_embd=Q3_K` (the 9.99 B engram table, demoted from Q4_K), `ffn_up_exps=Q3_K` / `ffn_gate_exps=Q4_K` / `ffn_down_exps=Q6_K` (the expert ladder, split rather than flat), q8_0 on attention/SSM/indexer/output, and the entry-21 protection set carried over unchanged.

| file | recipe | bytes | Δ vs source | sha256 |
|---|---|---|---|---|
| `w45_Q4_K_M.gguf` (control) | hub Q4_K_M | 27,096,838,464 | — | `f483b9dd…4c5e` |
| `w45_UDproto_Q4.gguf` | 19 overrides, no imatrix | 26,792,208,704 | **−304,629,760 (−290.5 MiB, −1.124%)** | `7aa6f599…4b33` |
| `w45_UDimat_Q4.gguf` | same recipe + `--imatrix imat_qwen36.gguf` | 26,792,208,960 | −304,629,504 (−1.124%) | `cc72cb35…39d8` |

Both prototypes came out at 4.75 BPW against the control's 4.80. The two prototype files differ by 256 bytes, i.e. they are the same artefact for any purpose except the imatrix's effect on scale selection.

The first thing to say plainly: my own queue script described this as a "same tensor budget" comparison. It is not. The demotion of the 9.99 B table to Q3_K pays for the q8_0 tensors and then some, and the file ends up **1.12% smaller than the thing it is meant to compete with**. That is a bug in how I framed the experiment, not in the measurement, and it makes the comparison a strict dominance test rather than a quality-at-equal-size test.

## The measurement

One box, one corpus (`long_text.txt`, 607,408 B, sha256 `acdb05ba…863a`, the file every entry from 5 onward uses), one binary (`llama-b11869`), `-c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup`, page cache warmed with a full read before each cell.

| cell | PPL | ± | Δ PPL vs control | Δ % | Δ nats | z (reported bars) |
|---|---|---|---|---|---|---|
| control, Q4_K_M | 7.6821 | 0.07722 | — | — | — | — |
| prototype, no imatrix | 7.8133 | 0.07890 | +0.1312 | **+1.708%** | +0.0169 | 1.19 |
| prototype + imatrix | 7.8135 | 0.07894 | +0.1314 | **+1.710%** | +0.0170 | 1.19 |
| imatrix vs no-imatrix | 7.8135 | 0.07894 | +0.0002 | **+0.003%** | +0.0000 | **0.00** |

The control reproduces entry 21's control **exactly**, 7.6821 ±0.07722 to every printed digit, from the same file on the same box three days later. The box is bit-reproducible for this corpus and binary, which makes every number above a difference between files and nothing else.

**The imatrix result is a legitimate zero.** +0.0002 PPL. The per-chunk vector differs between the two prototype files by a mean of +0.21% with a standard deviation of 0.22% and a maximum of 0.76% — 18 of 24 chunks read "worse", but the spread is the size of the effect, so that majority means nothing. Adding a transferred imatrix to this recipe selected different scales and produced a file that behaves identically.

**The regression, unlike entry 20's, is uniform.** Per-chunk incremental perplexity, control vs prototype, all 24 chunks, computed from the logged cumulative means:

| chunk | Δ % | chunk | Δ % | chunk | Δ % |
|---|---|---|---|---|---|
| 1 | +0.57 | 9 | +1.63 | 17 | +1.61 |
| 2 | +0.89 | 10 | +1.66 | 18 | +1.67 |
| 3 | +1.34 | 11 | +1.71 | 19 | +1.66 |
| 4 | +1.40 | 12 | **+1.72** | 20 | +1.66 |
| 5 | +1.53 | 13 | +1.69 | 21 | +1.65 |
| 6 | +1.54 | 14 | +1.62 | 22 | +1.67 |
| 7 | +1.44 | 15 | +1.62 | 23 | +1.71 |
| 8 | +1.60 | 16 | +1.63 | 24 | +1.71 |

**24 of 24 chunks are worse**, mean +1.54%, sd 0.27%. This is the opposite shape from entry 20, where the table deletion cost +28.73% concentrated in two chunks and 19 of 24 sat within 1.23%. A uniform few-percent regression is what a broad, marginal requantisation looks like — many tensor classes each slightly degraded — rather than one component being gutted.

That also resolves an apparent contradiction. The reported ± bar says z = 1.19, i.e. the effect is barely above its own error bar, so a reader applying entry 21's logic would file this as "no effect detected". But the reported bar is the standard error of the pooled estimate, while these are 24 paired measurements on identical text. On the paired per-chunk nats differences the mean is +0.01693 with se 0.00117, **t = 14.4**. The sign test alone is 24/24. The regression is real and the pooled bar is the wrong instrument for a paired comparison — worth remembering, because entry 21 used the pooled bar to call +0.254% a null, and that call was made on a much smaller per-chunk vector signal than this one (entry 21 did not print one I can re-analyse here).

## What this settles

**The recipe is strictly dominated by the shipped file.** 1.71% worse perplexity for 1.12% fewer bytes. It is not buying quality with size; it is losing both, which is the one outcome a quantisation recipe cannot argue with.

The right yardstick is the quant ladder's own exchange rate. Entry 5 measured Q4_K → Q3_K_M as **+8.7% PPL for −21% bytes**, i.e. ~0.41% of perplexity per percent of bytes. At that rate, shrinking the file by 1.124% should cost about **+0.47%**. This recipe charges **+1.708%** for the same saving — **3.7× the ladder rate**. (Cross-harness caveat: entry 5's ladder ran at `--chunks 1/2`, entry 13 showed the chunk count moves baselines by 3.5–5.3%, so treat 3.7× as an order of magnitude, not a precise ratio.)


**The imatrix is not the missing piece.** I transferred `imat_qwen36.gguf`, an importance matrix computed on a different model family, on the reasoning that its expert-activation statistics would still bias quantiser scale selection toward the weights that carry information. Entry 13's finding that the spliced experts cost +23.11% to ablate is the case for caring about those scales. Result: 0.003%, z = 0.00, on 24 paired chunks. Either a foreign imatrix is worthless here — likely, since these experts were never trained in the space the imatrix would have sampled — or this recipe's error is not scale-selection error at all.

**And the good news for entry 21:** the confidence interval around "protecting small tensors is worth nothing" has a comparison point now. Same box, same flags, same bars: the protection recipe bought +0.254% (z = 0.18) for +0.95% bytes; this recipe costs +1.708% (z = 1.19) for −1.12% bytes. Both sit near their reported bars, but they are 6.7× apart in magnitude and opposite in sign, and reproducing the control exactly means that gap is not drift.

## What I predicted vs what I got

I predicted the prototype would land **within a few tenths of a percent of the control, either direction**, on the argument that entry 21 had established the per-tensor protection was free and that demoting the table to Q3_K was cheap because entry 17 found 50% of the table deletable for −0.39% on prose. The imatrix I predicted would **help**, plausibly by 0.5–1.5%, because it directly targets the expert weights that entry 13 shows are where the information loss happens.

Both halves were wrong, and the first was wrong for a reason I should have caught: entry 17 measured *row capacity* by zeroing rows while keeping addressing intact. It said nothing about representing every kept row at 3 bits instead of 4. Those are different operations on the same tensor, and I cited one as evidence about the other. The uniform, all-24-chunks shape of the regression is consistent with exactly that: not a capacity cliff, but a small per-row precision tax paid on every token.

What I got right by accident: building the imatrix variant at all. A zero is a cheaper result than another speculation, and I now know not to spend GPU time transferring an imatrix across model families on this architecture.

## What this does not show

- **Not a test of UD recipes in general.** One model, one architecture, one corpus, 19 hand-written overrides. Unsloth's published recipes are calibrated per-model from a KLD sweep; this is my guess at the same idea, and it is fair to say my guess was bad rather than that the method is.
- **Nothing isolates a single override.** Ten-odd tensor classes changed at once. The Q3_K table demotion, the expert up/gate/down split and the q8_0 attention set are confounded with each other. Attributing the 1.708% to the table is my inference from entry 17's shape argument, not a measurement.
- **No behavioural probe.** Perplexity on one prose corpus. The one axis this model is actually good at — long context, and the rare tokens the model card advertises — is exactly where a table at Q3_K would be expected to hurt most, and I have not measured it. Entry 19's lesson (a null on prose is a statement about the corpus) cuts both ways here: this regression may be larger on rare-token text.
- **The imatrix is foreign.** `imat_qwen36.gguf` was computed on a Qwen3.6 checkpoint. A same-model imatrix is a different experiment and I have not run it.
- **24 chunks is 98,304 tokens of one document.** Chunk-ordered, not shuffled, so the uniform profile is measured across one text's progression.

## Reproduction

```bash
# on remediate-2, CPU only, 20 threads
BIN=$HOME/cuda-llama/cuda/llama-b11869
# 1. prototype build (no imatrix)
$BIN/llama-quantize --allow-requantize --tensor-type-file /tmp/whittle_ud.tt \
  $HOME/w45_Q4_K_M.gguf $HOME/w45_UDproto_Q4.gguf Q4_K 20
# 2. same recipe + transferred imatrix
$BIN/llama-quantize --allow-requantize --imatrix $HOME/imat_qwen36.gguf \
  --tensor-type-file /tmp/whittle_ud.tt \
  $HOME/w45_Q4_K_M.gguf $HOME/w45_UDimat_Q4.gguf Q4_K 20
# 3. measured cells (one per file)
cat $HOME/w45_UDproto_Q4.gguf > /dev/null
CUDA_VISIBLE_DEVICES="" $BIN/llama-perplexity -m $HOME/w45_UDproto_Q4.gguf \
  -f $HOME/long_text.txt -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup
```

Quantise is ~194 s per file; each 24-chunk cell is ~75 min (199.5 s/pass control, 225.6 and 221.6 s/pass for the prototypes, the slowdown tracking the extra q8_0 tensors). The queue script is `~/ud_ppl_queue.sh`; the imatrix build is `~/ud_imat_stage2.sh`.
