# Pinning the engram table at Q6_K buys a 17.05 GiB body that beats the shipped Q3_K_M by 6.49% on the same harness

*2026-10-31 — $0, no accelerator booked. One quantise run and one 24-chunk CPU perplexity cell on remediate-2, ~1.6 h of wall clock.*

Entries 21 and 24 both answered "can a smarter per-tensor recipe beat the shipped file" the same way, no. But neither of them tested the recipe the deployment constraint actually asks for. The constraint is a **20 GB card**: the file cannot live in VRAM, so the 7.64 GiB engram table can be pinned high and charged to RAM/SSD, and only the body has to fit. That is a different optimisation from shrinking the file, and I finally ran it.

**It works, measured against the only file that matters for this target.** A 17.05 GiB body at +3.59% over Q4_K_M beats the shipped Q3_K_M by **6.49%** (z = −4.59) while sitting **1.20 GiB larger on-card** than Q3_K_M's body. On the body bytes a 20 GB card actually pays for, this build is **1.66× more efficient than the quant ladder**. On total file bytes it is worse. Which of those two numbers you care about depends entirely on where you put the table.

## What was built

One source file, `w45_Q4_K_M.gguf`, sha256 `f483b9dd…4c5e` — the same control every entry since 21 has used. One `llama-quantize` invocation with `--allow-requantize`, base `Q4_K`, and a 15-rule `--tensor-type-file`:

- `^per_layer_token_embd\.weight$` → **Q6_K**, i.e. the 9.99 B table is *promoted* from Q4_K, not demoted as in entry 24
- `ffn_(gate|up)_exps` → Q3_K, `ffn_down_exps` → Q4_K: the expert ladder split so the body shrinks
- q8_0 on `attn_qkv`, `attn_{q,k,v,output}`, `attn_gate`, `ssm_out`, `token_embd`, `output`, the shared experts and the indexer projections; f16 on `ssm_{alpha,beta}`

637 individual tensors took a manual override; 1 of 1022 fell back to the base type. Quantise took 234.98 s.

| file | table | body | total | BPW | sha256 |
|---|---|---|---|---|---|
| `w45_Q4_K_M.gguf` (control) | 5,361.33 MiB | 20,469.73 MiB | 25,831.06 MiB | 4.80 | `f483b9dd…4c5e` |
| `w45_VRAM20.gguf` | **7,818.60 MiB (+2,457.3, +45.83%)** | **17,445.09 MiB (−3,024.6, −14.78%)** | 25,263.69 MiB | 4.70 | `c8d13639…c60e` |

All figures are tensor sizes printed by `llama-quantize`; the file on disk is 26,501,904,704 B (24.68 GiB, 26.50 GB), **567.4 MiB / 2.196% smaller** than the 27,096,838,464 B control. The table goes from 20.8% of the file to **30.9%** of it, and the body from 20.0 GiB to **17.04 GiB**.

So this is not a smaller file. It is the same file with 2.4 GB moved from the on-card side of the ledger to the RAM side, plus the 0.57 GB the ladder was going to give back anyway. Whether that is a win is not a quality question, it is a question about your memory hierarchy.

## The measurement

One box, one corpus (`long_text.txt`, 607,408 B, sha256 `acdb05ba…863a`), one binary (`llama-b11869`), `-c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup`, page cache warmed with a full read first. 237.52 s/pass, 90 minutes for the cell.

| cell | PPL | ± | vs control | Δ % | Δ nats | z (pooled bars) |
|---|---|---|---|---|---|---|
| Q4_K_M control | 7.6821 | 0.07722 | — | — | — | — |
| **`w45_VRAM20`** | **7.9581** | 0.08082 | **+0.2760** | **+3.593%** | +0.0353 | **+2.47** |
| Q3_K_M, shipped-size (entry 17 baseline) | 8.5104 | 0.08909 | +0.8283 | +10.782% | +0.1022 | +7.03 |
| **`w45_VRAM20` vs Q3_K_M** | | | **−0.5523** | **−6.490%** | −0.0671 | **−4.59** |

The control is 7.6821 to every printed digit, the third independent time this box has read it (entries 21, 24, now). The Q3_K_M baseline is the local artefact `3544d100…a824` that entries 5–19 measured, at the same `--chunks 24` on the same box; its `ple_sweep.log` invocation did not pass `-t`, so it ran on the default 8 threads rather than 20. That is the only flag difference in the table, and I have priced it: the same Q4_K_M file read 7.6957 in `q4k24.log` and 7.6821 here, **0.18% apart**, an order of magnitude below the 6.49% being claimed.

**The regression against control is uniform again.** All 24 paired chunks, computed from the logged cumulative means:

| chunk | Δ % | chunk | Δ % | chunk | Δ % |
|---|---|---|---|---|---|
| 1 | +2.31 | 9 | +3.72 | 17 | +3.08 |
| 2 | **+5.36** | 10 | +3.75 | 18 | +4.40 |
| 3 | +3.19 | 11 | +2.99 | 19 | +4.03 |
| 4 | +3.80 | 12 | +3.44 | 20 | +2.98 |
| 5 | +4.24 | 13 | +3.55 | 21 | +4.59 |
| 6 | +3.33 | 14 | +2.12 | 22 | **+5.32** |
| 7 | +2.98 | 15 | +3.58 | 23 | +3.85 |
| 8 | +4.67 | 16 | +3.33 | 24 | +3.08 |

**24 of 24 chunks worse**, mean +3.654%, sd 0.817%. On the paired per-chunk nats differences the mean is +0.03586 with se 0.00161, **t = 22.3**, so the effect is real even though the pooled bar only says z = 2.47. The shape is the same broad requantisation tax as entry 24's, at twice the size.

## What this settles: the ladder is not the right answer for a 20 GB card

Two exchange rates, computed from the ladder I already measured:

| denominator | ladder rate (Q4_K_M → Q3_K_M) | this recipe (Q4_K_M → VRAM20) | ratio |
|---|---|---|---|
| total file bytes | 0.506 %PPL per %byte | **1.636** | **3.23× worse** |
| on-card body bytes | 0.520 %PPL per %byte | **0.243** | **0.47× (2.1× better)** |

The ladder itself: +10.782% PPL for −21.32% total bytes, i.e. −20.72% of the body. The recipe: +3.593% PPL for −2.20% total bytes but −14.78% of the body.

That split is the whole finding. Charged on total bytes, this recipe is a bad deal by 3.2×, consistent with entry 24's verdict on a different recipe. Charged on the bytes that a 20 GB card has to hold, it is the **best deal in the project so far** — and the difference between those two statements is exactly the 2.46 GB of table it moved off the card.

Measured directly against the file that a 20 GB card would otherwise run: **−6.490% PPL for +7.50% body bytes** = 0.865 %PPL per %body-byte, **1.66× the ladder's body rate**. If you are going to run this model on a card too small for it, Q3_K_M is the wrong file.

## What I predicted vs what I got

I expected the body cut (`ffn_gate_exps`/`ffn_up_exps` → Q3_K, 14.8% of the body) to dominate, and to cost something close to what the ladder charges for a comparable body reduction — order +5 to +7% — with the Q6_K table promotion acting as a near-free size trade that moved PPL very little, on the strength of entry 17's finding that half the table is deletable on prose for −0.39%.

I got **+3.593%**, about half my estimate. The most likely reason is that entry 17 was the wrong prior for the same reason entry 24 found: entry 17 deleted *rows* while keeping addressing intact, and this changes the *precision* of every row, in the opposite direction from entry 24's Q3_K demotion. Entry 24 charged +1.708% for demoting the table to Q3_K; if that cost is mostly the table, then promoting it to Q6_K should recover a comparable amount, and the numbers are at least consistent with that (1.708 recovered out of my ~7% expectation).

But I want to be clear that this is a **consistency argument, not a decomposition**. Two tensor classes moved in opposite directions in this run and there is no cell that isolates either.

What I got right: building the file for the constraint instead of for the file size. Entries 21 and 24 both optimised the wrong objective and both read as failures.

## What this does not show

- **Nothing is decomposed.** The Q6_K table promotion, the Q3_K gate/up demotion and the q8_0 always-on set all changed at once. The claim "the table promotion is what paid" is inference from entry 24's opposite-signed result, not measurement. The clean cell — Q4_K table with the same reduced body — has not been run.
- **The 17.05 GiB body has not been loaded on a 20 GB card.** I claim a body size, not a successful fit. Entry 7 put a 21.42 GiB file on a 24 GB Arc B60 with ~17.8 GB resident; this body is smaller, but offload behaviour for a file whose table is 7.64 GiB and pinned high is untested. It may page the table differently than the Q3_K_M file does.
- **Prose perplexity is the wrong instrument for the table.** Entries 19 and 20 established that the table's value shows up on rare tokens, not on Gutenberg prose (+28.73% to halve it on the rare corpus, −0.44% on prose). A table at 6 bits is presumably worth *more* there than the prose numbers show, which means **−6.49% is probably a floor**, not an estimate. It is unmeasured.
- **+6.49% is a single corpus, single box, single 98,304-token document**, chunk-ordered rather than shuffled. It is not a Bencheval or any task score.
- **The file is 4.83 GiB bigger than shipped Q3_K_M** (24.68 vs 19.85 GiB). If your storage or page cache is the binding constraint rather than VRAM, this is the wrong build and entry 5's ladder still applies.
- **The rare-token corpus, NIAH and the HumanEval/GSM cells from entry 23 have not been run on this file.** The behavioural claims in the README all still belong to `w45_Q3_K_M.gguf`.

## Reproduction

```bash
# on remediate-2, CPU only, 20 threads
BIN=$HOME/cuda-llama/cuda/llama-b11869
# 1. the build (tensor-type file is ~/ud_stage4.sh's heredoc, 15 rules)
$BIN/llama-quantize --allow-requantize --tensor-type-file /tmp/whittle_vram20.tt \
  $HOME/w45_Q4_K_M.gguf $HOME/w45_VRAM20.gguf Q4_K 20     # 234.98 s
# 2. the cell
cat $HOME/w45_VRAM20.gguf > /dev/null
CUDA_VISIBLE_DEVICES="" $BIN/llama-perplexity -m $HOME/w45_VRAM20.gguf \
  -f $HOME/long_text.txt -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup
# Final estimate: PPL = 7.9581 +/- 0.08082
```

Driver script `~/ud_stage4.sh`, log `~/ud_stage4.log`.
