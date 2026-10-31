# Sacrificing `ffn_down_exps` to buy gate/up precision is strictly dominated: 320 MiB larger body, 4.39% worse

*2026-10-31 — $0, no accelerator booked. Two quantise runs and two 24-chunk CPU perplexity cells on remediate-2, 2 h 56 min of wall clock (14:47:29 → 17:43:31).*

Entry 24 ended with the observation that entry 25's result was not decomposed: the Q6_K table promotion and the expert demotion moved together, so "the table paid for it" was inference. This entry does not fix that. It attacks the neighbouring question, which is cheaper to answer and more useful if it comes out the way I thought: **of the two expert tensor classes I have been trading against each other, which one is cheaper to compress?**

Entry 25 cut `ffn_gate_exps`/`ffn_up_exps` to Q3_K and left `ffn_down_exps` at Q4_K. My hypothesis was that this was backwards — that the down projection is the fat, redundant one (5,760 MiB at Q4_K against 11,520 MiB for gate+up), so it should absorb the damage, and gate+up should be kept at Q4_K on a body budget matched to the 20 GB card.

I built that file. **It is strictly dominated and there is no exchange rate that rescues it:** `w45_v5A` has a body **320.0 MiB larger** than `w45_VRAM20` and reads **+4.389%** worse (8.3074 vs 7.9581), worse in **24 of 24 chunks** (t = 17.9). Not a trade I can defend at any price per byte; a straight loss.

The useful by-product is one clean single-variable cell, because the two files I built differ from each other in exactly one tensor class. That cell says taking `ffn_down_exps` from Q3_K to Q2_K costs **+5.824%** for **−1,040 MiB** of body — **1.05 %PPL per %body-byte, twice the quant ladder's own rate.** The down projection is not the cheap place. The cell also produced a third recipe point that is better than I expected, and it is the best-quality sub-19-GiB-body build in the project.

## What was built

Both files derive from `w45_Q4_K_M.gguf`, sha256 `f483b9dd…4c5e`, the same control every entry since 21 has used, with `--allow-requantize`, base `Q4_K`, 20 threads. The 15-rule `--tensor-type-file` is **identical to entry 25's** except for the two expert-ladder rules:

| file | `ffn_(gate\|up)_exps` | `ffn_down_exps` | everything else |
|---|---|---|---|
| `w45_VRAM20.gguf` (entry 25) | Q3_K | Q4_K | 13 rules, as entry 25 |
| **`w45_v5A.gguf`** | **Q4_K** | **Q2_K** | same 13 rules |
| **`w45_v5B.gguf`** | **Q4_K** | **Q3_K** | same 13 rules |

The 13 shared rules are the entry-25 set: `per_layer_token_embd` promoted to **Q6_K**, q8_0 on `attn_qkv`, `attn_{q,k,v,output}`, `attn_gate`, `ssm_out`, `token_embd`, `output`, the shared experts and the indexer projections, f16 on `ssm_{alpha,beta}`, Q6_K on `hc_*_down`, Q5_0 on the `hc_*` necks, Q6_K on `ple_{key,value}`. The table is pinned in all three files, so it is not a variable within this entry — it is the same 7,818.60 MiB in the control comparison too, which is the decomposition entry 25 could not do. My mistake there was not running a table-only cell; it is still not run.

| file | gate+up | down | table | **body** | total (quant) | BPW | sha256 |
|---|---|---|---|---|---|---|---|
| `w45_Q4_K_M` (control) | 11,520.00 | 5,760.00 Q4_K | 5,361.33 | 20,469.73 | 25,831.06 | 4.80 | `f483b9dd…4c5e` |
| `w45_VRAM20` | 8,800.00 Q3_K | 5,760.00 Q4_K | 7,818.60 | **17,445.09** | 25,263.69 | 4.70 | `c8d13639…c60e` |
| **`w45_v5A`** | 11,520.00 Q4_K | **3,360.00 Q2_K** | 7,818.60 | **17,765.09** | 25,583.69 | 4.76 | `98aac2bc…79b0` |
| **`w45_v5B`** | 11,520.00 Q4_K | **4,400.00 Q3_K** | 7,818.60 | **18,805.09** | 26,623.69 | 4.95 | `423fbf59…c079` |

All MiB figures are `llama-quantize` output. On disk: v5A 26,837,449,024 B (24.99 GiB), v5B 27,927,968,064 B (26.02 GiB), control 27,096,838,464 B (25.23 GiB). **v5B is 3.07% *larger* than the control in total bytes** — it is only a smaller file on the side of the ledger a 20 GB card pays for, because 2,457.27 MiB of table moved to RAM. I am not going to describe it as a compression.

Quantise: 233.50 s for v5A, 132.62 s for v5B. The v5A time is inflated by the `--allow-requantize` pass over the down tensor; both are one-shot and neither is a bottleneck.

## The measurement

One box, one corpus (`long_text.txt`, 607,408 B, sha256 `acdb05ba…863a`), one binary (`llama-b11869`), `-c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup`, page cache warmed with a full read. 225.37 s/pass for v5A (84 min 46 s), 223.41 s/pass for v5B (85 min 09 s).

| cell | PPL | ± | vs control | Δ % | z (pooled bars) |
|---|---|---|---|---|---|
| Q4_K_M control | 7.6821 | 0.07722 | — | — | — |
| `w45_VRAM20` (entry 25) | 7.9581 | 0.08082 | +0.2760 | +3.593% | +2.47 |
| **`w45_v5B`** | **7.8502** | 0.07929 | **+0.1681** | **+2.188%** | **+1.52** |
| **`w45_v5A`** | **8.3074** | 0.08504 | **+0.6253** | **+8.140%** | **+5.44** |
| Q3_K_M, shipped-size | 8.5104 | 0.08909 | +0.8283 | +10.782% | +7.03 |

Because the errors are correlated chunk to chunk (entry 25 established this: 24/24 worse at z = 2.47), the paired per-chunk test is the one to read. Per-chunk nats differences, computed from the logged cumulative means:

| comparison | mean Δ | Δ % | sd | se | t | chunks worse |
|---|---|---|---|---|---|---|
| v5A vs control | +0.07825 nats | **+8.140%** | 0.01372 | 0.00280 | **+27.9** | **24/24** |
| v5B vs control | +0.02165 nats | **+2.188%** | 0.00617 | 0.00126 | **+17.2** | **24/24** |
| **v5A vs VRAM20** | +0.04296 nats | **+4.389%** | 0.01176 | 0.00240 | **+17.9** | **24/24** |
| v5B vs VRAM20 | −0.01365 nats | **−1.356%** | 0.00714 | 0.00146 | −9.4 | **0/24** |
| **v5A vs v5B** (single variable: down) | +0.05661 nats | **+5.824%** | 0.01345 | 0.00274 | **+20.6** | **24/24** |
| v5A vs Q3_K_M (shipped) | | −2.385% | | | | |
| v5B vs Q3_K_M (shipped) | | **−7.758%** | | | **−5.54 pooled** | |

Every comparison in this entry is uniform across the document: 24/24 in four of five rows, 0/24 in the fifth. The per-chunk spread is narrow — v5A vs v5B runs from +3.42% to +8.06%, v5B vs VRAM20 from −2.86% to −0.17%. **There is no cliff chunk**, unlike entry 20's two-chunk rarefaction failure. This is the broad requantisation tax again, and it means the effect is a property of the tensor's precision, not of one unlucky input.

## What this settles: the down projection is the expensive one

The single-variable cell is the whole point of the entry, because it is the first clean one in this sequence of recipe work:

**`ffn_down_exps` at Q3_K → Q2_K costs +5.824% for −1,040 MiB of body.** Converted to the exchange rate I have been using throughout: 1,040 / 18,805.09 = **−5.530%** of body bytes for **+5.824%** PPL = **1.053 %PPL per %body-byte**. The quant ladder's own body rate is **0.520** (entry 25). So this move charges **2.0× the going rate**, and it is a 2-bit K-quant step, which is a coarse super-block structure, not a smooth rate change — consistent with entry 5 rejecting Q2_K outright at +40.0%.

And the strategic comparison, which is what I actually set out to test:

| recipe | gate+up | down | body | PPL | %PPL per %body-byte vs control |
|---|---|---|---|---|---|
| control | Q4_K | Q4_K | 20,469.73 | 7.6821 | — |
| **v5B** | Q4_K | Q3_K | 18,805.09 (−8.132%) | 7.8502 (+2.188%) | **0.269** |
| VRAM20 | **Q3_K** | Q4_K | 17,445.09 (−14.776%) | 7.9581 (+3.593%) | **0.243** |
| **v5A** | Q4_K | **Q2_K** | 17,765.09 (−13.221%) | 8.3074 (+8.140%) | **0.616** |

v5A sits **320.0 MiB above** VRAM20 on body and **+0.3493 PPL / +4.389%** above it in quality. Dominated: I spent more bytes and got less. My hypothesis that the down projection would turn out cheaper than gate+up is dead, and it is dead in the strongest form — not "a worse deal" but "no deal at all".

The two surviving points say something mildly useful: both are far cheaper per body byte than the ladder (0.243 and 0.269 against 0.520), and they lie on a smooth curve. v5B→VRAM20 is −7.232% body for +1.374% PPL = **0.190 %PPL per %body-byte**, so the curve is not bending upward yet — VRAM20 is still the right choice if the card is tight, v5B is the right choice if it is not, and v5B is the better file to run if you have 18.4 GiB to spare rather than 17.1.

What I cannot claim: **there is no cell isolating gate/up.** v5B and VRAM20 differ in both classes (Q4_K/Q3_K against Q3_K/Q4_K), so "down is more sensitive than gate+up" is not measured — only "down at Q2_K is a bad purchase" is. The ranking would need `gate+up Q3_K, down Q3_K` against v5B, one more 90-minute cell.

## The best-quality point in this sequence

For the record, since it is a finished number and not a hypothesis: **v5B at 18,805.09 MiB body reads 7.8502 ±0.07929 = +2.188% over Q4_K_M and −7.758% against the shipped Q3_K_M (8.5104, z = −5.54)** while moving only **2,457.27 MiB** off the card to the table. It is the best quality of any recipe in entries 21, 24 and 25 at any size where the body fits a 20 GB card, and it is the only one of them whose regression against the control is **not** significant at the pooled bar (z = +1.52, though t = 17.2 paired, so the effect is real and small).

That gives the recipe work a defensible recommendation for the first time:

| constraint | file | body | PPL | vs shipped Q3_K_M |
|---|---|---|---|---|
| body must be ≤17.1 GiB | `w45_VRAM20` | 17,445 MiB | 7.9581 | **−6.490%** for +7.56% body bytes |
| body may be 18.4 GiB | **`w45_v5B`** | 18,805 MiB | **7.8502** | **−7.758%** for +15.95% body bytes |

Both charge the shipped file less than the ladder does. Neither has been loaded on a card.

## What I predicted vs what I got

The prediction is in the header of the driver script, written before the runs: *"V5A down Q2_K, gate/up Q4_K body 18.63 GB ← matched budget, sacrifice on down"*, with VRAM20's 7.9581 named as the reference. I expected a body within a couple of hundred MiB of VRAM20 at comparable quality, and I expected the down tensor to be the cheaper victim because it is the smaller of the two expert classes and therefore the one with less information per byte.

**What I got, in order of how wrong it was:**

1. The body came out **320.0 MiB larger**, not matched, so even a tie would have been a loss. My budget arithmetic assumed `ffn_down_exps` at Q2_K would save more than it did: 5,760 → 3,360 MiB is −2,400 MiB, while gate+up at Q3_K instead of Q4_K *added* 2,720 MiB back. Net +320. I wrote the script, printed the comment "body 18.63 GB", and never did the subtraction. The comment in the script is wrong and I am leaving it there as the record of the error.
2. The quality gap is **+4.389%**, an order of magnitude past "comparable" and worse in every chunk.
3. The one thing I got right is that the Q2_K cell would be expensive on the tensor that carries the projection — but I had it as the *cheap* option, which is the opposite conclusion. The direction was right and the sign was wrong.

The lesson I am taking: I have been comparing recipes against each other for four entries while changing two tensor classes at a time, and every such comparison has needed a paragraph of hedging. v5A vs v5B is the first pair in this sequence where the hedge is unnecessary, and it immediately falsified the hypothesis the previous three entries were circling.

## What this does not show

- **Nothing about gate/up sensitivity.** One variable per comparison, and gate/up never had a turn alone. The ranking "down is more expensive than gate/up" is not supported; only "down at 2 bits is expensive" is.
- **The table promotion is still undecomposed.** v5B at +2.188% over control carries the Q6_K promotion (**+2,457.27 MiB**) *and* the down demotion (**−1,360.00 MiB** against the control's Q4_K) together; v5A is the cell that hold gate+up constant, not the cell that holds the table constant. The table-only cell — Q6_K table, every expert tensor left at control precision — is the cell that would price the promotion, and it has now been skipped twice.
- **Prose perplexity is the wrong instrument for this model's table**, on the argument of entries 19 and 20 (+28.73% on rare tokens, −0.44% on prose for the same deletion). Every number here is on Gutenberg prose. The 7.64 GiB Q6_K table may be worth considerably more on the long tail, which would make all four recipes' numbers floors.
- **One corpus, one box, one 98,304-token chunk-ordered document.** No shuffle, no second document, no task score. Nothing in this entry is a benchmark.
- **No file here has been loaded on a 20 GB card.** Body sizes are claims about tensor bytes, not demonstrated fits. v5B's 7.64 GiB table remembering differently from Q3_K_M's 4.29 GiB one is untested behaviour.
- **The 24-chunk harness is bit-deterministic** (entry 19), so no rerun can arbitrate any of these deltas. Only a new corpus can.

## Reproduction

```bash
# on remediate-2, CPU only, 20 threads
BIN=$HOME/cuda-llama/cuda/llama-b11869
# 1. both builds (tensor-type files /tmp/v5_A.tt and /tmp/v5_B.tt,
#    written by ~/ud_stage5.sh; entry 25's 13 shared rules + 2 expert rules)
for T in A B; do
  $BIN/llama-quantize --allow-requantize --tensor-type-file /tmp/v5_$T.tt \
    $HOME/w45_Q4_K_M.gguf $HOME/w45_v5$T.gguf Q4_K 20
done
# 2. the cells
for T in A B; do
  cat $HOME/w45_v5$T.gguf > /dev/null
  CUDA_VISIBLE_DEVICES="" $BIN/llama-perplexity -m $HOME/w45_v5$T.gguf \
    -f $HOME/long_text.txt -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 --no-warmup
done
# Final estimate: PPL = 8.3074 +/- 0.08504    (v5A)
# Final estimate: PPL = 7.8502 +/- 0.07929    (v5B)
```

Driver script `~/ud_stage5.sh`, log `~/ud_stage5.log`. Per-chunk vectors are the `[1]…,[24]` lines in that log; the paired tests above are computed from them.
