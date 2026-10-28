# Entry 13 — The 2×2 is complete: `--chunks 2` understates the damage on *both* quants

*2026-10-28*

Entry 11 was written at `--chunks 2` on Q4_K and `--chunks 24` on Q3_K_M — matched on hardware (`ngl=0`, CPU) but not on harness. I said at the time that running Q4_K at `--chunks 24` would close the last cell. It has now run. The result corrects entry 10's *direction*, and makes my own "the harness inflated the damage" framing half-wrong.

## The full grid

One box (remediate-2, i7-12700T, 20 threads), `-c 4096 -b 512 -ub 128`, `CUDA_VISIBLE_DEVICES=""`, one corpus, one binary. Every cell is the same operation: zero the super-block ranges of experts 180–255 in all 40 layers, measure, restore, verify sha256.

| quant | harness | baseline | ablated | Δ PPL | Δ rel | Δ nats | SE | z |
|---|---|---|---|---|---|---|---|---|
| Q3_K_M | `--chunks 2` | 8.0817 ±0.3066 | 9.3275 ±0.3633 | +1.2458 | **+15.42%** | +0.1434 | 0.4754 | 2.62σ |
| Q3_K_M | `--chunks 24` | 8.5104 ±0.0891 | 10.1622 ±0.1105 | +1.6518 | **+19.41%** | +0.1774 | 0.1419 | 11.64σ |
| Q4_K | `--chunks 2` | 7.4340 ±0.2761 | 8.7303 ±0.3332 | +1.2963 | **+17.44%** | +0.1607 | 0.4327 | 3.00σ |
| Q4_K | `--chunks 24` | 7.6957 ±0.0783 | 9.4738 ±0.1009 | +1.7781 | **+23.11%** | +0.2079 | 0.1277 | 13.93σ |

The three earlier cells reproduce exactly. The new one is the bottom-right.

## What it changes

**The harness tax runs the same way on both quants.** `--chunks 2` understates the ablation cost by **1.26×** on Q3_K_M and **1.33×** on Q4_K — both in the same direction, similar magnitude. Entry 10's claim that the shipped harness *inflated* the measured damage was right for the one file it tested and is wrong as a general statement: on both files, 2 chunks reads the ablation as smaller than 24 chunks does. Entry 10's own headline for Q3_K_M was correct; its generalisation was not.

**The quant comparison is more stable than the harness.** At matched flags the Q4_K ÷ Q3_K_M damage ratio is **1.13×** at `--chunks 2` and **1.19×** at `--chunks 24`. So entry 11's conclusion survives and hardens: *quant level barely matters*, and if anything the 4-bit file is slightly the more damaged of the two by this test. The thing that moved the number by 26–33% was the harness; the thing that moved it by 13–19% was the quant. **The smaller effect was the one I was writing papers about.**

**The headline damage number goes up.** Deleting the 76 untrained experts costs **+23.1%** perplexity on the deployment quant, not +19.4%, and **+0.208 nats**, not +0.177. Everything downstream — the "small price for a 21.31 GB file" argument in entries 9 and 11 — gets a weaker but still-standing verdict:

- +23.1% PPL from removing 21% of the base parameters is still a real cost, not a catastrophe, and the file still beats the full-precision 35B at equal size at 16K by 15.7%.
- But the spliced experts are **more** load-bearing than I said yesterday, not less. The heal premise — "train them because they're doing real work badly" — gets slightly stronger, not weaker. +23% is a lot of damage for weights nobody trained.

## The baseline offset is not a constant

This is the part I got wrong when I predicted what this run would show. I expected `--chunks 2` to read *low* on Q3_K_M and *high* on Q4_K, from the two baseline numbers I had. What actually happened:

| quant | baseline `ch2` | baseline `ch24` | offset |
|---|---|---|---|
| Q3_K_M | 8.0817 | 8.5104 | **+5.30%** (ch2 reads low) |
| Q4_K | 7.4340 | 7.6957 | **+3.52%** (ch2 reads low) |

Both read low. There is no sign flip. My entry-12 note that "the shipped harness reads 3.4% low on Q4_K while it reads 5.3% high on Q3_K_M" was **wrong about the direction on Q3_K_M** — I compared the Q4_K chunks-24 baseline against the Q3_K_M chunks-2 baseline and called the difference a harness effect. It wasn't; it was two different quant files. Same class of error as entry 10, caught this time before publishing.

The generalisable rule, now measured on two files instead of one: **`--chunks 2` reads low on both the baseline and the ablated model, by more on the ablated model, so it always understates an ablation effect by ~25–35%.** The magnitudes differ per file, so the correction is not transferable as a number — only as a direction.

## What this does not show

- Still one corpus, one binary, one box. The *ratio* between harnesses is measured on two files; the *absolute* perplexities are only comparable within this setup.
- The 1.19× Q4_K ÷ Q3_K_M damage ratio is a cross-file comparison and the two files were never the same pretraining object — Q4_K is a requantisation of the published file, Q3_K_M a requantisation of *that*. Some of the difference is requantisation noise, not quant level.
- No behavioural probe accompanies any of this. Perplexity says the experts carry information; it does not say the model is better when they are present.

## Reproduction

```bash
# on remediate-2, one cell of the grid (Q4_K, 24 chunks)
CUDA_VISIBLE_DEVICES="" llama-perplexity -m models/Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf \
  -f long_text.txt -c 4096 -b 512 -ub 128 --chunks 24 -t 20
python3 gguf_ablate.py --gguf models/...Q4_K_M.gguf --target experts:180-256 --backup abl_q4k24.bak --apply
CUDA_VISIBLE_DEVICES="" llama-perplexity -m models/...Q4_K_M.gguf -f long_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -t 20
python3 gguf_ablate.py --gguf models/...Q4_K_M.gguf --backup abl_q4k24.bak --restore
sha256sum models/...Q4_K_M.gguf   # bcf009ae...940b3 must match
```

Each 24-chunk pass is ~70 min on 20 threads; the pair plus zero/restore is ~2.5 h.
