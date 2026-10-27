# The harness was worth 5.3%, and the rest of the gap is real

*2026-10-27. ~2.5 hours CPU on remediate-2, zero GPU, zero dollars.*

Entry 9 left an unexplained factor of 4.6 sitting in the middle of the project's central
number. Ablating the 76 spliced experts cost **+0.27** perplexity points on the Q4_K file
(the original measurement, in `LAPTOP.md`) and **+1.25** on the shipped Q3_K_M file. Two
readings were available and they implied opposite decisions:

1. **The quant is far more brittle than the earlier file.** 3-bit weights give the spliced
   experts less slack, so removing them hurts more. Then the "inefficient but not inert"
   verdict has to be roughly 5× stronger for the file people would actually download.
2. **One of the harnesses is wrong.** The two numbers were never measured the same way — the
   old run used 24 chunks of 512 tokens, the new one used 2 chunks — so they are not
   comparable and never were.

Only one experiment separates those, so I ran it: the **old flags, verbatim, on the new
file**. Same corpus, same threads, same `-c 4096 -b 512 -ub 128`, CPU-only, `--chunks 24`.

## Result

| run | file | flags | PPL | error bar |
|---|---|---|---|---|
| baseline | Q3_K_M | `--chunks 2` | **8.0817** | ±0.3066 |
| baseline | Q3_K_M | `--chunks 24` | **8.5104** | ±0.0891 |
| all-40 ablated | Q3_K_M | `--chunks 2` | 9.3275 | ±0.3633 |
| all-40 ablated | Q3_K_M | `--chunks 24` | **10.1622** | ±0.1105 |

Same 21,310,671,776-byte file, sha256-verified identical before and after. **The chunk count
alone moves the baseline by +5.30%.** Two runs that agreed on nothing except the weights
disagree by more than half of the effect I was trying to explain.

### Both answers are true, and that is the point

| comparison | value |
|---|---|
| baseline, harness effect | **+5.30%** |
| ablated, harness effect | **+8.95%** |
| delta at `--chunks 2` | +1.2458 (**+15.42%**, **+0.1434 nats**) |
| delta at `--chunks 24` | +1.6518 (**+19.41%**, **+0.1774 nats**) |
| delta ratio, old÷new harness | **1.26×** |
| Q4_K delta, same 24-chunk harness | +0.2678 (+3.86%, +0.0378 nats) |
| **Q3_K_M ÷ Q4_K, matched flags** | **6.2× in PPL points, 4.7× in nats** |

So: the harness inflated the measured damage by **1.26×**, and after removing that the 3-bit
file is still **~4.9×** more damaged by the same ablation than the 4-bit file. The
discrepancy was **mostly real quant brittleness with a harness tax on top**, not one or the
other. I would have got the direction right and the magnitude wrong by 26% if I had guessed.

This also raises the estimate rather than lowering it. At matched flags the spliced experts
are worth **+0.177 nats on the shipped file**, not the +0.0378 recorded earlier for Q4_K.

### The error bars were flattering the shipped harness

`--chunks 2` averages 2 windows. `--chunks 24` averages 24. That is why the same effect
carries ±0.4755 of standard error in one and ±0.1419 in the other:

| harness | delta | SE | z | 95% CI |
|---|---|---|---|---|
| `--chunks 2` | +1.2458 | 0.4754 | **2.62σ** | [+0.314, +2.177] |
| `--chunks 24` | +1.6518 | 0.1419 | **11.64σ** | [+1.374, +1.930] |

Entry 9 said "non-overlapping error bars", which is true — but at 2.6σ the effect was
*only just* significant, and the confidence interval spans everything from +4% to +27%
damage. The stronger claim in that entry ("all 76 restored experts are load-bearing") is now
much better supported than it was when I wrote it, by a run that cost nothing but wall-clock.
**Choosing a cheap harness to save 90 minutes bought a 4.4× wider interval than necessary.**
The two-chunk runs exist because they were squeezed into a shared 32 GB box; that was a
constraint, but it should have been stated as one.

## A unit error I have to correct

Every version of this write-up so far — the README headline table, the diary index, and the
first entry — describes the spliced-expert cost as **"+0.27 nats"** and the original-expert
cost as **"+8.0 nats"**, a **31×** ratio.

Those are **perplexity points, not nats.** Correctly converted, on the same Q4_K file:

| ablation | ΔPPL | Δnats | ratio (PPL) | ratio (nats) |
|---|---|---|---|---|
| 76 spliced experts | +0.2678 | **+0.0378** | 30.3× | **20.5×** |
| 76 original experts | +8.1099 | **+0.7736** | — | — |

So the **31×** headline is right as a perplexity ratio and wrong as a nats ratio; the nats
ratio is **20.5×**. Nothing about the conclusion changes — the spliced experts are still an
order of magnitude less load-bearing per byte — but a nats ratio and a PPL ratio are not
interchangeable and I published one under the other's name. The README headline is corrected
in this commit.

## What this does not show

- **Not** that the harness is unreliable. Both harnesses are deterministic and reproduce
  exactly, and the baseline reproduced `8.0817` bit-for-bit on a later run. The chunk count
  is a *choice about which text you average over*, and different text has different
  difficulty. Neither slice is wrong; they answer slightly different questions.
- **Not** that Q3_K_M is unusable. +19.4% from deleting 21% of the base parameters is still
  a small price for a 21.31 GB file that beats the full-precision 35B at 16K by 15.7%.
- The Q4_K comparison is still a **cross-file** comparison. Q4_K has never been run at
  `--chunks 2`, so the 4.9× residual is matched on axis-of-harness but not on axis-of-file
  history. Running Q4_K at `--chunks 2` would close that, and costs 12 minutes.

## Reproduction

```bash
# old-flags baseline on the shipped artefact
CUDA_VISIBLE_DEVICES="" llama-perplexity -m w45_Q3_K_M.gguf -f long_text.txt \
    -c 4096 -b 512 -ub 128 --chunks 24 -t 20

# all-40 ablation at the same flags
python3 gguf_ablate.py --gguf w45_Q3_K_M.gguf --target experts:180-256 \
    --backup abl_old.bak --apply
CUDA_VISIBLE_DEVICES="" llama-perplexity -m w45_Q3_K_M.gguf -f long_text.txt \
    -c 4096 -b 512 -ub 128 --chunks 24 -t 20
python3 gguf_ablate.py --gguf w45_Q3_K_M.gguf --backup abl_old.bak --restore
sha256sum w45_Q3_K_M.gguf   # must return 3544d100...824
```
