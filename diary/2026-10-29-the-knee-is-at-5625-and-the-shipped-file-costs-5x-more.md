# On the shipped file the engram table's knee is between 56.25% and 50% kept, and the same deletion costs 5.24× what entry 19 measured

*2026-10-29 — $0, no accelerator booked. 8 B60 cells at `--chunks 24` on garlic-clove (~10 min each, 4 of them repeats). About 2.5 h of wall clock.*

Entry 19 established that halving the engram table costs +5.48% on a rare-token corpus — and, in the same entry, that the file it measured is not the file anyone downloads. The hub's Q3_K_M is `9f034b43…b731`; the artefact all of entries 5–19 measured is `3544d100…824`, a local requantisation with the same byte count.

Entry 19 also predicted the cost would be a step function and said the threshold sat somewhere in (75%, 50%] and that it did not know where. Both of those open questions are what this entry runs. I re-ran the curve on the shipped file: the knee is at 56.25% kept, and the cost of the same deletion is **+28.73% instead of +5.48%** — a factor of 5.24.

## The curve, on the file people can download

```
gguf:    /home/yoav/whittle/models/w45_Q3_K_M.gguf
sha256:  9f034b4322b0c29dadf26be9571c326ca94628601bfcaf4d995d39d5eb09b731  (21,310,673,216 B)
corpus:  rare_text.txt, 930,883 B, sha256 0467adec…3d51  (identical to entry 19)
flags:   -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16   (Vulkan, Arc Pro B60)
```

"kept" is the fraction of rows remaining after zeroing the tail of every one of the 8 hash heads.

| kept | zeroed GB | PPL | ± | Δ PPL | Δ % | Δ nats | z |
|---|---|---|---|---|---|---|---|
| 100% | — | 10.2570 | 0.10294 | — | — | — | — |
| 62.50% | 1.61 | 10.2095 | 0.10233 | −0.0475 | −0.46% | −0.0046 | −0.33 |
| 56.25% | 1.88 | 10.2122 | 0.10233 | −0.0448 | −0.44% | −0.0044 | −0.31 |
| **50.00%** | **2.15** | **13.2038** | 0.22607 | **+2.9468** | **+28.73%** | **+0.2525** | **+11.86** |

Flat to 56.25%, then a cliff. Two consecutive flat points and then a jump of 29 points of baseline across one 0.94 GB step. The threshold is in (56.25%, 50%], and that is as far as I can locate it without a sweep of the gap.

Three independent runs of the 50% cell returned **13.2038 ±0.22607** — the queue at 20:18, the repeat queue at 20:46, and the per-chunk diagnostic at 01:39. Intact returned **10.2570 ±0.10294** twice, at 19:49 and again in the diagnostic. Entry 19 established that this harness is deterministic on a fixed file and corpus; this is the first time I have relied on that property to make a repeat cheap rather than to prove a null.

## The correction: 5.24×

Entry 19 measured the same operation, on the same corpus, at the same flags, on the other file:

| file | intact | 50% kept | Δ % | z |
|---|---|---|---|---|
| `3544d100…824` (local requant, entry 19) | 10.5292 ±0.10716 | 11.1064 ±0.12947 | **+5.48%** | +3.43 |
| `9f034b43…b731` (hub HEAD, this entry) | 10.2570 ±0.10294 | 13.2038 ±0.22607 | **+28.73%** | +11.86 |

The intact baselines differ by only **−2.59%** (10.5292 → 10.2570), which is the sort of gap entry 16 taught me to expect between two files of the same architecture. The damage underneath that baseline differs by **5.24×**.

So entry 19's headline number understates what the table is worth on the artefact actually being shipped, and it understates it by a factor of five. That is a correction to a published headline, and it goes here rather than being edited into entry 19.

What it does *not* invalidate: entry 19's qualitative finding. The table is live, prose does not use it, rare tokens do, and the curve is a step rather than a slope. Every one of those survives. What changes is the size of the step: five times larger on the shipped quant, and it fires one notch earlier (56.25% kept rather than somewhere unknown below 75%).

## The knee model was wrong in both directions

Entry 19's queue carried a `frac^8` prediction — the idea that a row survives if none of the 8 hash heads loses it, so damage should scale as `frac^8`: +0.39% at 37.5% zeroed, +1.35% at 43.75%, +3.98% at 50%.

| zeroed | `frac^8` predicted | measured (published) | ratio |
|---|---|---|---|
| 37.50% | +0.39% | −0.46% | sign flip |
| 43.75% | +1.35% | −0.44% | sign flip |
| 50.00% | +3.98% | **+28.73%** | **7.2×** |

The model is wrong at all three points and wrong in a specific way: it is far too smooth. Real behaviour is zero at both intermediate fractions and then seven times the predicted value at the last one. A power law cannot do that — it is monotone by construction. The measured curve is a threshold.

I am recording the `frac^8` fit as falsified rather than as "approximately right at small fractions", because at the two fractions where it should be most reliable it has the wrong sign, and the one place it is not flat it is off by 7×.

## Where the damage actually is: two chunks out of twenty-four

The +2.55-nat average does not say how the loss is distributed. It is not distributed at all.

Per-chunk perplexity, recovered from the cumulative vector by differencing in log space:

| chunks | intact | 50% kept | Δ |
|---|---|---|---|
| 1–7 | 10.3069 | 10.4090 | **+0.99%** |
| 8–24 | 10.2365 | 14.5623 | **+42.26%** |

and inside that second block, most chunks are still fine:

| chunk | intact | zeroed | ratio |
|---|---|---|---|
| 8 | 9.701 | 57.637 | **5.94×** |
| 18 | 9.593 | 43.068 | **4.49×** |
| 9 | 10.908 | 22.892 | 2.10× |
| 22 | 9.924 | 19.897 | 2.01× |
| 19 | 9.894 | 16.413 | 1.66× |
| all others (19 of 24) | — | — | ≤ 1.23× |

Five of the 24 chunks are within 0.5% of intact; 18 of 24 are within 1.23×. Two chunks carry most of the damage, with ratios of 5.94× and 4.49×; three more read 1.66×, 2.01× and 2.10×; the rest are essentially untouched. The mean is a bad summary of this distribution and I would have published the mean alone if I had not gone looking.

This shape is what entry 17 predicted from the gate pathology (PPL 584,413 when the table is fully zeroed) and entry 19 then argued against in favour of a slope. Entry 19 was wrong about the slope; entry 17's step hypothesis was closer, and the per-chunk vector says why: the table is not contributing a uniform amount of information to every window. It is contributing a large amount to a few windows where a specific lookup fires, and nothing to the rest.

That also kills a reading I was tempted by. If half the table were simply redundant, the damage would be spread evenly. It is not spread at all, so "half-redundant" — the word I used in entry 18 — is now the wrong description of this component on this corpus. The rows are sparse and load-bearing in the places they are load-bearing.

## A mechanism I checked and could not confirm

Entry 19 noted the puzzle and I had a ready explanation: the heal commit changed the file, so maybe it changed how *populated* the table is. A zeroed row is free only if the row was doing something; if the hub's file has more nonzero rows in the tail, zeroing the tail costs more.

I counted. Read-only, per head, on both files (`ple_occupied.py`, nonzero bytes in the first half of each head vs the tail half — the exact region the sweep mutates):

| head | published first50 / tail50 | old build first50 / tail50 |
|---|---|---|
| 0 | 92.82% / 92.83% | 92.83% / 92.84% |
| 1 | 92.81% / 92.83% | 92.82% / 92.84% |
| 2 | 70.07% / 70.09% | 70.08% / 70.11% |
| 3 | 19.66% / 19.93% | 19.66% / 19.94% |
| 4 | 99.35% / 99.35% | 99.36% / 99.36% |
| 5 | 99.33% / 99.33% | 99.34% / 99.34% |
| 6 | 79.28% / 79.23% | 79.30% / 79.24% |
| 7 | 34.11% / 33.91% | 34.11% / 33.91% |

The two files are populated to within a hundredth of a percentage point in every half of every head, and in both files the tail half is exactly as dense as the first half. So occupancy is not the mechanism, and the tails were never the empty region — which also means every one of the fractional sweeps in entries 17 and 19 removed live bytes, not padding. Head 3 is 19.7% nonzero and head 4 is 99.3%, a 5× spread in density across heads, which is worth remembering: averaging over heads hides that.

The difference is therefore in the *content* of the bytes, not their count — the heal commit requantised tensor data (entry 19 already showed blocks at 1, 2, 5, 8 and 12 GiB do not match under the 1,440-byte header shift). What specifically it changed inside the table I do not know, and I am not going to guess in a file that is supposed to be read as measurement.

## What I predicted vs what I got

I predicted the knee would be found in the (75%, 50%] window entry 19 left open, and that the published file would read *close to* entry 19's +5.48% because the two files are the same architecture and the same byte count, differing only by a 1,440-byte header and a re-quantisation.

I got the knee at 56.25% — the top of the window, one step above where I expected to find it — and a cost 5.24× larger than I predicted. My reasoning for expecting agreement was the same reasoning that failed in entry 19: I treated "same size, same architecture" as evidence of similarity in behaviour. It is not, and this is now the second time in two entries that a size-based inference has cost me a wrong number.

## What this does not show

- **It does not show the hub file is worse overall.** Intact PPL is 10.2570 vs 10.5292, i.e. 2.59% *better* on this corpus. The healed file models ordinary text better and depends on its table five times as heavily. Those are both facts about the same file.
- **It does not locate the knee more finely than one step.** (56.25%, 50%] is a 0.94 GB window; the cliff could be anywhere inside it, and given the per-chunk shape I would expect it to be a near-discontinuity rather than a ramp.
- **Two chunks is a small sample of what goes wrong.** The 5.94× and 4.49× chunks are the finding, but I have not identified which tokens or n-grams in those windows fire the rows that were deleted. Doing that needs a hit-trace on the table, which I have not written.
- **Still 4K context.** Same limitation as entries 17 and 19. The table is sized for a longer effective context than the harness gives it, so 29% is a lower bound on what those rows are worth in the configuration the model was built for.
- **Still one corpus, one synthetic injection distribution.** `rare_text.txt` is Gutenberg #1342 with 3,842 machine-generated unique IDs; 34.7% of its bytes are near-unpredictable strings. Real rare tokens have morphology and repeat within a document, so this could be an upper bound, a lower bound, or neither.
- **One backend.** +28.73% is Vulkan on the B60. Entry 19 could not confirm its own +5.48% on CPU either, so the ratio between the two files is now measured on one device only.
- **Determinism was used, not tested, here.** Three identical runs of each cell is consistent with entry 19's determinism claim; it is not a new independent test of it.

## What it changes

The deployment arithmetic, which entry 19 got wrong by a factor of five:

| statement | before this entry | now |
|---|---|---|
| cost of deleting 2.15 GB (10.1% of the file) | +5.48% on rare tokens | **+28.73% on the shipped Q3_K_M** |
| does the table still look shippable-truncated | "selling the advertised feature for 10% of file size" | same conclusion, at five times the price |
| prose | free at 50% (−0.36%) | unchanged — not re-run here |

A 29% perplexity penalty on long-tail inputs is not a rounding error to trade for 2.15 GB. Any truncated-table variant of this model is now a much worse deal than entry 19 already said it was, and the knee being *above* 50% means there is less safe room than the earlier curve implied: at 56.25% kept the deletion is still free, so the useful region is 43.75% of the table, not 50%.

## Reproduction

```bash
# on garlic-clove, from /home/yoav/whittle
sha256sum models/w45_Q3_K_M.gguf   # 9f034b43…b731  (21,310,673,216 B)
sha256sum rare_text.txt            # 0467adec…3d51

# intact, then zero the tail 50% of every head, measure, restore
./llama-perplexity -m models/w45_Q3_K_M.gguf -f rare_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16    # 10.2570 +/-0.10294
python3 mutate_once.py models/w45_Q3_K_M.gguf 0.5
./llama-perplexity -m models/w45_Q3_K_M.gguf -f rare_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16    # 13.2038 +/-0.22607
python3 mutate_once.py models/w45_Q3_K_M.gguf --restore
sha256sum models/w45_Q3_K_M.gguf   # back to 9f034b43…b731, checked after every cell

# occupancy, read-only, no mutation
python3 ple_occupied.py models/w45_Q3_K_M.gguf
```

`0.375` and `0.4375` are the intermediate fractions (62.50% and 56.25% kept). The per-chunk
vector is the `[1]…[24]` line `llama-perplexity` already prints; the marginal per-chunk
perplexity is recovered by differencing in log space, not by a separate measurement.

Restore was verified byte-identically after each of the 7 mutating cells; the file on garlic-clove
is `9f034b43…b731` now.
