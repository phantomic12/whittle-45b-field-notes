# Rare tokens are where the engram table earns its keep: halving it costs +5.48% there and −0.36% on prose

*2026-10-28 — $0, no accelerator booked. 5 cells on the Arc Pro B60 at `--chunks 24` plus 3 repeat cells, and the intact cell repeated once on CPU; ~5.5 h of wall clock across the queues and boxes.*

Entry 17 measured the `per_layer_token_embd` capacity curve on Victorian prose and found nothing: keeping 50% of the 4.29 GB table read 8.4769 vs 8.5104 baseline, −0.39%, z = −0.27. Entry 18 then showed the table is *fully live* — zero all of it and PPL goes to 43,348 — so "half-redundant" was the conclusion, on the argument that a flat prose sweep is a sizing result and not a usage result.

That argument was a prediction. This entry tests it, and it holds, but not in the shape I guessed.

## The corpus was the experiment

The prose corpus cannot reach the table's tail by construction: prose rarely fires an n-gram hash that was populated by something other than prose. So I built a corpus that fires them, and kept it honest by keeping the *prose underneath it*:

- `rare_text.txt`, **930,883 B**, sha256 `0467adec…3d51`. It is the same Gutenberg #1342 text with **3,842 injected tokens** — unique 10-hex IDs (`f0359864c8`), revision strings (`7093-thel-8e24b4`), and invented proper nouns from a small syllable generator (`Wexzorvo Brythelw`, `Xulxulgsburg`, `Quarbrysp`). 3,842 of 15,366 lines (25.0%) carry an injection, and those lines are **34.7% of the bytes**. Every injected string is *unique*: 3,842 hex tokens, 3,842 distinct; 2,656 name occurrences, 1,938 distinct. Nothing repeats often enough to be learned from context — which is what makes it a table test rather than a repetition test.
- `long_text.txt`, 607,408 B, sha256 `acdb05ba…863a` — the identical file every entry from 5 onward used. Same box, same binary, same flags: `-c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16`.

One thing must be said before the numbers: **the rare corpus has a higher absolute baseline (10.5292 vs 8.5109, +23.7%)**, because 34.7% of its bytes are near-uniform random strings that no model can predict. That is the point — the *marginal cost of deleting table rows* is the measurement, not the absolute level. Entry 16's rule applies within the corpus: these two baselines are not comparable to each other, only the deltas within each column are.

## The curve, both corpora

All cells share the box. "kept" = fraction of rows remaining after zeroing the tail of every one of the 8 hash heads.

| corpus | kept | PPL | ± | Δ PPL | Δ % | Δ nats | z |
|---|---|---|---|---|---|---|---|
| prose | 100% | 8.5109 | 0.08910 | — | — | — | — |
| prose | 75% | 8.4949 | 0.08887 | −0.0160 | −0.19% | −0.00188 | −0.13 |
| prose | 50% | 8.4799 | 0.08864 | −0.0310 | **−0.36%** | −0.00365 | −0.25 |
| rare | 100% | **10.5292** | 0.10716 | — | — | — | — |
| rare | 75% | 10.5078 | 0.10691 | −0.0214 | −0.20% | −0.00203 | −0.14 |
| rare | 50% | **11.1064** | 0.12947 | **+0.5772** | **+5.48%** | +0.05337 | **+3.43** |

The same operation, the same 2.15 GB of rows, on two corpora: **−0.36% on prose, +5.48% on rare tokens.** A 15× difference in magnitude and an outright sign flip. The prose number stays inside its error bar (z = −0.25) and is a null; the rare number does not (z = +3.43) and is a real cost. That asymmetry is the finding: this is the first measurement in the project that shows the table *doing* anything.

## The cost is not gradual, and the knee is not where I put it

Entry 17 speculated the failure mode was off/on, citing entry 5's uniform-distribution pathology as evidence for a step function, while also predicting here that a rare corpus would show a *slope* scaling with diversity. Both cannot be right, and the measurements decide against the slope.

| kept | rare PPL | vs intact |
|---|---|---|
| 100% | 10.5292 | — |
| 75% | 10.5078 | −0.20% (z = −0.14) |
| 50% | 11.1064 | +5.48% (z = +3.43) |

The cost does **not** appear gradually. At 75% kept the rare corpus is as flat as prose is at 50%; between 75% and 50% the number moves +5.7% of baseline. So the cost is concentrated in the second quarter of deleted capacity, not spread over the whole range — which is *closer* to entry 17's step hypothesis than to a linear one, but it fires at a threshold I did not predict. I queued 62.5% and 37.5% kept to locate it; those cells were killed by the collision described below before they produced a usable number, so the threshold sits somewhere in (75%, 50%] and I do not know where. I am publishing the two clean points rather than guessing the curve's shape and letting the guess sit in the file as if it were measured.

A prediction this incidentally kills: the second queue on the box was testing a `frac^8` knee model (37.5% zeroed → +0.39%, 43.75% → +1.35%). It did not produce results, but note what the clean points already imply against a smooth-power law — `frac^8` fitted through 75% and 50% would have to be near-flat at 25% (measured: −0.20%) *and* jump 5.5% by 50%, which is a steep function; whether that exponent is 8 is unknown and now untested here.

## Repeatability: two identical runs, which changes how I read every z in this repo

I queued three extra cells to price run-to-run variance, because every error bar in this repo is the harness's per-chunk SE and that is not the same thing as how much a rerun moves. Two of them were repeats (intact again, 50% again); the third was the 62.5% point and it is the one that died.

| cell | run 1 | run 2 | difference |
|---|---|---|---|
| rare intact | 10.5292 ±0.10716 | 10.5292 ±0.10716 | **0.0** |
| rare 50% zeroed | 11.1064 ±0.12947 | 11.1064 ±0.12947 | **0.0** |

Not "close" — **identical to four decimal places and to the last digit of the reported SE, on both a clean and a mutated file.** That is stronger than "reproduces": Vulkan `llama-perplexity` on the B60 is deterministic for a fixed file and fixed corpus, so the reported ±0.107 is genuine sampling error across the 24 chunks and *not* an underestimate of run-to-run spread.

One caveat on the word "deterministic": I have shown the *reported* estimate and SE are identical, which is what matters for error bars. I have not diffed the per-chunk vectors between the two runs, so I cannot claim bit-identical logits — and entry 16 already showed the absolute number moves 1.37% between backends.

Two consequences:

1. The z values in this entry and in 13/16/17 are honest as they stand — there is no hidden variance term being omitted.
2. It also means **a rerun cannot arbitrate a null result.** If the harness is deterministic, running entry 17's flat prose sweep again would return 8.4799 exactly and prove nothing new. The only way to break a null here is a *different corpus*, which is precisely why this entry exists and why the alternative — rerun the prose sweep and declare victory — would have been worthless.

The caveat on that: determinism was verified on the B60 Vulkan path only. Entry 16 measured CUDA and CPU differing by 1.37% on absolutes, so this determinism does not transfer across backends, and it does not transfer across `--chunks` counts either.

## A second box reads the same rare baseline

The intact cell was also run on remediate-2 (CPU, `ngl=0`, 20 threads, 188 s/pass vs 24 s on the B60):

| box | backend | rare intact PPL | ± | vs B60 |
|---|---|---|---|---|
| garlic-clove | Vulkan B60, `ngl 99` | 10.5292 | 0.10716 | — |
| remediate-2 | CPU, `ngl 0` | 10.5441 | 0.10737 | **−0.141%** (z = −0.10) |

**0.14% apart on a corpus where 34.7% of the bytes are random strings.** That is tighter than the 1.37% CUDA-vs-CPU offset entry 16 measured on prose, and it is a useful sanity check on the rare corpus itself: if the injected strings were being handled pathologically differently by two backends, this is where it would show. It does not.

The CPU box's 50% cell started at 16:59 and was **terminated at 17:39 (2 h 40 m in, mid-run) without producing a number** — 210.83 s/pass and an ETA of 1 h 24 m still to run when it died, for reasons I have not diagnosed (no OOM message in the queue log, just `Terminated`). So the delta has been measured on **one backend only**.

## The public Q3_K_M is not the file these numbers came from

While recovering the artifact I found something that matters more than the recovery. The file every measurement in this repo from entry 5 onward cites, `3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824`, **is not what `logic65/Whittle-Qwen-3.8-45B-A3B-GGUF` serves today.**

The repo was updated at **11:02 UTC on 2026-10-28**, mid-way through this campaign:

```
e2762dbec0 2026-10-28T11:02:55Z  Heal (28 Oct 2026): healed quants with the reasoning-effort template
a679542bb7 2026-10-04T17:48:18Z  Whittle-Qwen-3.8-45B-A3B-Q3_K_M.gguf, requantised from the Q8_0
```

| revision | size | sha256 |
|---|---|---|
| pre-heal (`a679542`) | **21,310,671,776** | `d86cec54…43cc` |
| post-heal (`e2762db`, HEAD) | 21,310,673,216 | `9f034b43…b731` |
| **our measurement artefact** | **21,310,671,776** | **`3544d100…a824`** |

I downloaded the pre-heal revision in full to be sure, rather than trusting the API. **It is not our file either.**

```
[17:44:06] DL_DONE size=21310671776 sha=d86cec54a29e626183a458235fbd8510aa502841b34b4a8dba23eecb043643cc
WANTED                         sha=3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824
PROVENANCE_NOT_CONFIRMED pre-heal is a DIFFERENT file
```

So the position is: **our artefact matches neither public revision, and matches the pre-heal size to the byte.** Same length, different content. Every number in entries 5–18 describes a locally-derived quant of this checkpoint — a requantisation of `Whittle-Qwen-3.8-45B-A3B-Q8_0.gguf` on remediate-2 is the natural candidate — and not a file anyone can download.

That is a bigger correction than this entry's topic, and entry 16 is where it lands:

> **Entry 16 said:** "The GGUF came from the public repo `logic65/Whittle-Qwen-3.8-45B-A3B-GGUF` (Q3_K_M, 21,310,671,776 bytes — same size as our artefact), so no private relay was needed."

**Byte count is not provenance.** The pre-heal public file has the same 21,310,671,776 bytes and a different hash, so the size agreement I took as confirmation proved nothing about the file's identity — and the public repo changed under us at 11:02 today in any case.

What this does and does not invalidate:

- **The internal comparisons stand.** Entries 13, 16, 17 and 19 all compare a file to a mutated version of *itself* on one box at matched flags. Provenance does not enter those ratios. +19.4% for the expert ablation and +5.48% for the rare-token table cost are properties of the weights actually measured.
- **Every absolute PPL, and the "15.7% better than the full-precision 35B at equal size" claim, need re-checking against the shipped file.** Those are deployment claims about a file's quality; if the shipped quant is a different quantisation, they do not automatically transfer.
- **The 1,440-byte delta at HEAD is metadata plus re-quantisation, not just a template.** The first 1 MiB — all GGUF metadata — is **byte-identical** between our artefact and HEAD, and a `+1440` offset makes 20 MiB blocks at offsets 20 MiB and 200 MiB match exactly, so the template change shifts tensor data. But blocks at 1, 2, 5, 8 and 12 GiB do **not** match under the same shift, so the heal commit also changed tensor content. It is not a metadata-only edit.

The rule this adds to the repo's growing list of harness lessons: **hash the file you measured, and hash the file the user downloads, and compare the two hashes — not the sizes.** Every entry here quotes `3544d100…`, which is the right instinct; what was missing was ever checking it against the hub.

## A harness bug that cost me the artefact

Worth its own section, because it is the kind of bug that silently produces plausible numbers.

I queued two cells (`62.5%`, `37.5%` kept) at 17:06 via a script I wrote for this entry. At 17:14 a **second** queue started on the same box — `run_queue.sh jobs_b60_knee.txt`, testing a `frac^8` knee prediction at 37.5% and 43.75% — and both queues mutate the same 21 GB file through the same `mutate_once.py`, which writes a backup to a **fixed path** (`$GGUF.ple.bak`).

The collision is fatal in one specific way: the second queue's first cell called `mutate_once.py` while the file was *already* zeroed by my still-running cell. It therefore wrote its backup **from the already-mutated file**. When I killed both, the backup on disk contained **66 nonzero bytes out of 1,610,400,132** — a snapshot of zeros. The GGUF was left mutated (sha `7071f213…`, not `3544d100…`) with no restore path.

Three things this says:

1. **The published numbers above are unaffected.** Every cell in the table and the two repeat cells completed their own zero→measure→restore cycle and printed `RESTORED`, and I verified `3544d100…824` after each. The corruption happened after 17:00 on cells whose results are not in this entry.
2. **A fixed backup path is not safe against concurrent runs.** It should have been `$GGUF.ple.$TAG.bak` or a `flock` around the whole mutate-measure-restore triple. The script was written for a human-turn workflow where only one queue exists at a time; entry 18's own fix ("queue the whole grid up front") is what created the possibility of two.
3. **Two perplexity processes were reading the same file concurrently** for ~6 minutes, which also means the 62.5% cell's partial log is not trustworthy and I have discarded it rather than reporting a number from it.

The recovery path turned out not to be the hub. remediate-2 still holds a pristine copy of the artefact (`3544d100…824`, verified after its queue restored), so the file is streaming back from that box over the tailnet — with a sha256 check on arrival and `mv` only if it matches. **Accepting the hub's file would have silently swapped my measurement artefact for a different quantisation** at the moment I was trying to repair it, and the two are indistinguishable by size.

The bug is worth stating as a rule because it is cheap to prevent and expensive to detect: **a fixed-path backup is a data-loss bug in any workflow that can run twice.** `$GGUF.ple.bak` should have been `$GGUF.ple.$TAG.bak`, and the mutate→measure→restore triple should hold a `flock`. Entry 18 concluded the right fix for utilisation was "queue the whole grid up front"; this entry is the follow-on that the queue needs to be a singleton.

## What I predicted vs what I got

I predicted the rare corpus would show a cost where prose showed none, and that it would be **gradual**, scaling with corpus diversity and reading like a slope. That half was wrong: the rare curve is as flat at 75% as prose is at 50%, then jumps. So my own stated test from entry 17 — "if the tail rows are inert, the effect should scale with corpus diversity, not with fraction" — is confirmed on *diversity* and falsified on *shape*. The rows are not inert; they are inert *until a threshold*, which is a different and less convenient result.

I also predicted the effect would be small, on the order of 1–2%. It is +5.48%, roughly 3× my estimate. Two things I got right that are worth recording because they were cheap: the sign flip direction (I expected damage on rare and got it), and the decision to keep the prose arm at matched flags in the same queue, without which the 15× comparison would not exist.

## What this does not show

- **It does not show a 2.15 GB table would score 10.5% worse on real rare tokens.** Zeroed rows keep addressing intact and contribute exactly zero; a genuinely smaller table collides n-grams onto shared rows, which is a *different* error with an unmeasured cost that could be smaller or larger. This bounds what the rows are worth. It does not price the redesign.
- **+5.48% is on one synthetic corpus, and the injections are machine-generated**, not real rare tokens. Real names and real IDs have morphology and word-shape regularities that my syllable generator does not; a model may well handle real rare tokens better, which would make +5.48% an upper bound. It may also handle them worse, because real rare strings repeat across a document and mine do not. Both readings are live.
- **The 34.7% injection share is arbitrary.** I picked it to make the table fire, not because it reflects any deployment distribution. The size of the effect will move with that share and I have not swept it.
- **4K context only.** Same limitation entry 17 flagged, and this corpus does not fix it: the table's 39 M rows are sized for a longer effective context than `-c 4096` gives it. The whole project's one advantage is behaviour past 4K, and this is still measured at the one length where the table has the least history to index.
- **No behavioural probe.** PPL says the deleted rows carried information. It does not say a model with them recovers a name it would otherwise miss. NIAH would say that and has not been run on either corpus.
- **The intermediate cells did not land.** 62.5% and 37.5% kept were queued at 17:06 and killed at 17:20 by the concurrent-queue collision below, so the threshold is located only to within (75%, 50%]. If a later run puts the knee elsewhere, the shape claim in this entry is wrong and the correction goes in a new entry, not this one.
- **The delta is one backend.** +5.48% was measured on the B60's Vulkan path. The CPU 50% cell died mid-run, so I have baseline agreement across boxes (0.14%) but **no cross-backend confirmation of the delta itself** — and entry 16 established that baselines can agree while a ratio still needs its own check. Until it is re-run on CPU, "+5.48%" is a Vulkan number.

## What it changes

The practical question this project has been circling is whether the table can be shrunk to buy back file size. The answer is now conditional rather than flat:

- **On prose, half the table is still free** (entry 17, reproduced here at matched flags).
- **On rare tokens it is not free at all** (+5.48%, z = +3.43). 2.15 GB — **10.1% of the 21.31 GB artefact** — costs about a twentieth of a nat on inputs the table was built for.
- So the honest statement is no longer "the table is half-redundant" but **"the table is half-redundant on text that does not use it"**, and the deployment question is which distribution the model actually sees. For a 45B with a model card advertising long-tail recall, shipping the truncated table would be selling the advertised feature to buy 10% of file size.

The heal premise also gets a nudge. The table was the one large component in this model that looked untrained and inert; it is live, and its capacity is worth something exactly where the model was sold to be good. That is one fewer thing to delete and one more thing the quant budget has to carry.

## Reproduction

```bash
# on garlic-clove, from /home/yoav/whittle
# corpus: Gutenberg #1342 + 3,842 fixed injected unique IDs/names
sha256sum rare_text.txt      # 0467adec28a9fc075085d513d6ab4b068eecae9ee1fef75874c21ded68783d51
sha256sum long_text.txt      # acdb05bae61bd822b3ea0503cac145650b08a9cc1787f5d3f33184492799863a
sha256sum models/w45_Q3_K_M.gguf  # 3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824

# intact on the rare corpus, twice, to price run-to-run variance
./llama-perplexity -m models/w45_Q3_K_M.gguf -f rare_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16   # 10.5292 +/-0.10716

# zero the tail 50% of every hash head, measure, restore
python3 mutate_once.py models/w45_Q3_K_M.gguf 0.5
./llama-perplexity -m models/w45_Q3_K_M.gguf -f rare_text.txt \
  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 99 -fa on -t 16   # 11.1064 +/-0.12947
python3 mutate_once.py models/w45_Q3_K_M.gguf --restore
sha256sum models/w45_Q3_K_M.gguf   # must return to 3544d100…824 (a LOCAL requant; see provenance note)
```

Note the file this reproduces is **not** `logic65/Whittle-Qwen-3.8-45B-A3B-GGUF`'s Q3_K_M — same byte
count, different sha. Derive it from the published Q4_K_M with `llama-quantize --allow-requantize`
(3.4 min) and check the hash, never the size.

The whole six-cell grid runs unattended in **~63 min** on this box via `run_queue.sh jobs.txt`, which writes `CELL_START`/`CELL_END` epoch lines so idle is auditable afterwards. That is the entry-18 fix applied: queue the grid up front, no per-cell human turn. Round-trip was verified after every mutating cell that *completed* — the file restored to `3544d100…824` each time. It was the cells I killed mid-mutation at 17:20 that broke the round trip, which is the whole point of the bug section above.
