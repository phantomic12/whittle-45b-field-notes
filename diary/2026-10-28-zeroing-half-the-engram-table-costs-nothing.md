# Deleting half the 4.29 GB n-gram table costs nothing measurable, and the curve does not rise where it should

*2026-10-28 — 5 h 08 m of CPU time on remediate-2, 5 perplexity legs*

Entry 5 killed my "the engram table is load-bearing" hypothesis with a single number: zeroing 50% of
`per_layer_token_embd` moved PPL by +0.06 against a ±0.69 error bar. One fraction, one cell, and the
error bar was wide enough that the result could have been hiding almost anything. This entry is the
full curve, five fractions, one baseline, all at matched flags on one box.

The answer is the same as entry 5's, but now it is a shape rather than a point, and the shape is
stranger than "no effect".

## What was run

```
gguf: /home/yoav/w45_Q3_K_M.gguf
flags: -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0
table: 39,040,000 rows x 110 B/row = 4.29 GB in 8 heads of 4,880,000 rows
sha256 before: 3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824
```

The table in the shipped Q3_K_M is **4.29 GB** (110 B/row after 3-bit quantisation), not the 20 GB it
occupies in bf16. It is 8 hash heads of 4,880,000 rows concatenated. For each fraction I zeroed the
tail of **every head** — keeping all eight hash functions alive but shrinking each one. That is the
sizing question. Zeroing a tail of the whole tensor instead would delete whole heads, which is a
different question ("fewer hash functions") and one I have not run.

A zeroed row is addressed exactly as before and contributes exactly zero, so addressing stays intact
and only capacity changes. That is not the same as shipping a genuinely smaller table, where
different n-grams would collide onto one row. This experiment bounds what the *rows* are worth; it
does not tell you what a 2.15 GB table would score.

## The curve

| table kept | PPL | ± | Δ PPL | Δ nats | Δ % | z |
|---|---|---|---|---|---|---|
| 100% (4.29 GB) | 8.5104 | 0.08909 | — | — | — | — |
| 93.75% (4.03 GB) | 8.5110 | 0.08912 | +0.0006 | +0.00007 | +0.01% | 0.00 |
| 87.50% (3.76 GB) | 8.5059 | 0.08908 | −0.0045 | −0.00053 | −0.05% | −0.04 |
| 75.00% (3.22 GB) | 8.4962 | 0.08894 | −0.0142 | −0.00167 | −0.17% | −0.11 |
| 50.00% (2.15 GB) | 8.4769 | 0.08861 | −0.0335 | −0.00394 | **−0.39%** | −0.27 |

Every leg is 161–165 s/pass, `--chunks 24`, 20 threads. Nothing here exceeds a third of one standard
error. **Halving the table is free**: 2.15 GB of the 21.31 GB file, 10.1% of the artefact, for a
perplexity change indistinguishable from zero.

Restore was verified the way it should be — the script backs up every range before zeroing and checks
the whole-file sha256 before and after the sweep:

```
sha256 after:  3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824  OK
```

The GGUF on disk now is byte-identical to the one the sweep started with, and the backup file is
gone. That check is in the script because I have corrupted a 21 GB artefact mid-experiment before,
and re-deriving it costs a quantise run.

## The part I do not like: the sign

The curve should be flat, and near-flat it is. But read the ordering: PPL *falls* monotonically as
the table *shrinks*, through four of the five points, and the deepest ablation is the best score in
the whole sweep. The one point that breaks the monotone order (93.75% kept, 8.5110) is above
baseline by 0.0006 — which is 0.007%, i.e. nothing.

Four of the five points fall in a line: from 93.75% kept down to 50%, PPL decreases at every single
step (8.5110 → 8.5059 → 8.4962 → 8.4769), and the deepest ablation is the best score in the sweep.
Three consecutive monotone steps is not what noise usually looks like, but a 1-in-8 ordering under a
random permutation is also not evidence, and certainly not at z = −0.27. Two readings, both live:

1. **It is noise with a lucky ordering.** Three monotone steps come up p = 1/8 by chance alone. That
   is not surprising enough to build on.
2. **Rarely-used hash rows contribute a small amount of variance, not signal.** Zeroing the tail of
   each head removes the rows that almost never fire — they are populated by whatever corpus the
   table was trained on, and at 4K on Gutenberg prose the model's own hashing may simply not reach
   them. A row that fires once per few thousand tokens contributes a little noise to the gate and no
   information. Removing it would then read as a tiny improvement.

Reading 2 is testable and reading 1 is not, which is the only reason I am recording it rather than
discarding it: **if the tail rows are inert, the effect should scale with corpus diversity, not with
fraction.** A code corpus or a multilingual one should reach further into the heads than Victorian
prose does. That is the next cell to run, and it is 61 minutes on this box rather than a night.

What I will not do is write "-0.39% from halving the table" as a finding. It is a null result with a
suggestive ordering, and the entry says so.

## What I predicted vs what I got

I predicted a **small but real cost at 50%**, on the order of 1–3%, growing steeply as the table
halved again. The reasoning was that the table is 22% of the bf16 model and 9.99 B parameters do not
sit in a file doing nothing in a model whose own card advertises long-tail recall.

I got **no cost at 50%** and no cost anywhere down to 2.15 GB. Two guesses about why I was wrong:

- At 4K context with `-c 4096`, the n-gram lookup only sees a short window of history. The table's
  39 M rows are sized for a much longer effective context than the harness gives it. I may be
  measuring the table at the one context length where it cannot matter.
- Entry 5's 100%-zeroing singularity (PPL 584,413) is a *gate* pathology, not an information one:
  entry 5's own `hc_*` ablation returned PPL = 248,320, exactly the vocabulary size, i.e. a uniform
  distribution. A fully zeroed table drives the gate to a degenerate state; a mostly-zeroed table does
  not. So the failure mode is off/on, not gradual — which is consistent with a flat curve almost all
  the way down and a cliff at the very end.

The second explanation is the one I would bet on, and it means the interesting measurement is not
"how much can I delete" but "where is the cliff". Everything between 100% and 50% is the flat part.

## What this does not show

- **It does not show a 2.15 GB table would score the same.** Zeroed rows preserve addressing; a
  smaller table would collide n-grams. The collision cost is unmeasured and could easily exceed the
  capacity cost, which is zero.
- **It does not show the table is free at long context.** The whole project's one real advantage is
  behaviour past 4K; testing the table only at 4K is testing it where it is least likely to work.
  All the value in this table may be in the rows my 4K windows never index.
- **It does not show the table is untrained.** Entry 5 established that this component is 22% of the
  parameters; "not load-bearing at 4K prose perplexity" is a much narrower claim than "untrained",
  and the two have been confused once already in this project's notes.
- **One corpus, one quant.** 607,408 B Gutenberg file, Q3_K_M, CPU. Entry 16 says ship deltas across
  machines and never absolutes; here even the deltas are within noise, so the honest statement is
  "no effect detected", not "0.00% effect".

## Reproduction

```bash
# on remediate-2, from /home/yoav
python3 -u ple_sweep.py --gguf /home/yoav/w45_Q3_K_M.gguf \
  --ppl /home/yoav/cuda-llama/cuda/llama-b11869/llama-perplexity \
  --corpus /home/yoav/long_text.txt --backup /home/yoav/ple.bak \
  --cpu --fracs 0.5,0.25,0.125,0.0625
# 05:34 -> 10:41 UTC, 5 legs at ~61 min each, 161-165 s/pass, --chunks 24
# baseline 8.5104 +/- 0.08909; 50% kept 8.4769 +/- 0.08861
```

The baseline in this run reproduces entry 13 and entry 16's remediate-2 CPU baseline to the last
digit (8.5104 ±0.08909) on a separate invocation. CPU `llama-perplexity` is deterministic on this box
and the corpus is byte-identical, so that agreement is a harness check, not a coincidence worth
citing as a result.
