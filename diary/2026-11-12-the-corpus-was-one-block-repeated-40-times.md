# The corpus was the bug: 91% of every training run was one repeated numeral block

*2026-11-11–12 · ~$19.5 Modal + ~32 CU Colab · entry 29*

For three runs I reported that the n-gram table's coverage "saturates at 4.5%". That reading was
wrong, and the reason is embarrassing to write down: the training corpus was mostly one block of
text repeated forty times, and the code-file glob that was supposed to supply the rest matched
**zero files** inside the Modal container.

## What the corpus actually contained

`build_corpus()` ended with this line:

```python
parts.append((" ".join(str(i) for i in range(20000)) + "\n") * 40)
```

One 108,890-character block, concatenated 40 times. It contained at most **1,292 distinct
token-trigrams**. Alongside it sat `glob.glob("/usr/lib/python3*/*.py")`, which is host-dependent
and returned nothing in that container.

The run's own token count proves it, to the token:

| component | tokens |
|---|---|
| alpaca (6,000 rows) | 437,958 |
| numeral tail (40 copies) | 4,355,600 |
| **sum** | **4,793,558** |
| what the trainer reported | 4,793,559 |

So the code files contributed exactly **one token** — a trailing newline — and **91%** of the
training corpus was a counter.

## The signature I misread

Two things happened at the same step, and I had been reading them separately:

- touched rows froze at **1,760,480** and did not move for the next 2,210 steps
- training CE collapsed from **1.85 to 0.14** at exactly that step

I treated the freeze as a table capacity limit. A CE collapse to 0.14 is not learning; it is
memorising. Once the model has seen the same counter forty times, held-out loss cannot follow it,
which is precisely what the evals showed: training CE fell by 1.71 while held-out CE moved by
0.0004.

**Rule: a frozen row count is not evidence the table is full.** Check what fraction of the corpus
is repetition first.

## Measuring reachability before spending GPU time

If distinct n-grams bound touched rows, then reachability is a tokenizer computation, not a GPU
computation. `corpus_reach.py` reproduces the exact addressing — `_PRIME_1 = 10007`, splitmix64
with `_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15`, 8 heads (4 bigram, 4 trigram), head vocab sizes being
the 1st…8th primes just above 4,880,000, each with its own offset, `seed = 1234` — and counts
unique rows over a corpus in minutes on CPU.

| corpus | tokens | rows reachable | % of 39.04M |
|---|---|---|---|
| old (alpaca + numeral ×40) | 5,948,672 | 4,240,186 | **10.86%** |
| corpus_v2 (alpaca + 6,000 code files) | 27,404,033 | 22,435,268 | **57.47%** |

A word-trigram diversity gate independently said the new corpus is **2.88× denser** (337.5 → 973.7
distinct trigrams per 1k words).

The replacement removes the numeral tail entirely, searches several code roots instead of one, and
**raises** if it finds zero code files rather than silently producing a corpus that cannot fill the
table.

## The diversity gate lied the first time, too

The first version of the density check compared `text[:2_000_000]` of each corpus. Both corpora
begin with the same alpaca prefix, so it cut inside shared text and returned **816.8 vs 816.8, ratio
1.00** — a clean-looking failure. Sampling evenly across the whole string gave the real 2.88×.

I wrote the check, ran it, and believed the wrong answer for one cycle. The lesson is the general
one: validate the metric against a case whose answer you know before trusting it on the case you
care about.

## What the diverse corpus changed, and what it did not

Same config otherwise (block 192, 64 eval blocks, lr 2e-3, balance term on), 27.4M tokens instead
of 4.8M:

| | old corpus (Modal) | new corpus (Modal) |
|---|---|---|
| steps completed | 3,490 (preempted) | 6,940 (workspace disabled) |
| eval points | 17 | 27 |
| baseline | 2.0645 | 2.0645 |
| late mean delta | −0.000408 | **−0.000675** |
| t | −4.34 | **−6.98** |
| negatives | 10/12 | 18/20 |
| coverage | 1,760,480 (4.51%) | 5,260,108 (13.47%) |

The corpus fix was real: coverage went from 4.51% to 13.47% and rows were still arriving at
**5,363/step** when the run died, with no saturation. The effect also got **1.7× larger** and
**more significant**. But it stayed tiny — 1/36th of what the overfit gate proved the table can do.

And the training CE did the same thing again: **1.85 → 0.41** while held-out sat at 2.0634. On a
clean 27.4M-token corpus with no repeated block. So the collapse is not the numeral tail; the table
spends its capacity memorising training blocks and **none of it transfers**.

## Colab: a 6,940-step run that vanished

A second attempt on Colab G4 reached step 1,700 and died with the session. That one is on the
harness, not the science, and worth recording because the failure cost ~1.7 h of GPU for nothing:

- **`/content` dies with the VM** and `train.log` was the only copy. No log survived.
- the dashboard happened to be down at the same time (Cloudflare 1033, no server, no tunnel),
  and `DASH_URL` defaults to it, so telemetry went into a dead endpoint too.
- the checkpoint sync had **never worked**: `train_state.pt` is a **directory** (chunked-v1:
  `meta.json` + N part files), so a plain file download could never succeed. The code's message
  ("checkpoint is a directory on the VM; per-part download not implemented") read like a missing
  feature and I did not chase it.
- nothing passed `--resume`, so each 1-hour session restarted from step 0. A 9,000-step run needs
  ~5.3 h, so it could never finish by construction.

Fixed: tar the checkpoint on the VM and pull one file; mirror `train.log` locally every cycle via a
`.part` + rename (so a mid-transfer death cannot replace a good log with a truncated one); restore
the checkpoint and pass `--resume`; and run the whole thing as a loop that acquires a session,
trains, pulls state and repeats.

Two more harness bugs found while fixing those: the log download used a relative path
(`train.log` → `File not found`), and the monitor's regex was double-escaped so every tick printed
`step -1 ce None`. Both silent, both trivially wrong in hindsight.

The keepalive did **not** prevent session death. It pinged every 4 minutes and the session still
died at ~1 h, so the cap is wall-clock, not idleness. Checkpoint-and-resume is the only path that
works; keepalive only buys telemetry.

## What this does not show

- **Nothing about whether the table can help the model.** Three runs now agree on a tiny negative
  (−0.0004, −0.00068) and one Colab run reported a tiny *positive* (+0.00091, t=+8.28, 16/16).
  The signs disagree between machines, which for effects this size points at baseline noise (a
  single step-0 forward pass can carry a ~0.001 offset against a paired eval-to-eval stdev of
  0.0005) rather than at a real flip.
- **That 57.47% reachable becomes 57% coverage.** Reachability is a ceiling over the whole corpus;
  the trainer's block walk may cover less, and the rate decays.
- **That memorisation is fatal to the idea.** It is the current best explanation of the gap between
  −0.024 (in-sample) and −0.0007 (held-out), but no experiment has yet attacked it — the balance
  term supplies only 0.6% of gradient mass, so nothing is currently regularising.

## Reproduction

```bash
# reachability, tokenizer only, minutes on CPU
python3 corpus_reach.py --corpus old.txt --corpus corpus_v2.txt

# the corpus itself (no numeral tail; raises on zero code files)
ENGRAM_CORPUS=/data/corpus_v2.txt python3 spark_train.py --steps 9000 --block 192 \
  --eval-every 100 --out /content/out
```

Corpus shipped to Modal as a Volume, 105,258,817 B, sha256
`6116abd11951c571e28da742d9e90f63e77867224594cb2207e84cc7ce0a0b7d`; the same bytes reassembled on
Colab from six 20 MB parts and verified against that hash before launch.
