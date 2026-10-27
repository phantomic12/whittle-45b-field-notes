# Day 4 — Serving bugs and a legitimate zero

*2026-10-25 05:52 empty-content bug, 06:46 fleet ceilings, 08:14 MiMo baseline, 09:38 GSM8K
full, 09:56 server-path ngl retune.*

## The bug that made me think the model was broken

Six creative prompts and four reasoning prompts, at the default sampling, through the
OpenAI-compatible endpoint:

| mode | creative (n=6) | reasoning (n=4) |
|---|---|---|
| thinking ON (default) | **empty 3/6**, reason-loop 3/6 | empty 1/4 |
| thinking OFF (`enable_thinking: false`) | **empty 0/6** | empty 0/4 |

Reproduced identically on **both** the 45B and the 35B, so it is the family, not one
checkpoint. Later reproduced a third time on a third machine (laptop, ngl=22) on the prompt
"Who painted the ceiling of the Sistine Chapel?" → 48 tokens, empty content. Three machines,
two checkpoints, same behaviour.

### What actually happens

The tokens are generated, but into `reasoning_content`. `content` comes back as an empty
string with `finish_reason: "length"`. The reasoning trace is a runaway syllable-counting
loop:

> *"Soft drops fall from the sky" 5? Soft(1) drops(1) fall(1) from(1) the(1) sky(1) =6.
> Need 5. "Soft drops fall from the sky" no. "Soft drops fall from the sky" 6. Need 5…*

It never exits, never emits an answer, and burns the entire budget. A haiku is the canonical
trip-wire because syllable counting is exactly the kind of self-verification this model
cannot terminate.

### The separate bug I nearly conflated with it

`/v1/completions` (no chat template) with "Write a haiku about rain." returns the same verse
**six-plus times**, verbatim:

> Rain falls from the sky, / A gentle, soft, sweet rain, / It falls on the ground. *(×6)*

So raw completion needs a repeat-penalty / `--repeat-last-n` guard. The chat path does not
show this; it shows the empty-content bug. Two different defects behind one symptom
("the model won't answer"), and fixing the wrong one wastes a day.

### Why my first loop test missed it

The initial probe used "Count from 1 to 30", "List ten animals", "Write a paragraph about
the sea" — short, evaluative tasks where the model stops cleanly. It reported **0/3 loops**
and looked fine.

The bug only appears on *creative* prompts with hard internal constraints (5-7-5 syllables).
**A loop test must include constrained creative tasks, not just counting tasks.** This is
the general lesson and it generalises past this model: your test set's difficulty profile is
a hypothesis about where the failure lives, and I had picked tasks where I already knew the
answer would be fine.

### Workaround that ships today

Send `chat_template_kwargs: {"enable_thinking": false}` for creative and short-form
requests. Verified 0/6 empty, clean output, `finish=stop`:

> **Rain's Whisper** — Soft drops fall from the sky, / A gentle rhythm on the earth, /
> Whispering secrets to the soil.

For reasoning tasks leave thinking on. Two lines in any client. It is a real fix for anyone
running this model for chat, and it is the reason every later measurement in these notes
either disables thinking or gives a 160-token budget with eyes open.

## GSM8K full: the first gate with actual power

1,319 items, greedy (temp 0, top_k 1), cap 3,072, thinking on.

| metric | value |
|---|---|
| correct | **1198 / 1319 = 90.83%** |
| 95% CI (Wilson) | [89.15%, 92.27%] |
| wrong | 87 |
| ran out of tokens | 34 (2.6%) |
| wall | 369.6 min (6.2 h) |
| median tokens | 249 |
| p90 tokens | 705 |

A heal that moves accuracy by **2 points is now detectable** (CI width ±1.6). The previous
GSM-50 gate could not see anything under ~8 points, which is why the earlier 42-vs-44
comparison was decorative.

Two things the mean hides, both of which change how the number should be quoted:

1. **The 34 cap hits are a distinct failure mode**, not wrong answers. They are 2.6% of the
   set, i.e. ~2.6 points of headroom that a different `max_tokens` would move. Any
   comparison must use the same cap or it is measuring the cap.
2. **Median 249 vs p90 705 tokens.** Heavily right-skewed; a handful of items eat thousands
   of tokens (max 3,072, at the cap). This is why the run took 6.2 h rather than the ~2 h a
   median-based estimate promised. Estimate cost from the p90, not the median.

**Greedy vs sampled matters more than expected.** An earlier 50-item run at default sampling
gave 42/50 (84%); greedy on the full set gives 90.8%. Sampling noise alone accounts for ~6
points on a 50-item set. Never judge this model on a sampled small sample.

## MiMo agentic baseline: 0/22

Real SWE verifiers, no proxy metric. Each instance is a Docker image with a `test_patch` and
a `test_command` that either passes or does not.

| outcome | n (of 22) |
|---|---|
| `no_diff` — no unified diff in the reply | **14** |
| `apply_fail` — diff present, `git apply` rejects it | 7 |
| `fail` — applied, tests still red | 1 |
| `pass` | **0** |

Solve rate 0/22 = 0.0%. Only 8/22 (36%) produced anything diff-shaped at all.

**This is a legitimate zero, not a broken harness.** The one instance that applied cleanly
still failed its tests, which proves the verifier path works end to end — including the
grading step that can say "no". A harness that cannot fail cannot certify a pass either.

The bottleneck is upstream: the model answers with prose, or with a diff that does not match
the container's file state.

### Caveat on scope, which I want quoted with the number

These are the MiMo `code` split's `format-code-*` instances, which skew to C++/CMake build
tasks with 30-minute verifier timeouts (56 min wall for 12). A model with no
code-specialisation was never expected to do well. The point of the number is that it
**exists now**, measured on real tests rather than a multiple-choice proxy.

Do not quote 0/22 as "this model cannot code". Quote it as "0/22 on 22 SWE-style repo tasks
with a 3k-token answer budget and thinking disabled".

### The harness bug worth remembering

One HTTP 400 (prompt longer than the 8k context) killed the **entire batch**, because the
model call was unguarded. Any long-running eval must catch per-instance request errors and
record them as an outcome. Fixed with a `model_error` status, problem statements truncated
to 4,000 chars, repo listings to 1,200.

This is the same failure class as the empty-content bug and the NIAH bugs in entry 6: the
harness's error path is where false negatives live, and it is the part nobody tests.

## Fleet notes: VRAM ceilings are model-size dependent

| box | CPU | GPU | 45B | 35B |
|---|---|---|---|---|
| laptop (Zephyrus) | Ryzen 9 7945HX, 32T | RTX 4080 12 GB | **27.4 t/s** (ngl=20) | — |
| remediate (100.113.95.43) | i7-12700T, 20T | T1000 8 GB | — | 7.2 t/s (ngl=18, t=8) |
| remediate-2 (100.119.50.61) | i7-12700T, 20T | T1000 8 GB | **7.5 t/s** (ngl=14, t=8) | — |

**Do not copy ngl between model sizes.**
- 45B Q4 (27.1 GB): ngl=14 max, 16 OOMs. 5.19 t/s CPU-only → 7.49 at ngl=14.
- 35B Q4 (21.3 GB): ngl=18 max, 24 OOMs. 4.60 CPU-only → 6.68 at ngl=18.

Two config bugs on the first T1000 box, both of which produced plausible-but-wrong numbers:

1. **Installed the CPU-only build.** `llama-b11869-bin-ubuntu-x64.tar.gz` — `-ngl 4` was
   silently a no-op and `llama-bench` reported `backend: CPU` for every run. The T1000 sat
   idle the whole time. Fix: the CUDA tarball plus the matching `cudart-` tarball, **both**
   on `LD_LIBRARY_PATH`. Verify with `llama-bench --list-devices`; if it says `(none)` you
   have the wrong build.
2. **Thread count was the worst possible value.** With layers on the GPU, fewer CPU threads
   win, because the CPU has less memory-bound work:

   | threads | gen t/s (ngl=18, tg96) |
   |---|---|
   | **8** | **7.23** |
   | 10 | 7.08 |
   | 12 | 6.88 |
   | 14 | 6.52 |
   | 16 | 6.35 |
   | 18 | 6.26 |
   | 20 | 6.10 |

   Monotonic — and the original `-t 18` was the worst of the seven. CPU-only, the ordering
   *reverses*: t=18 beat t=12, and t=24 thrashed hyperthreads. **The thread optimum moves
   with ngl and must be re-swept whenever ngl changes.**

Net 4.8 → 6.1–7.2 t/s, and still slower than the laptop. That is not a config problem, it is
the silicon: an i7-12700T (35 W, AVX2 only, no AVX-512) with a T1000 (Pascal-class,
128 GB/s). A 21 GB model cannot fit in 8 GB of VRAM, so ~55% of layers are on a slow CPU no
matter what ngl is. **Rule: 45B on the laptop, 35B wherever it fits; a T1000 box is a
grading/Docker box, not a throughput machine.**

## The retune that contradicted my own earlier note

I had recorded ngl=20 as the laptop ceiling. That came from `llama-bench`. Through
`llama-server` with 128-token greedy generations:

| ngl | server t/s |
|---|---|
| 18 | 24.27 |
| 20 | 26.03 |
| **22** | **27.41** |
| 24 | OOM (`cudaMalloc failed`) |
| 26+ | OOM |

`llama-bench` does not include chat-template and tokenisation overhead, so it under-counts
exactly the cost that the extra two layers relieve. Through the actual serving path, 22 is
both faster and still fits. Laptop server now runs ngl=22, `-c 16384`, k=8: **26.6 t/s** on a
real 128-token generation.

**Rule: tune ngl through `llama-server` with a real HTTP request, not `llama-bench`.**

## Where this left the project

By the end of this day the model had: a working 27.4 t/s laptop deployment, a documented
serving defect with a two-line workaround, a 1,319-item arithmetic gate with real power, and
one honest 0/22 on agentic coding. What it did not have was a size that fits anything
smaller than a 24 GB card. That is entry 5.
