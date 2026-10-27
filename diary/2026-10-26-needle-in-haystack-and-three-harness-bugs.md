# Day 6 — Needle in a haystack: 8/8, after three harness bugs produced a false negative

*2026-10-26 21:53.*

## Why perplexity was not enough

Every number in entry 5 answers "does 3-bit degrade text modelling". None of them answers
"can it still find a specific fact in a long context" — a different capability, and the one
that decides whether a 21.31 GB quant is shippable. A model can have excellent average
log-loss on a document and still fail to retrieve a single sentence from the middle of it,
because PPL rewards local fluency and retrieval needs the attention path to work.

Q3_K_M had +8.7% PPL at 4K and 96% of the long-context arc retained. Both are compatible
with "retrieval is fine" and with "retrieval is subtly broken". So: needle in a haystack.

## Method

8 probes, 2 needles at each of 4 depths (5%, 33%, 66%, 95%) in a Project Gutenberg document.
58,577-char prompts, ~13.8K tokens. Seed 20261026. Each probe plants a `KEY → value` pair at
the given depth and asks for the value only.

Scoring is exact-match on the value, and the verdict **separates a wrong value from no
answer at all** — a real retrieval failure versus a harness or refusal problem. Those mean
different things and pooling them is how you get a false negative that looks like a model
limitation. Run CPU-only (`CUDA_VISIBLE_DEVICES=""`, `-ngl 0`, 20 threads) on remediate-2, so
it is comparable to the perplexity runs on the same box.

## Result: 8/8, every depth

```
d05_0    5%  correct  9879-DBQX     566s
d05_1    5%  correct  5440-TNPC     554s
d33_0   33%  correct  9131-GQRM     555s
d33_1   33%  correct  5666-DNPG     553s
d66_0   66%  correct  7037-SNJC     501s
d66_1   66%  correct  4668-TKCR     564s
d95_0   95%  correct  1175-VKDS     580s
d95_1   95%  correct  5489-ZFPF     563s
```

Every answer exact. No wrong values, no refusals, no truncation failures. Mean **555 s per
case**, dominated by prompt evaluation on CPU — generation is trivial at 32 tokens.

**Depth does not matter.** 95% (needle at the very end, furthest from the query under naive
positional reasoning) is as reliable as 5%. Whatever the 30 SSM + 10 sparse-attention mix is
doing, 16K retrieval is intact.

That is the confirming check entry 3 asked for. It does not prove the SSM mechanism is the
cause — that would need an ablation of the linear layers, which I did not run — but it rules
out the most likely way the long-context PPL win could have been an artifact.

## Three harness bugs that all produced the same false negative

This is the part worth keeping. **My first NIAH run reported that the model cannot retrieve.
It was wrong three separate ways**, and each failure was individually plausible. I would have
published a "the 3-bit quant breaks retrieval" conclusion that was entirely my own code.

### 1. The raw prompt had no chat template

```
llama-cli -st -p "<doc>...Answer:"
```

That does not apply the chat template. The text goes in as a **raw completion prompt**, so
the model has no instruction to answer — it simply continues the document, and it emitted
book front-matter. My scorer saw a non-answer and recorded `no_answer`.

This is the single most instructive bug in the project. A completion model given a document
and then "Answer:" will do the most probable thing, which is to keep writing the document.
Nothing in the output says "I was not asked a question". The fix is `llama-server --jinja`
and a real chat request.

**Generalisation: a benchmark that scores "no answer" as "cannot answer" is measuring the
prompt template, not the model.** Hence the separate `wrong_value` verdict — a wrong value is
at least evidence the model tried.

### 2. `n_predict` is not a valid field on the OpenAI endpoint

It needs `max_tokens`. Every request 400'd. And then:

### 3. The error handler shadowed `body`

```python
body = e.read().decode()      # rebinds the request payload to a string
```

so every retry after the first failed for a second, unrelated reason — and the retry loop's
bare `except` recorded those as generic errors rather than surfacing the 400. Three bugs
stacked: wrong parameter name, wrong error handling, and an exception handler that destroyed
the state needed to diagnose either.

### 4. (related) thinking was on by default

An 8-token `max_tokens` test returned an empty string because all 8 tokens went to reasoning.
`chat_template_kwargs: {"enable_thinking": false}` gives a clean answer. Same defect as
entry 4, now in a context where it masquerades as retrieval failure.

`--jinja` and `enable_thinking: false` are both required for clean short answers. Without
them the model spends the token budget on reasoning and returns empty content.

## One design decision that was forced, and was correct anyway

The original build used a 600K-char document. No box here can hold a 21.3 GB model plus a
150K-token cache on CPU, so it was abandoned. Testing 16K matches both the PPL measurement
and the hardware that can actually serve the model — a 32K test would have been a nicer
number and a less honest one, because nobody is going to run this quant at 150K on CPU.

## What this changes

The ladder from entry 5 was: Q4_K 27.10 GB (7.4340) · Q3_K_M 21.31 GB (8.0817, +8.7%) ·
Q2_K 16.44 GB (10.4041, +40.0%). On perplexity alone Q3_K_M looked like a defensible trade
and Q2_K was rejected.

The needle result makes Q3_K_M a **different class of recommendation**, not merely a cheaper
one: its +8.7% cost is a *text-modelling* cost that does not touch long-context factual
retrieval. Combined with —

- 15.7% better PPL than the 35B at equal package size (21.31 vs 21.28 GB) at 16K;
- 96% of the 4K→16K arc retained under matched `chunks=1`;
- 8/8 retrieval at 16K across all four depths;

— the 21.31 GB file is the shipping artefact, not a compromise. sha256
`3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824`.

Reproduce:

```bash
export CUDA_VISIBLE_DEVICES=""          # -ngl 0 alone still allocates a CUDA buffer
llama-server -m w45_Q3_K_M.gguf -ngl 0 -t 20 -c 32768 --jinja --port 8080
python3 niah_run2.py --set niah16k.json --tag niah45v3
```

## Transport note, because it cost an afternoon

Moving 21.31 GB between boxes: the tailnet between remediate-2 and garlic-clove measured
**~2 MB/s**, i.e. ~3 h for the file, and saturating a tailnet is antisocial. The HF
round-trip touches the tailnet not at all: remediate-2 → HF at **13.3 MB/s** (872 s),
HF → garlic-clove at **~34–52 MB/s**. ~20 minutes total, and it needs only a write-scoped HF
token. Syncthing over the open internet would work but adds install and hole-punching for no
throughput gain over the CDN round-trip.

That route is what made entry 7 possible the same evening.
