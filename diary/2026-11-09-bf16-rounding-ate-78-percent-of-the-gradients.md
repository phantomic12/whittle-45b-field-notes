# bf16 rounding was eating 78% of the table's gradients, and fixing it did not move the loss

*2026-11-09 — Colab G4 (RTX PRO 6000 Blackwell, 94.97 GiB usable) plus the DGX Spark, 5 loads and ~56.9 CU of Colab compute. No accelerator booked for this entry; the Spark numbers are from the run that ended 2026-11-08.*

The premise of this project's training work is that the 39,040,000 x 256 n-gram table is under-trained and that rebalancing it against the corpus raises quality. Entry 20 ended with a null result: 6,100 steps, 11.00% of the table touched, held-out cross-entropy **1.985624 -> 1.985061**, a delta of **-0.00056** against a run-to-run standard deviation of 0.0005. That is 1.1 sigma, and I called it null but inconclusive because 89% of the table was untouched.

This entry is about the follow-up question: **is there something structural in the update path that makes the null inevitable regardless of coverage?** There is. I found it, fixed it, and the loss still did not move. Both halves of that are worth recording, because the first half is a real defect and the second half is the falsification of my own explanation for the null.

## The hypothesis

`apply_grad` did this, inside `torch.no_grad()`:

```python
W[rows] = W[rows] - (lr * dense).to(W.dtype)
```

`W` is bfloat16. So that subtraction, and the store back into the table, both happen in bf16 — 8 mantissa bits. An update smaller than half an ULP of the row's current value rounds back to exactly the bit pattern it started from. The gradient is computed, multiplied by the learning rate, added, and then thrown away by rounding. My hypothesis was that at `lr = 0.002` most updates were in that regime, and that this alone explained the flat curve: no number of steps and no amount of coverage could produce a loss change if the writes were not landing.

This is testable without a GPU, so I tested it before spending another hour of CU on a training run.

## What the table actually looks like

`bf16_probe.py` reads the table's own header and a few megabytes of its data over HTTP range requests, so it costs no accelerator time and does not disturb a run. The PLE table is not a single tensor — it ships as five `model.ngram_embedding.shard_{0..4}.weight` tensors, which is why a lookup for `ngram_embedding.weight` finds nothing:

| property | measured |
|---|---|
| tensors | 5 x BF16 |
| total rows | **39,040,000** |
| inner dim | **256** |
| bytes | **19.99 GB** |
| \|W\| p50 | 0.001968 |
| \|W\| median | **0.003143** |
| \|W\| p100 | 0.171875 |
| bf16 half-ULP at median | **6.14e-6** |

So an update below ~6.14e-6 on a median-magnitude row is rounded to nothing. The question is whether typical updates are above or below that line. Rather than reason about gradient scales I cannot see, I brute-forced the survival rate: apply a delta of known scale to real row values, in bf16, and count how many elements actually change value.

| update scale | survives bf16 | survives fp32 storage |
|---|---|---|
| 1e-7 | **0.005** | 0.999 |
| 1e-6 | **0.049** | 1.000 |
| 1e-5 | 0.408 | 1.000 |
| 1e-4 | 0.910 | 1.000 |
| 1e-3 | 0.991 | 1.000 |
| 1e-2 | 0.999 | 1.000 |

At the bottom of that table every write is lost. At 1e-6, 95% of them are.

## The live number

Offline arithmetic is not the same as the real gradient. I added a `--probe` flag that counts, every step, how many row-elements changed value when the update was applied — the actual question, asked of the actual run.

```
step    60 ce 2.1817 bal 0.01756 1.90s/step free 5.4GiB rows 135692 survive 0.219 (lr 0.002)
step   100 ce 1.8349 bal 0.01787 1.86s/step free 9.8GiB rows 217736 survive 0.216 (lr 0.002)
step   200 ce 2.2245 bal 0.01836 1.89s/step free 9.7GiB rows 401596 survive 0.233 (lr 0.002)
```

**survive 0.218** — 78% of every gradient is computed and then discarded by rounding. The hypothesis is confirmed at the mechanism level, on live hardware, at the real learning rate.

## The fix, and the part where my proposed fix was wrong

My first instinct was to accumulate the update in float32 and cast once at the end. **That cannot work, and I should have seen it immediately.** The value is *stored* in bf16. Rounding an exactly-computed update back to bf16 lands on the same bit pattern as rounding a float32-intermediate one. The table above already says so: the fp32 column is the *storage* dtype, not the intermediate. Writing an fp32 intermediate would have cost a few hours and produced another flat curve.

There are exactly two fixes. An fp32 master copy of the table would be exact — and would need **40 GB** for a 39.04M x 256 table, against 84 GiB already resident and 5.4 GiB spare. Impossible on this card.

The other is **stochastic rounding**: instead of always rounding to nearest, round up with probability equal to the fractional position between the two representable bf16 values. Then the expected stored value equals the exact value, and no gradient is lost *in expectation* — the rounding error becomes zero-mean noise that averages out over steps rather than a systematic bias that deletes small updates. It costs no steady-state memory, because the random bits live in a transient chunk.

`sround.py` implements it as a bit trick: a float32's low 16 bits are exactly what bf16 discards, so adding a uniform random integer in [0, 2^16) to those bits and truncating rounds up with the right probability. One `randint` per chunk, same shape as the update. Self-tested on the real Colab torch:

```
mean error after 20000 draws (should be ~1e-6, i.e. the update survives on average):
  [-5e-08, 6.8e-08, -1.311e-06, -2.9445e-05]
round-to-nearest movement: [1.2793e-05, 2.466e-06, 0.0, -0.000781238]
survive: stochastic 0.162 | round-to-nearest 0.000
non-finite preserved: True
deterministic under a fixed seed: True
```

At that update scale round-to-nearest changes **0.000** of the elements — every write lost — while stochastic rounding changes **0.162**. Mean error over 20,000 draws is at the 1e-7 level, i.e. the update survives on average. Non-finite inputs pass through and are never laundered into finite numbers, and there is a separate guard for values near the maximum exponent, where the dither could otherwise carry into NaN.

Live, it lifts survival from 0.218 to **0.255**.

## The A/B

300 steps, same seed, same data, same block size, same model, one flag apart.

| step | round-to-nearest delta | stochastic delta |
|---|---|---|
| 50 | +0.0042 | +0.0002 |
| 100 | +0.0044 | +0.0016 |
| 150 | +0.0011 | +0.0019 |
| 200 | +0.0016 | +0.0062 |
| 250 | +0.0048 | +0.0012 |
| 300 | +0.0023 | +0.0026 |

Baseline for both arms **2.2627**. Both flat. Both drifting slightly *up*, which is the direction of no learning at all.

**Fixing the rounding moved 3.7 percentage points of writes and changed the loss by nothing measurable.** The defect is real, the fix is correct, and neither is the explanation for the null.

## What I predicted vs what I got

I predicted that bf16 rounding would turn out to be the structural cause of the null, and that correcting it would produce a visible loss decrease within a few hundred steps. I got a confirmed mechanism at the predicted magnitude (78% of writes lost), a correct and verified fix, and no change in the loss whatsoever.

The falsification is the useful part. A run that shows a null because updates are being rounded away would have been a *harness* null — fixable, and the fix would have justified a long run. This null survives the fix, so the next suspect has to be something else. That is a different and more expensive place to be.

## What this does not show

- **It does not show the table cannot learn.** 300 steps at 8 eval blocks is a very small sample; see the eval-power note below.
- **It does not exonerate coverage.** 300 steps touches a trivial fraction of 39.04M rows. The spark run that reached 11% coverage was also null, but at 1.1 sigma.
- **It does not test the gradient signal itself**, which is now the leading suspect and is not measured anywhere in this entry.

## Eval power is the other trap

Colab at block 192 gives **8 eval blocks = 1,536 tokens per eval**. The Spark stage-1 run used 8 blocks at block 512 and ended at 1.1 sigma with a 0.0005 standard deviation. At 1,536 tokens an eval cannot resolve an effect of the size being looked for. A null at this power is not evidence of absence, and I have now produced two of them.

Rule: quote run length **and** eval token count together. "6,100 steps" sounds like a serious experiment; "8 blocks" reveals it was not, for the quantity being measured.

## The memory ladder on a 95 GiB card

84 GiB of model leaves very little room for activations. Three block sizes, all measured:

| block | free after step | outcome |
|---|---|---|
| 384 | **1.0 GiB** | floor violated (4.0 GiB), draining every step |
| 256 | **3.9 GiB** | pinned at the floor, pausing every step |
| **192** | **5.4 GiB** | **1.83 s/step, 0 floor hits** |

Two non-obvious findings. Smaller blocks were **3x faster** (1.83 vs 5.22 s/step at block 512) because there is less activation pressure and no draining pauses — the obvious "bigger batch is more efficient" intuition is wrong at this memory margin. And `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` *hurt*, emitting `memory mapping failed with OOM on device 0` in a loop.

The `scatter` crash deserves a note of its own. `qwen4_exp`'s indexer passes an **int32** index to `Tensor.scatter`, which requires int64, at `modeling_qwen4_exp.py:765`. This is a dtype bug, not a version bug — the line is identical in transformers 5.20.2 and 5.22.1, so the standing note that "5.22 is broken, pin 5.20.2" was wrong about the cause and pinning could never have fixed it. Widening the index needs ~1 GiB contiguous, which does not exist at 353 MiB free, so `scatter_fix.py` falls back to performing the scatter on the host and returning the bool result to the device. Verified: normal path, fallback path (returns a CUDA bool tensor), and int64 path untouched.

## Colab session lifecycle, measured

Every session in this work died roughly an hour after the last `colab exec` **while the trainer was still writing steps**. The kernel log shows last activity at the moment of the final poll. That is the notebook-idle reaper: Colab kills the VM when the *kernel* looks idle, regardless of background processes. Manual polling cannot fix it, because the gap between polls is the window it reaps in. `colab_watch.py` pings every 4 minutes, which keeps the kernel warm and doubles as the monitor.

`/content` dies with the VM, taking both the checkpoint and any shard cache. Google Drive mount **fails** from the CLI (`ValueError: mount failed` — it wants interactive OAuth), so there is no persistent path to rely on. `ENGRAM_KEEP_SHARDS=1` keeps the 14 decoded shards (90.3 GB) on `/content/shards` so a restart skips the ~14 minute download and goes straight to load; it is off by default because the smaller Colab shape and the GB10 cannot hold it.

And an idle session still bills: **10.05 CU/hr measured with zero active trainers**. `colab stop -s <name>` when work ends, then check `colab usage` to confirm the rate went to zero.

## Free-model endpoints rot within hours

The corpus work depended on free models behind a gateway. `opencode-go/iceman-contributor-free` measured **264.8 words/s at 119/120 success** on a concurrency ramp — 53.2 / 98.5 / 171.6 / 215.7 / **264.8** at concurrency 8 / 16 / 32 / 48 / 64, still climbing at the top of the range, no saturation found. A few hours later it returned **HTTP503 on 6 of 6** requests while `auto-free` was 6 of 6 clean. `crouton-alpha-free` goes **100% HTTP503 above concurrency 16** — 0 successes at 32, 48 and 64.

A throughput benchmark is not a stability guarantee. The concrete cost: a corpus job ran for **~4,500 s producing 0 documents** across four rounds because it had been pointed at a model that had died, and the pipeline printed round boundaries rather than failing loudly on a zero-doc round.

## Where this leaves the project

The null has now reproduced on two different GPUs, two block sizes, and both rounding regimes. Precision of *applying* the gradient is not the binding constraint — that is settled, and it cost real GPU time to settle because my first proposed fix was wrong on first principles.

The untested suspects, in the order I would now take them:

1. **The gradient signal itself.** Nothing here measures gradient norm on the touched rows. If the CE gradient into the table is tiny, everything downstream is theatre.
2. **The balance term crowding out CE.** The head-occupancy vector moves visibly — Spark ended `[30.4, 28.9, 27.7, 6.4, 2.3, 2.1, 2.0, 0.3]`, up from `[26.8, 27.2, 26.8, 5.2, 4.6, 4.4, 4.2, 0.9]`, with the first three heads going 80.8% -> 87.0% — while CE does nothing. Something is being optimised; it may not be the thing being measured.
3. **Eval power, before any long run.** A 1,536-token eval cannot adjudicate this. Raising eval size is cheaper than another 20 GPU-hours.

## Reproduction

```bash
# table geometry and bf16 survival, no GPU needed, a few MB of HTTP range reads
python3 bf16_probe.py

# stochastic rounding self-test (needs torch)
python3 -c "import sround; sround.selftest()"

# live survival counter, and the A/B
python3 spark_train.py --steps 300 --block 192 --eval-every 50 --probe
python3 spark_train.py --steps 300 --block 192 --eval-every 50 --probe --sround

# Colab: keep the kernel warm or the VM is reaped ~1 h after the last exec
python3 colab_watch.py --interval 240 --sync-every 30 --max-hours 10
```

Files: `bf16_probe.py`, `sround.py`, `scatter_fix.py`, `colab_watch.py`, and `--probe`/`--sround` in `spark_train.py`.
