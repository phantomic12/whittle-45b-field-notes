# Q4_K is more damaged than Q3_K_M, not 4.9× less — entry 10 was wrong

*2026-10-27. ~25 minutes CPU on remediate-2. Zero GPU, zero dollars. Written one hour after
the entry it retracts.*

Entry 10 concluded that the same ablation damages Q3_K_M **4.9×** more than Q4_K, and I
called that "matched flags". It was not matched. I compared two numbers measured on
**different hardware**:

| number | source | hardware | chunks |
|---|---|---|---|
| Q4_K 6.9466 → 7.2144 | `LAPTOP.md` | **ngl=20, RTX 4080 laptop** | 24 |
| Q3_K_M 8.0817 → 9.3275 | new run | **CPU, ngl=0, remediate-2** | 2 |

I had spent the whole entry arguing that `--chunks` must match, then compared a GPU run
against a CPU run and shipped the ratio. The tell was already in the file: `LAPTOP.md`'s Q4_K
baseline is **6.9466** and the same file on remediate-2 at `--chunks 2` is **7.4340** —
**6.6% apart**, which is *larger than the ablation effect I was measuring*. A 6.6% offset
should have made a 4.9× ratio impossible to believe before I wrote it down.

## The missing cell

The grid needed four cells; I had three, and the fourth was Q4_K at `--chunks 2` on a CPU
box. Filling it:

| quant | chunks | baseline | ablated | ΔPPL | Δ% | Δnats |
|---|---|---|---|---|---|---|
| **Q4_K** | **2** | 7.4340 ±0.276 | **8.7303 ±0.333** | +1.2963 | **+17.44%** | +0.1607 |
| Q3_K_M | 2 | 8.0817 ±0.307 | 9.3275 ±0.363 | +1.2458 | +15.42% | +0.1434 |
| Q3_K_M | 24 | 8.5104 ±0.089 | 10.1622 ±0.110 | +1.6518 | +19.41% | +0.1774 |
| Q4_K | 24 | *running — see below* | | | | |

**At genuinely matched settings the 4-bit file is slightly MORE damaged, not 4.9× less:
ratio 0.88× in percent, 0.89× in nats.** Both land at ~+16%, within each other's error bars
(z = 3.00 and 2.62; CIs [+0.45, +2.14] and [+0.31, +2.18] overlap heavily).

The correct conclusion is the dull one: **quantisation level does not meaningfully change how
much the spliced experts are worth.** 4-bit and 3-bit both lose ~16% perplexity when you
delete them. There is no "3-bit is brittle" story, and there is no "4-bit tolerates it"
story either.

## What this does to the decision, which is nothing

Worth stating plainly because it looks like a big correction and is not:

- The experts are load-bearing at **+17%** PPL on Q4_K and **+15%** on Q3_K_M. Entry 9's
  "all 76 restored experts, in all 40 layers, are load-bearing" survives intact and is now
  measured on both quants.
- The **+0.27 PPL** figure that `LAPTOP.md` used to argue these experts were "removable at no
  cost" is a **GPU-at-ngl=20** number that does not reproduce on CPU. On remediate-2 the same
  ablation costs **+1.30 PPL**. So the original "dead weight" verdict was wrong by ~5×, and
  it was wrong for a reason nobody had checked: the baseline it was measured against moves
  6.6% depending on how many layers you put on the GPU.

That reframes the whole arc. Entries 1–2 concluded the spliced experts were near-inert and
"removable at no cost"; entry 9 reversed that on perplexity; entry 10 tried to explain the
magnitude gap as quant brittleness; entry 11 says the gap was **hardware**, and the truth is
that these experts have always cost ~16% to remove on a CPU baseline. The model was never
carrying 5.79 GB of dead weight. It was carrying 5.79 GB of *inefficient* weight — the
original entry 8 reading — and four separate measurements have now failed to make it inert.

## What this does not show

- **Not** that entry 10's harness finding is wrong. The chunk-count effect is real and
  reproducible (`--chunks 2` vs `24` moves the baseline +5.30% on the same file, same box).
  Entry 10 was right about the harness and wrong about the hardware, in the same table.
- **Not** that ngl=20 numbers are invalid. They describe a GPU-served model, which is how
  people run it. They are just not interchangeable with ngl=0 numbers, and I treated them as
  if they were.
- The Q4_K `--chunks 24` cell is still running at time of writing; it affects the
  harness-sensitivity ratio for Q4_K, not the Q3-vs-Q4 comparison above (which is matched at
  `--chunks 2`).

## The rule this earned

**A ratio is only as matched as its worst-matched factor.** I checked chunk count because I
had just discovered it mattered, and did not check `ngl` because I had not just discovered
it. The check that would have caught this costs nothing: every table in this repo should
carry a hardware and an `ngl` column, not just a flags column. Entries 5–11 all state
`--chunks` and none of the earlier ones state `ngl`.

## Reproduction

```bash
# the missing cell
CUDA_VISIBLE_DEVICES="" llama-perplexity -m Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf \
    -f long_text.txt -c 4096 -b 512 -ub 128 --chunks 2 -t 20
python3 gguf_ablate.py --gguf Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf \
    --target experts:180-256 --backup abl_q4k.bak --apply
CUDA_VISIBLE_DEVICES="" llama-perplexity -m Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf \
    -f long_text.txt -c 4096 -b 512 -ub 128 --chunks 2 -t 20
python3 gguf_ablate.py --gguf Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf --backup abl_q4k.bak --restore
sha256sum Whittle-Qwen-3.8-45B-A3B-Q4_K_M.gguf   # bcf009ae...940b3
```
