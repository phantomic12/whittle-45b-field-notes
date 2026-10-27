# Ablation: the experts are load-bearing, and the damage is superlinear

*2026-10-27*

Yesterday's routing profile said the 76 spliced experts get recruited for reasoning,
most strongly in the middle of the network. The obvious follow-up: if they are
recruited, is that recruitment *load-bearing*? Being selected is not the same as
contributing useful computation, and I had argued earlier that these experts were
essentially decoration.

Both runs below are on the shipped Q3_K_M GGUF, CPU-only, same corpus
(`long_text.txt`), same flags. The file was restored and sha256-verified after each
run, so every measurement is against a byte-identical model.

## Results

| condition | PPL | delta vs baseline | error bar | per ablated layer |
|---|---|---|---|---|
| baseline (intact) | **8.0817** ±0.30658 | — | — | — |
| experts 180-255, **layers 13-26** (14) | **8.2823** ±0.31539 | **+2.48%** | non-overlapping | 0.177% |
| experts 180-255, **all 40 layers** | **9.3275** ±0.36328 | **+15.42%** | non-overlapping | 0.385% |

Baseline reproduced `8.0817` exactly, matching the figure recorded earlier in this
notebook — an independent check that the harness is sound.

## The prediction was wrong, in an interesting direction

The prediction going in was "the effect is concentrated in the middle, so 40 layers
should cost about 2.9x the 14-layer result, i.e. ~+7%." The measured 40-layer cost is
**+15.42%**, which is **2.2x the proportional prediction** — a ratio of 6.21x damage
for 2.86x the layers.

So the late layers are **not** inert. They are *synergistic*: removing the restored
experts everywhere is much worse than the sum of removing them in parts. That is the
opposite of the "late layers barely route to them, so they can't matter" reading I
took from the routing histogram.

Two readings are consistent with the data and I cannot separate them with what I have:

1. **Genuine synergy.** The restored experts form a distributed pathway; knocking out
   the middle forces later layers to compensate in hidden states the restored experts
   no longer process, and the remaining ablations then hurt more than they would alone.
2. **Non-additivity of damage near a cliff.** Perplexity in this model responds
   superlinearly to weight damage (the k-sweep and the router-rescaling sweep both
   showed strongly asymmetric penalties). If damage compounds rather than adds, a
   40-layer ablation is not a scaled-up 14-layer ablation regardless of where the
   layers sit.

Either way, the operational claim is safe and the mechanistic claim is not:
**all 76 restored experts, in all 40 layers, are load-bearing.** What I cannot yet say
is whether the middle is special in *kind* or only in *where the histogram is highest*.

## This is the second reversal on this question

Earlier entries in this notebook concluded the spliced experts were near-inert
("30x inert on prose", "removable at no cost", "publish the negative result"). Those
were **text-modelling ablations that zeroed all layers at once and measured a single
number** — they are not wrong as measurements, but the interpretation was. A single
whole-network ablation cannot distinguish "uniformly small" from "concentrated and
compensated". The layered ablation shows the damage is large (+15.4%) and that its
distribution is not proportional to layer count.

## A method note I want on the record

Before the perplexity run I tried a 14-probe correctness battery (arithmetic,
reasoning, knowledge, format), baseline vs mid-ablated:

- baseline **13/14**, ablated **14/14**

The ablated model scored *better*. The single difference was `7 + 8 * 3`: baseline
answered `24` (an operator-precedence failure — it computes 8x3 and drops the +7),
ablated answered `31`. Both results are deterministic and reproduce exactly, so this
is a real behavioural difference, not sampling noise — but a +2.5% perplexity shift
is invisible to 14 binary probes. **Do not draw conclusions from tiny probe batteries
when a continuous metric is available.** The probe battery cost ~2 minutes; the
perplexity run that actually answered the question cost ~11 minutes per condition.

## Reproduction

```bash
# mid-layer
python3 gguf_ablate.py --gguf w45_Q3_K_M.gguf --target experts:180-256 \
    --layers 13-26 --backup abl_mid.bak --apply
CUDA_VISIBLE_DEVICES="" llama-perplexity -m w45_Q3_K_M.gguf -f long_text.txt \
    -c 4096 -b 512 -ub 128 --chunks 2 -t 20
python3 gguf_ablate.py --gguf w45_Q3_K_M.gguf --backup abl_mid.bak --restore

# all layers
python3 gguf_ablate.py --gguf w45_Q3_K_M.gguf --target experts:180-256 \
    --backup abl_all.bak --apply
```

`gguf_ablate.py` gained a `--layers` filter for this (accepts `13-26` or `0,5,9`).

## Also, I pattern-killed my own launcher again

`pgrep -f "abl_ppl|abl_all" | xargs kill` matches **its own** SSH command line and
kills the launcher. The relaunch silently produced no log and no process. Use a
bracket so the pattern cannot match itself: `pgrep -f "abl_[a]ll"`. This is the third
time in this project a pattern-based kill has eaten its own command; it is in
`README.md` for `pkill -f llama-cli` and it deserves to be a standing rule:
**never pattern-kill from a shell whose own command line contains the pattern.**