# Protecting the SSM and gate tensors at q8_0 costs 245 MiB and buys +0.25% perplexity, well inside noise

*2026-10-29 — $0, no accelerator booked. One quantise pair on remediate-2 (~75 s each) plus 2 CPU cells at `--chunks 24`, 76 min each. About 3 h of wall clock.*

The stated reason to prefer a higher-bit quant is that a few tensors are fragile. For hybrid architectures the usual claim names the same short list every time: `ssm_out`, `attn_gate`, `ssm_alpha`, `ssm_beta` — the small always-on tensors that a per-tensor KLD sweep tends to flag. llama.cpp at `ff0d396e` leaves them at the ftype default (q4_K inside Q4_K_M), so a "protected" build is one line of `--tensor-type` overrides.

I believed that enough to build it. On this model it is not worth the bytes, and the experiment that shows so also shows why the belief is plausible and wrong.

## What was built

Both files start from the same source and both use `--allow-requantize`, because llama.cpp refuses to change a tensor's type on an already-quantised input:

| file | recipe | bytes | sha256 |
|---|---|---|---|
| `w45_Q4_K_M.gguf` (source) | hub Q4_K_M | 27,096,838,464 | `f483b9dd…4c5e` |
| `w45_Q4KM_ctl.gguf` (A) | Q4_K_M → Q4_K_M, no overrides | 27,096,838,464 | `f483b9dd…4c5e` |
| `w45_Q4KM_prot.gguf` (B) | same + `ssm_out=q8_0 attn_gate=q8_0 ssm_alpha=f16 ssm_beta=f16` | 27,354,149,184 | `da3f8e31…8c0d` |

The control is **byte-identical to its own source**, same hash. Requantising Q4_K_M to Q4_K_M with `--allow-requantize` is a no-op on this file, so there is no requantisation confound to subtract — A is the source, and B−A is the protection alone.

B is **+257,310,720 B = +245.4 MiB = +0.95%** of the artefact.

The first attempt to build a protected file failed and is worth one line: quantising *from* Q3_K_M aborts at tensor 2/1022 with `requantizing from type q3_K is disabled`, because q3_K→q4_K is a type change the flag did not cover. The pair above was rebuilt from Q4_K_M instead.

## The measurement

Same box, same corpus, same flags, `.gguf` mutated only by the quantiser:

```
corpus: long_text.txt, 607,408 B, sha256 acdb05ba…863a  (the file every entry from 5 onward uses)
flags:  -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20   (remediate-2, CPU, 20 threads)
```

| cell | PPL | ± | Δ PPL | Δ % | Δ nats | z |
|---|---|---|---|---|---|---|
| A (control = source) | 7.6821 | 0.07722 | — | — | — | — |
| B (protected, +245.4 MiB) | 7.7016 | 0.07745 | +0.0195 | **+0.254%** | +0.0025 | **+0.18** |

**z = +0.18.** The protected build is 0.25% *worse* and the error bar is four times the effect. If the protection did anything to this model — either direction — this measurement cannot see it, and the honest reading is "no effect detected", not "0.25% worse".

The wall clock moved by a comparable amount, +1.24%, and for the same reason: 4524 s for A (188.5 s/pass) vs 4580 s for B (190.8 s/pass). That is not a real slowdown from q8_0 on the SSM tensors; it is the same slop that produced the +0.25%, and I am not going to report it as a throughput cost either.

## Why the belief is plausible and why it fails here

The per-tensor-fragility story comes from models where a small tensor carries a disproportionate share of the computation. It is a reasonable prior and I would still use it on a dense hybrid. It fails on this model for a reason the anatomy in entry 1 supplies: what is fragile here is not a 4096×2048 gate, it is 9.56 B of spliced experts and a 9.99 B hash table. Those are 43% of the parameters. Protecting 245 MiB of small tensors addresses the parts of the file that were never the problem, and leaves the two components that actually lose information at the ftype default.

Entry 13 already measured the size of the correct lever: ablating all 76 spliced experts costs **+23.11%** on Q4_K at these same flags. The protection is worth 0.25% of that, and 0.25% is not distinguishable from zero.

## What I predicted vs what I got

I predicted the protection would be **worth roughly its byte cost** — that +0.95% file size should buy something on the order of a few tenths of a percent of perplexity, on the reasoning that the KLD sweeps that nominate these tensors do so because they cost more than their share.

I got a number smaller than the error bar in the wrong direction. Either the KLD result does not transfer to this architecture (plausible: those sweeps are run on standard hybrids, and this model's always-on weights are 6.5% of parameters), or the effect exists and is below 0.3% on this corpus. I cannot distinguish those with one corpus, and I am not going to report a null as proof of absence.

What I got right cheaply: building the control as a *requantise-to-same-type* rather than comparing against the hub file. That turned out to be even cleaner than designed — byte-identical — but the design decision is what made it checkable rather than assumed.

## What this does not show

- **It does not show per-tensor protection is useless in general.** One model, one architecture family, one recipe (q8_0/f16 on four tensor classes), one corpus. A different protected set, or q6_K instead of q8_0, is a different experiment.
- **It does not show the tensors are unimportant.** They are always-on and small; "protecting them from q4_K changes nothing measurable at 4K prose perplexity" is much narrower than "they do not matter".
- **It does not test the quantisation the project actually ships.** This is Q4_K_M-derived; Q3_K_M is 5.8 GB smaller and its q3_K blocks are where fragility would be expected first. Testing protection on the Q3 build is the cell this entry does *not* contain — the build aborted, as noted above, and the retry is a different recipe.
- **No behavioural or long-context probe.** PPL at 4K on one Gutenberg file. The claim the model card makes is long-tail recall, and nothing here touches it.
- **The 0.95% byte cost is real even though the perplexity effect is not.** That asymmetry is the whole finding: I can pay the bytes, but there is no measured reason to.

## Reproduction

```bash
# on remediate-2, from /home/yoav
sha256sum w45_Q4_K_M.gguf     # f483b9dd…4c5e
bash two_build.sh             # writes w45_Q4KM_ctl.gguf and w45_Q4KM_prot.gguf
sha256sum w45_Q4KM_ctl.gguf   # f483b9dd…4c5e — identical to source, requant is a no-op
sha256sum w45_Q4KM_prot.gguf  # da3f8e31…8c0d

for F in w45_Q4KM_ctl.gguf w45_Q4KM_prot.gguf; do
  CUDA_VISIBLE_DEVICES="" ./llama-perplexity -m $F -f long_text.txt \
    -c 4096 -b 512 -ub 128 --chunks 24 -ngl 0 -t 20 | grep "Final estimate"
done
# A 7.6821 +/-0.07722 | B 7.7016 +/-0.07745
```

Each pass is ~190 s on 20 threads; a cell is ~76 min. Nothing is mutated, so there is no
restore to verify — both files are static artefacts.

The A baseline reproduces entry 13's Q4_K CPU cell (7.6957 ±0.0783) to **−0.18%**, on a
different day, from a fresh requantisation. That is a harness check, not a result.
