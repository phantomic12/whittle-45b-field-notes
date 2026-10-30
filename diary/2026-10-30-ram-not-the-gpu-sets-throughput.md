# RAM, not the GPU, sets throughput on this file: 15 GB instead of 31 GB on the same laptop costs 8.7× prompt speed

*2026-10-30 — $0, no accelerator booked. Five hosts benched once (llama-bench, `-r 2`), then ~40 control runs on the two 3060 boxes: cache-hold sweep (14 cells), disk probes, wall-clock I/O accounting. About 4 h of wall clock.*

I benched Clover-35B-A3B Q4_K_M on every machine I have, expecting a clean GPU ladder: 4080 laptop > 3060 laptop > T1000, with `--n-cpu-moe 99` holding the routed experts in system RAM as the common mode. The ladder came out, but an **11.8×** outlier appeared inside the 3060 rung. Two boxes have the identical CPU (i9-12900H, 14C/20T, both reading a 200 W RAPL PL1), the identical GPU (RTX 3060 Laptop, 6144 MiB), the identical driver (615.78.08) and the identical binary (`4838ffb0c`, portable x86-64-v3, sm_75;86;89). They differ in RAM — 31 GB vs 15 GB — and in throughput by **8.7×** on means.

The file is 21.42 GiB. That is the whole story: one box can hold it in page cache, the other cannot, and `--n-cpu-moe 99` does not mean "streams from disk", it means "streams from disk *only if it isn't cached*".

## The fleet, one cell each

`llama-bench -m Q4_K_M.gguf -ngl 99 --n-cpu-moe 99 -fa 1 -p <pp> -n 32 -r 2 -o md`. Weights 21.41 GiB, 34.66 B params, backend CUDA on all five rows.

| host | IP | CPU | GPU (VRAM) | RAM | pp256 (t/s) | pp2048 (t/s) | tg32 (t/s) |
|---|---|---|---|---|---|---|---|
| laptop-nvidia | 50.89 | Ryzen 9 7945HX, 32T | RTX 4080 Laptop (12282 MiB) | 30 GB | **231.33 ± 1.25** | **384.91 ± 3.66** | **40.67 ± 0.31** |
| localhost | 50.77 | i9-12900H, 20T | RTX 3060 Laptop (6144 MiB) | **31 GB** | **193.65 ± 0.97** | **318.17 ± 2.45** | **34.92 ± 0.14** |
| DESKTOP-JKJQB5R | 50.119 | i9-12900H, 20T | RTX 3060 Laptop (6144 MiB) | **15 GB** | **16.42 ± 11.05** | **28.52 ± 16.50** | **13.27 ± 1.45** |
| remediate-2 | 50.111 | i7-12700T, 20T | T1000 (8192 MiB) | 31 GB | 33.98 ± 0.38 | 51.94 ± 0.59 | 10.30 ± 0.36 |
| remediate | 50.142 | i7-12700T, 20T | T1000 (8192 MiB) | 31 GB | 32.26 ± 0.29 | 51.42 ± 0.12 | 9.70 ± 0.14 |

Two sanity checks fall out for free. The T1000 twins agree to **5.3%** on pp256, **1.0%** on pp2048 and **6.2%** on tg32 — that is my repeatability floor for a single `-r 2` cell, and it is why the 8.7× gap is not noise. The 4080 beats the 3060 by **+21.0%** on pp2048 and **+16.5%** on tg32, and `--n-cpu-moe 28` (28 expert layers resident in 12 GB VRAM) buys a further **+29.9%** and **+15.9%** on the 4080 only. The ladder is real and it is small.

The DESKTOP error bars, ±67.3% and ±57.9%, are the box telling me it is not doing the same work per run. That is the finding, not a measurement defect to average away.

## What is actually happening

Same-model, same-driver pair, repeated. pp256 on 50.77: seven runs, mean **195.2 ± 0.7** (0.4% CV). pp256 on 50.119: five runs, 22.47 / 18.77 / 19.85 / 23.85 / 27.64, mean **22.5 ± 3.5** (15.6% CV). Ratio of means **8.67×**; best-against-worst **10.4×**.

Page-residency of the `.gguf` (`fincore --bytes`) before the run:

| box | file resident | % of 21.42 GiB |
|---|---|---|
| 50.77 (31 GB) | 21.42 GiB | 100.0% |
| 50.119 (15 GB) | 13.5–13.9 GiB | 63–65% |

So 50.119 is missing ~7.6 GiB of the model, on every pass, forever. It reads it back off NVMe. Measured over the whole `llama-bench` process (load included), from `/sys/block/nvme0n1/stat` sector counts:

| run | wall | device read | after subtracting the 21.42 GiB one-time load | effective rate |
|---|---|---|---|---|
| 50.119 pp256 | 67.41 s | 32.5 GiB | **11.1 GiB = 0.52× the model** | 0.48 GiB/s |
| 50.119 pp2048 | 151.13 s | 44.6 GiB | **23.2 GiB = 1.08×** | 0.30 GiB/s |
| 50.119 pp2048 (repeat) | 174.58 s | 47.9 GiB | **26.5 GiB = 1.24×** | 0.27 GiB/s |
| 50.77 pp256 | 5.82 s | **0 MiB** | 0 | — |
| 50.77 pp2048 | 14.37 s | **0 MiB** | 0 | — |

The box reads the entire model back off disk for every 2048-token prefill. It does so at **0.27–0.30 GiB/s** against a sequential `dd iflag=direct` rate of **1.9 GB/s** on the same device — a **7.4×** penalty, because reclamation hands back scattered 4 KiB pages in address order rather than contiguous extents. Wall clock on pp2048 is **10.5–12.2×** the cached box; on pp256, **11.6×**.

Decode survives better than prefill — 2.5 GiB of reads over a 32-token generation, **13.27 vs 34.92 t/s = 2.6×** — because one token touches only 8 of 256 experts, so the working set per step is smaller than a full pass over the experts.

## The control I needed, and the part of my own story it killed

The correlation is clean, but "correlated with RAM" is not "caused by page cache". The cheap test: take the box that *fits* the model and force the kernel to evict it, by holding N GiB of touched anonymous memory while `llama-bench` runs. If the gap is the cache, the fast box should collapse to the slow box's number. One sweep, two reps per step, on 50.77:

| hold | resident before | pp256 | tg32 |
|---|---|---|---|
| 0 GiB | 21.42 GiB | 194.33 / 195.63 | 34.50 / 34.38 |
| 10 GiB | 19.54 GiB | 167.12 / **67.20** | 33.67 / 26.26 |
| 14 GiB | 15.41 / 18.31 GiB | **20.21** / 76.03 | 17.83 / 24.33 |
| 16 GiB | 13.39 / 13.52 GiB | **146.06 / 44.70** | 33.44 / 18.66 |
| 18 GiB | 11.61 / 11.83 GiB | 32.76 / 30.00 | 14.42 / 15.79 |
| 20 GiB | 9.59 / 9.49 GiB | 9.86 / 24.09 | 9.96 / 11.40 |
| 22 GiB | 11.46 / 7.45 GiB | 5.16 / 18.34 | 3.96 / 6.20 |

The collapse is real: at 14 GiB held, the fast box reads **20.21 t/s**, inside the slow box's distribution. At 22 GiB held it reads 5.16. Evicting the cache reproduces the symptom on demand, which is the causal test.

**And the sweep is not monotone, which is the half of this I have to report against myself.** Two reps at the same hold disagree by up to **3.8×** (14 GiB: 20.21 vs 76.03; 16 GiB: 146.06 vs 44.70), and 13.39 GiB resident read **146.06** while 15.41 GiB resident read **20.21**. A log-linear fit of pp256 on resident GiB gives R² = 0.61 — most of the variance is unexplained by residency alone. What I believe is happening: residency measured *before* the run is not residency *during* it, because the kernel keeps reclaiming while the bench runs, and my 10 s settle after starting the holder is not enough to reach steady state. The honest statement is **the cliff sits between ~63% and ~91% of the file resident and its position within that band is not reproducible run to run** — not "pp ≈ 20.8% × per resident GiB", which is what the fit would let me write. The mechanism is established; the dose–response curve is not, and I am not going to publish the fit.

## What this does not show

- **It does not show RAM is the only factor.** The two 3060 boxes also have different NVMe (WDC PC SN540 vs Micron 2450) — and the *slower* box's disk is the **faster** one, 1.9 GB/s vs 1.3 GB/s by sequential direct read. So the 8.7× gap runs *against* the storage difference and cannot be explained by it. But 15 GB vs 31 GB is not the only asymmetry I did not control; the machines are nominally the same model and I have no proof they are.
- **It does not show `--n-cpu-moe 99` is a bad setting.** Every box here used it. The alternative, `--n-cpu-moe 28`, was only measurable on the 12 GB 4080, where it is worth **+29.9%** on pp2048. What the 15 GB result shows is that *unbacked* RAM residency is the thing to check before trusting any `ncmoe` number, not that the flag is wrong.
- **It does not touch perplexity.** These are throughput cells, `-p`/`-n` only, no `-f`. Nothing here says the model computes differently when it streams — llama.cpp's mmap path should return identical bytes regardless of where they came from, and I did not verify that it does.
- **No quality or long-context probe.** Nothing about the model's behaviour, only about how fast the box feeds it.
- **The 8.7× is one cell type.** pp256 is the harshest case: a 256-token prompt over a 21 GiB file is the most page-fault-dense workload in the suite. At pp2048 the ratio is 11.2× on means but the slow box's own CV is 50%+, so quote the wall clock (10.5–12.2×) and not the t/s.

## Two harness notes, because they cost me time

`clover_bench.sh` prints a `build=` line hard-coded to `4838ffb0c` rather than reading it from the binary. All five logs say the same build. I verified it from `--version` on one box and from `llama-bench -o csv`'s own `build_commit` field on another, but the printout is an assertion, not an observation, and should be treated as such if these logs are ever cited.

`llama-bench`'s `avg_ts` excludes model load. The first version of this entry divided bytes read by `avg_ts` and got bandwidths that closed suspiciously well; that arithmetic is invalid because the I/O happens partly during the load that `avg_ts` does not count. Everything in the table above is wall clock around the whole process, with the one-time 21.42 GiB load subtracted explicitly. The "effective rate" column is therefore a lower bound on the streaming rate and an honest over-estimate of nothing.

## Reproduction

```bash
# the fleet cell, identical on every host
export LD_LIBRARY_PATH=$HOME/clover/lib
$HOME/clover/llama-bench -m $HOME/models/clover/Q4_K_M.gguf \
  -ngl 99 --n-cpu-moe 99 -fa 1 -p 2048 -n 32 -r 2 -o md

# is the model cached? (this is the whole diagnostic)
fincore --bytes $HOME/models/clover/Q4_K_M.gguf     # want 22998011904

# is it reading the disk?
awk '{print $3}' /sys/block/nvme0n1/stat            # sectors, x512

# force the eviction that proves causality
~/clover_cache_ctl.sh 14    # holds 14 GiB of touched anon memory during one run
~/clover_cache_sweep.sh     # the 14-cell sweep; NOT monotone, see above
```

Scripts: `clover_cache_ctl.sh`, `clover_cache_sweep.sh`, `clover_disk_probe.sh`, `clover_disk_split.sh`, `clover_io_clock.sh` — all in the lab tree and reproduced verbatim above in behaviour.

For the record, the file was verified identical on all five hosts by hashing its first 64 MiB (`8541a7a9ad9981748efabb57e5b8a280`) and by `stat` size (`22998009248`), so this is not two different artefacts.
