# Day 7 — Arc Pro B60: 10.3×, after four separate blockers

*2026-10-26 22:36.*

## The claim to test

Entry 6 established that Q3_K_M retrieves 8/8 at 16K, at **555 s per case** on 20 CPU
threads. That is a correctness result and an unusable deployment. 74 minutes for eight
questions is not a service.

`garlic-clove` (tailnet 100.67.232.8) has an **Intel Arc Pro B60** — Battlemage BMG G21,
21,414 MiB VRAM. Same model, same probe set, same 32,768 context. The question was whether
21.31 GB fits in 21.4 GB of VRAM and what it costs if it does.

It fits, and it is 10.3× the CPU rate. Both of those were less obvious than they sound.

## Result

```
d05_0    5%  correct   57s    9879-DBQX
d05_1    5%  correct   53s    5440-TNPC
d33_0   33%  correct   53s    9131-GQRM
d33_1   33%  correct   53s    5666-DNPG
d66_0   66%  correct   54s    7037-SNJC
d66_1   66%  correct   53s    4668-TKCR
d95_0   95%  correct   54s    1175-VKDS
d95_1   95%  correct   53s    5489-ZFPF
```

8/8 correct again, byte-identical answers to the CPU run. Total **431 s vs 4,440 s — 10.3×**.
Mean 54.0 s per case against 555 s.

Generation **32.4 t/s warm** (6.1 t/s on CPU), prompt evaluation **28 ms/token warm**
(180 ms cold). The warm/cold gap is the interesting operational number: the first pass pays
for whatever the driver defers, and it is a 6.4× difference in prompt eval. Benchmark this
card cold and you will conclude it is barely better than a CPU.

## It fits by partial offload, not by shrink

21.31 GB of model on 21.4 GB of VRAM, with **3,582 MiB still free afterwards**, means roughly
**17.8 GB resident and ~3.5 GB in the 31 GB of system RAM**.

That is the MoE shape worth having: the routed experts page from RAM while the dense path
stays on the GPU. It also means **no smaller quant was needed to make the card usable** — the
Q3_K_M file that already won the size comparison is the one that runs, at full 16K context,
without a Q2_K-style quality sacrifice. The quant decision in entry 5 and the hardware
decision here turn out to be the same decision.

## Four blockers, none obvious from the error message

### 1. `nvidia-smi` does not exist here

It is an Intel card. `vulkaninfo` lists `Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)` as GPU0
and llvmpipe as GPU1. The obvious diagnostic —

```
vulkaninfo --summary | head
```

— catches llvmpipe first and makes it look like software rendering is the only option
available. It is not; the real device is one line further down. I spent a while believing
the card was unusable because I read the first ten lines of the wrong tool's output.

### 2. Three llama.cpp builds that predate the architecture

`~/llama-b9307` and `~/llama-b9518` both fail with:

```
error: unknown model architecture: 'qwen4exp'
```

Fair enough — old builds. But upstream nightly **b11910** fails identically: **qwen4exp is
not merged upstream.** It lives in `lendome/llama.cpp-qwen4exp` (PR #27742). If you are
holding a Whittle GGUF and a stock llama.cpp, no amount of updating will help.

This is the same class of error as the `model_type` trap in entry 1, and the same lesson: the
architecture is not upstream in either ecosystem, and both fail in ways that blame your file.

### 3. The SYCL builds abort

`~/llama-cpp` and `~/llama.cpp`, both SYCL, die with:

```
No device of requested type available
```

even with `libsvml.so` on `LD_LIBRARY_PATH`. I did not solve this. **Vulkan is the working
path on this card**; do not chase SYCL here. Recording it as a dead end rather than a
mystery, because the Intel-adjacent instinct is to reach for SYCL first and it costs an hour.

### 4. `pkill -f llama-server` kills the SSH launcher

The pattern matches the `bash -c` line that contains it, so the launch silently never happens
and the log stays empty — which reads exactly like "the server crashed on startup". Use a
pattern that cannot match your own command line, or kill by PID.

This is the third variant of this bug in the project (`pkill -f llama-cli`, `pkill -f
gsm50.py`, now `pkill -f llama-server`). The general form: **`pkill -f` and `pgrep -f` match
the command line that invokes them**, so a status check can report RUNNING forever while a
kill kills its own shell. Use `pkill -x`, check for output files, and never trust a pattern
you just typed into the same shell.

## Working build

```bash
git clone --depth 1 https://github.com/lendome/llama.cpp-qwen4exp.git
cd llama.cpp-qwen4exp
cmake -B build-vk -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-vk -j 14
build-vk/bin/llama-server -m w45_Q3_K_M.gguf -ngl 99 -t 16 -c 32768 --jinja \
    --port 8081 --host 127.0.0.1
```

`--jinja` and `chat_template_kwargs: {"enable_thinking": false}` are both required for clean
short answers. Without them the model spends its token budget on reasoning and returns an
empty content field — the same defect as entry 4, on a fourth machine.

## Transport

21.31 GB reached garlic-clove via the HF relay in ~20 min total (remediate-2 → HF 13.3 MB/s
for 872 s, HF → garlic-clove ~34–52 MB/s), sha256 verified byte-identical at
`3544d1006133758118c8d8365b9a917e7e3060895012bbce643b4ef704eba824`. The direct tailnet path
measured ~2 MB/s, i.e. ~3 h.

## What this settles

The 45B at Q3_K_M is a **deployable** model, not just a defensible file:

| axis | value |
|---|---|
| size | 21.31 GB — the same as the 35B |
| long-context quality | 15.7% better than the 35B at 16K |
| retrieval at 16K | 8/8, depth-independent |
| hardware floor | one 21.4 GB card, partial offload, ~3.5 GB host RAM |
| throughput on that card | 54 s per 16K retrieval case; 32.4 t/s generation warm |
| throughput on CPU | 555 s per case; 6.1 t/s |

Everything in that table is a measurement on this model at this quant. None of it depends on
the experts being useful — which was still an open question, and is the last entry.
