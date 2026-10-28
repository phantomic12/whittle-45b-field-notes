# Entry 12 — The Spark session died loading 45B bf16, and the booked window sat idle

*2026-10-28*

## What I set out to do

Two questions, one load:

1. Does a 90.3 GB bf16 checkpoint load and run on 121 GB of unified memory with room left to train? This is the gate on the whole heal plan — if the answer is no, Spark is a tuning box, not a training box, and the expert repair has to happen elsewhere.
2. What is terminal P(EOS) on the *HF* forward pass? Entries 8 and 9 measured the "never emits EOS" defect at mean terminal P(EOS) **0.1972**, with 6 of 8 probes running to the token cap. That number is the BEFORE state any training run has to move, and every measurement of it so far came through the GGUF path with `--jinja` papering over the symptom. The raw forward is where it shows.

`spark_baseline.py` (10 probes: 4 arithmetic, 2 reasoning, 2 knowledge, 2 formatting) does both: it loads bf16, records where the parameters actually landed, then generates greedily at `max_new_tokens=160` and reports terminal P(EOS) plus whether EOS was ever emitted.

## What happened

The first session's link expired (single-use, 12 h/week quota, nightly window). A fresh `/lab link` got me back in with the 85 GB checkpoint already staged at `/workspace/hf` from the earlier `snapshot_download` — that part of the platform works exactly as advertised: `/workspace` persists across sessions, `fla` 0.6.0 was still installed, `torch 2.13.0a0+9186a08b2c.nv26.07`, `transformers 5.20.2`.

The load ran unbounded:

```
[load] starting
Loading weights:   2%|▏         | 16/975 [00:14<08:47,  1.82it/s]
```

Then, at **01:09 UTC**, the whole notebook host vanished — not a Python traceback, not a CUDA OOM, the entire Cloudflare tunnel:

```
Error 1033
The host (spark.reagent-systems.com) is configured as a Cloudflare Tunnel,
and Cloudflare is currently unable to resolve it.
```

`/`, `/docs`, `/colab/lab` and `/v1/models` all returned **HTTP 530**. The status page and the inference API went down together with my notebook, and they stayed down.

## Why I think it was the load

Stated as inference, not proof — I lost the box before I could read `dmesg`:

- The failure was **the host**, not the process. A CUDA OOM in the 45B load raises `torch.OutOfMemoryError` in `base.log` and leaves the notebook up. A tunnel that cannot resolve means the machine or the tunnel daemon died.
- 121 GB unified is **not** 121 GB free to the process. `free -g` at idle showed `total 121, used 2, available 118`, and `memory.max` was `max` (no cgroup ceiling), so there was no container-level guard that would have killed just my process early — the OOM killer would have gone hunting across services. On a box that also runs the community inference server, killing the OOM killer's chosen victims is how a shared host goes dark.
- The timing matches: vetoed nothing else was running (GPU 0%, 4 W, 0 of 8 slots working), and the last thing written before the tunnel died was `Loading weights: 2%`.

The load was also structurally doomed on this box. Of the 45 B parameters, **~10 B are the n-gram table** (`per_layer_token_embd`) which the loader deliberately keeps on CPU, and the remaining ~35 B in bf16 is ~70 GB. Add the loader's habit of materialising shards, the `max_memory` default of "all of it", and a box already holding other people's services, and 118 GB available gets eaten in one allocation sweep.

**Fix applied:** explicit `max_memory={0: "78GiB", "cpu": "26GiB"}`, tunable by `GPU_CAP`/`CPU_CAP` env vars, so the load either fits or fails loudly inside the process instead of taking the host down. Both caps together are 104 GiB against 118 GiB available — the point is to leave a margin, not to maximise.

## The part that actually matters for the budget

I had a booked training window. The status page said it plainly:

> The Spark is booked for ML training. Inference is back in about 4h 5m.
> **0** of 8 slots working · GPU **0%** busy · **42°C · 4 W**

A booked window with nothing running in it. That is the failure mode nobody warns about: the booking reserves the *machine*, it does not keep your *process* alive. When the host died, the remaining booked time did not pause, warn, or email anyone — it just ticked down at 4 W. If this had been a 6-hour training run at hour 2, the run would be dead and the hours spent.

**Rule I am taking from this:** for anything longer than a smoke test on Spark, the process has to be restart-capable — a shell loop that relaunches from the last checkpoint, writing progress to `/workspace` every N steps. `/workspace` persists; my process does not. The 12 h/week cap is the scarce resource and it is spent whether or not the job is running.

## What this does and does not settle

- **Settled:** the checkpoint stages fine, `/workspace` persists, the env is real (`torch 2.13`, `transformers 5.20.2` with `qwen4_exp`, `fla` 0.6.0). The platform works.
- **Not settled:** whether 45B bf16 + the table fits with room to train. The honest answer right now is *unbounded it does not fit — it takes the host with it*, and the bounded retry has not run yet. That is the next thing I do when the tunnel comes back.
- **Unchanged:** P(EOS) on the raw forward is still unmeasured. No BEFORE number, so no training claim.

This entry is a record of a zero-information session minus one hard lesson about shared hosts. I would rather write it than quietly retry and pretend the first attempt never happened.
