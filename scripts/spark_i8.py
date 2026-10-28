"""Baseline behavioural probe for Whittle-45B on the DGX Spark, int8-weight-only.

WHY THIS EXISTS
---------------
Two bf16 attempts to load this checkpoint on the Spark were OOM-killed by the kernel
(cgroup `oom_kill 3`, peak 93.1 GB, no cgroup ceiling, no traceback) at 74% and 80% of
the 975 weight shards. The arithmetic explains it: 45.1 B params in bf16 is 90.3 GB of
weights, and loading them via safetensors mmap also leaves the 85 GB of shard files in
page cache. On a 121 GB unified pool that sum has no headroom, and the kernel chooses
the 90 GB process when it needs pages back.

So the load is done in int8 weight-only (torchao, 0.18.0 present), ~45 GB of weights.
That is also how this model would realistically be trained on this box: LoRA on an
int8 base is ordinary QLoRA, whereas a bf16 base plus optimizer state does not fit at
all.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS SCRIPT
--------------------------------------------------------
int8 weight-only is not bf16. Terminal P(EOS) measured here is NOT directly comparable
to the 0.1972 bf16 figure in entries 8/9 -- it is a separate measurement of a separate
model. Read it as "does the defect survive quantisation", not as "the bf16 number".

Run:  python3 spark_i8.py
Out:  /workspace/w/baseline_i8.json
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/hf")
sys.path.insert(0, "/workspace/w")
import torch

REPO = "logic65/Whittle-Qwen-3.8-45B-A3B"
PROBES = [
    ("arith", "What is 17 * 23? Give only the number."),
    ("arith", "If I have 12 apples, give away a third, then buy 5 more, how many do I have?"),
    ("arith", "A train travels 60 km/h for 2.5 hours. How far does it go?"),
    ("arith", "What is 15% of 240?"),
    ("reason", "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
               "How much does the ball cost?"),
    ("reason", "List three differences between TCP and UDP."),
    ("know", "What is the capital of Australia? Answer in one word."),
    ("know", "What is the chemical symbol for gold?"),
    ("fmt", "List three colors, one per line, nothing else."),
    ("fmt", "Rewrite this more formally: 'the thing didnt work so we binned it'"),
]


def mem_report(tag):
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in f
                if l.split(":")[0] in ("MemAvailable", "MemFree", "Cached")}
    print(f"[mem:{tag}] avail={info.get('MemAvailable')} free={info.get('MemFree')} "
          f"cached={info.get('Cached')}", flush=True)


def main():
    from transformers import AutoTokenizer, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from whittle_load import load_whittle

    t0 = time.time()
    print("[load] int8 weight-only, starting", flush=True)
    mem_report("pre")
    model = load_whittle(
        REPO, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: os.environ.get("GPU_CAP", "86GiB"),
                    "cpu": os.environ.get("CPU_CAP", "20GiB")},
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
    )
    tok = AutoTokenizer.from_pretrained(REPO)
    model.eval()
    print(f"[load] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    mem_report("post")

    dev = {}
    for _, p in model.named_parameters():
        dev[str(p.device)] = dev.get(str(p.device), 0) + p.numel()
    print(f"[mem] params by device (B): "
          f"{ {k: round(v/1e9, 3) for k, v in dev.items()} }", flush=True)

    results = []
    for i, (kind, prompt) in enumerate(PROBES):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        n_prompt = ids.input_ids.shape[1]
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        secs = time.time() - t1
        gen = out.sequences[0][n_prompt:]

        eos_ids = {tok.eos_token_id}
        for extra in ("<|im_end|>", "<|endoftext|>"):
            e = tok.convert_tokens_to_ids(extra)
            if isinstance(e, int) and e >= 0:
                eos_ids.add(e)
        last = out.scores[-1][0].float()
        p_eos = sum(float(last[j]) for j in eos_ids if j < last.shape[0])
        ended = any(int(t) in eos_ids for t in gen)
        txt = tok.decode(gen, skip_special_tokens=True).strip()

        results.append({"i": i, "kind": kind, "prompt": prompt, "p_eos": round(p_eos, 4),
                        "ended": bool(ended), "words": len(txt.split()),
                        "secs": round(secs, 2), "new_tokens": int(gen.shape[0]),
                        "preview": txt[:200]})
        print(f"[{kind:6s}] p_eos={p_eos:.4f} ended={ended} tok={gen.shape[0]:3d} "
              f"{secs:5.1f}s {txt[:60]!r}", flush=True)

    summ = {"n": len(results),
            "ended_frac": sum(r["ended"] for r in results) / len(results),
            "mean_p_eos": round(sum(r["p_eos"] for r in results) / len(results), 4),
            "mean_words": round(sum(r["words"] for r in results) / len(results), 1),
            "mean_secs": round(sum(r["secs"] for r in results) / len(results), 2)}
    with open("/workspace/w/baseline_i8.json", "w") as f:
        json.dump({"quant": "int8-weight-only (torchao)", "load_min":
                   round((time.time() - t0) / 60, 1), "probe_summary": summ,
                   "probes": results}, f, indent=1)
    print("\n<<<BASELINE_BEGIN>>>")
    print(json.dumps(summ, indent=1))
    print("<<<BASELINE_END>>>")


if __name__ == "__main__":
    main()
