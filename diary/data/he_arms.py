#!/usr/bin/env python3
"""HumanEval, two arms, both models — resolves the thinking-on confound.

The first arm (in head2head.py) used thinking ON + the card sampler with a
1024-token cap. Measured consequence: 41% of Whittle's completions stopped at
exactly 1024 tokens, so it scored 51.8% against a published 81.1%. The published
number was taken thinking-OFF, greedy, so the two are not the same measurement.

This script runs the protocol the card actually used:
  arm A: thinking OFF, greedy (temp 0), 1024-token cap   <- comparable to 81.1%
  arm B: thinking ON,  card sampler, 3072-token cap      <- room to finish

Both arms, both models. Same items, same prompts.

Usage: python3 he_arms.py --base http://127.0.0.1:8091 --tag whittle
"""
import argparse
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLER = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
SYSMSG = ("Complete the following Python function. Output ONLY the code, "
          "no markdown fences, no explanation. Include the full function "
          "signature and body.")


def post(base, payload, timeout=1800):
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, prompt, cap, thinking, greedy):
    p = {"model": "m", "messages": [{"role": "user", "content": prompt}],
         "max_tokens": cap}
    if greedy:
        p.update({"temperature": 0.0, "top_k": 1})
    else:
        p.update(SAMPLER)
    p["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    d = post(base, p)
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def clean_code(txt, prompt):
    m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
    code = m.group(1) if m else txt
    code = code.replace("```python", "").replace("```", "").strip()
    if not re.search(r"^\s*def\s", code, re.M):
        code = prompt + "\n" + code
    return code


def run(base, tag, arm, workers=4):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cap, thinking, greedy = {"A": (1024, False, True),
                             "B": (3072, True, False)}[arm]
    out = f"{HERE}/he{arm}_{tag}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["task_id"] for l in open(out)}
    items = [json.loads(l) for l in open(f"{HERE}/HumanEval.jsonl")]
    lock = threading.Lock()
    fh = open(out, "a")

    def one(it):
        try:
            txt, usage = chat(base, SYSMSG + "\n\n```python\n" + it["prompt"] + "```",
                              cap, thinking, greedy)
        except Exception as e:
            print(f"[{arm} {it['task_id']}] ERR {type(e).__name__}", flush=True)
            return
        row = {"task_id": it["task_id"], "completion": clean_code(txt, it["prompt"]),
               "ctok": usage.get("completion_tokens", 0), "cap": cap}
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        print(f"[{arm} {it['task_id']}] {row['ctok']}tok", flush=True)

    todo = [it for it in items if it["task_id"] not in done]
    if workers <= 1:
        for it in todo:
            one(it)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(one, it) for it in todo]):
                f.result()
    fh.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", default="AB")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for arm in a.arm:
        print(f"=== arm {arm} [{a.tag}] ===", flush=True)
        run(a.base, a.tag, arm, a.workers)
