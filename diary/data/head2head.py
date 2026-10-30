#!/usr/bin/env python3
"""Matched head-to-head: two models, one box, one build, one harness.

Every suite is identical in prompt text, item set, token cap and answer
extraction for both models. The ONLY thing that varies is the model file.

Design rules (learned the hard way in this project):
  - Resumable JSONL per (model, suite), so a crash never loses a run and a
    re-score never needs a re-generation.
  - Full transcripts stored. Extraction bugs are fixed by re-scoring.
  - `hit_cap` recorded separately from `wrong`, like the author's tables.
  - Sampler is the Whittle card's (0.7/0.8/20/1.05) for BOTH models, because
    that is the harness every published Whittle number was taken on. The card
    notes Qwen's own sampler would be 0.6/0.95; using one sampler for both is
    the only way the comparison is about the weights.

Usage:
    python3 head2head.py --base http://127.0.0.1:8091 --tag qwen38 --suite gsm50
    python3 head2head.py --base http://127.0.0.1:8080 --tag whittle --suite all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLER = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
SEED = 20261004
N_GSM = 50


# ------------------------------------------------------------------ transport

def _post(base, path, payload, timeout):
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, prompt, max_tokens, timeout=1800, thinking=True):
    p = {"model": "m", "messages": [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens, **SAMPLER}
    if thinking:
        p["chat_template_kwargs"] = {"enable_thinking": True}
    d = _post(base, "/v1/chat/completions", p, timeout)
    ch = d["choices"][0]
    return ch["message"].get("content") or "", d.get("usage", {}), ch.get("finish_reason")


def raw(base, prompt, max_tokens=12, timeout=180):
    """Greedy raw completion — how the knowledge rows were measured."""
    d = _post(base, "/v1/completions", {
        "model": "m", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_k": 1}, timeout)
    return d["choices"][0]["text"]


# ------------------------------------------------------------------ extraction

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def strip_think(t):
    return re.sub(r"</?think>", "", t, flags=re.I)


def num_after_answer(text):
    t = strip_think(text)
    m = re.search(r"ANSWER:\s*(-?[\d,\.]+)", t, re.I)
    cand = [m.group(1)] if m else []
    if not cand:
        nums = NUM.findall(t.rstrip(" .,;:"))
        cand = nums[-1:] if nums else []
    if not cand:
        return None
    s = cand[0].replace(",", "").rstrip(".")
    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except ValueError:
        return None


BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def boxed_answer(text):
    t = strip_think(text)
    m = BOXED.findall(t)
    if not m:
        return None
    s = m[-1].strip()
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    return s


def norm_math(s):
    """Loose normalisation: both sides go through it before comparing."""
    if s is None:
        return None
    s = str(s).strip().lower()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("x=", "").replace("=", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    # unwrap a single \frac{a}{b} into a decimal, so \frac{3}{4} == 0.75
    def _frac(m):
        a, b = m.group(1), m.group(2)
        try:
            if re.fullmatch(r"-?\d+", a) and re.fullmatch(r"-?\d+", b) and int(b) != 0:
                v = int(a) / int(b)
                return f"{v:.10g}"
        except Exception:
            pass
        return f"{a}/{b}"
    s = re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", _frac, s)
    s = re.sub(r"^\.0*|\.0+$", "", s)
    try:
        v = float(s)
        return f"{v:.6g}"
    except ValueError:
        return s


def math_ok(pred, gold):
    a, b = norm_math(pred), norm_math(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    # numeric tolerance for rationals/decimals
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


# ------------------------------------------------------------------ items

def load_gsm50(path):
    rows = [json.loads(l) for l in open(path)]
    items = []
    for r in rows:
        m = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
        if not m:
            continue
        g = m.group(1).replace(",", "")
        try:
            v = float(g)
        except ValueError:
            continue
        items.append({"q": r["question"],
                      "gold": int(v) if v.is_integer() else v})
    rnd = random.Random(SEED)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    return [items[i] for i in idx[:N_GSM]]


def load_cloze(path):
    """Pull the 200 (prompt, accept-list) tuples out of cloze_big.py."""
    src = open(path).read()
    body = src[src.index("ITEMS = ["):]
    ns = {}
    exec(compile(body.split("\n]\n")[0] + "\n]", "<cloze>", "exec"), ns)
    return [{"prompt": p, "accept": list(a)} for p, a in ns["ITEMS"]]


def load_math60(path):
    items = json.load(open(path))
    return items[:60]


def load_humaneval(path):
    return [json.loads(l) for l in open(path)]


# ------------------------------------------------------------------ runners

def already(path, key):
    if not os.path.exists(path):
        return set()
    done = set()
    for l in open(path):
        try:
            done.add(json.loads(l)[key])
        except Exception:
            pass
    return done


def pmap(fn, items, workers, keyfn):
    """Thread pool that writes are serialised behind a lock in each runner."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if workers <= 1:
        for it in items:
            fn(it)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"[worker] {type(e).__name__}: {e}", flush=True)


def _writer(path):
    """One lock per output file; append-and-flush under the lock."""
    import threading
    lock = threading.Lock()
    fh = open(path, "a")

    def emit(row):
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return emit, fh


def run_gsm50(base, tag, args):
    out = f"{HERE}/h2h_{tag}_gsm50.jsonl"
    items = load_gsm50(args.gsm8k)
    done = already(out, "i")
    emit, fh = _writer(out)
    suffix = ("\n\nShow your reasoning step by step, then write the final numeric "
              "answer on its own line as:\nANSWER: <number>")

    def one(i):
        it = items[i]
        txt, usage, fin = chat(base, it["q"] + suffix, 3072)
        pred = num_after_answer(txt)
        ok = pred is not None and abs(float(pred) - float(it["gold"])) < 1e-6
        emit({"i": i, "q": it["q"], "gold": it["gold"], "pred": pred, "ok": ok,
              "hit_cap": fin == "length", "ctok": usage.get("completion_tokens", 0),
              "raw": txt})
        print(f"[gsm {i+1:2d}/{len(items)}] {'OK ' if ok else 'MISS'} "
              f"pred={pred} gold={it['gold']}", flush=True)

    pmap(one, [i for i in range(len(items)) if i not in done], args.workers, None)
    fh.close()
    return out


def run_cloze(base, tag, args):
    out = f"{HERE}/h2h_{tag}_cloze.jsonl"
    items = load_cloze(args.cloze_src)
    done = already(out, "i")
    emit, fh = _writer(out)

    def one(i):
        it = items[i]
        txt = raw(base, it["prompt"], 12)
        first = txt.strip().split("\n")[0]
        ok = any(a.lower() in first.lower() for a in it["accept"])
        emit({"i": i, "prompt": it["prompt"], "out": first[:160],
              "accept": it["accept"], "ok": ok})
        print(f"[cloze {i+1:3d}/{len(items)}] {'OK ' if ok else 'MISS'} "
              f"{first[:50]!r}", flush=True)

    pmap(one, [i for i in range(len(items)) if i not in done], args.workers, None)
    fh.close()
    return out


def run_math60(base, tag, args):
    out = f"{HERE}/h2h_{tag}_math60.jsonl"
    items = load_math60(args.math_items)
    done = already(out, "i")
    emit, fh = _writer(out)

    def one(i):
        it = items[i]
        txt, usage, fin = chat(base, it["problem"] + "\n\nPut the final answer in \\boxed{}.", 6144)
        pred = boxed_answer(txt)
        ok = math_ok(pred, it["gold"])
        emit({"i": i, "level": it["level"], "gold": it["gold"], "pred": pred,
              "ok": ok, "hit_cap": fin == "length",
              "ctok": usage.get("completion_tokens", 0), "raw": txt})
        print(f"[math {i+1:2d}/{len(items)}] {'OK ' if ok else 'MISS'} "
              f"L{it['level']} pred={str(pred)[:30]!r}", flush=True)

    pmap(one, [i for i in range(len(items)) if i not in done], args.workers, None)
    fh.close()
    return out


def run_humaneval(base, tag, args):
    out = f"{HERE}/h2h_{tag}_humaneval.jsonl"
    items = load_humaneval(args.humaneval)
    done = already(out, "task_id")
    emit, fh = _writer(out)
    sysmsg = ("Complete the following Python function. Output ONLY the code, "
              "no markdown fences, no explanation. Include the full function "
              "signature and body.")

    def one(it):
        tid = it["task_id"]
        prompt = it["prompt"]
        txt, usage, fin = chat(base, sysmsg + "\n\n```python\n" + prompt + "```", 1024)
        code = txt
        m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
        if m:
            code = m.group(1)
        code = code.replace("```python", "").replace("```", "").strip()
        if not re.search(r"^\s*def\s", code, re.M):
            code = prompt + "\n" + code
        emit({"task_id": tid, "prompt": prompt, "completion": code, "raw": txt,
              "ctok": usage.get("completion_tokens", 0)})
        print(f"[he {tid}] {usage.get('completion_tokens',0)} tok", flush=True)

    pmap(one, [it for it in items if it["task_id"] not in done], args.workers, None)
    fh.close()
    return out


SUITES = {"gsm50": run_gsm50, "cloze": run_cloze, "math60": run_math60,
          "humaneval": run_humaneval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suite", default="all")
    ap.add_argument("--gsm8k", default=f"{HERE}/gsm8k_test.jsonl")
    ap.add_argument("--cloze-src", default=f"{HERE}/cloze_big.py")
    ap.add_argument("--math-items", default=f"{HERE}/math60_items.json")
    ap.add_argument("--humaneval", default=f"{HERE}/HumanEval.jsonl")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(f"{args.base}/health", timeout=10) as r:
            print(f"server: {r.read().decode()[:80]}", flush=True)
    except Exception as e:
        sys.exit(f"server not reachable at {args.base}: {e}")

    names = list(SUITES) if args.suite == "all" else args.suite.split(",")
    for n in names:
        print(f"\n=== {n} [{args.tag}] ===", flush=True)
        SUITES[n](args.base, args.tag, args)


if __name__ == "__main__":
    main()
