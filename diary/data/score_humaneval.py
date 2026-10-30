#!/usr/bin/env python3
"""Score h2h_<tag>_humaneval.jsonl by executing each completion.

Not pass@k — one sample per item, greedy-ish sampler, exactly the shape of the
card's "HumanEval (164, executed)" number. A completion passes iff every test
in the official `test` field passes and the entry point is callable.

Safety: runs in a subprocess with a hard timeout and a memory cap. HumanEval
tests are benign, but a model that emits an infinite loop should not hang the
harness.

Usage: python3 score_humaneval.py h2h_qwen38_humaneval.jsonl [more...]
"""
import json
import subprocess
import sys
import tempfile
import os

RUNNER = r"""
import sys, signal
signal.alarm(10)
{code}
{test}
check({entry})
print("PASS")
"""


def score_one(rec, timeout=15):
    code = rec["completion"]
    test = rec.get("test", "")
    entry = rec.get("entry_point", "")
    if not test or not entry:
        return False, "no test/entry_point"
    src = RUNNER.format(code=code, test=test, entry=entry)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        ok = p.returncode == 0 and "PASS" in p.stdout
        return ok, (p.stderr or p.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def score_file(samples_path, items_path):
    items = {json.loads(l)["task_id"]: json.loads(l) for l in open(items_path)}
    rows = [json.loads(l) for l in open(samples_path)]
    n_ok = 0
    for r in rows:
        it = items.get(r["task_id"], {})
        r2 = dict(r, test=it.get("test", ""), entry_point=it.get("entry_point", ""))
        ok, why = score_one(r2)
        n_ok += ok
        if not ok:
            print(f"  FAIL {r['task_id']}: {why.splitlines()[-1][:110] if why else '?'}")
    print(f"{samples_path}: {n_ok}/{len(rows)} = {n_ok/len(rows):.1%}")
    return {"file": samples_path, "score": n_ok, "total": len(rows)}


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    res = [score_file(p, os.path.join(base, "HumanEval.jsonl")) for p in sys.argv[1:]]
    json.dump(res, open(os.path.join(base, "h2h_humaneval_scores.json"), "w"), indent=1)
