#!/usr/bin/env python3
"""Execution-score a HumanEval samples file into a h2h_*-shaped jsonl.

The h2h report expects an `ok` field; the generation step cannot know it, so
this runs the official tests and rewrites the rows with `ok` set. Output goes
to h2h_<tag>_humaneval.jsonl so the report picks it up unchanged.

Usage: python3 he_score.py <samples.jsonl> <tag>
       python3 he_score.py h2h_whittle_humaneval.jsonl whittle
       python3 he_score.py heA_whittle.jsonl whittle_armA
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_humaneval import score_one  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src, tag = sys.argv[1], sys.argv[2]
    items = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(f"{HERE}/HumanEval.jsonl")}
    rows = [json.loads(l) for l in open(src)]
    out = f"{HERE}/h2h_{tag}_humaneval.jsonl"
    n_ok = 0
    with open(out, "w") as fh:
        for r in rows:
            it = items[r["task_id"]]
            ok, why = score_one(dict(r, test=it["test"],
                                     entry_point=it["entry_point"]))
            n_ok += ok
            fh.write(json.dumps({"task_id": r["task_id"], "ok": bool(ok),
                                 "ctok": r.get("ctok", 0),
                                 "why": (why or "")[-200:]}) + "\n")
    print(f"{src} -> {out}: {n_ok}/{len(rows)} = {n_ok/len(rows):.1%}")


if __name__ == "__main__":
    main()
