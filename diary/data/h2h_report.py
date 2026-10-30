#!/usr/bin/env python3
"""Aggregate the matched head-to-head into one table, with paired tests.

Reads h2h_<tag>_<suite>.jsonl, produces:
  - per-suite accuracy per model
  - McNemar-style paired comparison on the items BOTH models answered
  - hit_cap counts separated from wrong answers
  - mean completion tokens (the "thinking budget actually spent" column)

Paired is the whole point: the two models saw byte-identical prompts, so the
discordant-pair count is a much tighter statistic than two independent CIs.

Usage: python3 h2h_report.py
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = sys.argv[1:] or ["whittle", "qwen38"]
SUITES = {"cloze": "i", "gsm50": "i", "math60": "i", "humaneval": "task_id"}


def load(tag, suite, key):
    p = os.path.join(HERE, f"h2h_{tag}_{suite}.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for l in open(p):
        try:
            r = json.loads(l)
            out[r[key]] = r
        except Exception:
            pass
    return out


def mcnemar(b, c):
    """Exact binomial two-sided p for b discordant-one-way, c the other."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    s = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def main():
    lines = []
    for suite, key in SUITES.items():
        data = {t: load(t, suite, key) for t in TAGS}
        if not any(data.values()):
            continue
        ids = set()
        for d in data.values():
            ids |= set(d)
        common = set(ids)
        for d in data.values():
            if d:
                common &= set(d)
        lines.append(f"\n### {suite}")
        for t in TAGS:
            d = data[t]
            if not d:
                lines.append(f"- {t}: (no data)")
                continue
            n = len(d)
            ok = sum(1 for r in d.values() if r.get("ok"))
            lo, hi = wilson(ok, n)
            caps = sum(1 for r in d.values() if r.get("hit_cap"))
            toks = [r.get("ctok", 0) for r in d.values() if r.get("ctok")]
            mt = sum(toks) / len(toks) if toks else 0
            lines.append(f"- **{t}**: {ok}/{n} = {ok/n:.1%} "
                         f"[{lo:.1%}–{hi:.1%}] · cap-hit {caps} · "
                         f"mean {mt:.0f} tok")
        if len(TAGS) == 2 and common:
            a, b = TAGS
            if a not in data or b not in data or not data[a] or not data[b]:
                lines.append("- paired: waiting for both models")
                continue
            common = set(data[a]) & set(data[b])
            both = only_a = only_b = neither = 0
            for i in common:
                oa, ob = data[a][i].get("ok"), data[b][i].get("ok")
                if oa and ob:
                    both += 1
                elif oa:
                    only_a += 1
                elif ob:
                    only_b += 1
                else:
                    neither += 1
            p = mcnemar(only_a, only_b)
            lines.append(f"- paired on {len(common)} shared items: "
                         f"both {both}, only-{a} {only_a}, only-{b} {only_b}, "
                         f"neither {neither} → McNemar exact p = {p:.3g}")
            gap = (only_a - only_b)
            lines.append(f"- net advantage to **{a if gap > 0 else b}**: "
                         f"{abs(gap)} items of {len(common)}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "h2h_report.md"), "w").write("# Head-to-head\n" + out + "\n")


if __name__ == "__main__":
    main()
