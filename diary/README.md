# Diary

These are working notes, not write-ups: one file per sitting, in the order the measurements
came off the machines, including the ones that contradicted what I had written the day
before. The method is deliberately narrow — **measure, state the claim the measurement
supports, then try to break that claim with a cheaper experiment before spending GPU time on
it**. Concretely: perplexity on a fixed non-repetitive corpus with matched `--chunks`,
`-ngl` and thread counts; ablation by zeroing GGUF super-block scales so a weight can be
removed and restored byte-identically; paired item-level tests (McNemar, Wilson intervals)
instead of comparing two percentages; and, whenever a result would change a decision, a
control that proves the harness can produce the *other* answer. Three of the eight entries
below exist mainly because that control failed first and told me the result was mine, not
the model's. Where I got something wrong, the entry says so on the day I found out, with the
number that corrected it — a diary that only records the surviving hypotheses is a sales
document.
