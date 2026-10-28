"""Load Whittle-45B on the Spark with the n-gram table in FILE-BACKED memory.

WHY
---
Three bf16/int8 loads were OOM-killed at 74%, 80% and 87% of the weight shards.
Measured facts that rule out capacity:
  * the container is uncapped (memory.max = max) and holds 132 GiB of touched pages;
  * a synthetic 68 GiB + 20 GiB + 20 GiB + 10 GiB anonymous pattern succeeds.
What the loader does that the synthetic test does not: it allocates the 8-head n-gram
table as ONE 20 GB anonymous tensor and then concatenates five shards into it, while
85 GB of safetensors mmap is being actively faulted. The shards holding the table are
model-00012/13/14 -- the last three of fourteen -- and the int8 attempt died at 87%
(850/975 keys), i.e. exactly there.

THE FIX
-------
Give the 20 GB table a file-backed mapping instead of anonymous memory. Pages written
into it are clean-after-writeback and therefore reclaimable, so the kernel has somewhere
to go that is not "kill the 90 GB process". Everything else is the stock path.

Run:  LOAD_ONLY=1 python3 spark_load_fix.py
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/hf")
sys.path.insert(0, "/workspace/w")
import torch

REPO = "logic65/Whittle-Qwen-3.8-45B-A3B"
TABLE_FILE = "/workspace/w/ple_table.bin"


def meminfo():
    with open("/proc/meminfo") as f:
        d = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in f}
    return d.get("MemAvailable"), d.get("Cached")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import whittle_load as wl
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _qwen4exp

    cfg = AutoConfig.from_pretrained(REPO)
    prefix = f"model.layers.{cfg.ple_layer_ids[0] - 1}.ple.ple_embedding"
    hb, rows = wl._hash_buffers(REPO)
    sizes, offsets = hb["ngram_heads_vocab_sizes"], hb["ngram_heads_offsets"]
    total = offsets[-1] + sizes[-1]
    # The table's width is NOT hidden_size. It is 256 (`ple_embed_dim`/head dim); using
    # cfg.hidden_size (2048) asks for 159.9 GB instead of 20.0 GB. Read it off the shard.
    import json as _json
    from safetensors import safe_open
    wmap = _json.load(open(wl._local(REPO, "model.safetensors.index.json")))["weight_map"]
    shard_key = next(k for k in wmap if k.startswith("model.ngram_embedding.shard_"))
    with safe_open(wl._local(REPO, wmap[shard_key]), "pt") as f:
        dim = f.get_slice(shard_key).get_shape()[1]
    nbytes = rows * dim * 2  # bf16
    log(f"table: {rows:,} rows x {dim} = {nbytes/1e9:.1f} GB -> {TABLE_FILE}")

    # Pre-size the backing file; a sparse file costs nothing until written.
    if not os.path.exists(TABLE_FILE) or os.path.getsize(TABLE_FILE) != nbytes:
        with open(TABLE_FILE, "wb") as f:
            f.truncate(nbytes)
        log("backing file created")

    original_init = _qwen4exp.Qwen4ExpTextNGramEmbedding.__init__
    state = {}

    def fb_init(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        original_init(self, config, embedding_dim, layer_idx, ple_layer_index)
        assert len(sizes) == self.ngram_heads, (len(sizes), self.ngram_heads)
        self.head_vocab_sizes, self.head_offsets, self.total_vocab_size = list(sizes), list(offsets), total
        self.ngram_heads_vocab_sizes = torch.nn.Buffer(torch.tensor(sizes, dtype=torch.long))
        self.ngram_heads_offsets = torch.nn.Buffer(torch.tensor(offsets, dtype=torch.long))
        d = self.ngram_embedding.embedding_dim
        # file-backed: pages are reclaimable under pressure instead of anonymous.
        # Validated on this box: torch.from_file accepts bf16 directly, forward() works,
        # in-place writes land in the file, requires_grad=False. Do NOT build the
        # Embedding via __new__ -- sparse.py reads self.padding_idx and crashes.
        flat = torch.from_file(TABLE_FILE, shared=False, size=rows * d, dtype=torch.bfloat16)
        self.ngram_embedding = torch.nn.Embedding(rows, d)
        self.ngram_embedding.weight = torch.nn.Parameter(flat.view(rows, d), requires_grad=False)
        state["ok"] = True
        log(f"table mapped file-backed, {nbytes/1e9:.1f} GB")

    _qwen4exp.Qwen4ExpTextNGramEmbedding.__init__ = fb_init
    try:
        from transformers import AutoModelForCausalLM
        log("loading")
        model = AutoModelForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: os.environ.get("GPU_CAP", "86GiB"),
                        "cpu": os.environ.get("CPU_CAP", "24GiB")},
            key_mapping={r"^model\.ple_embedding\.": f"{prefix}.",
                         r"^model\.ngram_embedding\.": f"{prefix}.ngram_embedding."},
        )
    finally:
        _qwen4exp.Qwen4ExpTextNGramEmbedding.__init__ = original_init

    log(f"loaded; avail={meminfo()}")
    dev = {}
    for _, p in model.named_parameters():
        dev[str(p.device)] = dev.get(str(p.device), 0) + p.numel()
    log("params by device (B): " + str({k: round(v / 1e9, 3) for k, v in dev.items()}))
    emb = model.get_submodule(prefix)
    log(f"table shape {tuple(emb.ngram_embedding.weight.shape)} file-backed={state.get('ok')}")
    log(f"final avail={meminfo()}")

    with open("/workspace/w/loadfix.json", "w") as f:
        json.dump({"ok": True, "rows": rows, "dim": dim, "bytes": nbytes,
                   "params_by_device": dev, "avail": meminfo()}, f, indent=1)
    print("<<<LOAD_OK>>>")


if __name__ == "__main__":
    main()
