# Optimization profiles

A profile is a folder whose name is the profile's name. It turns the generic
engine into an engine for one model on one machine.

```
profiles/<name>/
  profile.yaml     identity, model facts, runtime options, index of the other files
  hardware.yaml    the host the profile was tuned for, and throughput assumptions
  memory.yaml      tiers, block classes, cache policy, warm-up, swap
  model.seil       the forward pass in SEIL
  ops.py           model-specific VM ops            (optional)
  tokenizer.py     tokenizer / chat-format plugin   (optional)
  facts/           data scraped from the model's repository
```

Profiles are found in `./profiles`, in the `profiles` folder of the
installation, and in the directories listed in `INFERENCESE_PROFILES`.

`ops.py` and `tokenizer.py` are Python and run inside the engine process.
Treat a profile like any other program you install.

## Starting one

```bash
inferencese new-profile <org>/<model> <name>
```

This reads the Hugging Face repository — API record, model card,
`config.json`, the file list with sizes, and the header of every safetensors
shard through HTTP range requests — and writes a skeleton. No weights are
downloaded. `facts/tensor_patterns.txt` lists every tensor name pattern with
its dtype and count, which is the starting point for the program and the
memory rules. For LLaMA-shaped checkpoints the skeleton already runs; for
anything else `model.seil` is a stub to fill in.

## profile.yaml

```yaml
name: myprofile                  # must equal the folder name
model:
  id: org/model
  path: /data/models/model       # or pass --model, or set INFERENCESE_MODEL
  config: facts/config.json      # used when the model folder has no config.json
  config_key: text_config        # expose this sub-dictionary as `cfg`
  manifest: facts/catalog.json.xz   # tensor index, for plan/simulate without the weights
  aliases: {lm_head.weight: model.embed_tokens.weight}
program: model.seil
ops: ops.py
tokenizer: {type: auto}          # or {plugin: tokenizer.py}
hardware: hardware.yaml
memory: memory.yaml
runtime:
  backend: auto                  # numpy | torch | auto
  device: auto                   # cuda | cpu | auto
  prefill_chunk: 256             # prompt tokens per sweep over the weights
  max_context: 0                 # 0 = unlimited
sampling: {temperature: 0.7, top_p: 0.9}
```

Keys under `runtime` are also visible to ops as `ctx.opts`, so a profile can
define its own knobs.

## memory.yaml

```yaml
tiers:
  gpu: {capacity: auto, reserve: 5GiB}     # "auto" = free memory minus reserve
  ram: {capacity: 80%}                     # or a percentage, or "96GiB"
  ssd: {direct_io: true, io_threads: 32, swap: {path: /fast/swap.bin}}

policy: {name: plan, alpha: 0.02}          # plan | lru | lfu

classes:                                   # first match wins
  expert:
    match: ['\.experts\.\d+\.']            # regular expressions on tensor names
    group: '^(.*\.experts\.\d+)\.'         # tensors with the same capture and adjacent on disk
                                           # form one block
    tier: ram                              # best tier this class may occupy: gpu | ram | ssd
    prior: 0.0179                          # expected use per step before any observation
  embedding:
    match: ['embed_tokens\.weight$']
    tier: ram
    chunk_bytes: 1MiB                      # split a large tensor by rows
  dense:
    match: ['.*']
    tier: gpu
    planned: true                          # touched in a fixed order every step

state: {tier: gpu, rows_per_block: 1024}   # per-token caches (pageable)

warmup:                                    # initial cache fill, in order
  - {class: dense, device: gpu}
  - {class: expert, device: ram, heat: true}   # by popularity learnt in earlier runs
```

How to choose:

- Mark as `planned` everything every token needs. The policy then knows its
  next use exactly.
- Give data-dependent tensors their own class with a `prior`, and `group`
  tensors that are only ever used together so they are read with one request.
- Use `chunk_bytes` for tensors of which a step needs only a few rows.
- `tier: ssd` with `cache: false` keeps a class out of the caches entirely.

Check the result with `inferencese plan <profile>`. Sizes use binary units;
`GB` and `GiB` both mean 2^30 bytes.

## SEIL

One statement per line. `;` or `#` starts a comment. A trailing `\`, an open
bracket or a trailing comma continues the line.

```
program name

func main(%tokens) -> %logits
  let n = cfg.num_hidden_layers
  %x = embed @"model.embed_tokens.weight", %tokens
  for i in range(n)
    %x = call block(i, %x)
  end
  if cfg.get("final_norm", True)
    %x = rmsnorm %x, @"model.norm.weight", eps=cfg.rms_norm_eps
  end
  %x = last %x
  %logits = linear %x, @"lm_head.weight"
  ret %logits
end
```

| form | meaning |
|---|---|
| `%name` | register holding a tensor (or any value an op returns) |
| `@"a.{i}.b"` | weight reference; `{expr}` is evaluated and substituted |
| `key=value` | keyword argument of an op |
| `[%a, %b]` | list |
| anything else | scalar expression |
| `%a, %b = op ...` | ops may return several values |
| `%y = call f(args)` | call a function of the program |
| `let v = expr` | bind a scalar |
| `for v in expr` … `end` | loop |
| `if` / `elif` / `else` / `end` | branch |
| `ret values` | return |

Scalar expressions are Python expressions restricted to constants, names,
arithmetic, comparisons, boolean logic, conditional expressions, subscripts,
attribute access, f-strings, and calls to `range len min max int float abs
bool str list sum any all sorted round`. Names resolve to `let` bindings, loop
variables, function parameters, `cfg` (the model configuration) and `opts`
(the profile's `runtime` section). There are no builtins, no comprehensions
and no access to names starting with an underscore.

`main` receives the new token ids of the current sequence and returns the
logits of the last position.

### Built-in ops

| op | |
|---|---|
| `embed w, tokens` | row lookup; reads only the chunks that hold the rows |
| `linear x, w, bias=` | `x @ w.T`; streams a chunked `w` piece by piece |
| `rmsnorm x, w, eps=` · `layernorm x, w, b, eps=` | |
| `add` `sub` `mul` `scale x, s` | |
| `silu` `gelu x, tanh=` `relu` `sigmoid` `tanh` `softmax x, axis=` | |
| `attention x, q=,k=,v=,o=, heads=, kv_heads=, key=, rope_theta=, rope_style=` | causal multi-head / grouped-query attention with a paged KV cache; `rope_style` is `half` (Hugging Face) or `interleaved` (GGUF) |
| `posembed x, w` | learned absolute positions |
| `concat [..], axis=` · `last x` · `weight w` | |
| `prefetch pattern…` · `release pattern…` | memory hints on weight-name globs |
| `trace x, label=` | print statistics of a register |

## ops.py

```python
from inferencese.runtime.store import W

@op("my_gate")                      # `op` is injected by the loader
def my_gate(ctx, x, prefix):
    B, ws = ctx.B, ctx.ws           # backend, weight store
    w = ws.get(W(f"{prefix}.weight"))          # through the virtual memory
    return x * B.sigmoid(B.linear(x, w))
```

`ctx` provides:

- `ctx.B` — the backend (`linear`, `einsum`, `softmax`, `topk`, `cat`, …);
  the same code runs on NumPy and PyTorch.
- `ctx.ws` — weights: `get(name)`, `get_many(names)`, `raw_many(names)` for
  storage-format bytes, `rows(name, ids)`, `blocks(names)`.
- `ctx.mem` — the unified memory: `prefetch(blocks)`, `release(blocks)`.
- `ctx.state` — the sequence: `vars` for small recurrent tensors,
  `log(key, width)` for paged per-token rows (`append`, `read_all`,
  `read_rows`), `pos` for the number of tokens already processed.
- `ctx.cfg`, `ctx.opts`, and `ctx.cache` for derived constants.

Ask for several weights with one `raw_many`/`get_many` call when they may be
on the SSD: the blocks are then read in parallel.

## tokenizer.py

Expose `load(model_dir, profile)` returning an object with `encode(text)`,
`decode(ids)`, `apply_chat(messages, add_generation_prompt=True, **kw)`,
`eos_ids` and `bos_id`. Optionally `output_filter()` returning an object whose
`push(text)` yields `(channel, text)` pairs, to separate reasoning from the
answer.
