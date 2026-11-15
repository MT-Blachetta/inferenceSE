# myKimi3

Optimization profile for **Kimi K3.1 Pro**
([moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro),
revision `d9e99b07`) on this machine:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 |
| CPU | AMD Ryzen 9 9950X3D, 16 cores |
| RAM | 128 GB DDR5-6000 (2 × 64 GB) |
| SSD | WD_BLACK SN8100 4 TB NVMe |
| Board | ASRock X870E Taichi |

The model is 1.56 TB. The machine has 224 GB of VRAM and RAM together. The
profile runs it by keeping what every token needs in memory and reading the
rest from the SSD as it is needed.

**This profile has never been run against the real weights or on a GPU.**
Read [What was checked](#what-was-checked) before using it.

## The model, as published

Taken from the model card, `config.json`, the repository file list and the
headers of all 96 safetensors shards (`facts/`).

| | |
|---|---|
| Type | LLM, text + image input (this profile: text only) |
| Architecture | Mixture of experts, 93 decoder layers, hidden size 7168 |
| Parameters | 2.8 T total, 104 B active per token |
| Attention | 69 layers Kimi Delta Attention (linear, recurrent), 24 layers gated sparse MLA with a top-2048 indexer |
| Feed-forward | layer 0 dense; layers 1–92 latent MoE: 896 experts, 16 per token, 2 shared, latent width 3584 |
| Residual | attention residuals over blocks of 12 layers |
| Files | 96 `.safetensors` shards, one per layer plus head, projector and vision tower |
| Tensors | 497,340 |
| Tensor types | `U8` 1,446.5 GB (MXFP4-packed expert weights and their scales), `BF16` 114.7 GB, `F32` 0.04 GB |
| Size | 1,561,217,377,280 bytes |
| Context | 1,048,576 tokens |
| Vocabulary | 163,840 (tiktoken BPE) |

One expert is six tensors (three matrices, each packed 4-bit plus one scale
byte per 32 weights) that are contiguous on disk: 17,547,264 bytes. A layer
holds 896 of them, 15.7 GB.

## What a token costs

Every decoded token needs:

- all dense weights — attention projections, shared experts, routers, the
  output head: **103.9 GiB**;
- 16 of 896 experts in each of 92 layers: 1,472 experts, **24.0 GiB** out of
  1,347 GiB;
- one row of the embedding table.

Dense weights alone exceed the 96 GB of VRAM. The plan, computed from the
real tensor layout by `inferencese plan myKimi3`:

```
profile myKimi3: 497,340 tensors in 87,428 blocks, 1,454.0 GiB mapped from the SSD (no copy)
tier 0 VRAM cache    91.0 GiB  (filled 90.7)
tier 1 RAM  cache   116.0 GiB  (filled 13.1)

class         blocks       GiB  home    in VRAM    in RAM   SSD only
expert        82,432   1347.12  ram        0.00      0.00    1347.12
dense          2,578    101.67  vram      88.53     13.14       0.00
lm_head            5      2.19  vram       2.19      0.00       0.00
embedding      2,245      2.19  ram        0.00      0.00       2.19
vision           168      0.83  ssd        0.00      0.00       0.83
```

- **VRAM** holds dense weights and nothing else. They are used by every token,
  so nothing is worth more there.
- **RAM** holds the 13 GiB of dense weights that did not fit. The remaining
  103 GiB fill with experts as they are used: about 6,300 of them (7.6 %),
  chosen by how often each has been needed. The table shows the state right
  after start, before any expert has been read.
- **SSD** holds everything. The model files are used where they are; no copy
  is made, so the drive keeps about 2.4 TB free. Reads bypass the page cache.

The expert cache is where the profile adapts. Routing is not uniform in
practice, and the cache keeps the experts that have been popular, remembers
them across restarts (`~/.cache/inferencese/myKimi3/heat-*.json`), and refuses
to evict a popular expert for a rare one.

## Expected speed

These are estimates from `inferencese simulate myKimi3`, which replays a
synthetic workload against the plan above. They rest on the throughput
assumptions in `hardware.yaml` (9 GiB/s sustained SSD reads, 20 GiB/s RAM→VRAM)
and on how skewed expert usage is, which is unknown until the real model runs.

| expert usage | SSD read per token | experts found in RAM | seconds per token | tokens per second |
|---|---|---|---|---|
| uniform | 22.3 GiB | 7 % | 5.0 | 0.20 |
| mildly skewed (Zipf 0.6) | 17.6 GiB | 27 % | 4.5 | 0.22 |
| skewed (Zipf 1.0) | 10.8 GiB | 55 % | 3.8 | 0.27 |
| strongly skewed (Zipf 1.3) | 6.7 GiB | 72 % | 3.3 | 0.30 |

Of the fixed cost, about 1.9 s per token is copying RAM→VRAM (13 GiB of
spilled dense weights and 24 GiB of experts) and 0.7 s is GPU compute.

A prompt is processed in chunks of 1,024 tokens. Together the tokens of a
chunk use nearly every expert, so each chunk is one sweep over the whole
model: roughly 4 minutes at these rates, about 4 tokens per second of prompt.
A follow-up turn in the same conversation only processes its new tokens.

Expect an answer of a few hundred tokens, including the model's reasoning, to
take tens of minutes. Lower `thinking_effort` to shorten it.

## Run it

Put the 96 shards, `config.json`, `tiktoken.model` and `tokenizer_config.json`
in one folder.

```bash
inferencese plan myKimi3 -m /data/models/Kimi-K3.1-Pro
```

```bash
inferencese run myKimi3 -m /data/models/Kimi-K3.1-Pro --chat -p "Explain MoE routing in two sentences." -n 400
```

```bash
inferencese serve myKimi3 -m /data/models/Kimi-K3.1-Pro
```

Starting the engine loads 104 GiB of dense weights, about 12 seconds of SSD
time at the assumed rate. Set `model.path` in `profile.yaml` to avoid passing
`-m`.

## Settings worth knowing

| setting | default | effect |
|---|---|---|
| `memory.tiers.gpu.reserve` | 5 GiB | VRAM left for activations and state; lower it to keep more dense weights in VRAM |
| `memory.tiers.ram.reserve` | 12 GiB | RAM left for the OS and the process; each GiB freed caches 61 more experts |
| `memory.policy.alpha` | 0.02 | how fast expert popularity follows recent use |
| `runtime.pin_host_memory` | true | read SSD blocks into page-locked buffers for faster RAM→VRAM copies; falls back to ordinary buffers if the allocation fails |
| `runtime.prefill_chunk` | 1024 | prompt tokens per sweep; larger is faster per token and needs more VRAM |
| `runtime.expert_prefetch_layers` | 2 | reads ahead the experts the previous token used in the next layers |
| `tokenizer.thinking_effort` | max | `low`, `high`, `max`; also settable per request as `reasoning_effort` |
| `tokenizer.trust_model_code` | false | allow importing the model's `encoding_k3.py` for tool calls and JSON mode |

## Files

| file | content |
|---|---|
| `profile.yaml` | model facts, runtime options |
| `hardware.yaml` | this machine, throughput assumptions |
| `memory.yaml` | tiers, block classes, policy, warm-up |
| `model.seil` | the 93-layer forward pass |
| `ops.py` | attention residuals, SiTU, KDA, gated sparse MLA, MXFP4 latent MoE |
| `tokenizer.py` | tiktoken vocabulary, chat format, reasoning/answer split |
| `facts/config.json` | the model's configuration |
| `facts/catalog.json.xz` | name, dtype, shape, file and offset of all 497,340 tensors |

## What was checked

The repository publishes no modelling code, so the forward pass was written
from vLLM's implementation (`vllm/models/kimi_k3`), which the model card
names as a supported engine.

Checked:

- Every tensor the program asks for, in all 93 layers, exists in the real
  checkpoint with the expected shape, and no tensor of the text model is left
  unused (`tests/test_kimi.py`, against `facts/catalog.json.xz`).
- The memory rules cover the real layout: 82,432 expert blocks of exactly
  17,547,264 bytes, dense weights spill to RAM and never to SSD.
- On a miniature model with the same tensor names, dtypes and MXFP4 packing
  but random weights: processing a prompt at once, in chunks and token by
  token gives the same logits; so do NumPy and PyTorch; so does a cache many
  times smaller than the model. That exercises the KDA state, the
  convolution history, the sparse-attention cache and its top-k path, and
  attention residuals across block boundaries.
- An end-to-end run on the real file layout. `inferencese replica myKimi3
  <dir>` recreates all 96 shards as sparse files with their true sizes,
  headers and tensor offsets (1,454 GiB apparent, 87 MB on disk; weights read
  as zeros, small tensors random). On a machine with 15 GB of RAM and no GPU,
  with a 6 GiB cache and the NumPy backend, the profile processed a 2-token
  prompt and generated 2 tokens: three passes over all 93 layers, 1,472
  experts read per pass, 419 GiB read from the SSD in 14,803 requests, context
  state paged to the swap file, peak process memory 9.0 GiB, finite logits of
  size 163,840. The passes took 897 s, 566 s and 527 s. This shows that the
  catalog, the memory plan, the paging and the program work at full size; it
  says nothing about output quality.
- MXFP4 decoding against the bit-level definition; attention-residual mixing
  against a transcription of the reference kernel's loop; the router's
  selection and weighting.
- The tokenizer loads the real 163,840-token vocabulary and renders plain
  system/user/assistant conversations in the model's format.

Not checked:

- **Output against the real model.** No reference outputs were available and
  the weights were not downloaded. A mistake in the transcription of the
  maths would show up as incoherent text.
- **Anything on CUDA.**
- **Speed.**

Known deviations from the reference:

- Expert inputs are not quantised to MXFP8. The model was trained with
  MXFP8 activations; the profile computes in higher precision.
- KDA's `A_log` has 128 entries for 96 heads. The profile uses the first 96,
  one per head, as vLLM's kernels index it. `runtime.kda_a_log_axis: dim`
  selects the other reading.
- `dsa_config.index_skip_topk_offset` is not used (vLLM does not use it
  either).
- Images are not supported.
- Tool calls and JSON mode need `trust_model_code`.

The first thing to do on the real machine is a short greedy generation and a
look at whether the text is coherent.

To repeat the full-size dry run anywhere:

```bash
inferencese replica myKimi3 /tmp/kimi-replica
```

```bash
inferencese run myKimi3 -m /tmp/kimi-replica --backend numpy --ram 6GiB --set memory.warmup=[] -p "ab" -n 2
```
