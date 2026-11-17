# inferenceSE

An inference engine that runs models **larger than the machine's memory** by
treating the SSD as process memory. GPU VRAM and CPU RAM are caches in front
of it; the model files on disk are the memory.

```
            one virtual address space (model weights + conversation state)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ tier 0  VRAM   fastest, smallest   ─┐                                    │
 │ tier 1  RAM                         ├─ caches: hold the blocks that are  │
 │                                     ─┘  needed soonest                   │
 │ tier 2  SSD    holds everything: the model files themselves, mapped in   │
 │                place (no copy), plus a swap file for writable state      │
 └──────────────────────────────────────────────────────────────────────────┘
```

If the SSD can hold the model, the model runs. How fast depends on how much of
what each token needs fits in the caches, and on the SSD.

Status: early (v0.1). See [What is verified](#what-is-verified) before relying
on it.

## How it works

**Unified virtual memory** (`inferencese/uvm`). Every tensor is mapped to a
virtual address range, and every range to the place on the SSD that
permanently holds its bytes: an offset inside the original model file for
weights, a slot in a swap file for writable state. Opening a 1.5 TB model
reads only headers. An access looks in VRAM, then RAM, then reads from the SSD
and evicts whatever will be needed last. Weights stay in their storage
encoding (BF16, packed 4-bit, GGML blocks) in every tier and are decoded at
the moment of use.

**A virtual machine with an intermediate language** (`inferencese/vm`). The
forward pass of a model is not compiled into the engine. It is a small program
in SEIL, interpreted by the VM; each weight it names is a reference into the
virtual memory:

```
func block(i, %x) -> %x
  %h = rmsnorm %x, @"model.layers.{i}.input_layernorm.weight", eps=cfg.rms_norm_eps
  %a = attention %h, q=@"model.layers.{i}.self_attn.q_proj.weight", ...
  %x = add %x, %a
  ...
```

**Optimization profiles** (`profiles/<name>/`). A profile is a folder that
specialises the generic engine for one model on one machine: the SEIL
program, model-specific kernels, the memory layout (which tensors belong in
VRAM, which are grouped, what gets evicted first), the tokenizer, and facts
scraped from the model's repository. Three ship with the engine:

| profile | for |
|---|---|
| `myKimi3` | Kimi K3.1 Pro (1.56 TB) on an RTX PRO 6000 Blackwell 96 GB, 128 GB RAM, SN8100 4 TB — see [its README](profiles/myKimi3/README.md) |
| `generic-llama` | LLaMA-family models as Hugging Face safetensors |
| `generic-llama-gguf` | LLaMA-family models as `.gguf` files, quantised or not |

**Compute backends** (`inferencese/backend`). Tensor kernels come from
existing libraries: NumPy on CPU, PyTorch for CUDA. With CUDA, VRAM is tier 0.

**API and UI** (`inferencese/server`). An OpenAI-compatible HTTP server
(`/v1/chat/completions` with streaming, `/v1/completions`, `/v1/models`) and
a built-in chat page.

Design notes, the eviction policy and the reasoning behind these choices are
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Writing a profile is covered
in [docs/PROFILES.md](docs/PROFILES.md).

## Install

New to Linux, or setting up the Kimi K3.1 Pro machine from a fresh Ubuntu?
[how-to-run.md](how-to-run.md) walks through every step, from the graphics
driver to the chat window.

```bash
pip install -e ".[tokenizers]"
```

For GPU use, add PyTorch with CUDA (`pip install -e ".[gpu,tokenizers]"`).
Python 3.10 or newer. The core needs only NumPy and PyYAML.

## Use

Run a model that is on disk:

```bash
inferencese run generic-llama -m /models/SmolLM2-135M-Instruct --chat -p "What is the capital of France?"
```

Force it to page, by giving it a cache smaller than the model:

```bash
inferencese run generic-llama -m /models/SmolLM2-135M-Instruct --chat -p "Hello" --ram 48MiB
```

A GGUF file:

```bash
inferencese run generic-llama-gguf -m /models/SmolLM2-135M-Instruct-Q4_K_M.gguf --chat -p "Hello"
```

Serve the API and the chat page on http://127.0.0.1:8080:

```bash
inferencese serve myKimi3 -m /data/models/Kimi-K3.1-Pro
```

Plan before running: what ends up in VRAM, RAM and on the SSD, and an
estimate of decode speed (neither needs the weights):

```bash
inferencese plan myKimi3
```

```bash
inferencese simulate myKimi3
```

Create a profile skeleton from a Hugging Face repository (reads the model
card, config, file list and tensor headers; downloads no weights):

```bash
inferencese new-profile moonshotai/Kimi-K3.1-Pro myprofile
```

Other commands: `profiles`, `hardware`, `inspect <model>`, `replica`. Any
profile setting can be overridden on the command line with
`--set key.path=value`, for example `--set memory.tiers.ram.capacity=64GiB`.

## What is verified

Measured in the development sandbox (4 CPU cores, 15 GB RAM, no GPU):

- **Correctness of the engine.** `generic-llama` on SmolLM2-135M-Instruct
  matches the Hugging Face `transformers` logits to 0.01 and answers "The
  capital of France is Paris." The same answer comes out with a 48 MiB RAM
  cache for the 256 MiB model, with a simulated 24 MiB VRAM tier in front, and
  from Q8_0 and Q4_K_M GGUF files of the same model.
- **Paging is transparent.** Tests assert identical logits whether everything
  is cached or almost nothing is, and whether a prompt is processed at once,
  in chunks, or token by token — for the LLaMA program and the Kimi program.
- **Context pages too.** A KV cache larger than the RAM tier is written to the
  swap file and read back with unchanged results.
- **A model about 100 times larger than the machine's RAM runs.** `inferencese
  replica` rebuilds the exact file layout of the 1.56 TB Kimi K3.1 Pro
  checkpoint as sparse files (true sizes, headers and offsets; weights read as
  zeros). On that layout the `myKimi3` program completed three forward passes
  over all 93 layers with a 6 GiB cache: 419 GiB read in 14,803 requests,
  peak process memory 9.0 GiB, finite logits over the 163,840-token
  vocabulary. With NumPy on 4 cores a pass took 9 to 15 minutes.
- **Formats.** GGML block decoders (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K) agree with
  the reference decoder of the `gguf` package.

Not verified, and you should assume it needs work:

- **Nothing has run on a GPU.** The PyTorch backend is exercised on CPU only.
  The CUDA paths (VRAM tier, page-locked read buffers, BF16 matmuls on device)
  are written but untested.
- **`myKimi3` has never run the real weights.** See the profile's README for
  exactly what was and was not checked.
- **Speed figures are estimates** from a workload simulation, not benchmarks.

## Limits

- One sequence at a time; no batching across requests.
- Text only. Vision towers are mapped but not executed.
- GGUF-embedded tokenizers are supported for byte-level BPE vocabularies;
  SentencePiece vocabularies need a `tokenizer.json` next to the file.
- Profiles contain Python (`ops.py`, `tokenizer.py`) that runs in the engine
  process. Install only profiles you trust.

## Tests

```bash
python -m pytest
```

## License

MIT. `profiles/myKimi3/ops.py` follows the mathematics of vLLM's Kimi K3
implementation (Apache-2.0); see the notes in that file.
