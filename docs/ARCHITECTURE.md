# Architecture

## The idea

A process normally sees one flat memory, and the operating system decides
which of its pages are in RAM. inferenceSE builds that abstraction itself, one
level up, for three kinds of memory with very different speeds:

| tier | medium | role |
|---|---|---|
| 0 | GPU VRAM | primary cache |
| 1 | CPU RAM | secondary cache |
| 2 | SSD | holds the whole address space |

The inference program never holds a pointer into host RAM or VRAM. It holds
*virtual* references. The engine resolves each one at the moment of use: look
in tier 0, then tier 1, then read it from the SSD and decide what to evict.

Doing this in the engine rather than leaving it to the OS (`mmap`) matters for
three reasons. The OS knows nothing about VRAM. It pages in 4 KiB units with
no idea that a 17 MB expert is used as a whole or not at all. And it cannot
know what the model will touch next, while the engine largely can.

## Virtual memory index and cache index

`inferencese/uvm/memory.py`

- **Virtual memory index** (`AddressSpace`): a sorted table of *blocks*. Each
  block has a virtual address range and an SSD extent, either
  `(model file, offset)` for read-only data or a slot in the swap file for
  writable data. This table is what makes the SSD the backing store of the
  whole address space. Building it copies nothing: a 1.56 TB checkpoint is
  mapped by reading 73 MB of headers.
- **Cache index**: per block, the handles of its copies in VRAM and RAM. A
  block that has neither is only a reference to its SSD extent.

Blocks are variable in size because that is how inference touches memory. A
block is one tensor, or a slice of a very large tensor (an embedding table is
cut into 1 MiB pieces so a token lookup reads 1 MiB, not gigabytes), or a
group of tensors that are always used together (the six tensors of one MoE
expert form one 17.5 MB block that is contiguous on disk).

There is also a byte-level view. `UnifiedMemory.read(vaddr, n)` and
`write(vaddr, data)` behave like one flat RAM regardless of where the bytes
are; a read may cross from a block in VRAM into one that has to be faulted in
from the SSD.

### No second copy of the model

Read-only blocks point into the original model files. Evicting one is free:
the cache copy is dropped, the SSD still has it. Only writable blocks
(conversation state) ever get written, into a sparse swap file that grows only
by what actually left RAM. For a 1.56 TB model on a 4 TB drive this is the
difference between fitting and not fitting.

Reads use `O_DIRECT` so the kernel page cache does not keep a second copy of
what the RAM tier already holds, and fall back to buffered reads with
`posix_fadvise(DONTNEED)` on filesystems that refuse it.

## What gets evicted

`inferencese/uvm/policies.py`

The best possible rule is to evict the block that will not be needed for the
longest time. For general programs that is unknowable. For inference much of
it is known:

- **Planned blocks** (dense weights) are touched in the same order on every
  decoding step, because the layer schedule is fixed. The `plan` policy
  records the access sequence of one step and from then on knows each block's
  exact distance to its next use.
- **Stochastic blocks** (MoE experts, context pages) depend on the data. For
  these the policy keeps an exponentially weighted estimate `p` of "used in a
  given step". The expected distance to the next use is `step length / p`.

Both are on one scale, so a dense tensor needed in 40 accesses outranks an
expert used in 2 % of steps, and a popular expert outranks a rare one.

Two refinements:

- **Admission control.** A block read from the SSD is only kept if it ranks
  above what it would displace. Otherwise it is used once and dropped. Without
  this, a stream of cold experts would flush the hot ones.
- **Heat map.** The per-expert estimates are saved when the engine shuts down
  and loaded at start, so a restart warms the cache with the experts that
  were actually used rather than arbitrary ones.

`lru` and `lfu` are available for comparison. In the simulation of the
`myKimi3` plan with uniformly used experts, `plan` reads 22 GiB from the SSD
per token and `lru` 36 GiB, because `lru` lets experts push out dense weights
that are needed again a moment later.

## VRAM and RAM together

A block class names the best tier it may occupy. Dense weights may live in
VRAM; when VRAM is full the rest stays in RAM and is copied to the GPU each
time it is used. By default tiers are exclusive: a block promoted to VRAM
releases its RAM copy, because RAM is more useful holding something else.
`mirror_gpu_in_ram: true` keeps both, which makes eviction from VRAM free at
the cost of RAM.

With a CUDA backend and `pin_host_memory`, SSD reads land in page-locked
buffers so the RAM→VRAM copy runs at bus speed.

## The virtual machine

`inferencese/vm`

The forward pass is a SEIL program inside the profile. The interpreter walks
it; ops receive weight *names* and materialise them through the weight store,
which is what routes every access through the virtual memory. Interpretation
overhead is microseconds per op against milliseconds to seconds of I/O and
matrix work, so nothing is lost by not compiling.

Expressions in SEIL (loop bounds, conditions, name templates) are a
restricted Python expression grammar evaluated without builtins. Programs are
checked when loaded: unknown ops, calls to undefined functions and unbalanced
blocks are errors before any weight is read.

Ordinary transformer ops are built in. Anything unusual is supplied by the
profile as Python (`ops.py`) against the backend interface, so one
implementation serves NumPy and PyTorch.

## State and context

Per-conversation state lives in the same address space:

- Small recurrent state (for example the KDA matrices of Kimi) is held as
  device tensors.
- Per-token caches are *paged logs*: append-only tables stored in writable
  blocks. They can sit in VRAM, fall back to RAM, and page out to the swap
  file. A context larger than RAM therefore still works, and a sparse
  attention that needs 2,048 of a million cached tokens faults in only the
  blocks that hold those rows.

A follow-up request usually extends the previous one. The engine reuses the
longest processed prefix: logs are truncated, and recurrent state, which
cannot be unwound, is restored from a snapshot taken at the end of the last
prompt.

## Prompt processing

Decoding one token needs a thin slice of a sparse model. A long prompt needs
nearly all of it, because many tokens together route to nearly all experts.
The engine processes the prompt in chunks (`prefill_chunk`) and each chunk
sweeps the weights once, layer by layer, in file order. That is the
layer-streaming idea of AirLLM, applied only where it pays.

## What was reused and what was not

The brief asked for an existing framework to build on. The candidates:

| project | what it offers | why it is not the base |
|---|---|---|
| llama.cpp | `mmap` of weights, mature CPU/GPU kernels, a good server | GGUF only; one fixed C++ graph per architecture; memory is paged by the OS, which cannot see VRAM or MoE structure |
| AirLLM | loads one layer at a time from disk | built on Hugging Face `transformers` model classes; needs the model's Python modelling code, which Kimi K3.1 does not publish; whole layers only, so an MoE layer costs 16 GB of reads instead of 0.3 GB |
| vLLM | the reference implementation of Kimi K3.1 | a datacentre server: its recipe for this model asks for 1,874 GB of VRAM |
| PyTorch / NumPy | tensor kernels on CUDA and CPU | used, as the compute backends |

Colibri, also named in the brief, could not be located from the build
environment and was not evaluated.

So the tensor maths is delegated to PyTorch and NumPy, tokenisation to
`tiktoken` and `tokenizers`, and three things are taken as designs rather than
code: mapping model files in place (llama.cpp), streaming layers for prompts
(AirLLM), and the OpenAI wire format for the API (llama.cpp server, vLLM).
The parts with no existing equivalent were written for this project: the
three-tier virtual memory, the policy, the VM and its language, and the
profile system. The Kimi kernels in `profiles/myKimi3/ops.py` follow the
mathematics of vLLM's implementation.

## Source map

```
inferencese/
  uvm/        address space, tiers, pager, policies, SSD and swap I/O
  formats/    safetensors / GGUF / npy headers, storage dtype decoding
  backend/    NumPy and PyTorch
  vm/         SEIL parser, interpreter, built-in ops
  runtime/    weight store, paged state, engine, sampler
  profile/    profile loader, hardware detection, generator from Hugging Face
  tokenizers/ wrappers for tokenizers / tiktoken / GGUF vocabularies
  server/     OpenAI-compatible API and chat page
  tools/      plan, simulate, sparse replica, test fixtures
profiles/     myKimi3, generic-llama, generic-llama-gguf
```
