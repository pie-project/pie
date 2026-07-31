# Storage Program

`StorageProgram` is the executable loading plan returned by the Rust compiler.
It contains physical instructions only: allocations, source extents, tiled
transforms, views, metadata attachments, releases, and finalization.

The executor must not need checkpoint naming rules, model-family knowledge, or
runtime ABI lookup to run it.

Version `1` covers the Rust migration boundary:

- `Allocate`: allocate one device buffer by stable `buffer_id`.
- `ExtentWrite`: copy one explicit `{file_id, tensor_id, file_offset, span}`
  source extent into one destination buffer extent.
- `TileMap`: apply a typed tiled transform. It may read directly from a source
  extent, from input buffers, or both. Current transform kinds are cast, decode,
  encode, transcode, reblock, and reorder.
- `CreateView`: create a layout view over an existing buffer.
- `Attach`: attach metadata buffers to a tensor buffer.
- `Release`: end a temporary buffer lifetime.
- `Finalize`: publish a buffer under a runtime tensor name.

Every executable read names both the file and tensor ID. The C++ executor must
not infer source identity from a tensor name.

## Version lineage

The format forked after version `3` and was developed on two branches at once.
`STORAGE_PROGRAM_VERSION` linearises them, so the numbers below are the only
ones a merged compiler may emit:

| Version | Owner | Change |
| --- | --- | --- |
| `4` | tts-arena fork | Offline GPTQ/AWQ INT4 lowering to Marlin layout. |
| `5` | upstream | Deferred routed-expert streaming (originally numbered `4` on the upstream branch). |
| `6` | upstream + merged head | Offline GPT-OSS expert packs (originally numbered `5` upstream), and the merged head that carries both branches. |

Upstream's `4` and the fork's `4` describe different formats, so upstream's
numbers are shifted up by one here. A program stamped `4` is fork INT4
lowering; a program stamped `5` or `6` is upstream streaming as renumbered
above. Anything reading `StorageProgram.version` or a dump's
`compiler_version` must use this table, not the upstream branch's own history.

Version `4` — offline GPTQ/AWQ INT4 lowering — repacks each `.qweight` /
`.scales` (and AWQ `.qzeros`) pair into a resident Marlin-layout buffer at
compile time. It requires the CUDA backend and `tp_size = 1`.

Version `4` and version `5` are mutually exclusive: streaming defers whole
expert tensors to cache slots while INT4 lowering repacks every `.qweight` into
a resident buffer, and the two have never been designed against each other.
The compiler refuses `stream_routed_experts` on a GPTQ/AWQ checkpoint rather
than emit a program whose experts are half deferred and half repacked.

## Version 5 — deferred expert streaming

When `StorageTarget.stream_routed_experts` is set, routed MoE expert weights are
excluded from the resident `schedule` (and from `memory.persistent_bytes`).
They are described by `StorageProgram.stream`:

- `stream.template`: instruction IDs into `instrs` that are **not** on
  `schedule`. These are `ExtentWrite`s whose `dest.offset` is relative to a
  cache-slot base and whose `dest.buffer` is the sentinel `BufferId(u32::MAX)`.
  Section count is arch-defined (DeepSeek-V4: 6; GPT-OSS RoutedDequant: 4;
  GPT-OSS Native Marlin pack: 6; GPT-OSS Eager BF16 pack: 3 `gate/up/down`;
  Mixtral: 3 BF16 `w1/w2/w3.weight`;
  Qwen3-MoE: 3 named `gate/up/down_proj`; Qwen3.5-MoE: 2 fused `gate_up`/`down`).
- `stream.bindings`: flat `[num_layers × num_experts × sections]` source
  extents that instantiate the template at decode time. An arch plugin may
  map one checkpoint tensor per cell (DSv4, Mixtral, Qwen3-MoE) or slice
  fused `[E, …]` banks into per-expert extents (GPT-OSS RoutedDequant,
  Qwen3.5-MoE). GPT-OSS native / eager-BF16 bindings describe pack-relative
  offsets; the driver builds the pack with bounded staging and remaps
  `stream.files` to the pack path.
- `stream.files` / `section_offsets` / `section_bytes` / `slot_bytes`: layout
  the driver's expert stream cache needs to open shards and size the slab.
- `stream.pack_kind`: selects the driver's offline pack builder (`None`,
  GPT-OSS Native Marlin, GPT-OSS Eager BF16); set by the same arch selectors
  that choose section layout.

Boot execution runs `schedule` only. On a cache miss the driver executes the
template into `slot_base` with sources taken from `bindings` for that
`(layer, expert)` — deferred loader execution, not a parallel I/O path.

Supported arches today: `deepseek_v4`, `gpt_oss`, `mixtral`, `qwen3_moe`,
`qwen3_5_moe` (plain ExtentWrite; GPT-OSS RoutedDequant streams HF packs;
GPT-OSS native streams a Marlin expert pack; GPT-OSS eager_bf16 streams a
BF16 expert pack; biases stay resident; Qwen shared expert / router stay
resident).
