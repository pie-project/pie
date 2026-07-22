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

## Version 4 — deferred expert streaming

When `StorageTarget.stream_routed_experts` is set, routed MoE expert weights are
excluded from the resident `schedule` (and from `memory.persistent_bytes`).
They are described by `StorageProgram.stream`:

- `stream.template`: instruction IDs into `instrs` that are **not** on
  `schedule`. These are `ExtentWrite`s whose `dest.offset` is relative to a
  cache-slot base and whose `dest.buffer` is the sentinel `BufferId(u32::MAX)`.
  Section count is arch-defined (DeepSeek-V4: 6; GPT-OSS RoutedDequant: 4;
  Mixtral: 3 BF16 `w1/w2/w3.weight`).
- `stream.bindings`: flat `[num_layers × num_experts × sections]` source
  extents that instantiate the template at decode time. An arch plugin may
  map one checkpoint tensor per cell (DSv4, Mixtral) or slice fused `[E, …]`
  banks into per-expert extents (GPT-OSS).
- `stream.files` / `section_offsets` / `section_bytes` / `slot_bytes`: layout
  the driver's expert stream cache needs to open shards and size the slab.

Boot execution runs `schedule` only. On a cache miss the driver executes the
template into `slot_base` with sources taken from `bindings` for that
`(layer, expert)` — deferred loader execution, not a parallel I/O path.

Supported arches today: `deepseek_v4`, `gpt_oss`, `mixtral` (plain ExtentWrite;
GPT-OSS RoutedDequant only — biases stay resident).
