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
  GPT-OSS Native Marlin, GPT-OSS Eager BF16, GPT-OSS RoutedDecode TP MXFP4);
  set by the same arch selectors that choose section layout.

Boot execution runs `schedule` only. On a cache miss the driver executes the
template into `slot_base` with sources taken from `bindings` for that
`(layer, expert)` — deferred loader execution, not a parallel I/O path.

Supported arches today: `deepseek_v4`, `gpt_oss`, `mixtral`, `qwen3_moe`,
`qwen3_5_moe` (plain ExtentWrite; GPT-OSS RoutedDequant streams HF packs at
`tp_size=1`; GPT-OSS native streams a Marlin expert pack; GPT-OSS eager_bf16
streams a BF16 expert pack; biases stay resident; Qwen shared expert / router
stay resident).

## Version 6 — TP + streaming packs

`STORAGE_PROGRAM_VERSION = 6` adds TP-local expert packs so
`stream_routed_experts && tp_size > 1` can page contiguous per-rank extents
(cache key already includes `tp_rank`/`tp_size`):

- GPT-OSS `GptOssRoutedMxfp4` / Native Marlin / Eager BF16: section sizes use
  `I_local = I_full / tp_size`; builders apply the same row/column offsets as
  resident TP (RoutedDecode densifies strided down groups).
- DeepSeek-V4 `Dsv4TpMxfp4`: contiguous MXFP4 `w1`/`w3` row slices and densified
  `w2` columns (weight+scale) under `tp_size > 1`; shared experts / MTP /
  router stay resident. At `tp_size = 1`, named experts page as full HF
  tensors (`pack_kind = None`).
- Mixtral `MixtralTpBf16`: contiguous BF16 `w1`/`w3` row slices and densified
  `w2` columns under `tp_size > 1`; at `tp_size = 1` Mixtral still pages full
  HF experts (`pack_kind = None`).
- Qwen3.5-MoE `Qwen35MoeTpBf16`: contiguous fused BF16 `gate_up`/`down` local-I
  sections under `tp_size > 1` (gate/up halves concatenated, down columns
  densified); shared expert and router stay resident. At `tp_size = 1`, fused
  banks page as full HF expert slices (`pack_kind = None`).
- Qwen3-MoE `Qwen3MoeTpBf16`: contiguous named BF16 `gate`/`up`/`down` local-I
  sections under `tp_size > 1` (down columns densified); at `tp_size = 1`,
  named experts page as full HF tensors (`pack_kind = None`).

Arches without a per-rank pack still reject `stream_routed_experts &&
tp_size > 1`.
