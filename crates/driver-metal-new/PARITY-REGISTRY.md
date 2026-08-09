# Parity ledger: `registry.cpp` + `descriptor_resolve.hpp` → `src/pipeline/`

Every entity is ported, dropped (with the reason the C++ needed it and the Rust
does not), or missing (with the blocker). Same rules as `PARITY.md`.

Like `PARITY-INTERP.md`, this ledger is written after the fact. Both files were
ported as their subjects came up and neither got a ledger, which is how
`CUTOVER.md` came to list them as prerequisites it still had to satisfy. They
are the third and fourth entries this session found already done and still
recorded as outstanding.

## `registry.cpp` (452) → `src/pipeline/registry.rs`

Programs, channels and instances, and the rules deciding whether an instance
may attach to a channel.

| C++ | Rust | |
|---|---|---|
| `Registry::register_program` | `Registry::register_program` | ported |
| `Registry::register_channel` | `Registry::register_channel` | ported |
| `Registry::bind_instance` | `Registry::bind_instance` | ported |
| `Registry::close_instance` | `Registry::close_instance` | ported |
| `Registry::close_channel` | `Registry::close_channel` | ported |
| `Registry::find_program` ×2 (const and not) | `Registry::program` | ported |
| `Registry::find_channel` ×2 | `Registry::channel` | ported |
| `Registry::find_instance` ×2 | `Registry::instance` / `instance_mut` | ported |
| `ChannelRecord::numel` | `Channel::numel` | ported |
| `ChannelRecord::program_dtype` | `Channel::state` / `channel_dtype` | ported |
| `host_role_for(Channel)` | `HostRole` | ported |
| `extern_dir_for(Channel)` | `Direction` | ported |
| `channel_dtype_for(DType)` | `channel_dtype` | ported |
| `cell_bytes_for(Channel)` | `pipeline::wire_cell_bytes` | dropped |
| `chainable(Decl)` | the bind rules | ported |
| `shape_text(Dims)` | `Display` on the error | ported |
| `ProgramRecord` / `ChannelRecord` / `InstanceRecord` | `Program` / `Channel` / `Instance` | ported |

Each `find_*` is one Rust method rather than a const/non-const pair: the
constness the C++ spells twice is the borrow, and a signature that cannot be
written twice cannot disagree with itself.

`shape_text` built a human string for a refusal by concatenation at the throw
site. The Rust renders it in the error's `Display`, so a refusal that is never
printed costs nothing to format — and, more usefully, the refusal carries the
dims rather than a sentence about them, so a caller can act on it.

`cell_bytes_for` is dropped because `pipeline::wire_cell_bytes` already answers
it and is the same function the ring, the codec and `Ring::new` use. A second
spelling of the cell width is the way a ring and its reader come to disagree.

## `descriptor_resolve.hpp` (400) → `src/pipeline/resolve.rs`

Reading a fire's geometry out of the channels the last fire wrote it into.

| C++ | Rust | |
|---|---|---|
| `resolve_fire_geometry_typed` ×3 overloads | `resolve` | ported |
| `resolve_fire_geometry` (bool + out-params) | `resolve -> Resolution` | ported |
| `GeometryResolveResult` | `Resolution` | ported |
| `read_port_cell` | `Bindings` + `resolve`'s port walk | ported |
| `read_mask_cell` | `read_mask` | ported |
| `value_as_u32` | `as_u32` | ported |
| the last-page-length arithmetic | `last_page_len` | ported |
| `translate_kv_pages` | — | **missing** |

The three `resolve_fire_geometry_typed` overloads and the `bool`-returning
`resolve_fire_geometry` are one function in Rust. The C++ pair is the usual
shape — a `bool` beside out-parameters the caller may read either way —
and `Resolution` is an enum, so a geometry that did not resolve does not exist
to be read.

### The one missing entry

`translate_kv_pages(tr, tr_len, …)` maps a frame's logical KV page ids through
the working-set translation table. It is **not** part of resolving a fire's
geometry from its channels — it is frame-level bookkeeping the executor does
before a fire, over `FrameSubmission::kv_translation` and its CSR partition.
It belongs with `forward.cpp`, and it is one of the concrete reasons
`CUTOVER.md`'s gate item 2 cannot be attempted yet: the engine seam's `launch`
hands the driver exactly that table.

## Closed out

Both files are accounted for. `registry.cpp` has nothing missing.
`descriptor_resolve.hpp` has one entry, `translate_kv_pages`, blocked on the
forward executor that owns the frame it describes.
