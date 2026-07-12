# Storage Program

`StorageProgram` is the executable loading plan returned by the Rust compiler.
It contains physical instructions only: allocations, source extents, tiled
transforms, views, metadata attachments, releases, and finalization.

The executor must not need checkpoint naming rules, model-family knowledge, or
runtime ABI lookup to run it.

Version `4` is the runtime-owned compiler boundary:

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
- `BulkExtentWrite` and `SlabScatter`: coalesced arena-relative payload copies.

Every executable read names both the file and tensor ID. The C++ executor must
not infer source identity from a tensor name.

The wire document also carries the compiler source hash and a source-tensor
catalog. Deserializers reject both format-version and compiler-hash mismatches;
native executors use the catalog for dtype/shape metadata but read payload bytes
only from the explicit file offsets in instructions.
