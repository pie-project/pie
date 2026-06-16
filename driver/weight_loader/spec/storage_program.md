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
