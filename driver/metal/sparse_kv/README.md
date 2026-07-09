# Metal 4 placement sparse buffers for elastic KV cache — verification

**Question:** Can Metal 4 *placement sparse buffers* give a CUDA-VMM-style elastic KV
cache — reserve a huge **virtual** buffer with no physical backing, then map/unmap
placement-heap tiles onto specific byte ranges, and actually **shrink process
footprint** when a request ends?

**Verdict: TRUE.** Every assertion holds on Apple M1 Max / macOS 26.3 / Metal 4.

## Model (as implemented in `sparse_kv_probe.mm`)

- `newBufferWithLength:options:placementSparsePageSize:` → a sparse buffer whose
  virtual length / GPU address range exists but has **no backing** (costs ~0 physical).
- Physical tiles come from an `MTLHeapTypePlacement` heap; the heap allocates its
  full size **up front** and is the real memory cost.
- `-[MTL4CommandQueue updateBufferMappings:heap:operations:count:]` aliases heap
  tiles onto buffer tile-ranges (`MTL4UpdateSparseBufferMappingOperation`, mode
  Map/Unmap, `bufferRange` and `heapOffset` **in tiles**). `heap:nil` is legal only
  for Unmap.
- **Unmap ≠ free**: an unmapped tile returns to *your* heap's free list; the heap
  (and OS footprint) stay put. To actually reclaim footprint you run a **chunked
  heap pool** and release a heap chunk once it is fully empty.
- Mapping runs on the `MTL4CommandQueue`; cross-queue ordering vs GPU reads/writes
  is done with an `MTLSharedEvent` (the probe's legacy compute queue waits on it).

## Measured results (`measured_output.txt`)

| step | `device.currentAllocatedSize` |
|---|---|
| start | 0.39 MB |
| **+ 8 GB virtual sparse buffer** | **0.39 MB — free** |
| + 128 MB placement heap | 134.6 MB (heap = physical, up front) |
| **map all 2048 tiles** | **134.6 MB — mapping adds nothing** |
| GPU writes 134 MB into mapped tiles | **0 mismatches (backing is real)** |
| **unmap all tiles, heap kept** | **134.6 MB — unmap does NOT free** |
| release the `MTLHeap` | 0.39 MB (128 MB reclaimed) |
| elastic pool: 8×64 MB chunks mapped | 537 MB peak |
| **shrink → release 6 empty chunks** | **134.6 MB — footprint really shrinks** |
| baseline: dense (non-sparse) 4 GB buffer | +4295 MB (the whole point of going sparse) |

**Map/unmap latency:** ~0.12–0.16 ms including a full GPU event round-trip, and
**~constant from 1 to 2048 tiles** — cheap enough for per-decode-step KV grow/shrink.

## Gotchas / findings

- **Placement sparse buffers must be `Private` storage.** `Shared` aborts with
  `Invalid Storage Mode 0 for Placement Sparse Buffer`. So no direct CPU `contents`;
  read/write via compute/blit (the probe does this).
- `buf.sparseBufferTier` throws *unrecognized selector* on M1 (`AGXG13XFamilyBuffer`),
  yet placement-sparse map/unmap **works** on M1 Max. Don't gate on that property.
- `bufferRange` / `heapOffset` are in **tiles**, not bytes.
- `device.currentAllocatedSize` is the precise memory metric; process
  `phys_footprint` (task_info) is sticky/laggy and lags reclaim.

## Implications for a real KV runtime

1. One large virtual KV buffer per sequence (or global), sized for `max_ctx`, costs
   nothing until mapped.
2. Back it from a **chunked placement-heap pool** (e.g. 64/128 MB chunks), not one
   max-size heap, so evicting/finishing a request can release whole empty chunks and
   drop real footprint.
3. Per-page epoch/fence tracking: unmap a KV page only after its last GPU read
   completes, and reuse the tile only after the unmap op completes (event-ordered).
4. Attention must read only resident token blocks (block table + bounds/mask); do not
   rely on unmapped-region access semantics.

## Reproduce

```sh
cd driver/metal/sparse_kv
clang++ -fobjc-arc -fmodules -O2 -framework Metal -framework Foundation \
      sparse_kv_probe.mm -o sparse_kv_probe
./sparse_kv_probe
```
