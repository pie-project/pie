# Device memory inventory, all backends

Branch `memory-followups` at dbb5e043d. Read-only audit, 2026-11-13. Five per-backend sweeps starting from the allocation primitives; nothing was built or run. Items marked (unverified) were not confirmed by reading the code; the four defects in section 6 marked "checked" were re-read by hand.

Classes: **model** (load to unload, sized by the model file), **state** (survives between fires, sized by demand), **fire** (one fire or a ring of in-flight frames, sized by the rung), **overhead** (driver or library, not ours).

Cell codes: where it lives / how the load-time fit sees it.
- Where: `P` state pool chunk, `A` arena carved from the pool span, `a` arena as its own buffer, `S` separate buffer, `K` kernel slab map, `D` driver, `X` inside the XLA executable, `–` absent.
- Fit: `✓` priced by a formula, `~` only netted out because it is already resident when free memory is measured, `✗` not counted at all.

## 1. Allocators

| Backend | Allocators that put bytes on the device |
|---|---|
| CUDA | elastic range (reserve + map on demand); device heap (256 MiB chunks, never freed); recycle shelf (buffers over 64 MiB, keeps up to 8 GiB); endpoint pool; kernel slab map (grow-only, old slabs retired); raw `cudaMalloc` (three sites); driver |
| ROCm | same code, but no virtual memory: every arena is one flat `hipMalloc`, the pool range is fixed after load, commit and release of segments move no memory |
| Metal | separate Shared buffers; one Private sparse buffer over 16 MiB placement heaps for the pool, a second one for the arena |
| Vulkan | one `vkCreateBuffer` + dedicated `vkAllocateMemory` per buffer; no sub-allocator |
| wgpu | one `create_buffer` per buffer; plus raw frame pools (8 MiB scratch chunks, 1 MiB uniform chunks) that the allocation counter does not see |
| XLA | PJRT upload, compile, and executable outputs; the executable's own temporaries are invisible |
| Cerebras | none at run time: every array is static in the linked PE image |
| CPU | process heap; not an engine |

## 2. Owners by class

### model

| Owner | CUDA | ROCm | Metal | Vulkan/wgpu | XLA |
|---|---|---|---|---|---|
| Resident weight planes | S ✓ | S ✓ | S ✓ | S ✓ | S ✓ |
| Load-time decoded or re-laid copies (bf16 decode, fp8 shadow, relabelled conv, repacked banks, prescaled banks, tied-embedding copy) | S, dense ✓ rest ~ | S, dense ✓ rest ~ | S ✓ (not in the first fit pass) | S ~ | S ✓ |
| Weight-cache atoms (experts, row blocks) | P ✓ | P ✓ | P ✓ | P ✓ | – |
| Weight-cache tables and hit counters | S ~ | S ~ | S ✗ | S ✗ | – |
| Adapter banks (written after load) | (unverified) | (unverified) | S ✓ | S ✓ | S ✓, re-upload copy ✗ |
| State segment-table maps | – | – | S ✗ | – | – |
| Compiled fire bodies / executables | D ✓ (bodies allowance) | D ✓ | D ✗ (pipelines) | D ✗ | D ✗, lazy without prewarm |
| Kernel modules | D ~ | D ~ | D ✗ | D ✗ | – |
| Guest program modules | D ✗ | D ✗ | D ✗ | D ✗ | D ✗ |
| FP8 converted block scales (process static) | S ✗ | – | – | – | – |
| n-gram hash constants (inside the scratch plane) | – | – | S ✗ | S ~ | – |
| ANE weight and scale surfaces, ANE programs | – | – | S ✗ | – | – |

### state

| Owner | CUDA | ROCm | Metal | Vulkan/wgpu | XLA |
|---|---|---|---|---|---|
| KV list pages, window lists, fixed records, rs list | P ✓ | P ✓ | P ✓ | P ✓ | P ✓ |
| Compressor side planes (per physical kv page) | (in pool, unverified) | (unverified) | (unverified) | S, inside the scratch plane ✓ | P ✓ |
| Channel rings and registry | S ✗ | S ✗ | S ✗ | S ✗ | carried cells, executable output ✗ |
| Channel endpoint cells | S pooled ✗ | S pooled ✗ | – | S ✗ (kept cells) | – |
| Guest per-instance tables | S ✗ | S ✗ | S ✗ | S ✗ | – |
| Expert-parallel loads | S ✗ | unreachable | – | – | – |
| Host-tier chunks counted as device (unified or wgpu) | – | – | host | S ✗ | – |

### fire

| Owner | CUDA | ROCm | Metal | Vulkan/wgpu | XLA |
|---|---|---|---|---|---|
| Activation arena | A ✓ | A ✓ | a ✓ | a ~ | X |
| Staged inputs × frames in flight | S ~ | S ~ | S ✗ | S ~ | upload ✗ |
| Attention plan workspaces | S ✓ (in inputs) | S ✓ | (in scratch) | (in scratch) | X |
| Readout rows × frames | S ✓ | S ✓ | S ✗ | S ~, regrowth ✗ | output plane ✗ (one logits plane cited at 254 MB) |
| Decoded weight tile per stream | K ✓ | K ✓ | – | – | X |
| BLAS workspace per stream | K ✗ (8 MiB) | K ~ (128 MiB each) | – | – | – |
| All other kernel scratch (about 100 call sites on CUDA) | K ✗ | K, blanket hold then ~ | S ✗ (one plane) | S ~ (one plane) | X |
| rs extended-run scratch | own arena, beside the budget ✗ | same, first allocated at first fire ✗ | S ✗ | S ✗ | X |
| rs predicate region, snapshot tables | S ~ | S ~ | – | S ✗ | – |
| Mask and inject run tables, group masks (outgrown ones kept) | S ✗ | S ✗ | S ✗ | S ✗ | – |
| Voxel / pixel stores | S ✓ or ~ | S ~ | – | S ~ | – |
| Guest program scratch | S ✓ (reserve) | S ✓ (reserve) | S ✗ | S ✗ | upload ✗ |
| Guest batch tables, wave arena | S ✗ | S ✗ | S ✗ (never removed) | S ✗ | – |
| MoE host-expert staging (process static, never freed) | S ✗, up to 1.5 GiB | S ✗, up to 1.5 GiB | – | gathered seats + staging ring ~ | – |
| Weight-cache stage and fill upload buffers | – (pinned host) | – | – | S ✗ (Vulkan row fill: 2 × at least 64 MiB) | – |
| Transfer staging | – | – | – | Vulkan 256 MiB slab ~; wgpu read staging ✗ | upload |
| Load-time temporaries over 64 MiB | recycle shelf, see 6 | recycle shelf | – | – | – |
| TP / EP wire mappings (process-lived) | S ✗ | unreachable | – | S ✗ (no caller found) | – |
| cuBLAS tuning arena | S ✗ | – | tuner 256 MiB / 1 GiB (offline) | tuner (offline) | – |
| ANE input and partial surfaces, ANE staging | – | – | S ✗ | – | – |
| wgpu frame scratch and uniform pools, bind-cache arena | – | – | – | raw ✗ | – |

### overhead

CUDA/HIP context, streams, events, cuBLAS / rocBLAS / hipBLASLt handles, Vulkan descriptor and command pools, wgpu internals, Metal queues and residency sets; on CUDA and ROCm also the device heap chunks and the recycle shelf themselves. All are either measured implicitly or covered only by a safety floor of `min(128 MiB, total/10)`.

Cerebras and CPU do not fit the table. Cerebras: every wafer array is static and fixed at link; fit is by retrying presets when the link runs out of PE memory; the wafer holds the KV of one sequence, other lanes are kept as tokens on the host and replayed. CPU: a content-keyed host page pool bounded by one `capacity`.

## 3. What the budget equation covers today

| Backend | Does `state chunks + act_chunks(rung) <= total chunks` hold? | How the pool is actually sized |
|---|---|---|
| CUDA | Partly: arena is in the span, rungs exist; every other fire owner is outside | three mechanisms: admit (weights + one sequence), working-set fit (formulas), card fit (measured free bytes) |
| ROCm | Same accounting, but one flat allocation: nothing is returned after load | same, range re-made at the fitted size during load |
| Metal | No: `act_chunks` empty, arena is a second sparse buffer | working set × share − weights − arena − floor |
| Vulkan / wgpu | No: `act_chunks` empty | whatever is left when the pool is created, or `device_kv_budget` |
| XLA | No: fire lives inside the executable | share × HBM − held − weights − headroom |
| Cerebras, CPU | Not applicable | link-time / one capacity |

## 4. Growth after load

- Pool segments on demand (CUDA, Metal, Vulkan/wgpu). Inside the pledge, except that Vulkan/wgpu does not re-check free memory at commit.
- Kernel scratch slabs on CUDA and ROCm: grow on any non-captured fire with a wider shape; the old slab is kept, so the peak is old + new. Largest uncounted growth.
- rs scratch, mask tables, readout regrowth, guest rings / tables / scratch / modules on every backend that has them.
- Never-freed growth: group masks keep every outgrown buffer (CUDA, ROCm, Metal, Vulkan/wgpu); Metal guest batches are insert-only; MoE host-expert staging and FP8 scales are process statics; device heap chunks are never returned.
- CUDA arena: a program that reads `attn.scores` backs the whole reservation beside the budget.
- Compiled objects: pipelines on first use (Metal, Vulkan, wgpu), XLA executables for any shape the warm ladder missed, graph bodies under deferred arming.

## 5. Knobs, constants and heuristics that size device memory

Config keys still live: `device_memory_share`, `device_kv_budget`, `host_kv_budget`, `disk_kv_budget`, `max_context`, `max_forward_tokens`, `max_forward_requests`, `bodies_mem`, `working_set_percent` (50), `window_spare`, `gemm_workspace_mb`, `[xla] headroom / seat_tokens / prewarm`, `[wgpu] device_memory`, `[cerebras.wafer] preset`.

Rules: halve forward tokens until the arena fits (floor 1024 on CUDA, 512 on Metal); one sequence takes at most half the room; bodies get a quarter with a 512 MiB floor and a 1/20 stretch; fp8 shadow only under free/8; XLA prescale and tied copy only when weights are under 5/8 of HBM; voxel fit uses half the remainder (Vulkan/wgpu, XLA); safety floor `min(128 MiB, total/10)`; XLA headroom HBM/16.

Fixed sizes: activation rungs 256 / 1024 / 4096; chunk 16 MiB (CUDA, ROCm, XLA target) or 4 MiB (Vulkan/wgpu); segments 64 MiB (Metal), 1 GiB (CUDA), a quarter of the pool (Vulkan/wgpu); default state slots 256; attention workspace grant at least 64 MiB float + 1 MiB int; slab grain 8 MiB / 256 KiB; hipBLASLt workspace 128 MiB; rocBLAS grouped cap 512 MiB; staged experts cap 1.5 GiB; recycle shelf 8 GiB; device heap 256 MiB chunks; Vulkan staging 256 MiB; upload batches 64 MiB; guest scratch cap 4 GiB; ring cap 64; rs replay 16; rs snapshots 4.

## 6. Defects and risks found along the way

Checked by re-reading the code:
1. **Vulkan allocation counter drifts.** `Raw::new` adds the driver's `need.size` to `allocated` (engine-vulkan/src/device/alloc.rs:137) and `Drop` subtracts the requested `size` (:291). Each allocate/free pair leaves the padding behind, and `left()` shrinks over time.
2. **wgpu bind-cache uniform arena never shrinks.** At the 16384-group cap the cache clears `groups` and resets `used`, but keeps every 1 MiB chunk and goes on writing into the last one (engine-wgpu/src/device/pipelines.rs:240-262). Slow leak under cache churn.
3. **Vulkan/wgpu: a stated `device_kv_budget` is not compared with free memory** where the pool is sized (engine-gpu/src/serve.rs:1066).
4. **CUDA/ROCm recycle shelf is not drained before the card fit.** `release_idle` has two callers, context bind and shell drop. Load temporaries over 64 MiB are shelved on drop (up to 8 GiB) and the pool allocations do not go through the path that drains the shelf, so the fit reads them as used. Size of the effect not measured.

Reported, not checked:
5. FP8 block-scale cache is a process static keyed by device address; a later model whose atoms land at the same address with the same shape would read stale scales.
6. XLA: unknown whether pool donation updates in place or makes a transient pool-sized copy (also open in XLA-TPU-CHECKLIST.md 3.2).
7. Vulkan/wgpu `LoadFacts` reports the whole pool range as committed while one segment is backed.
8. Metal: decoded banks are missing from the first fit pass.

## 7. Not traced

The compiler side that sets chunk and segment sizes and the arena map; runtime-crate callers of commit / release; device bytes of compiled objects on every backend; whether kernel modules are ever unloaded; XLA executable temporaries; Cerebras non-dense families; adapter bank placement on CUDA and ROCm; where compressor side planes live on CUDA, ROCm and Metal.
