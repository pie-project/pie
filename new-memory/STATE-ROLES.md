# State roles: guest instances and channels in the pool (REDESIGN step 4)

Proposal, 2026-11-13, Mac session. Scope: REDESIGN 2.6's first two rows (channel rings, guest per-instance tables). Adapters and the other 2.6 rows follow the same contract and are not designed here. One choice is left to the user: the size rule (section 3).

## 1. What is actually state

Measured on Metal (Qwen3-0.6B, text-completion, two instances, `Plane::bytes` and `currentAllocatedSize` deltas at bind and first stage):

| Buffer, per instance | Bytes | Written | Class |
|---|---|---|---|
| Channel ring (capacity + 1 cells × cell stride) | 32-192 (cells 16-64 B, capacity 1-11) | host per message, device per fire | **state** |
| Launch tables built at bind: descriptors, params, offsets, status, grouped layout | < 1 KiB | once at bind | **state** |
| Stage value scratch (`scratch_stride` × values, vocab-wide) | 3,688,704 | zeroed every fire | **fire** |
| Output seats, pending flags, lane indices, row meta, batch tables | < 1 KiB each | every fire | **fire** |

So 2.6's "guest per-instance tables" are two things. The state part is about 1-2 KiB an instance. The large part (the value scratch, 99.9% of the bytes) and every per-fire table are fire. They belong in the fire carve (REDESIGN 2.4, `FrameMap` / workspaces, step 3b), not in state slots. **This proposal moves only the state part.** The ledger already prices both exactly (sc/mac-ledger).

The CUDA path has the same split. `Rings` holds the ring slabs plus a registry of `MAX_RING` entries a slot. Per-fire launch words are staged on the stream. This has not been measured on CUDA, Vulkan, wgpu or XLA.

## 2. Roles and the replayable attribute

`engine::chunk::SlotRole` gains:

| Role | Holds | Replayable |
|---|---|---|
| `State` | kv list slots, recurrent records | yes |
| `RsBuffer` | unfolded rs rows | yes |
| `Instance` (new) | one guest instance's bind-time launch tables | no |
| `Channel` (new) | one channel's ring (cells, head / tail words) | no |

`SlotRole::replayable() -> bool` is the attribute. Non-replayable slots may be parked on the host tier (copy out, copy back to any slot) but never dropped. The existing rule "the runtime never drops state on its own; PLEX decides" covers both kinds. PLEX may offer a non-replayable slot for parking only.

## 3. Size rule (user decision)

The constraint: a fixed shape's `slot_bytes` is stated in `ChunkFacts` at load, and a program's sizes are known only at `register_program`.

| Option | How | Cost |
|---|---|---|
| **A. One small fixed shape per role; an item is a run of slots through a slot-ref table** (New Pie's lean) | Load states `Instance` and `Channel` shapes of a few KiB. An item of `n` bytes takes `ceil(n / slot)` slots. The engine addresses slot `k` of the item through the item's ref table, as kv pages are addressed. | Nothing new is stated after load. Indirection for items past one slot. Waste is under one slot an item. |
| B. Size classes | Load states power-of-two shapes per role (e.g. 256 B .. chunk) | No indirection. But each class holds whole chunks, so about 10 classes × one chunk (16 MiB on CUDA) is about 160 MiB per role before anything is used. |
| C. Shapes stated at registration | `register_program` adds shapes to `ChunkFacts` | Exact fit. But `ChunkFacts` and the runtime's store are no longer fixed after load, which breaks the governor's and evacuation's reasoning about shapes. |
| D. Contiguous runs inside one chunk | One shape per role, and the runtime hands out `n` adjacent slots in one chunk | No indirection. But the store needs a run allocator with fragmentation, and an item larger than a chunk is impossible. |

**Recommendation: A**, with the slot sized so the common item is one slot:
- With section 1's numbers, a 4 KiB `Instance` slot and a 1 KiB `Channel` slot hold every measured item in one slot, so the common path has no indirection: the item's base is its one slot.
- The slot sizes are structural constants, stated by the compiler's chunk plan beside the others. They are not config.
- Kernels address a multi-slot item the way kv pages are addressed. Byte `b` of the item lies at offset `b % slot` of slot `table[b / slot]`. A small cell never straddles a slot (cells are packed `floor(slot / cell)` a slot), so a read of a small cell is one table load and one contiguous read.
- **Open: cells larger than a slot.** A vocab-wide probabilities channel (cell about 600 KB, capacity 4, about 3 MiB) exists in the CUDA suite. Under A its cell is paged across about 600 `Channel` slots, and every element read goes through the table. That is correct, but it is a per-element lookup in the sampler path, with a cost to measure before adoption. If the cost shows, the fallback for cells past a slot is D (a contiguous run inside one chunk) for those channels only. This case, not the common one, decides between A and A+D.

## 4. Binding API change

1. `register_program` returns, besides the id, what an instance of it needs: `ProgramFacts { instance_bytes: u64, channels: Vec<ChannelFacts { cell_bytes, capacity }> }`.
2. The runtime takes the slots from its store before binding: `ceil(bytes / slot)` of role `Instance`, and per channel the `Channel` slots the ring needs.
3. `InstanceBinding` gains `instance_slots: Vec<u32>` (slot refs, role `Instance`). Channel registration (`register_channel`, and the channels an `InstanceBinding` names) gains `slots: Vec<u32>` (role `Channel`).
4. `close_instance` / `close_channel` return the slots to the runtime. The engine no longer frees anything for them.
5. Evacuation and `copy_slots` apply as for kv. **New engine duty:** guest launch tables bind ring and table addresses by pointer (Metal argument tables, CUDA registry), so after an evacuation that moves an `Instance` or `Channel` chunk, the engine rewrites those addresses before the next fire. kv does not need this because it is read through the chunk table at fire. This is the one non-mechanical change per backend.
6. A shared ring (several instances on one channel) is one set of `Channel` slots held by the channel, not by an instance. The runtime frees it when the last attachment closes, which mirrors `SharedRing::detach` today.

## 5. Host-write path, by backend

Rings are written by the host per message (`write_cell`), and the bind-time tables once. Both are small.

| Backend | Pool memory | Host write into a pool slot |
|---|---|---|
| Metal | Private placement-sparse buffer over Shared heaps | Through the heap's Shared alias (`Elastic::host_span` / `write_from`): a plain store, no copy. Same path kv staging uses. |
| CUDA | Device memory in the elastic range | `cuMemcpyHtoDAsync` on the instance's stream from pinned staging, as rings are staged today (`registry.stage(stream, ..)`). Per-message writes stay small async copies. |
| ROCm | One flat `hipMalloc` range | As CUDA (`hipMemcpyAsync`). |
| Vulkan | Pool segment buffers (device-local, not host-visible) | Done (sc/gpu-roles). A ring lies on the host (the interpreter's `ChannelState`), so a message writes no device memory. Only the bind-time tables are host-written: one staged copy (synchronous `vkCmdCopyBuffer` + fence) per `Instance` slot, about 27-36 us on an RTX PRO 4500 against 3 such copies per stage before. The device's channel state is the kept cell (capacity-1 cells only the device reads), now in `Channel` slots, written by the device only. |
| wgpu | Pool segment buffers | Done (sc/gpu-roles): tables by `queue.write_buffer`, one per `Instance` slot, about 5 us. Rings lie on the host and wgpu keeps no cell on the device, so it states no `Channel` shape. |
| XLA | Pool tensor donated to the executable | `Store::write_slots` exists for slot bytes. But rings today are carried cells mirrored on the host and passed through executable outputs. Moving them into the pool changes the executable's inputs. **Proposal: XLA states no `Instance` / `Channel` shapes at first and keeps today's path, priced in the ledger.** Revisit with the XLA session. |

## 6. Order

1. `SlotRole::{Instance, Channel}`, `replayable()`, and `ProgramFacts`, in `engine` (pure, no behaviour).
2. Runtime store: takes and returns the two roles, and carries refs in `InstanceBinding` and channel registration.
3. Per backend: state the two shapes, place tables and rings at slot refs, and rebind after evacuation. Metal first (direct host writes, sparse pool), then CUDA / ROCm, then Vulkan / wgpu. XLA last or never (section 5).
4. Separately, in step 3b: the value scratch and per-fire tables move into the fire carve. That is where the bytes are (3.7 MB of about 3.7 MB an instance on Metal).

## 7. Vulkan / wgpu: what is allocated after load (audit, 2026-11-14, sc/gpu-span)

Every reservation either backend makes once `Shell::load` has returned, from `crates/engine-gpu`, `crates/engine-vulkan` and `crates/engine-wgpu` at e41113458. "Device" means a device-local heap (entered in the books); "host" means a host-visible type off every device-local heap (NVIDIA picks `HOST_CACHED` system memory first; wgpu books no mappable buffer), which is not in the device ledger. Load-time reservations (weights, pool segments, arena, scratch, inputs, readout, expert seats, weight-cache tables) are not listed except where they regrow.

| What | Where | When | Memory, owner | Size | Class / goes to |
|---|---|---|---|---|---|
| Pool segments past the first | `store/pool.rs` `commit_segments` | runtime `commit_segments(n)` off the fire path | device, `pool` | `segment_chunks × chunk` each | state (already in the ledger, released by the runtime) |
| rs scratch | `serve.rs` `prepare` (`rs_scratch`) | first fire with an rs replay, regrown when a wider one comes (drains first; the old buffer dropped) | device, `rs scratch` | `rs::Seat::scratch_bytes(layout, rows_ext, lanes)` | fire: belongs in the span (FrameMap); not moved here |
| Group-mask table | `inputs.rs` `group_masks` | a fire whose groups mask outgrows the table; old buffer kept in `outgrown` until the arm is reused | host, `inputs` | next power of two of the masks' image | fire (FrameMap bound by the rung, REDESIGN 2.4) |
| Snapshot table | `serve.rs` (`snaps`) | a fire with a lane snapshotting rs state; regrown | host, `snapshot table` | next power of two of the entries, 256 B at least | fire |
| Readout seat | `serve.rs` `enqueue` | a fire reading back more rows than the load sized (`readout_rows(max_lanes)`) | host, `readout`; wgpu also regrows its `Readback` mirror (`Api::grown`) | the fire's readout layout | fire |
| Weight-cache stage and uploads | `store/rows.rs` `fill`, `store/experts.rs` `fill` / `stage` | each residency change of streamed row tables / experts; uploads live for one fill | host, `weight cache` | one cache's stage; uploads up to `UPLOAD` | model (cache) |
| Spill stage | `store.rs` `Spill` | each copy to / from the kv slot file where the API maps no host memory (wgpu); one standing stage on Vulkan | host, `spill staging` | 32 MiB batches | state tier |
| Guest session: stage value heap | `engine-vulkan/src/guest/session.rs` `build_stage`, wgpu `build_stage` | every `bind_instance` (one per instance, per stage) | **device**, `guest state` | the stage plan's `scratch` (vocab-wide values: about 3.7 MB an instance on Qwen3-0.6B) | fire: the large guest item still outside the pool (STATE-ROLES section 6 step 4, `ProgramFacts`) |
| Guest session: status, staging, landing; wgpu outbox | same | every `bind_instance`; `lend_landing` / outbox grow when a pass lands late or reads more | host, `guest` | status `STATUS_BYTES`; staging and landing the stage scratch | fire (host) |
| Guest own cell | `engine-vulkan/src/guest/session.rs` `Cell::own` | a channel cell past one chunk | device, `guest state` | the cell | state (Channel slots past a chunk) |
| Readout widen config | `engine-vulkan/src/guest/widen.rs` | first instance built | host, `guest` | `SLOTS × SLOT_BYTES` | model-like, once |
| Guest descriptor pool | `engine-vulkan/src/guest/session.rs` | every session | driver (outside) | one set a stage | outside |
| Descriptor pools of recorded and walked frames | `engine-vulkan/src/device/ctx.rs` `set` | a frame needing a set with no spare: pools of 1024 sets / 8192 storage descriptors, never freed; sets of up to 64 kept recordings (`record.rs` `SHAPES_KEPT`) are held for their life | driver (outside) | per pool | outside, priced by measurement only |
| Command buffers of kept recordings | `ctx.rs` `frame_with` / segment cuts | each recorded shape (LRU of 64) | driver (outside) | per buffer | outside |
| Timestamp query pools | `ctx.rs` `frame_with` | only under kernel profiling | driver | — | outside (diagnostic) |
| wgpu uniform chunks | `engine-wgpu/src/device/ctx.rs` `uniform_chunk`, `pipelines.rs` uniform arena | encoding a frame that runs out of uniform space; returned to a free list, never freed | device, `uniforms` | 1 MiB each | fire (per-frame) |
| wgpu frame scratch | `ctx.rs` `scratch_chunk` | a binding at an offset the device cannot bind (staged copy), per frame; pooled | device, `frame scratch` | 8 MiB, or the view when larger (not returned) | fire (per-frame) |
| Pipelines | both backends, built on first use | a kernel's first dispatch | driver (outside) | priced at load as `unbuilt pipelines` | outside |
| Tuning buffers | `engine-vulkan/src/tune/qmm.rs` | only under `pie tune` | device, `tuning` | — | not serving |

Moved by sc/gpu-span: the arena and the scratch rooms (`scratch.rs`), from buffers of their own into the top of the pool (`act_chunks` per rung). Still outside the span after it: rs scratch, the group-mask and snapshot tables, readout regrowth, the guest stage heaps, and wgpu's per-frame uniform and scratch chunks.
