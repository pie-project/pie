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
| Vulkan | Pool segment buffers | Copy from the staging slab with `vkCmdCopyBuffer` in the next submission (unverified: whether today's guest buffers are host-visible, in which case this adds a copy per message). |
| wgpu | Pool segment buffers | `queue.write_buffer` into the slot range (unverified, as Vulkan). |
| XLA | Pool tensor donated to the executable | `Store::write_slots` exists for slot bytes. But rings today are carried cells mirrored on the host and passed through executable outputs. Moving them into the pool changes the executable's inputs. **Proposal: XLA states no `Instance` / `Channel` shapes at first and keeps today's path, priced in the ledger.** Revisit with the XLA session. |

## 6. Order

1. `SlotRole::{Instance, Channel}`, `replayable()`, and `ProgramFacts`, in `engine` (pure, no behaviour).
2. Runtime store: takes and returns the two roles, and carries refs in `InstanceBinding` and channel registration.
3. Per backend: state the two shapes, place tables and rings at slot refs, and rebind after evacuation. Metal first (direct host writes, sparse pool), then CUDA / ROCm, then Vulkan / wgpu. XLA last or never (section 5).
4. Separately, in step 3b: the value scratch and per-fire tables move into the fire carve. That is where the bytes are (3.7 MB of about 3.7 MB an instance on Metal).
