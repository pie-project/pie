# XLA: what lives in the pool, and what does not (design note)

Status, 2026-11-14. Measurements on v6e; models qwen3-30b-a3b (cut to 8
layers) and qwen3-30b-a3b-mini; speed harness `tests/engine_xla/speed.rs`.

## The fact that decides most of this

A pool write placed anywhere after a fire's first layer changes XLA's
memory-space assignment for the whole program:

| pool write in the fire (zeros into the sink chunk) | mini decode 64 tok/s |
|---|---|
| none | 40.6–40.9k |
| at the start, 512 rows (256 KiB) | 40.8k (free) |
| at the start, 75 968 rows (39 MiB) | 38.2k (−6%) |
| at the end, 512 rows | 39.2k (−4%) |
| at the end, 75 968 rows | 37.1k (−9%) |
| one small write in every layer | 38.4k (−5%) |
| same, 30B-8L prefill 2048 | 93.8k vs 107.0k (−12%) |
| counters in a donated buffer of their own, end of fire | 40.8k (free) |

The write's kind does not matter: an XLA in-place dynamic-update-slice and
the `pool_scatter` DMA kernel (input/output-aliased, 1 MiB scoped VMEM)
cost the same. `kv_append`'s writes, part of the baseline, cost nothing.

**Dump pair** (`/workspace/xd-base`, `/workspace/xd-endw`, mini decode 64,
one end write): without the write, 533 ops get scoped VMEM at offset 16 MiB
and the first 16 MiB holds the cross-program-prefetched `bf16[2048,4096]`
q-projection weight for the whole program (18 `S(1)` uses of q weights, 56
slice-starts, 326 copy-starts). With it, all 919 ops get scoped VMEM at
offset 0, the cross-program prefetch region is not kept, and later layers
lose their VMEM placement (6 `S(1)`, 32 slice-starts, 282 copy-starts).

**Flags tried, none recovers** (fresh compiles): 
`xla_msa_cross_program_prefetch_permissive_mode=true` (39.3k),
`xla_tpu_enable_cross_program_prefetch_freeing=false` (39.2k),
`xla_msa_enable_cross_program_prefetch_freeing=false` (39.4k).

**Open item.** Hypotheses left: the HLO schedule orders differently once a
late pool write exists (memory-pressure scheduling); the extra alias chain
changes MSA's cost model. Until one is confirmed and steered, every design
below avoids pool writes after a fire's first layer.

## Placement, item by item

| memory | where | why |
|---|---|---|
| kv, recurrent state, rs buffer | pool, slots (State, RsBuffer) | as today |
| streamed expert planes, row-table blocks | pool, weight slots; misses staged in the program from pinned host banks | A7 |
| use counters | **one small donated buffer, sized at load (exception, user's decision)** | in the pool they cost 4% decode, 12% prefill; their own buffer costs nothing |
| readout rows | **the same buffer (exception, user's decision)**, read by `CopyRawToHost` at an offset | an end-of-fire pool write costs ≥4% (−9% for full decode-64 logits) |
| staged inputs (pack, arrays) | pool region, written at the fire's start from a pinned host ring | start writes measure free |
| device-fed tokens (today: pack patches) | read by the fire from where the guest pass left them in the pool | no patch program, no fresh pack |
| guest carried one-cell rings | pool, Channel slots | today: a fresh output buffer each pass |
| guest rings, bind-time tables | host | the host interpreter owns them; no executable reads them as tables |
| guest stage inputs | the staged-inputs path | as fire inputs |
| adapter banks | pool, Adapter slots | GPU's 5b153be5a |
| MoE intermediates (224 of 254 MiB at the 30B 4096-token rung) | XLA temp, measured in the ledger | writing them into `act_chunks` mid-fire pays the layout switch above |
| the rest of activation (15–30 MiB), executable code, the shared temp reservation | XLA's, measured in the ledger | not ours to place |

## Staged inputs

Today each fire uploads its pack (every i32 input) as a fresh device buffer,
runs one patch program per device source to splice in words the device
produced (tokens a guest pass sampled), and uploads f32/raw arrays as fresh
buffers. All are device allocations after load.

Design:

1. **Pinned host ring**, sized at load: `fires in flight + 1` slots, each a
   `pinned_host` buffer of the largest pack (and array set) any warmed rung
   stages. The host writes a fire's inputs into its slot in place
   (`PJRT_Buffer_UnsafePointer`, measured: a program sees each write; a
   4 KiB write costs nanoseconds) and never into a slot a queued fire has
   not read. Entered in the ledger's host side: class Fire, host tier.
2. **Pool region**: a reserved run the fire copies its slot into at its
   start (a start write: free), then reads every input from. The program's
   inputs become one host parameter of a fixed shape (the slot), so one
   program a rung, as today.
3. **Device-fed words**: a guest pass leaves its outputs in pool slots
   (Channel role for carried cells; the same for sampled tokens), and the
   fire reads them there at its start by pool word index the host stages
   in the pack. No patch program, no fresh pack.

## Guest programs

The host interpreter owns every ring (as on wgpu, which states no Channel
shape for that reason): `the ring lives on the host` is the design. Device
parts move: carried one-cell rings into Channel slots (stage executables
take the donated pool and slot rows as scalars, read and write in place;
WARM_STRICT holds), stage inputs through the staged-inputs path, and
stages are compiled at registration for every warmed extent and answered
through `fire_facts` (contract 0f2b546f5), an extent outside them refused.
Guest stages are small executables of their own; the layout switch above is
to be measured there before relying on it.

## Readout

Read straight from where the fire left it with `PJRT_Buffer_CopyRawToHost`
at an offset (measured correct, ~120 µs a small read against ~290 µs for a
2 MiB download). Where the fire writes it is the user's decision: the
counters' small buffer (free) or the pool (≥4%).
