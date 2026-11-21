# The carried gated-delta decode kernel: its shape on CUDA, ROCm and Metal

A decode lane in a buffering fire reads its carried rows (rows written past
the committed state, `lagged-fold.md`, `BEAM-PAGE-LISTS.md` §11) as a linear
attention from the committed state. This note fixes the kernel's shape on
every engine: a short-carry form for the natural lag (all of c1 / c4 / c64
decode) and a long-carry split (beam lanes, long run-ahead, deep buffered
prefixes).

## Where the cost is

CUDA, Qwen3.5-0.8B, per GDN layer per decode step:

- c1: the plain fused step 5.78 µs; the carried kernel 8.10 µs at ~1 carried
  row, 8.33 µs at 7. The gap is ~2.3 µs of structure; rows add ~0.03 µs each.
- c64: 94.7 vs 80.5 µs.
- ROCm at c64: 61.8 vs 33.2 µs a launch, the whole decode gap there.
- Metal, one lane: plain 15 µs; carried 23 / 34 / 36 µs at 0 / 3 / 7 rows.
- Long carry (beam, ~100+ rows on 4 lanes): ~0.12 µs a row a layer,
  latency-bound: one block a (lane, head) streams 16-row chunks with
  barriers, ~100 GB/s against the device's ~860.

## 1. Short-carry form (the natural lag: at most one 16-row chunk)

The fused step's shape plus at most one barrier.

- The fused step's tiling: a warp / simdgroup holds a 16-column value tile of
  the state in registers, loaded once, stored once.
- No block staging of carried rows. Each warp reads its own 16 columns of each
  row's `u` straight from global (64 B a row: coalesced and L2-resident, the
  rows written a step ago). The staged path moves the whole 512 B `u` row
  through shared memory with cp.async and two barriers a chunk: pure
  overhead at n <= 16, and its 24.7 KB of shared memory holds occupancy to
  one block an SM.
- The per-row scalars once a head, shared through shared memory with a single
  barrier: row s's keys against the own row's normalized q and k (two
  K_d-long dots), its key norm and its log-gate. Row s to warp s mod W, which
  reads the 256 B key row coalesced and dots it against q / k in registers.
  (Metal's register-only variant, every simdgroup redoing every row, was no
  better: the work must be once a head.)
- Each warp then forms the weights (a suffix scan of the log-gates over at
  most 16 rows, one shuffle scan) and accumulates `a` / `o` from its `u`
  columns; the rest as the fused step.
- Under-occupation at c1: one head a block is 16 blocks on 142 SMs (CUDA) or
  16 threadgroups on 24 cores (Metal). Split a head's value tiles over 2-4
  blocks / threadgroups, each recomputing the per-row scalars (cheap with
  the stored scalars below: the two dots, 7 x 2 x 128 MACs a block). The
  blocks a head follow from the shape (lanes x heads against the resident
  blocks), as the fused step already picks its warps.

Measured on CUDA (L40S, Qwen3.5-0.8B): the no-stage form above, as
written, is slower than the staged one: c1 / c4 / c64 351 / 1385 / 9564
tok/s against 355 / 1400 / 10008. Without the stage, each row's keys and
`u` load after the state instead of being prefetched with it a tile ahead,
which is what made the staged form fast; folding the committed rows from
the staged chunk changed nothing. Metal measured the opposite: the no-stage
form wins there (one lane 19.5 / 23.4 / 24.8 µs at 0 / 3 / 7 rows against
staged 23 / 34 / 36; a c4 step 1.420 against 1.684 ms): the M1 has no
async copy to prefetch with, and its barriers cost more. The stage is
per-engine: kept where an async prefetch exists (CUDA's cp.async), dropped
where it does not (Metal); ROCm measures before choosing. On CUDA about half the c1 gap to
the plain step is host-side (per-step prepare and rs staging), not the
kernel's shape.

### Stored per-row scalars (a buffered-row layout change)

Store each row's key norm (kinv) and its log-gate g
(`-exp(A_log) * softplus(a + dt_bias)`), f32, a head each, written by the
step that lands the row (it holds both). Later reads drop the third dot
(k . k) and the softplus / exp. Cost: 8 B a head a row (2 x V_h x 4 B, 128 B
on Qwen3.5) beside the 4 KB keys and 8 KB `u` kept already; the `[b | a]`
plane then goes unread by gated-delta reads (kept for ROCm's recompute path
or dropped). One contract commit on the line: a plane in
`ChunkProfile::rs_corrected` (the copy planes and their tiling check pick it
up through `page_planes`), the tail kernel (`gdn_tail_corrected`) writes it,
and the fold kernels read g instead of recomputing it.

## 2. Long-carry split (beam lanes, long run-ahead, deep prefixes)

- A unit is one (lane, head) block of value tiles. Where a fire's grid holds
  more blocks than units and a unit's rows span more than one chunk, its
  chunks split over `parts = min(blocks / units, chunks)` blocks: part 0 the
  latest chunks, later parts earlier ranges.
- Each part runs the staged chunk loop with its decay relative to its own
  range's end, and publishes its partial sums (`a`, `o`, V_d each) and its
  range's log-gate total G_k to a small buffer laid at the load.
- Last-arriver combine, no inter-block wait: each part fences, then counts
  itself in on the unit's counter. The block that brings it to `parts`
  combines, `a = sum_k exp(off_k) a_k` with `off_0 = g_own` and
  `off_{k+1} = off_k + G_k` (`o` alike, the state's decay `g_own + sum G`),
  finishes the tile (the state's term, the own row's `u` and output, folds,
  the store) and resets the counter. A part that waits for the others
  deadlocks wherever the grid's blocks are not all resident (a concurrent
  kernel, a profiler serializing dispatches): never that form.
- A prepare-time variant: the host knows every lane's carried count when it
  stages the rs tables, so a fire splits only where some lane carries more
  than a chunk; a plain fire keeps the plain kernel and grid exactly (a
  device-side choice on every fire launched the full-device grid on every
  decode, +2% at c1).
- CUDA: beam-4 1.155x -> 1.116x a 4-lane plain decode.

## Order on every engine

1. The short-carry form: the decode gap at c1 / c4 / c64 on all three.
2. The stored kinv + g planes (the layout change, agreed first).
3. The long-carry split as a prepare-time variant: beam and long-carry
   fires only.
