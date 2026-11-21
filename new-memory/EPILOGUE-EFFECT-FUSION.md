# Epilogue effect fusion (design note, for review; not built)

## Problem

A guest epilogue that mixes per-lane scalars with vocabulary rows launches
several tiny one-block regions a fire. On MI300X each costs about 5 us
whatever it does (launch plus the region prologue: params, descriptors,
offsets staged into shared memory), and beam-4 on Qwen3.5-0.8B runs 3-4 of
them a step (26 us a step of one-block `ptir_region`s, traced; the plain
4-lane step is 1.72 ms, beam-4 2.56 ms). The same shape appears in any
epilogue that reads or writes small channel cells around row work: samplers
with temperature or penalty scalars, parallel samples, scoring.

The order-only coalescing pass (independent pure Single nodes moved across a
Rows run) was built and measured: no change (beam-1 1.696 / 1.698 ms vs 1.704
/ 1.689; beam-4 2.562 / 2.567 vs 2.563 / 2.563). The remaining tiny regions
are not stranded pure nodes. Each is bound to a channel op or to a library
region:

- the scores take at the start (`CHAN_TAKE`, its own region: an effect node
  never joins or moves);
- the follow-ups of the library top-k (`parent = i / v`, `token = i % v`, the
  reorder) and the puts of tokens, parents and scores (`CHAN_PUT`), which
  read the top-k's outputs and cannot join a library region.

## What a channel op is inside a fire

From the SIMT emitter (`codegen/simt/fused.rs`):

- `CHAN_TAKE` / `CHAN_READ` copy the channel's cell (its pending cell where
  this fire already put one, else its committed cell) into the value's
  scratch. Nothing in the fire advances the ring: the cursor moves in the
  engine's channel kernels after the fire (`pull_validate`, `commit_bump`,
  `scatter_publish`, `settle`).
- `CHAN_PUT` writes the value into the fire's pending cell and sets the
  channel's pending flag from thread 0: an idempotent write, published by
  stream order when the kernel ends.

So within a fire a take is a read of a cell, and a put a write of a
fire-private cell. Their only ordering constraint is per channel: a take or
read after a put on the same channel must see the put (it reads the pending
cell by the flag), so it must run in a later region, or after the put in
the same block.

## Proposal

Three rules in eta's region planning (`plan/compile/region.rs`) and the
SIMT emitter, each general (no guest named), each landable alone:

1. **A take or read joins its readers.** A `CHAN_TAKE` / `CHAN_READ` of a
   cell a block covers is a pure read for planning: it may join the region
   that first reads it (every block copies the cell into its own view, as a
   broadcast is computed in each block today), or a Single run before it.
   Constraint: never placed before a `CHAN_PUT` on the same channel that
   precedes it in stage order. Removes the scores-take launch.
2. **A put joins its producer.** A `CHAN_PUT` whose value is produced in a
   generated region joins that region, as its last op. Every block writes
   its share of the value (its stretch, for a split row) and thread 0 of
   each block sets the flag (idempotent). A put after another put on the same
   channel keeps their order. Removes the puts' own launches where their
   producers are generated.
3. **A library select carries a small tail.** A library top-k (or sort)
   region may carry a tail of generated ops that read only its outputs, are
   Single (cover a block), and end in puts. The select kernel emits the
   tail in the block that holds its final candidates (the merging block of a
   split select, the single block otherwise), after the merge, over shared
   memory. The parent / token arithmetic and the puts of a beam step become
   part of the select's launch. This needs the select kernels
   (`codegen/cuda/order*.rs`) to splice a generated tail, the largest of the
   three.

Expected: beam-4 loses 2-4 one-block launches a step (about 10-20 us of
26), any epilogue with a scalar take or a trailing put loses one each, plain
decode's standard sampler loses its trailing put's region where it has one.

## Correctness

- Results identical: no op's arithmetic changes, only which launch holds it.
  The order between a put and a later take on one channel is kept by rule.
- Commit and fault paths: a region's status and commit word already cover
  every op it holds. A take or put moved into a region inherits them; a
  fault after a put in the same region leaves the pending flag set with the
  fire not committed, which the channel kernels already treat as an
  uncommitted fire (to confirm in review against `settle`).
- Split rows: a put of a row-valued value in a split region writes by
  stretches, which every block owns alone (`ptir_stretch`), as its other
  stores do.

## Tests

- eta: plan pins where regions merge (with the merge noted); a test per rule
  that a take after a put on one channel never moves before it; the emitter
  fingerprint and versions moved.
- engine-cuda: the epilogue-vs-interpreter suite over subjects with a take
  before rows, a put after rows (split and not), and a top-k with a tail;
  every cell and every channel against the interpreter.
- served: beam verify w1 / w4, greedy and top-p, beam-1 / beam-4 / plain
  c1 / 4-lane step costs before and after; every backend boots it.

## Related, separate

The engine's channel kernels run 5-6 launches a fire after every epilogue
(`pull_validate`, `commit_bump`, `scatter_publish`, `settle`,
`gather_words` x 2: about 21 us a beam-4 step, traced). One channel kernel a
fire doing all of it would remove 4-5 launches from every fire, plain decode
included, and is engine work (engine-cuda, `serve/settle.rs` and the channel
kernels), not eta's.

## Open questions

- Rule 3's tail in Metal's and XLA's select paths (Metal forms its own
  passes; XLA lowers top-k to its own op).
- Whether rule 1 should replicate a take into every reader region or keep
  one copy per fire in scratch read by later regions (memory vs launch).
