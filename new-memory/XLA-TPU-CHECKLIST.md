# XLA chunk pool — TPU checklist

Branch `sc/xla-chunk-pool` puts engine-xla on one shared chunk pool fixed at load (contract BC 1-22 on XLA):
every pool plane is a bitcast view of ONE u32 buffer `[chunks * C / 512, 128]`, a page / slot id reaches its plane's
block through a per-plane `PageMap` (`p * stride + offset`), lanes carry slot refs (`Lane.tables`, `Lane.fixed`,
`RsVerb::Buffer.pages`, `Lane.slot_copies`), host/disk tiers serve `copy_slots`. Verified on the CUDA PJRT plugin only
(bitwise vs the base build). Nothing below has run on a TPU.

## 0. Setup

```sh
# a TPU host with libtpu; one process holds the chip
export TPU_LIBRARY_PATH=/path/to/libtpu.so      # or PIE_XLA_PLUGIN
BASE=origin/new-memory                         # the commit sc/xla-chunk-pool sits on (0596e5176)
```

## 1. Bitwise gate: branch vs base on the same TPU

The harness (`tests/engine_xla/common_llm/`, `a_pool_answers_what_the_base_build_answered.rs`) loads catalog models
with seeded random bf16 weights through `Engine::load` and fires a fixed scenario (two prompts, two-lane decodes, a
prompt beside a decode; gemma: a 700-token prompt past its 512 window, then decodes). It writes every readout's bits
with `PIE_XLA_GOLDEN_DUMP=<dir>` and compares with `PIE_XLA_GOLDEN=<dir>`. The same files compile on the base build
only with a trimmed test `main.rs` (the rest of the branch's tests use the pool API):

```sh
G=/tmp/xla-golden; mkdir -p $G
# base build: branch's harness over base sources
git checkout sc/xla-chunk-pool
git checkout $BASE -- crates/engine-xla/src crates/kernels-xla
rm -f crates/engine-xla/src/store/layout.rs crates/engine-xla/src/store/tier.rs
printf 'mod common;\nmod common_dit;\nmod common_llm;\nmod a_pool_answers_what_the_base_build_answered;\n' \
  > crates/engine-xla/tests/engine_xla/main.rs
PIE_XLA_GOLDEN_DUMP=$G cargo test --release -p engine-xla --test engine_xla -- \
  a_pool_answers_what_the_base_build_answered::a_dense \
  a_pool_answers_what_the_base_build_answered::a_gated \
  a_pool_answers_what_the_base_build_answered::a_windowed --test-threads=1
git checkout HEAD -- crates/engine-xla crates/kernels-xla
# branch
PIE_XLA_GOLDEN=$G cargo test --release -p engine-xla --test engine_xla -- \
  a_pool_answers_what_the_base_build_answered --test-threads=1
```

Expect `... readouts bit-identical` for qwen3-micro-text (dense), qwen35-tiny (gated-delta hybrid) and
gemma4-e4b-mini-l1 (window list, trimmed past the window), and `a_parked_working_set_decodes_on_as_it_would_have`
(host tier + slot file park/restore, same build) to pass.

On TPU the Mosaic kernels are ON (on CUDA they were off via `PIE_XLA_FLASH_PREFILL=0 PIE_XLA_KV_KERNEL=0
PIE_XLA_ROPE_KERNEL=0 PIE_GDN_CHUNK_KERNEL=0 PIE_GDN_STEP_KERNEL=0`). Differences to expect and how to read them:
- The Mosaic kv append runs only on identity-mapped planes, so on the branch every pool plane appends through the HLO
  scatter. Values written are the same bytes; a mismatch here is a bug.
- If a gate differs, rerun both builds with the Mosaic switches above set to 0: equal then = a Mosaic path under a
  non-identity map (section 2); different then = the pool addressing itself.

## 2. TPU-only kernel tests (mapped pools; skip off a TPU)

```sh
cargo test -p engine-xla --test engine_xla -- \
  kernels_attn::the_page_copy_kernel_reads_mapped_planes_tpu_only \
  kernels_attn::a_prefill_reads_its_fresh_rows_over_mapped_planes \
  kernels_ssm::the_gdn_kernels_step_a_mapped_bank_tpu_only \
  kernels_attn::a_mapped_pool kernels_attn::kv_append \
  kernels_mla:: kernels_index:: kernels_pool:: kernels_ssm:: kernels_mosaic::
cargo test -p kernels-xla --lib pages::
```

What they cover: the decode page-copy kernel (`attn/pages.rs`) copying keys and values by their own maps; the flash
fresh-row read (`attn/flash.rs`) comparing table pages and gathering older pages by map; the GDN step / fused-chunk
Mosaic kernels on mapped block ids; every HLO path under stride 3 / offset 1 maps, bitwise against the identity layout.
Main's kernels added after the fork (v5e page-copy for 64+ rows #1596, v5e GDN decode via MXU #1600) take the same
mapped tables; they have no mapped-pool test of their own yet — the gate in section 1 covers them end to end.

## 3. Speed: branch vs base on v5e and v6e

Measure with the usual bench (decode tok/s at c = 1, 64, 256; prefill tok/s at 512 and 2048-token prompts;
`PIE_XLA_TIMING=1` / `PIE_XLA_PROFILE` for per-op time) on Qwen3-0.6B, Qwen3.5-0.8B (hybrid) and one MoE, base vs
branch, same profile file. Watch especially:

1. **Kv append without Mosaic.** Every pool plane is offset into the one buffer, so `attn/append.rs` is gated off and
   appends go through `scatter_slots` (row scatter, or whole-page writes for tall prefills). Expect prefill to lose
   what #1567/the append kernel won. Fix path: an append kernel that takes the pool once and addresses each plane by
   its map (no two aliased outputs).
2. **Bitcast-view relayout.** A plane is read as `bitcast_convert(u32 pool) -> reshape [rows, width]`. On GPU that is
   free; on TPU the tiled layouts of u32 `[R,128]` and bf16 `[N, w]` differ, so XLA may copy whole views per read and
   per write — O(pool) per layer per fire. Check the HLO profile for `copy`/`transpose`/`bitcast-convert` fusions
   over pool-sized operands and the fire's HBM traffic. If present this dominates everything; the fix is the spec's
   Pallas realization (kernels read u32 pool tiles and bitcast in VMEM), not available yet.
3. **Program reuse.** Programs are keyed as before; the copy / zero / tier programs are keyed by counts only. A new
   compile per fire at steady state is a regression (`a_prompt_length_reuses_its_programs` checks this on a dry
   device).
4. **Memory.** Pool bytes are what the old per-plane pools took (`pool_mib` in the load log; chunks x chunk size in
   `xla state pool laid`). Host tier defaults to one pool's worth of host memory (calloc, touched on use).

## 4. Known limits (not TPU questions)

- One window list (widest window, T_w = kv page), as CUDA's interim.
- Fixed rows larger than a chunk are never cut (the chunk stretches to the largest block); a record spans pieces only
  between blocks.
- rs snapshots: served by firing the pass in pieces (section 5); a kernel-level form is a speed item.
- Weight streaming: section 5 (dense bf16 only; quantized banks stay resident; a streaming load seats every atom at load).
- Frame caches (DiT conv3d) read the record's first piece only.

## 5. Weight streaming and snapshots (`sc/xla2`, on top of the chunk pool)

What it is (design, `crates/engine-xla/src/store/streamed.rs`, `serve/stream.rs`, `snapshot.rs`):

- **Held loads stream nothing.** `Residency::weights_held` (the worker sets it unless `weight_streaming = "on"`):
  no weight shape is stated, every plane lands as the base build lands it, and every program is the base build's
  (checked on the CUDA plugin by dumping programs, section 5.1). On XLA "held" is therefore the dense planes, not
  pool slots: a pool slot and a dense buffer cannot alias in PJRT, and a held program reading weights out of the
  pool would pay the bitcast relayout of section 3.2 on every read.
- **A streaming load** (`weights_held = false`) states `WeightKind::Rows` shapes (a bf16 table every reader of
  which is a plain token-id `Gather::Embed` of the fire's tokens: PLE, an untied embedding) and `WeightKind::Expert`
  shapes (dense bf16 banks read only by plain routed selects, alone or in a fused routed MLP; an atom = one expert
  of one layer, every bank's plane back to back). Every atom is seated in the pool at load, just below the sink
  (`WeightShape::seated`); the host keeps the landed bytes (the miss copy) until a held residency names every atom
  (`WeightResidency::held`), then drops them. A residency that moves an atom after that is refused (refilling from
  the serving file is not built).
- **Rows**: the host knows a fire's token ids (ids the device feeds are downloaded first: a sync, the documented
  fallback), so each token row's row of the pool is a per-fire input; a block in no slot is uploaded into a staging
  slot before the fire. The program gathers rows from a bf16 view of the pool: no block table on the device.
- **Experts**: a select reads its bank as the dense `[E*rows, cols]` value the base build reads, gathered from the
  pool (512-byte rows) by a per-fire input of each expert's first row: the kernels are the base build's. Routes are
  decided inside the fire, so a fire with a layer whose experts are not all in slots is **cut before that layer's
  first select** (`Tracer::cut`): the segment runs, its activations stay on the device as carried buffers, the
  routes come back (D2H), the routed experts in no slot are uploaded into staging slots, the layer's expert table
  is rewritten in the pack, and the next segment runs. A layer whose experts are all resident costs no cut.
- **Snapshots inside a pass** (contract 16): a step whose lanes snapshot is fired in pieces, the lane's rows up to
  each snapshot row alone, the record copied whole into the snapshot record after each piece, the last piece with
  the step's other lanes. Bitwise the cut pass by construction. `rs_snapshots = 4` where the load holds state.
  Contract 16 amended: `RsSnapshot.fixed` names the snapshot record in every fixed shape (XLA states one fixed shape
  a distinct state block size).

### 5.1 Bitwise on TPU (branch vs base, Mosaic on)

```sh
# base dumps as in section 1, plus the MoE gate (a_routed_model_answers_bit_for_bit, qwen3-30b-a3b-mini):
PIE_XLA_GOLDEN_DUMP=$G PIE_XLA_DUMP=/tmp/progs-base/<test> cargo test --release -p engine-xla --test engine_xla -- \
  a_pool_answers_what_the_base_build_answered::<test> --test-threads=1      # base build
# branch, held (programs must be the base build's: same file names under PIE_XLA_DUMP)
PIE_XLA_WEIGHTS_HELD=1 PIE_XLA_GOLDEN=$G PIE_XLA_DUMP=/tmp/progs-held/<test> cargo test --release -p engine-xla \
  --test engine_xla -- a_pool_answers_what_the_base_build_answered::<test> --test-threads=1
diff <(ls /tmp/progs-base/<test>) <(ls /tmp/progs-held/<test>)             # expect no output
# branch, streaming: every residency (seated / moved / none / half / held) = a held load at the same pool size
PIE_XLA_GOLDEN=$G cargo test --release -p engine-xla --test engine_xla -- streamed_weights:: snapshots:: \
  --test-threads=1 --nocapture
```

- `streamed_weights::a_streamed_row_table_*` (gemma4-e4b-mini-l1, PLE): every case must also print `= the base
  build's dump, bit for bit` (CUDA plugin: all five did).
- `streamed_weights::streamed_experts_*` (qwen3-30b-a3b-mini: experts + its untied embedding as rows): every case
  must equal the held load grown to the same pool (the control). On TPU the MoE runs the Mosaic grouped-matmul kernel
  on a bank gathered from the pool: equal bits there are the real check of "same kernels on the same values".
- `snapshots::` (qwen35-tiny): snapshots / record / readouts / decodes = the cut pass. On TPU the GDN Mosaic step and
  fused-chunk kernels run here (off on the CUDA plugin).
- **Pool-size sensitivity (found here, pre-existing, M63).** On the CUDA plugin the base build's own readouts
  change with the pool's chunk count alone: qwen3-30b-a3b-mini held, +1 / +4 / +37 chunks changes the 50-row prompt
  fired beside a decode (and the decodes after it), max |d| 0.016, no top-token change; +130 chunks (pool past
  2 GiB) changes every readout (max |d| 0.11). gemma4-e4b-mini-l1 +16 chunks: no change. Check on TPU: run the
  gate's held load grown (`Model::load_held_grown`, `DeviceBoot::pool_extra_chunks`) against the base dump; a
  difference past 2 GiB of pool is likely an i32 offset somewhere in the pool views (TPU pools are tens of GiB).

### 5.2 Speed (v5e / v6e, branch streaming vs held)

1. **Expert bank gathered per layer per fire.** A streamed select reads `[E*rows, cols]` gathered from the pool every
   fire: O(bank) HBM traffic per layer even for a decode whose base reads top-k experts. Measure decode tok/s at
   c = 1 / 64 on a MoE, streaming all-resident vs held. Fix path: hand the kernels the pool view and an expert ->
   row table (the gmm kernel's scalar-prefetched group offsets, the slice path's dynamic offsets) so no bank is
   materialized.
2. **Cut fires.** Each layer whose experts are not all resident costs a program boundary, a D2H of its routes and an
   H2D of its misses; the activations cross as carried buffers. Measure fire latency vs resident fraction (none /
   half / all) and the compile count: a new cut set compiles L+1 programs per fire shape (the warm ladder compiles
   only the uncut program). XLA cannot overlap the next segment's launch with the staging upload today.
3. **Row gathers from a bitcast view.** The PLE / embedding gather reads a bf16 view of the whole u32 pool
   (section 3.2): check the HLO for a pool-sized copy per gather.
4. **Device-fed token ids with streamed rows.** A lane whose tokens come off the device (a guest pass's output)
   downloads them before the fire: a sync per fire for device token loops.
5. **Staging uploads.** Misses land through maintenance programs (back-to-back slots as one block, lone slots 64 a
   program): one extra program per fire with misses.
6. **Snapshots.** A fire per snapshot row (the cut pass's cost). A scan that lands its state as it walks (as
   Vulkan's `PIE_SNAPS` twin) would take one fire; the Mosaic GDN kernels would need the same.
7. **Expert uses of resident layers are not counted** (no routes come back without a cut): the runtime's LRU sees
   only cut layers' experts. Counting them needs the routes as extra outputs downloaded at settle.
