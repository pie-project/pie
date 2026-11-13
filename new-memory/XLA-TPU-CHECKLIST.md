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
- rs snapshots: served by firing the pass in pieces (`snapshot.rs`, contract 16 with `RsSnapshot.fixed`), up to 4 a
  lane; the record copies are warmed at load. v6e: qwen35-tiny's 3 snapshots, record, readouts and decodes equal the
  cut pass bit for bit (GDN Mosaic kernels on). A kernel-level form (the scan writing state as it walks) is a speed
  item.
- Pool chunk-count dependence (MoE readouts changing with pool size alone): CUDA PJRT plugin only. v6e at
  memory-followups 9113919c2: qwen3-30b-a3b-mini and the dense gate are bit for bit vs base at pool +16, +144, +1040
  and +3072 MiB (`PIE_XLA_TEST_EXTRA_MIB`; +3072 takes the pool past 2 GiB, where the plugin changed every readout).
- Weight streaming (PLE rows, experts): not implemented; all weights stay resident. Agreed design (A7): streamed banks
  in pinned-host PJRT buffers (`mhlo.memory_kind = "pinned_host"`), read inside the fire by an expert -> slot table
  (pool slot, or host); hot atoms in pool slots, filled by a warmed copy program; no cut fires.
- Frame caches (DiT conv3d) read the record's first piece only.
