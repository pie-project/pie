# State Chunk redesign — shared spec for implementation agents

Design doc (human-facing): https://claude.ai/artifact/3QFaEVRYHGJrZMquXdsG2H
Integration branch: `new-memory` on origin (state-chunk is its base) (git@github.com:pie-project/pie.git), cut from main f2f1353df.

## North star
The model says only WHAT it remembers (`state(shape, dtype)` handles). The compiler decides how that is packed
into chunks (`ChunkRecipe`). The engine counts device memory as one number of chunks. Adding a new kind of state
must need one poem line and no engine change.

## Decisions (final)
1. Memory by lifetime: weight (model) / state (holder) / activation (one fire). Weights are out of scope.
2. State units: **slot** = allocation unit (refcounted); **chunk** = physical unit holding slots of ONE slot shape,
   shared by many working sets; **segment** = the backend's grow/shrink unit; a chunk never straddles a segment.
3. Chunk size C and slot sizes are chosen by the compiler as `f(model, DeviceProfile)`: large on server CUDA
   (~16MB), small on on-device Metal. C is a load-time constant for the engine.
4. Slot shapes: **fixed** (GDN/mamba/conv state, VAE frame cache, engram/replay ids, snapshots), **list** (full-attention kv, compressed pools via `tokens // r`,
   drafter kv: T tokens x all rows, layer-major inside the slot), **ring** (sliding-window kv only).
   A ring with a small static cap folds into fixed. Fixed records larger than C split into equal pieces that never
   cross an op's atom (GDN/KDA/mamba: head axis; conv: channel; conv3d frame cache: frame-position axis 0;
   attention: token; n-gram ids: uncuttable). Do not cut pieces below 1MiB until prefill GDN is measured.
   Fixed sets: ONE per working set (resting or running) plus ONE copy-on-write set per lane that runs in a fire
   (fork / snapshot restore within the same fire) — i.e. the CoW reserve scales with max lanes, not with working
   sets. Derived, not configured. (Earlier text said "two per working set": that was wrong.)
5. Working set is logical: slots are refcounted; holders = working sets, prefix-cache entries, snapshots.
6. Budget: `state chunks + act_chunks(rung) <= total chunks`. No rs:kv ratio, no `max_state_slots`, no kv budget knob.
7. Roles: memory governor (runtime/governor.rs) decides HOW MUCH (target bytes); PLEX (policy) decides WHAT stays;
   the pool decides HOW to move (segments, evacuation). The runtime never drops checkpoints or state on its own;
   under pressure it emits `tier-pressure` with byte metrics. No new PLEX verbs.
8. Growth: pool = list of segments. CUDA/Metal (sparse VA) may grow state from the bottom and activation from a
   fixed top of one VA; other backends keep activation in its own buffer. Boundary moves and segment release go
   through evacuation (batched D2D copy + logical->physical chunk table update on the stream). Never map/commit on
   the fire path; commit a segment ahead of need.
9. poem: `state(shape, dtype)` declared at model construction like `weight()`; first axis `tokens` (or
   `tokens // r`) means per-token (list/ring), otherwise fixed. Names come from the model-tree path
   (`layers.9.delta`). `caches()`, `HybridSpec`, `kv_space`, string lookups, `write_page/offset` go away.
   Retention (window) is inferred from the reading op's window; TP split from shard inference; head_dim/live from
   the attention op. kv space disappears from IR; stream scheduler orders per state row, not per space; mask and
   attention-plan geometry read the All axis (not "space 0").
10. Kernel contract on every backend: `addr = base + slot * stride + offset` (kv: `stride_page = S`,
   `page_size = T * rate`; state: segment table + per-lane chunk ids). Default layout inside a list slot is
   layer-major (method 1). Per-layer contiguous run `T * K-row >= 32KB`.
11. Backends: CUDA (VMM segments), ROCm (flat, fixed at load), Metal (sparse heap segments), Vulkan/wgpu
   (buffer segments <= binding limit; wgpu default 128MiB binding, 8 storage buffers/stage), XLA/TPU (one pool
   fixed at load; target "tiled" realization `[chunks, rows, 128]` u32 with Pallas bitcast), Cerebras (host state).
12. Migration: **degenerate mode first** — every kernel change takes the new addressing with settings that
   reproduce today's layout exactly (page stride = today's, one segment, chunk = today's slot), verified
   bit-for-bit, before any pool switch. Legacy paths exist only as temporary verification scaffolding and are
   deleted right after each backend verifies; removed config keys are rejected with a clear message.

## Verification rules
- SPEED RULE (user, 2026-11-10): do NOT run full test suites. Run `cargo check` plus only the tests that exercise
  what you changed (filter by module/test name) and the new tests you wrote. Bitwise checks for the kernel family
  you touched still count as necessary; unrelated suites and catalog-wide sweeps do not.
- Bitwise comparisons: same process, or share the GEMM algorithm cache (cuBLAS picks algorithms per process).
  Concurrent requests can diverge by batching timing; keep them out of bitwise sets.
- Known pre-existing failures on main (do not chase, but confirm on main before claiming a regression):
  sm120 `moe_select` / `every_unit_lowers`, `elemwise::synth`, engine_cuda_ep (3 tests),
  GDN chunked prefill flake when tests run in parallel (uninitialized `ssm_gdn_wide_out`),
  `a_paged_epilogue_serves_the_logits_of_its_kv_append` (glm45-air-mini e4m3 routing).
  Confirm these yourself on the base commit if they matter to you.
- GPU on this machine: one RTX PRO 4500 Blackwell (sm_120, 32GB) shared by several agents. Wrap every GPU-using
  command in `flock /tmp/pie-gpu.lock <cmd>` and keep runs short.

## Git rules (strict)
- Work only on your own branch in your worktree. Commit with plain messages: NO `Co-Authored-By`, NO
  `Claude-Session`, NO "Generated with" lines. Author is the repo's configured user.
- Push your branch to origin as `sc/<your-package>` (`git push origin HEAD:sc/<name>`; the push URL is SSH).
  Never push to `main`, `state-chunk` or `new-memory`; the coordinator merges.
- Before starting, make sure your branch is based on `origin/state-chunk` or later `origin/new-memory` (run `git fetch origin` and rebase if not).
- Follow CONTRIBUTING.md and the repo's existing style; run clippy/rustfmt on crates you touch.

## Weight cache (MoE experts, PLE) — decided 2026-11-09
- Routed experts (expert axis) and PLE / token-id-gathered table rows (row blocks) live in the chunk pool as
  read-only **weight slots** owned by a weight-cache holder. Everything else (attention, shared experts, router,
  embedding/head, norms) stays resident as today. Dense-weight streaming is out of scope.
- No new poem declaration: the compiler infers streamability from the reading op — read only by a router-selected
  MoE op => expert-streamable (atom = expert); read only by a token-id row gather => row-streamable (atom = row block);
  anything else or mixed => resident. `Weight::host()` / `ParamPlacement::Host` are replaced by this inference and
  deleted. The result goes into `ChunkRecipe` as weight slot shapes.
- Deployment mode `weight_streaming = auto | on | off` (auto: off if the model fits, else on).
  off: weights never evicted; pressure goes to KV/state via tier-pressure (PLEX). on: under pressure, weight slots are
  evicted FIRST down to the floor, then KV/state via PLEX. Recovery: live KV/state first, then weights refill.
- Floor: MoE = current layer + next layer staging (double buffer) of top-k experts; PLE = this fire's rows.
- Residency inside the weight cache: LRU order, but (1) a miss never inserts into the cache — it streams through a
  staging slot and is dropped; (2) cache membership changes only at the governor tick (~1s) from the use record of
  the last window. This prevents thrashing on the per-token layer cycle. The runtime/engine decides (not PLEX).
- Kernels: MoE kernels take an expert -> (chunk, offset) table (degenerate = today's contiguous expert bank);
  inside CUDA graphs, missing experts are filled into staging by a zero-copy gather kernel reading pinned host memory.
  PLE is first (rows are known before the fire).

## rs buffer is a list (decided 2026-11-10, supersedes "rs buffer as fixed [P, width] rows")
- The rs buffer (unfolded recurrent inputs) is its own LIST slot shape ("rs list"), separate from the kv list:
  one slot = T_rs tokens x every recurrent layer's rs planes, T_rs chosen by the compiler small enough that a slot fits
  a chunk (e.g. 1 token for kimik3-2.8t at 6.8 MB/token; 4-16 for small models). It grows with unfolded tokens and
  holds nothing when everything is folded. No 16-row limit.
- Slots are refcounted like kv pages: forks / beam branches share their common unfolded prefix rows; a lane writes
  new rows into its own slots (copy-on-write only for a partially filled shared tail slot).
- The fixed record holds only state (no rs rows). Engines address buffered rows through the rs-list page table
  (`RsVerb::Buffer.pages` are real slot refs again), as degenerate mode did with buffer pages.
