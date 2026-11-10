# Baseline failures on origin/state-chunk (f2f1353df) — P8

Build: `cargo test -p engine-cuda -p kernels-cuda --features engine-cuda/cuda,kernels-cuda/cuda`
(debug), RTX PRO 4500 Blackwell (sm_120). Every chunk run serially (`--test-threads=1`) under
`flock /tmp/pie-gpu.lock`. Raw logs: `/tmp/claude-0/-root-pie/6bd922f5-9344-4b29-b26a-5ed65abcff59/scratchpad/baseline/`.

The run was cut short at the coordinator's request (GPU shared, ~4h of lock contention), so
coverage is partial. "Not run" is NOT "passes" — confirm those on the base commit yourself.

## kernels-cuda, lib (35 tests): RUN — 34 passed, 1 failed

- `jit::nvrtc::tests::every_unit_lowers` — FAILED (known, sm120 moe_select).
  `crates/kernels-cuda/src/jit/nvrtc.rs:764`:
  `linear/moe/select_fp4_sm120.cu: sm_80 has no arch-specific twin`,
  `linear/moe/select_fp8_sm120.cu: sm_80 has no arch-specific twin`, and
  `select_fp4_sm120.cu ::pie::linear::moe_select_mxfp4_w4a8<1, 3, 2> at sm_100a: would not compile`
  (`moe_w4.cuh(220): warning #128-D: loop is not reachable`, treated as an error by the strict sweep).
  Root cause: the sm120-only MoE select units are swept at every arch; not a regression.

## kernels-cuda, integration (145 tests): PARTIAL — 60 of 145 run, all 60 passed

- Chunk 000 (first 60 tests in `--list` order, `attn::*` through the start of the list): 60 passed, 0 failed (1272 s).
- Chunks 001–002 (85 tests, incl. `elemwise::synth*`, `ssm::*`, `linear::*`): NOT RUN.
  Known from the spec, unconfirmed here: `elemwise::synth` failure; GDN chunked-prefill flake under
  parallel runs (uninitialized `attn.ssm_gdn_wide_out` plane, `crates/kernels-cuda/src/ssm/gdn_chunk.rs:528`:
  the non-tcgen05 bf16 path lands f32 rows into a scratch plane sized `2 * y.elements()` that is never
  cleared, so rows a staged window does not write are read back as stale bytes — shows only when the
  plane's previous contents differ, i.e. under parallel tests).

## engine-cuda, lib: PARTIAL / NOT RUN

- Three host-only catalog sweeps run for hours in a debug build (no GPU use, but anyone running them
  under the GPU lock holds the lock that long):
  - `tests::no_region_of_a_catalog_deployment_lands_over_its_inputs` — ok (> 1 h)
  - `tests::every_fused_op_its_rules_form_is_one_dispatch_fires` — ok (> 1 h)
  - `tests::every_site_a_retired_rule_fused_folds_into_a_region` — did not finish in 2 h 40 min (killed).
    Run these three OUTSIDE the GPU lock, or skip them (`--skip`).
- The rest of the lib tests: NOT RUN.

## engine-cuda, integration (`engine_cuda`): NOT RUN
Known from the spec, unconfirmed here:
`a_synthesized_region_serves_the_logits_of_its_ops::a_paged_epilogue_serves_the_logits_of_its_kv_append`
(glm45-air-mini e4m3 routing).

## engine-cuda, `engine_cuda_ep`: NOT RUN
Known from the spec: 3 failing tests.

Per the user's speed rule, full suites were stopped; everything marked NOT RUN stays not run.

## Golden-logits probe (branch `sc/p8-golden-baseline`, commit e536895fe)

`crates/engine-cuda/tests/engine_cuda/golden_logits.rs`. Hashes on the base commit:
`/tmp/claude-0/-root-pie/6bd922f5-9344-4b29-b26a-5ed65abcff59/scratchpad/spec/golden-logits-base.txt`
(use as `PIE_GOLDEN_CHECK`). Reuse the shared cache
`PIE_GOLDEN_CACHE=/tmp/claude-0/-root-pie/6bd922f5-9344-4b29-b26a-5ed65abcff59/scratchpad/golden-cache`
(the seeded fixtures, ~38 GB, plus the recorded GEMM picks) so you skip the slow seeding and get the same algorithms.

## Notes for other agents

- Put `timeout` INSIDE the lock (`flock /tmp/pie-gpu.lock timeout N cmd`), never `timeout N flock ...`:
  the outer form spends the timeout waiting for the lock.
- A test binary inherits the flock fd from `flock ... cargo test`; if cargo dies, the orphaned test
  binary keeps the GPU lock until it exits (seen 20:02–21:35 with an orphaned engine-cuda lib run).
