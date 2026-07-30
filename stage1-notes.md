# Stage 1 — hooks become a parameter (running notes)

Hardware/env: L40S (sm_89), CUDA 13.0, Qwen3-0.6B, debug build. Companion to
`stage0-l40s.md`; plan context in `pie-application-plan.md` §2.2/§7 Stage 1.

## Landed

- `01655163` — the four body-scoped thread_locals (active_stage_hooks,
  attention observation, scores, mask sink) became parameters end-to-end.
  Verified: full build; naive-baseline (hook-free) and trackb-snapkv
  (prefill score capture + OnAttn AttnScore, 28 layers, mass 28.0, 0 NaNs).
- `77446890` — `fused_decode_qkv_post` gates on a row count.
  `Dispatch::launch_hook_free_prefix_rows` = min row start over
  attention-stage programs (0 if one is unlocatable); rows [0, fast_rows)
  keep the fused QKV+norm+rope+KV-write kernel, the tail runs the unfused
  path over its own rows (new `first_token` on `write_kv_to_pages`,
  native-bf16 only). §2.2's "one tap disables the fused kernel for every
  request" is no longer true in llama_like.

## First pie-side measurements

Live mixed workload, 3× naive-baseline + 1× trackb-snapkv, 24–48 tok:

- Mechanism: with arrival-order rows, mixed decode fires log
  `R=4 fast_rows=1 fused=1` (22 fires) — one hook-free lane kept the fused
  kernel while the snapkv lane ran hook-visible, token-correct on all lanes.
  `fast_rows` is 1 rather than 3 because nothing orders hook programs last
  yet (in flight).
- Cost of carrying a hook lane (48 tok, 3 reps, naive-lane completion in
  mixed vs pure fires): 1.09–1.17× before row ordering, 0.95–1.14× after.
  Confounds folded into both numbers: the 4th lane's real work, snapkv's
  capture cost, and mixed fires running eager (the `has_stage_hooks` graph
  gate — a Stage 6 concern). On a debug build at R=4 the host overhead
  dominates the step, so the kernel savings sit inside run-to-run noise;
  the honest claim is the mechanism (below), not a throughput number.
  Re-measure on a release build at wider R when Stage 5 needs the cost
  model.
- Row ordering (scheduler): with hook programs sorted last within the wire
  class, the same workload logs `R=4 fast_rows=3 fused=1` (21 fires) — all
  three hook-free lanes in the fused prefix, only the snapkv lane
  hook-visible. Before ordering the prefix ended at whichever hook lane
  arrived first (`fast_rows=1`).
- Path-dependent rounding: a lane that moves between the fused prefix and
  the unfused tail can flip a late sampled token (bf16 rounding boundaries
  differ between the fused and unfused QKV paths — the same difference that
  already existed between hook-free and hook-present fires). Expected;
  parity within each path is what Stage 3's harness will pin.

## Deliberately not done

- gemma4's `can_fuse_packed_qkv_post` still gates fire-wide on
  `hooks == nullptr`. Same row-count treatment applies (its fused kernel is
  also r-relative), but there is no gemma checkpoint in this environment to
  E2E-verify against; defer rather than land unverified.
- Remaining fire-wide predicates (grep `hooks == nullptr` /
  `active_stage_hooks` reported 112 pre-refactor) migrate as their families
  get the llama_like treatment; the derived row count is the pattern.

## Operational notes (test harness)

- One `PieClient` connection per launched process; several processes on one
  connection stalls event delivery (client library limitation).
- Engine for tests: `PIE_HOOK_PREFIX_TRACE=1 nohup timeout 1800
  ./target/debug/pie local > log 2>&1 &`. The trace prints
  `[hook-prefix] R=.. fast_rows=.. fused=..` once per fire at layer 0.
- quest-attention needs `envelope_dot` (KV envelopes), capability-gated off
  in this config — snapkv/h2o are the hook-path exercisers.
