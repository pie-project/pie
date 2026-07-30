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

## Stage 5 — why the "dual derivation" of fast_rows stays

`fire_plan.rs` computes a member-count prefix; the driver's
`launch_hook_free_prefix_rows` independently derives the wire-row prefix
from the program row CSR. Handing the plan's number across the ABI was
considered and rejected: the planner's job is the ORDER (which creates the
prefix), and measuring the prefix off the wire layout it ordered is the
driver applying device-side knowledge to a device-shaped fact — exactly
the "emit alternatives, runtime picks" split of plan §4.4. The CSR
derivation also survives composition (device-geometry placeholders) that a
scheduler-side row count would have to re-model. Revisit only if a site
appears whose lowering the driver cannot derive locally.

## Stage 4 — cache_domain × adapters (why there is no digest fold yet)

The concern: `cache_domain` (runtime/engine/src/store/kv/hash.rs:30) is
per-STORE (boot nonce), adapters are per-INSTANCE, and KV produced under a
LoRA correction is not the vanilla model's KV — so two instances with
different adapters (or one with, one without) must never share prefix pages
through the semantic-hash chain. Recon of every `chain_token_slot_hash`
caller, verdict first: **adapter-carrying instances cannot reach any
prefix-sharing path in the production build today**, so an adapter digest
folded into the slot-hash domain would be unreachable machinery. Documented
here instead; one latent seam was closed (below).

Evidence, per hashing site:

- The standard PTIR fire path never creates matchable identity. It commits
  pages via `prepare_explicit_reserved` (pipeline/fire.rs:487) with EMPTY
  `token_hashes` and no page hash — nothing to index, nothing to match.
- The prefix-sharing consumer is compiled out of production:
  `KvStore::lookup_cached_page` is `#[cfg(test)]`; the `#[cfg(not(test))]`
  body returns `None` (store/kv.rs:1119), so `adopt_cached_prefix` can
  never hit, and `fire::kv::match_prefix` has no production caller
  (increment 2a, store tests only).
- The one production producer of canonical hashes is the prefill-offload
  path (offload.rs:1356 scratch prepare + `adopt_offloaded_prefix`,
  store/kv.rs:1247). Its `mutates_context` gate (offload.rs:1279) rejects
  any program with a non-Epilogue stage carrying ops — the lora prologue is
  exactly that. The rejection is semantically required, not incidental: the
  remote surrogate runs a plain context-extension program that does not
  carry the adapter, so an offloaded lora prefill would compute WRONG KV
  before hashing even enters the picture.
- No adapter identity reaches any hashing site anyway. The sites see
  `store.domain()` only; the adapter contents are per-instance channel
  seeds ("an adapter swap is a channel re-seed, never a re-trace",
  driver/cuda/src/model/lora.hpp) and are NOT in the container or the
  program hash (`ChannelDecl` carries just `seeded: bool`) — two instances
  with different adapters share one program identity.

The latent seam, closed this chunk: `canonical_kv_shape`
(pipeline/fire/kv.rs) — the unit-tested bind-time gate that increment 2a
will wire into the fire path — accepted prologue programs on the claim that
they "only shape sampling". The lora sink falsified that claim: a prologue
`SinkCall` folds an adapter delta into the q/v projections (hidden states,
hence KV at every layer). Had the gate gone live as written, a lora
instance would have hashed canonically under the bare store domain and its
KV could impersonate (or adopt) no-adapter KV. The gate now rejects a
`SinkCall` in any stage (`lora` and `minference_sparse` both perturb KV;
second-party sinks have unknown semantics and must not be presumed
canonical); sink-free prologues stay canonical. Boolean gate, unwired in
production, hash functions untouched — no existing hash moves (pinned by
the untouched chain/page-hash tests; full engine suite 369/369 green).

When sharing between SAME-adapter instances is actually wanted, the honest
shape is a derived domain — `blake3(store_domain ‖ adapter_digest)` with
the digest taken over the instance's seeded lora-channel bytes at
registration (contents, not program identity) — handed to the hashing
sites alongside `store.domain()`; no-adapter instances keep the store
domain byte-for-byte. That needs the seed bytes plumbed from instance
registration to `fire::kv`, and only pays once lora fires can pass a
(relaxed, lora-aware) canonical gate — deferred until increment 2a gives
it a consumer.

## Stage 2 — verdict

Recon result: 4 of the 5 plan claims are obsolete or were never true.

- "Multi-row fires are forced solo": false. `preserves_inner_rows` only
  solos in conjunction with an empty qo tail
  (`PendingRequest::requires_solo_submission`, worker.rs — the clause is
  `preserves_inner_rows() && qo_indptr.last() == Some(0)`); ordinary
  multi-row fires co-batch.
- "Quest can't co-batch": false today —
  `tests/inferlets/test_quest_pages.py::test_quest_two_requests` enforces
  two quest requests batching together.
- "Needs per-lane page tables": landed pre-plan in ea3c868b ("Track A/B
  L4: consume the page mask, so Quest actually evicts") — the mask is
  honoured by gathering the page table per request.
- "Fused-QKV tap blocks hooks": fixed in stage 1 (`77446890` —
  `fused_decode_qkv_post` gates on `hook_free_prefix_rows`; hook lanes run
  the unfused tail, hook-free lanes keep the fused kernel).

RS-buffer solo relax: rejected. `FoldBuffered` is unbatchable by WIT
contract (the driver rejects a batch mixing folded and forward rows), and
`Buffer` has zero callers — there is nothing to relax for.

Remaining real item A — structured device-mask co-batch. Honest cost:
packing a structured mask sets `has_custom_mask`, and the driver's fused
paths gate on `!has_custom_mask` (llama_like.cpp ~:636-663), so the WHOLE
fire loses the fused path. Measure before making it a default. Quest
enablement note: `PIE_CUDA_KV_ENVELOPES=1` at boot is the only switch.

Item B — the `is_pure_decode` hazard — is fixed at admission (this
branch): a fire carrying an `attn_page_mask`-writing program throws
mid-body if the fire is not pure decode (llama_like.cpp ~:1040), and no
scheduler predicate separated prefill from quest-class decode.
`ProgramFacts::page_mask_sink` is derived at registration,
`PendingRequest::page_mask_program` is stamped at admission, and
`LaunchGrouping::accepts` refuses (defers, order-independently) any
page-mask + multi-token mix. Hook programs without the sink (snapkv
prefill capture) stay freely mixable.
