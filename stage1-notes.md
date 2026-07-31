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

**Measured** (2026-07-31, branch head `32507463`, L40S, Qwen3-0.6B,
config as-is except `PIE_FRAME_SIZE=1` — required, see finding 1 below).

Method. Plan (a) — force `fwd_cfg.force_prefill_path` for a clean env A/B —
is unavailable: the flag is derived solely from
`flashinfer_decode_supports_gqa` (context.cpp:1414) and XQA (default on)
clears it; no env reaches it. Plan (b) instead, with a purpose-built
instrument: `tests/inferlets/naive-masked` (UNCOMMITTED measurement
inferlet, left in the tree with a workspace-member line in
tests/inferlets/Cargo.toml — delete both when this note is settled). It is
chat-completion's decode shape (all descriptor ports channel-bound,
re-put per fire) with naive-baseline's gumbel sampler and three mask
modes that differ ONLY in the mask: `none` (control), `dense` (causal
mask from iota/le — packs a dense custom mask, the custom-mask prefill
path), `structured` (the `causal_mask` opcode — the driver's
structured-mask recognizer lowers it to the window-override decode path,
`runtime_window_for_tail_aligned`, llama_like `supports_runtime_window`).
All three produce token-identical output (checked at ctx 512 and 3072).
Metric: per-token decode ms by endpoint differencing (`bench_quest.py`
discipline: wall time at max_tokens 8 vs 104, interleaved conditions,
min/median over 4-5 reps; driver script preserved at the session
scratchpad, `bench_maskpath.py`). Mixed conditions run 3 naive-baseline
lanes + 1 masked lane on separate client connections; the metric is the
PLAIN lanes' time.

Numbers (per-token ms, min/median). Debug build, ctx~512: every condition
— naive 25.0/25.2, control 24.9/24.8, dense 25.1/25.1, structured
25.3/25.3, 4×naive 24.8/24.8, mixed 24.4/24.5 — sits in one ±2% band: the
~25 ms/step debug host floor swallows the entire effect, as this file's
first measurements predicted. The decision numbers below are from
`target/release/pie` (binary in-tree, built this same day from this
branch):

    ctx~512   naive 2.79/2.58   control 2.41/2.43   dense 2.34/2.42
              struct 2.64/2.58  4xnaive 3.10/3.08   mixed(dense) 6.09/5.56
    ctx~512   (fresh server)    4xnaive 3.00/2.77   mixed(dense) 5.68/5.51
                                mixed(struct) 6.14/6.27
    ctx~3072  naive 5.49/5.53   control 5.55/5.53   dense 3.28/3.39
              struct 5.76/5.76  4xnaive 5.87/6.01
    ctx~512, PIE_CUDA_DECODE_FUSED_POST=0:
              naive 2.69/2.42   control 2.55/2.57   dense 2.56/2.64
              4xnaive 3.32/3.48 (vs 3.10/3.08 default: +7-13%, weak signal)

Reading. The fused-path loss the verdict worried about is NOT visible at
these scales: dense (custom-mask prefill path, unfused QKV) is within
noise of the unmasked control at ctx 512 and is ~40% FASTER per step at
ctx 3072 (3.3 vs 5.5 — per-fire host/planning work differs by scheduling
class and dominates; the kernels the mask path forfeits are worth tens of
µs on a 0.6B/L40S step whose release-build floor is ~2.4 ms). Killing
`fused_decode_qkv_post` outright costs at most ~0.2-0.4 ms/step at R=4
(inside a noisy band). What IS expensive and reproducible is the status
quo the relax would replace: a masked lane firing solo beside 3 plain
lanes costs the plain lanes 1.8-2.3x per token — every step becomes two
serialized waves — and the structured (window-override) mask is no
cheaper than the dense one there (6.1-6.3 vs 5.5-5.7), because the cost
is the extra wave, not the kernel.

Verdict on the verdict: on this evidence, NO — co-batch admission should
not gate on the window-override path; the window override buys nothing
measurable end-to-end here (struct ≈ naive ≈ dense at R≤4), and the
co-batch side of the ledger (one fire instead of two) is worth ~2x to the
plain lanes. Caveats before acting: R≤4 and 0.6B keep every step
host-bound — on a kernel-bound config (larger model, much wider R, longer
kv) the mask-prefill + unfused-QKV loss and the O(R x pool) dense pack
could reappear; re-measure there before defaulting co-batch on. And two
liveness seams found while measuring block the relax anyway:

1. FIXED (engine, `fire::submit_frame`). At the default `PIE_FRAME_SIZE=2`,
   EVERY dense-device-mask decode inferlet (chat-completion included) died
   at frame slot 1: the pooled device-geometry pass resolves descriptor
   ports on the host at frame prepare (`descriptor_resolve.hpp` "not
   ready"), and FramePrepare for every step of a v14 frame runs before ANY
   step reaches the stream (driver `context.cpp` `launch`) — so a
   device-geometry fire behind another fire of the same frame read cells
   whose producing fire (its own previous fire, slot 0) was not even
   enqueued. Structured mode died the same way on its OTHER ports (tokens/
   kv-len — the device-composed template covers only mask-free
   DecodeEnvelope shapes; DeviceGeometry class always host-reads at
   prepare). The fix: `submit_frame` detects a DeviceGeometry-class pass in
   any slot after the first and submits the frame's fires as single-slot
   frames (one frame per lane seals per boundary, so the producer's frame
   is on-stream before the consumer's prepare readback syncs it — the
   `PIE_FRAME_SIZE=1` shape, paid only by that lane). Verified: dense
   12/12, structured 5/5, stock chat-completion clean at default frame
   size; all modes byte-identical to `PIE_FRAME_SIZE=1`; mixed
   masked+plain (repro_maskmix) 12/12 clean at BOTH frame sizes; FS=1
   outputs byte-identical pre/post fix. Note while verifying: dense-mode
   text diverges from none/structured at ctx≈512 (repro_maskmix --parity)
   — pre-existing kernel-path numerics, present identically in this
   session's pre-fix captures and in the item A session's own
   modes_prefix/postfix_512_104.json captures, NOT a frame-size effect
   (all three modes byte-identical at the short-prompt config across both
   frame sizes).
2. Twice under mixed masked+plain load, the scheduler handed a
   dense-masked fire into a multi-program batch; the driver's v1 mask
   scope throws `RetryableLaunchError` ("dense device mask in a
   multi-program batch requires solo retry", frame.cpp:938-947) but the
   wave POISONED all lanes instead of retrying solo, and the dead
   instances leaked pages until later frames hit "exceeds the driver's
   physical budget ceiling". The same lesson as item B: the invariant
   lives in a driver throw, and the throw is reachable. Whatever admission
   policy item A lands on, this seam needs the item-B treatment first
   (scheduling decision, not launch-time refusal), plus a working solo
   retry.

Item B — the `is_pure_decode` hazard — is fixed at admission (this
branch): a fire carrying an `attn_page_mask`-writing program throws
mid-body if the fire is not pure decode (llama_like.cpp ~:1040), and no
scheduler predicate separated prefill from quest-class decode.
`ProgramFacts::page_mask_sink` is derived at registration,
`PendingRequest::page_mask_program` is stamped at admission, and
`LaunchGrouping::accepts` refuses (defers, order-independently) any
page-mask + multi-token mix. Hook programs without the sink (snapkv
prefill capture) stay freely mixable.
