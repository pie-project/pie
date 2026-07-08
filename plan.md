# Plan: Delete the PTIR beam hardwire, device-resolved geometry, multi-pass channels

Status: approved direction, not started. Date: 2026-07-07.
Scope: runtime (`runtime/src/ptir/*`), driver (`driver/cuda/*`), SDK (`sdk/rust/ptir`), ABI (`interface/driver/*`), WIT (`interface/inferlet/*` + 3 synced copies).

This document is a handover plan. It assumes no context beyond the repo; every claim
is anchored to a file/line as of commit `33d038bc` (branch `dev`, with uncommitted
work described in §2.1).

---

## 1. Goal

Remove every beam-search-specific piece of host and driver code, and make beam an
ordinary run-ahead PTIR program, by:

1. Letting the **driver resolve forward geometry from the program's descriptor-port
   channels at fire time** (today the host prefills it, and for beam the host
   *replays the epilogue arithmetic* to do so).
2. Making the **PTIR program compute wire-form geometry itself** (CSR page lists,
   dense masks) so the driver's new code is a program-agnostic 1:1 port→field mapper.
3. Making **channels first-class across forward passes**: one channel bindable to
   many passes (draft→verify chaining etc.), which requires channel identity to be
   global rather than per-program-instance.

### 1.1 Non-negotiable architectural principles (owner's constraints)

These override any implementation convenience:

- **`pipeline.submit` must never block.** Run-ahead is the most important feature.
  Fire t+1 may be submitted before fire t completes. (Already implemented for the
  non-beam path, see §2.1.)
- **The inferlet (traced PTIR program) does all heavy lifting. The driver is dumb
  and general.** No program-specific assembly, CSR construction, or mask logic in
  the executor. If a proposed driver change contains an algorithm name (e.g.
  "beam"), the design is wrong.
- **No beam-named files or symbols anywhere in `driver/cuda`** when this plan is
  done. General mechanisms extracted from beam code must be renamed to describe
  the mechanism.
- **A channel/register must NOT be owned by one forward pass.** It must be
  bindable to multiple passes.

---

## 2. Current state

### 2.1 What is already implemented (uncommitted, this working tree)

The WIT surface was redesigned from a program-registry model to first-class
objects, and the host side implements non-blocking run-ahead:

- **WIT** ([interface/inferlet/deps/core/ptir.wit](interface/inferlet/deps/core/ptir.wit),
  synced to `interface/inferlet/core/wit/`, `sdk/rust/inferlet/wit/deps/core/`,
  `sdk/tools/bakery/src/bakery/wit/deps/core/`): three resources.
  - `channel` — guest-constructed `(shape, dtype, capacity)`; `put`/`take`/`read`.
  - `forward-pass` — `new(container-bytes, channels, kv-working-sets, rs-working-sets)`;
    decode+bind+validate happens here (hash-deduped compile/bind cache).
  - `pipeline` — `constructor()`, `submit(fwd)` (non-blocking, run-ahead), `close()`.
- **Host** ([runtime/src/ptir/ptir_host.rs](runtime/src/ptir/ptir_host.rs)):
  - `submit` prepares (seed bind, host-put coalescing, KV/RS projection), calls
    `submit_async`, pushes a `PendingFire` (oneshot rx + open `PtirKvTxn`/`PtirRsTxn`
    + bound cells) onto the pass's FIFO, returns immediately. `committed_tokens`
    advances optimistically at submit so fire t+1 prepares against t's post-state.
  - `channel.take`/`read` are the await points: if the cell is empty, pop pending
    fires FIFO, await + finalize (commit/abort txns, marshal outputs into cells),
    recheck. `Empty` error only when no fire remains (a wasi-p2 guest is
    single-threaded; truly blocking would self-deadlock — real cross-task blocking
    is p3 follow-up).
  - Fire failure: poisons the pass's host-reader channels with the error string
    (first-poison-wins) and marks the pass `failed`; later submits error with the
    root cause. Under non-blocking submit, poison IS the error channel.
  - `ForwardPass::drop` drains all pending fires (pins safety).
  - RS working sets wired: `ptir_rs_prepare`/`finalize` fill
    `ForwardRequest.rs_slot_ids`/`rs_slot_flags` (RESET on first fire).
- **v1 restrictions currently enforced in `HostForwardPass::new`** (to be lifted
  by this plan): one channel binds to at most one forward-pass; exactly one
  kv-working-set; at most one rs-working-set.
- Known debt: unit-test target for the `pie` lib is blocked by a stale test module
  in [runtime/src/api/inference.rs](runtime/src/api/inference.rs) owned by a
  parallel migration (references deleted `sampling_edsl`); the SDK DSL
  (`sdk/rust/ptir`) guest bridge to the new WIT is not wired; existing test
  inferlets are intentionally NOT migrated.

### 2.2 The beam hardwire (what this plan deletes)

Beam §6.2 works today via **Design X, host replay**: the epilogue's geometry
channels are device-produced, but the driver cannot read them into the next
fire's geometry, so the host re-derives them.

Runtime side:

- [runtime/src/ptir/ptir_beam.rs](runtime/src/ptir/ptir_beam.rs) (~430 lines):
  `BeamState::step` replays the freeze/heir/page-turn arithmetic host-side from
  the harvested `out_par` channel; `BeamGeometry::{page_indptr,last_page_lens,
  live_page_slots,masks}` convert [B,P] form to wire form. Contains the golden
  vectors (`golden_charlie_fork_freeze_csrs`: `np=[3,2], pages=[5,6,7,5,6],
  klen=[9,7]`) — these are the locked contract and must survive as tests elsewhere.
- [runtime/src/ptir/ptir_host.rs](runtime/src/ptir/ptir_host.rs): `BeamRun`,
  `detect_beam` (structural: WSlot/WOff ports present), `fire_beam` (~230 lines:
  per-lane slot→physical resolution, `write_slot_shared_inplace` for heirs /
  `cow_write_slot` for forks, folds B lanes with
  `request::append_request_with_options`, fires via `submit_prebuilt_async`,
  awaits inline — **synchronous by construction**, since fire t+1's geometry
  needs t's harvested outputs), `beam_channel_u32`, placeholder prompt tokens
  `vec![1u32; b]`, eager B*P working-set alloc in `new`.

Driver side — an isolated island, **not wired into the serve path** (verified:
`beam_attention_forward` and `beam_build_csrs` have no callers outside their own
files and the two goldens; `executor.cpp` mentions beam only in comments):

- [driver/cuda/src/ops/beam_attention.{cpp,hpp}](driver/cuda/src/ops/beam_attention.hpp)
  — SEAM 1+3 orchestration (write→pack→custom-mask prefill). Its input contract
  already expects **physical** page ids and **dense** kvm bytes.
- [driver/cuda/src/ops/beam_csrs.{cpp,hpp}](driver/cuda/src/ops/beam_csrs.hpp)
  — host CSR staging helper.
- [driver/cuda/src/kernels/beam_mask_adapter.{cu,hpp}](driver/cuda/src/kernels/beam_mask_adapter.hpp)
  — dense [B, P*PAGE] kvm → FlashInfer bit-packed mask (`launch_beam_pack_kvm`).
  Signature is already lane-generic.
- [driver/cuda/src/kernels/kv_paged.cu:136](driver/cuda/src/kernels/kv_paged.cu#L136)
  `write_kv_beam_kernel` / `launch_write_kv_beam_bf16` (kv_paged.hpp:74) — writes
  each lane's new-token K/V at an **explicit** `(w_page[B], w_off[B])` descriptor,
  bypassing the standard geometry derivation. Already a general mechanism; only
  the name is beam.
- Goldens: `driver/cuda/tests/beam_csrs_test.cpp`, `tests/beam_mask_adapter_test.cu`
  (CMakeLists.txt:261-262, 279, 628-636).

### 2.3 Driver-side facts a new owner must know

1. **PTIR programs already execute on device.** `PtirDispatch::run`
   ([driver/cuda/src/ptir/ptir_dispatch.cu:65](driver/cuda/src/ptir/ptir_dispatch.cu#L65))
   decodes/caches the container (`PtirProgramCache`), keeps persistent
   per-instance state, applies host-puts, runs the trace via `Tier0Runner`,
   harvests host-reader channels into `ForwardResponse.ptir_output_*`.
   **But it is invoked only post-forward** (after logits), at
   [executor.cpp:4732-4763](driver/cuda/src/executor/executor.cpp#L4732-L4763).
   There is no pre-forward step that could source geometry from channels — that
   is the central missing mechanism.
2. **Channel storage is per-instance.** `ChannelArena`
   ([driver/cuda/src/ptir/channels.hpp:112](driver/cuda/src/ptir/channels.hpp#L112))
   is owned by `PtirInstance` (`program_runtime.hpp:106`), a single device blob
   with per-channel rings (cap `kMaxRing=8`), full/empty bits, readiness/commit
   kernels. This per-instance ownership is what forbids channel sharing across
   passes (§4.3 replaces it).
3. **The descriptor phase already exists in the runner.**
   [tier0_runner.hpp:82-95](driver/cuda/src/ptir/tier0_runner.hpp#L82-L95) handles
   §5.1 port consumption discipline (token family takes, geometry peeks) for
   readiness/commit. Only the *value* read into forward geometry is missing.
4. **The instance map leaks.** `PtirDispatch`'s `instances`
   (`ptir_dispatch.cu:18-30`) has no eviction path. A release ABI is needed
   regardless of this plan (W0.3).
5. **The standard serve path already handles the beam wire shape.** `fire_beam`
   ships a standard decode wire (physical `kv_page_indices`, `kv_last_page_lens`,
   BRLE `masks`, 1 query/beam) via `submit_prebuilt_async`; the custom-mask
   decode routing at [llama_like.cpp:343](driver/cuda/src/model/llama_like.cpp#L343)
   serves it. This is why the beam_attention island can be deleted rather than
   integrated.
6. **The wire type** is `pie_driver_abi::ForwardRequest`
   ([interface/driver/src/schema.rs:87](interface/driver/src/schema.rs#L87));
   in-proc it crosses as a `#[repr(C)]` desc aliasing runtime heap
   (`fill_forward_view`, [interface/driver/include/pie_driver_abi/view.hpp:464](interface/driver/include/pie_driver_abi/view.hpp#L464)).
   PTIR fields: `ptir_program_hashes/_instances/_bytes/_sidecar_bytes`, seed table
   `ptir_program_seed_*`, host-put table `ptir_program_host_put_*`
   (schema.rs:326-363). Outputs return in
   `ForwardResponse.ptir_output_*` (schema.rs:1466-1504), marshaled by
   `finalize_fire` → `marshal_response`
   ([runtime/src/ptir/ptir_channel_store.rs](runtime/src/ptir/ptir_channel_store.rs)).
7. **Op coverage supports program-computed wire form.** The PTIR op set
   ([interface/ptir/src/op.rs:99](interface/ptir/src/op.rs#L99)) already has
   `CumSum`, `Gather`/`GatherRow`, `ScatterSet`/`ScatterAdd`, `Iota`, `Select`,
   `Rem`, compares, `Reshape`. Result shapes must be trace-known, but scatter
   *indices* may be data-dependent — which is exactly what CSR packing needs.
   (Tier-0 *launcher* coverage for CumSum/ScatterSet/Gather must be audited: W1.2.)
8. **The reference implementation of port→geometry mapping** is host-side
   [runtime/src/ptir/ptir_geometry.rs `map_geometry`](runtime/src/ptir/ptir_geometry.rs#L74):
   ports (`EmbedTokens/EmbedIndptr/Positions/Pages/PageIndptr/KvLen/Readout`) →
   request fields, 1:1, program-agnostic. The driver's new pre-forward reader
   (W1.1) is its device mirror. Port enum:
   [interface/ptir/src/registry.rs:97](interface/ptir/src/registry.rs#L97).

---

## 3. Target contracts

### 3.1 Ports carry wire-form geometry (the program does the math)

Descriptor ports are defined to carry the forward's geometry **in final wire
form**. The program (traced by the inferlet via the SDK) computes it; the driver
only copies port values into the standard per-request fields.

- **CSR-prefix contract**: for CSR port pairs, the indptr port's last element
  defines the valid prefix length of the data port. Channels keep trace-known
  fixed shapes (e.g. `pages: [B*P]`); the program densely packs live entries at
  the front via `ScatterSet` with `CumSum`-derived destinations; the driver reads
  `page_indptr[B]` entries and ignores the rest. General rule, no beam mention.
- **AttnMask contract**: a dense u8/bool channel, one row per lane, valid span
  per lane defined by `KvLen`. BRLE remains a *host wire* compression only; a
  device-resident mask channel is consumed dense (packed to FlashInfer layout by
  a general adapter, W1.3).
- **`KvLen` = physical span**; the fixed derivation
  `last_page_len = ((len-1) % page) + 1` stays in the generic mapper (it is port
  semantics, not program policy — already the locked contract in `map_geometry`).
- `WSlot`/`WOff` = explicit KV write descriptors (physical page id + in-page
  offset per lane), consumed by the explicit write kernel (W1.4).

### 3.2 Physical-space page ids (no slot→physical table)

Geometry channels operate on **physical page ids from the start**. The epilogue's
freeze/heir arithmetic is pure gather/scatter on ids and does not care about the
id space. The runtime seeds fire 0's pages and grants per-fire fresh pages as
physical ids; the whole loop stays physical; the driver needs no translation
table. Supporting evidence: the driver side already expects physical ids
(beam_attention.hpp: "Pages/WSlot are PHYSICAL page ids (slot→physical resolved
upstream)"), and `fire_beam` already ships physical ids on the wire today.

Correctness rests on the **freeze discipline**: a frozen (shared) page is never
rewritten; forks write only into fresh granted pages; the heir continues the tail
in place. This is structural in the epilogue program. Recommended cheap guard: the
driver errors a fire whose `w_page` is outside the pass's leased set (a [B]-sized
membership check against the lease list shipped per fire or tracked per instance
— decide in W1.4; owner preference was to have the check).

### 3.3 Global channel identity (multi-pass channels)

A channel is bindable to **multiple forward passes**. Three couplings must break:

1. Host: the `already-bound` validation in `HostForwardPass::new` (v1 restriction
   — delete).
2. Host: `PendingFires` is owned by the pass and `Channel.fires` is fixed at bind
   — moves to the pipeline (§3.4).
3. Device: `ChannelArena` owned by `PtirInstance` — replaced by a **global device
   channel table**.

New model:

- The runtime mints a **global channel id** (u64, unique per runtime; inferlet
  scoped) when the WIT `channel` resource is constructed.
- `forward-pass.new` builds a binding table `dense container index → global id`
  per program; it owns nothing.
- Driver: a device-side registry `global id → DeviceChannel { ring cells,
  full/empty bits, decl }`. Allocated on first reference (decl comes with the
  first program that binds it — validate decl equality on later binds).
  `PtirInstance` holds views (index → table entry), not storage.
- ABI additions (schema.rs + view.hpp + derive):
  - per-program `ptir_program_channel_ids` (CSR over programs, parallel to the
    existing per-program tables),
  - re-key seeds / host-puts / outputs by **global channel id** instead of
    program-local dense index,
  - **release markers**: lists of channel ids and instance ids to free (fixes the
    existing instance-map leak too). Ridden on any ForwardRequest, or on a
    dedicated lightweight message if a request isn't imminent — decide in W0.3
    (riding the next request is simpler; a pass/channel dropped with no further
    fires can flush on the runtime's next heartbeat submit or on driver detach).

### 3.4 Pipeline is the ordering domain

`PendingFires` moves from `ForwardPass` to `Pipeline` (currently a stateless
struct in ptir_host.rs). Semantics:

- Fires submitted to the same pipeline are issued to the **same CUDA stream in
  submission order**. Therefore: fire t's epilogue channel puts happen-before
  fire t+1's descriptor reads — the entire run-ahead correctness argument, now
  extended across passes.
- `take`/`read` await the pending-fire FIFO of the pipeline that feeds the
  channel (the channel records the pipeline at submit).
- **v1 constraint**: all passes binding a given channel must be submitted on the
  same pipeline. Checked at `submit`; cross-pipeline sharing is an error.
- Poison unchanged: a failed fire poisons the channels it feeds; other passes
  sharing those channels observe the same poison (correct — the value they would
  consume is gone). Pass-level `failed` flag stays per-pass.
- WIT: **no signature changes.** Only the doc line "a channel already bound to
  another forward-pass (one pass per channel for now)" is removed from
  `forward-pass.new`'s doc comment; sync all 4 WIT copies.

### 3.5 Device-resolved geometry fires

For a program whose descriptor ports bind channels (device-produced geometry):

- The runtime sends the wire's `token_ids`/`positions`/`qo_indptr`/`kv_page_*`/
  `masks` **empty** and marks the fire solo (`submit_prebuilt_async` /
  scheduler `prebuilt` flag, [runtime/src/scheduler.rs:698](runtime/src/scheduler.rs#L698)
  — the batcher cannot co-batch what it cannot see). Host-known/const ports may
  still be prefilled by `map_geometry` as today.
- The driver resolves the ports pre-forward (W1.1) and feeds the **standard**
  batch/attention machinery.
- **Not-ready is an error, not a wait**: on a solo fire, if a descriptor channel
  is not full (the producing fire failed), nothing will ever fill it — the driver
  must fail the fire (not Tier0's silent dummy-run). The runtime's poison plumbing
  (§2.1) surfaces it to the guest.
- Scheduler/workspace sizing uses **static bounds from the container's channel
  decls** (e.g. nnz ≤ B from the embed channel's shape) since the host no longer
  knows exact per-fire geometry.

---

## 4. Workstreams

Dependency graph: W0 → W1 → (integrate) ← W3;  W2 ∥ (W0,W1);  W4 last.
W2 is CPU-only (no GPU needed); W0/W1 need CUDA + coordination with the driver
owner (goldens); W3 is runtime-only Rust.

### W0 — Driver + ABI: global channel table and release ABI

*Prerequisite for W1 (write the port reader against the new table from the start;
don't build it on the per-instance arena and refactor twice).*

- W0.1 Replace per-instance `ChannelArena` ownership with a device channel
  registry keyed by global channel id (§3.3). `PtirInstance` keeps a
  `dense idx → registry entry` view. Preserve the ring/bits/readiness kernel
  machinery (channels.hpp) — it moves, it doesn't change.
- W0.2 ABI: add `ptir_program_channel_ids`; re-key `ptir_program_seed_*`,
  `ptir_program_host_put_*`, `ptir_output_*` by global id. Update schema.rs,
  the `#[schema]` derive, view.hpp `fill_forward_view`, and the runtime writers
  (`PtirInstance::submission`, `drain_host_puts`, `marshal_response`).
- W0.3 Release markers for channels + instances (fixes the pre-existing
  `instances` leak). Runtime sends them from `Channel::drop` / `ForwardPass::drop`.
- Verify: existing PTIR e2e (greedy decode) unchanged; a new driver unit test:
  two instances binding the same channel id observe the same cell.

### W1 — Driver: generic pre-forward port reader; de-beam the driver tree

- W1.1 **Pre-forward descriptor resolution** (the core mechanism). Split PTIR
  handling into `resolve_descriptors(view) → FireGeometry` (pre-forward) +
  existing `PtirDispatch::run` (post-logits epilogue). For each program whose
  ports bind channels: read the port channels' current cells (v1: D2H of a few
  hundred bytes — [B]/[B,P] cells; same-stream ordering makes this correct under
  run-ahead) and fill `FireGeometry { token_ids, positions, qo_indptr,
  kv_page_indices, kv_page_indptr, kv_last_page_lens, w_page, w_off, mask }`
  applying the CSR-prefix and KvLen contracts (§3.1). Program-agnostic: it is
  the device mirror of `map_geometry` — keep the two in explicit correspondence
  (same port→field table). Feed the standard executor batch assembly
  (fwd_in.* at executor.cpp:1197-1204) instead of the wire fields.
- W1.2 **Tier-0 launcher audit**: ensure `CumSum`, `ScatterSet`, `Gather`,
  `Iota`, `Rem`, `Select` have device launchers in
  [tier0_launch.hpp](driver/cuda/src/ptir/tier0_launch.hpp) (the beam program
  needs them to compute wire form, W2). Add any missing as plain generic ops.
- W1.3 **Generalize the mask adapter**: rename `beam_mask_adapter` →
  `pack_dense_mask` (kernels/), same kernel; it is the general "dense device mask
  → FlashInfer packed" mechanism for any `AttnMask`-port program. Re-point its
  golden test.
- W1.4 **Rename the explicit KV write**: `write_kv_beam_kernel` /
  `launch_write_kv_beam_bf16` → `write_kv_explicit_*` (semantics: "write where
  the descriptor says"). Wire it to W1.1's `w_page`/`w_off`. Add the lease
  membership guard (§3.2) if adopted.
- W1.5 **Delete the beam island**: `ops/beam_attention.{cpp,hpp}`,
  `ops/beam_csrs.{cpp,hpp}`, `tests/beam_csrs_test.cpp`; strip CMakeLists
  entries. Re-target the golden *vectors* (§5) at the generic path.
- W1.6 **Not-ready → fire error** (§3.5): a solo device-geometry fire with an
  unfull descriptor channel returns a fire error carried in the normal error
  path (runtime poisons readers).
- End-state check: `grep -ri beam driver/cuda/src --include='*.cpp' ...` finds
  no *symbols/files* (incidental comments describing history are acceptable but
  prefer scrubbing). Optionally add this grep as a CI gate.

### W2 — SDK: the beam program computes wire-form geometry (parallel with W0/W1)

- W2.1 Extend the §6.2 beam epilogue trace so the arithmetic currently in
  `BeamGeometry`'s host methods is in-graph:
  - `page_indptr` = `CumSum(np)` with a leading 0 (`ScatterSet` into a `[B+1]`
    zeros + `Iota` destinations),
  - packed live pages = `ScatterSet` of the `[B,P]` matrix's live entries into a
    `[B*P]` channel at `CumSum`-derived destinations,
  - `klen`, `kvm` (dense), `w_slot`, `w_off`, `out`, `out_par` already emitted;
    bind `kvm` to the `AttnMask` port; `EmbedIndptr` stays const `[0..=B]`.
  - All ids in physical space (§3.2); the `fresh` channel receives host-granted
    physical ids.
- W2.2 Goldens: port the vectors from
  [ptir_beam.rs tests](runtime/src/ptir/ptir_beam.rs#L287) — especially
  `golden_charlie_fork_freeze_csrs` (`np=[3,2]`, `pages=[5,6,7,5,6]`,
  `klen=[9,7]`, `w_slot=[7,6]`, `w_off=[0,2]`, `w_cont=[false,true]`) and the
  continue-tail / page-turn cases — into SDK trace tests: run the extended
  program through the CPU reference interpreter and assert the emitted wire-form
  port values. These vectors ARE the contract; they must exist somewhere before
  W4 deletes ptir_beam.rs.
- W2.3 D4 compaction is **out of scope** (see Follow-ups); v1 documents the
  constraint: size P for the run's max length, or accept `np ≤ P` growth bounds
  without repeated-fork densification.

### W3 — Runtime: pipeline-owned fires, multi-pass channels, page leasing

- W3.1 **Move `PendingFires` from `ForwardPass` to `Pipeline`** (mechanical over
  the existing run-ahead implementation; semantics unchanged for the
  single-pass case). `Channel` records the feeding pipeline at submit;
  `take`/`read` await that pipeline's FIFO. Enforce the same-pipeline
  constraint at submit (§3.4). `Pipeline::drop`/`close` drains its FIFO (the
  pins-safety drain currently in `ForwardPass::drop` follows the queue).
- W3.2 **Lift the one-pass-per-channel restriction** in `HostForwardPass::new`
  (delete the already-bound check; keep decl validation per container, and keep
  duplicate-handle-within-one-pass detection). Remove the WIT doc line; sync 4
  copies. Seeds: a seeded channel's staged put is consumed by the first fire of
  whichever pass ships that channel first (seed table now keyed by global id,
  W0.2).
- W3.3 **PageLease** (per device-geometry pass): grant B physical pages at fire 0
  (seed values), B fresh per submit delivered as a host-put on the `fresh`
  channel (existing D1 coalescing path); reclaim unused grants at
  `finalize_fire` by reading the harvested `w_cont` (host-reader channel);
  reclaim everything on pass drop/failure. Pin float is bounded by
  (run-ahead depth) × B pages. Pins ride the existing per-fire arena txns.
- W3.4 **Unify submit**: delete the `detect_beam`/`fire_beam` branch. For
  device-bound ports, relax `map_geometry`'s `MissingChannelValue` gate
  ([ptir_geometry.rs:155](runtime/src/ptir/ptir_geometry.rs#L155)) to "leave the
  wire field empty"; mark the fire solo/prebuilt. Assert the FIFO invariant:
  fires of one pipeline keep submission order through the scheduler onto one
  stream (this is the whole correctness story — make it an explicit, tested
  invariant, not an accident).
- W3.5 **Shadow verify (bring-up safety net)**: keep `ptir_beam.rs` alive for
  one phase; in `finalize_fire`, debug-compare harvested geometry channels
  (pages/np/klen) against the host replay. Run N steps green on the 4090
  before W4.

### W4 — Deletion (single commit, after shadow green)

Runtime:
- `runtime/src/ptir/ptir_beam.rs` — entire file (goldens live in W2.2 by now).
- In `ptir_host.rs`: `BeamRun`, `detect_beam`, `fire_beam`, `beam_channel_u32`,
  `ForwardPass::beam`, placeholder toks (`vec![1u32; b]`), eager B*P alloc,
  the submit branch. (`submit_prebuilt_async` STAYS — it is the solo-fire carrier.)
- `KvWorkingSet::write_slot_shared_inplace` if no other caller remains.

Driver (done in W1.5 but verify nothing crept back):
- `ops/beam_attention.*`, `ops/beam_csrs.*`, `tests/beam_csrs_test.cpp`;
  `beam_mask_adapter` and `write_kv_beam` exist only under their generic names.

### Follow-ups (recorded, out of scope)

- **D4 compaction** under the same principle: the program computes the gather
  plan (densify pages when repeated forks would exceed P) and emits it on
  channels; the driver provides one generic KV-gather mechanism. Until then the
  P-bound constraint of W2.3 applies.
- Device-side geometry assembly without the D2H hop (performance).
- Co-batching device-geometry fires with other requests (v1 is solo).
- True blocking `take` across guest tasks (wasi-p3).
- SDK DSL → new WIT guest bridge; migrating the test inferlets.

---

## 5. Verification plan

1. **W0**: existing ptir-greedy e2e unchanged; shared-channel driver unit test.
2. **W1**: charlie's three beam goldens (continue-tail / page-turn / fork-freeze)
   re-targeted: seed channels on an instance, run `resolve_descriptors`, assert
   the produced geometry equals the locked vectors (source of truth in §2.2 /
   W2.2). Mask adapter golden re-pointed at the generic name.
3. **W2**: SDK trace tests assert the program's emitted wire-form port values
   equal the same vectors (CPU reference interpreter).
4. **W3**: channel-store and host unit tests extended for pipeline-owned FIFOs,
   same-pipeline enforcement, multi-pass binding, lease reclaim; then
   [bin/pie/tests/cuda_beam_e2e.rs](bin/pie/tests/cuda_beam_e2e.rs) rerouted
   through the ordinary run-ahead submit; shadow verify green for N steps.
5. **W4**: full e2e green with the replay deleted; grep gate clean.

---

## 6. Risks and open decisions

- **Stream/FIFO ordering is the entire correctness argument** for both run-ahead
  and multi-pass chaining. If the scheduler reorders a pipeline's fires or splits
  them across streams, t+1 reads stale/empty descriptor cells. W3.4's asserted
  invariant is mandatory, not advisory.
- **Freeze discipline** is assumed, not enforced, unless the W1.4 lease guard is
  adopted (recommended; one [B] membership check per fire).
- **Decl conflicts on shared channels**: two containers binding the same global
  channel id must declare identical shape/dtype/capacity; validate at bind
  (W0.1) with a clear error.
- **Ring capacity** (`kMaxRing=8`, channels.hpp:106) bounds run-ahead depth per
  channel; a deeper pipeline back-pressures at submit-prepare time (host-put
  staging) — acceptable, but document it.
- **Who reclaims shared channels**: with multi-pass binding, a channel's device
  storage frees on the WIT resource drop (release marker), not on any pass drop.
  Passes keep cells alive only through the runtime's `Arc<Mutex<ChannelCell>>`
  host mirror — device lifetime follows the resource, host mirrors follow Arc.
- **Coordination**: W0/W1 touch the driver owner's (charlie's) goldens and the
  4090 bring-up slot is the schedule bottleneck. W2 needs no GPU; start it first.
- Parallel in-flight migration (X2 bridge / `pie_sampling_ir` → `pie-ptir`,
  `forward_prepare` → `paging` rename) owns
  [runtime/src/api/inference.rs](runtime/src/api/inference.rs)'s stale test
  module; don't fix it from this workstream, and don't revert the `paging` rename.
