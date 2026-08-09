# Batch parity: `csrc/src/batch/` against `driver-metal-new`

The batch subsystem is ~11.6k lines: the forward executor (`forward.cpp`,
4.5k), the model-family glue (`simple_family.cpp`, 2k), and around them the
pure logic — scheduling, ticket composition, scratch coloring, decode
timing. Same rules as the other ledgers: every entry is **ported**,
**dropped** (with the reason the C++ needed it and the Rust does not), or
**missing** (with what blocks it). The portable half goes first.

> **A `missing` row is only true on the day it is written.**
>
> A `ported` row is written by the commit that ports the thing, so it cannot
> drift. A `missing` row is written by a *different* commit — the one that
> found the gap — and nothing brings the author back when the gap closes. On
> 2026-08-09 an audit found **five of thirteen** `missing` rows here describing
> work that was already done: the whole of `scratch.{hpp,cpp}` and
> `scratch_color.hpp` (twice, in two rows), `golden_tap.cpp`, `run_segments`,
> and `load_multibatch_psos`. Each had landed under a name the row did not
> predict — `schedule_scratch`, `batch/golden.rs`, `Stepper::run_segments`,
> `psos_mb.rs` — so no text search would have caught it either.
>
> This is the same defect as `HANDOFF.md`'s stale "What is left", in the file
> that document tells readers to trust instead. The rule that follows:
> **when a slice lands, re-read the `missing` rows that named it as their
> blocker** — a row saying "with the family port" is a note addressed to
> whoever finishes the family port, and it is their job to answer it. The
> `PARITY-M1.md` struct-zoo rows were the same shape and got the same fix.

## The batch shape — `src/batch/schedule.rs`

From `batch_schedule.hpp` (220 lines), the one file of `batch/` the C++
kept deliberately pure — and shipped without a checked build.

| C++ | Rust | |
|---|---|---|
| `BatchSchedule` | `BatchSchedule` | ported |
| `RequestSpan` | `RequestSpan` | ported |
| `build_batch_schedule` | `build_schedule` | ported |
| `find_request` | `find_request` | ported |
| `validate_paged_batch` | `validate_paged` | ported |
| `validate_paged_batch_capacity` | `validate_capacity` | ported |
| `BatchSchedule::m1` | `BatchSchedule::single` | ported |
| `kRsFlagReset` | `driver_abi::local::PIE_RS_FLAG_RESET` | dropped |
| `BatchStepInputs` | `batch::StepInputs` | ported |
| the fleet concatenation (`forward.cpp` 3844-3930) | `batch::concat_fleet` | ported; request-local indices rebased onto the fire, both CSRs closed, the mask materialised only when a member carries one |
| `run_paged_batch_forward`'s per-member admission + span derivation | `batch::marshal_fleet` / `plan_member` → `RequestPlan` | ported; the portable half of the marshaling |

The build/validate split survives — the geometry gate needs fire-time
arrays the build does not — but the build stops trusting its inputs:
`qo_hi - qo_lo` and `seqlen - new_tokens` were unchecked `u32` subtractions,
so a non-ascending `qo_indptr` or a span longer than its sequence produced
*wrapped* spans, and whether anyone noticed depended on whether that caller
also ran the validator. `build_schedule` refuses at construction, naming the
request; a `BatchSchedule` that exists has coherent spans.

`find_request` answered an out-of-range token with `R - 1` — a wrong request
shaped like a right one — and answers `None` here. `page_size <= 0` silently
became 32 inside the build; the default is now the caller's to take
(`DEFAULT_PAGE_SIZE`), not the build's to impose. The validator's
`bool` + static-string answers become `Rejected`, which names the request or
token at fault. `kRsFlagReset` was a hand copy of `PIE_RS_FLAG_RESET`,
"duplicated rather than included" for a test harness this crate does not
need; the masked read the C++ comment insists on (FOLD is not RESET) is kept
and tested.

Ten tests, portable, including the write-descriptor formula held exactly and
both wrap refusals.

### `close_linear_sequence` cannot close what `copy_state` created

The C++ clears a slot only when

```cpp
state.has_resident && (state.ring_backed || state.paged_backed) &&
    state.resident_sequence_id == sequence_id
```

and `copy_state` produces precisely the excluded state: it copies a source
slot's bookkeeping to a destination and sets `ring_backed = false`, so a copy
taken from a **ring-backed** source has neither flag while `has_resident` stays
true. That slot can then never be closed. Its metadata outlives every
`close_sequence` the caller issues, and the entry becomes a permanent resident
in the table the close exists to keep clean.

The backing describes *how* a sequence is resident, not *whether* it is, so it
has no business in the predicate. `close_sequence` matches on the id alone.

## The wire mask — `src/batch/mask.rs`

From `wire_mask.hpp` (142 lines): whether a wire attention mask says
anything the kernel's own causal predicate does not already enforce.

| C++ | Rust | |
|---|---|---|
| `row_is_prefix` | `row_prefix` (private) | ported |
| `causal_prefix_lengths` | `causal_prefix_lengths` | ported |
| `first_kv_len_disagreement` | `kv_len_disagreement` | ported |

The semantics are kept exactly — a causal-prefix mask can be *dropped*, and
the answer is bit-identical because the kernel's predicate and the mask's
are then the same predicate; anything else refuses. What changes is the
plumbing: the C++ restates the CSR KV-length formula for the third time and
indexes `kv_page_indptr[r + 1]` / `kv_last_page_lens[r]` with no length
check anywhere — a mask table describing more requests than the CSR carries
is an out-of-bounds read. The Rust comparison takes the schedule's own
`RequestSpan`s (checked at construction, one owner for the formula), and a
mask describing more requests than the schedule is itself the first
disagreement. The `int` + out-param answer becomes `Disagreement`, carrying
the request and both numbers.

Eight portable tests, including multi-word rows, the sink and window
refusals, and the classic PAGE_T-16-vs-32 mismatch.

## The admission gate — `src/batch/admit.rs`

The first half of `compose.cpp`: the recurrent-state shapes this driver
refuses to run, extracted from `build_launch_view`'s refusal battery.

| C++ | Rust | |
|---|---|---|
| the RS refusals in `build_launch_view` | `admit_recurrent` | ported |
| `build_launch_view` (the slice wrapping) | — | dropped |
| `OwnedLaunchView::capture` | — | dropped |
| `OwnedLaunchView::view` | — | dropped |
| `kRsFlagFold` / `kRsFlagBufferWrite` / `kRsFlagFoldLenDevice` / `kRsFlagReset` | `driver_abi::local::PIE_RS_FLAG_*` | dropped |

The refusals are the file's real content and survive exactly — buffered
replay, device-resident fold lengths, mid-page buffer heads, a fold
boundary inside a fire's own tokens, a fold row with no tokens, mixed
persistence — each one a shape that fails *quietly* if admitted, corrupting
a recurrent state that cannot be recovered once folded. What changes is
their identity: the C++ throws `std::runtime_error` with prose as the only
discriminator, conflating "your launch is malformed" with "this driver
lacks the capability". `Refused` names each decision, keeps the C++'s prose
as `reason()`, carries the row at fault, and `is_malformed` answers the
question the exception type could not. A descending token CSR is refused as
malformed where the C++ wrapped the subtraction.

The wrapping half is dropped because its reason to exist is the C ABI: the
engine hands the C++ borrowed slices that die at return, so every launch was
re-wrapped (`build_launch_view`), deep-copied (`capture`) and re-wrapped
again (`view`). The Rust engine hands the driver an owned
`driver_abi::plan::LaunchPlan`; there is nothing to capture. The four
`kRsFlag*` hand copies fall to the ABI crate's own constants.

Nine portable tests, one per refusal plus the whole-row/no-fold admissions
and the malformed-rows fail-closed cases.

## The member description — `src/batch/member.rs`

The second half of `compose.cpp`: `build_member_forward_desc` and the type
it fills.

| C++ | Rust | |
|---|---|---|
| `MemberForwardDesc` | `ForwardDesc` | ported |
| `build_member_forward_desc` | `build_member_desc` | ported |
| `FireGeometry` (the consumed subset) | `ResolvedGeometry` | ported |
| `has_rs_slot` / `rs_slot_id` / `rs_reset` | `ForwardDesc::rs_slot` | dropped |
| `kv_last_page_len = 0` as "derive later" | `Option<u32>` | dropped |
| `StructuredMaskDescriptor` (here) | the `structured_mask` bit | dropped |

The RS dual-indexing fix — the shipped bug where the launch-wide form was
read from index 0, giving member 1 member 0's slot and barring two decodes
from ever sharing a forward — is kept with both forms matched explicitly
and a test that member 1 gets its own slot. The `bool` + out-param + error
string becomes `Result<ForwardDesc, BuildError>` with the C++'s own words
as `reason()`. The three-deep nested ternary deriving the final page fill
is `derive_key_len`, named and documented; the member-level RS triple is
derived from the per-request vectors instead of stored beside them; a zero
page size is refused where the C++ silently clamped it to one.

`ForwardDesc::extents`/`extents_from_readout` are
`m1_extents_from_forward_desc`/`m3_extents_from_forward_desc` — the two
entries `PARITY-M1.md` carried as missing — as methods on the type that
owns the fields.

Nine portable tests: wire slicing for both members, the pageless and
derived-fill paths, both RS forms and the dual-indexing regression, mask
shape and structured-mask refusals, malformed spans named, and the extents
round-trip.

## The activation colouring — `src/batch/color.rs`

From `scratch_color.hpp` (111 lines): the linear-scan colouring of
activation live ranges onto the ping-pong pool, extracted by the C++ "so a
second model family does not arrive with a second copy that drifts".

| C++ | Rust | |
|---|---|---|
| `Use` | `Use` | ported |
| `Coloring` | `Coloring` | ported |
| `color_live_ranges` | `color_live_ranges` | ported |
| `Coloring::hazard_free` | `ColoringError::HazardDetected` | dropped |

The algorithm survives byte for byte — first-use order, strictly-before
reuse, the concurrency-run extension, inclusive overlap. What does not: an
unused value took a fresh colour anyway (its `def = last = -1` never
matched `free_at < def`), inflating the count the scratch region is sized
by; it colours to `None` and costs nothing. An ordinal past the run table
silently skipped the extension — the one case the barrier-free-run rule
exists for — and is refused as malformed. And `hazard_free` was a
self-check behind a bool the caller must remember to read; the check now
runs unconditionally and a detected hazard is an `Err` that cannot be
ignored.

Seven portable tests: the ping-pong chain, same-dispatch write-after-read,
the run extension separating what solo ordinals may share, `no_recycle`,
unused values costing nothing, the refusals, and def-order robustness.

## Not yet started

The independent leaves of `batch/` are done; everything below hangs off
shared vocabulary. The porting order that follows from the includes:

1. **`decode_abi.hpp` (650) is the trunk** — Region/IoSlot enums, the ~30
   `bind::` argument-table layouts, `ArgmaxParams`, the `Kernel` kind enum
   and `ForwardGraphKey`. Pure ("NO Metal/ObjC — every lane includes it
   without a Metal dependency") and next. Its own best argument is the
   `KindCount` story at the bottom: the count used to be spelled
   `G4PleResidual + 1`, forty kinds short, so `psos[LmHeadUntied]` indexed
   past the array and the untied head ran the wrong pipeline — every logit
   zero, every token 0, and not one error anywhere. The enum's numeric
   values are ABI ("APPEND ONLY" ×5 in the comments), so the Rust port owes
   discriminant-pinning tests.
2. `decode_timing` (365) and the scratch schedule (`scratch.hpp/.cpp`,
   ~540) consume `Kernel`/`Dispatch`/`DecodeGeometry`; `Dispatch` and the
   geometry live under `model/<family>/` and land with the family port.
   (`DecodeGeometry` itself has since landed — see the geometry section.)
3. `expert_paging.hpp` (195): ported -- see "Expert paging" below.
4. `worker.hpp` (171) — ported (`worker.rs`); `simple_family` (2176)
   remains, and `forward.cpp/.hpp` (5393) is now PARTIALLY covered by
   `src/metal/decoder.rs` — see "The multibatch step and the decode
   runner" below for what is ported and what still is not.

## The multibatch step and the decode runner

`decode_step_mb.cpp` (1179) is fully ported across four modules, and the
first arm of `forward.cpp`'s orchestration exists as `metal/decoder.rs`.

| C++ | Rust | |
|---|---|---|
| `build_decode_dag_mb` / `build_decode_prefill_dags` / `mb_kind` / `mb_geometry` + `decode_dispatch_mb.hpp`'s decisions | `batch/dispatch_mb.rs` | ported; split-K OFF and the routed decode GEMM SHUT, both with their numbers |
| `bind_decode_dag_mb` / `bind_gdn_conv_parity` / `MbBindOffsets` | `metal/bind_mb.rs` | ported; per-row offsets are `gpu_address + offset`, one number per table entry |
| `mb_pso` selection ladder | `metal/step_mb.rs::mb_pso` | ported; FP16 staging (`bind_mb_fp16_qmm`) deferred — mb_pso is always told FP16 is unavailable until the staging pair lands |
| slotted GDN / paged KV / paged SDPA const arms | `metal/bind.rs` | ported — found by the paged bisect after being omitted (an unbound constant is zero output, not a fault) |
| `ab_*` A/B levers, `PIE_METAL_PAGING_TRACE` | — | dropped: crate policy denies prints; the smoke's env levers are the equivalents |
| `forward.cpp`: fire loop, page/slot assignment, conv orientation | `metal/decoder.rs` | ported in first form: per-width step cache, per-token prefill streams, fleet fires, per-slot orientation with the join copy |
| `ensure_elastic_storage`'s sizing arithmetic | `batch::sizing::{kv_pool_target_bytes, ring_target_bytes, conv_state_target_bytes, recurrent_state_target_bytes, row_scaled_target_bytes}` + `Target` | ported; the clamp is a value, not a silent `min` |
| `kv_pool_row_bytes`, and its inline second spelling in `ensure_elastic_storage` | `batch::sizing::kv_pool_row_bytes` | ported, one spelling |
| `ensure_elastic_buffers_atomically` (the commit itself) | `metal::Elastic` | ported |
| `logits_convert.hpp` (30) | `batch/logits.rs` | ported; one spelling of the widening, which `batch/golden.rs` now calls instead of carrying its own |
| the EOS substrate: `ArgmaxParams` (vocab + `n_eos` + 8 ids), the `eos_flag` buffer, `bind::Argmax::{Params, EosFlag}` | `batch::ArgmaxParams` (size-asserted at 40), `metal::storage`'s `eos_flag`, the family bind walks, `Decoder::greedy` | ported |
| the "resident loop" the C++ comments describe | — | **there is nothing to port**: `heap_bind.cpp` sets `n_eos = 0` and says "executor/resident loop rewrites vocab+eos per gen", and `mtl4_context.hpp` calls it the shippable fix it does not have. It is an aspiration in comments, not code |
| `LinearSequenceState` | `batch::SequenceState` + `Backing` | ported; the two `*_backed` bools become one enum, because a slot is ring-backed **or** paged **or** neither and the pair could spell a fourth thing |
| `validate_linear_sequence_geometry` | `batch::validate_continuation` | ported; the nine refusals kept one for one, each a fire that would otherwise answer with someone else's history |
| `close_linear_sequence` | `batch::close_sequence` | ported, defect fixed — see below |
| `request_rs_binding` | `batch::rs_binding` | ported; the legacy member-level arm is dropped — `ForwardDesc::rs_slot` already derives that view from the per-request vectors, so there is no arm to pick |
| `validate_paged_request_state` | `batch::validate_paged_continuation` | ported; the five not-matching cases stay five answers, for the reason the C++ gives |
| `commit_paged_request_state` | `batch::commit_paged` | ported, defect fixed: the C++ is `void` and returns silently on a guard the validation should have made unreachable — but if it fires, the fire has already run and the slot keeps pre-fire state, so the NEXT continuation validates against a position the device moved past and passes |
| `rs_slot_bytes_for` | `batch::rs_slot_bytes` | ported; the conv stride counts twice because the state is a ping-pong pair |
| `rs_slot_budget_bytes` | `batch::rs_slot_budget_bytes` + `RS_SLOT_BUDGET_FLOOR` | ported |
| `rs_slots_for_budget` | `batch::rs_slots_for_budget` + `MAX_RS_SLOTS` | ported, with the shipped bug it records: `requested` was applied as `max(slots, requested)`, which made the budget decorative — 64 slots at 170 MiB each is 10.6 GiB, 5.2 GiB over the device, and the OOM arrived as a command buffer that never ran so every PTIR lane read its own zero fill back as a fault |
| `fits_on_this_gpu`'s decision | `batch::fits` → `Fit` | ported; the two bounds stay two answers, and a device-bound footprint is reported as such even when the machine is also short |
| its refusal prose, incl. the `ElasticBreakdown` advice | `batch::Refusal`'s `Display` | ported; the numbers are the answer and the sentence is a rendering of them, so a caller that only decides never formats |
| `warn_once_if_the_gpu_leaked_memory_before_this_run` | — | dropped: a `std::call_once` warning printed from inside the fit check. The fit is a function of its inputs; a process-global latch is not one of them |
| `forward.cpp`: `copy_state`/reset ABI arms, logits views, PTIR hooks, timing attribution wiring | — | missing: the engine-facing surface; lands with the cutover wiring |

Verified on device (Qwen3.6-27B, `tests/device_smoke.rs`): the M=1 ring
decode, the paged sequential and per-row-stream prefills, the equal
fleet, and the mixed-length fleet all reproduce one token-exact
reference; the bisect instruments (`PIE_SMOKE_SEQ_PREFILL`,
`PIE_SMOKE_PAGED_TAPS`, the per-step KV/sdpa host checks) stay in the
test as the accuracy gate's tooling.

## The multibatch PSO plan — `src/batch/psos_mb.rs`

The portable half of `load_multibatch_psos` (`decode_psos.cpp`, the M>1
side of the 582): feature gating and the name grammar, emitted as
`MbRequest { slot, file, entry }` for the device half to batch-compile.

| C++ | Rust | |
|---|---|---|
| `MultiBatchPsoFeatures` (10 flags) | `MbFeatures` | ported, each flag's absence documented |
| `MultiBatchPsos` field table | `MbSlot` (39 variants, rung indices as payload) | ported: the C++ routes by pointer into a struct; the plan names the slot instead |
| `pie::kernels::entrypoint()` grammar | grammar inline + dev test | ported: names are products; existence is pinned against `kernels_metal::entrypoints()` on any host, as for the M=1 plan |
| `kQmmBMs` static_assert | `QMM_BMS` + rung-indexed slots | ported |
| batch-compile-per-file argument | module doc + `file` on each request | ported as a fact about the list |
| `compile_psos_from_files` + fill loop | — | device half; lands with the Metal PSO table |

Arguments preserved: `quant` has no default (half-supplied formats
compiled, bound, dispatched, and answered wrongly); the routed GEMM's
FP16 form is a NAME choice decided by tuning, identical in contract, and
llama — whose routed top-k moved under FP16 — never reaches this table;
the wide matvec follows the CHECKPOINT's format rather than the fp16
gate, because tying it to `fp16_strided` left an alt-quant kind with no
batched shape at all (pinned by test: g64/b8 still gets one).

## The golden taps — `src/batch/golden.rs`

`batch/golden_tap.hpp/.cpp` (316 together): the env-gated activation dump
the accuracy gate diffs against the MLX reference, tap by tap.

| C++ | Rust | |
|---|---|---|
| `tap_for` (the kernel→name/bind/width table) | `tap_for` | ported, with its three stories: GatedRms is `gdn_core` (the reference taps after the gate RMSNorm); routed SiluMul is `shared_act` at its own width (the `swiglu` tap was present-but-empty on every routed checkpoint); the mixture taps exist because the routed FFN was the one block no parity run could see |
| `Dispatch` parameter | `TapSite { kind, layer }` | ported narrower: tapping reads two fields, so it asks for two — the family `Dispatch` needn't exist yet |
| `write_npy` (silent on failed open) | `write_npy -> io::Result` | ported; the silence is deleted, not preserved — a "successful" dump that left nothing behind cost a bisect once, which is also why `dir_from_env` still mkdirs |
| `golden_tap_dir()` static | `dir_from_env()` | ported; the create failure is returned, not swallowed |
| `golden_taps_recycle` | `taps_recycle` | ported with its argument (the one defect class the no-recycle dump cannot see) |
| `dump_golden_taps` | `dump_taps` (unsafe, over `[R: Region]`) | ported; the last-writer-of-a-colour rule and its reason (in-place rewriters would publish the later tensor under the earlier name) pinned by test |
| `dump_golden_bf16` / `_sorted` / `_tokens` | `dump_bf16` / `dump_bf16_sorted` / `dump_tokens` | ported over slices; the sorted dump's perm-not-recomputed argument kept (within-expert order is atomics-decided) |

The C++ silently skipped a row whose read left the pool slot; the Rust
zero-fills it after a named bounds check — a dump is diagnostic, but a
short slot is now visible in the data instead of shortening it.

## The geometry — `src/batch/geometry.rs`

`DecodeGeometry` (`model/qwen3_5/geometry.hpp`, generic despite its path)
and `AffineFormat` (declared beside the quant kernels, shared with them).

| C++ | Rust | |
|---|---|---|
| `AffineFormat` / `kernel_suffix` | `AffineFormat` / `kernel_suffix` | ported |
| `DecodeGeometry` fields + defaults | `DecodeGeometry` / `Default` | ported |
| `is_full_attn` / `full_attn_layers` | same, interval<=1 semantics kept | ported |
| `gdn_conv_stride_bytes` / `gdn_recurrent_stride_bytes` | same | ported |
| `is_moe` / `has_shared_expert` / `ffn_width` | same | ported |
| `has_alt_quant` | `AffineFormat::is_set` + `has_alt_quant` | ported |

The stories stay on the fields that carry them: the affine width and
group are one fact (g64/b8 and g128/b4 pack identically; a pipeline built
for the wrong pair "compiled, bound, dispatched, and lied" — token 3504,
repeated), and `alt_quant` exists because mlx_lm spares the two routing
projections at 8 bits inside a 4-bit body — read as 4-bit they route to
almost the right experts (cosine 0.84) and weight them wrongly.

The deferral recorded here previously is closed: the scratch footprints
landed as `src/batch/sizing.rs` (below) and `plan_heap` as
`src/loader/heap.rs` (see `PARITY-LOADER.md`). `Tuning::moe_tile_rows`
already existed in `src/tuning.rs`; only the kernel bound needed porting.

## Scratch sizing — `src/batch/sizing.rs`

The sizing half of `batch/scratch.hpp`, plus the two helpers it reaches:
`pie::kernels::moe::sorted_rows` (`kernels-metal/include/pie/kernels/
moe.h`) and `moe_sorted_rows` (`model/qwen3_5/decode_step.hpp`, generic
despite its path).

| C++ | Rust | |
|---|---|---|
| `moe::sorted_rows(pairs, experts, tile)` | `sorted_rows` | ported |
| `moe_sorted_rows(g, n_tokens, batched)` | `moe_sorted_rows(g, tuning, n, RoutedProjection)` | ported |
| `shared_kernels::moe_sorted_rows(pairs, experts)` | — | dropped: it only glued the tile lookup to the bound; the two calls read better than a third name |
| `scratch_widest_elems(g)` | `scratch_widest_elems(g, tuning)` | ported |
| `scratch_slot_elems(g, max_tokens)` | `scratch_slot_elems(g, tuning, max_tokens)` | ported |
| `batched: bool` parameter | `RoutedProjection { Matmul, Matvec }` | ported: a bare `true` at a call site says nothing; the enum says which projection runs |
| `ScratchBind` / `ScratchDispatch` / the schedule | `ScratchBind` / `ScratchSchedule` / `schedule_scratch` (`color.rs`) | ported — see below |

The arguments preserved: the slot width was derived twice (here and in the
heap layout) and the copies drifted — 8320 elements against a 16384-element
row pitch, every row past the halfway point writing into the next colour —
so the second derivation is deleted, not synchronized. The mixture's stack
does not scale linearly with tokens (12800 rows where the linear bound says
5120, pinned by test), and `sorted_rows` takes its tile as a parameter
because a bound the kernel guarantees must not depend on a tuning decision.

The kernel bound has no Rust home on the kernels side yet: `kernels-metal`
is a signature table. When it grows launch-shape helpers, `sorted_rows`
should migrate there; until then the doc names its authority
(`moe_route.metal`'s sort).

`model/family_coloring.hpp` (72) folds into `color.rs` as
`schedule_scratch`: the C++ adapter was a template because every family
declared its own `Use` struct with identical fields, and widening them was
half its body. Rust families produce the shared `Use` directly, so what
remains is the fan-out from the per-value colouring to per-dispatch bind
lists — plus the argument for `color_of_value` travelling alongside
(sizing needs the widest VALUE sharing a colour; a routed expert stack is
`experts_per_token` times taller than the dense tensor beside it, and no
bind index shows that). Its `hazard_free` flag stays unrepresentable: a
hazard is an `Err` through the adapter too, pinned by test. A use whose
ordinal is past the DAG is refused (`OrdinalPastDag`) where the C++
indexed the table with it.

## Expert paging — `src/batch/paging.rs`

The portable half of `batch/expert_paging.hpp` (195), now that
`ExpertSlab` exists (`src/loader/slab.rs`).

| C++ | Rust | |
|---|---|---|
| `ExpertPaging::plan` | `plan_paging(cuts, dag_size, SlabShape, …)` | ported; five refusals become four named `PagingRefused` variants |
| ids-not-host-readable refusal | — | stays with the device half: a `SlotHandle`'s readability is a Metal fact |
| the in-place id rewrite inside `fire` | `renumber_routing` | ported; takes the slab's `ensure_resident` as a closure, so the rewrite is tested without a device |
| `fire`'s segment loop / `run_segments` | `metal::Stepper::run_segments` | ported; the pins-back-FIRST rule is stated in the budget module's docs because it is a budget fact, not a queue fact |
| `PIE_METAL_PAGING_TRACE` stderr dump | — | dropped: the crate denies `print_stderr`; a caller that wants the trace logs the buffer it owns |
| `[pie-metal] … experts paged through …` banner | `PagingPlan::worst_case_experts` + slab accessors | dropped as a print; the numbers it printed are readable off the plan and the slab |

Arguments preserved: the worst case is every expert ONE dispatch can read
(`min(n_experts, rows × experts_per_token)`) resident at once -- there is
no order in which a smaller cache could serve it, which is also why the
slab never needs to exceed one layer's bank. The strided-vs-packed story
is pinned by test: reading a strided prefill as packed renumbers row 0
`rows` times and leaves the rest holding true expert ids -- fluent wrong
text, not an error. `renumber_routing` refuses a short buffer before
touching a byte, because a partial rewrite is a state nobody asked for.

## The decode-step ABI — `src/batch/abi.rs`

The vocabulary half of `decode_abi.hpp`: regions, IO slots, the kernel-kind
enum, the argmax params and the graph key.

| C++ | Rust | |
|---|---|---|
| `Region` | `Region` | ported |
| `SCRATCH_POOL` | `SCRATCH_POOL` | ported |
| `IoSlot` / `kIoSlotCount` | `IoSlot` / `IO_SLOT_COUNT` | ported |
| `ArgmaxParams` | `ArgmaxParams` (+ size pin) | ported |
| `Kernel` (98 kinds) | `Kernel`, macro-derived | ported |
| `Kernel::KindCount` | `Kernel::COUNT` via `Kernel::ALL` | ported |
| `ForwardGraphKey` / `PAGE_BUCKET_GRAN` | `ForwardGraphKey::of` | ported |
| the 36 `bind::` layouts (`decode_abi.hpp` 123-475) | `metal::bind::slot`, `bind_mb::slot_mb`, `gptoss_bind::slot` and the family bind walks | ported — audited 2026-08-09, see below |

The argument is the count that was forty kinds short: `KindCount` was once
spelled `G4PleResidual + 1`, so `psos[LmHeadUntied]` indexed past every
kind-sized table — the untied head ran the wrong pipeline, every logit
zero, every token 0, not one error anywhere. The C++ fix made the count an
enum member; the Rust fix derives `ALL` and `COUNT` from the same token
list the variants come from, so there is no second spelling of the end to
fall behind, and a `[T; COUNT]` table indexed through a `Kernel` value
cannot be indexed past. The numeric values are ABI ("APPEND ONLY" five
times over), so eleven anchor discriminants pin every block boundary — an
insertion upstream of one fails loudly instead of renumbering silently.

| C++ | lines | |
|---|---|---|
| `ChannelTicket` + `kNoChannelTicket` | `batch::ChannelTicket` + `pipeline::NO_TICKET` | ported |
| the ticket composition loop (`context.cpp` 1061-1092) | `batch::compose_tickets` | ported; the pin rules (a take pins its head, a put pins its tail) refused at composition, where the member and dense index are still in hand |
| `LaunchMember::{instance_id, needs_forward, requires_m2, mtp_draft_row, fwd_slot, build_err}` | — | missing: per-member launch state, with the forward port |
| `LaunchMember::terminal_cell` / `LaunchJobData::{completion, lease_id}` | — | missing: the completion broker, with the engine seam |
| `LaunchJobData` | — | missing: the async job container — `worker_.post`, the owned deep copy, the settle/publish walk — with the worker port |
| `scratch.hpp` / `scratch.cpp`: `build_scratch_schedule`, `bind_scratch`, the footprint helpers | `batch::schedule_scratch` + `metal::bind_scratch` + `batch::sizing` | ported — see the scratch row below |
| — | — | — (`decode_timing` ported below) |

## The attribution — `src/batch/timing.rs`

From `decode_timing.cpp/.hpp` (365 lines), plus the kind names moved to
where the kinds live.

| C++ | Rust | |
|---|---|---|
| `attribute_step` | `attribute_step` | ported |
| `StepAttribution` / `DispatchAttribution` | same names | ported |
| `StepAttribution::valid` | `Result` + `BoundaryMismatch` | dropped |
| `kernel_name` | `Kernel::name`, macro-total | ported |
| `kernel_ablated` | `Ablation::parse` / `from_env` / `ablated` | ported |
| `print_attribution` | `StepAttribution::report` | ported |
| the `Dispatch` subset it reads | `DispatchInfo` | ported |

`kernel_name` was a hand-kept switch with `default: return "unknown"`, and
for a while 50 of the 99 kinds fell through it — the attribution report was
blind to gemma4's mixture and PLE, all of gpt-oss and both untied kinds at
once, and the ablation knob could not name any of them. The name is now an
argument of the `kernels!` macro: a variant without one does not compile,
and `from_name` is its exact inverse (tested unique and total). The one
legacy exception, `AttnGate = "gate"`, is pinned.

The ablation spec parser keeps its hard-won lesson — a typo'd token
"ablates NOTHING and this run reports the baseline while looking armed" —
but returns the unmatched tokens instead of printing them (`parse` is
env-free and testable; `from_env` is the one place the environment is
read). The substring-with-boundary-checks walk becomes an exact per-token
lookup. `valid = false` becomes an `Err` carrying both counts, and the
`FILE*` report becomes a returned `String`, because this crate denies
stdout/stderr by policy.

Five timing tests plus the abi name test; the monotonic guard (a clock
wrap attributes zero, not a negative share) is kept and tested.
| `expert_paging.hpp` (195): `ExpertPaging::plan` | `batch::plan_paging` → `PagingPlan` | ported; a rejected plan is not a value — the C++ assigns `slab_`, `cuts_` and both counts BEFORE validating, so a `plan` that returns false leaves `cuts_` non-empty and `active()` answering **true**, and a caller that trusts `active()` fires a shape that was refused |
| `ExpertPaging::fire` | `metal::fire_paged` | ported; drives `Stepper::run_segments`, pins back before the page-in |
| the in-place id rewrite | `batch::renumber_routing` | ported; refuses `ShortIds` before touching a byte, where the C++ walks `rows` rows at a trusted stride with no bound on the buffer |
| `ids_row_stride_bytes = 0` (defaulted `size_t`) | `IdsLayout` | ported; the packed/strided distinction the C++ spends 22 comment lines on — "fluent wrong text rather than an error" — carried in a named struct beside the two counts it has to agree with, rather than in a defaulted integer |
| `Cut::ids` (a handle per cut, tail cut holds a null one) | `fire_paged`'s `ids: &[Handle]` against `PagingPlan::mixture_layers` | ported; "the tail has no ids" becomes a length that cannot be wrong |
| the `fprintf` on the success path | — | dropped: a successful plan printed a line to stderr unconditionally |
| `scratch.cpp` / `scratch.hpp` / `scratch_color.hpp` (650) | `batch/color.rs` + `batch/sizing.rs` + `metal/bind.rs` | ported: `Use`/`Coloring`/`color_live_ranges` and `build_scratch_schedule` → `schedule_scratch`; `scratch_widest_elems`/`scratch_slot_elems` → `batch::sizing`; `bind_scratch` → `metal::bind`. `ScratchDispatch` is dropped — the C++'s per-dispatch struct is one row of `ScratchSchedule::per_dispatch`, and a struct holding one vector is a name for a row |
| `batch_schedule.hpp` (done above) | — | — |
| — | — | — (`decode_psos`'s M=1 half ported below) |

## The PSO plan — `src/batch/psos.rs`

The pure half of `decode_psos.cpp`: which `(file, entrypoint)` pairs a
configuration compiles and which kinds each serves. The metal half is
`Compiler::compile_batch`, which already exists.

| C++ | Rust | |
|---|---|---|
| `PsoSpec` / the `want` gathering | `PsoRequest` / `plan_decode_psos` | ported |
| `DecodePsoFeatures` | `Features` | ported |
| the format-dependent `entrypoint()` names | `EntryNames`, table-checked | ported |
| `DecodeStepPsos` fan-out | `DecodePsoPlan::source_of` | ported |
| `load_decode_psos` (the compile loop) | `Compiler::compile_batch` + `source_of` | dropped |
| `load_multibatch_psos` / `MultiBatchPsos` | `batch::plan_multibatch_psos` + `MbRequest` / `MbFeatures` (`psos_mb.rs`) | ported |

The C++ `entrypoint()` refuses a name no shader instantiates, so an
uninstantiated format fails at load naming the formats that exist instead
of inside the Metal compiler. The plan holds the same line one step
earlier and on any host: a dev test validates every emittable name against
`kernels-metal`'s signature table (a new `default-features = false`
dev-dependency — the table, no Metal) and every file path against the
shipped tree. Each feature flag keeps its load-bearing absence documented
(`untied` once handed llama a wrong-format pipeline that answered wrongly;
`routed` would let an unrelated shader error fail a dense load), fan-out
order is later-claims-win (the GDN recurrent override), and `routing_only`
still clears the world down to the two second-format projections.

Five portable tests: the 25-kind base surface, the full feature set with
single-claim disjointness, the override ordering, `routing_only`, and the
signature-table validation.
| `golden_tap.cpp` (238) | `batch/golden.rs` | ported function for function: `tap_for`, `golden_tap_dir` → `dir_from_env`, `golden_taps_recycle` → `taps_recycle`, `write_npy`, `dump_golden_bf16*` → `dump_bf16*`, `dump_golden_tokens` → `dump_tokens`. Same file names and shapes as the MLX reference, so the two trees diff tap by tap |
| — | — | — (`worker.hpp` ported below) |

## The executor worker — `src/batch/worker.rs`

| C++ | Rust | |
|---|---|---|
| `ExecutorWorker` | `Worker<S>` | ported |
| `run` / `post` / `drain` | `run` / `post` / `drain` | ported |
| `submitted` | `submitted` | ported |
| the same-thread inline re-entry | a refusal with instructions | dropped |
| `worker_thread_id` | — | dropped |

The C++ serializes every executor touch through one FIFO thread — the
thread-affinity Metal requires and the forward/control-op exclusion in one
mechanism — but the guarantee holds only *as long as everyone remembers to
go through the worker*: the context pointer stays reachable from any
thread. `Worker<S>` closes that by ownership: the state is constructed ON
the worker thread by a factory and never leaves it, jobs receive `&mut S`,
and `S` need not be `Send` — which is exactly what lets it hold the
runtime's `Rc`s and a `Stepper`. "Another thread touched the context" goes
from a discipline to unrepresentable, and a test proves an `Rc`-holding
state works.

`run` still resumes the job's panic on the caller (the C++ rethrows the
captured exception) and the worker survives; `post` contains panics so one
bad job cannot tear the thread down; drop drains before stopping. What did
not survive is inline same-thread re-entry: a Rust job already holds
`&mut S`, an inline nested job would alias it, so re-entry refuses with
instructions instead of deadlocking. `worker_thread_id` existed for that
inline check and for tests; the refusal owns the former and the tests ask
the worker directly.

Five portable tests: the `!Send`-state fence, FIFO + drain-as-barrier,
panic resume + survival, contained post panics, drop-as-barrier.
| `simple_family.cpp`: `*_pool_elems` / `*_value_extents` | `sizing.rs::pool_colour_elems` + `dispatch_llama.rs::llama_value_extent`/`llama_pool_elems` | ported (llama first): each colour sized by its WIDEST VALUE, not its kind — a routed stack is k× taller than the dense tensor sharing its colour. The int32 traps kept verbatim (ids are TWO two-byte elements; one kind writes ids and weights so the wider claims); in-place writers keep their widest claim. gemma4/gptoss extents + `extra_heap_bytes` + the budget bisection follow |
| `simple_family.cpp`: `extra_heap_bytes` / `max_forward_tokens_for_budget` / `stream_predicate` + the gptoss/gemma4 extents | `batch/heap_budget.rs` + `gptoss_pool_elems`/`gemma4_pool_elems` | ported; the budget sums the SAME pure pieces the engine stages (gemma's KV term walks owning layers at their own widths — the same shape `stage_gemma4_kv` allocates), the bisection spends the DIFFERENCE from a one-row fire (charging the floor made every model answer "one row"), and the stream predicate is one pattern with the C++'s three findings intact: `experts.` not `mlp.experts.` (gemma hangs its bank off the layer), no family argument (the llama proxy streamed nothing on Qwen3-MoE while gpt-oss streamed 10.75 GB), and the bias splits on paging because paging RENUMBERS |
| `FireCsr` + the fire's IO pass | `batch/fire_csr.rs` + `metal/storage.rs::write_fire_io` | ported; validation COMPOSES the schedule machinery (build_schedule/validate_paged/validate_capacity) rather than restating it, plus the fire's own gates: the two ownership statements must agree (`req_of_token` for the append, `qo_indptr` for the attention — a row they place differently is KV written to one request's pages and attended from another's) and one sample per request. Device-proven: both paged prefill smokes refactored onto it and re-run exact |
| the llama engine (`LlamaEngine::init/reset/fire`) | `metal/llama_engine.rs` | ported; a cache of PREPARED steps keyed by the fire's shape — the port bakes launches and tiles into the DAG, so a new shape is a new prepared step rather than a re-encode, and a decode fleet and a same-count prefill are different shapes (the 64-fleet row block). The KV pages live in the STORAGE, not the steps; `the_llama_engine_decodes_across_fires` proves the continuity no single-fire smoke could — a prefill fire's pages read by the decode fires that follow, greedy chain exact vs mlx_lm |
| the gpt-oss engine | `metal/gptoss_engine.rs` | ported; the llama assembly plus the family's one load-time difference — the trio is SOLVED off the staged heap before a single pipeline is planned, inside `new()`, because the plan's entry names depend on the answer. `the_gptoss_engine_decodes_across_fires` cross-validates the PAGED KV path against the ring: the same eight-token greedy chain the M=1 smoke produced through the contiguous ring, reproduced through pages |
| the gemma4 MB path + engine | `dispatch_gemma4.rs::build_gemma4_dag_mb` + `psos_gemma4.rs::gemma4_mb_plan` + `metal/gemma4_step.rs::Gemma4MbStep` + `metal/gemma4_engine.rs` | ported and verified: the decide-once MB DAG at per-layer widths, ONE `SdpaPaged` kind whose instantiation the step resolves from the layer (d256 sliding / d512 full — missing refuses: the d256 pipeline over 512-wide heads strides past every head), the strided PLE geglu claimed by kind, the alt-quant probe inside `new()`. `the_gemma4_engine_decodes_across_fires` lands the ring's exact chain through pages, first run — all four families now have engines, three with ring↔page cross-validation |
| the multi-request fleet contract | `tests/device_smoke.rs::the_llama_engine_isolates_two_requests` | verified: two conversations share every fire — disjoint physical pages (request 1 at page 64 under logical positions 0..17, so the indirection itself is under test), per-request page walks, a 2-row decode fleet — and each continues ITS chain token-exact, same first token and different second, so contamination would show immediately |
| the M=1 ring engine surface; split-K/fp16/tiled rungs; forward.cpp's engine-facing wiring (PTIR hooks, the timing hand-off, logits views) | — | missing/deferred. Since landed: `copy_state`'s deciding half (`store/control.rs`), taps (`batch/golden.rs`), the elastic sizing (`batch/sizing.rs`), the EOS substrate, and the timing attribution itself (`batch/timing.rs`) — what is left is the wiring, not the logic |
| `forward.cpp` / `forward.hpp` | 5393 | missing — the executor; last, over everything above |

### The bind audit (2026-08-09)

`decode_abi.hpp`'s `namespace bind` declares **36** `enum class` layouts, one
per kernel, each naming that kernel's buffer ordinals. The row above carried
them as missing; they are not.

* All 36 have a Rust home. Twenty-seven are findable by their C++ name; the
  other nine landed under a family or kernel spelling the row could not have
  guessed — `MoeRouteSort`/`MoeRouteRows` are `Kernel::LlMoeSort`/`LlMoeGather`
  with `MoeRouteParams`, `MoeCombineSorted` is `ExpertCombineParams`, and
  `Embed`, `Geglu`, `Softcap`, `PleCombine`, `SdpaSliding` and `SharedCombine`
  sit in the family bind files that own them.
* The ordinals match, spot-checked against the C++ enums:
  `Sdpa` (`K`=1, `V`=2, `GqaFactor`=4, `N`=5, `KHeadStride`=6),
  `SdpaPaged` (`GqaFactor`=4, `PageSize`=9, `NKvHeads`=10, `Scale`=11),
  `KvAppendPaged` (`HeadDim`=5, `KHeadStride`=6, `KSeqStride`=7, `PageSize`=10,
  `NKvHeads`=12) and `RowGather` (`Rows`=2) are identical on both sides.
  106 `slot::` constants carry them, plus the per-family walks.

The table is the weaker evidence. The stronger is that four families decode
**token-exact against mlx_lm** on device, and a bind ordinal that is off by one
is not a subtle error — it is a different buffer. The one bind the audit found
genuinely wrong is the C++'s: `KvAppendPaged::SrcRowStride` (ordinal 15) is
declared and read by the kernel and never bound by the C++ walk, which the
llama row above already records as found and fixed.

## The gpt-oss family — `csrc/src/model/gptoss/`

The third family, built in the SHARED `Kernel` vocabulary (the C++ kept
its kinds in their own namespace and mapped at encode time), so the M=1
step machinery is reused rather than mirrored.

| C++ | Rust | |
|---|---|---|
| `geometry.hpp` | `batch/gptoss.rs` | ported; refusal ladder shares the router bounds with llama's |
| `encode.cpp`: the DAG walk + `launch_shape` | `batch/dispatch_gptoss.rs` | ported; 508 dispatches pinned, `RowGather` deliberately absent at M=1 |
| `encode.cpp`: `qmv_kn` + params structs | `batch/gptoss_consts.rs` | ported; YaRN table host-computed (mlx `YarnRoPE` port), routed looks dense |
| `bind.cpp`: `router_bits_from_extents` + the mxfp4/proj probes | `batch/gptoss_solve.rs` | ported portable (name→bytes lookup); refusals carry the extents |
| `kernels.cpp`: the compile list | `batch/psos_gptoss.rs` | ported; both solver outcomes validated against the signature table |
| `encode.cpp`: `pso_for` | `psos_gptoss.rs::gptoss_kinds` / `gptoss_step_plan` | ported as plan DATA: the C++ re-decided per fire and fell back to qwen's base table for norms/residuals — the plan is self-contained and a kind nothing claims refuses at prepare, not at dispatch |
| `bind.cpp`: `bind_gptoss_dag` | `metal/gptoss_bind.rs` + the SHARED `bind_decode_dag` over `gptoss_decode_geometry` | ported; the all-full-attention view gives every layer its KV pair — the window is a property of the attention READ (a per-layer const), not of what is stored |
| the assembled step | `metal/gptoss_step.rs::GptOssStep` | ported; `fire_prefix` is the bisect's stage probe — truncated fires read any stage off the ordinary recycled pool |
| device smoke vs gpt-oss-20b-MXFP4-Q4 | `tests/device_smoke.rs::the_gptoss_assembly_decodes_the_reference_tokens` | verified TOKEN-EXACT: the staged trio solves (router 8 / proj 4 / mxfp4), and eight greedy fed-back argmaxes reproduce mlx_lm's continuation of "The capital of France is" |
| `encode.cpp`: `pso_for_paged` / `pso_for_mb` / `pso_for_mb_rows` + `launch_shape_mb` | `dispatch_gptoss.rs::build_gptoss_dag_mb` + `psos_gptoss.rs::gptoss_mb_plan` | ported portable: the C++ kept the tile decision in TWO switches that had to mirror each other across four predicates and says so itself ("wrong numbers, not a crash"); the builder decides once and the answers ride the `Dispatch` |
| `encode.cpp`: `gptoss_qmm_rows` / `_pool_rows` / `_moe_qmm_bn` / `_qmm_bn` / `is_dense_proj` | `dispatch_gptoss.rs`, same names | ported; the moe GEMM's widest-dividing-tile rule kept distinct from the dense no-split rule, with the C++'s 448-row measurement carried |
| `kernels.cpp`: the mxfp4 routed GEMM table | `psos_gptoss.rs::QmmRoutedBias{width,bn}` | ported; compiled only for the bank that has the instantiation, names pinned against the signature table |
| `RowGather` (prefill tail compaction) | `Kernel::G4RowGather` in the MB DAG + `dataflow.rs` arm | ported; a value of its own, never aliased — `[N,hidden]` in, `[S,hidden]` out — and the MB DAG colours under the shared scheduler |
| `bind.cpp`: `bind_gptoss_dag_paged` (the sinks remap, sample rows) | `bind_mb.rs` arms for `GoSdpaSinkPaged`/`G4RowGather` + `binds.rs::SDPA_PAGED_SINKS` | ported; the C++ remapped the sinks at bind time because index 14 means `Sinks` on the ring ABI and `AttnMaskEnabled` on the paged one — one Kind, two ABIs. Two kinds answer for themselves and there is nothing to remap; the naive widened arm (14 for both) was caught by exactly that test |
| `decode_consts.cpp`: `bind_gptoss_consts(rows, paged, head_rows)` | `gptoss_bind.rs`, ONE kind-driven walk | ported; the `paged` bool is gone — the DAG's kinds already say which ABI each dispatch is on — and every row-dependent const (sort tile, padded stack, SwiGLU count, paged sink Rows, RowGatherParams) reads the SAME helpers the MB builder wrote into the launches. At rows=1 it collapses to the M=1 constants: the token-exact smoke re-ran green unchanged |
| `encode_gptoss_step_mb` / `encode_gptoss_step_paged` + the three tables | `metal/gptoss_step.rs::GptOssMbStep` / `gptoss_mb_pso` / `load_gptoss_mb_psos` | ported; each table is asked only what it owns, and a dispatch that DECIDED a tile whose pipeline is missing REFUSES — the C++'s fallback would run the matvec under the GEMM's grid, the exact mismatch this arc is named after |
| device paged-prefill smoke vs gpt-oss-20b-MXFP4-Q4 | `tests/device_smoke.rs::the_gptoss_prefill_answers_in_one_paged_fire` | verified: one packed 16-row paged fire — biased dense GEMM, routed MXFP4 GEMM (first fire of both), paged sink, paged append, RowGather to the one sampled row — answers 5542 (' known'), mlx_lm's exact continuation. Found en route: `IoSlot::SampleRows` was the one MB slot the paged staging never allocated |
| the tiled/MMA paged sinks, `bind_gptoss_fp16_qmm`, FP16 precast staging | — | deferred WITH the shared family's same rungs, same reasons; the scalar paged sink and BF16 GEMM serve every fleet until those arms open for both |

The smoke's first run decoded fluent wrong tokens, and the tap bisect
(cosine per stage against mlx_lm, first divergence at `0.moe_out`,
magnitude 25× low with three of four expert rows zero) found the port's
`GoQmv` stride constants one slot off the ABI: `XSlotStride/XRowStride/
SlotsPerRow` are 9/10/11 in `decode_abi.hpp` and were 10/11/12 here, so
the kernel's `slots_per_row` read the row stride (2880) and every expert
row past the first wrote out of its slot. Qwen3.6-27B is DENSE — no
dispatch ever read those three slots, so five token-exact verification
matrices sat green over the defect until the first routed family fired.

## The llama family — `csrc/src/model/llama/`

The plain transformer decoder that `llama`, `llama3`, `mistral`,
`qwen2`, `qwen3` and `qwen3_moe` all are. Two axes vary and both are
FIELDS (qk_norm; dense-or-routed FFN), so one geometry serves six
config families. The Ll* mixture kinds are already in the shared enum —
gpt-oss fires them — and the rope's llama3 table is the family's one
piece of new arithmetic.

| C++ | Rust | |
|---|---|---|
| `geometry.hpp` | `batch/llama.rs` | ported; the refusal ladder intact, including `norm_topk_prob: false` (softmax-over-all weights sum below one — same tokens, quietly wrong magnitudes) and the `factor` split (llama3's divides FREQUENCIES in the table, linear's divides POSITIONS; one field meaning both applies it twice) |
| `decode_step.hpp`: the DAG | `batch/dispatch_llama.rs::build_llama_dag` + `llama_dag_stats` | ported; NOT the shared builder under a flag — that walk is qwen3.5's (2×-wide q\|gate, QSplit, AttnGate, structural QK-norm), and one builder trying to be both would carry every difference as a branch the other family dodges. The mixture is the shared nine dispatches minus the shared expert; `RowGather` deliberately absent at M=1, as with gpt-oss |
| `kernels.cpp`: `llama3_inv_freq` + the compile list | `batch/psos_llama.rs` | ported; the table pinned against mlx_lm's `Llama3RoPE` in float32 at both ends and on the ramp; the plan is the SHARED table under this family's features plus the two claims only the geometry can spell — the attention at its OWN width (the C++ records `_d128` as a literal that strode 64-wide heads past their end), and the freq-table rope claimed after the base so `source_of` answers with it |
| `decode_consts.cpp`: `qmv_kn` + `bind_llama_consts` | `batch/llama.rs::llama_qmv_kn` + `metal/llama_bind.rs` | ported as ONE kind-driven walk (ring + paged, both rope ABIs — buffer 3 is a pointer to one and a float to the other, and the base is log2(theta), not theta); the family's OWN KN table because the shared one answers for qwen's 2×-wide q. Found and fixed: the C++ walk leaves `KvAppendPaged::SrcRowStride` unbound while the kernel declares AND reads it — the exact class its own header opens by naming |
| the assembled M=1 step | `metal/llama_step.rs::LlamaStep` | ported; shared weight/scratch passes over `llama_decode_geometry`, family consts walk, `fire_prefix` bisect probe; the freq-table keepalive is `Option` — a geometric-series checkpoint has no table and says so |
| device smoke vs Llama-3.2-1B-Instruct-4bit | `tests/device_smoke.rs::the_llama_assembly_decodes_the_reference_tokens` | verified TOKEN-EXACT on the first fire: tied ends, `_d_64` attention, the llama3 frequency table (factor 32) on device — greedy 8 reproduce mlx_lm's continuation exactly. (Qwen3.6-35B-A3B turned out to be `qwen3_5_moe` — the GDN-hybrid family, not this one; the routed+qk-norm llama arm awaits a qwen3_moe checkpoint) |
| `encode.cpp`: `launch_shape(rows)` + `pso_for_mb_rows` + the qmm rules | `dispatch_llama.rs::build_llama_dag_mb` + the `llama_qmm_*`/`llama_moe_*` helpers | ported decide-once, as the gpt-oss arc closed it; with split-K deferred every dispatch is unsplit, which by the C++'s OWN measurement makes `qmm_bn_unsplit` the right width (its 448-row table: BN=32 beats both neighbours on every dense projection the split never covered). The 64-fleet BM=32 finding kept (`llama_dense_qmm_bm`), the head pinned to 32 |
| the MB assembly + paged smoke | `metal/llama_step.rs::LlamaMbStep` / `llama_mb_pso` + `psos_llama.rs::llama_mb_plan` + `tests/device_smoke.rs::the_llama_prefill_answers_in_one_paged_fire` | verified EXACT on the first fire: one packed 17-row paged prefill (padded to a whole 32-row GEMM block) answers 12366 (' Paris'), mlx_lm's continuation — the unbiased dense GEMM at the unsplit widths, the `_p32` paged attention, the paged append with the SrcRowStride fix live, the MB freq-table rope, the RowGather compaction. This family owns NO slot table: its GEMMs are entirely the shared lattice |
| split-K; fp16 staging; tiled paged sdpa; the `_sg8` paged rung | — | deferred with the shared family's, same reasons; `_sg8` additionally needs its 256-thread launch moved with the pipeline choice |
| device smoke | — | missing; local checkpoints: Qwen3.6-35B-A3B-4bit (routed), gemma-4 pair for the sibling family |

## The gemma4 family — `csrc/src/model/gemma4/`

The family where the DAG carries the most schedule: per-layer head
widths and rope bases, the KV-shared tail, the double-wide MLP over
exactly that range, the PLE chain, the norm sandwich, and — on the 26B —
a mixture BESIDE the dense MLP and full-attention layers whose V is the
K projection. The G4* kinds were already in the shared enum; this arc
gives them a geometry and a DAG.

| C++ | Rust | |
|---|---|---|
| `geometry.hpp` | `batch/gemma4.rs` | ported; the full predicate set (`kv_source` resolves each sharer to an earlier owner of its own attention type or the geometry refuses), the per-layer KN table, the two mixture refusals, the irregular `layer_types` refusal |
| `decode_step.hpp`: the DAG | `batch/dispatch_gemma4.rs::build_gemma4_dag` | ported; which dispatches EXIST moves per layer (a sharer drops six, a k-eq-v layer swaps its v projection for `G4VNormFromK` ordered BEFORE KNorm — it reads the projection KNorm overwrites), sliding/full attention as two kinds, E2B 694 dispatches pinned |
| `Kernel::G4VNormFromK` | appended at 99, nothing renumbered | the one kind the shared enum lacked: a flag on `G4VNorm` could not carry the ordering constraint |
| `scratch.cpp`: the dataflow | `dataflow.rs` G4 arms | ported into the SHARED walk: where gemma's shape IS a shared shape the kind joins the existing arm (the geglu is the silu slots, the scaled top-k threads like the plain one, the sorted mixture is the Ll arms), and only what differs gets an arm of its own — the PLE stream, the sandwich's fused norm+residuals, the k-reading V norm, and a mixture whose router reads its OWN norm of the residual, not the dense branch's. Both shapes colour under the shared scheduler, pinned |
| `kernels.cpp`: the compile list + `encode.cpp`: `pso_for` | `batch/psos_gemma4.rs::gemma4_step_plan` | ported as plan data, self-contained; every matvec is the TAIL form (the PLE projections' K=256 divides no reduction block), both attention widths spelled from the geometry through the SWA kernel (a full layer is window 0), one geglu serves three kinds, and the SECOND affine format peels exactly the exempted kinds — the C++ records the cost of guessing (router logits at cosine 0.10 while every feeding tensor agreed at 0.9999) and both shipped exemption sets are pinned |
| `decode_consts.cpp` | `metal/gemma4_bind.rs::bind_gemma4_consts` + `batch/gemma4_consts.rs` | ported; the walk that cannot be a switch on the kind — four constants move with the LAYER — with the C++'s paid-for findings kept: the router norm's `1/√hidden` gain in a case of its OWN (without the root every logit is 53× too large, survives the top-k, and the softmax one-hots the mixture), the attention scale at 1.0 (folded into the q-norm weights), the window bound for BOTH attention types, the embed scale that cannot fold into the tied table |
| `bind.cpp`: `bind_gemma4_dag` + the KV region | `metal/gemma4_step.rs::bind_gemma4_dag` / `stage_gemma4_kv` / `Gemma4Step` | ported; the family's own walk exists for ONE reason — the attention reads `kv[kv_source(layer)]`, and only this geometry can answer whose pages those are. The KV vector is rebuilt at each OWNING layer's own `[n_kv_heads_of, max_ctx, head_dim_of]` extent, sharers at None; the softcap runs in place over the logits IO |
| device smoke vs gemma-4-26B-A4B-it-4bit | `tests/device_smoke.rs::the_gemma4_assembly_decodes_the_reference_tokens` | verified TOKEN-EXACT under the chat template (margins ~26 logits; a raw completion prompt sits in a degenerate regime where bf16 jitter flips thin argmaxes — first five tokens exact, then noise, recorded in the test). The run itself caught the alt-router live: the probe first missed the runtime name, the 4-bit pipeline ran the 8-bit router's bytes, and the drift vanished the moment the solver found `layers.N.router.proj` at 8 bits — the exact defect the C++ recorded at cosine 0.10. Everything-at-once shape: mixture beside the MLP, k-eq-v with 2 global KV heads, interval 6, per-layer widths, PLE, sandwich, softcap |
