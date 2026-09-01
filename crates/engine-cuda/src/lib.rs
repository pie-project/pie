//! The menlo CUDA engine's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `kernels-cuda`
//! (design §8, decisions #13–#16).
//!
//! **Lineage.** This crate took `engine-cuda`'s name when the string-plan
//! shell it re-imagines was deleted with the rest of the old stack (design,
//! porting order step 6); serving plumbing (bind, pools, serve) rejoins it
//! as the fabric is rewired.
//!
//! **Prepare/capture.** The split itself is the model compiler's and the
//! walk over it the engine substrate's (`model_exec::fire::walk`, design
//! decisions #11–#12); graph capture policy — whether to capture at all,
//! rows-bucketing vs graph-update, when a bucket is re-captured — stays the
//! shell's; this crate only makes the split *executable*. The prepare-phase
//! arms run the pure plan builders and `stage` their pageable-host uploads
//! eagerly (#16), so nothing they do can leak into a capture; the
//! capture-phase arms enqueue only (#15), so the same walk runs identically
//! inside `cudaStreamBeginCapture`. Two words cross the boundary:
//! [`FireBindings::capture`] carries the shell's policy *in* (the builders
//! carve graph-shaped, padded schedules under it), and
//! `PrefillPlan::graph_capturable` carries the builders' answer *out* (a
//! schedule that would not fit fell back to an uncapturable one — the shell
//! reads it before capturing).
//!
//! **The seam** — the engine side of the `MENLO-SEAM` markers in
//! `kernels_cuda`. A `Run` binds more than the ops name, and on this
//! plane the deepest seam is a *duality*: the IR declares kv geometry as
//! device inputs (design §7), but the plan builders are host functions that
//! walk that geometry's **contents** — and a device handle cannot be read
//! host-side. So every geometry the planners consume (`kv_indptr`,
//! `kv_len`, the qo side) is bound twice: the device tensor
//! [`Run::tensor`] serves to launches, and the host copy the same engine
//! kept when it wrote that tensor ([`Window::indptr_host`] per window,
//! [`CachePlanning`] per cache space). The ledger of extras with no IR seat
//! is short now — the IR seats the write descriptors, `row_valid`,
//! `request_of_token`, and the mask bits as declared inputs — leaving:
//!
//! - the **qo boundaries**: `qo_indptr` is no longer a runtime input
//!   (design §5) — ragged views assemble from the current window's
//!   ([`Run::qo_indptr`]), and its host twin is what `plan_prefill`/`plan_mla`
//!   walk;
//! - the **mask span table** for `attention.masked`'s op-named bits: the
//!   plan-prefill arm binds [`FireTables::mask_indptr`] onto the plan at
//!   build, sliced to its own window, because the table is indexed by the
//!   schedule's request number and the bits it points into are fire-wide
//!   (the offsets stay ABSOLUTE, exactly as `GeomKind::Indices`' bounds do);
//! - the dsv4 **compressor slabs** ([`PoolSlabs`]) `attention.pool_gather`
//!   reads beside its cache;
//! - the split-plane **mxfp4 banks**: one weight id, two device planes —
//!   [`WeightRow::Planes`] seats what the metal shell's one-handle rows
//!   refused, resolved through [`Run::planes`].
//!
//! # The shell (palo porting order, step 4)
//!
//! Everything above is the dispatch layer — one op in, one launch out. The
//! rest of this crate is the shell around it, and it is thin on purpose
//! (design §6, decision #13: *shells are thin call-order crates*):
//!
//! ```text
//! device/    the stream, the cuBLAS handle, and `cudaMalloc` — the bytes no
//!            kernel entry allocates, because an entry that allocated per
//!            fire could not be captured
//! weights.rs a `ModelContract` and a checkpoint in, a resident
//!            `WeightTable` out — through checkpoint, with no model family
//!            named on this side
//! store/     the pools: `kv.rs` is backend-neutral page arithmetic marked
//!            for `model_exec::store`, the rest is bytes
//! arena.rs   one allocation at the compiler's `ArenaMap::bytes`, and the
//!            per-fire `SlotTable` that is `base + offset` and nothing else
//! inputs.rs  the pointer-stable resident fire inputs and the plan grants
//! mask.rs    a lane's run-length mask, expanded into the (query, key) bits
//!            `attention.masked` reads, with the causal bound folded in
//! window.rs  which rows and lanes each region runs over — design §0's
//!            window-split, resolved per fire and read by every `Run::*`
//! record.rs  the bodies: one exec per `(bucket, present set)`, armed at
//!            LOAD over the realizable lattice and replayed forever, cut
//!            around the regions no capture can name
//! serve.rs   `Shell::load` and `Shell::fire` — call order, top to bottom
//! ```
//!
//! **The guest-program plane is beside the fire, not inside it.**
//! [`program`] compiles a `LaunchPackage`'s emitted CUDA, binds an instance's
//! channel rings and runs its stages; design §9 attaches those stages before
//! and after the immutable graph and never within one. The attachment itself
//! is the runtime's step and is deliberately not wired here — see
//! [`program`]'s module docs for the seam.
//!
//! **The walk is not here.** `model_exec::fire::walk` is written once, over
//! `Dispatch` and `Sink`, and this shell hands it a [`Run`] and its
//! [`Cursor`] (decision #11). [`record`] is the second mode of the SAME walk:
//! it runs the prepare regions on the open stream and the capture regions
//! inside `cudaStreamBeginCapture`, so a replayed fire is an eager fire by
//! construction rather than by assertion. Eager stays the golden it is diffed
//! against, and [`Graphs::Off`] is a mode of this shell rather than a build
//! without the other one.
//!
//! Inside `kernels_cuda` a residue of derive-addressed writers remains
//! (quantized kv schemes, the mla latent writer, the dsv4 store): those
//! entries accept the op's `write_page`/`write_offset` and mark where the
//! device text still re-derives the cells.

/// **THE LORA SINK'S RESOLVER** (alto adapter §6.1, §6.4): reading the
/// `lora` sink off a launch package, and turning the f32 cell a guest seeded
/// into the bank's own bytes. Pure host arithmetic — the half of the wave
/// that needs no device to be judged.
pub mod adapter;
pub mod api;
pub mod arena;
/// **The shared-adapter store** (alto adapter §3.3, promoted to wave 1 by
/// §6.1): a read-only mount whose files are the truth, a single-flight
/// refcounted host cache over them, and the bank slots keyed by blob
/// identity — which is what makes N instances of one adapter one device copy.
/// Nothing in it is reachable from the fire path.
pub mod blob;
/// Reading a boot document — this crate's half of it, in this crate.
pub mod boot;
pub mod device;
mod dispatch;
mod error;
/// The export seam and the op-vocabulary scans beside it — pure IR analysis,
/// lifted out of `serve.rs` (alto article 9's "shells are call order").
/// **The routed-expert tier** (alto design §7, wave D2): the residency plan a
/// budget decides, the pinned host copy of every expert, the device slab of a
/// few of them, and the indirection table that lets a captured graph read
/// either without knowing which.
pub mod experts;
pub mod exports;
pub mod inputs;
pub mod mask;
pub mod program;
pub mod record;
pub mod rotate;
pub mod run;
pub mod scores;
pub mod serve;
/// Who is airborne, and where the settlement callbacks ride (survey §7, I7).
pub mod settle;
/// **The pinned double-buffered H2D pump** (alto streaming §1 and build-order
/// item 1), ported from `origin/dev`'s `loader/staged_h2d.hpp`. The one path
/// bulk weight bytes take: four lanes, each overlapping a host memcpy with the
/// DMA already in flight, because one lane already outruns NVMe by 1.6×.
pub mod staged_h2d;
pub mod store;
pub mod weight_cache;
pub mod weights;
pub mod window;

/// **THE ENTRIES THAT CLAIM A WORKSPACE TWO STREAMS WOULD SHARE**, by
/// `model_ir::Operands::name` — what this shell hands `model_compiler`'s P6 as
/// `DeviceProfile::exclusive`, so that two of them are never scheduled onto
/// two streams at once.
///
/// **IT IS EMPTY, AND THAT IS THE POINT.** It held eleven names, and every
/// one of them was on it for one reason: `kernels_cuda::Ctx::scratch` handed
/// back a slab keyed by a static NAME, so two launches inside one slab at the
/// same instant staged over each other and the fire computed anyway. A slab
/// is keyed by `(arena, name, stream)` now — one arena per CUDA context, one
/// slab per stream inside it, growth broadcast across the arena's streams so
/// the eager warm pass still warms what the capture reads
/// (`kernels_cuda::Slabs`, and `kernels_cuda::jit::device`'s header for the
/// argument). Two arms of a fork group take two slabs, so there is nothing
/// left for the compiler to order apart, and the linear-attention layers of
/// qwen and kimi — the eleven names' whole cost, build log 24 — fork.
///
/// **THE SEAT STAYS, BECAUSE THE NEXT SUCH ENTRY WILL NEED IT.** The list is
/// the shell's answer to a question the compiler cannot ask: no `Operands`
/// method says which entries reach a device-wide workspace, and a
/// backend-neutral pass that knew would be a compiler that knows an
/// allocator. An entry that acquires a workspace this shell cannot key per
/// stream — a device-wide semaphore, a library handle with hidden state —
/// belongs here, and `tests/no_forked_pair_shares_a_slab.rs` re-derives the
/// question from the other end.
pub const EXCLUSIVE: [&str; 0] = [];

/// **THE OPS THIS SHELL CAN RUN OVER A SEGMENT LIST IN ONE LAUNCH** — what
/// `DeviceProfile::grouped` is handed, and therefore what lets P4 answer
/// `Fallback::Grouped` for a consumer it could not seat (design §3,
/// decision #24).
///
/// The list is the shell's answer to a question the compiler cannot ask, in
/// the shape [`EXCLUSIVE`] above already established. What a name here
/// promises is one sentence: handed the union of the consumer's row intervals
/// PLUS the intervals, the op computes what a launch per interval computes,
/// and touches no row in the gaps between them. `Windows::of` cuts the union
/// window, `Run::segments` carries the list, and `linear/lora.cuh` is where
/// the promise is kept.
///
/// **ONE NAME, AND THE REASON THE SECOND CANDIDATE IS NOT HERE.** The other
/// windowed consumer this catalog cannot seat is `attention.prefill_lse`, and
/// it fails the second clause twice over: flashinfer's `q_indptr` doubles as
/// the offset and the length of a request's query rows, so there is no seat
/// for a second offset; and under capture the split-kv fold writes densely
/// over the whole row extent it was handed, which would clobber every
/// neighbour standing in a gap. Neither is a kernel this tree owns. The
/// correction is: it is ours, it is one file, and its weight side was already
/// runtime-indexed — which is the whole of why it is the one that could go
/// first.
pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// **THE OPS THAT ADDRESS OFF THE SEAT'S START AND NOT OFF THEIR POINTER** —
/// by `model_ir::Operands::name`, the third of this shell's answers to a
/// question the compiler cannot ask, in the shape [`EXCLUSIVE`] and
/// [`GROUPED`] above established.
///
/// **WHAT A NAME HERE PROMISES, IN ONE SENTENCE.** Handed the PLANE'S BASE
/// pointers and an armed staged-geometry seat `(count, start)`, the op
/// computes over plane rows `[start, start + count)` and touches no other
/// row. Two halves, and both are load-bearing. The COUNT half is the guard
/// every seated kernel already keeps — `if (win && r >= win[0]) return;` —
/// which retires the tail of a grid carved at a bucket. The START half is the
/// one this list is about: `r + win[1]` is the plane row, so a pointer handed
/// UNSLICED still addresses this launch's own rows. A kernel that keeps only
/// the first half is GUARD-ONLY: correct for a launch whose pointers were
/// already advanced, wrong for one whose were not, and off this list.
///
/// **WHO READS IT.** `exports::regions_shifting` turns it into a per-region
/// fact, and `Windows::admits` spends that fact: a WINDOWED region — one that
/// does not begin at the fire's row zero — may still carry a recorded body's
/// replay when everything in it is named here. Neither gathered rectangles
/// nor grouped unions are ever admitted by it, whatever this list says,
/// because neither is an interval and `(count, start)` names nothing else.
///
/// **AND SINCE THE TIER-2 CAMPAIGN A NAME MISSING FROM THIS LIST COSTS ITS
/// REGION AND NOT ITS COMPOSITION.** A region the rule refuses is an ISLAND:
/// the body is captured in segments around it and the fire path re-issues it
/// eagerly between the execs (`record::Cut`). That is a real cost — that
/// stretch's launch overhead, and P6's overlap across its span — and it is why
/// the discipline is SEAT-FIRST, SEGMENT-SECOND. An op that could keep both
/// halves of the promise belongs here; cutting around it is the answer for the
/// ones that cannot, not a reason to stop adding names.
///
/// **DERIVED FROM THE LAUNCHERS, NOT FROM THE DEVICE TEXT ALONE**, because a
/// name admits ALL of its instantiations. `kernels_cuda`'s launcher decides
/// per fire whether the seat argument is `Ctx::stage` or `ArgValue::ABSENT`,
/// and one absent seat under a name is enough: the kernel then sees a null
/// `win`, reads neither word, and wants pre-shifted pointers. So a name is
/// here only when EVERY `Fire::at` it can reach both takes an unconditional
/// seat and reads `win[1]` off it.
///
/// **AND THE SECOND AXIS, WHICH THIS PROMISE STILL DOES NOT SPEAK FOR AND NOW
/// HAS TWO EXCEPTIONS.** The promise is about ROWS. Most names here also read
/// PER-LANE tables — `qo_indptr`, `slot_ids`, `commit_len`, page bounds — and
/// index them with the UNSHIFTED ordinal, because their request number IS a
/// grid coordinate: `attention.index_topk` and `attention.pool_lse` read
/// `page_indptr[r]` at the `r` their own grid counts, and
/// `Run::recurrent_cut` hands the per-STEP scans their slot map already sliced
/// at the window's `lane_offset`. For those, window-local lane tables are the
/// contract, and handing them fire-wide ones breaks them silently.
///
/// **THE FIVE FA2 NAMES ARE THE FIRST EXCEPTION, AND THEY ARE ONE BECAUSE
/// THEIR REQUEST NUMBER IS A DATUM** (chunk 2c-b). An FA2 launch reads its
/// request id out of `request_indices[bx]` — a vector the plan builder stages
/// per fire — so it is the one family whose lane axis can be moved without
/// touching a kernel's indexing at all: `Run::planning` stages
/// `lane_offset + r` under a plane base, and `Run::pool_absolute` and
/// `Run::mask_indptr` hand those launches the fire's tables to match. Absolute
/// ids and fire-wide tables are one change with two halves, they are gated on
/// the same `Run::plane_base` the row axis is.
///
/// **THE FOUR CHUNKED ARMS ARE THE SECOND, AND THEY EARNED IT THE OTHER WAY:
/// BY READING THEIR LANE OFF THE SEAT** (the chunked-arm wave). Their request
/// number IS a grid coordinate — one block, or one grid row, per request — so
/// on the face of it they belong with the steppers. What moved them is that
/// their per-lane tables cannot survive a body: `Run::recurrent_cut`'s slice
/// is `base + lane_offset * 4`, `lane_offset` is the sum of the LANES of the
/// classes in front of the window, and a `record::BodyKey` deliberately does
/// not fix it — the ladder it carries fixes the sum of their row RUNGS, which
/// is a bound on that number and not that number — so a recorded slice is
/// stale on every replay but its recording one. The seat grew a LANE half for them: `win[2]` is the window's live lane
/// count, `win[3]` its first fire lane, and the kernels read `slot_ids`, the
/// fold predicate, the commit length and the segment origin at `r + win[3]`
/// off vectors `Run::recurrent_absolute` hands over WHOLE. The window's own
/// rebased CSR is the one per-lane vector that stays on `r`, because it
/// belongs to the window and not to the fire; that split is stated in every
/// one of those kernels' comments. What the grid-at-ceiling wave added is the
/// only thing about that CSR a body may not bake — its LENGTH, which is these
/// four launches' grid: `Run::ragged_lanes` declares it out to the key's lane
/// ceiling and `win[2]` retires the requests past the fire's own, so the grid
/// stops following the batch without either pointer moving. `Run::recurrent_absolute` is a SECOND DOOR
/// and not a widening of the first, because the per-step scans beside them
/// index `slot_ids[r]` with a ROW ordinal and would break silently on a
/// fire-wide map.
///
/// A further name that wants the lane axis has to earn it one of those two
/// ways: by reading its lane from something a fire STAGES, or by reading it
/// off the seat's `win[3]` against fire-wide tables it was handed on purpose.
///
/// **AND THE NAMES THAT ARE NOT HERE, BY NAME, BECAUSE EACH IS A DIFFERENT
/// REASON.**
///
/// * `linear.matmul` and `linear.lm_head` — the dense form is cuBLAS and
///   cuBLASLt with no kernel of ours to seat (and `gemm` rounds M up to a
///   bucket on purpose, writing rows past the live count); the quantised
///   forms — mlx-affine, fp8, q4k/q6k — DO take the seat and are guard-only,
///   reading `win[0]` and never `win[1]`.
/// * `linear.lora_correct` — two origins that were never reconciled, and the
///   MoE wave settled one of them: its select-gemv leg is `moe.cuh`'s, which
///   now takes the full pair (a lora fire can never see it armed — a grouped
///   region is refused a body — so the seat it passes is always `ABSENT`),
///   while `linear/lora.cuh`'s combine leg still takes one seat and reads only
///   the count. It is [`GROUPED`]'s single name and the start word is what
///   that leg is owed.
/// * `elementwise.res_blend` — seated, and it does read `win[1]`; but it
///   walks its candidate blocks as `blocks + (j * block_rows + row) * hidden`
///   where `block_rows` is a row count the RECORDING baked. The pitch it
///   needs is the PLANE'S height — a fact only the arena knows, and one a
///   `Tensor`'s `ptr/rows/width` cannot spell once `rows` is the window's.
///   The name is owed that height from the side that owns it (a handle field
///   or a further seat word — the seat holds four now, and `res_blend` reads
///   only the row pair) before it can go on this list.
/// * (the per-head rmsnorm arms were here — the coverage wave taught them to
///   derive their row as `b / heads` and shift in the flattened frame, the
///   `rmsnorm_grouped_plus_one` idiom, and the conditional seats became
///   unconditional; the four names are on the list now.)
/// * (`attention.decode`, `attention.decode_lse`, `attention.prefill`,
///   `attention.prefill_lse` and `attention.masked` were here, and what it
///   took to move them is the record chunk 2c left. The first reason written
///   here — "a `q_indptr` that is already both an offset and a length" — was
///   wrong: a length is a difference and survives a shift, so the q axis rode
///   an ABSOLUTE qo vector with no kernel change at all (2c-a,
///   `Run::ragged_q`). The three real blockers each cost their own edit.
///   The OUTPUT axis: under a body `split_kv` is always taken — a body means
///   `Graphs::On` means `FireBindings::capture`, and both planners set
///   `split_kv` unconditionally under it — so the real plane write is the
///   cascade fold, and the fold got the `(count, start)` seat every other
///   shifting kernel already had, plus the row-count pointer the plan had
///   been staging and nobody had ever wired. The LANE axis: the sliced page
///   tables a body would have baked are handed over WHOLE now, and the
///   schedule stages `lane_offset + r` to match. And DECODE's one line:
///   `decode.cuh` took `batch_idx` for its query row where prefill had always
///   read a `q_indptr`, which is the same number for a launch reading its own
///   rebased vector — one query row per decode request — and the wrong row
///   the instant `batch_idx` is a fire lane.)
/// * `attention.dense` is ours and simply never took a seat, and the sm90
///   prefill arm (`attention.prefill_sm90` in the launcher's vocabulary)
///   refuses before it launches at all.
/// * (`attention.kv_append` and `attention.kv_append_shared` were here — the
///   coverage wave seated `kv_append_explicit` with its fp8 siblings' exact
///   split, so every instantiation the two names reach now reads the seat;
///   both are on the list.)
/// * `attention.mla_decode`, `attention.mla_prefill` — a compute-capability
///   fork: the naive engine is seated, the flashinfer one is not, and a name
///   cannot be on this list on some cards only. `attention.mla_absorb_q` and
///   `attention.mla_absorb_out` are cuBLAS batched GEMMs with nothing to seat.
/// * (the four CHUNKED arms were here — `attention.ssm_causal_conv1d_chunked`,
///   `attention.ssm_gated_delta_chunked`, `attention.ssm_kda_chunked`,
///   `attention.ple_ngram_ids_chunked` — with the reason "they take no seat at
///   all: they carry their own per-request begin/commit geometry and were
///   never taught the word". The chunked-arm wave taught it, and the word it
///   had to learn was not `win[0]`: their grids count REQUESTS, so the guard
///   is `win[2]` and the addressing is a SPLIT — the fire's per-lane tables at
///   `r + win[3]`, the window's own CSR at `r`, the fire's activation planes
///   at `+ win[1]`, the staged scratch the preps write at the launch-local row
///   with nothing added, and the recurrent slabs, addressed by a slot's VALUE,
///   shifted by nothing at all. The lane half of the seat exists for them, and
///   `Run::recurrent_absolute` is the engine half. All four names are on the
///   list now.)
///   `attention.mla_kv_append`, `attention.index_kv_append` and the four
///   `attention.pool_*` writers are seat-less for the plain reason the chunked
///   arms used to be.
///
/// A name whose every instantiation could not be verified stays off. That is
/// the safe direction: an op wrongly absent costs a body that could have been
/// replayed, and an op wrongly present costs the right number of rows read
/// from the wrong place, silently.
pub const SHIFTED: [&str; 81] = [
    // ── attention: the arms this tree owns — the linear-attention scans, the
    //    mla prologue and its naive selected engine, the index and pool
    //    readers, and the two lse folds — and, since chunk 2c-b, the five FA2
    //    names, whose seat is the cascade fold's and whose lane axis is the
    //    schedule's (the exclusion paragraphs above carry the whole account).
    //
    //    **AND SINCE THE CHUNKED-ARM WAVE, THE FOUR PREFILL SCANS**:
    //    `ssm_causal_conv1d_chunked`, `ssm_gated_delta_chunked`,
    //    `ssm_kda_chunked` and `ple_ngram_ids_chunked`. They are the first
    //    names whose GRID counts requests rather than rows, so they read the
    //    seat's lane half — `win[2]` to retire a ceiling grid's padded lanes,
    //    `win[3]` to name a fire lane — and they are the reason that half
    //    exists. With them a hybrid model's windowed mixer regions are all on
    //    this list, which is what makes a mixed fire on an SSM hybrid
    //    body-servable at all.
    //
    //    **AND A PREFILL BODY NO LONGER RESHAPES WHILE ITS ROWS MOVE.** The
    //    prefill planner's `cta_tile_q` and padded batch are functions of the
    //    row total it is CARVED at, and both ride `Run::schedule_shape` — so
    //    while that total was the fire's, a key whose fires oscillated
    //    between two row counts re-captured on each of them. The
    //    plan-at-bucket-ceiling design moved the carve: `Run::planning` hands
    //    the builders the KEY's rows and the key's lane ceiling on exactly the
    //    fires a body serves — the fire's bucket for a whole-fire window and,
    //    since the ceiling design's Option B, prefix sums over the key's
    //    per-class rung ladder for a WINDOWED one — the rows between the
    //    fire's own and the ceiling are ones no work item is emitted for, and
    //    the payload numbers are therefore a function of the
    //    `record::BodyKey` and the load. `record::Graphs::fire_body`'s demotion stays, and what it
    //    means has inverted: `record::BodyStats::reshapes` is an anomaly
    //    counter now, and a nonzero one names a builder whose hashed image
    //    still follows the fire rather than a batch that will not sit still.
    "attention.decode",
    "attention.decode_lse",
    "attention.index_layernorm_rope",
    "attention.kv_append",
    "attention.kv_append_shared",
    "attention.index_rope",
    "attention.index_topk",
    "attention.masked",
    "attention.merge_lse",
    "attention.mla_decode_selected",
    "attention.mla_latents",
    "attention.mla_latents_rope",
    "attention.mla_prefill_selected",
    "attention.mla_split_q_b",
    "attention.ple_ngram_ids",
    "attention.ple_ngram_ids_chunked",
    "attention.pool_lse",
    "attention.prefill",
    "attention.prefill_lse",
    "attention.sink",
    "attention.ssm_causal_conv1d",
    "attention.ssm_causal_conv1d_chunked",
    "attention.ssm_gated_delta",
    "attention.ssm_gated_delta_chunked",
    "attention.ssm_gdn_prep",
    "attention.ssm_kda_chunked",
    "attention.ssm_kda_step",
    // ── elementwise: the row-plane arms, each one block (or one flattened
    //    lane) per token row, every plane it touches shifted together.
    "elementwise.add_bias",
    "elementwise.clamp",
    "elementwise.clamp_learned",
    "elementwise.gate_sigmoid_mul",
    "elementwise.hc_expand",
    "elementwise.hc_fold",
    "elementwise.hc_gates",
    "elementwise.hc_inject",
    "elementwise.hc_mix",
    "elementwise.hc_rmsnorm_f32",
    "elementwise.layernorm",
    "elementwise.layernorm_no_scale",
    "elementwise.mul_scalar",
    "elementwise.ple_gate",
    "elementwise.residual_add",
    "elementwise.rmsnorm",
    "elementwise.rmsnorm_gated",
    "elementwise.rmsnorm_gated_by",
    "elementwise.rmsnorm_grouped_plus_one",
    "elementwise.rmsnorm_no_scale",
    "elementwise.rmsnorm_per_head",
    "elementwise.rmsnorm_per_head_plus_one",
    "elementwise.rmsnorm_plus_one",
    "elementwise.rope_full",
    "elementwise.rope_mrope",
    "elementwise.rope_partial",
    "elementwise.rope_partial_last",
    "elementwise.rope_partial_q",
    "elementwise.rope_yarn",
    "elementwise.scale",
    "elementwise.silu_scaled",
    // ── layout: the gathers and splits whose id, weight and destination
    //    planes share one row axis; the banks they read are id-indexed.
    "layout.embed",
    "layout.embed_concat",
    "layout.embed_weighted",
    "layout.scatter_live_rows",
    "layout.select",
    "layout.split_q_gate",
    "layout.split_qkv",
    "layout.split_rows",
    // ── linear: the fused activations, the moe routers and folds, and the
    //    routed bf16 select. Not the dense GEMMs — see the exclusions above.
    //
    //    **THE ROUTERS AND THE SELECT ARE THE MOE WAVE'S**, and between them
    //    a windowed MoE region is admissible whole: the four routers are
    //    token-row-gridded and read the pair the way every other row entry
    //    does, and `moe_matmul_select` is the first name on this list whose
    //    GRID counts something else — ROUTES, `top_k` of them per row — so it
    //    multiplies the seat's pair by the fan-out and addresses in route
    //    space (`linear/moe.cuh` states the conversion, and it is the same
    //    token axis `moe_weighted_sum` folds back onto).
    //
    //    **AND THE TWO MXFP4 SELECT NAMES STAY OFF, FOR A REASON THAT IS NOT
    //    ABOUT THEM.** `linear.moe_matmul_select_bias` and
    //    `linear.moe_matmul_select_quant` fire `linear/quant.cuh`, which is
    //    not seated; a name admits ALL of its instantiations, so one unseated
    //    kernel under a name keeps the name off. A quantized MoE's region is
    //    therefore still refused, and what it is owed is the quant plane's own
    //    seat and not a word here.
    "linear.mlp_geglu_tanh",
    "linear.mlp_geglu_tanh_packed",
    "linear.mlp_gelu_tanh",
    "linear.mlp_situ",
    "linear.mlp_swiglu",
    "linear.mlp_swiglu_clamp",
    "linear.mlp_swiglu_clamp_alpha",
    "linear.moe_bias_sum",
    "linear.moe_matmul_select",
    "linear.moe_sigmoid_gate_add",
    "linear.moe_topk_sigmoid",
    "linear.moe_topk_softmax",
    "linear.moe_topk_softmax_scaled",
    "linear.moe_topk_sqrt_softplus",
    "linear.moe_weighted_sum",
];

/// **THE OPS THAT LAUNCH NOTHING** — the prepare-phase planners, admitted to
/// a shifting region for the inverse of [`SHIFTED`]'s reason. A plan op runs
/// host builders during the PREPARE half of the walk and puts no node in the
/// captured graph, so there is no launch to address a row wrongly; and its
/// plans are rebuilt per fire against that fire's own staged geometry, so a
/// body replay reads a schedule as fresh as an eager fire's. A region that
/// holds only planners and `SHIFTED` names may therefore be windowed under a
/// body — which is not a corner: the class-split attention regions of every
/// mixed fire are exactly (planner + FA2 arm), and refusing the planner was
/// the last thing standing between a mixed composition and a body.
pub const PLANNED: [&str; 2] = ["attention.plan_decode", "attention.plan_prefill"];


pub use error::{Fault, Result};
pub use mask::{LaneMask, Staged as StagedMask};
pub use program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{BodyKey, BodyStats, Graphs as GraphCache};
pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, Planning,
    PoolSlabs, Run, SlotTable, StructSlot, WeightRow, WeightTable,
};
pub use api::{ContractFor, Cuda, DeviceBoot};
pub use boot::open;
pub use serve::{
    Boot, DEFAULT_GPU_MEM_UTILIZATION, FireCost, Graphs, Knobs, Lane, Media, Seated, Shell,
};

/// What a capturing lane's fire hands back, one entry per exported attention
/// layer — the contract's own type, re-exported so a caller of
/// [`Shell::fire_captured`] need not reach two crates deep for the noun its
/// own signature is written in (design §9, palo C4b).
pub use engine::fire::LayerScores;
pub use blob::{
    Adapters, Binding, Site as AdapterSite, Source as AdapterSource, layer_of, role_of,
    site_of,
};
pub use weights::{AdapterPlane, BankSeat};
pub use window::{Cursor, Window, Windows};
