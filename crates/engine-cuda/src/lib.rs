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
//! walk over it the engine substrate's (`engine::fire::walk`, design
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
//!            `WeightTable` out — through model-loader, with no model family
//!            named on this side
//! store/     the pools: `kv.rs` is backend-neutral page arithmetic marked
//!            for `engine::store`, the rest is bytes
//! arena.rs   one allocation at the compiler's `ArenaMap::bytes`, and the
//!            per-fire `SlotTable` that is `base + offset` and nothing else
//! inputs.rs  the pointer-stable resident fire inputs and the plan grants
//! mask.rs    a lane's run-length mask, expanded into the (query, key) bits
//!            `attention.masked` reads, with the causal bound folded in
//! window.rs  which rows and lanes each region runs over — design §0's
//!            window-split, resolved per fire and read by every `Run::*`
//! record.rs  the graph cache: one exec per shape key, captured once behind
//!            two warm eager fires, replayed forever
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
//! **The walk is not here.** `engine::fire::walk` is written once, over
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

pub mod api;
pub mod arena;
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
pub mod run;
pub mod serve;
/// Who is airborne, and where the settlement callbacks ride (survey §7, I7).
pub mod settle;
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


pub use error::{Fault, Result};
pub use mask::{LaneMask, Staged as StagedMask};
pub use program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{Graphs as GraphCache, Key, Mode, Stats};
pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, Planning,
    PoolSlabs, Run, SlotTable, StructSlot, WeightRow, WeightTable,
};
pub use api::{ContractFor, Cuda, DeviceBoot};
pub use serve::{Boot, FireCost, Graphs, Knobs, Lane, Seated, Shell};

/// What a capturing lane's fire hands back, one entry per exported attention
/// layer — the contract's own type, re-exported so a caller of
/// [`Shell::fire_captured`] need not reach two crates deep for the noun its
/// own signature is written in (design §9, palo C4b).
pub use engine::engine_api::fire::LayerScores;
pub use weights::AdapterPlane;
pub use window::{Cursor, Window, Windows};
