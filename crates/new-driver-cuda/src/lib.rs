//! The menlo CUDA driver's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `new-kernels-cuda`
//! (design §8, decisions #13–#16).
//!
//! **Successor note.** This crate is the menlo rewrite of `driver-cuda`'s
//! execution layer — the resolving and dispatching business its walk, bind,
//! and graph plumbing did for the string-plan stack. It merges into
//! `driver-cuda` when the old stack is deleted in one commit (design,
//! porting order step 6).
//!
//! **Prepare/capture.** Graph capture policy — whether to capture at all,
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
//! **The seam** — the driver side of the `MENLO-SEAM` markers in
//! `new_kernels_cuda`. A `Run` binds more than the ops name, and on this
//! plane the deepest seam is a *duality*: the IR declares kv geometry as
//! device inputs (design §7), but the plan builders are host functions that
//! walk that geometry's **contents** — and a device handle cannot be read
//! host-side. So every geometry the planners consume is bound twice: the
//! device tensor [`Run::tensor`] serves to launches, and the host copy the
//! same driver kept when it wrote that tensor ([`FireBindings::indptr_host`]
//! fire-wide, [`CachePlanning`] per cache space). Beside the duality sit the
//! extras with no IR seat at all, each seated in [`FireBindings`] and marked
//! at its arm:
//!
//! - the fire's shared **indptr**: `qo_indptr` is no longer a runtime input
//!   (design §5) — ragged views assemble from [`FireBindings::indptr`], and
//!   its host twin is what `plan_prefill`/`plan_mla` walk;
//! - **`kv_len`** (per-request kv lengths): derived host-side from
//!   kv_indptr + last_page_len, named on [`CachePlanning`], read by the
//!   sm90 and mla builders;
//! - the **mask** pair for `attention.masked`: no op names it; the
//!   plan-prefill arm binds [`FireTables::mask`] onto the plan at build;
//! - **`row_valid`** and **`request_of_token`**: the graph-padding mask and
//!   the owning-request table the pool entries read from
//!   [`FireTables`];
//! - the dsv4 **compressor slabs** ([`PoolSlabs`]) `pool.gather` reads
//!   beside its cache;
//! - the split-plane **mxfp4 banks**: one weight id, two device planes —
//!   [`WeightRow::Planes`] seats what the metal shell's one-handle rows
//!   refused, resolved through [`Run::planes`].

mod dispatch;
pub mod run;

pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, PoolSlabs, Run,
    SlotTable, StructSlot, WeightRow, WeightTable,
};

/// The walk is `new-driver`'s, written once and generic over any `Dispatch`;
/// re-exported so a shell driving this `Run` needs one crate in scope.
pub use new_driver::{Phases, fire, phases, walk};
