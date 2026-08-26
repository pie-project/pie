//! The menlo CUDA driver's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `kernels-cuda`
//! (design §8, decisions #13–#16).
//!
//! **Lineage.** This crate took `driver-cuda`'s name when the string-plan
//! shell it re-imagines was deleted with the rest of the old stack (design,
//! porting order step 6); serving plumbing (bind, pools, serve) rejoins it
//! as the fabric is rewired.
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
//! `kernels_cuda`. A `Run` binds more than the ops name, and on this
//! plane the deepest seam is a *duality*: the IR declares kv geometry as
//! device inputs (design §7), but the plan builders are host functions that
//! walk that geometry's **contents** — and a device handle cannot be read
//! host-side. So every geometry the planners consume (`kv_indptr`,
//! `kv_len`, the qo side) is bound twice: the device tensor
//! [`Run::tensor`] serves to launches, and the host copy the same driver
//! kept when it wrote that tensor ([`FireBindings::indptr_host`] fire-wide,
//! [`CachePlanning`] per cache space). The ledger of extras with no IR seat
//! is short now — the IR seats the write descriptors, `row_valid`,
//! `request_of_token`, and the mask bits as declared inputs — leaving:
//!
//! - the fire's shared **indptr**: `qo_indptr` is no longer a runtime input
//!   (design §5) — ragged views assemble from [`FireBindings::indptr`], and
//!   its host twin is what `plan_prefill`/`plan_mla` walk;
//! - the **mask span table** for `attention.masked`'s op-named bits: the
//!   plan-prefill arm binds [`FireTables::mask_indptr`] onto the plan at
//!   build;
//! - the dsv4 **compressor slabs** ([`PoolSlabs`]) `attention.pool_gather`
//!   reads beside its cache;
//! - the split-plane **mxfp4 banks**: one weight id, two device planes —
//!   [`WeightRow::Planes`] seats what the metal shell's one-handle rows
//!   refused, resolved through [`Run::planes`].
//!
//! Inside `kernels_cuda` a residue of derive-addressed writers remains
//! (quantized kv schemes, the mla latent writer, the dsv4 store): those
//! entries accept the op's `write_page`/`write_offset` and mark where the
//! device text still re-derives the cells.

mod dispatch;
pub mod run;

pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, PoolSlabs, Run,
    SlotTable, StructSlot, WeightRow, WeightTable,
};

/// The walk is `kernels::exec`'s, written once and generic over any
/// `Dispatch`; re-exported so a shell driving this `Run` needs one crate in
/// scope.
pub use kernels::{Phases, fire, phases, walk};
