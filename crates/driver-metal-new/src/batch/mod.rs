//! The batch subsystem: scheduling, composition, and the forward's shape.
//!
//! `csrc/src/batch/` is ~11.6k lines and is mostly not about the GPU: it
//! derives batch shapes from the CSR view the engine marshals, composes
//! channel tickets, colors scratch, and only at the end encodes a forward.
//! The port follows the crate's rule — portable half first, into modules a
//! Linux `cargo test` reaches — and `PARITY-BATCH.md` is its ledger.
//!
//! [`schedule`] is the batch shape: request spans, the token→request
//! expansion, and the paged-geometry gate that runs before any pool cell can
//! be addressed. [`mask`] answers whether a wire attention mask says
//! anything the kernel's own causal predicate does not already enforce.

mod abi;
mod color;
mod consts;
mod dispatch;
mod dispatch_mb;
mod geometry;
mod geometry_facts;
mod logits;
mod psos_mb;
mod sizing;
mod timing;

pub use abi::{
    ArgmaxParams, ForwardGraphKey, IO_SLOT_COUNT, IoSlot, Kernel, PAGE_BUCKET_GRAN, Region,
    SCRATCH_POOL,
};
pub use color::{
    Coloring, ColoringError, ScheduleError, ScratchBind, ScratchSchedule, Use, color_live_ranges,
    schedule_scratch,
};
pub use consts::{
    ExpertCombineParams, GatedRmsParams, GdnCoreParams, KN, MoeRouteParams, RmsParams,
    RouterParams, gdn_core_params, is_qmv, is_routed, qmv_kn,
};
pub use dispatch::{
    DagOptions, Dispatch, Launch, attn_gate, barrier_after, build_decode_dag, concurrent_run_ends,
    embed, gated_rms, kv_append, q_split, qmv, residual, rms, rope, route_rows, route_sort,
    routed_qmv, router_lane_width, router_topk, sdpa, silu_mul,
};
pub use dispatch_mb::{
    PREFILL_ORDINAL_BASE, PREFILL_ORDINAL_STRIDE, ROUTED_DECODE_BATCHED, SDPA_QUERY_TILE,
    build_decode_dag_mb, build_decode_prefill_dags, elementwise_mb, fp16_format, mb_geometry,
    mb_kind, qmm_bm, qmm_bm_slot, qmm_bn, qmm_bn_unsplit, qmm_mb_rows, qmm_t, qmv_mb, qmv_out_size,
    rms_mb, uses_alt_quant,
};
pub use geometry::{AffineFormat, DecodeGeometry};
pub use geometry_facts::{
    GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, geometry_from_facts,
};
pub use logits::{LengthMismatch, bf16_to_f32, widen, widen_into};
pub use psos_mb::{
    MOE_TILE_WIDTHS, MbFeatures, MbRequest, MbSlot, QMM_BMS, QMM_SPLIT_BN, plan_multibatch_psos,
};
pub use sizing::{
    RoutedProjection, RowAxis, Target, ValueExtent, conv_state_target_bytes, kv_pool_row_bytes,
    kv_pool_target_bytes, moe_sorted_rows, pool_colour_elems, recurrent_state_target_bytes,
    ring_target_bytes, row_scaled_target_bytes, scratch_slot_elems, scratch_widest_elems,
    sorted_rows,
};
pub use timing::{
    Ablation, BoundaryMismatch, DispatchAttribution, DispatchInfo, StepAttribution, attribute_step,
};
