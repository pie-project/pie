use core::mem::{align_of, offset_of, size_of};

use crate::attn::fa2::geometry::Device;
use crate::attn::plan::info::{DecodePlanInfo, PrefillPlanInfo};

/// A device address.
pub type DevicePtr = u64;

/// `cuda::fast_mod_div<uint32_t>` — **the shim's**, `shim/cuda/cmath:175`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FastModDiv {
    /// `__divisor_`, `cmath:234`.
    pub divisor: u32,
    /// `__magic_`, `cmath:235`. `floor(2^64 / divisor) + 1`.
    pub magic: u64,
}

impl FastModDiv {
    /// The shim's constructor, `cmath:196-206`, transcribed.
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor };
        let all_ones = u64::MAX;
        let q = all_ones / d as u64;
        let r = all_ones % d as u64;
        let carry = if r + 1 == d as u64 { 1 } else { 0 };
        Self { divisor: d, magic: q.wrapping_add(carry).wrapping_add(1) }
    }
}

const _: () = assert!(size_of::<FastModDiv>() == 16);
const _: () = assert!(align_of::<FastModDiv>() == 8);
const _: () = assert!(offset_of!(FastModDiv, divisor) == 0);
const _: () = assert!(offset_of!(FastModDiv, magic) == 8);

/// `flashinfer::uint_fastdiv`, `fastdiv.cuh`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UintFastdiv {
    /// `uint_fastdiv::impl_`, `fastdiv.cuh`.
    pub magic: FastModDiv,
    /// `uint_fastdiv::d_`, `fastdiv.cuh` — the divisor again, which is what
    pub d: u32,
}

impl UintFastdiv {
    /// `fastdiv.cuh:36`'s guarded `__host__` constructor, in Rust.
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor };
        Self { magic: FastModDiv::new(d), d }
    }
}

const _: () = assert!(size_of::<UintFastdiv>() == 24);
const _: () = assert!(align_of::<UintFastdiv>() == 8);
const _: () = assert!(offset_of!(UintFastdiv, magic) == 0);
const _: () = assert!(offset_of!(UintFastdiv, d) == 16);

/// `flashinfer::paged_kv_t<DTypeKV, IdType>`, `page.cuh:44-65`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PagedKv {
    /// `page.cuh:45`. **24 bytes** — see [`UintFastdiv`].
    pub page_size: UintFastdiv,
    /// `page.cuh:46`. Measured at **+24**, not +20; the shim's `fast_mod_div`
    pub num_heads: u32,
    /// `page.cuh:47`.
    pub head_dim: u32,
    /// `page.cuh:48`.
    pub batch_size: u32,
    /// `page.cuh:49`.
    pub stride_page: u32,
    /// `page.cuh:50`.
    pub stride_n: u32,
    /// `page.cuh:51`.
    pub stride_h: u32,
    /// `page.cuh:56`.
    pub k_data: DevicePtr,
    /// `page.cuh:57`.
    pub v_data: DevicePtr,
    /// `page.cuh:58`.
    pub indices: DevicePtr,
    /// `page.cuh:61` — `[batch_size + 1]`, first element 0, last `nnz_pages`.
    pub indptr: DevicePtr,
    /// `page.cuh:63` — `[batch_size]`, the offset of the last page.
    pub last_page_len: DevicePtr,
    /// `page.cuh:65` — `[batch_size]`, each request's start position.
    pub rope_pos_offset: DevicePtr,
}

const _: () = assert!(size_of::<PagedKv>() == 96);
const _: () = assert!(align_of::<PagedKv>() == 8);
const _: () = assert!(offset_of!(PagedKv, page_size) == 0);
const _: () = assert!(offset_of!(PagedKv, num_heads) == 24);
const _: () = assert!(offset_of!(PagedKv, head_dim) == 28);
const _: () = assert!(offset_of!(PagedKv, batch_size) == 32);
const _: () = assert!(offset_of!(PagedKv, stride_page) == 36);
const _: () = assert!(offset_of!(PagedKv, stride_n) == 40);
const _: () = assert!(offset_of!(PagedKv, stride_h) == 44);
const _: () = assert!(offset_of!(PagedKv, k_data) == 48);
const _: () = assert!(offset_of!(PagedKv, v_data) == 56);
const _: () = assert!(offset_of!(PagedKv, indices) == 64);
const _: () = assert!(offset_of!(PagedKv, indptr) == 72);
const _: () = assert!(offset_of!(PagedKv, last_page_len) == 80);
const _: () = assert!(offset_of!(PagedKv, rope_pos_offset) == 88);

/// `flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>`,
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DecodeParams {
    /// `:111`.
    pub q: DevicePtr,
    /// `:112`.
    pub q_rope_offset: DevicePtr,
    /// `:113`. **96 bytes.**
    pub paged_kv: PagedKv,
    /// `:114`.
    pub o: DevicePtr,
    /// `:115`.
    pub lse: DevicePtr,
    /// `:116`.
    pub maybe_alibi_slopes: DevicePtr,
    /// `:117` — the planner's, not the batch size.
    pub padded_batch_size: u32,
    /// `:118`.
    pub num_qo_heads: u32,
    /// `:119`.
    pub q_stride_n: i32,
    /// `:120`.
    pub q_stride_h: i32,
    /// `:121` — `-1` for no window.
    pub window_left: i32,
    /// `:122`.
    pub logits_soft_cap: f32,
    /// `:123`.
    pub sm_scale: f32,
    /// `:124` — the RECIPROCAL. `1.f / rope_scale`, `:180`.
    pub rope_rcp_scale: f32,
    /// `:125` — the RECIPROCAL. `1.f / rope_theta`, `:181`.
    pub rope_rcp_theta: f32,
    /// `:127`.
    pub request_indices: DevicePtr,
    /// `:128`.
    pub kv_tile_indices: DevicePtr,
    /// `:129`.
    pub o_indptr: DevicePtr,
    /// `:130`.
    pub kv_chunk_size_ptr: DevicePtr,
    /// `:131` — `bool*`. Written only on the split path; an unsplit padded
    pub block_valid_mask: DevicePtr,
    /// `:132` — `bool`, one byte, with seven of tail padding after it.
    pub partition_kv: bool,
}

const _: () = assert!(size_of::<DecodeParams>() == 224);
const _: () = assert!(align_of::<DecodeParams>() == 8);
const _: () = assert!(offset_of!(DecodeParams, q) == 0);
const _: () = assert!(offset_of!(DecodeParams, q_rope_offset) == 8);
const _: () = assert!(offset_of!(DecodeParams, paged_kv) == 16);
const _: () = assert!(offset_of!(DecodeParams, o) == 112);
const _: () = assert!(offset_of!(DecodeParams, lse) == 120);
const _: () = assert!(offset_of!(DecodeParams, maybe_alibi_slopes) == 128);
const _: () = assert!(offset_of!(DecodeParams, padded_batch_size) == 136);
const _: () = assert!(offset_of!(DecodeParams, num_qo_heads) == 140);
const _: () = assert!(offset_of!(DecodeParams, q_stride_n) == 144);
const _: () = assert!(offset_of!(DecodeParams, q_stride_h) == 148);
const _: () = assert!(offset_of!(DecodeParams, window_left) == 152);
const _: () = assert!(offset_of!(DecodeParams, logits_soft_cap) == 156);
const _: () = assert!(offset_of!(DecodeParams, sm_scale) == 160);
const _: () = assert!(offset_of!(DecodeParams, rope_rcp_scale) == 164);
const _: () = assert!(offset_of!(DecodeParams, rope_rcp_theta) == 168);
const _: () = assert!(offset_of!(DecodeParams, request_indices) == 176);
const _: () = assert!(offset_of!(DecodeParams, kv_tile_indices) == 184);
const _: () = assert!(offset_of!(DecodeParams, o_indptr) == 192);
const _: () = assert!(offset_of!(DecodeParams, kv_chunk_size_ptr) == 200);
const _: () = assert!(offset_of!(DecodeParams, block_valid_mask) == 208);
const _: () = assert!(offset_of!(DecodeParams, partition_kv) == 216);

/// `flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>`,
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillPagedParams {
    /// `:327`.
    pub q: DevicePtr,
    /// `:328`. **96 bytes.**
    pub paged_kv: PagedKv,
    /// `:329` — `uint8_t*`.
    pub maybe_custom_mask: DevicePtr,
    /// `:330`.
    pub q_indptr: DevicePtr,
    /// `:331`.
    pub maybe_mask_indptr: DevicePtr,
    /// `:332` — fused-rope attention only.
    pub maybe_q_rope_offset: DevicePtr,
    /// `:333`.
    pub o: DevicePtr,
    /// `:334`.
    pub lse: DevicePtr,
    /// `:335`.
    pub maybe_alibi_slopes: DevicePtr,
    /// `:336`. **24 bytes** — the second `uint_fastdiv`, and the one that puts
    pub group_size: UintFastdiv,
    /// `:337`.
    pub num_qo_heads: u32,
    /// `:338`.
    pub q_stride_n: i32,
    /// `:339`.
    pub q_stride_h: i32,
    /// `:340` — the scale-factor strides, for the FP8/FP4 KV variants. Zero
    pub k_sf_stride_page: u32,
    /// `:341`.
    pub k_sf_stride_n: u32,
    /// `:342`.
    pub k_sf_stride_h: u32,
    /// `:343`.
    pub v_sf_stride_page: u32,
    /// `:344`.
    pub v_sf_stride_n: u32,
    /// `:345`.
    pub v_sf_stride_h: u32,
    /// `:346` — `-1` for no window.
    pub window_left: i32,
    /// `:347`.
    pub logits_soft_cap: f32,
    /// `:348`.
    pub sm_scale: f32,
    /// `:349` — the RECIPROCAL.
    pub rope_rcp_scale: f32,
    /// `:350` — the RECIPROCAL.
    pub rope_rcp_theta: f32,
    /// `:352`.
    pub request_indices: DevicePtr,
    /// `:353`.
    pub qo_tile_indices: DevicePtr,
    /// `:354`.
    pub kv_tile_indices: DevicePtr,
    /// `:355`.
    pub merge_indptr: DevicePtr,
    /// `:356`.
    pub o_indptr: DevicePtr,
    /// `:357` — `bool*`.
    pub block_valid_mask: DevicePtr,
    /// `:358`.
    pub kv_chunk_size_ptr: DevicePtr,
    /// `:359` — the graph-capture bound, a VALUE.
    pub max_total_num_rows: u32,
    /// `:360` — `uint32_t*`, a device POINTER, and not the same thing as the
    pub total_num_rows: DevicePtr,
    /// `:361`.
    pub padded_batch_size: u32,
    /// `:362`.
    pub partition_kv: bool,
    /// `:363` — `uint32_t*`.
    pub maybe_prefix_len_ptr: DevicePtr,
    /// `:364` — `uint16_t*`.
    pub maybe_token_pos_in_items_ptr: DevicePtr,
    /// `:365`.
    pub token_pos_in_items_len: u32,
    /// `:366` — `uint16_t*`.
    pub maybe_max_item_len_ptr: DevicePtr,
}

const _: () = assert!(size_of::<PrefillPagedParams>() == 352);
const _: () = assert!(align_of::<PrefillPagedParams>() == 8);
const _: () = assert!(offset_of!(PrefillPagedParams, q) == 0);
const _: () = assert!(offset_of!(PrefillPagedParams, paged_kv) == 8);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_custom_mask) == 104);
const _: () = assert!(offset_of!(PrefillPagedParams, q_indptr) == 112);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_mask_indptr) == 120);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_q_rope_offset) == 128);
const _: () = assert!(offset_of!(PrefillPagedParams, o) == 136);
const _: () = assert!(offset_of!(PrefillPagedParams, lse) == 144);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_alibi_slopes) == 152);
const _: () = assert!(offset_of!(PrefillPagedParams, group_size) == 160);
const _: () = assert!(offset_of!(PrefillPagedParams, num_qo_heads) == 184);
const _: () = assert!(offset_of!(PrefillPagedParams, q_stride_n) == 188);
const _: () = assert!(offset_of!(PrefillPagedParams, q_stride_h) == 192);
const _: () = assert!(offset_of!(PrefillPagedParams, k_sf_stride_page) == 196);
const _: () = assert!(offset_of!(PrefillPagedParams, k_sf_stride_n) == 200);
const _: () = assert!(offset_of!(PrefillPagedParams, k_sf_stride_h) == 204);
const _: () = assert!(offset_of!(PrefillPagedParams, v_sf_stride_page) == 208);
const _: () = assert!(offset_of!(PrefillPagedParams, v_sf_stride_n) == 212);
const _: () = assert!(offset_of!(PrefillPagedParams, v_sf_stride_h) == 216);
const _: () = assert!(offset_of!(PrefillPagedParams, window_left) == 220);
const _: () = assert!(offset_of!(PrefillPagedParams, logits_soft_cap) == 224);
const _: () = assert!(offset_of!(PrefillPagedParams, sm_scale) == 228);
const _: () = assert!(offset_of!(PrefillPagedParams, rope_rcp_scale) == 232);
const _: () = assert!(offset_of!(PrefillPagedParams, rope_rcp_theta) == 236);
const _: () = assert!(offset_of!(PrefillPagedParams, request_indices) == 240);
const _: () = assert!(offset_of!(PrefillPagedParams, qo_tile_indices) == 248);
const _: () = assert!(offset_of!(PrefillPagedParams, kv_tile_indices) == 256);
const _: () = assert!(offset_of!(PrefillPagedParams, merge_indptr) == 264);
const _: () = assert!(offset_of!(PrefillPagedParams, o_indptr) == 272);
const _: () = assert!(offset_of!(PrefillPagedParams, block_valid_mask) == 280);
const _: () = assert!(offset_of!(PrefillPagedParams, kv_chunk_size_ptr) == 288);
const _: () = assert!(offset_of!(PrefillPagedParams, max_total_num_rows) == 296);
const _: () = assert!(offset_of!(PrefillPagedParams, total_num_rows) == 304);
const _: () = assert!(offset_of!(PrefillPagedParams, padded_batch_size) == 312);
const _: () = assert!(offset_of!(PrefillPagedParams, partition_kv) == 316);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_prefix_len_ptr) == 320);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_token_pos_in_items_ptr) == 328);
const _: () = assert!(offset_of!(PrefillPagedParams, token_pos_in_items_len) == 336);
const _: () = assert!(offset_of!(PrefillPagedParams, maybe_max_item_len_ptr) == 344);

/// `fa2::DecodeScoreParams` — `PieScoreParams<BatchDecodeParams, int32_t>`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DecodeScoreParams {
    /// The `Base` subobject — every field the uncaptured decode fills.
    pub base: DecodeParams,
    /// `float*`, the ragged score sink:
    pub score_out: DevicePtr,
    /// `const IdType*`, `R + 1` entries, last is the total element count.
    pub score_indptr: DevicePtr,
}

const _: () = assert!(size_of::<DecodeScoreParams>() == 240);
const _: () = assert!(align_of::<DecodeScoreParams>() == 8);
const _: () = assert!(offset_of!(DecodeScoreParams, base) == 0);
const _: () = assert!(offset_of!(DecodeScoreParams, score_out) == 224);
const _: () = assert!(offset_of!(DecodeScoreParams, score_indptr) == 232);

/// `fa2::PrefillScoreParams` —
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillScoreParams {
    /// The `Base` subobject — every field the uncaptured prefill fills.
    pub base: PrefillPagedParams,
    /// `float*`, the ragged score sink:
    pub score_out: DevicePtr,
    /// `const IdType*`, `R + 1` entries.
    pub score_indptr: DevicePtr,
    /// `std::uint32_t` — the observation window's width in query rows.
    pub score_window: u32,
}

const _: () = assert!(size_of::<PrefillScoreParams>() == 376);
const _: () = assert!(align_of::<PrefillScoreParams>() == 8);
const _: () = assert!(offset_of!(PrefillScoreParams, base) == 0);
const _: () = assert!(offset_of!(PrefillScoreParams, score_out) == 352);
const _: () = assert!(offset_of!(PrefillScoreParams, score_indptr) == 360);
const _: () = assert!(offset_of!(PrefillScoreParams, score_window) == 368);

// ── The filling ─────────────────────────────────────────────────────────────
//
// # ONE FILLER AND ONE READER, WHICH IS WHAT CLOSED THE HAZARD
//
// Everything above is pinned to the layout `shim/cuda/cmath`'s
// `__fast_div_modulo` produces -- `{u32 @0, u64 @8}`, align 8, putting
// `paged_kv_t::num_heads` at **+24**. The deleted `attention_flashinfer.cu`
// compiled against real CCCL, whose `uint_fastdiv` is `{u32,u32,u32,i32}`
// align 4 and puts the same field at **+20**, with `sizeof` reconverging at 96
// under both. Both were correct for their own reader and **a block filled on
// one side and read on the other is a silent wrong answer, not a crash**.
//
// The filling therefore lives beside the mirrors it fills, in the crate whose
// `const _: () = assert!(offset_of!(..))` lines are the layout: a second
// filler in another crate could not be checked by them.

/// Where a fire reads and writes. Every field is a device address.
///
/// One struct rather than fourteen positional arguments, which is what
/// `dispatch_attention_flashinfer_decode_bf16` (`:490-503`) had. The C++ could
/// not do this without a header both sides included; Rust can.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buffers {
    /// `[num_tokens, num_q_heads, head_dim]`, or one row broadcast — see
    /// `broadcast_q`.
    pub q: DevicePtr,
    /// The paged K cache.
    pub k_pages: DevicePtr,
    /// The paged V cache.
    pub v_pages: DevicePtr,
    /// `[num_tokens, num_q_heads, head_dim]`, written.
    pub o: DevicePtr,
    /// `[nnz_pages]`.
    pub kv_page_indices: DevicePtr,
    /// `[batch_size + 1]`.
    pub kv_page_indptr: DevicePtr,
    /// `[batch_size]`.
    pub kv_last_page_lens: DevicePtr,
    /// `[batch_size + 1]` QO row offsets. Prefill only; decode has one row per
    /// request and passes 0.
    pub qo_indptr: DevicePtr,
    /// Optional `[num_tokens, num_q_heads]` log-sum-exp output, or 0.
    pub lse: DevicePtr,
    /// The plan's int workspace. The descriptor was uploaded here.
    pub int_buffer: DevicePtr,
    /// The plan's float workspace, where the split path stages partials.
    pub float_buffer: DevicePtr,
}

/// The split-path staging pointers a split fire leaves behind, and everything
/// the fold that consumes them needs.
///
/// # Why this carries nine fields and not two
///
/// They are filled in [`make_decode_params`] and [`make_prefill_params`],
/// because this is where they are unambiguous: the same function that
/// redirects `params.o` to `tmp_v` is the one that knows what `o` used to be.
/// A caller one layer up also holds an `o` and a head count, and would be
/// filling them from a second reading of the same request — two derivations of
/// one fact, with nothing checking that they agree. `prefill.cuh:4339-4342`
/// makes the same choice for the same reason: it saves `o` and `lse` into
/// locals immediately before overwriting the fields.
///
/// The two dispatches disagree about three of the nine, and the disagreement
/// is upstream's:
///
/// | | prefill (`prefill.cuh:4350-4352`) | decode (`decode.cuh:822-824`) |
/// |---|---|---|
/// | `indptr`      | `params.merge_indptr` | `params.o_indptr` |
/// | `max_seq_len` | `params.max_total_num_rows` | `params.paged_kv.batch_size` |
/// | `seq_len`     | `params.total_num_rows` | `nullptr` |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Partials {
    /// Partial outputs, `plan_info.v_offset` into the float workspace.
    ///
    /// **`params.o` points here too**, after the redirect. That is the whole
    /// mechanism: the attention kernel writes its per-chunk answers where the
    /// merge will read them.
    pub tmp_v: DevicePtr,
    /// Partial log-sum-exps, `plan_info.s_offset`. `params.lse` points here.
    pub tmp_s: DevicePtr,
    /// Where each merged row's partials start.
    ///
    /// `plan_info.merge_indptr_offset` for prefill, `plan_info.o_indptr_offset`
    /// for decode — see the table above.
    pub indptr: DevicePtr,
    /// The caller's real output, saved before `params.o` was redirected.
    pub o: DevicePtr,
    /// The caller's real log-sum-exp output, or 0. Saved likewise.
    pub lse: DevicePtr,
    /// Rows to fold: `plan_info.total_num_rows` for prefill, the request
    /// count for decode.
    pub max_seq_len: u32,
    /// A DEVICE `uint32_t*` overriding `max_seq_len`, or 0.
    ///
    /// Taken from `params.total_num_rows` verbatim rather than recomputed, so
    /// the fold folds exactly the rows the attention kernel wrote. See
    /// [`make_prefill_params`] for the one case where that is currently 0 and
    /// upstream's would not be.
    pub seq_len: DevicePtr,
    /// Query heads.
    pub num_heads: u32,
    /// 64, 128, 256 or 512.
    pub head_dim: u32,
}

/// What a decode fire reads out of `driver-cuda`'s plan cache, and the device
/// it will be launched on.
///
/// **This is the destructuring, and it is the whole of what crosses.** A
/// `DecodePlanCache` owns `Vec`s and is re-planned once per fire; none of that
/// can be an argument. What a launch reads out of one is this: the descriptor,
/// seven shape scalars and two flags — all `Copy`, all fixed for the fire.
///
/// `device` rides along because the geometry is derived against it and the
/// plan was sized against the same part. A fire that paired one plan with
/// another device's shared-memory budget would derive a valid-looking
/// `NUM_MMA_KV` for a kernel nothing launches, which is the pairing
/// `driver-cuda`'s own `facts()` exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    /// The descriptor the kernel reads, whose offsets index `int_buffer`.
    pub info: DecodePlanInfo,
    /// The device the geometry is derived against. See this struct's doc.
    pub device: Device,
    /// The batch this plan was built for.
    pub num_requests: i32,
    /// Query heads.
    pub num_q_heads: i32,
    /// KV heads. `num_q_heads / num_kv_heads` is the GQA group that picks the
    /// lattice point.
    pub num_kv_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// Tokens per page.
    pub page_size: i32,
    /// Byte offset of this plan's descriptor inside the shared int workspace.
    ///
    /// Not zero in general: several layers' plans share one int buffer, and
    /// the decode planner records where this one starts.
    pub int_base_bytes: u64,
    /// HND page layout.
    pub hnd_layout: bool,
    /// Whether the plan was built for the full-attention variant.
    pub full_attention_variant: bool,
    /// Whether anything above was written.
    pub valid: bool,
}

/// [`DecodePlan`]'s twin, for prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    /// The FA2 descriptor.
    pub info: PrefillPlanInfo,
    /// The device the geometry is derived against. See [`DecodePlan`].
    pub device: Device,
    /// Requests in the batch.
    pub num_requests: i32,
    /// Query heads.
    pub num_q_heads: i32,
    /// KV heads.
    pub num_kv_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// Tokens per page.
    pub page_size: i32,
    /// The `CTA_TILE_Q` this plan was built at, which names the prefill root.
    /// **Read back from the plan, never recomputed**: the planner split the
    /// batch against this tile.
    pub cta_tile_q: u32,
    /// Sliding-window span, `-1` for full attention. **From the CACHE, not
    /// from the request**: the window was fixed at planning time because the
    /// split was sized against it.
    pub window_left: i32,
    /// HND page layout.
    pub hnd_layout: bool,
    /// Whether the plan was built for the full-attention variant.
    pub full_attention_variant: bool,
    /// Whether the mask is causal.
    pub causal_mask: bool,
    /// Whether the SM90 route was chosen. **This lattice has not ported it**,
    /// so a fire refuses rather than reading an FA2 descriptor out of an FA3
    /// plan.
    pub use_sm90: bool,
    /// Whether anything above was written.
    pub valid: bool,
}

impl DecodePlan {
    /// `num_q_heads / num_kv_heads`, the GQA group that names the root.
    ///
    /// One, not zero, for a cache that states no KV heads: a divide here would
    /// be a fire that trapped on a plan it could have refused.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        if self.num_kv_heads > 0 { (self.num_q_heads / self.num_kv_heads) as u32 } else { 1 }
    }
}

impl PrefillPlan {
    /// [`DecodePlan::group_size`]'s twin — the divisor `group_size`'s
    /// `uint_fastdiv` carries.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        if self.num_kv_heads > 0 { (self.num_q_heads / self.num_kv_heads) as u32 } else { 1 }
    }
}

/// `GetPtrFromBaseOffset`, `attention_flashinfer_common.cuh:174-177`:
/// `(base + offset_bytes)`.
///
/// A saturating add rather than a wrapping one. Upstream's is a pointer
/// arithmetic that would be UB on overflow; here an overflow can only come
/// from a corrupt plan, and saturating to `u64::MAX` produces an address the
/// device faults on immediately rather than one that aliases the workspace.
const fn offset_ptr(base: DevicePtr, off: i64) -> DevicePtr {
    if off < 0 { base } else { base.saturating_add(off as u64) }
}

/// `sm_scale > 0 ? sm_scale : 1/sqrt(head_dim)`.
///
/// `attention_flashinfer_common.cuh:603-605` and
/// `attention_flashinfer.cu:735-737`, identically in both.
#[must_use]
pub fn sm_scale_or_default(sm_scale: f32, head_dim: i32) -> f32 {
    if sm_scale > 0.0 { sm_scale } else { 1.0 / (head_dim as f32).sqrt() }
}

/// `paged_kv_t`'s guarded `__host__` constructor, `page.cuh:103-120`.
///
/// The three strides are computed here because that constructor is
/// `#ifndef __CUDACC_RTC__` and device code never runs one — its `// PIE:`
/// marker says *"Under the JIT this struct is filled by the Rust caller"*.
/// This is that caller.
///
/// `hnd_layout` is `QKVLayout::kHND`; false is `kNHD`. `page.cuh:118-119`:
///
/// ```text
/// stride_page = num_heads * page_size * head_dim
/// stride_n    = kHND ? head_dim            : num_heads * head_dim
/// stride_h    = kHND ? page_size * head_dim : head_dim
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn make_paged_kv(
    num_heads: u32,
    page_size: u32,
    head_dim: u32,
    batch_size: u32,
    hnd_layout: bool,
    k_data: DevicePtr,
    v_data: DevicePtr,
    indices: DevicePtr,
    indptr: DevicePtr,
    last_page_len: DevicePtr,
) -> PagedKv {
    PagedKv {
        page_size: UintFastdiv::new(page_size),
        num_heads,
        head_dim,
        batch_size,
        stride_page: num_heads.wrapping_mul(page_size).wrapping_mul(head_dim),
        stride_n: if hnd_layout { head_dim } else { num_heads.wrapping_mul(head_dim) },
        stride_h: if hnd_layout { page_size.wrapping_mul(head_dim) } else { head_dim },
        k_data,
        v_data,
        indices,
        indptr,
        last_page_len,
        // Left null, and `run_decode`'s comment at
        // `attention_flashinfer_common.cuh:614-619` is why it must stay null:
        // `PieScoreCapture` records `kv_idx` verbatim and the kernel derives
        // it from `rope_pos_offset` (`decode.cuh:541`), so a non-null value
        // would silently land every captured score at the wrong position.
        // The C++ asserted this at runtime on the capture path; here it is
        // structural, because nothing writes the field.
        rope_pos_offset: 0,
    }
}

/// `run_decode`'s params filling, `attention_flashinfer_common.cuh:581-641`.
///
/// Returns the struct and the split-path staging pointers, which the C++
/// returned through two out-references (`tmp_v`, `tmp_s`).
#[must_use]
pub fn make_decode_params(
    plan: &DecodePlan,
    bufs: &Buffers,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    broadcast_q: bool,
) -> (DecodeParams, Partials) {
    let info = &plan.info;
    let mut p = DecodeParams {
        q: bufs.q,
        // `:583` — always null on this path. Fused rope is not wired.
        q_rope_offset: 0,
        paged_kv: make_paged_kv(
            plan.num_kv_heads as u32,
            plan.page_size as u32,
            plan.head_dim as u32,
            plan.num_requests as u32,
            plan.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        o: bufs.o,
        lse: bufs.lse,
        // `:586`.
        maybe_alibi_slopes: 0,
        num_qo_heads: plan.num_q_heads as u32,
        // `:588-590`. **Zero, not a stride**, when `broadcast_q`: one query row
        // is read by every token. That is how a single decoded token feeds a
        // batch, and a mirror that "fixed" the zero would read past the row.
        q_stride_n: if broadcast_q { 0 } else { plan.num_q_heads * plan.head_dim },
        q_stride_h: plan.head_dim,
        window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, plan.head_dim),
        // `:598-599`. Rope is not fused into FA2 here; it ran earlier as its
        // own kernel, so both reciprocals are 1 and the kernel's rope path is
        // an identity.
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..DecodeParams::default()
    };

    // `:632-633`. The decode int base is NOT the workspace base: several
    // layers' plans share one int buffer and `set_decode_plan_int_base`
    // (`attention_flashinfer.cu:215-217`) says where this one starts.
    let int_buf = bufs.int_buffer.saturating_add(plan.int_base_bytes);
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    let mut partials = Partials::default();
    if info.split_kv {
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);
        // `decode.cuh:809-812`. **The redirect, and it is the whole split
        // mechanism**: the attention kernel writes per-chunk partials to
        // `tmp_v`/`tmp_s` and `o`/`lse` are filled by the merge afterwards.
        // Without these two lines the kernel writes partial answers straight
        // into the caller's output and the merge folds a buffer nothing
        // staged.
        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;
        // `decode.cuh:823`. Decode's indptr is `o_indptr` and its row count
        // is the batch size; there is no `merge_indptr` on this path because
        // a decode step has exactly one query row per request.
        partials.indptr = p.o_indptr;
        partials.max_seq_len = plan.num_requests as u32;
        // `decode.cuh:823` passes `nullptr`.
        partials.seq_len = 0;
        partials.num_heads = plan.num_q_heads as u32;
        partials.head_dim = plan.head_dim as u32;
        // `:648-651`. Only under graph capture: outside it the grid is exactly
        // the work list and every block is valid.
        if info.enable_cuda_graph {
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

/// `make_prefill_params`, `attention_flashinfer.cu:693-775`.
#[must_use]
pub fn make_prefill_params(
    plan: &PrefillPlan,
    bufs: &Buffers,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> (PrefillPagedParams, Partials) {
    let info = &plan.info;
    let mut p = PrefillPagedParams {
        q: bufs.q,
        paged_kv: make_paged_kv(
            plan.num_kv_heads as u32,
            plan.page_size as u32,
            plan.head_dim as u32,
            plan.num_requests as u32,
            plan.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        // `:722` — null. The causal mask is a KernelTraits constant, not a
        // buffer; `maybe_custom_mask` is upstream's arbitrary-mask path and
        // pie does not use it.
        maybe_custom_mask: 0,
        q_indptr: bufs.qo_indptr,
        maybe_mask_indptr: 0,
        maybe_q_rope_offset: 0,
        o: bufs.o,
        lse: bufs.lse,
        maybe_alibi_slopes: 0,
        // `:728-729`. **24 bytes, and a computed magic** — see
        // [`UintFastdiv`]. The prefill kernel divides by the GQA group on
        // every row, which is why upstream carries a reciprocal rather than a
        // divisor.
        group_size: UintFastdiv::new(plan.group_size()),
        num_qo_heads: plan.num_q_heads as u32,
        // `:731`. Note there is no `broadcast_q` here: prefill always has a
        // real QO row per token, so the decode path's zero stride has no
        // prefill analogue.
        q_stride_n: plan.num_q_heads * plan.head_dim,
        q_stride_h: plan.head_dim,
        // `:733` — from the PLAN, not the request. The window was fixed at
        // planning time because the split was sized against it.
        window_left: plan.window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, plan.head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..PrefillPagedParams::default()
    };

    // `:742`. Prefill reads the workspace base directly — there is no prefill
    // analogue of `int_base_bytes`, because one prefill plan serves one fire.
    let int_buf = bufs.int_buffer;
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.qo_tile_indices = offset_ptr(int_buf, info.qo_tile_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;
    // `:753`. A VALUE, and the field below it is a POINTER — the names differ
    // by two characters and the types by eight bytes.
    p.max_total_num_rows = info.total_num_rows as u32;
    p.total_num_rows = 0;

    let mut partials = Partials::default();
    if info.split_kv {
        p.merge_indptr = offset_ptr(int_buf, info.merge_indptr_offset);
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);
        // `prefill.cuh:4339-4342` — the redirect. See [`make_decode_params`]
        // for what it is for; the two are the same three lines.
        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;
        // `prefill.cuh:4351`. Prefill folds by ROW rather than by request —
        // `merge_indptr` has `total_num_rows + 1` entries
        // (`plan/prefill.rs:124`) — which is why it has an indptr of its own
        // where decode reuses `o_indptr`.
        partials.indptr = p.merge_indptr;
        partials.max_seq_len = p.max_total_num_rows;
        // **Verbatim, not recomputed.** `prefill.cuh:4352` passes
        // `params.total_num_rows`, and the field is written 0 four lines
        // above. So the fold uses `max_total_num_rows`, which is exactly what
        // the attention kernel used, and the two cannot disagree.
        //
        // Under `enable_cuda_graph` upstream would have a real pointer here —
        // `plan/prefill.rs:414-416` allocates `total_num_rows_offset` only in
        // that mode, for a grid captured with a dummy row count. This driver
        // does not fill it, on either side, and reading it here would make
        // the merge fold a different row count from the kernel that produced
        // the partials. That gap is FA2's and predates the split; it is
        // recorded here because this is where a reader will look for it.
        partials.seq_len = p.total_num_rows;
        partials.num_heads = plan.num_q_heads as u32;
        partials.head_dim = plan.head_dim as u32;
        if info.enable_cuda_graph {
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

#[cfg(test)]
mod tests {
    use super::{FastModDiv, UintFastdiv, offset_ptr};

    /// The shim's magic, checked the way the shim's own banner says it should
    #[test]
    fn the_magic_divides() {
        for d in [1u32, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256] {
            let m = FastModDiv::new(d);
            assert_eq!(m.divisor, d, "the divisor is kept verbatim");
            for n in [0u32, 1, 2, d, d - 1, d + 1, 63, 64, 65, 1023, 4096, u32::MAX] {
                if d == 1 {
                    continue;
                }
                let hi = ((u128::from(n) * u128::from(m.magic)) >> 64) as u32;
                assert_eq!(hi, n / d, "mul.hi disagreed with the divide at n={n} d={d}");
            }
        }
    }

    /// A zero divisor recovers rather than traps, exactly as `fastdiv.cuh:36`
    #[test]
    fn a_zero_divisor_becomes_one() {
        assert_eq!(UintFastdiv::new(0).d, 1);
        assert_eq!(UintFastdiv::new(0).magic.divisor, 1);
    }

    /// A corrupt plan produces an address that faults, not one that aliases.
    #[test]
    fn a_negative_offset_does_not_walk_backwards() {
        assert_eq!(offset_ptr(4096, -8), 4096);
        assert_eq!(offset_ptr(4096, 8), 4104);
        assert_eq!(offset_ptr(u64::MAX, 8), u64::MAX);
    }
}
