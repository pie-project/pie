//! `#[repr(C)]` mirrors of the two FlashInfer FA2 params structs, pinned
//! field-for-field against layouts **measured out of NVRTC's PTX**.
//!
//! # Why these exist
//!
//! Every FA2 `__global__` takes exactly one argument: a `__grid_constant__`
//! params struct, by value. `decode.cuh:618-621` and `prefill.cuh:3966-3967`
//! take no others. So the whole calling convention of this lattice is these
//! two structs' byte layouts, and a mirror wrong in one field is a kernel
//! dereferencing a stride — not a wrong answer, a fault.
//!
//! # Where the numbers came from
//!
//! **Not from reading the header.** `default_decode_params.cuh:105-190` and
//! `default_prefill_params.cuh:321-366` give the field ORDER, and the order is
//! all a human can safely read off: two of these fields are `uint_fastdiv`,
//! which is 24 bytes and looks exactly like a `uint32_t` at the declaration
//! site.
//!
//! The offsets were measured by `nvrtc-probes/params_layout.py`, which
//! compiles the real struct under the real carried header set and reads the
//! offsets back out of the emitted PTX. **Re-running that script is how these
//! assertions get checked against a new upstream**, and the script is the
//! artefact — a table in a comment goes stale the first time a field is added,
//! and silently.
//!
//! Its method is worth keeping, because two obvious routes do not work under
//! NVRTC:
//!
//! - `offsetof` is not available — neither the macro nor `__builtin_offsetof`.
//!   `offsetof(S, b)` parses as a cast and fails with *"type name is not
//!   allowed"*.
//! - `(unsigned)(unsigned long long)&(((S*)0)->b)` is rejected both as a
//!   `__constant__` initialiser (*"dynamic initialization is not supported"*)
//!   and inside an `enum` (*"expression must have a constant value"*).
//! - `(unsigned)((char*)&((S*)0)->b - (char*)(S*)0)` **works**. The
//!   *difference* of two pointers folds to a constant; the cast of one does
//!   not.
//!
//! # The two traps, neither visible from a field name
//!
//! 1. **`paged_kv_t::page_size` and `BatchPrefillPagedParams::group_size` are
//!    `uint_fastdiv`, 24 bytes, not 4.** Read either as a `u32` and every
//!    following field shifts by twenty: decode's `num_heads` sits at **24**,
//!    prefill's `num_qo_heads` at **184**.
//! 2. **Decode has four bytes of tail padding at 172**, where the float block
//!    meets the pointer block — `rope_rcp_theta` at 168, `request_indices` at
//!    176. A `#[repr(packed)]` mirror loses it. These are `#[repr(C)]` and the
//!    padding falls out of the alignment rules; the assertions are what proves
//!    it fell out where the device put it.
//!
//! # The trap the measurement did NOT show, and it is the larger one
//!
//! `csrc/shim/cuda/cmath` carries this crate's own `cuda::fast_mod_div`, and
//! its banner states the hazard in as many words:
//!
//! > this shim  `{ u32 __divisor_ @0, u64 __magic_ @8 }`   size 16, align 8
//! > CCCL       `{ u32 __divisor @0, u32 __multiplier @4, u32 __add @8,
//! >              i32 __shift @12 }`                       size 16, align 4
//! >
//! > `paged_kv_t::num_heads` measured at **+24 under this shim and +20 under
//! > CCCL**, with the totals reconverging at 96 by luck of the pointer block
//! > that follows.
//!
//! > **So: a Rust mirror of anything containing a `uint_fastdiv` must be
//! > pinned against THIS layout, and must say so.**
//!
//! This is that mirror and this is it saying so. `params_layout.py` compiled
//! under the carried set, so it measured the **shim's** layout, which is the
//! one the JIT compiles against — correct for every fire in this crate.
//!
//! It is *not* the layout `driver-cuda`'s surviving `cc::Build` produced:
//! that one pointed at real CCCL through `DEP_PIE_KERNELS_CUDA_CCCL`, so the
//! ahead-of-time `attention_flashinfer.cu` filled a `paged_kv_t` whose
//! `num_heads` was at +20. **The two paths must never share a filled params
//! block**, and they cannot once that build is deleted — which is the same
//! deletion this module exists to enable. Until then, the C++ fills its own
//! and the Rust fills its own, and neither reads the other's.
//!
//! # What is deliberately not here
//!
//! - **No `SingleDecodeParams`, no ragged prefill, no MLA params.** The four
//!   dispatches this lattice serves are all paged and all batched. Adding one
//!   is adding a struct and its assertions, and the probe script already takes
//!   a type name on the command line.
//! - **No constructors mirroring upstream's.** Both structs' `__host__`
//!   constructors are `#ifndef __CUDACC_RTC__`-guarded in the vendored headers
//!   (`default_prefill_params.cuh:368-373` says why: NVRTC refuses an
//!   explicitly `__host__` function outright), so device code never runs one.
//!   Rust fills the fields. [`Default`] here is all-zero, which is what
//!   upstream's default constructor writes.
//! - **No `Drop`, no ownership.** Every pointer field is a `u64` device
//!   address whose lifetime belongs to the caller's allocator.

use core::mem::{align_of, offset_of, size_of};

/// A device address.
///
/// `u64` and not `*mut T` on purpose: these are `CUdeviceptr`s, and a raw
/// pointer in a Rust struct invites a host dereference that would fault at
/// best. Nothing in this module may be dereferenced on the host.
pub type DevicePtr = u64;

/// `cuda::fast_mod_div<uint32_t>` — **the shim's**, `csrc/shim/cuda/cmath:175`.
///
/// See the module banner: CCCL's has the same size and a different layout, and
/// the difference moves `paged_kv_t::num_heads` by four bytes.
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
    ///
    /// `floor(2^64/d)` without a 128-bit intermediate: `2^64-1 = q*d + r`, so
    /// `2^64 = q*d + (r+1)`, which carries into the quotient exactly when
    /// `r+1 == d`.
    ///
    /// Precondition, upstream's and ours: `divisor > 0`. This takes `d ? d : 1`
    /// as `fastdiv.cuh:36` does, so a zero divisor produces a magic of 1 and
    /// quotients of 0 — garbage, not a trap, exactly as a hardware divide by
    /// zero on the device produces garbage.
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor };
        let all_ones = u64::MAX;
        let q = all_ones / d as u64;
        let r = all_ones % d as u64;
        let carry = if r + 1 == d as u64 { 1 } else { 0 };
        Self { divisor: d, magic: q + carry + 1 }
    }
}

const _: () = assert!(size_of::<FastModDiv>() == 16);
const _: () = assert!(align_of::<FastModDiv>() == 8);
const _: () = assert!(offset_of!(FastModDiv, divisor) == 0);
const _: () = assert!(offset_of!(FastModDiv, magic) == 8);

/// `flashinfer::uint_fastdiv`, `fastdiv.cuh:26-49`.
///
/// **24 bytes.** It is written at a declaration site as one word — `uint_fastdiv
/// page_size;` — and a mirror that reads it as a `u32` shifts every following
/// field by twenty.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UintFastdiv {
    /// `impl_`, `fastdiv.cuh:47`.
    pub magic: FastModDiv,
    /// `d_`, `fastdiv.cuh:48` — the divisor again, which is what
    /// `operator unsigned int()` at `:39` returns and what `divmod`'s
    /// remainder at `:43` multiplies by.
    pub d: u32,
}

impl UintFastdiv {
    /// `fastdiv.cuh:36`'s guarded `__host__` constructor, in Rust.
    ///
    /// That constructor is exactly the reason this type is filled here: the
    /// vendored header's `// PIE:` marker at `fastdiv.cuh:29-34` says *"Under
    /// the JIT a divisor is computed by the Rust caller and arrives inside the
    /// params struct, so device code never constructs one."* This is that
    /// caller.
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
///
/// Measured `sizeof` 96, `alignof` 8 at `<bf16, int32_t>`. Every instantiation
/// this lattice uses has 2-byte or 1-byte KV and 4-byte ids, so the layout is
/// pointer-dominated and does not vary with the type arguments — but nothing
/// asserts that, and a `<..., int64_t>` instantiation would need its own
/// measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PagedKv {
    /// `page.cuh:45`. **24 bytes** — see [`UintFastdiv`].
    pub page_size: UintFastdiv,
    /// `page.cuh:46`. Measured at **+24**, not +20; the shim's `fast_mod_div`
    /// is what puts it there.
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
/// `default_decode_params.cuh:105-190`.
///
/// Measured `sizeof` 224, `alignof` 8 at `<bf16, bf16, bf16, int32_t>` — the
/// only instantiation the decode half of this lattice compiles.
///
/// This is the single argument of
/// `BatchDecodeWithPagedKVCacheKernel<..., BatchDecodeParams<...>>`
/// (`decode.cuh:618-621`), passed by value as a `__grid_constant__`.
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
    ///
    /// Four bytes of tail padding follow this field, at 172. See the banner.
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
    /// grid that read it would read uninitialised work assignments, which is
    /// the reason `attention_flashinfer.cu:386-388` refuses graph capture with
    /// split disabled.
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
/// `default_prefill_params.cuh:321-366`.
///
/// Measured `sizeof` 352, `alignof` 8 at `<bf16, bf16, bf16, int32_t>`.
///
/// This is the second of the two template arguments of
/// `BatchPrefillWithPagedKVCacheKernel<KTraits, Params>` (`prefill.cuh:3966-3967`
/// — it takes **two**, not fifteen; the fifteen belong to `KernelTraits`).
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
    /// `num_qo_heads` at 184 rather than 164.
    pub group_size: UintFastdiv,
    /// `:337`.
    pub num_qo_heads: u32,
    /// `:338`.
    pub q_stride_n: i32,
    /// `:339`.
    pub q_stride_h: i32,
    /// `:340` — the scale-factor strides, for the FP8/FP4 KV variants. Zero
    /// for every bf16 instantiation this lattice compiles.
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
    /// field above it despite the names. `attention_flashinfer.cu:755` sets it
    /// null outside capture.
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

// ─── the two capture params, DERIVED and then MEASURED ──────────────────────
//
// `attention_score_capture.cuh:44-63` adds fields by DERIVATION:
//
// ```text
// template <typename Base, typename IdT> struct PieScoreParams : Base {
//     float* score_out = nullptr;
//     const IdT* score_indptr = nullptr;
// };
// template <typename Base, typename IdT>
// struct PieScoreWindowParams : PieScoreParams<Base, IdT> {
//     std::uint32_t score_window = 0;
// };
// ```
//
// # THE PROBE WAS RUN, AND THE DERIVATION HELD IN ALL SIX PLACES
//
// `params_layout.py`'s technique, pointed at these two: a `__constant__`
// array of `(unsigned)((char*)&((P*)0)->field - (char*)(P*)0)` compiled by
// NVRTC for `compute_89`, with the initialiser read back out of the emitted
// PTX. `offsetof` and every other constant-expression route is rejected by
// NVRTC, which is why the offsets are laundered through a `__constant__`.
// No GPU is involved and nothing is linked — NVRTC compiles to PTX in
// process.
//
// Measured, and every value below matches the assertion it was checked
// against:
//
// * `PieScoreParams<BatchDecodeParams<bf16,bf16,bf16,i32>, i32>` —
//   `sizeof` **240**, `alignof` **8**, `score_out` **224**,
//   `score_indptr` **232**.
// * `PieScoreWindowParams<BatchPrefillPagedParams<bf16,bf16,bf16,i32>, i32>`
//   — `sizeof` **376**, `alignof` **8**, `score_out` **352**,
//   `score_indptr` **360**, `score_window` **368**.
//
// The same run re-measured the two BASE structs through the derived ones
// (the base subobject is at 0, so its offsets come out of the same array):
// all 21 `DecodeParams` offsets and all 39 `PrefillPagedParams` offsets
// asserted above reproduced exactly, including `paged_kv` at **16** with `o`
// at **112**, i.e. `sizeof(paged_kv_t)` = **96**.
//
// WHICH `uint_fastdiv` THIS MEASURED, because it is the whole hazard. The
// probe compiles against `csrc/shim`, so it measures the SHIM's
// `__fast_div_modulo` — `{u32 @0, u64 @8}`, `alignof` 8 — and not CCCL's
// `{u32, u32, u32, i32}`, `alignof` 4. They disagree on interior offsets
// (`paged_kv_t::num_heads` is **+24** under the shim and **+20** under CCCL)
// and reconverge at `sizeof` 96, so a `sizeof` check cannot tell them apart
// and every offset here can. The shim is the right answer because the shim
// is what every JIT fire compiles against: `families/fa2.rs`' rows are all
// NVRTC and the mirrors below are pinned to the shim's layout to match.
//
// It was the WRONG answer for the `cc::Build` that used to compile
// `driver-cuda/csrc/attn/attention_flashinfer.cu` against real CCCL and fill
// the +20 layout. That build and that file are deleted, so there is no
// longer a second filler of these structs and no way for a block filled on
// one side to be read on the other.
//
// # The derivation is kept below because it is what a re-run has to agree with
//
// The ABI lets a derived class place its own members in a non-POD base's
// **tail padding**, starting at the base's `dsize` (its last member's end)
// rather than at its `sizeof`. That is the one degree of freedom, and here it
// closes:
//
// * `BatchDecodeParams`: `sizeof` **224**, `alignof` **8**, last member
//   `partition_kv` a `bool` at **216** (all three asserted above), so
//   `dsize = 217`. The first derived member is a pointer, `alignof` 8, so its
//   offset is `ceil(217 / 8) * 8` = **224** = `sizeof(Base)`. Tail padding
//   cannot be reached by a pointer, so there is nothing for the ABI to choose.
// * `BatchPrefillPagedParams`: `sizeof` **352**, last member
//   `maybe_max_item_len_ptr` a pointer at **344**, so `dsize = 352` and there
//   is no tail padding to reach in the first place.
//
// Then `PieScoreParams<Base>` ends on a pointer, so it has no tail padding
// either, and `PieScoreWindowParams`' `uint32_t` lands immediately after it.
//
// **RE-RUN THE PROBE ON A FLASHINFER BUMP, not this arithmetic.** The
// arithmetic depends on `partition_kv` being the last member of
// `BatchDecodeParams`, which is an upstream fact a version bump can change
// WITHOUT changing `sizeof` — the one failure this file's `sizeof` assertions
// cannot see. The measurement above is against the pinned v0.6.15 and is a
// statement about that version only.

/// `fa2::DecodeScoreParams` — `PieScoreParams<BatchDecodeParams, int32_t>`.
///
/// The single `__grid_constant__` argument of the two capturing decode arms
/// ([`crate::families::fa2::DecodeArm::CaptureFull`] and
/// [`crate::families::fa2::DecodeArm::CaptureWindow`]).
///
/// The base is a FIELD rather than a flattening, which is what makes the
/// derivation above checkable: `#[repr(C)]` puts `base` at 0 and the two
/// pointers after it at the base's own `size_of`, which is exactly what the
/// C++ derivation does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DecodeScoreParams {
    /// The `Base` subobject — every field the uncaptured decode fills.
    pub base: DecodeParams,
    /// `float*`, the ragged score sink:
    /// `score_out[score_indptr[b] + h * kv_len(b) + kv_idx]`.
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
/// `PieScoreWindowParams<BatchPrefillPagedParams, int32_t>`.
///
/// [`DecodeScoreParams`] plus the observation window, which is what keeps the
/// capture off the O(n²) diagonal: only the last `score_window` query rows of
/// each request are recorded.
///
/// **`score_window` is not `window_left`.** The launcher refuses `window == 0`
/// and `window_left` is `-1` on a family that attends its whole context, so
/// the same number reads as "no window" to one field and "invalid" to the
/// other — `table::attn`'s `flashinfer_prefill_capture` row says so from its
/// side.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillScoreParams {
    /// The `Base` subobject — every field the uncaptured prefill fills.
    pub base: PrefillPagedParams,
    /// `float*`, the ragged score sink:
    /// `score_out[score_indptr[b] + (head * window + w) * kv_len(b) + kv_idx]`.
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

#[cfg(test)]
mod tests {
    use super::{FastModDiv, UintFastdiv};

    /// The shim's magic, checked the way the shim's own banner says it should
    /// be: against the arithmetic it claims, not against a table.
    ///
    /// `cmath:214` is `__umul64hi(n, magic)`, so `(n * magic) >> 64` must equal
    /// `n / d` for every 32-bit `n`. Spot-checked over the divisors this
    /// lattice actually builds — page sizes and GQA group sizes.
    #[test]
    fn the_magic_divides() {
        for d in [1u32, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256] {
            let m = FastModDiv::new(d);
            assert_eq!(m.divisor, d, "the divisor is kept verbatim");
            for n in [0u32, 1, 2, d, d - 1, d + 1, 63, 64, 65, 1023, 4096, u32::MAX] {
                // `cmath:212` short-circuits `d == 1` on the device, so the
                // magic path is only required to be right for `d > 1`.
                if d == 1 {
                    continue;
                }
                let hi = ((u128::from(n) * u128::from(m.magic)) >> 64) as u32;
                assert_eq!(hi, n / d, "mul.hi disagreed with the divide at n={n} d={d}");
            }
        }
    }

    /// A zero divisor recovers rather than traps, exactly as `fastdiv.cuh:36`
    /// does with `d ? d : 1`. Stated as a test because the recovery is silent
    /// and someone will otherwise assume it panics.
    #[test]
    fn a_zero_divisor_becomes_one() {
        assert_eq!(UintFastdiv::new(0).d, 1);
        assert_eq!(UintFastdiv::new(0).magic.divisor, 1);
    }
}
