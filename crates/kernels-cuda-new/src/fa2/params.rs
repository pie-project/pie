use core::mem::{align_of, offset_of, size_of};

/// A device address.
pub type DevicePtr = u64;

/// `cuda::fast_mod_div<uint32_t>` — **the shim's**, `csrc/shim/cuda/cmath:175`.
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

#[cfg(test)]
mod tests {
    use super::{FastModDiv, UintFastdiv};

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
}
