//! FlashInfer's paged decode, planned in Rust and compiled from text in this
//! binary, against the same kernel built by `nvcc` and driven by the C++
//! `DecodePlan`.
//!
//! # What was missing, and what this closes
//!
//! Two halves were already measured and neither touched the other.
//! `examples/flashinfer_probe.rs` proved that the 28-file internalised closure
//! compiles under NVRTC against this crate's shims — with no include path on
//! disk — and that `BatchDecodeWithPagedKVCacheKernel<...>` instantiates to a
//! cubin. `tests/plan.rs` proved that `src/plan` reproduces `scheduler.cuh`
//! byte for byte over 638 cases. Neither fired anything. A cubin is not an
//! answer and a plan is not an answer; the answer is what comes out of `o` and
//! `lse` when the schedule the Rust planner computed is uploaded under a
//! params struct the Rust host filled and the kernel NVRTC compiled reads it.
//!
//! So this file does the join, and it does it on the real device: build a
//! paged-KV batch, plan it with [`plan::decode::plan`], upload the bytes that
//! plan returns, compile the kernel through `runtime::nvrtc`, and fire it with
//! `cuLaunchKernel`.
//!
//! # The reference is nvcc, not a host attention
//!
//! **Parity, not self-consistency.** A host-side attention over the same
//! inputs would answer a different question — "does FlashInfer compute
//! attention" — which FlashInfer's own tests already answer and which no
//! amount of agreement would turn into evidence about *this* compile path. The
//! thing at risk here is not the algorithm; it is the ABI, the schedule and
//! the header set. So the reference is the kernel that ships today: the same
//! `BatchDecodeWithPagedKVCacheDispatched`, compiled by `nvcc` against the
//! unmodified FlashInfer headers under `target/*/build/kernels-cuda-*/out`,
//! planned by the real `DecodePlan`, and launched by the real dispatcher —
//! including its `cudaFuncSetAttribute` and, when the plan splits, its
//! `VariableLengthMergeStates` follow-up. Both sides see the same device, the
//! same inputs, and the same float flags. Any disagreement is then a
//! disagreement about the port and nothing else.
//!
//! The float flags are part of the reference and not a detail: `runtime::nvrtc`
//! compiles with `--fmad=false --prec-div=true --prec-sqrt=true`, and `nvcc`
//! contracts multiply-adds by default. Without those three flags on the
//! reference command line the two kernels compute genuinely different
//! arithmetic and the comparison would have to be a tolerance — which is
//! exactly the kind of soft result that hides an ABI bug.
//!
//! # Two cases, because the dispatcher has two shapes
//!
//! `split_kv` false is one kernel writing straight into `o`. `split_kv` true is
//! the same kernel writing PARTIALS into a float workspace and a second kernel,
//! `PersistentVariableLengthMergeStatesKernel`, reducing them — a different
//! params fill (`o`/`lse` swapped for `tmp_v`/`tmp_s`), a `block_valid_mask`
//! that may be null, and a grid that comes from an occupancy query rather than
//! from the batch. Testing only the first would leave the entire float
//! workspace, `o_indptr` and the merge unexercised, and those are precisely the
//! fields a wrong plan corrupts. Both cases are fired end to end and both are
//! compared on the FINAL `o` and `lse`.
//!
//! # What is deliberately not here
//!
//! MLA. `BatchMLAPagedAttentionKernel` needs `grid.sync()` and
//! `cudaLaunchCooperativeKernel`; the crate's `cooperative_groups.h` shim omits
//! `this_grid()` on purpose so that such a kernel fails to compile rather than
//! silently doing nothing, and `KernelModule::fire` has no cooperative launch.
//! See the module tail for what it would take.
//!
//! # Skipping
//!
//! No GPU, no NVRTC, no `nvcc`, or no upstream FlashInfer checkout — the test
//! prints why and passes. That is the same contract `tests/plan.rs` and
//! `tests/fire.rs` keep: a machine that cannot run the measurement must not
//! report a failure that means "this box has no card".

#![cfg(feature = "_cuda")]
#![allow(clippy::too_many_lines)]

use std::ffi::c_void;
use std::mem::offset_of;
use std::path::{Path, PathBuf};
use std::process::Command;

use cudarc::driver::sys as dr;
use kernels_cuda_new::device::DeviceKernel;
use kernels_cuda_new::plan::{Workspace, decode};
use kernels_cuda_new::runtime::{Args, ArgValue, KernelModule, Launch, Stream, cache, nvrtc};
use kernels_cuda_new::source;
use kernels_cuda_new::unit::Unit;
use kernels::{kernel, operands};
use kernels_cuda_new::KernelSig;

// ---------------------------------------------------------------------------
// the geometry
// ---------------------------------------------------------------------------

/// The instantiation this file fires, and every number derived from it.
///
/// Not invented here: it is the one `kernels-cuda`'s AOT launcher already
/// dispatches for a head dimension of 128 and a GQA group of 4, worked through
/// `BatchDecodeWithPagedKVCacheDispatched`'s own arithmetic —
/// `vec_size = max(16/2, 128/32) = 8`, `bdx = 128/8 = 16`, `bdy = GROUP_SIZE =
/// 4`, `num_threads = max(128, 64) = 128`, `bdz = 2`, `tile_size_per_bdx = 1`,
/// and `NUM_STAGES_SMEM = 2` because `DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM`
/// takes the `>= 8` branch on sm_89. A number here that disagreed with the
/// dispatcher would be a kernel with the wrong shared-memory budget reading
/// past its tiles, so they are spelled once and used by both the launch and the
/// smem computation below.
const HEAD_DIM: u32 = 128;
const NUM_KV_HEADS: u32 = 2;
const GQA_GROUP: u32 = 4;
const NUM_QO_HEADS: u32 = NUM_KV_HEADS * GQA_GROUP;
const PAGE_SIZE: u32 = 16;

const VEC_SIZE: u32 = 8;
const BDX: u32 = HEAD_DIM / VEC_SIZE;
const BDY: u32 = GQA_GROUP;
const BDZ: u32 = 2;
const NUM_STAGES_SMEM: u32 = 2;
const TILE_SIZE_PER_BDX: u32 = 1;
const DECODE_THREADS: u32 = BDX * BDY * BDZ;

/// `BatchDecodeWithPagedKVCacheDispatched`'s `smem_size`, restated.
///
/// `2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM *
/// sizeof(DTypeKV) + max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
/// 2 * bdy * bdz * sizeof(float))` = 8192 + 1024 = **9216 bytes**, which is
/// under the 48 KB static default — so this configuration needs no
/// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`
/// and the fact that `KernelModule::fire` offers no seam for one costs nothing
/// HERE. It costs at head_dim 256 with a GQA group of 1, where the same formula
/// gives 65 KB and the launch fails with `CUDA_ERROR_INVALID_VALUE`; that is a
/// finding for the crate, not something to bolt onto this test.
const DECODE_SMEM: u32 = 2 * NUM_STAGES_SMEM * TILE_SIZE_PER_BDX * BDY * BDZ * HEAD_DIM * 2
    + if TILE_SIZE_PER_BDX * DECODE_THREADS * 8 > 2 * BDY * BDZ * 4 {
        TILE_SIZE_PER_BDX * DECODE_THREADS * 8
    } else {
        2 * BDY * BDZ * 4
    };

/// `VariableLengthMergeStates`'s geometry for bf16 at head dim 128.
///
/// `vec_size = max(16/sizeof(bf16), 128/32) = 8`, `bdx = 128/8 = 16`,
/// `num_threads = 128`, `bdy = 8`, `num_smem_stages = 4`, and
/// `smem = 4 * 8 * 128 * 2 + 128 * 4 = 8704`. Also under 48 KB, and also
/// restated rather than recomputed for the same reason.
const MERGE_BDX: u32 = 16;
const MERGE_BDY: u32 = 8;
const MERGE_THREADS: u32 = MERGE_BDX * MERGE_BDY;
const MERGE_SMEM: u32 = 4 * MERGE_BDY * HEAD_DIM * 2 + MERGE_THREADS * 4;

// ---------------------------------------------------------------------------
// the params ABI
// ---------------------------------------------------------------------------

/// `flashinfer::paged_kv_t<__nv_bfloat16, int32_t>`, as NVRTC lays it out.
///
/// # Why a mirror and not a constructor
///
/// `paged_kv_t`'s two `__host__` constructors are behind `#ifndef
/// __CUDACC_RTC__` in this crate's copy, and they have to be: NVRTC rejects a
/// function marked `__host__` outright — *"a function explicitly marked as a
/// `__host__` function is not allowed in JIT mode"* — whether or not anything
/// calls it. `uint_fastdiv(uint32_t)` is guarded for the same reason. So under
/// the JIT nothing on the device can build one of these, and the host has to
/// hand the kernel a struct it filled itself. This is that struct.
///
/// # The one place the two compilers disagree
///
/// `page_size` is a `uint_fastdiv`, which holds a `cuda::fast_mod_div<uint32_t>`
/// — and that class is NOT the same object on the two sides. NVRTC resolves
/// `<cuda/cmath>` to `csrc/shim/cuda/cmath`, this crate's 16-byte shim
/// `{uint32_t divisor; unsigned long long magic;}` with alignment 8; `nvcc`
/// resolves it to FlashInfer's bundled CCCL, a 16-byte
/// `{uint32_t divisor; uint32_t multiplier; unsigned add; int shift;}` with
/// alignment **4**. Measured, both with `nvcc`, by putting the shim first on
/// the include path: `uint_fastdiv` is 24 bytes here and 20 bytes there, so
/// `num_heads` sits at offset **24 under the shim and 20 under CCCL** and every
/// `uint32_t` after it moves with it. The pointer block realigns at 48 and the
/// two agree again, which is why `sizeof(paged_kv_t)` is 96 on both sides and
/// why "the sizes match" was never the question worth asking.
///
/// This mirror therefore describes the SHIM's layout, because the shim is what
/// the JIT kernel was compiled against. The reference harness builds its own
/// with the real constructor and never sees these bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PagedKv {
    /// `uint_fastdiv::impl_::__divisor_` — the page size itself.
    page_size_divisor: u32,
    /// Padding the shim's `unsigned long long __magic_` forces to offset 8.
    _pad0: u32,
    /// `uint_fastdiv::impl_::__magic_` — `floor(2^64/d) + 1`, the constant the
    /// device multiply-high needs. Computed here by [`fast_div_magic`] because
    /// the guarded constructor that used to compute it cannot run under NVRTC,
    /// and because `src/plan` never had to: a planner returns indices, not
    /// reciprocals. This is the one piece of host arithmetic the port did not
    /// already cover, and getting it wrong is a silently wrong page index, not
    /// a crash.
    page_size_magic: u64,
    /// `uint_fastdiv::d_` — the divisor again, which `divmod` uses for the
    /// remainder and `operator unsigned int()` returns.
    page_size_d: u32,
    /// The shim's alignment-8 tail.
    _pad1: u32,
    num_heads: u32,
    head_dim: u32,
    batch_size: u32,
    stride_page: u32,
    stride_n: u32,
    stride_h: u32,
    k_data: u64,
    v_data: u64,
    indices: u64,
    indptr: u64,
    last_page_len: u64,
    rope_pos_offset: u64,
}

/// `flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
/// int32_t>`, as NVRTC lays it out.
///
/// **This struct is an ABI, and a field in the wrong place is a wrong answer
/// rather than a crash.** The kernel takes it `const __grid_constant__` by
/// value, reads `request_indices[blockIdx.x]` out of it without a bounds check
/// and dereferences `kv_chunk_size_ptr` unconditionally; a mirror that put
/// `o_indptr` where `kv_chunk_size_ptr` belongs would run to completion and
/// produce numbers. So the offsets are asserted three ways: `offset_of!` here
/// (below), a device kernel compiled in the SAME NVRTC unit that writes what
/// the compiler actually chose ([`params_abi`]), and the reference harness's
/// own `offsetof` under `nvcc`. Measured: **224 bytes, alignment 8, on both
/// sides**, with every field at the same offset — the divergence is entirely
/// inside `paged_kv_t`, where the padding absorbs it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DecodeParams {
    q: u64,
    q_rope_offset: u64,
    paged_kv: PagedKv,
    o: u64,
    lse: u64,
    maybe_alibi_slopes: u64,
    padded_batch_size: u32,
    num_qo_heads: u32,
    q_stride_n: i32,
    q_stride_h: i32,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    rope_rcp_scale: f32,
    rope_rcp_theta: f32,
    _pad0: u32,
    request_indices: u64,
    kv_tile_indices: u64,
    o_indptr: u64,
    kv_chunk_size_ptr: u64,
    block_valid_mask: u64,
    partition_kv: u8,
    _pad1: [u8; 7],
}

/// The offsets this mirror claims, in the order [`params_abi`] writes them.
///
/// One list, read by the compile-time assertions below, by the device probe's
/// comparison and by the report — so a field that moves is a mismatch in all
/// three rather than a silent divergence between an assertion and a print.
const PARAM_OFFSETS: [(&str, usize); 22] = [
    ("q", offset_of!(DecodeParams, q)),
    ("q_rope_offset", offset_of!(DecodeParams, q_rope_offset)),
    ("paged_kv", offset_of!(DecodeParams, paged_kv)),
    ("o", offset_of!(DecodeParams, o)),
    ("lse", offset_of!(DecodeParams, lse)),
    ("maybe_alibi_slopes", offset_of!(DecodeParams, maybe_alibi_slopes)),
    ("padded_batch_size", offset_of!(DecodeParams, padded_batch_size)),
    ("num_qo_heads", offset_of!(DecodeParams, num_qo_heads)),
    ("q_stride_n", offset_of!(DecodeParams, q_stride_n)),
    ("q_stride_h", offset_of!(DecodeParams, q_stride_h)),
    ("window_left", offset_of!(DecodeParams, window_left)),
    ("logits_soft_cap", offset_of!(DecodeParams, logits_soft_cap)),
    ("sm_scale", offset_of!(DecodeParams, sm_scale)),
    ("rope_rcp_scale", offset_of!(DecodeParams, rope_rcp_scale)),
    ("rope_rcp_theta", offset_of!(DecodeParams, rope_rcp_theta)),
    ("request_indices", offset_of!(DecodeParams, request_indices)),
    ("kv_tile_indices", offset_of!(DecodeParams, kv_tile_indices)),
    ("o_indptr", offset_of!(DecodeParams, o_indptr)),
    ("kv_chunk_size_ptr", offset_of!(DecodeParams, kv_chunk_size_ptr)),
    ("block_valid_mask", offset_of!(DecodeParams, block_valid_mask)),
    ("partition_kv", offset_of!(DecodeParams, partition_kv)),
    ("paged_kv.num_heads", offset_of!(DecodeParams, paged_kv.num_heads)),
];

// The layout, pinned. `src/plan/info.rs` pins the four `PlanInfo` structs the
// same way and for the same reason: these numbers came out of a measurement,
// and a struct that drifts from them compiles fine and answers wrong.
const _: () = {
    assert!(size_of::<DecodeParams>() == 224);
    assert!(align_of::<DecodeParams>() == 8);
    assert!(size_of::<PagedKv>() == 96);
    assert!(offset_of!(DecodeParams, paged_kv) == 16);
    assert!(offset_of!(DecodeParams, o) == 112);
    assert!(offset_of!(DecodeParams, padded_batch_size) == 136);
    assert!(offset_of!(DecodeParams, request_indices) == 176);
    assert!(offset_of!(DecodeParams, partition_kv) == 216);
    // The shim's own signature: 24, not CCCL's 20. If `csrc/shim/cuda/cmath`
    // ever grows a field or loses its 8-byte magic, this is where it shows.
    assert!(offset_of!(DecodeParams, paged_kv.num_heads) == 16 + 24);
};

/// `floor(2^64 / d) + 1`, the shim's magic, computed without a 128-bit
/// intermediate.
///
/// `2^64 - 1 = q*d + r`, so `2^64 = q*d + (r+1)` and the carry into the
/// quotient happens exactly when `r + 1 == d`. This is `csrc/shim/cuda/cmath`'s
/// constructor transcribed into Rust — it has to be transcribed, because that
/// constructor is `__host__ __device__` C++ that the JIT never runs on the
/// host. Wrong by one and `__umul64hi` returns the wrong page for some but not
/// all indices, which is a corruption that shows up on one request in a batch.
const fn fast_div_magic(d: u32) -> u64 {
    let d = d as u64;
    let q = u64::MAX / d;
    let r = u64::MAX % d;
    // `wrapping_add`, because the C++ this transcribes is unsigned arithmetic
    // and `d == 1` overflows it deliberately: the magic wraps to 1 and the
    // multiply-high then returns 0 for every dividend, which is why the shim
    // treats `d == 1` as a COMPARISON rather than a magic. A checked add here
    // would panic on a page size of one instead of reproducing the shim.
    q.wrapping_add(if r + 1 == d { 1 } else { 0 }).wrapping_add(1)
}

// ---------------------------------------------------------------------------
// the unit
// ---------------------------------------------------------------------------

/// The root a vendored unit compiles: the FlashInfer headers it needs, plus the
/// two names `DeviceKernel::instantiation` forces into existence.
///
/// # The namespace alias is not decoration
///
/// `DeviceKernel::instantiation` formats
/// `::pie_cuda_driver::kernels::{template_path}<::pie_cuda_driver::kernels::{elem}>`
/// — the template path AND the first template argument are both prefixed, and
/// nothing else is. A `::flashinfer::` kernel therefore cannot be named by a
/// row at all unless something under `pie_cuda_driver::kernels` resolves to it,
/// which is what `namespace fi = ::flashinfer` is for. The remaining arguments
/// are pasted verbatim, so they are spelled from `::` here.
///
/// `merge_vec_size` exists for the harsher half of the same problem: the merge
/// kernel's first template parameter is a NON-TYPE one, so the row's `elem`
/// would have to be the literal `8` and the prefix would produce
/// `::pie_cuda_driver::kernels::8`, which is not a name. A `constexpr` under
/// the right namespace is the only way a row can say "eight" here.
///
/// `params_abi` is this file's own, and it is the point of compiling it INSIDE
/// the unit rather than measuring the layout somewhere else: it reports the
/// offsets the same compiler, with the same header set and the same options,
/// chose for the struct the decode kernel is about to read.
const ROOT: &str = r#"
#include "attn/flashinfer/attention/decode.cuh"
#include "attn/flashinfer/attention/default_decode_params.cuh"
#include "attn/flashinfer/attention/variants.cuh"
#include "attn/flashinfer/attention/cascade.cuh"

namespace pie_cuda_driver {
namespace kernels {

namespace fi = ::flashinfer;
constexpr unsigned int merge_vec_size = 8;
using decode_params =
    ::flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;

// `Tag` is unused and the body works on the concrete alias, deliberately:
// `DeviceKernel::instantiation` can only name a TEMPLATE, so the row supplies
// a tag it does not read.
//
// The offsets are pointer differences off a real object rather than
// `offsetof`: NVRTC has no `<cstddef>` in this header set, and its
// `__builtin_offsetof` refuses even a non-dependent typedef here -- measured,
// *"type name is not allowed"* once per field. A default-constructed local
// works because BOTH structs kept a `__device__ __host__` default constructor
// when their `__host__` ones were guarded away, which is the one thing device
// code can still do with them.
template <typename Tag>
__global__ void params_abi(unsigned long long* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  using Params = decode_params;
  using Pages = ::flashinfer::paged_kv_t<__nv_bfloat16, int32_t>;
  Params s;
  const char* b = reinterpret_cast<const char*>(&s);
#define PIE_AT(field) \
  static_cast<unsigned long long>(reinterpret_cast<const char*>(&s.field) - b)
  unsigned long long* p = out;
  *p++ = sizeof(Params);
  *p++ = alignof(Params);
  *p++ = sizeof(Pages);
  *p++ = alignof(Pages);
  *p++ = PIE_AT(q);
  *p++ = PIE_AT(q_rope_offset);
  *p++ = PIE_AT(paged_kv);
  *p++ = PIE_AT(o);
  *p++ = PIE_AT(lse);
  *p++ = PIE_AT(maybe_alibi_slopes);
  *p++ = PIE_AT(padded_batch_size);
  *p++ = PIE_AT(num_qo_heads);
  *p++ = PIE_AT(q_stride_n);
  *p++ = PIE_AT(q_stride_h);
  *p++ = PIE_AT(window_left);
  *p++ = PIE_AT(logits_soft_cap);
  *p++ = PIE_AT(sm_scale);
  *p++ = PIE_AT(rope_rcp_scale);
  *p++ = PIE_AT(rope_rcp_theta);
  *p++ = PIE_AT(request_indices);
  *p++ = PIE_AT(kv_tile_indices);
  *p++ = PIE_AT(o_indptr);
  *p++ = PIE_AT(kv_chunk_size_ptr);
  *p++ = PIE_AT(block_valid_mask);
  *p++ = PIE_AT(partition_kv);
  *p++ = PIE_AT(paged_kv.num_heads);
#undef PIE_AT
}

}  // namespace kernels
}  // namespace pie_cuda_driver
"#;

/// The three rows this unit carries.
///
/// `LaunchRule::Unstated` on all three, honestly: a decode grid is
/// `(padded_batch_size, num_kv_heads)` where `padded_batch_size` comes out of
/// the PLAN, and the merge grid is `num_sm * min(occupancy, ...)`. Neither is a
/// function of a rectangle, so there is no rule to state and stating one would
/// be a lie the launch path would then believe.
///
/// The operand lists are honest in the same way and they are where the crate's
/// argument marshalling runs out: `params_abi` and the merge kernel take
/// pointers and scalars and go through [`Args::bind`] unchanged, while the
/// decode kernel takes ONE `const __grid_constant__ Params` by value, which
/// `Ty` cannot spell. Its row therefore declares no operands and its launch
/// builds the `void*` array by hand.
///
/// **`indptr` and `seq_len` were `I32s` and `U32s` — read-only — and the
/// merge kernel declares both `IdType*` and `uint32_t*`, non-const.** Found by
/// applying `table-rows`' offline device typecheck to this unit, which their
/// sweep could not reach because a vendored unit is not in `unit::UNITS`. It
/// was latent, not live: `Args::bind` marshals a device pointer either way, so
/// split-KV was bit-exact against nvcc both before and after. It was still a
/// row that misdescribed its kernel, and `I32sMut`/`U32sMut` already existed —
/// unlike the four rows their sweep found, this one was a row bug and not a
/// `Ty` gap.
///
/// Two operands here remain unspellable and are deliberately left wrong rather
/// than papered over, because the fix belongs in `Ty` and not in a row:
/// `Ty::BufMut` derives its element from the row's `elem` HEAD, and these rows'
/// heads are `merge_vec_size` and `fi::PosEncodingMode::kNone` — a `constexpr`
/// and an enumerator, not types. `params_abi`'s `out` is worse still: the
/// kernel writes `unsigned long long*` and no `Ty` spells a 64-bit array.
/// Narrowing the probe to `::std::uint32_t*` to dodge that was tried and
/// reverted — it crashed the harness, and it was the wrong shape of fix
/// regardless.
static SIGS: [KernelSig; 3] = [
    kernel!(flashinfer_decode_bf16 "flashinfer_decode_bf16", operands = operands![]),
    kernel!(flashinfer_merge_bf16 "flashinfer_merge_bf16", operands = operands![
        v: BufMut,
        s: F32sMut,
        indptr: I32sMut,
        v_merged: BufMut,
        s_merged: F32sMut,
        max_seq_len: U32,
        seq_len: U32sMut | null,
        num_heads: U32,
    ]),
    kernel!(flashinfer_params_abi "flashinfer_params_abi", operands = operands![out: BufMut]),
];

/// The rows, as instantiations.
///
/// The decode row's `elem` is nine template arguments in one string, which
/// `DeviceKernel::elem` supports because it is pasted rather than parsed —
/// `src/device.rs` says so explicitly and `examples/argform_probe.rs` measured
/// it. Only the first of the nine is prefixed; see [`ROOT`].
static ROWS: [DeviceKernel; 3] = [
    DeviceKernel {
        sig: &SIGS[0],
        template_path: "fi::BatchDecodeWithPagedKVCacheKernel",
        elem: concat!(
            "fi::PosEncodingMode::kNone, 2, 1, 8, 16, 4, 2, ",
            "::flashinfer::DefaultAttention<false, true, false, false>, ",
            "::flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>"
        ),
    },
    DeviceKernel {
        sig: &SIGS[1],
        template_path: "fi::PersistentVariableLengthMergeStatesKernel",
        elem: "merge_vec_size, 16, 8, 4, __nv_bfloat16, __nv_bfloat16, int32_t",
    },
    DeviceKernel {
        sig: &SIGS[2],
        template_path: "params_abi",
        // `decode_params`, not the type spelled out: `instantiation()` prefixes
        // the first template argument unconditionally, and
        // `::pie_cuda_driver::kernels::::flashinfer::BatchDecodeParams<...>` is
        // not a name. Measured -- NVRTC answers *"expected an identifier"* at
        // the fourth colon. The alias in [`ROOT`] is the only way a row can
        // name a vendored TYPE, exactly as `merge_vec_size` is the only way it
        // can name a vendored non-type argument.
        elem: "decode_params",
    },
];

/// The unit.
///
/// `--device-as-default-execution-space` is the whole reason `Unit::options`
/// exists as a field. FlashInfer is full of unannotated `constexpr` helpers
/// that `nvcc` forgives inside a `.cu` and NVRTC does not, and without the flag
/// `decode.cuh` is refused at the first one. It is per-unit and deliberately
/// not global: on this crate's OWN sources it would compile an unannotated
/// HOST helper onto the device silently instead of reporting it — the
/// `yarn_original_ramp_bounds` defect, which the flag would have hidden.
///
/// `Unit::cache_key` spans `options`, so this unit and a hypothetical one
/// without the flag cannot be served each other's cubin.
static UNIT: Unit = Unit {
    name: "attn/flashinfer_decode",
    root: ROOT,
    rows: &ROWS,
    options: &["--device-as-default-execution-space"],
};

// ---------------------------------------------------------------------------
// the batch
// ---------------------------------------------------------------------------

/// One case: a name, the page counts per request, and how full the last page is.
///
/// Page counts are what decide the shape of the plan, so they are the case's
/// only real parameter. `last_page_len` is separate because it decides `kv_len`
/// and therefore the mask, and a case where every last page is full would never
/// exercise the `chunk_end` clamp.
struct Case {
    name: &'static str,
    pages: &'static [u32],
    last_page_len: &'static [u32],
    /// What the plan is expected to decide. Checked, not assumed: a case that
    /// silently stopped splitting would still pass a parity comparison and
    /// would stop testing the merge.
    expect_split: bool,
}

/// The cases.
///
/// The first stays under the estimator's floor — `min_num_pages_per_batch` is
/// `max(128 / page_size, 1)`, which is 8 pages at page size 16, so a request of
/// at most 8 pages produces one work item and `new_batch_size == batch_size`
/// turns the split off. The second is deliberately above it: 40 and 33 pages
/// become 5 chunks each, ten work items for two requests, and the dispatcher
/// takes its partition-KV path.
const CASES: [Case; 2] = [
    Case {
        name: "no split -- every request under the 8-page chunk floor",
        pages: &[2, 5, 1, 8],
        last_page_len: &[16, 3, 9, 16],
        expect_split: false,
    },
    Case {
        name: "split -- 40 and 33 pages become ten work items and a merge",
        pages: &[40, 33],
        last_page_len: &[7, 16],
        expect_split: true,
    },
];

/// A deterministic bf16 bit pattern, so both sides see identical inputs and a
/// rerun sees the same ones.
///
/// The exponent is clamped into `[0x3e, 0x40]` — roughly `[0.25, 2)` — rather
/// than left to the generator. Uniform random 16-bit patterns are mostly
/// denormals, infinities and NaNs, and a comparison over those measures the
/// hardware's handling of specials instead of the attention: one NaN anywhere
/// in a row makes the whole row NaN on both sides, which compares EQUAL and
/// proves nothing. The sign bit is kept random, because a softmax over
/// same-signed logits never exercises the running-max path.
fn bf16_stream(seed: u64, count: usize) -> Vec<u16> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            let sign = ((bits >> 20) & 1) as u16;
            let exponent = 0x3e + ((bits >> 8) & 0x3) as u16;
            let mantissa = (bits & 0x7f) as u16;
            (sign << 15) | (exponent << 7) | mantissa
        })
        .collect()
}

/// The inputs one case runs on, in the exact layouts both sides read.
struct Batch {
    kv_indptr: Vec<i32>,
    kv_indices: Vec<i32>,
    last_page_len: Vec<i32>,
    q: Vec<u16>,
    k_pool: Vec<u16>,
    v_pool: Vec<u16>,
    num_pages: u32,
}

impl Batch {
    /// Build a case's batch.
    ///
    /// The page indices are deliberately NOT `0, 1, 2, ...`: they are strided
    /// through the pool by a coprime step so that consecutive pages of one
    /// request are far apart and two requests interleave. A test whose page
    /// table happened to be the identity would pass with `stride_page` wrong,
    /// with `indices` ignored, and with two requests reading each other's
    /// pages — which is exactly the corruption
    /// `attention_flashinfer_common.cuh` carries a twenty-line comment about.
    fn build(case: &Case) -> Self {
        let batch = case.pages.len();
        let mut kv_indptr = vec![0i32];
        for pages in case.pages {
            kv_indptr.push(kv_indptr.last().unwrap() + *pages as i32);
        }
        let nnz = *kv_indptr.last().unwrap() as u32;
        // One more page than the batch uses, so an out-of-range index would
        // land somewhere addressable and produce a wrong answer rather than a
        // fault -- a wrong answer is what this test can see.
        let num_pages = nnz + 3;
        let step = 7u32;
        let kv_indices: Vec<i32> =
            (0..nnz).map(|i| ((i * step) % num_pages) as i32).collect();
        assert_eq!(
            {
                let mut seen = kv_indices.clone();
                seen.sort_unstable();
                seen.dedup();
                seen.len()
            },
            kv_indices.len(),
            "the page permutation must not alias: two requests sharing a page \
             would make a mis-attribution invisible"
        );

        let per_page = (PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize;
        Self {
            last_page_len: case.last_page_len.iter().map(|n| *n as i32).collect(),
            q: bf16_stream(0x51ed_1234, batch * (NUM_QO_HEADS * HEAD_DIM) as usize),
            k_pool: bf16_stream(0x0bad_c0de, num_pages as usize * per_page),
            v_pool: bf16_stream(0xfeed_face, num_pages as usize * per_page),
            kv_indptr,
            kv_indices,
            num_pages,
        }
    }

    fn batch_size(&self) -> u32 {
        (self.kv_indptr.len() - 1) as u32
    }
}

// ---------------------------------------------------------------------------
// device memory
// ---------------------------------------------------------------------------

/// A device allocation that frees itself.
///
/// A test that fires two cases and two kernels each leaks a dozen allocations
/// per run without this, and a leaked allocation on a shared box is a later
/// test's out-of-memory rather than this one's failure.
struct Buf(dr::CUdeviceptr, usize);

impl Buf {
    fn zeroed(bytes: usize) -> Self {
        let mut ptr: dr::CUdeviceptr = 0;
        // SAFETY: `ptr` is a live out-parameter; the size is non-zero.
        let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes.max(1)) };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuMemAlloc_v2({bytes})");
        // SAFETY: the allocation is live and exactly `bytes` long.
        unsafe { dr::cuMemsetD8_v2(ptr, 0, bytes.max(1)) };
        Self(ptr, bytes)
    }

    fn upload<T: Copy>(data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::zeroed(bytes);
        // SAFETY: the allocation is exactly `bytes` long and `data` is live.
        let code =
            unsafe { dr::cuMemcpyHtoD_v2(buf.0, data.as_ptr().cast(), bytes) };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD_v2({bytes})");
        buf
    }

    fn download<T: Copy + Default>(&self, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        let bytes = std::mem::size_of_val(out.as_slice());
        assert!(bytes <= self.1, "download of {bytes} from a {}-byte buffer", self.1);
        // SAFETY: the allocation covers `bytes` by the assertion above.
        let code = unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.0, bytes) };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH_v2({bytes})");
        out
    }

    const fn ptr(&self) -> u64 {
        self.0
    }

    const fn raw(&self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemAlloc_v2` and not freed twice.
        unsafe { dr::cuMemFree_v2(self.0) };
    }
}

/// `cuCtxSynchronize`, with the kernel's own fault reported where it happened.
///
/// A launch is asynchronous, so an out-of-bounds store inside the decode kernel
/// surfaces at whatever call comes next — usually a `cuMemcpyDtoH` in the
/// comparison, where the message reads as a download failure. Synchronising
/// right after each fire keeps the blame on the launch.
fn sync(what: &str) {
    // SAFETY: a context is bound; the call takes nothing.
    let code = unsafe { dr::cuCtxSynchronize() };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}: the launch faulted");
}

// ---------------------------------------------------------------------------
// the reference
// ---------------------------------------------------------------------------

/// The C++ that is being matched: the real `DecodePlan`, the real dispatcher,
/// the real headers.
///
/// It is a string here for the reason `tests/plan.rs` keeps its harness as one
/// — the file this test compiles cannot then drift from the test that describes
/// it, and the crate ships no C++ of its own, which is the entire point of the
/// exercise. It reads a binary spec, allocates, plans, launches, and dumps: the
/// ABI it measured, the occupancy it was told, the plan info, the staged
/// workspace bytes, and `o` and `lse`.
///
/// Nothing here judges anything. Every comparison is on the Rust side.
const HARNESS: &str = r##"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>

using DType = __nv_bfloat16;
using IdType = int32_t;
using Params = flashinfer::BatchDecodeParams<DType, DType, DType, IdType>;
using Pages = flashinfer::paged_kv_t<DType, IdType>;
using Variant = flashinfer::DefaultAttention<false, true, false, false>;
constexpr uint32_t HEAD_DIM = 128;
constexpr flashinfer::PosEncodingMode POS_ENC = flashinfer::PosEncodingMode::kNone;

#define CHECK(expr)                                                              \
  do {                                                                           \
    cudaError_t _e = (expr);                                                     \
    if (_e != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr,        \
                   cudaGetErrorString(_e));                                      \
      return 2;                                                                  \
    }                                                                            \
  } while (0)

static std::vector<unsigned char> slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); exit(3); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> out(n);
  if (n && fread(out.data(), 1, n, f) != (size_t)n) { std::fprintf(stderr, "short read\n"); exit(3); }
  fclose(f);
  return out;
}

struct Reader {
  const unsigned char* p;
  template <class T> T get() { T v; memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
  template <class T> const T* take(size_t n) { const T* v = (const T*)p; p += n * sizeof(T); return v; }
};

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: harness <spec> <out>\n"); return 3; }
  std::vector<unsigned char> spec = slurp(argv[1]);
  Reader r{spec.data()};
  const uint32_t batch_size = r.get<uint32_t>();
  const uint32_t num_qo_heads = r.get<uint32_t>();
  const uint32_t num_kv_heads = r.get<uint32_t>();
  const uint32_t page_size = r.get<uint32_t>();
  const uint32_t num_pages = r.get<uint32_t>();
  const uint32_t nnz = r.get<uint32_t>();
  const float sm_scale = r.get<float>();
  const int32_t window_left = r.get<int32_t>();
  const IdType* kv_indptr_h = r.take<IdType>(batch_size + 1);
  const IdType* kv_indices_h = r.take<IdType>(nnz);
  const IdType* last_page_len_h = r.take<IdType>(batch_size);
  const size_t q_elems = (size_t)batch_size * num_qo_heads * HEAD_DIM;
  const size_t pool_elems = (size_t)num_pages * page_size * num_kv_heads * HEAD_DIM;
  const DType* q_h = r.take<DType>(q_elems);
  const DType* k_h = r.take<DType>(pool_elems);
  const DType* v_h = r.take<DType>(pool_elems);

  void *d_q, *d_k, *d_v, *d_o, *d_lse, *d_indptr, *d_indices, *d_lastlen;
  CHECK(cudaMalloc(&d_q, q_elems * sizeof(DType)));
  CHECK(cudaMalloc(&d_k, pool_elems * sizeof(DType)));
  CHECK(cudaMalloc(&d_v, pool_elems * sizeof(DType)));
  CHECK(cudaMalloc(&d_o, q_elems * sizeof(DType)));
  CHECK(cudaMalloc(&d_lse, (size_t)batch_size * num_qo_heads * sizeof(float)));
  CHECK(cudaMalloc(&d_indptr, (batch_size + 1) * sizeof(IdType)));
  CHECK(cudaMalloc(&d_indices, (nnz ? nnz : 1) * sizeof(IdType)));
  CHECK(cudaMalloc(&d_lastlen, batch_size * sizeof(IdType)));
  CHECK(cudaMemcpy(d_q, q_h, q_elems * sizeof(DType), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_k, k_h, pool_elems * sizeof(DType), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_v, v_h, pool_elems * sizeof(DType), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_indptr, kv_indptr_h, (batch_size + 1) * sizeof(IdType), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_indices, kv_indices_h, nnz * sizeof(IdType), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_lastlen, last_page_len_h, batch_size * sizeof(IdType), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_o, 0, q_elems * sizeof(DType)));
  CHECK(cudaMemset(d_lse, 0, (size_t)batch_size * num_qo_heads * sizeof(float)));

  // The occupancy the estimator will read, recomputed here so the Rust planner
  // can be given the same `max_grid_size` rather than guessing at one.
  constexpr uint32_t vec_size = 8, bdx = 16, bdy = 4, bdz = 2, stages = 2, tile = 1;
  constexpr uint32_t decode_threads = bdx * bdy * bdz;
  const uint32_t decode_smem =
      2 * stages * tile * bdy * bdz * HEAD_DIM * sizeof(DType) +
      (tile * decode_threads * sizeof(DType*) > 2 * bdy * bdz * sizeof(float)
           ? tile * decode_threads * sizeof(DType*)
           : 2 * bdy * bdz * sizeof(float));
  auto decode_kernel =
      flashinfer::BatchDecodeWithPagedKVCacheKernel<POS_ENC, stages, tile, vec_size, bdx, bdy, bdz,
                                                    Variant, Params>;
  int num_sm = 0, blocks_per_sm = 0, dev = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, decode_kernel,
                                                      decode_threads, decode_smem));

  // Size the workspaces the way the planner asks to be sized, then plan.
  size_t float_ws = 0, int_ws = 0;
  auto estimator = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      4, HEAD_DIM, POS_ENC, Variant, Params>;
  cudaError_t st = flashinfer::DecodePlanWorkspaceSize<HEAD_DIM, POS_ENC, Variant, Params>(
      float_ws, int_ws, (IdType*)kv_indptr_h, batch_size, num_qo_heads, page_size,
      /*enable_cuda_graph=*/false, /*stream=*/nullptr, estimator);
  if (st != cudaSuccess) { std::fprintf(stderr, "workspace size failed\n"); return 2; }

  void* float_buffer = nullptr;
  void* int_buffer = nullptr;
  CHECK(cudaMalloc(&float_buffer, float_ws ? float_ws : 16));
  CHECK(cudaMalloc(&int_buffer, int_ws ? int_ws : 16));
  CHECK(cudaMemset(float_buffer, 0, float_ws ? float_ws : 16));
  CHECK(cudaMemset(int_buffer, 0, int_ws ? int_ws : 16));
  // Zeroed, not merely allocated: the allocator pads for alignment and upstream
  // copies whatever was in the staging buffer there. The Rust planner
  // zero-fills its padding, so anything else here would compare unequal on
  // bytes neither side means.
  std::vector<unsigned char> staging(int_ws ? int_ws : 16, 0);

  flashinfer::DecodePlanInfo plan_info;
  st = flashinfer::DecodePlan<HEAD_DIM, POS_ENC, Variant, Params>(
      float_buffer, float_ws, int_buffer, staging.data(), int_ws, plan_info,
      (IdType*)kv_indptr_h, batch_size, num_qo_heads, page_size,
      /*enable_cuda_graph=*/false, /*stream=*/nullptr, estimator);
  if (st != cudaSuccess) { std::fprintf(stderr, "plan failed\n"); return 2; }
  CHECK(cudaStreamSynchronize(nullptr));

  Pages paged_kv(num_kv_heads, page_size, HEAD_DIM, batch_size, flashinfer::QKVLayout::kNHD,
                 (DType*)d_k, (DType*)d_v, (IdType*)d_indices, (IdType*)d_indptr,
                 (IdType*)d_lastlen);
  Params params;
  params.q = (DType*)d_q;
  params.q_rope_offset = nullptr;
  params.paged_kv = paged_kv;
  params.o = (DType*)d_o;
  params.lse = (float*)d_lse;
  params.maybe_alibi_slopes = nullptr;
  params.num_qo_heads = num_qo_heads;
  params.q_stride_n = (IdType)(num_qo_heads * HEAD_DIM);
  params.q_stride_h = (IdType)HEAD_DIM;
  params.window_left = window_left;
  params.logits_soft_cap = 0.f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;
  auto at_int = [&](int64_t off) { return (void*)((unsigned char*)int_buffer + off); };
  params.request_indices = (IdType*)at_int(plan_info.request_indices_offset);
  params.kv_tile_indices = (IdType*)at_int(plan_info.kv_tile_indices_offset);
  params.o_indptr = (IdType*)at_int(plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr = (IdType*)at_int(plan_info.kv_chunk_size_ptr_offset);
  params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;
  params.partition_kv = plan_info.split_kv;

  DType* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = (DType*)((unsigned char*)float_buffer + plan_info.v_offset);
    tmp_s = (float*)((unsigned char*)float_buffer + plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask = (bool*)at_int(plan_info.block_valid_mask_offset);
    }
  }

  st = flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENC, Variant, Params>(
      params, tmp_v, tmp_s, /*enable_pdl=*/false, /*stream=*/nullptr);
  if (st != cudaSuccess) { std::fprintf(stderr, "dispatch failed: %s\n", cudaGetErrorString(st)); return 2; }
  CHECK(cudaDeviceSynchronize());

  std::vector<unsigned short> o_h(q_elems);
  std::vector<float> lse_h((size_t)batch_size * num_qo_heads);
  CHECK(cudaMemcpy(o_h.data(), d_o, q_elems * sizeof(DType), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(lse_h.data(), d_lse, lse_h.size() * sizeof(float), cudaMemcpyDeviceToHost));

  FILE* out = fopen(argv[2], "wb");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 3; }
  auto put_u64 = [&](unsigned long long v) { fwrite(&v, sizeof(v), 1, out); };
  put_u64(sizeof(Params));
  put_u64(alignof(Params));
  put_u64(sizeof(Pages));
  put_u64(alignof(Pages));
  put_u64(offsetof(Params, q));
  put_u64(offsetof(Params, q_rope_offset));
  put_u64(offsetof(Params, paged_kv));
  put_u64(offsetof(Params, o));
  put_u64(offsetof(Params, lse));
  put_u64(offsetof(Params, maybe_alibi_slopes));
  put_u64(offsetof(Params, padded_batch_size));
  put_u64(offsetof(Params, num_qo_heads));
  put_u64(offsetof(Params, q_stride_n));
  put_u64(offsetof(Params, q_stride_h));
  put_u64(offsetof(Params, window_left));
  put_u64(offsetof(Params, logits_soft_cap));
  put_u64(offsetof(Params, sm_scale));
  put_u64(offsetof(Params, rope_rcp_scale));
  put_u64(offsetof(Params, rope_rcp_theta));
  put_u64(offsetof(Params, request_indices));
  put_u64(offsetof(Params, kv_tile_indices));
  put_u64(offsetof(Params, o_indptr));
  put_u64(offsetof(Params, kv_chunk_size_ptr));
  put_u64(offsetof(Params, block_valid_mask));
  put_u64(offsetof(Params, partition_kv));
  put_u64(offsetof(Params, paged_kv) + offsetof(Pages, num_heads));
  put_u64((unsigned long long)num_sm);
  put_u64((unsigned long long)blocks_per_sm);
  put_u64((unsigned long long)decode_smem);
  std::vector<int64_t> info = plan_info.ToVector();
  put_u64(info.size());
  fwrite(info.data(), sizeof(int64_t), info.size(), out);
  put_u64((unsigned long long)float_ws);
  put_u64((unsigned long long)int_ws);
  fwrite(staging.data(), 1, int_ws, out);
  put_u64(o_h.size());
  fwrite(o_h.data(), sizeof(unsigned short), o_h.size(), out);
  put_u64(lse_h.size());
  fwrite(lse_h.data(), sizeof(float), lse_h.size(), out);
  fclose(out);
  return 0;
}
"##;

/// Where FlashInfer's headers are, if they are anywhere.
///
/// `kernels-cuda`'s build script vendors them under its `OUT_DIR`; the hash in
/// that path changes with the build, so this globs rather than hard-codes.
/// `PIE_FLASHINFER_INCLUDE` overrides it. Lifted from `tests/plan.rs`, which
/// needs the same tree for the same reason — and deliberately NOT the crate's
/// own `csrc/src/attn/flashinfer`, because the reference must be the UNMODIFIED upstream
/// headers that ship today, not this crate's guarded copy of them.
fn flashinfer_src() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PIE_FLASHINFER_INCLUDE") {
        let dir = PathBuf::from(dir);
        return dir.join("include/flashinfer/attention/scheduler.cuh").exists().then_some(dir);
    }
    let mut dir: &Path = Path::new(env!("OUT_DIR"));
    let target = loop {
        let parent = dir.parent()?;
        if parent.file_name().is_some_and(|n| n == "target") {
            break parent.to_path_buf();
        }
        dir = parent;
    };
    for profile in std::fs::read_dir(&target).ok()? {
        let build = profile.ok()?.path().join("build");
        let Ok(entries) = std::fs::read_dir(&build) else { continue };
        for entry in entries.flatten() {
            if !entry.file_name().to_string_lossy().starts_with("kernels-cuda-") {
                continue;
            }
            let src = entry.path().join("out/kernels-cuda/build/_deps/flashinfer-src");
            if src.join("include/flashinfer/attention/scheduler.cuh").exists() {
                return Some(src);
            }
        }
    }
    None
}

/// `nvcc`, wherever it is.
fn nvcc() -> Option<PathBuf> {
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"] {
        let path = PathBuf::from(candidate);
        if Command::new(&path).arg("--version").output().is_ok_and(|o| o.status.success()) {
            return Some(path);
        }
    }
    None
}

/// Compile the reference, or say why not.
///
/// `-arch=sm_89` because the reference must be SASS for the card the JIT is
/// compiling for; a PTX-only build would be JIT-compiled by the driver with
/// different arithmetic decisions than `ptxas` made for the NVRTC cubin, and
/// the comparison would be between two compilers AND two backends.
///
/// **`--fmad=false --prec-div=true --prec-sqrt=true` is the load-bearing part.**
/// `runtime::nvrtc::options` passes exactly those three and `Unit::cache_key`
/// restates them, because contracting a multiply-add moves a lane by more than
/// a bit. `nvcc` contracts by default. Without them the two kernels are doing
/// different arithmetic and no amount of ABI correctness makes the outputs
/// equal — which would leave this test comparing with a tolerance and unable to
/// tell a rounding difference from a wrong index.
fn build_reference(arch: &str) -> Result<PathBuf, String> {
    let out = scratch();
    std::fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let exe = out.join("decode_reference");
    let src = out.join("decode_reference.cu");
    std::fs::write(&src, HARNESS).map_err(|e| format!("cannot write the reference: {e}"))?;

    let Some(nvcc) = nvcc() else { return Err("nvcc is not on PATH or in /usr/local/cuda".into()) };
    let Some(flashinfer) = flashinfer_src() else {
        return Err("FlashInfer's headers are not vendored under target/ -- \
                    build kernels-cuda once, or set PIE_FLASHINFER_INCLUDE"
            .into());
    };

    let output = Command::new(&nvcc)
        .args(["-std=c++20", "-O2", "-w"])
        .arg(format!("-arch={arch}"))
        .args(["--fmad=false", "--prec-div=true", "--prec-sqrt=true"])
        .arg("-I")
        .arg(flashinfer.join("include"))
        .arg("-I")
        .arg(flashinfer.join("3rdparty/cccl/libcudacxx/include"))
        .arg("-o")
        .arg(&exe)
        .arg(&src)
        .output()
        .map_err(|e| format!("could not run {}: {e}", nvcc.display()))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc refused the reference:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(exe)
}

/// Where the reference binary and its per-case spec and dump live — **one
/// directory per PROCESS, not one per crate.**
///
/// `env!("OUT_DIR")` alone is per-crate-per-feature-set, which is to say
/// shared by every concurrent `cargo test -p kernels-cuda-new` that resolves
/// to the same metadata hash. Two of them race twice over, and this file was
/// written with both bugs before either was measured:
///
/// * `nvcc -o decode_reference` rewrites the binary **in place** while the
///   other run is `execve`-ing it, which the kernel answers with `ETXTBSY`
///   (*"Text file busy"*) or `EACCES`. Reproduced deliberately with the GPU
///   idle: two copies of `tests/plan.rs`'s binary, which has the same shape,
///   and one lost six cases to `Text file busy (os error 26)` while the other
///   stayed green. That failure is loud and merely confusing.
/// * `spec.bin` and `reference.bin` are worse. They are not indexed by case —
///   there are only ever two of them — so a concurrent run overwrites this
///   one's INPUT between the write and the exec, and the reference then
///   answers the other case's question. The dump decodes, the comparison runs,
///   and the disagreement is reported against the wrong reference. **A silent
///   wrong answer in a test whose entire job is to be believed about a
///   bit-exact comparison.**
///
/// The pid makes both impossible and costs one `nvcc` per process, which is
/// 3 seconds against the 11 the test already takes.
fn scratch() -> PathBuf {
    PathBuf::from(env!("OUT_DIR")).join(format!("flashinfer-decode-{}", std::process::id()))
}

/// What the reference produced.
struct Reference {
    /// `sizeof`, `alignof` and the offsets, in [`PARAM_OFFSETS`] order after
    /// the first four.
    abi: Vec<u64>,
    num_sm: u32,
    blocks_per_sm: u32,
    decode_smem: u32,
    info: Vec<i64>,
    float_bytes: u64,
    int_bytes: u64,
    upload: Vec<u8>,
    o: Vec<u16>,
    lse: Vec<f32>,
}

/// Run the reference on one case's spec.
fn run_reference(exe: &Path, spec: &[u8], scratch: &Path) -> Result<Reference, String> {
    let spec_path = scratch.join("spec.bin");
    let out_path = scratch.join("reference.bin");
    std::fs::write(&spec_path, spec).map_err(|e| format!("cannot write the spec: {e}"))?;
    let _ = std::fs::remove_file(&out_path);
    let status = Command::new(exe)
        .arg(&spec_path)
        .arg(&out_path)
        .output()
        .map_err(|e| format!("could not run the reference: {e}"))?;
    if !status.status.success() {
        return Err(format!(
            "the reference exited {:?}:\n{}",
            status.status.code(),
            String::from_utf8_lossy(&status.stderr)
        ));
    }
    let bytes = std::fs::read(&out_path).map_err(|e| format!("cannot read the dump: {e}"))?;

    let mut at = 0usize;
    let u64_at = |at: &mut usize| {
        let v = u64::from_le_bytes(bytes[*at..*at + 8].try_into().expect("eight bytes"));
        *at += 8;
        v
    };
    // Four size/align words, then one per entry of `PARAM_OFFSETS`.
    let abi: Vec<u64> = (0..4 + PARAM_OFFSETS.len()).map(|_| u64_at(&mut at)).collect();
    let num_sm = u64_at(&mut at) as u32;
    let blocks_per_sm = u64_at(&mut at) as u32;
    let decode_smem = u64_at(&mut at) as u32;
    let info_len = u64_at(&mut at) as usize;
    let info: Vec<i64> = (0..info_len)
        .map(|i| i64::from_le_bytes(bytes[at + i * 8..at + i * 8 + 8].try_into().expect("eight bytes")))
        .collect();
    at += info_len * 8;
    let float_bytes = u64_at(&mut at);
    let int_bytes = u64_at(&mut at);
    let upload = bytes[at..at + int_bytes as usize].to_vec();
    at += int_bytes as usize;
    let o_len = u64_at(&mut at) as usize;
    let o: Vec<u16> = (0..o_len)
        .map(|i| u16::from_le_bytes(bytes[at + i * 2..at + i * 2 + 2].try_into().expect("two bytes")))
        .collect();
    at += o_len * 2;
    let lse_len = u64_at(&mut at) as usize;
    let lse: Vec<f32> = (0..lse_len)
        .map(|i| f32::from_le_bytes(bytes[at + i * 4..at + i * 4 + 4].try_into().expect("four bytes")))
        .collect();

    Ok(Reference {
        abi,
        num_sm,
        blocks_per_sm,
        decode_smem,
        info,
        float_bytes,
        int_bytes,
        upload,
        o,
        lse,
    })
}

/// The binary spec the reference reads, built from the same `Batch` the Rust
/// side fires on.
///
/// One buffer, written once, read by both — so "the two sides saw the same
/// inputs" is a property of the code rather than a claim in a comment.
fn spec_bytes(batch: &Batch, sm_scale: f32, window_left: i32) -> Vec<u8> {
    let mut out = Vec::new();
    for word in [
        batch.batch_size(),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        PAGE_SIZE,
        batch.num_pages,
        batch.kv_indices.len() as u32,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&sm_scale.to_le_bytes());
    out.extend_from_slice(&window_left.to_le_bytes());
    for v in &batch.kv_indptr {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &batch.kv_indices {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &batch.last_page_len {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for stream in [&batch.q, &batch.k_pool, &batch.v_pool] {
        for v in stream {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// the test
// ---------------------------------------------------------------------------

/// Decode, end to end, against the kernel that ships.
#[test]
fn decode_fires_and_matches_nvcc() {
    let Some(arch) = cache::arch() else {
        println!("SKIPPED: no CUDA device or no NVRTC -- nothing to fire");
        return;
    };
    if let Err(why) = cache::bind_context() {
        println!("SKIPPED: {why}");
        return;
    }
    let reference = match build_reference(arch) {
        Ok(exe) => exe,
        Err(why) => {
            println!("SKIPPED: no reference to compare against -- {why}");
            return;
        }
    };
    let scratch = scratch();

    // One compile for the whole file: three rows, one cubin, one module. The
    // decode kernel, the merge kernel it needs when the plan splits, and the
    // ABI probe that reports what layout this very compile chose.
    let rows: Vec<&DeviceKernel> = ROWS.iter().collect();
    let compiled = match nvrtc::compile_with(&UNIT, arch, &rows, source::ALL_HEADERS) {
        Ok(c) => c,
        Err(why) => panic!(
            "the vendored FlashInfer unit did not compile -- this is the half \
             `examples/flashinfer_probe.rs` measured, so a failure here is a \
             regression in the header set or the vendored closure:\n{why}"
        ),
    };
    println!(
        "compiled {} rows into {} bytes of cubin in {:.0} ms, no include path on disk",
        rows.len(),
        compiled.cubin.len(),
        compiled.elapsed.as_secs_f64() * 1e3
    );
    let sigs: Vec<&'static KernelSig> = SIGS.iter().collect();
    let module = KernelModule::load_mangled(UNIT.name, &compiled.cubin, &sigs, &compiled.lowered)
        .expect("the cubin loads and every row resolves");

    let decode_fn = module.entry(SIGS[0].symbol).expect("the decode row resolved");
    let merge_fn = module.entry(SIGS[1].symbol).expect("the merge row resolved");

    // ---- the ABI, measured on the JIT side by the JIT's own compiler --------
    let abi_out = Buf::zeroed((4 + PARAM_OFFSETS.len()) * 8);
    let mut abi_args =
        Args::bind(&SIGS[2], &[ArgValue::Ptr(abi_out.raw())]).expect("one pointer operand");
    module
        .fire(&SIGS[2], Launch { grid: [1, 1, 1], block: [1, 1, 1], smem: 0 }, &mut abi_args, Stream::NULL)
        .expect("the ABI probe launches");
    sync("params_abi");
    let jit_abi: Vec<u64> = abi_out.download(4 + PARAM_OFFSETS.len());

    let mine: Vec<u64> = [
        size_of::<DecodeParams>() as u64,
        align_of::<DecodeParams>() as u64,
        size_of::<PagedKv>() as u64,
        align_of::<PagedKv>() as u64,
    ]
    .into_iter()
    .chain(PARAM_OFFSETS.iter().map(|(_, at)| *at as u64))
    .collect();
    let names: Vec<&str> = ["sizeof(Params)", "alignof(Params)", "sizeof(paged_kv_t)", "alignof(paged_kv_t)"]
        .into_iter()
        .chain(PARAM_OFFSETS.iter().map(|(n, _)| *n))
        .collect();
    let mut abi_bad = 0;
    for ((name, jit), rust) in names.iter().zip(&jit_abi).zip(&mine) {
        if jit != rust {
            abi_bad += 1;
            println!("  ABI MISMATCH {name:<22} NVRTC {jit:>4}   Rust mirror {rust:>4}");
        }
    }
    assert_eq!(
        abi_bad, 0,
        "the Rust params mirror disagrees with the layout NVRTC chose for the \
         struct the decode kernel reads -- every field after the first bad one \
         is being read from the wrong place, which produces numbers rather than \
         a fault"
    );
    println!(
        "params ABI: the Rust mirror and the layout NVRTC chose agree on all {} measurements \
         -- sizeof(Params) {}, alignof {}, sizeof(paged_kv_t) {}, and paged_kv.num_heads {} \
         bytes into paged_kv (the shim's fast_mod_div is 8-aligned and 24 bytes wide)",
        names.len(),
        jit_abi[0],
        jit_abi[1],
        jit_abi[2],
        jit_abi[jit_abi.len() - 1] - jit_abi[6],
    );

    // The occupancy the Rust planner must be given, from the JIT function
    // itself rather than from a constant. `plan::decode::estimate` takes
    // `max_grid_size = num_blocks_per_sm * num_sm`, and upstream computes it by
    // asking the occupancy API about the KERNEL -- so a port that guessed would
    // silently plan a different schedule on a different card.
    let (num_sm, blocks_per_sm) = occupancy(decode_fn, DECODE_THREADS, DECODE_SMEM);
    let max_grid_size = blocks_per_sm * num_sm;
    println!(
        "occupancy from the JIT cubin: {blocks_per_sm} blocks/SM x {num_sm} SMs = \
         max_grid_size {max_grid_size} (block {DECODE_THREADS} threads, smem {DECODE_SMEM} B)"
    );

    let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let window_left = -1i32;
    let mut fired = 0usize;
    let mut compared_o = 0usize;
    let mut compared_lse = 0usize;
    let mut bad_o = 0usize;
    let mut bad_lse = 0usize;

    for case in &CASES {
        println!("\n--- {} ---", case.name);
        let batch = Batch::build(case);
        let spec = spec_bytes(&batch, sm_scale, window_left);
        let reference = match run_reference(&reference, &spec, &scratch) {
            Ok(r) => r,
            Err(why) => panic!("the nvcc reference failed on `{}`: {why}", case.name),
        };

        // The reference's own view of the ABI, under real CCCL. Everything
        // agrees except the interior of `paged_kv_t`, which is the finding.
        assert_eq!(reference.abi[0], 224, "sizeof(Params) under nvcc");
        assert_eq!(reference.abi[2], 96, "sizeof(paged_kv_t) under nvcc");
        for (i, (name, _)) in PARAM_OFFSETS.iter().enumerate().take(PARAM_OFFSETS.len() - 1) {
            assert_eq!(
                reference.abi[4 + i], jit_abi[4 + i],
                "`{name}` sits at a different offset under nvcc than under NVRTC"
            );
        }
        println!(
            "nvcc's Params: {} bytes, paged_kv_t {} bytes, and paged_kv.num_heads {} bytes into \
             paged_kv against NVRTC's {} -- every OUTER offset identical, the interior four \
             bytes apart, because CCCL's fast_mod_div is 4-aligned and 20 bytes wide",
            reference.abi[0],
            reference.abi[2],
            reference.abi[PARAM_OFFSETS.len() + 3] - reference.abi[6],
            jit_abi[jit_abi.len() - 1] - jit_abi[6],
        );

        assert_eq!(
            (reference.num_sm, reference.blocks_per_sm, reference.decode_smem),
            (num_sm, blocks_per_sm, DECODE_SMEM),
            "the two cubins have different occupancy, so the two planners are \
             being asked different questions -- compare the schedules before \
             the outputs"
        );

        // ---- the Rust plan ------------------------------------------------
        let request = decode::Request {
            kv_indptr: &batch.kv_indptr,
            batch_size: batch.batch_size(),
            num_qo_heads: NUM_QO_HEADS,
            gqa_group_size: GQA_GROUP,
            page_size: PAGE_SIZE,
            head_dim: HEAD_DIM,
            enable_cuda_graph: false,
        };
        let estimate = decode::estimate(&request, max_grid_size).expect("the batch is well formed");
        assert_eq!(
            estimate.split_kv, case.expect_split,
            "`{}` no longer exercises the shape it was written for",
            case.name
        );
        let sizes = decode::workspace_size(&request, max_grid_size).expect("sizing cannot fail");
        let plan = decode::plan(
            &request,
            max_grid_size,
            Workspace::new(sizes.float_bytes, sizes.int_bytes),
        )
        .expect("the workspace was sized by the same arithmetic");

        assert_eq!(
            (plan.float_bytes as u64, plan.int_bytes as u64),
            (reference.float_bytes, reference.int_bytes),
            "the two planners want different workspaces"
        );
        assert_eq!(
            plan.info.to_vector().as_slice(),
            reference.info.as_slice(),
            "the Rust plan info differs from `DecodePlan`'s"
        );
        assert_eq!(
            plan.int_upload, reference.upload,
            "the Rust planner staged different bytes than `DecodePlan` did"
        );
        println!(
            "plan: split_kv {} padded_batch_size {} chunk {} pages, {} int bytes and {} float \
             bytes -- byte-identical to DecodePlan's",
            plan.info.split_kv,
            plan.info.padded_batch_size,
            estimate.kv_chunk_size_in_pages,
            plan.int_bytes,
            plan.float_bytes,
        );

        // ---- upload, fill, fire -------------------------------------------
        let d_q = Buf::upload(&batch.q);
        let d_k = Buf::upload(&batch.k_pool);
        let d_v = Buf::upload(&batch.v_pool);
        let d_indptr = Buf::upload(&batch.kv_indptr);
        let d_indices = Buf::upload(&batch.kv_indices);
        let d_lastlen = Buf::upload(&batch.last_page_len);
        let d_o = Buf::zeroed(batch.q.len() * 2);
        let d_lse = Buf::zeroed(batch.batch_size() as usize * NUM_QO_HEADS as usize * 4);
        // The schedule, exactly as the planner returned it. `int_upload` is the
        // block upstream would have handed `cudaMemcpyAsync`, padding and all.
        let d_int = Buf::upload(&plan.int_upload);
        let d_float = Buf::zeroed(plan.float_bytes.max(16));

        let mut params = DecodeParams {
            q: d_q.ptr(),
            q_rope_offset: 0,
            paged_kv: PagedKv {
                page_size_divisor: PAGE_SIZE,
                page_size_magic: fast_div_magic(PAGE_SIZE),
                page_size_d: PAGE_SIZE,
                num_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                batch_size: batch.batch_size(),
                // kNHD: `[max_num_pages, page_size, num_heads, head_dim]`.
                // The `__host__` constructor that computes these three is
                // behind `#ifndef __CUDACC_RTC__`, so they are computed here.
                stride_page: NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM,
                stride_n: NUM_KV_HEADS * HEAD_DIM,
                stride_h: HEAD_DIM,
                k_data: d_k.ptr(),
                v_data: d_v.ptr(),
                indices: d_indices.ptr(),
                indptr: d_indptr.ptr(),
                last_page_len: d_lastlen.ptr(),
                rope_pos_offset: 0,
                ..PagedKv::default()
            },
            o: d_o.ptr(),
            lse: d_lse.ptr(),
            maybe_alibi_slopes: 0,
            padded_batch_size: plan.info.padded_batch_size as u32,
            num_qo_heads: NUM_QO_HEADS,
            q_stride_n: (NUM_QO_HEADS * HEAD_DIM) as i32,
            q_stride_h: HEAD_DIM as i32,
            window_left,
            logits_soft_cap: 0.0,
            sm_scale,
            rope_rcp_scale: 1.0,
            rope_rcp_theta: 1.0,
            request_indices: d_int.ptr() + plan.info.request_indices_offset as u64,
            kv_tile_indices: d_int.ptr() + plan.info.kv_tile_indices_offset as u64,
            o_indptr: d_int.ptr() + plan.info.o_indptr_offset as u64,
            kv_chunk_size_ptr: d_int.ptr() + plan.info.kv_chunk_size_ptr_offset as u64,
            block_valid_mask: 0,
            partition_kv: u8::from(plan.info.split_kv),
            ..DecodeParams::default()
        };
        // The dispatcher's own swap: when the plan splits, the kernel writes
        // PARTIALS into the float workspace and the merge kernel produces `o`.
        // `block_valid_mask` stays null because graphs are off, which is what
        // `run_decode` does and what the plan's own `enable_cuda_graph` says.
        if plan.info.split_kv {
            params.o = d_float.ptr() + plan.info.v_offset as u64;
            params.lse = d_float.ptr() + plan.info.s_offset as u64;
        }

        let grid_x = plan.info.padded_batch_size as u32;
        fire_decode(decode_fn, &params, grid_x);
        sync("BatchDecodeWithPagedKVCacheKernel");
        fired += 1;

        if plan.info.split_kv {
            // `VariableLengthMergeStates`, restated: the grid is
            // `num_sm * min(occupancy, ceil_div(max_seq_len * num_heads,
            // num_sm))` and the kernel is persistent, so the grid decides how
            // the (row, head) pairs are distributed but not which CTA computes
            // what -- each pair is reduced by exactly one CTA over the same
            // `indptr` range in the same order, so the result does not depend
            // on the grid.
            let (merge_sms, merge_per_sm) = occupancy(merge_fn, MERGE_THREADS, MERGE_SMEM);
            let rows = batch.batch_size() * NUM_QO_HEADS;
            let capped = merge_per_sm.min(rows.div_ceil(merge_sms));
            let mut merge_args = Args::bind(
                &SIGS[1],
                &[
                    ArgValue::Ptr((d_float.ptr() + plan.info.v_offset as u64) as *mut c_void),
                    ArgValue::Ptr((d_float.ptr() + plan.info.s_offset as u64) as *mut c_void),
                    ArgValue::Ptr((d_int.ptr() + plan.info.o_indptr_offset as u64) as *mut c_void),
                    ArgValue::Ptr(d_o.raw()),
                    ArgValue::Ptr(d_lse.raw()),
                    ArgValue::U32(batch.batch_size()),
                    ArgValue::Ptr(std::ptr::null_mut()),
                    ArgValue::U32(NUM_QO_HEADS),
                ],
            )
            .expect("the merge row's operands are all pointers and scalars");
            module
                .fire(
                    &SIGS[1],
                    Launch {
                        grid: [merge_sms * capped, 1, 1],
                        block: [MERGE_BDX, MERGE_BDY, 1],
                        smem: MERGE_SMEM,
                    },
                    &mut merge_args,
                    Stream::NULL,
                )
                .expect("the merge kernel launches");
            sync("PersistentVariableLengthMergeStatesKernel");
            fired += 1;
            println!(
                "merged {} partials into {rows} rows over {} CTAs",
                plan.info.padded_batch_size,
                merge_sms * capped
            );
        }

        // ---- compare -------------------------------------------------------
        let o = d_o.download::<u16>(batch.q.len());
        let lse = d_lse.download::<f32>(batch.batch_size() as usize * NUM_QO_HEADS as usize);
        assert_eq!(o.len(), reference.o.len(), "the two sides produced different shapes");
        assert_eq!(lse.len(), reference.lse.len(), "the two sides produced different shapes");

        let mut first_bad: Option<String> = None;
        for (i, (mine, theirs)) in o.iter().zip(&reference.o).enumerate() {
            compared_o += 1;
            if mine != theirs {
                bad_o += 1;
                first_bad.get_or_insert_with(|| {
                    format!(
                        "o[{i}] (request {}, head {}, lane {}): JIT 0x{mine:04x} vs nvcc 0x{theirs:04x}",
                        i / (NUM_QO_HEADS * HEAD_DIM) as usize,
                        (i / HEAD_DIM as usize) % NUM_QO_HEADS as usize,
                        i % HEAD_DIM as usize,
                    )
                });
            }
        }
        for (i, (mine, theirs)) in lse.iter().zip(&reference.lse).enumerate() {
            compared_lse += 1;
            if mine.to_bits() != theirs.to_bits() {
                bad_lse += 1;
                first_bad.get_or_insert_with(|| format!("lse[{i}]: JIT {mine} vs nvcc {theirs}"));
            }
        }
        println!(
            "compared {} bf16 output lanes and {} lse floats: {} disagreements",
            o.len(),
            lse.len(),
            bad_o + bad_lse
        );
        assert!(
            first_bad.is_none(),
            "`{}`: the JIT-compiled kernel disagrees with the nvcc-compiled one \
             ({bad_o} of {} lanes, {bad_lse} of {} lse) -- first at {}",
            case.name,
            o.len(),
            lse.len(),
            first_bad.unwrap_or_default()
        );
        // A kernel that never ran leaves the zeroed buffer behind and every
        // comparison passes. The reference is the one that has to be non-empty:
        // if IT wrote nothing, the parity above is vacuous.
        assert!(
            reference.o.iter().any(|v| *v != 0),
            "the reference wrote nothing, so the comparison proved nothing"
        );

        // ---- the negative control -----------------------------------------
        //
        // Bit-identical outputs are only evidence if the comparison can tell
        // the difference. So the non-split case is fired ONCE MORE against a
        // schedule with two entries of `request_indices` transposed -- the
        // exact corruption `attention_flashinfer_common.cuh` carries twenty
        // lines about, requests answered from each other's pages, and one that
        // stays perfectly in bounds and faults nothing. If that still matched,
        // this test would be measuring whether the kernel ran at all.
        if !plan.info.split_kv {
            let mut mutated = plan.int_upload.clone();
            let at = plan.info.request_indices_offset as usize;
            mutated.swap(at, at + 4);
            let d_bad = Buf::upload(&mutated);
            let mut bad = params;
            bad.request_indices = d_bad.ptr() + plan.info.request_indices_offset as u64;
            bad.kv_tile_indices = d_bad.ptr() + plan.info.kv_tile_indices_offset as u64;
            bad.o_indptr = d_bad.ptr() + plan.info.o_indptr_offset as u64;
            bad.kv_chunk_size_ptr = d_bad.ptr() + plan.info.kv_chunk_size_ptr_offset as u64;
            let out = Buf::zeroed(batch.q.len() * 2);
            bad.o = out.ptr();
            bad.lse = 0;
            fire_decode(decode_fn, &bad, grid_x);
            sync("BatchDecodeWithPagedKVCacheKernel (control)");
            fired += 1;
            let mutated_o = out.download::<u16>(batch.q.len());
            let moved = mutated_o.iter().zip(&reference.o).filter(|(a, b)| a != b).count();
            assert!(
                moved > 0,
                "transposing two work items in the schedule changed nothing, so \
                 the comparison above is not reading what the kernel wrote"
            );
            println!(
                "negative control: transposing request_indices[0] and [1] moved {moved} of {} \
                 lanes, so the parity above is a measurement and not a tautology",
                mutated_o.len()
            );
        }
    }

    println!(
        "\nDECODE FIRED END TO END: {fired} launches, {compared_o} bf16 output lanes and \
         {compared_lse} lse floats compared against the nvcc-built kernel driven by the C++ \
         DecodePlan, {} disagreements",
        bad_o + bad_lse
    );
    // Best effort, and only on the way out through the happy path: a panic
    // leaves the reference binary and the last spec on disk, which is what you
    // want when the parity broke and you need to run it by hand.
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `cuLaunchKernel` on the decode entry, with the params struct passed BY VALUE.
///
/// This is the one launch in the file that cannot go through
/// [`KernelModule::fire`], and the reason is a gap rather than a preference:
/// the kernel's only parameter is `const __grid_constant__ Params`, a 224-byte
/// aggregate, and `runtime::args::Ty` can spell pointers, four scalar widths
/// and pointer arrays — nothing that is a struct. `Args::bind` therefore has
/// no value to marshal and the row above declares no operands at all. What the
/// driver wants is what it always wanted: an array of pointers to the argument
/// values, so one pointer to 224 bytes of host memory.
///
/// # Safety
///
/// `function` must be a live entry point and every device pointer inside
/// `params` must address an allocation of the extent the plan sized. Neither
/// is checkable here; both are the caller's, exactly as they are for every
/// launch in this tree.
fn fire_decode(function: dr::CUfunction, params: &DecodeParams, grid_x: u32) {
    let mut bytes = *params;
    let mut slot: *mut c_void = (&raw mut bytes).cast();
    // SAFETY: `slot` points at `bytes`, which outlives the call and is exactly
    // the 224 bytes this very compile was measured to want; the geometry is
    // the dispatcher's own `(padded_batch_size, num_kv_heads)` by
    // `(bdx, bdy, bdz)`; the shared-memory request is the formula
    // `BatchDecodeWithPagedKVCacheDispatched` computes, restated in
    // [`DECODE_SMEM`].
    let code = unsafe {
        dr::cuLaunchKernel(
            function,
            grid_x,
            NUM_KV_HEADS,
            1,
            BDX,
            BDY,
            BDZ,
            DECODE_SMEM,
            std::ptr::null_mut(),
            (&raw mut slot).cast(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        code,
        dr::CUresult::CUDA_SUCCESS,
        "cuLaunchKernel refused the JIT decode kernel at grid ({grid_x}, {NUM_KV_HEADS})"
    );
}

/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` on a JIT function, with the SM
/// count beside it.
///
/// The driver-API twin of what `BatchDecodeWithPagedKVCacheWorkEstimationDispatched`
/// asks the runtime API. It has to be asked about the FUNCTION, not guessed:
/// the answer depends on the register count `ptxas` chose for this cubin, so a
/// hard-coded number would silently plan a different schedule the moment the
/// kernel or the compiler changed.
fn occupancy(function: dr::CUfunction, threads: u32, smem: u32) -> (u32, u32) {
    let mut device: dr::CUdevice = 0;
    // SAFETY: a context is bound, so there is a current device.
    unsafe { dr::cuCtxGetDevice(&raw mut device) };
    let mut num_sm = 0i32;
    // SAFETY: `num_sm` is a live out-parameter and `device` is valid.
    unsafe {
        dr::cuDeviceGetAttribute(
            &raw mut num_sm,
            dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        )
    };
    let mut blocks = 0i32;
    // SAFETY: `function` is a live entry point from a loaded module.
    let code = unsafe {
        dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            function,
            threads as i32,
            smem as usize,
        )
    };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    (num_sm as u32, blocks as u32)
}

/// The unit and its rows agree with the table, without a GPU.
///
/// `tests/units.rs` does this for every declared unit and cannot do it for this
/// one — it is not in `unit::UNITS`, because a vendored unit is not compilable
/// through `nvrtc::compile`: that path resolves includes against
/// [`source::DEVICE_HEADERS`], which deliberately excludes [`source::UPSTREAM`].
/// The check that survives is the cheap one: three rows, three signatures, and
/// an instantiation string that names the right templates.
#[test]
fn the_unit_names_what_it_compiles() {
    assert_eq!(UNIT.rows.len(), SIGS.len());
    for row in UNIT.rows {
        assert!(UNIT.hosts(row.sig.symbol), "`{}` is not hosted by its own unit", row.sig.symbol);
    }
    let decode = ROWS[0].instantiation();
    assert!(
        decode.starts_with("::pie_cuda_driver::kernels::fi::BatchDecodeWithPagedKVCacheKernel<"),
        "the row must resolve through the namespace alias the root declares: {decode}"
    );
    assert!(
        decode.contains("::flashinfer::BatchDecodeParams<__nv_bfloat16"),
        "only the FIRST template argument is prefixed, so the rest must be \
         spelled from the global namespace: {decode}"
    );
    // The flag that is not global, and must not become so.
    assert_eq!(UNIT.options, &["--device-as-default-execution-space"]);
    assert!(
        UNIT.cache_key("sm_89").contains("--device-as-default-execution-space"),
        "a cubin built with a different option set must not be served for this key"
    );
}

/// The shim's magic constant, checked against the division it stands for.
///
/// `csrc/shim/cuda/cmath` proves `floor(n * M / 2^64) == floor(n / d)` for every
/// 32-bit `n`; this checks the M this file computes is the M that proof is
/// about, over the page sizes anything here will ever use plus the awkward
/// ones. A magic that is off by one divides correctly for small dividends and
/// wrongly for large ones — a corruption that appears on one long request in a
/// batch and nowhere else — so it is worth checking without a GPU.
#[test]
fn the_fastdiv_magic_is_the_shims() {
    for d in [1u32, 2, 3, 5, 7, 8, 15, 16, 17, 32, 64, 128, 256, 4096, 65_535] {
        let magic = fast_div_magic(d);
        for n in [0u32, 1, d, d.saturating_sub(1), d + 1, 1024, 65_535, 1 << 20, u32::MAX - 1, u32::MAX]
        {
            let expected = n / d;
            // What `__umul64hi(n, magic)` computes, in the one case the device
            // takes it: `d == 1` is a comparison there, not a magic.
            let got = if d == 1 { n } else { ((u128::from(n) * u128::from(magic)) >> 64) as u32 };
            assert_eq!(got, expected, "magic for {d} is wrong at {n}");
        }
    }
}

/// What a vendored [`Unit`] needs that this crate does not yet have, and what
/// MLA would need on top of that.
///
/// Documentation only. Every item below was hit while getting this file to
/// fire, worked around inside the test, and left un-worked-around in the crate
/// — because each one is a decision for the file that owns it, and a test that
/// reached into `src/runtime` to make it would have made it silently.
///
/// # 1. `Unit` cannot say which header set it compiles against
///
/// `nvrtc::compile` — and therefore `cache::module` and `runtime::fire` — passes
/// [`source::DEVICE_HEADERS`], which is prelude plus shims and deliberately
/// excludes [`source::UPSTREAM`]. A unit whose root says
/// `#include "attn/flashinfer/attention/decode.cuh"` cannot be compiled through the
/// normal path at all; this test reaches for `nvrtc::compile_with(..,
/// source::ALL_HEADERS)` instead, which no launch path calls. A `headers` field
/// on `Unit` would fix it, and [`Unit::cache_key`] must span it — the key
/// already folds `source::digest` of the set it assumed, so a unit compiled
/// against a different set would otherwise be served the wrong cubin. That is
/// the same class of bug `driver-cuda/src/program/cache.rs` records in the past
/// tense.
///
/// # 2. `DeviceKernel::instantiation` cannot name a vendored template
///
/// It formats
/// `::pie_cuda_driver::kernels::{template_path}<::pie_cuda_driver::kernels::{elem}>`,
/// prefixing the path AND the first template argument and nothing after it.
/// Three consequences, all measured here:
///
/// * a `::flashinfer::` kernel needs a namespace alias under
///   `pie_cuda_driver::kernels` before a row can name it;
/// * a first template argument that is a TYPE needs an alias too, or NVRTC
///   answers *"expected an identifier"* at `kernels::::flashinfer`;
/// * a first template argument that is a NON-TYPE — the merge kernel's
///   `vec_size` — needs a named `constexpr`, because
///   `::pie_cuda_driver::kernels::8` is not a name in any language.
///
/// The root in this file declares all three. A `DeviceKernel` that carried the
/// instantiation as one already-qualified string, or that prefixed nothing and
/// let rows spell `::`, would need none of them.
///
/// # 3. `Ty` cannot spell a by-value struct, so `Args` cannot marshal one
///
/// The decode kernel's only parameter is `const __grid_constant__ Params`, 224
/// bytes. [`ArgValue`] has pointers and four scalar widths; there is no
/// `Ty::Struct(size)` and no `ArgValue::Bytes`. So `KernelModule::fire` is
/// unusable for the kernel this whole file is about, and [`fire_decode`] builds
/// the `void*` array itself. The merge kernel and the ABI probe go through
/// `Args::bind` unchanged, which is the useful half of the finding: the
/// marshalling is fine, the vocabulary is one variant short.
///
/// # 4. There is no seam for `cuFuncSetAttribute`
///
/// `BatchDecodeWithPagedKVCacheDispatched` calls
/// `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
/// smem_size)` before every launch, because FlashInfer's decode asks for more
/// than the 48 KB static default on many configurations. This one does not —
/// 9216 bytes, computed in [`DECODE_SMEM`] and confirmed against the
/// reference's own number every run — so nothing here needs the call and
/// nothing here makes it. It WILL be needed: head dim 256 with a GQA group of
/// 1 puts the same formula over 64 KB, and the launch then fails with
/// `CUDA_ERROR_INVALID_VALUE` rather than running slowly. `Launch` carries
/// `smem` and `KernelModule::fire` passes it to `cuLaunchKernel`; the missing
/// piece is a once-per-function
/// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`
/// at module load, where it belongs — it is a property of the entry point, not
/// of a launch, and calling it per fire would be a driver round trip per token.
///
/// # 5. Occupancy is a planner input and nothing exposes it
///
/// `plan::decode::estimate` takes `max_grid_size = num_blocks_per_sm * num_sm`,
/// and upstream gets `num_blocks_per_sm` by asking the occupancy API **about
/// the kernel** — so it depends on the register count `ptxas` chose for this
/// cubin. A caller of `crate::plan` therefore needs the `CUfunction`, which
/// only `KernelModule` has and only through `entry`. This test calls
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` itself and then checks the
/// answer against the reference's: **8 blocks/SM on 142 SMs, identical for the
/// NVRTC and the nvcc cubin.** They are not obliged to be, and if they ever
/// diverge the two planners are answering different questions and the output
/// comparison is meaningless — which is why that equality is asserted before
/// anything is compared.
///
/// # 6. `source::roots` is hand-written
///
/// `Unit::root` comes from a two-entry module that `build.rs` does not
/// generate — the generator walks headers only. A vendored unit needs an entry
/// there, or a root carried some other way. This test carries its own [`ROOT`]
/// as a `const`, which is fine for a test and wrong for the crate.
///
/// # 7. What MLA would need
///
/// Not this. `BatchMLAPagedAttentionKernel` calls `grid.sync()` and must be
/// launched with `cudaLaunchCooperativeKernel` / `cuLaunchCooperativeKernel`;
/// `BatchDecodeWithPagedKVCacheKernelMLA` is the non-cooperative sibling and is
/// a different question. Three things are missing rather than one:
///
/// * `cooperative_groups.h` in `csrc/src` omits `this_grid()` **on purpose**
///   (§13.5) so that a kernel needing it fails to compile instead of silently
///   not synchronising. Adding it is the first decision, and it is a decision
///   about the shim's honesty, not about MLA.
/// * `KernelModule::fire` calls `cuLaunchKernel`. A cooperative launch is a
///   different entry point with a different failure mode — it refuses outright
///   if the grid exceeds what fits concurrently, which means the GRID must come
///   from an occupancy query rather than from a rectangle. `LaunchRule` has no
///   shape for "as many blocks as fit", so MLA needs either a new rule or a
///   `Launch` the planner hands over whole.
/// * The plan is different: `plan::mla` exists and produces `MlaPlanInfo`, but
///   the kernel is driven from `work_indptr` and a persistent grid, so the
///   padded-batch reasoning this file relies on does not carry over.
///
/// Decode needed none of the three, which is why decode is what fires here.
///
/// [`source::DEVICE_HEADERS`]: kernels_cuda_new::source::DEVICE_HEADERS
/// [`source::UPSTREAM`]: kernels_cuda_new::source::UPSTREAM
#[allow(dead_code)]
mod what_a_vendored_unit_needs {}
