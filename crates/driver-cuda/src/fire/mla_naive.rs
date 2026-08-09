//! `attention_mla_naive.cuh`'s two launchers, in Rust.
//!
//! The whole host half of the naive/Blackwell MLA path: a scalar flash-softmax
//! kernel and a tensor-core one, the shape predicate that chooses between
//! them, and the shared-memory arithmetic each needs. The device text they
//! fire is `kernels-cuda-new/csrc/src/attn/attention_mla_naive.cuh`, which is
//! now device text and nothing else — see [`super::mla_paged`] for the same
//! shape done for a different file.
//!
//! # These were the last two launches nvcc could reach, and they were miscounted
//!
//! `kernels-cuda/tests/sources.rs` counts launches over `.cu` and `.cpp`. That
//! scope was correct for the entire life of this tree: a `.cuh` under
//! `kernels-cuda-new/csrc` is device text carried into NVRTC, and device text
//! does not launch, so the extension WAS the classification. It stopped being
//! the classification the moment one `.cuh` grew a launcher, and this was that
//! `.cuh` — so the whole-tree census read zero while these two were live
//! (`new-horizon.md` §63.3). The repair was not a wider scan: the launches
//! moved into `kernels-cuda/csrc/src/attn/attention_mla.cu`, which already
//! held their only caller, so the count there is 2 and honest.
//!
//! # The specification of this program, in four terms
//!
//! Written against the code below in [`super::xqa`]'s shape, so a reader can
//! check the code against a claim rather than infer the claim from the code.
//!
//! **1. Which JIT units it fires, in what order.** One unit,
//! `attn/attention_mla_naive`; two rows; exactly ONE fire per call. The two
//! kernels are alternatives, not a sequence: [`plan`] returns whichever the
//! shape admits and the other is not launched. There is no composition and no
//! `execution::Step`.
//!
//! **2. What intermediate buffers sit between them.** None. Both kernels read
//! the caller's paged cache and write the caller's output; the only scratch is
//! dynamic shared memory, which is per-block and never outlives a launch.
//! Unlike [`super::xqa`] there is no carve, so there is no layout to keep in
//! step with a second copy.
//!
//! **3. What it decides on the host.** Four decisions and three refusals:
//!
//! ```text
//!   #  what                                   from                picks
//!   1  total_tokens <= 0                      operand             DECLINE
//!   2  qo/kv_page_indptr/kv_last_page_lens     operand             DECLINE
//!   3  mla_mma_supported(ckv, kpe, heads)      operands, consts    WHICH KERNEL
//!   4  the head group G (scalar arm only)      operands, consts    grid.y, arg
//!   5  ckv % 32, ckv/32 <= 16                  operand             DECLINE
//!   6  kpe % 32, kpe/32 <= 4                   operand             DECLINE
//! ```
//!
//! Only 3 and 4 pick anything. 3 is a pure predicate over three integers and
//! is [`mma_supported`]. 4 is the wave-target loop and is [`head_group`] —
//! the one piece of real host arithmetic here, and the one a `LaunchRule`
//! could not state, because it is a search and not a formula.
//!
//! **4. What in `Source` / `LaunchRule` / `Specialisation` / `Execution` is
//! missing to state it.** In one line each:
//!
//! * **`LaunchRule`** — no rule opens either rectangle, and the two are not
//!   even the same rectangle transposed by accident: the scalar arm is
//!   `(tokens, heads/G)` and the tensor-core arm is `(heads/16, tokens)`, x
//!   and y swapped, which is a real difference in which axis gets the 2^31
//!   extent and which gets 65 535. Both rows are `LaunchRule::Unstated` and
//!   this module builds the [`Launch`] itself, exactly as
//!   [`super::mla_paged`] does for `mla_prepare`.
//! * **`Specialisation`** — the mma/scalar choice is a `Term` over three
//!   integers and could be spelled. It is not spelled because the row that
//!   would carry it (`attn::dispatch_attention_mla_bf16`) is not served from
//!   here yet; see "What blocks it" below.
//! * **`Source`** — `G` is `Control::Supplies`-shaped: passed to the kernel
//!   AND dividing the head axis of the grid, which is the case that variant's
//!   own doc names. Same for nothing else here.
//! * **`Execution`** — nothing missing. One fire, no walk, no composition.
//!
//! # What `std::call_once` became, and why the Rust is stronger
//!
//! Both C++ launchers guarded their `cudaFuncSetAttribute` with a function
//! `static std::once_flag`:
//!
//! ```text
//! attention_mla_naive.cuh:259-264   static std::once_flag smem_optin;
//!                                   std::call_once(smem_optin, [&] {
//!                                       cudaFuncSetAttribute(
//!                                           mla_naive_paged_kernel,
//!                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
//!                                           200 * 1024); });
//! attention_mla_naive.cuh:717-723   static std::once_flag opt_in;   // same, at smem_bytes()
//! ```
//!
//! **It became once per `(CUdevice, CUfunction)`, not once per process and not
//! once per module**, and it is not written here at all:
//! `kernels_cuda_new::runtime::module::raise_dynamic_smem_cap` does it inside
//! [`kernels_cuda_new::runtime::KernelModule::fire`], keyed on
//! `(device, function.addr())` with a high-water mark, whenever
//! `launch.smem > 48 KiB`. This module sets [`Launch::smem`] and nothing else,
//! which is why there is no `OnceLock` below. The north star's §5 step 1
//! `smem_opt_in` hook is that same mechanism seen from `x::Launch`'s side —
//! a statement by the author, checked against the driver fact rather than
//! replacing it — and adding a second opt-in path here would be the two
//! disagreeing.
//!
//! **Once per process was a latent bug and the port does not carry it.** A
//! `std::once_flag` fires on the first call in the process, on whatever device
//! that call's context belonged to. `cuFuncSetAttribute` is per (device,
//! function). On a two-GPU box, the second device never receives the opt-in and
//! its first launch fails with `CUDA_ERROR_INVALID_VALUE` — a diagnostic that
//! names the launch and not the missing attribute. `cache::module`'s lifetime
//! (once per compiled module) would have the same hole for the same reason,
//! which is why the answer to "which did you choose" is neither.
//!
//! # The geometry, cited
//!
//! ```text
//! attention_mla_naive.cuh:265   dim3 grid(total_tokens, num_heads / G);
//! attention_mla_naive.cuh:266   mla_naive_paged_kernel<grid, kMlaNaiveBlock, smem, stream>(...)
//! attention_mla_naive.cuh:45    constexpr int kMlaNaiveBlock = 256;
//! attention_mla_naive.cuh:252   smem = (kMlaNaiveWarps * CKV + 2 * kMlaNaiveWarps) * sizeof(float)
//!
//! attention_mla_naive.cuh:725   dim3 grid(num_heads / kBM, total_tokens);
//! attention_mla_naive.cuh:726   mla_mma_paged_kernel<grid, kThreads, smem, stream>(...)
//! attention_mla_naive.cuh:329   constexpr int kThreads = kWarps * 32;      // 8 * 32 = 256
//! attention_mla_naive.cuh:682   smem_bytes() = (kBM*kLdD + kStages*kBK*kLdD + kBM*kLdP) * 2
//!                                            + (kBM*kBK + 3*kBM) * 4
//! ```
//!
//! Line numbers are the file as it stood before the split, which is the state
//! `git show` reaches and the state every other citation in this migration
//! uses. The launchers now live at `attention_mla.cu:59-200`.
//!
//! **The two grids are transposed and it is not a typo.** The scalar arm puts
//! tokens on x and head groups on y; the tensor-core arm puts head blocks on x
//! and tokens on y. A port that "tidied" them into agreement would silently cap
//! one of them: `grid.y` and `grid.z` are 16-bit-limited to 65 535 and `grid.x`
//! is not, so the scalar arm supports 2^31 tokens and 65 535 head groups while
//! the mma arm supports 65 535 tokens and 2^31 head blocks. Both are stated as
//! the C++ stated them.
//!
//! # What blocked this module from replacing the C++, and what is left
//!
//! **IT HAS REPLACED IT.** `kernels-cuda/csrc/src/attn/attention_mla.cu` is
//! deleted and this module is the only thing that reaches either of these two
//! kernels.
//!
//! The blocker recorded here was that `attn::dispatch_attention_mla_bf16` is
//! one row with TWO arms — this naive one for sm_100, and
//! `flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>` for everything
//! below it — and that a row loses its shim entry whole or not at all, so both
//! arms had to be Rust before either could be. The FA2 arm passed ONE
//! `MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>` **by value**, which is
//! `runtime::args::ArgValue::Bytes`, which only `x::Abi` produces. That was
//! the identical blocker XQA's `KVCacheList` had, and one `Abi` capability
//! did clear both: `by_value!` grew an UNTAGGED arm and
//! `kernels_cuda_new::x::attn::mla_params` measured `MLAParams` at 288/8.
//! The FA2 arm is now `kernels_cuda_new::x::attn::mla_fa2`, unit and all, and
//! the row crossed to `x::attn::ATTENTION_MLA`.
//!
//! **WHAT IS LEFT IS THE ARM CHOICE AND NOTHING ELSE.** `attention_mla.cu:150`
//! made it on `cudaDevAttrComputeCapabilityMajor >= 10`, because FA2 MLA
//! writes ZERO OUTPUT on sm_100 — a wrong answer, not a fault. Neither `Cx`
//! nor `kernels-cuda-new`'s runtime states a compute capability, and `Cx`
//! states none of the MLA cache layer, the plan handle, the attention
//! workspace or `sm_scale` either. So the contract is a `none:` arm, which
//! names all five, and this module still waits on a caller — as it always
//! has, and now for a reason that is one query list rather than a capability.
//!
//! Everything in THIS module is unblocked: every operand of both kernels is a
//! pointer, an `i32`, an `f32` or a `bool`.

use std::ffi::c_void;

use kernels_cuda_new::ArgValue;
use kernels_cuda_new::jit::Launch;

/// `attn::mla_naive_paged` — the scalar arm's device row.
const NAIVE_DEVICE: &str = "attn::mla_naive_paged";

/// `attn::mla_mma_paged` — the tensor-core arm's device row.
const MMA_DEVICE: &str = "attn::mla_mma_paged";

/// `attention_mla_naive.cuh:45` — `constexpr int kMlaNaiveBlock = 256;`.
///
/// A block width AND, through `kMlaNaiveWarps`, the divisor of the shared
/// allocation: named once and used three times, which is why it is a constant
/// here rather than a literal in [`plan`].
pub const NAIVE_BLOCK: u32 = 256;

/// `attention_mla_naive.cuh:46` — `kMlaNaiveBlock / 32`, the warps per block.
///
/// It is the head-group ceiling as well as the warp count: the C++ starts `G`
/// at `kMlaNaiveWarps` and halves, so 8 is the largest group any shape gets.
pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

/// `attention_mla_naive.cuh:47` — `kv_lora_rank / 32` must not exceed this.
///
/// The comment beside it is the measurement and travels with the number:
/// *"kv_lora_rank <= 512 with 32 lanes"*.
pub const NAIVE_MAX_PER: i32 = 16;

/// `attention_mla_naive.cuh:48` — `qk_rope_head_dim / 32` must not exceed this.
///
/// *"qk_rope_head_dim <= 128 with 32 lanes"*.
pub const NAIVE_MAX_PE_PER: i32 = 4;

/// `attention_mla_naive.cuh:239` — `const int kMlaWaveTarget = 296;`.
///
/// The measurement it encodes is at `:235-237` and must survive the port
/// verbatim, because nothing about the number can be re-derived from the code:
///
/// > Pick the largest head group that still fills the machine. Every head in a
/// > block walks the same keys, so a bigger group means the latent KV is read
/// > from L1 instead of L2/HBM — but the grid is (tokens x head-groups), so
/// > shrinking it too far starves the SMs. Two waves is the target.
///
/// 296 is two waves of 148 SMs, which is a B200. It is a target and not a
/// bound: [`head_group`] stops halving when the grid first reaches it, so a
/// shape that cannot reach 296 at `G == 1` simply runs at `G == 1`.
pub const WAVE_TARGET: i64 = 296;

/// `attention_mla_naive.cuh:238` — `constexpr int kForcedGroup = 0;`.
///
/// An override left at its off value, kept because it documents that `G` is a
/// tuning knob someone reached for. `0` means "use the wave-target search";
/// any positive value pins `G` and then halves it until it divides both
/// `num_heads` and `kMlaNaiveWarps`. The pinned arm is transcribed in
/// [`head_group_forced`] rather than dropped, so that turning it back on is a
/// call-site change and not a re-derivation.
pub const FORCED_GROUP: i32 = 0;

/// `attention_mla_naive.cuh:324` — `constexpr int kBM = 16;`.
///
/// *"query rows per block == heads"*. It is the mma `m16n8k16` tile's M and the
/// divisor of the tensor-core grid's x axis, so a change here moves the grid.
pub const MMA_BM: i32 = 16;

/// `attention_mla_naive.cuh:329` — `kWarps * 32`, with `PIE_MLA_MMA_WARPS = 8`.
pub const MMA_THREADS: u32 = 256;

/// `attention_mla_naive.cuh:682-687` — `mma_detail::smem_bytes()`, evaluated.
///
/// ```text
/// (kBM*kLdD + kStages*kBK*kLdD + kBM*kLdP) * sizeof(__nv_bfloat16)
///     + (kBM*kBK + 3*kBM) * sizeof(float)
/// ```
///
/// with `kBM = 16` (`:324`), `kBK = PIE_MLA_MMA_BK = 64` (`:325`, `:303`),
/// `kStages = PIE_MLA_MMA_STAGES = 1` (`:327`, `:315`), `kLdD = kD + 8 = 584`
/// (`:333`) and `kLdP = kBK + 8 = 72` (`:334`):
///
/// ```text
/// (16*584 + 1*64*584 + 16*72) * 2 + (16*64 + 3*16) * 4
///   = (9 344 + 37 376 + 1 152) * 2 + (1 024 + 48) * 4
///   = 95 744 + 4 288
///   = 100 032
/// ```
///
/// **Two independent measurements in the file corroborate it**, which is why
/// it is written as a constant rather than recomputed here from five other
/// constants that could each drift:
///
/// * `:309-310` prices one extra pipeline stage at *"a full sK copy (73 KB)"*.
///   `kStages*kBK*kLdD*2` is `64*584*2 = 74 752` = 73.0 KiB exactly.
/// * `:311-313` says two stages *"on B200 drops the block occupancy from 2/SM
///   to 1/SM"*. B200 has 228 KiB of shared memory per SM; `2 * 100 032 =
///   195.4 KiB` fits and `2 * 174 784` does not, and `__launch_bounds__(kThreads,
///   PIE_MLA_MMA_MINBLK = 2)` at `:420`/`:321` asks for exactly the 2 that fits.
///
/// **`:334`'s trailing comment `// 40` is stale and is not evidence.** `kLdP`
/// is `kBK + 8`, which is 40 only at `kBK = 32`; the default has been 64 since
/// `PIE_MLA_MMA_BK` was introduced. The two measurements above are computed
/// against 64 and agree with it, so the comment is what drifted. Recorded here
/// rather than corrected in place, because the `.cuh` is device text under a
/// probe and this is the file that carries host arithmetic.
pub const MMA_SMEM_BYTES: u32 = 100_032;

/// The dynamic shared memory the scalar kernel asks for, in bytes.
///
/// `attention_mla_naive.cuh:251-254`:
///
/// ```text
/// smem = (kMlaNaiveWarps * CKV + 2 * kMlaNaiveWarps) * sizeof(float)
/// ```
///
/// One `float` accumulator row per warp across the latent width, plus the two
/// per-warp partial-softmax scalars (`m` and `l`).
#[must_use]
pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
    let per = NAIVE_WARPS as i64 * kv_lora_rank as i64 + 2 * NAIVE_WARPS as i64;
    let bytes = per * 4;
    if bytes < 0 { 0 } else { bytes as u32 }
}

/// Whether an MLA naive launch ran, and which kernel.
///
/// `#[must_use]` for [`super::gemv`]'s reason: *"it declined"* must not be
/// spellable like *"it ran"*.
#[must_use]
pub enum MlaNaive {
    /// The scalar flash-softmax kernel was launched on the caller's stream.
    LaunchedScalar,
    /// The tensor-core kernel was launched on the caller's stream.
    LaunchedMma,
    /// Nothing was launched, and why.
    Declined(NaiveDecline),
}

/// The four ways a naive MLA launch declines.
///
/// Three of them were a `throw` in the C++ and one was a bare `return`, and
/// the difference is preserved rather than flattened: [`NaiveDecline::NoTokens`]
/// is a legal empty fire and the other three are the caller having asked for a
/// shape this kernel pair cannot serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NaiveDecline {
    /// `attention_mla_naive.cuh:211` — `if (total_tokens <= 0) return;`.
    ///
    /// A bare `return` in the C++, not a throw. Both kernels open one grid
    /// lane per token, so an empty batch is an empty grid, which the driver
    /// rejects as `Error::Geometry`.
    NoTokens,
    /// `attention_mla_naive.cuh:212-217` — a null `qo_indptr`,
    /// `kv_page_indptr` or `kv_last_page_lens`.
    ///
    /// A `throw` in the C++, whose message names all three:
    /// *"naive MLA: missing device indptr/lens (qo/kv_page_indptr/
    /// kv_last_page_lens)"*. The naive path is the only consumer of these
    /// three, which is why `attn/attention_mla.hpp` documents them as
    /// *"Ignored by the FlashInfer FA2 path"* and defaulted to null.
    MissingIndptr,
    /// `attention_mla_naive.cuh:228-230` — `CKV % 32 != 0 || CKV / 32 > 16`.
    ///
    /// *"naive MLA: unsupported kv_lora_rank"*. 32 is the warp, and
    /// `kMlaNaiveMaxPer = 16` caps the per-lane accumulator array the kernel
    /// declares, so a wider latent would index past a register array.
    UnsupportedKvLoraRank,
    /// `attention_mla_naive.cuh:231-233` — `KPE % 32 != 0 || KPE / 32 > 4`.
    ///
    /// *"naive MLA: unsupported qk_rope_head_dim"*, and the same argument at
    /// `kMlaNaiveMaxPePer = 4`.
    UnsupportedRopeDim,
}

/// `attention_mla_naive.cuh:698-701` — whether the tensor-core kernel applies.
///
/// The C++ comment above it is the measurement and is reproduced whole,
/// because it is the only record of which models the fast path serves:
///
/// > Requires kv_lora_rank == 512, qk_rope_head_dim == 64 and
/// > num_heads % 16 == 0 (true for GLM-5.2, Kimi K2.6 and both DeepSeek-V4
/// > variants); anything else falls back to the scalar kernel.
///
/// The three constants are `mma_detail::kCkv` (`:330`), `kKpe` (`:331`) and
/// `kBM` (`:324`) — the tile the `mma.sync` shapes are written against, not a
/// tuning choice.
///
/// **The C++ `forced` override is not reproduced.** `:692-697` is a
/// `static const int forced = [] { return 0; if (false) return -1; if (false)
/// return 1; return 0; }();` — a debug switch whose two live arms are
/// unreachable after the first `return 0`, so it evaluates to 0 and the
/// function is exactly the predicate below. Transcribing dead statements into
/// a second language is how a debug switch becomes a feature.
#[must_use]
pub const fn mma_supported(kv_lora_rank: i32, qk_rope_head_dim: i32, num_heads: i32) -> bool {
    kv_lora_rank == 512 && qk_rope_head_dim == 64 && num_heads % MMA_BM == 0
}

/// `attention_mla_naive.cuh:241-249` — the head group `G`, by wave-target search.
///
/// ```text
/// int G = kMlaNaiveWarps;
/// while (G > 1 && (num_heads % G != 0 ||
///                  (long long)total_tokens * (num_heads / G) < kMlaWaveTarget)) {
///     G >>= 1;
/// }
/// ```
///
/// Start at 8 and halve until `G` divides the head count AND the resulting
/// grid is at least [`WAVE_TARGET`] blocks. The loop terminates at `G == 1`,
/// which divides everything, so a shape too small for two waves runs
/// key-parallel — the degenerate case `:56-58` argues for: *"With `G == 1`
/// this degenerates to the pure key-parallel layout, which is what small
/// batches want (there the grid is the only source of parallelism)."*
///
/// The multiplication is `long long` in the C++ and `i64` here for the same
/// reason: `total_tokens * (num_heads / G)` is the grid's block count and a
/// 32-bit product would wrap on a long prefill before it ever compared.
#[must_use]
pub fn head_group(num_heads: i32, total_tokens: i32) -> i32 {
    let mut g = NAIVE_WARPS;
    while g > 1
        && (num_heads % g != 0 || i64::from(total_tokens) * i64::from(num_heads / g) < WAVE_TARGET)
    {
        g >>= 1;
    }
    g
}

/// `attention_mla_naive.cuh:242-244` — the pinned arm of the same choice.
///
/// ```text
/// G = kForcedGroup;
/// while (G > 1 && (num_heads % G != 0 || kMlaNaiveWarps % G != 0)) G >>= 1;
/// ```
///
/// Reachable only when [`FORCED_GROUP`] is positive, which it is not. Kept
/// because the two loops differ in a way that is easy to get wrong if it is
/// ever re-derived: the forced arm tests `kMlaNaiveWarps % G` and NOT the wave
/// target, so it never trades the L1 hit rate for occupancy — which is the
/// whole reason someone would pin it.
#[must_use]
pub fn head_group_forced(num_heads: i32, forced: i32) -> i32 {
    let mut g = forced;
    while g > 1 && (num_heads % g != 0 || NAIVE_WARPS % g != 0) {
        g >>= 1;
    }
    g
}

/// Everything a naive MLA fire needs that is not a pointer.
///
/// Grouped because the two launchers take fourteen scalars between them and a
/// fourteen-argument `unsafe fn` is where an argument gets transposed. The
/// field order is the C++ launcher's parameter order.
#[derive(Clone, Copy, Debug)]
pub struct NaiveShape {
    /// `layer.kv_lora_rank` — the latent width, `CKV` in the C++.
    pub kv_lora_rank: i32,
    /// `layer.qk_rope_head_dim` — the rope tail, `KPE` in the C++.
    pub qk_rope_head_dim: i32,
    /// Tokens per page.
    pub page_size: i32,
    /// Query rows in this batch.
    pub total_tokens: i32,
    /// Requests in this batch, for the CSR walk.
    pub num_requests: i32,
    /// Query heads.
    pub num_heads: i32,
    /// The softmax scale.
    pub sm_scale: f32,
    /// Whether the mask is causal.
    pub causal: bool,
    /// `index_mask`'s row stride; 0 when the mask is null.
    pub index_mask_stride: i32,
}

/// The device pointers both kernels take.
///
/// Ordered as the `__global__`s declare them, which is what
/// `Args::bind` checks against the row's signature.
#[derive(Clone, Copy, Debug)]
pub struct NaivePtrs {
    /// `[tokens, heads, kv_lora_rank]` bf16.
    pub q_nope: *const c_void,
    /// `[tokens, heads, qk_rope_head_dim]` bf16.
    pub q_pe: *const c_void,
    /// `[pages, page_size, kv_lora_rank]` bf16.
    pub ckv_pages: *const c_void,
    /// `[pages, page_size, qk_rope_head_dim]` bf16.
    pub kpe_pages: *const c_void,
    /// Per-request query offsets.
    pub qo_indptr: *const u32,
    /// The page list.
    pub kv_page_indices: *const u32,
    /// Per-request page offsets.
    pub kv_page_indptr: *const u32,
    /// Tokens used in each request's last page.
    pub kv_last_page_lens: *const u32,
    /// `[tokens, heads, kv_lora_rank]` bf16, written.
    pub o: *mut c_void,
    /// The DSA top-k mask, or null for dense.
    pub index_mask: *const u8,
}

/// Which kernel a shape selects and the rectangle it runs at.
///
/// Separated from [`fire`] so the choice can be asserted without a CUDA
/// context — the geometry is the part a test can check and the launch is not.
#[must_use]
pub enum NaivePlan {
    /// Fire [`NAIVE_DEVICE`] at this rectangle, with this head group.
    Scalar {
        /// The rectangle, `attention_mla_naive.cuh:265-266`.
        launch: Launch,
        /// `G`, which the kernel takes as its last argument AND which divided
        /// the grid's y axis. [`head_group`] computed it.
        head_group: i32,
    },
    /// Fire [`MMA_DEVICE`] at this rectangle.
    Mma {
        /// The rectangle, `attention_mla_naive.cuh:725-726`.
        launch: Launch,
    },
    /// Neither, and why.
    Declined(NaiveDecline),
}

/// Choose the kernel and build its rectangle.
///
/// `attention_mla_naive.cuh:199-276` down to but not including the launch: the
/// three refusals, the mma predicate, the head-group search and both
/// shared-memory figures. No CUDA call, so this is the whole of what a test can
/// pin.
#[must_use]
pub fn plan(shape: NaiveShape, have_indptr: bool) -> NaivePlan {
    // `:211` — a bare `return`, not a throw.
    if shape.total_tokens <= 0 {
        return NaivePlan::Declined(NaiveDecline::NoTokens);
    }
    // `:212-217`.
    if !have_indptr {
        return NaivePlan::Declined(NaiveDecline::MissingIndptr);
    }
    // `:218-225` — the mma arm is tested BEFORE the scalar arm's shape
    // refusals, and the order is load-bearing: `kv_lora_rank == 512` passes
    // `CKV / 32 > kMlaNaiveMaxPer` at exactly 16, so the two agree today, but
    // the mma arm does not depend on the scalar arm's bounds and must not
    // start doing so if either constant moves.
    if mma_supported(shape.kv_lora_rank, shape.qk_rope_head_dim, shape.num_heads) {
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch {
            // `:725` — `dim3 grid(num_heads / kBM, total_tokens);`. Head
            // blocks on x, tokens on y: the transpose of the scalar arm.
            grid: [(shape.num_heads / MMA_BM).max(0) as u32, shape.total_tokens.max(0) as u32, 1],
            block: [MMA_THREADS, 1, 1],
            smem: MMA_SMEM_BYTES,
            cooperative: false,
        };
        return NaivePlan::Mma { launch };
    }
    // `:227-233`.
    let ckv = shape.kv_lora_rank;
    let kpe = shape.qk_rope_head_dim;
    if ckv % 32 != 0 || ckv / 32 > NAIVE_MAX_PER {
        return NaivePlan::Declined(NaiveDecline::UnsupportedKvLoraRank);
    }
    if kpe % 32 != 0 || kpe / 32 > NAIVE_MAX_PE_PER {
        return NaivePlan::Declined(NaiveDecline::UnsupportedRopeDim);
    }
    // `:238-250`.
    let g = if FORCED_GROUP > 0 {
        head_group_forced(shape.num_heads, FORCED_GROUP)
    } else {
        head_group(shape.num_heads, shape.total_tokens)
    };
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        // `:265` — `dim3 grid(total_tokens, num_heads / G);`.
        grid: [shape.total_tokens.max(0) as u32, (shape.num_heads / g.max(1)).max(1) as u32, 1],
        block: [NAIVE_BLOCK, 1, 1],
        smem: naive_smem_bytes(ckv),
        cooperative: false,
    };
    NaivePlan::Scalar { launch, head_group: g }
}

/// `attention_mla_naive.cuh:199` — `launch_mla_naive_paged_raw`, whole.
///
/// [`plan`] chooses; this binds the operands and fires. The operand order is
/// each `__global__`'s own, which is what the row's signature states and what
/// `Args::bind` refuses a drift from.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer in `ptrs` is a device address the caller keeps live across the
/// launch, and `stream` is the caller's stream.
pub unsafe fn launch(ptrs: NaivePtrs, shape: NaiveShape, stream: *mut c_void) -> MlaNaive {
    let have_indptr = !ptrs.qo_indptr.is_null()
        && !ptrs.kv_page_indptr.is_null()
        && !ptrs.kv_last_page_lens.is_null();
    match plan(shape, have_indptr) {
        NaivePlan::Declined(why) => MlaNaive::Declined(why),
        NaivePlan::Mma { launch } => {
            // `attention_mla_naive.cuh:420-431` — the `__global__`'s
            // parameters. It takes NEITHER `kv_lora_rank` NOR
            // `qk_rope_head_dim`: both are `mma_detail` constants the kernel
            // is compiled against, which is why `mma_supported` compares them
            // rather than forwarding them.
            let values = [
                ArgValue::Ptr(ptrs.q_nope.cast_mut()),
                ArgValue::Ptr(ptrs.q_pe.cast_mut()),
                ArgValue::Ptr(ptrs.ckv_pages.cast_mut()),
                ArgValue::Ptr(ptrs.kpe_pages.cast_mut()),
                ArgValue::Ptr(ptrs.qo_indptr.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_page_indices.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_page_indptr.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_last_page_lens.cast_mut().cast()),
                ArgValue::Ptr(ptrs.o),
                ArgValue::Ptr(ptrs.index_mask.cast_mut().cast()),
                ArgValue::I32(shape.index_mask_stride),
                ArgValue::I32(shape.num_requests),
                ArgValue::I32(shape.num_heads),
                ArgValue::I32(shape.page_size),
                ArgValue::F32(shape.sm_scale),
                ArgValue::Bool(shape.causal),
            ];
            super::hand::fire(
                &kernels_cuda_new::x::attn::mla_naive::ROOT,
                kernels_cuda_new::x::attn::mla_naive::inst::MMA_PAGED,
                launch,
                &values,
                stream,
            );
            MlaNaive::LaunchedMma
        }
        NaivePlan::Scalar { launch, head_group } => {
            // `attention_mla_naive.cuh:66-78` — the `__global__`'s parameters,
            // and note the tail: `R, H, CKV, KPE, page_size, sm_scale, causal,
            // G`. `G` is last and is the value the grid's y axis was divided
            // by, which is `Control::Supplies`' case exactly.
            let values = [
                ArgValue::Ptr(ptrs.q_nope.cast_mut()),
                ArgValue::Ptr(ptrs.q_pe.cast_mut()),
                ArgValue::Ptr(ptrs.ckv_pages.cast_mut()),
                ArgValue::Ptr(ptrs.kpe_pages.cast_mut()),
                ArgValue::Ptr(ptrs.qo_indptr.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_page_indices.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_page_indptr.cast_mut().cast()),
                ArgValue::Ptr(ptrs.kv_last_page_lens.cast_mut().cast()),
                ArgValue::Ptr(ptrs.o),
                ArgValue::Ptr(ptrs.index_mask.cast_mut().cast()),
                ArgValue::I32(shape.index_mask_stride),
                ArgValue::I32(shape.num_requests),
                ArgValue::I32(shape.num_heads),
                ArgValue::I32(shape.kv_lora_rank),
                ArgValue::I32(shape.qk_rope_head_dim),
                ArgValue::I32(shape.page_size),
                ArgValue::F32(shape.sm_scale),
                ArgValue::Bool(shape.causal),
                ArgValue::I32(head_group),
            ];
            super::hand::fire(
                &kernels_cuda_new::x::attn::mla_naive::ROOT,
                kernels_cuda_new::x::attn::mla_naive::inst::NAIVE_PAGED,
                launch,
                &values,
                stream,
            );
            MlaNaive::LaunchedScalar
        }
    }
}

/// The 200 KiB opt-in, and why this module does not carry it.
///
/// `attention_mla_naive.cuh:259-264` asked for **200 * 1024 = 204 800 bytes**
/// of dynamic shared memory for the scalar kernel, once per process, with this
/// justification at `:255-258`:
///
/// > Wide blocks are what make this kernel fast at decode: the grid is only
/// > (tokens x head-groups), so with a narrow block the SMs sit at single-digit
/// > occupancy and every key's load latency is exposed. The partial-softmax
/// > scratch that buys the extra warps can exceed the 48 KB default.
///
/// **The first two sentences are the design and stand. The third is false
/// against the file's own constants, and it is false by a factor of three.**
/// [`naive_smem_bytes`] is `(8 * CKV + 16) * 4 = 32 * CKV + 64`, and the
/// refusal at `:228` caps `CKV` at `32 * kMlaNaiveMaxPer = 512`. So the
/// largest allocation this kernel can ever request is
///
/// ```text
/// 32 * 512 + 64 = 16 448 bytes
/// ```
///
/// — 16.1 KiB, against a 48 KiB default it would have to exceed by 3x before
/// any opt-in were needed. To reach 49 152 the latent would have to be 1 535
/// wide, which `:228` rejects. **The `cudaFuncSetAttribute` was unreachable
/// dead weight**, not a live requirement, and the Rust does not reproduce it:
/// `raise_dynamic_smem_cap` is threshold-driven, sees 16 448, and correctly
/// does nothing.
///
/// It is recorded rather than dropped for two reasons. The 200 KiB is a
/// MEASUREMENT — someone chose it — and the rule is that a measurement
/// survives a port even when the port stops acting on it. And the comment's
/// first two sentences explain why the block is 256 wide, which IS live and
/// would have been lost with the number.
///
/// The tensor-core arm is the opposite case and needs no note beyond
/// [`MMA_SMEM_BYTES`]: 100 032 bytes genuinely exceeds 48 KiB, its C++ opt-in
/// asked for exactly `smem_bytes()` rather than a round number, and
/// `raise_dynamic_smem_cap` raises it to exactly that.
pub const NAIVE_OPT_IN_BYTES_UNREACHED: u32 = 200 * 1024;
