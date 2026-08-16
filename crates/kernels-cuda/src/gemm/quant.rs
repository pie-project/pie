//! `gemm/gemm.cpp`'s quantized arms: the router on the weight's dtype, and
//! the four cuBLASLt recipes behind it.
//!
//! # What this is
//!
//! `gemm.cpp` had 2,216 lines, zero `__global__` and 138 cuBLAS/cuBLASLt
//! calls. It was never a kernel file: it is host logic — a router on the
//! weight's dtype, six growable device scratches, a dequantized-weight cache
//! and four cuBLASLt matmul recipes. The four pure-cuBLAS bodies became
//! [`super`]'s dense entry points; this module is the QUANTIZED half, which
//! is everything the router reaches that is not a plain bf16 GEMM.
//!
//! It exists for one reason above all others: **`gemm.cpp` was the last C++
//! caller of four migrated `quant::dequant_*` rows.** C++ composing with C++
//! is a call no Rust dispatch can intercept, so those four rows could not be
//! routed to the JIT while this file's arms lived in the archive. Moving the
//! arms moved the calls:
//!
//! | row | fired from |
//! |---|---|
//! | `quant::dequant_fp8_e4m3_to_bf16` | [`dequant_then_bf16`], PerTensor |
//! | `quant::dequant_fp8_e4m3_to_bf16_per_channel` | [`dequant_then_bf16`] |
//! | `quant::dequant_fp8_e4m3_to_bf16_per_group` | [`dequant_then_bf16`] |
//! | `quant::dequant_mxfp4_to_bf16` | [`act_x_w`]'s MXFP4 arm |
//!
//! # Why it is in THIS crate, which it was not
//!
//! It was `driver-cuda`'s `bind/quant_gemm.rs`, and the three entry points
//! below were `driver-cuda`'s `bind/service.rs`. Both came down under
//! `.wiki/kernel-x/refactor-plan.md` §6.3, and the argument is the one the
//! FA2 host half came down on: **every callee this file has is in this
//! crate.** Nine fires reach `crate::quant`, five bf16 GEMMs reach
//! [`super::dense`], and the `norm::residual_add_bf16` at the end of the INT8
//! arm is `crate::norm`'s — fifteen calls, every one of them downward. It
//! reached UP for exactly nothing, which is what makes the boundary it sat on
//! the wrong way round rather than merely inconvenient.
//!
//! **What the move found is a duplicate.** `get_device_properties` — the
//! `#[cfg]` pair that names `cudaGetDeviceProperties_v2` under CUDA 12 and
//! `cudaGetDeviceProperties` under 13 — was written here and, byte for byte,
//! in [`super::dense`]. Neither copy could see the other, because a crate
//! boundary ran between them. One survives, as `jit::device::properties`.
//!
//! **What the move did NOT merge is two cuBLASLt contexts, and that is
//! deliberate.** [`LtCtx`] here and `dense`'s `Bf16LtCtx` are both per-device
//! `cublasLtCreate` + workspace singletons, and they are now two of them in
//! one directory: 32 MiB here (`gemm.cpp:50`'s `kDefaultLtWorkspaceBytes`)
//! and 64 MiB there (`:92`'s `cublaslt_bf16_workspace_bytes()`). The archive
//! had both, at `:1338` and `:130`, with those two sizes — so a merge is a
//! change to what the tuner and the router each get to work in, not a
//! deletion of a copy the boundary caused. It is now VISIBLE, which is all a
//! descent owes it.
//!
//! # The shape it keeps
//!
//! Deliberately the C++'s own shape, function for function, so a reader with
//! the archive open can diff them. [`act_x_w`] is `gemm.cpp:1999`'s router
//! with its seven arms in the same order; [`LtCtx`] is `:1338`'s per-device
//! singleton; [`GrowScratch`] is `:1385`'s; [`DequantWeightCache`] is
//! `:1508`'s. The three entry points at the end are the three `gemm.hpp`
//! inlines that built a `WeightView` and called the router — they still do,
//! and [`WeightView`] below is that struct.
//!
//! # A refusal is reproduced, never softened
//!
//! Five bodies here decline, and each declines for a measured reason:
//!
//! * The INT4 arm **panics**. It is not "unimplemented": §46 measured that
//!   no representation in this driver can construct an INT4 weight, removed
//!   the marlin tree that served it, and left the throw as the record.
//! * [`blockwise_w8a8`] returns `false` four different ways, and `false`
//!   means *"use the dequant path"*, not *"fail"*.
//! * The INT8 arm falls back on `M % 4 | N % 4 | K % 4` and on a non-success
//!   `cublasGemmEx`, both to the same dequant path.
//! * [`GrowScratch::reserve`] panics rather than growing once sealed: a
//!   CUDA-graph reservation that grows underneath a captured graph is a
//!   dangling device pointer in the replay.
//! * Two capability LATCHES turn themselves off permanently on a zero
//!   heuristic count. See [`LtCtx::fp8_native_supported`] and
//!   [`LtCtx::fp8_block_supported`] — those two `returned == 0` branches are
//!   measurements and the comments on them are the measurement.
//!
//! **None of the five is a `Refusal`**, and that is the same answer
//! [`super::dense`] gives: the C++ threw and the shim's `catch` aborted, so a
//! value here would be a fallback where the archive had none. The three entry
//! points below return `Result` for one reason only — [`Ctx::cublas`] can
//! refuse on a build with no handle — which is exactly why
//! [`super::act_x_wt_bf16`] returns one.
//!
//! # Why nothing new links
//!
//! `cudarc`'s `fallback-dynamic-loading` resolves every cuBLAS and cuBLASLt
//! symbol with `dlopen` on first use, so a crate holding this still builds
//! with no CUDA toolkit installed. `cublaslt` is a binding-generation feature
//! of `cudarc`, not a link flag.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::jit::abi::bf16;
// `In` and `Out` for the one cross-family call this file makes. Nothing here
// is a `#[routine]` -- the three symbols these bodies serve are
// `driver_bound!` rows in `gemm/mod.rs`, with no column and no binder -- so
// the wrappers arrive as a CALLER's vocabulary rather than a launcher's.
use kernels::routine::{In, Out};

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasGetStream_v2,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};
use cudarc::cublaslt::sys as lt;

// ─────────────────────────────────────────────────────────────────────────
// The vocabulary the router switches on — the archive crate's `DType` and
// `QuantMeta::Kind`, as they cross the ABI.
// ─────────────────────────────────────────────────────────────────────────

/// `tensor.hpp`'s `DType`, by its integer value.
///
/// The three entry points receive `w_dtype` and `scale_dtype` as `i32`
/// because that is how the row states them and how the C++ inline passed
/// them. Only the six the router names are spelled; anything else reaches
/// [`unsupported`], which is the C++'s own final line.
pub mod dtype {
    /// `DType::BF16` — every activation and every output this router serves.
    pub const BF16: i32 = 0;
    /// `DType::FP16`.
    pub const FP16: i32 = 1;
    /// `DType::FP32`.
    pub const FP32: i32 = 2;
    /// `DType::INT8` — W8A8's weight.
    pub const INT8: i32 = 3;
    /// `DType::INT32`.
    pub const INT32: i32 = 4;
    /// `DType::INT64`.
    pub const INT64: i32 = 5;
    /// `DType::UINT8` — MXFP4's scale dtype, raw E8M0 bytes.
    pub const UINT8: i32 = 6;
    /// `DType::FP8_E4M3`.
    pub const FP8_E4M3: i32 = 7;
    /// `DType::FP8_E5M2`.
    pub const FP8_E5M2: i32 = 8;
    /// `DType::INT4_PACKED` — the arm that refuses.
    pub const INT4_PACKED: i32 = 9;
    /// `DType::MXFP4_PACKED`.
    pub const MXFP4_PACKED: i32 = 10;
    /// `DType::E8M0` — OCP Microscaling's exponent-only scale byte, `b`
    /// denoting `2^(b - 127)`. Only ever a block-scale companion, never a
    /// weight, which is why no arm names it.
    pub const E8M0: i32 = 11;
}

/// `tensor.hpp`'s `QuantMeta::Kind`, by its integer value.
///
/// **The numbers are load-bearing beyond the match arms**:
/// [`DequantWeightCache`]'s key carries `kind` as an `i32` exactly as the
/// C++ key did (`static_cast<int>(scale_kind)`), so two recipes that differ
/// only in granularity get different cache entries.
pub mod quant_kind {
    /// One scale for the whole tensor.
    pub const PER_TENSOR: i32 = 0;
    /// One scale per output channel — `[N]`.
    pub const PER_CHANNEL: i32 = 1;
    /// One scale per group along K, or per 2-D block for FP8.
    pub const PER_GROUP: i32 = 2;
}

/// `gemm.hpp`'s `WeightView` — the weight and everything needed to read it.
///
/// Built by the three entry points at the end of this file from their
/// statements' operands, exactly as the `gemm.hpp` inlines built it from
/// their arguments. Carried by value; it owns nothing.
#[derive(Clone, Copy, Debug)]
pub struct WeightView {
    /// Device pointer to the packed weight, `[N, K]` row-major.
    pub data: *const c_void,
    /// Its element type — one of [`dtype`].
    pub dtype: i32,
    /// Bytes actually behind `data`. Checked by
    /// [`validate_quant_weight_view`] against what `N x K` requires.
    pub nbytes: usize,
    /// Device pointer to the scale tensor.
    pub scale_data: *const c_void,
    /// The scale's element type. FP32 for FP8 and INT8, UINT8 for MXFP4.
    pub scale_dtype: i32,
    /// Elements behind `scale_data`, checked against the recipe.
    pub scale_numel: usize,
    /// One of [`quant_kind`].
    pub quant_kind: i32,
    /// Group extent along K when [`quant_kind::PER_GROUP`]; `0` otherwise.
    pub group_size: i32,
}

/// `gemm.cpp:1226` — `unsupported(api, act, w, y)`.
///
/// The router's final line. A dtype triple with no arm is a programming
/// error in the caller, not a runtime condition, and the C++ threw; the
/// shim's `catch` turned that into an abort. A panic here reaches the same
/// place by the same route.
fn unsupported(api: &str, act_dtype: i32, w_dtype: i32, y_dtype: i32) -> ! {
    panic!("ops::{api}: unsupported dtype combo (act={act_dtype}, w={w_dtype}, y={y_dtype})");
}

// ─────────────────────────────────────────────────────────────────────────
// Status checking
// ─────────────────────────────────────────────────────────────────────────

/// `gemm.cpp:1282` — `check_lt(status, expr)`.
fn check_lt(status: lt::cublasStatus_t, expr: &str) {
    assert!(
        status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLASLt error ({}): at {expr}",
        status as i32
    );
}

/// The classic-cuBLAS twin. [`super::dense`] has a third spelling of this
/// one predicate, with a different message; see this module's header for why
/// the descent left both standing rather than merging them.
fn check(status: cublasStatus_t, api: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "{api} failed with cuBLAS status {}",
        status as i32
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Device memory — `tensor.cpp`'s `allocate_device_memory` / `free_device_memory`
// ─────────────────────────────────────────────────────────────────────────

/// `tensor.cpp:allocate_device_memory(bytes, align)`, minus the allocator
/// indirection.
///
/// **`tensor.cpp` dies with this function.** Its six entry points were
/// `alloc_logging_enabled`, `allocate_device_memory`, `free_device_memory`,
/// `sample_memory_callback`, `set_device_memory_allocator` and
/// `set_device_tensor_memory_callback`; four had NO consumer anywhere in the
/// tree — including the whole allocator-binding API, whose entire purpose is
/// to let a host bind an allocator, with nothing binding one — and the other
/// two were called only from `gemm.cpp`. So the 256-byte alignment and the
/// bound-allocator hook both collapse to one `cudaMalloc`: `cudaMalloc`
/// already returns memory aligned to at least 256 bytes on every CUDA
/// device, which is why the C++'s `align` argument was 256 at all four call
/// sites and never read by the default allocator.
fn allocate_device_memory(bytes: usize) -> *mut c_void {
    use cudarc::runtime::sys::{cudaError, cudaMalloc};
    let mut raw: *mut c_void = std::ptr::null_mut();
    let code = unsafe { cudaMalloc(&mut raw, bytes.max(1)) };
    assert!(
        code == cudaError::cudaSuccess && !raw.is_null(),
        "cudaMalloc({bytes}) for a quantized-GEMM scratch failed with {code:?}"
    );
    raw
}

/// `tensor.cpp:free_device_memory(block)`.
///
/// The status is ignored, as it was there: a failed free on a scratch being
/// replaced has nowhere to report to, and the growth path has already
/// committed to the new block.
fn free_device_memory(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }
}

/// `CUDA_CHECK(cudaStreamSynchronize(s))`.
fn stream_synchronize(stream: *mut c_void) {
    use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
    let code = unsafe { cudaStreamSynchronize(stream.cast()) };
    assert!(code == cudaError::cudaSuccess, "cudaStreamSynchronize failed with {code:?}");
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1385` — GrowScratch
// ─────────────────────────────────────────────────────────────────────────

/// A device buffer that only ever grows, and refuses to grow once sealed.
///
/// Six of these live in [`LtCtx`]. Monotonic growth is what lets the
/// quantized arms allocate nothing per call in steady state: the first fire
/// at a given shape pays a `cudaMalloc`, every later one reads `block`.
///
/// `seal` is the CUDA-graph contract. A captured graph records the device
/// ADDRESS of every buffer a launch touches, so a growth after capture frees
/// the pointer the graph replays against. Rather than corrupt the replay,
/// [`Self::reserve`] panics with the shape it wanted — which is an operator
/// message ("increase the planner reserve or disable CUDA graphs"), not an
/// internal error.
#[derive(Debug)]
struct GrowScratch {
    ptr: *mut c_void,
    bytes: usize,
    sealed: bool,
    name: &'static str,
}

impl GrowScratch {
    const fn new(name: &'static str) -> Self {
        Self { ptr: std::ptr::null_mut(), bytes: 0, sealed: false, name }
    }

    fn reserve(&mut self, want: usize) {
        if want <= self.bytes {
            return;
        }
        assert!(
            !self.sealed,
            "{} attempted to grow after CUDA graph reservation: want {want} bytes, \
             have {} bytes. Increase the planner reserve or disable CUDA graphs.",
            self.name, self.bytes
        );
        free_device_memory(self.ptr);
        self.ptr = allocate_device_memory(want);
        self.bytes = want;
    }

    fn ensure(&mut self, want: usize) -> *mut c_void {
        self.reserve(want);
        self.ptr
    }
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1338` — LtCtx
// ─────────────────────────────────────────────────────────────────────────

/// 32 MiB — `gemm.cpp:50`'s `kDefaultLtWorkspaceBytes`.
const DEFAULT_LT_WORKSPACE_BYTES: usize = 32 << 20;

/// The cuBLASLt handle, its workspace, the capability probe and the six
/// scratches — one per device.
///
/// **Per device, not per process.** Every rank of a tensor-parallel group
/// runs in this one process with its own current device, and the members
/// here own device memory: a shared context lets one rank hand another
/// rank's device a pointer, which poisons the context and surfaces as an
/// illegal access in whatever runs next. The C++ said the same of
/// `Bf16LtCtx`, `Bf16LtPlanCache` and `DenseGemmTuner`.
///
/// The C++ had a second layer here — `g_runtime_quant_context`, a
/// thread-local override installed by `ScopedRuntimeQuantContext` so a
/// planner could pre-size and then seal the scratches before capture. It is
/// **not ported, because it had no consumer**: `RuntimeQuantContext`,
/// `ScopedRuntimeQuantContext`, `reserve_runtime_quant_scratch` and
/// `runtime_quant_scratch_bytes` were declared in `gemm.hpp` and called from
/// nowhere in the tree. [`GrowScratch::sealed`] is kept anyway, because the
/// day a planner does pre-size these it is the mechanism it needs and the
/// panic message is the whole design.
struct LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace: *mut c_void,
    workspace_bytes: usize,
    /// `0` = unqueried. `gemm.cpp:1374`.
    compute_capability_major: i32,
    /// Set from the device query: `major > 8 || (major == 8 && minor >= 9)`.
    ///
    /// sm_89 (Ada) is the first architecture with FP8 tensor cores, so
    /// everything below it takes the dequant path. **Also latched off at
    /// run time** — see [`fp8_e4m3_w_bf16_act`]'s `returned == 0` branch.
    fp8_native_supported: bool,
    /// Starts `true` and latches off the first time cuBLASLt answers zero
    /// algorithms for a block-scaled FP8 matmul. See [`blockwise_w8a8`].
    fp8_block_supported: bool,
    /// sm<89 FP8 → bf16 weight scratch. Also the MXFP4 arm's, and the INT8
    /// arm's `beta != 0` staging buffer.
    dequant: GrowScratch,
    /// `[M, K]` int8 quantised activation.
    int8_act: GrowScratch,
    /// `[M]` fp32 `act_scale_inv`.
    int8_act_scale: GrowScratch,
    /// `[M, N]` int32 W8A8 accumulator.
    int32_acc: GrowScratch,
    /// `[M, K]` fp8 blockwise-quantised activation.
    fp8_act: GrowScratch,
    /// `[M, ceil(K/128)]` fp32 activation scales.
    fp8_act_scale: GrowScratch,
}

// SAFETY: every field is either a plain integer or a device handle/pointer
// whose owning device is fixed by the per-device keying below. The `Mutex`
// in `per_device` is what serialises access; nothing here is touched
// outside it.
unsafe impl Send for LtCtx {}

impl LtCtx {
    fn new() -> Self {
        Self {
            handle: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
            compute_capability_major: 0,
            fp8_native_supported: false,
            fp8_block_supported: true,
            dequant: GrowScratch::new("dequant"),
            int8_act: GrowScratch::new("int8_act"),
            int8_act_scale: GrowScratch::new("int8_act_scale"),
            int32_acc: GrowScratch::new("int32_acc"),
            fp8_act: GrowScratch::new("fp8_act"),
            fp8_act_scale: GrowScratch::new("fp8_act_scale"),
        }
    }

    /// `gemm.cpp:1353` — `ensure_init(ws_bytes)`.
    ///
    /// Idempotent in three independent parts: the handle, the workspace and
    /// the capability probe each guard on their own sentinel, because the
    /// router calls this on every quantized fire.
    fn ensure_init(&mut self) {
        if self.handle.is_null() {
            let mut h: lt::cublasLtHandle_t = std::ptr::null_mut();
            check_lt(unsafe { lt::cublasLtCreate(&mut h) }, "cublasLtCreate");
            self.handle = h;
        }
        if self.workspace.is_null() {
            self.workspace = allocate_device_memory(DEFAULT_LT_WORKSPACE_BYTES);
            self.workspace_bytes = DEFAULT_LT_WORKSPACE_BYTES;
        }
        if self.compute_capability_major == 0 {
            use cudarc::runtime::sys::{cudaError, cudaGetDevice};
            let mut dev: i32 = 0;
            if unsafe { cudaGetDevice(&mut dev) } == cudaError::cudaSuccess
                && let Some(prop) = crate::jit::device::properties(dev)
            {
                self.compute_capability_major = prop.major;
                // sm_89 (Ada) is the floor for FP8 tensor cores. Hopper
                // is 9.0 and Blackwell 10.0, both `major > 8`.
                self.fp8_native_supported =
                    prop.major > 8 || (prop.major == 8 && prop.minor >= 9);
            }
        }
    }
}

// `get_device_properties` STOOD HERE. It is `jit::device::properties`, one
// copy where the crate boundary had allowed two -- see that function's own
// note, and this module's header.

/// The per-device map. `cudaGetDevice` keys it, exactly as
/// `gemm.cpp:129`'s `per_device_singleton<T>()` did.
fn with_lt_ctx<R>(f: impl FnOnce(&mut LtCtx) -> R) -> R {
    static CTXS: OnceLock<Mutex<HashMap<i32, LtCtx>>> = OnceLock::new();
    let mut dev: i32 = 0;
    let _ = unsafe { cudarc::runtime::sys::cudaGetDevice(&mut dev) };
    let mut map = CTXS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("quantized-GEMM context mutex poisoned");
    let ctx = map.entry(dev).or_insert_with(LtCtx::new);
    ctx.ensure_init();
    f(ctx)
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1508` — DequantWeightCache
// ─────────────────────────────────────────────────────────────────────────

/// The full recipe, not the pointer.
///
/// **A pointer-only key is wrong and this is measured**: sub-slices of one
/// tensor share a base pointer — DeepSeek-V4's per-group `wo_a` starts group
/// zero at the base — so a pointer-keyed cache hands a caller a buffer that
/// was expanded for a different shape or a different scale block. The key
/// carries the weight, the scale, `N`, `K`, the group extent and the
/// granularity, which is everything the expansion depends on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DequantKey {
    weight: usize,
    scale: usize,
    n: i32,
    k: i32,
    group: i32,
    kind: i32,
}

/// Cached bf16 expansions of block-quantized FP8 weights.
///
/// Block-quantized FP8 (DeepSeek `weight_block_size = [128, 128]`) has no
/// native cuBLASLt path on this platform, so every GEMM re-expands the
/// weight to bf16. **That costs 5x the weight bandwidth of the matmul and
/// dominates decode.** Weights are immutable, so the expansion is cached.
///
/// Per device for [`LtCtx`]'s reason: the entries own device memory, and an
/// eviction from another rank's map would `cudaFree` a pointer belonging to
/// a device that is not current.
struct DequantWeightCache {
    entries: HashMap<DequantKey, (*mut c_void, usize)>,
    used: usize,
    /// `0` disables the cache entirely — every `get` answers `None` and the
    /// caller falls back to [`LtCtx::dequant`].
    budget: usize,
    /// A private, non-blocking stream used to fill a fresh entry while the
    /// caller's stream is capturing. `None` until first needed.
    fill_stream: Option<*mut c_void>,
}

// SAFETY: as `LtCtx` — device handles behind the per-device `Mutex`.
unsafe impl Send for DequantWeightCache {}

impl DequantWeightCache {
    fn new() -> Self {
        // Sized on first use, i.e. AFTER the KV arena is sized, so "free"
        // here is real headroom rather than the whole card. `min(free / 4,
        // 16 GiB)` — a quarter of what is left, capped.
        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        let budget =
            if unsafe { cudarc::runtime::sys::cudaMemGetInfo(&mut free_bytes, &mut total_bytes) }
                == cudarc::runtime::sys::cudaError::cudaSuccess
            {
                (free_bytes / 4).min(16usize << 30)
            } else {
                0
            };
        Self { entries: HashMap::new(), used: 0, budget, fill_stream: None }
    }

    /// Returns the entry's device pointer and whether it still needs
    /// filling, or `None` when the cache declines (disabled, or full).
    ///
    /// Declining is not an error: the caller uses [`LtCtx::dequant`] and
    /// re-expands every call, which is what the archive did before the cache
    /// existed.
    fn get(&mut self, key: DequantKey, bytes: usize) -> Option<(*mut c_void, bool)> {
        if self.budget == 0 {
            return None;
        }
        if let Some(&(ptr, _)) = self.entries.get(&key) {
            return Some((ptr, false));
        }
        if self.used + bytes > self.budget {
            return None;
        }
        let ptr = allocate_device_memory(bytes);
        self.entries.insert(key, (ptr, bytes));
        self.used += bytes;
        Some((ptr, true))
    }

    /// A private non-blocking stream for capture-safe fills.
    fn fill_stream(&mut self) -> *mut c_void {
        if let Some(s) = self.fill_stream {
            return s;
        }
        use cudarc::runtime::sys::{cudaStreamCreateWithFlags, cudaStreamNonBlocking};
        let mut s: cudarc::runtime::sys::cudaStream_t = std::ptr::null_mut();
        let _ = unsafe { cudaStreamCreateWithFlags(&mut s, cudaStreamNonBlocking) };
        let raw: *mut c_void = s.cast();
        self.fill_stream = Some(raw);
        raw
    }
}

fn with_dequant_cache<R>(f: impl FnOnce(&mut DequantWeightCache) -> R) -> R {
    static CACHES: OnceLock<Mutex<HashMap<i32, DequantWeightCache>>> = OnceLock::new();
    let mut dev: i32 = 0;
    let _ = unsafe { cudarc::runtime::sys::cudaGetDevice(&mut dev) };
    let mut map = CACHES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("dequant weight cache mutex poisoned");
    f(map.entry(dev).or_insert_with(DequantWeightCache::new))
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1231` — validate_quant_weight_view
// ─────────────────────────────────────────────────────────────────────────

/// Bytes per element of `dtype`. `tensor.hpp`'s `dtype_bytes`, restricted to
/// the non-nibble-packed types — every quantized scalar type is one byte per
/// logical element by that header's own convention, which is why the default
/// arm is 1 rather than a panic.
const fn dtype_bytes(dtype: i32) -> usize {
    match dtype {
        dtype::INT64 => 8,
        dtype::FP32 | dtype::INT32 => 4,
        dtype::BF16 | dtype::FP16 => 2,
        _ => 1,
    }
}

/// `gemm.cpp:1231`. Every check and every arithmetic identity preserved.
///
/// The scale-count arithmetic is the part worth reading twice, because one
/// branch of it is a real checkpoint layout and not a generalisation:
/// PerTensor is one scale, PerChannel is `N`, PerGroup is `N *
/// ceil(K/gs)` — **except** for FP8, where DeepSeek's 2-D block scaling
/// makes it `ceil(N/gs) * ceil(K/gs)`. Getting that wrong does not fail
/// here; it fails as a silent under-read inside the dequant kernel.
pub fn validate_quant_weight_view(api: &str, w: &WeightView, n: i32, k: i32) {
    assert!(!w.data.is_null(), "{api}: quant weight data is null");
    assert!(!w.scale_data.is_null(), "{api}: quant scale data is null");

    let is_nibble_packed = w.dtype == dtype::INT4_PACKED || w.dtype == dtype::MXFP4_PACKED;
    let nk = (n as usize) * (k as usize);
    let expected_weight_bytes =
        if is_nibble_packed { (nk + 1) / 2 } else { nk * dtype_bytes(w.dtype) };
    assert!(
        w.nbytes >= expected_weight_bytes,
        "{api}: quant weight buffer is smaller than GEMM shape requires; have {} bytes, \
         need {expected_weight_bytes} bytes for N={n} K={k}",
        w.nbytes
    );

    let mut expected_scales: usize = 1;
    if w.quant_kind == quant_kind::PER_CHANNEL {
        expected_scales = n as usize;
    } else if w.quant_kind == quant_kind::PER_GROUP && w.group_size > 0 {
        let gs = w.group_size;
        expected_scales = if w.dtype == dtype::FP8_E4M3 {
            (((n + gs - 1) / gs) as usize) * (((k + gs - 1) / gs) as usize)
        } else {
            (n as usize) * (((k + gs - 1) / gs) as usize)
        };
    }
    assert!(
        w.scale_numel >= expected_scales,
        "{api}: quant scale tensor is smaller than GEMM shape requires; have {} values, \
         need {expected_scales}",
        w.scale_numel
    );
}

// ─────────────────────────────────────────────────────────────────────────
// cuBLASLt RAII — `gemm.cpp:1290-1326`
// ─────────────────────────────────────────────────────────────────────────

struct LtMatmulDesc(lt::cublasLtMatmulDesc_t);

impl LtMatmulDesc {
    fn new(compute: lt::cublasComputeType_t, scale: lt::cudaDataType) -> Self {
        let mut d: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check_lt(
            unsafe { lt::cublasLtMatmulDescCreate(&mut d, compute, scale) },
            "cublasLtMatmulDescCreate",
        );
        Self(d)
    }

    fn set<T>(&self, attr: lt::cublasLtMatmulDescAttributes_t, value: &T) {
        check_lt(
            unsafe {
                lt::cublasLtMatmulDescSetAttribute(
                    self.0,
                    attr,
                    (value as *const T).cast(),
                    std::mem::size_of::<T>(),
                )
            },
            "cublasLtMatmulDescSetAttribute",
        );
    }
}

impl Drop for LtMatmulDesc {
    fn drop(&mut self) {
        let _ = unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
    }
}

struct LtMatrixLayout(lt::cublasLtMatrixLayout_t);

impl LtMatrixLayout {
    fn new(dtype: lt::cudaDataType, rows: u64, cols: u64, ld: i64) -> Self {
        let mut d: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(
            unsafe { lt::cublasLtMatrixLayoutCreate(&mut d, dtype, rows, cols, ld) },
            "cublasLtMatrixLayoutCreate",
        );
        Self(d)
    }
}

impl Drop for LtMatrixLayout {
    fn drop(&mut self) {
        let _ = unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
    }
}

struct LtMatmulPref(lt::cublasLtMatmulPreference_t);

impl LtMatmulPref {
    fn new(workspace_bytes: usize) -> Self {
        let mut d: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check_lt(
            unsafe { lt::cublasLtMatmulPreferenceCreate(&mut d) },
            "cublasLtMatmulPreferenceCreate",
        );
        let ws = workspace_bytes;
        check_lt(
            unsafe {
                lt::cublasLtMatmulPreferenceSetAttribute(
                    d,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&ws as *const usize).cast(),
                    std::mem::size_of::<usize>(),
                )
            },
            "cublasLtMatmulPreferenceSetAttribute",
        );
        Self(d)
    }
}

impl Drop for LtMatmulPref {
    fn drop(&mut self) {
        let _ = unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
    }
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1629` — the dequant-then-bf16 fallbacks
// ─────────────────────────────────────────────────────────────────────────

/// One bf16 GEMM through the dense autotuner.
///
/// `gemm::act_x_wt_bf16`'s body — the archive's `gemm_bf16_impl` — is
/// [`super::dense::act_x_wt_bf16`], and this is a direct call to it.
///
/// # This function is a boundary's last shadow
///
/// It has had three spellings, and each one names where the callee was
/// living at the time: `ffi::pie_k_gemm_act_x_wt_bf16` while the autotuner
/// was C++, `crate::fire::gemm::act_x_wt_bf16` while it was `driver-cuda`'s
/// re-export, and now a sibling module in this directory. Nothing about the
/// four call sites ever changed. **A one-line forwarder whose whole content
/// is a path is what a crate boundary looks like from the inside**, and this
/// one is kept only because "one bf16 GEMM through the dense autotuner" is
/// what the four arms mean and `super::dense::act_x_wt_bf16` is not.
///
/// The argument the middle spelling carried — that reaching the body through
/// the C shim *"remains correct and is not a regression"*, because what had
/// held `gemm_bf16_impl` in C++ was a `gemv_bf16` returning `bool` for *"I
/// did not launch"* — is void and stays recorded, because it is the reason
/// the file moved rather than a reason it should not have. The standard was
/// never *"C++ that no Rust dispatch can intercept"*; it is **no `.cpp`
/// anywhere**, and `gemm.cpp` held zero `__global__`, so what condemned it
/// was that it held a host program and not that it held kernels. The `gemv`
/// arm inside the callee is `matches!(gemv::gemv_bf16(..), Gemv::Launched)`
/// in the same short-circuiting position `gemm.cpp:560` and `:978` put it in.
unsafe fn gemm_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    // SAFETY: the caller's obligation -- the three matrices address live
    // device memory of the extents `m`, `n` and `k` describe.
    unsafe {
        super::dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
}

/// `gemm.cpp:1629` — `gemm_fp8_dequant_then_bf16_fallback`.
///
/// Materialises a bf16 copy of the FP8 weight and runs the classic bf16
/// GEMM. Strictly slower than plain bf16 in steady state — one extra memory
/// pass per layer per fire — but correct, and on sm_89+ the native path
/// takes over automatically.
///
/// # The capture rule this body encodes
///
/// A fresh cache entry must be filled FOR REAL, right now. Under CUDA-graph
/// capture the caller's stream only *records* work, so a dequant enqueued
/// there leaves the buffer unwritten until the first replay while the entry
/// already reads as filled — and every non-graph caller would then multiply
/// by garbage. So a fresh entry is filled on the cache's private stream
/// (capture is Relaxed, so off-stream work and a sync on it are legal) and
/// the graph records just the matmul, which is the entire point of caching.
#[allow(clippy::too_many_arguments)]
unsafe fn dequant_then_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w_fp8: *const c_void,
    w_scale_fp32_dev: *const c_void,
    scale_kind: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    stream: *mut c_void,
    group_size: i32,
) {
    /// `cudaStreamIsCapturing(stream) != cudaStreamCaptureStatusNone`.
    ///
    /// A failed query clears the sticky error and answers "not capturing", which
    /// is what `gemm.cpp:1653` did — the query failing is not a reason to change
    /// where the fill runs.
    fn stream_is_capturing(stream: *mut c_void) -> bool {
    use cudarc::runtime::sys::{cudaError, cudaStreamCaptureStatus, cudaStreamIsCapturing};
    let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    let code = unsafe { cudaStreamIsCapturing(stream.cast(), &mut status) };
    if code != cudaError::cudaSuccess {
    let _ = unsafe { cudarc::runtime::sys::cudaGetLastError() };
    return false;
    }
    status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone
    }

    let weight_elems = (n as usize) * (k as usize);

    let cached = with_dequant_cache(|c| {
        c.get(
            DequantKey {
                weight: w_fp8 as usize,
                scale: w_scale_fp32_dev as usize,
                n,
                k,
                group: group_size,
                kind: scale_kind,
            },
            weight_elems * 2,
        )
    });

    let (bf16_w, fill_stream) = match cached {
        Some((ptr, false)) => {
            // Already expanded for this exact recipe. The whole point.
            unsafe { gemm_bf16(handle, act, ptr, y, m, n, k, beta) };
            return;
        }
        Some((ptr, true)) => {
            let fs = if stream_is_capturing(stream) {
                with_dequant_cache(DequantWeightCache::fill_stream)
            } else {
                stream
            };
            (ptr, fs)
        }
        // The cache declined — re-expand into the growable scratch, on the
        // caller's stream, every call.
        None => (with_lt_ctx(|ctx| ctx.dequant.ensure(weight_elems * 2)), stream),
    };

    if scale_kind == quant_kind::PER_GROUP && group_size > 0 {
        // `quant::dequant_fp8_e4m3_to_bf16_per_group`, `LaunchRule::RouteRows`:
        // `(fp8_in, bf16_out, scale_dev, cols, group_size)`. The launcher's
        // `rows` became the rule's grid and its `scale_cols` is derived
        // device-side as `ceil(cols / group_size)`.
        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16_per_group", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to_bf16_per_group(
                &ctx,
                kernels::routine::In { ptr: w_fp8.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: bf16_w.cast(), rows: 0, width: 0 },
                kernels::routine::In { ptr: w_scale_fp32_dev.cast(), rows: 0, width: 0 },
                // A LAUNCHER CALLING A LAUNCHER, so the wrappers are written
                // out: this caller is the host program's, not a statement's,
                // and `n`/`k`/`group_size` are its own numbers rather than
                // anything a trace placed.
                <kernels::keys::Rows as kernels::keys::Fact>::env(n),
                k,
                kernels::routine::Param(group_size),
            )
        });
    } else if scale_kind == quant_kind::PER_CHANNEL {
        // `(fp8_in, bf16_out, scale_inv_dev, cols)` — `rows` is the grid.
        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16_per_channel", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to_bf16_per_channel(
                &ctx,
                kernels::routine::In { ptr: w_fp8.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: bf16_w.cast(), rows: 0, width: 0 },
                kernels::routine::In { ptr: w_scale_fp32_dev.cast(), rows: 0, width: 0 },
                kernels::routine::Param(n),
                kernels::routine::Param(k),
            )
        });
    } else {
        // PerTensor. The scale is a DEVICE scalar and the kernel takes it by
        // VALUE, so the host must read it back — one `cudaMemcpyAsync` and a
        // full stream sync before the launch can be issued. That sync is
        // why this arm is the slow one and why nothing calls it: no live row
        // constructs a PerTensor FP8 weight (`gemm::act_x_wt_tensor_scaled`
        // was deleted from the table when `ScaleLayout::PerTensor` was found
        // to have no constructor outside `dsl.rs`). It is ported anyway
        // because re-stating that row is an eight-line edit the day the
        // loader grows the granularity, and a router arm that silently
        // routed PerTensor somewhere else would be the §45.2 failure.
        let mut scale: f32 = 0.0;
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let code = unsafe {
            cudaMemcpyAsync(
                (&mut scale as *mut f32).cast(),
                w_scale_fp32_dev,
                std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                fill_stream.cast(),
            )
        };
        assert!(
            code == cudaError::cudaSuccess,
            "cudaMemcpyAsync of the PerTensor FP8 scale failed with {code:?}"
        );
        stream_synchronize(fill_stream);
        // `quant::dequant_fp8_e4m3_to_bf16`, `LaunchRule::Elementwise`:
        // `(fp8_in, bf16_out, scale, n)`.
        // The scale crosses BY VALUE as an `f32`, which is the one thing the
        // deleted row could not spell: `Ty::F32` with `Source::Slot(Kind::Param, 0)` is
        // ungenerable — `abi.rs:1119` turns a `Param` into an `i32` and
        // `cast_for` adds no conversion — so this fire was ALWAYS the live
        // path and the row's `ParamF32` operand was never reached. The host
        // program takes an `f32` parameter and the question does not arise.
        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to::<bf16>(
                &ctx,
                kernels::routine::In { ptr: w_fp8.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: bf16_w.cast(), rows: 0, width: 0 },
                // The host program's own scale, not a statement's param slot;
                // the wrapper is written out for the same reason the one
                // above is.
                kernels::routine::ParamF32(scale),
                kernels::routine::Param(n),
                kernels::routine::Param(k),
            )
        });
    }

    if fill_stream != stream {
        stream_synchronize(fill_stream);
    }
    unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
}

/// `gemm.cpp:1701` — `gemm_int8_dequant_then_bf16_fallback`.
///
/// No cache: INT8 weights are per-channel and the expansion is cheap
/// relative to the FP8 block case the cache was built for. The archive did
/// not cache it either.
#[allow(clippy::too_many_arguments)]
unsafe fn int8_dequant_then_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w_int8: *const c_void,
    w_scale_inv: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    stream: *mut c_void,
) {
    let weight_elems = (n as usize) * (k as usize);
    let bf16_w = with_lt_ctx(|ctx| ctx.dequant.ensure(weight_elems * 2));
    // `quant::dequant_int8_to_bf16_per_channel` — a JIT row already, fired
    // through `unit_of` like the FP8 three.
    empty_or_panic("quant::dequant_int8_to_bf16_per_channel", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::dequant_int8_to_bf16_per_channel(
            &ctx,
            kernels::routine::In { ptr: w_int8.cast(), rows: 0, width: 0 },
            kernels::routine::Out { ptr: bf16_w.cast(), rows: 0, width: 0 },
            kernels::routine::In { ptr: w_scale_inv.cast(), rows: 0, width: 0 },
            n,
            k,
        )
    });
    unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1748` — DeepSeek-style W8A8 block FP8 GEMM
// ─────────────────────────────────────────────────────────────────────────

/// `gemm.cpp:1748` — returns `true` if it ran the matmul, `false` to mean
/// **"use the dequant path"**.
///
/// The checkpoint stores `weight [N, K]` as FP8 E4M3 with one FP32 scale per
/// 128x128 weight tile (`quantization_config.weight_block_size = [128,
/// 128]`). The historical path dequantized the *entire* weight to bf16 on
/// every call; this one quantizes the ACTIVATION to FP8 per 128-element
/// token group instead and lets cuBLASLt consume both scale tensors
/// directly.
///
/// # Four refusals, each meaning "the other path"
///
/// 1. `!fp8_block_supported` — the latch, already off.
/// 2. `group_size != 128` — the block scales assume 128.
/// 3. `K % 128 != 0 || N % 16 != 0` — block scales assume a whole number of
///    128-wide groups along K, and the FP8 tensor-core path additionally
///    needs 16-byte-aligned leading dimensions.
/// 4. Zero algorithms from the heuristic — **and this one latches**, so
///    every later block-FP8 call skips the whole round trip. The heuristic
///    is not cheap and its answer does not change within a process.
#[allow(clippy::too_many_arguments)]
unsafe fn blockwise_w8a8(
    act: *const c_void,
    w_fp8: *const c_void,
    w_scale_fp32_dev: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    stream: *mut c_void,
    group_size: i32,
) -> bool {
    /// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F`. Same note as above.
    const SCALE_MODE_VEC128_32F: i32 = 4;

    /// `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F`, written as its raw value.
    ///
    /// The C++ set both scale modes as a `std::int32_t` and so does this. The
    /// enumerator is not in `cudarc`'s `cuda-12080` binding — it appears at
    /// `cuda-12090` — and the number is the ABI either way, which is exactly why
    /// the archive spelled it as an integer attribute rather than a typed one.
    const SCALE_MODE_BLK128X128_32F: i32 = 5;

    if !with_lt_ctx(|ctx| ctx.fp8_block_supported) {
        return false; // latched-off
    }
    if group_size != 128 {
        return false; // group_size
    }
    if k % 128 != 0 || n % 16 != 0 {
        return false; // shape
    }

    let k_blocks = k / 128;
    let (handle_lt, workspace, workspace_bytes, act_fp8, act_scale) = with_lt_ctx(|ctx| {
        let a = ctx.fp8_act.ensure((m as usize) * (k as usize));
        let s = ctx
            .fp8_act_scale
            .ensure((m as usize) * (k_blocks as usize) * std::mem::size_of::<f32>());
        (ctx.handle, ctx.workspace, ctx.workspace_bytes, a, s)
    });

    // `quant::quantize_bf16_to_fp8_e4m3_per_token_group` — a driver-owned
    // `Launch`, because `families::quant`'s row is `LaunchRule::Unstated`:
    // its grid is `(ceil(k / group_size), m)`, an axis that divides one
    // operand by another, and §10.5 refuses vocabulary grown for one kernel.
    // `fire/quant_int8.rs` states the rectangle beside the `<<<>>>` it came
    // from. This was `ffi::pie_k_quant_*` until `kernels/quant/` was deleted.
    empty_or_panic("quant::quantize_bf16_to_fp8_e4m3_per_token_group", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::quantize_bf16_to_fp8_e4m3_per_token_group(
            &ctx,
            kernels::routine::In { ptr: act.cast(), rows: 0, width: 0 },
            kernels::routine::Out { ptr: act_fp8.cast(), rows: 0, width: 0 },
            kernels::routine::Out { ptr: act_scale.cast(), rows: 0, width: 0 },
            m,
            k,
            128,
        )
    });

    let desc = LtMatmulDesc::new(
        lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        lt::cudaDataType::CUDA_R_32F,
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &(cublasOperation_t::CUBLAS_OP_T),
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &(cublasOperation_t::CUBLAS_OP_N),
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        &SCALE_MODE_BLK128X128_32F,
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        &SCALE_MODE_VEC128_32F,
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &w_scale_fp32_dev,
    );
    let act_scale_const: *const c_void = act_scale.cast_const();
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &act_scale_const,
    );

    let a_layout =
        LtMatrixLayout::new(lt::cudaDataType::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64);
    let b_layout =
        LtMatrixLayout::new(lt::cudaDataType::CUDA_R_8F_E4M3, k as u64, m as u64, k as i64);
    let d_layout = LtMatrixLayout::new(lt::cudaDataType::CUDA_R_16BF, n as u64, m as u64, n as i64);
    let pref = LtMatmulPref::new(workspace_bytes);

    let mut heur: lt::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut returned: i32 = 0;
    let hs = unsafe {
        lt::cublasLtMatmulAlgoGetHeuristic(
            handle_lt,
            desc.0,
            a_layout.0,
            b_layout.0,
            d_layout.0,
            d_layout.0,
            pref.0,
            1,
            &mut heur,
            &mut returned,
        )
    };
    if hs != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || returned == 0 {
        with_lt_ctx(|ctx| ctx.fp8_block_supported = false);
        return false; // no-algo
    }

    let alpha: f32 = 1.0;
    check_lt(
        unsafe {
            lt::cublasLtMatmul(
                handle_lt,
                desc.0,
                (&alpha as *const f32).cast(),
                w_fp8,
                a_layout.0,
                act_fp8.cast_const(),
                b_layout.0,
                (&beta as *const f32).cast(),
                y.cast_const(),
                d_layout.0,
                y,
                d_layout.0,
                &heur.algo,
                workspace,
                workspace_bytes,
                stream.cast(),
            )
        },
        "cublasLtMatmul[fp8 blockwise w8a8]",
    );
    true
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1817` — native FP8 x bf16
// ─────────────────────────────────────────────────────────────────────────

/// `gemm.cpp:1817` — `gemm_fp8_e4m3_w_bf16_act_impl`.
///
/// cuBLASLt supports mixed FP8(weight) x BF16(act) -> BF16(out) with FP32
/// accumulation and a scale pointer for the FP8 operand. Adapted from
/// FlashInfer's `include/flashinfer/gemm/bmm_fp8.cuh`.
///
/// # The row-major-as-col-major reinterpretation
///
/// The same one the bf16 path uses, and it is worth spelling out because
/// every layout below depends on it. We compute `D'[N, M] = op(A=W) *
/// op(B=act)` where
///
/// * `A` col-major view of row-major `W[N, K]` -> `[K, N]` ld=K, `OP_T` -> `[N, K]`
/// * `B` col-major view of row-major `act[M, K]` -> `[K, M]` ld=K, `OP_N` -> `[K, M]`
/// * `D` col-major view of row-major `y[M, N]` -> `[N, M]` ld=N
///
/// so cuBLASLt sees `m = N`, `n = M`, `k = K`.
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_e4m3_w_bf16_act(
    handle: *mut c_void,
    act: *const c_void,
    w_fp8: *const c_void,
    w_scale_fp32_dev: *const c_void,
    scale_kind: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    stream: *mut c_void,
    group_size: i32,
) {
    assert!(
        !w_scale_fp32_dev.is_null(),
        "act_x_w[FP8_E4M3]: scale pointer is null — weight_scale_inv must be attached to \
         the materialized WeightStore as an FP32 device tensor before calling FP8 GEMM"
    );

    if scale_kind == quant_kind::PER_GROUP
        && unsafe {
            blockwise_w8a8(act, w_fp8, w_scale_fp32_dev, y, m, n, k, beta, stream, group_size)
        }
    {
        return;
    }

    let (native, handle_lt, workspace, workspace_bytes) = with_lt_ctx(|ctx| {
        (ctx.fp8_native_supported, ctx.handle, ctx.workspace, ctx.workspace_bytes)
    });

    if !native || scale_kind == quant_kind::PER_CHANNEL || scale_kind == quant_kind::PER_GROUP {
        unsafe {
            dequant_then_bf16(
                handle,
                act,
                w_fp8,
                w_scale_fp32_dev,
                scale_kind,
                y,
                m,
                n,
                k,
                beta,
                stream,
                group_size,
            );
        }
        return;
    }

    let desc = LtMatmulDesc::new(
        lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        lt::cudaDataType::CUDA_R_32F,
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &(cublasOperation_t::CUBLAS_OP_T),
    );
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &(cublasOperation_t::CUBLAS_OP_N),
    );
    // FAST_ACCUM: fp32 accumulation with a reduced-precision inner loop. The
    // archive pinned it on and every FP8 parity check in this tree is
    // written against that arithmetic.
    let fast_accum: i8 = 1;
    desc.set(lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum);
    // cuBLASLt multiplies A by `*scale` BEFORE the matmul. mistral3 stores
    // `weight_scale_inv` such that `bf16 = fp8 * scale`, which matches this
    // contract exactly — the name says "inv" and the arithmetic does not.
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &w_scale_fp32_dev,
    );

    let a_layout =
        LtMatrixLayout::new(lt::cudaDataType::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64);
    let b_layout = LtMatrixLayout::new(lt::cudaDataType::CUDA_R_16BF, k as u64, m as u64, k as i64);
    let d_layout = LtMatrixLayout::new(lt::cudaDataType::CUDA_R_16BF, n as u64, m as u64, n as i64);
    let pref = LtMatmulPref::new(workspace_bytes);

    let mut heur: lt::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut returned: i32 = 0;
    check_lt(
        unsafe {
            lt::cublasLtMatmulAlgoGetHeuristic(
                handle_lt,
                desc.0,
                a_layout.0,
                b_layout.0,
                d_layout.0,
                d_layout.0,
                pref.0,
                1,
                &mut heur,
                &mut returned,
            )
        },
        "cublasLtMatmulAlgoGetHeuristic[fp8 x bf16]",
    );
    if returned == 0 {
        // LATCHED FALLBACK. The device advertised FP8 (sm_89+) and cuBLASLt
        // still had no algorithm for this shape, which on the machines that
        // hit it was every shape. Cache the negative so subsequent FP8 calls
        // skip the heuristic round-trip entirely — and note that the latch
        // is per device and per process, so a machine that answers zero once
        // takes the dequant path for the rest of its life.
        //
        // **This branch is the one §45.2 flags.** Note what it does NOT do:
        // it does not pass `group_size` on to the fallback. The C++ omitted
        // it too (the call at `:1888` has nine arguments where the one at
        // `:1849` has ten), and it is correct rather than a bug, because
        // this branch is only reachable when `scale_kind` is PerTensor: the
        // PerChannel and PerGroup cases were routed to the fallback three
        // lines above and never reach the heuristic. A PerTensor recipe has
        // no group extent. Passing `group_size` here would change the
        // `DequantWeightCache` key for a recipe that has none.
        unsafe {
            dequant_then_bf16(
                handle,
                act,
                w_fp8,
                w_scale_fp32_dev,
                scale_kind,
                y,
                m,
                n,
                k,
                beta,
                stream,
                0,
            );
        }
        return;
    }

    let alpha: f32 = 1.0;
    check_lt(
        unsafe {
            lt::cublasLtMatmul(
                handle_lt,
                desc.0,
                (&alpha as *const f32).cast(),
                w_fp8,
                a_layout.0,
                act,
                b_layout.0,
                (&beta as *const f32).cast(),
                y.cast_const(),
                d_layout.0,
                y,
                d_layout.0,
                &heur.algo,
                workspace,
                workspace_bytes,
                stream.cast(),
            )
        },
        "cublasLtMatmul[fp8 x bf16]",
    );
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1911` — W8A8 INT8
// ─────────────────────────────────────────────────────────────────────────

/// `gemm.cpp:1911` — `gemm_int8_w_bf16_act_impl`.
///
/// bf16 activation -> int8 (per token), int8 weight (per-channel scale
/// already attached), `cublasGemmEx` INT8 -> int32 accumulator, dequant to
/// bf16 via the per-row x per-col scale product.
///
/// **sm_80 has native INT8 tensor-core GEMM (`CUDA_R_8I` +
/// `CUBLAS_COMPUTE_32I`) at ~2x bf16 throughput, so this is the real Ampere
/// quant performance win** — FP8 on sm_80 is bf16-equivalent via the dequant
/// fallback, and that asymmetry is the whole reason this arm exists.
#[allow(clippy::too_many_arguments)]
unsafe fn int8_w_bf16_act(
    handle: *mut c_void,
    act_bf16: *const c_void,
    w_int8: *const c_void,
    w_scale_inv: *const c_void,
    y_bf16: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    stream: *mut c_void,
) {
    // The INT8 tensor-core path needs all three extents divisible by four.
    // Not a preference: `cublasGemmEx` refuses `CUDA_R_8I` otherwise, and
    // refusing inside a graph capture invalidates the capture.
    if m % 4 != 0 || n % 4 != 0 || k % 4 != 0 {
        unsafe {
            int8_dequant_then_bf16(
                handle,
                act_bf16,
                w_int8,
                w_scale_inv,
                y_bf16,
                m,
                n,
                k,
                beta,
                stream,
            );
        }
        return;
    }

    let act_int8_bytes = (m as usize) * (k as usize);
    let act_scale_bytes = (m as usize) * std::mem::size_of::<f32>();
    let acc_bytes = (m as usize) * (n as usize) * std::mem::size_of::<i32>();
    let (act_int8, act_scale, acc_int32) = with_lt_ctx(|ctx| {
        (
            ctx.int8_act.ensure(act_int8_bytes),
            ctx.int8_act_scale.ensure(act_scale_bytes),
            ctx.int32_acc.ensure(acc_bytes),
        )
    });

    // Stage 1: per-token activation quant. A JIT fire through
    // `LaunchRule::Rms`, which IS the launcher's grid — one block per row,
    // 256 wide, `(256 / 32) * 4` bytes of shared memory. The C++
    // `quantize_bf16_to_int8_per_token` was a forwarder onto
    // `quantize_bf16_to_int8_per_channel` and this calls that row directly,
    // which is what `table/driver_internal.rs` prescribed for it.
    empty_or_panic("quant::quantize_bf16_to_int8_per_channel", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::quantize_bf16_to_int8_per_channel(
            &ctx,
            kernels::routine::In { ptr: act_bf16.cast(), rows: 0, width: 0 },
            kernels::routine::Out { ptr: act_int8.cast(), rows: 0, width: 0 },
            kernels::routine::Out { ptr: act_scale.cast(), rows: 0, width: 0 },
            m,
            k,
        )
    });

    // Stage 2: `cublasGemmEx` INT8. Same row-major-as-col-major
    // reinterpretation as the bf16 path:
    //   `y_int32[m, n] = sum_k act_int8[m, k] * w_int8[n, k]`
    //   A = w_int8   [K, N] ld=K, OP_T -> [N, K]
    //   B = act_int8 [K, M] ld=K, OP_N -> [K, M]
    //   D = acc      [N, M] ld=N col-major = [M, N] row-major.
    //
    // `CUBLAS_GEMM_DEFAULT`, not `..._TENSOR_OP`: the integer path selects
    // its own tensor-core kernels and the TENSOR_OP hint is a float concept.
    let alpha: i32 = 1;
    let c_beta: i32 = 0;
    let status = unsafe {
        cublasGemmEx(
            handle.cast::<cublasContext>(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            (&alpha as *const i32).cast(),
            w_int8,
            cudaDataType::CUDA_R_8I,
            k,
            act_int8.cast_const(),
            cudaDataType::CUDA_R_8I,
            k,
            (&c_beta as *const i32).cast(),
            acc_int32,
            cudaDataType::CUDA_R_32I,
            n,
            cublasComputeType_t::CUBLAS_COMPUTE_32I,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        // A refusal, not an error: the INT8 kernel set is narrower than the
        // divisibility check above can predict, so a non-success status here
        // means "this shape has no INT8 kernel" and the answer is the
        // dequant path — the same one the divisibility check takes.
        unsafe {
            int8_dequant_then_bf16(
                handle,
                act_bf16,
                w_int8,
                w_scale_inv,
                y_bf16,
                m,
                n,
                k,
                beta,
                stream,
            );
        }
        return;
    }

    // Stage 3: dequant int32 -> bf16 with per-row x per-col scales.
    //   `y[m, n]  = acc[m, n] * act_scale_inv[m] * w_scale_inv[n]`  (beta = 0)
    //   `y[m, n] += acc[m, n] * act_scale_inv[m] * w_scale_inv[n]`  (beta != 0)
    //
    // For `beta != 0` (residual-add fusion) dequant into a scratch then
    // residual-add — the same trick marlin used. For `beta == 0` dequant
    // straight into `y_bf16`.
    if beta == 0.0 {
        empty_or_panic("quant::dequant_int32_w8a8_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::quant::dequant_int32_w8a8_to_bf16(
                &ctx,
                kernels::routine::In { ptr: acc_int32.cast::<i32>().cast_const(), rows: 0, width: 0 },
                kernels::routine::In { ptr: act_scale.cast::<f32>().cast_const(), rows: 0, width: 0 },
                kernels::routine::In { ptr: w_scale_inv.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: y_bf16.cast(), rows: 0, width: 0 },
                m,
                n,
            )
        });
    } else {
        let mn = (m as usize) * (n as usize);
        let dq_dst = with_lt_ctx(|ctx| ctx.dequant.ensure(mn * 2));
        empty_or_panic("quant::dequant_int32_w8a8_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::quant::dequant_int32_w8a8_to_bf16(
                &ctx,
                kernels::routine::In { ptr: acc_int32.cast::<i32>().cast_const(), rows: 0, width: 0 },
                kernels::routine::In { ptr: act_scale.cast::<f32>().cast_const(), rows: 0, width: 0 },
                kernels::routine::In { ptr: w_scale_inv.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: dq_dst.cast(), rows: 0, width: 0 },
                m,
                n,
            )
        });
        // `y += x`, through the same fn-world entry the dequant above uses.
        // The dynamic path resolved a `LaunchRule` from the row's sig, and a
        // crossed contract states none, so it refused `Geometry { Unstated }`
        // -- which `bind::jit::fire` reports only for `Unknown` and otherwise
        // swallows, making this a launch that silently did not happen.
        //
        // THE RECTANGLE THIS CALLER ALREADY HELD, HANDED OVER AS ONE THING.
        // The call read `residual_add(&ctx, y_bf16.cast(), dq_dst.cast(),
        // mn)`, where `mn` was `m * n` folded flat by this tuner and then
        // unfolded by nobody: two loose extents travelling beside the
        // pointers they describe, which is §1.3 exactly. `norm::residual_add`
        // takes `Out<0, _>` and `In<1, _>` now and multiplies them back out
        // itself (`norm.rs`'s `y.rows.saturating_mul(y.width)`), so the
        // flattening happens once, at the callee, on numbers that arrived
        // attached to the buffer.
        //
        // `mn` IS STILL COMPUTED AND STILL USED, four lines up, to size the
        // scratch -- `ctx.dequant.ensure(mn * 2)` is a BYTE count and not an
        // extent, which is why it did not travel into a region with the rest.
        //
        // The two regions are honest here in a way a hand-built region is not
        // always: `m` and `n` are this function's own parameters, the same
        // pair the dequant above was handed, and `dq_dst` was allocated from
        // their product. Nothing is invented to fill a field.
        empty_or_panic("norm::residual_add_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::norm::residual_add::<bf16>(
                &ctx,
                Out { ptr: y_bf16.cast::<bf16>(), rows: m, width: n },
                In { ptr: dq_dst.cast::<bf16>().cast_const(), rows: m, width: n },
            )
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.cpp:1999` — the router
// ─────────────────────────────────────────────────────────────────────────

/// `gemm.cpp:1999` — `act_x_w(handle, act, w, y, M, N, K, beta, act_dtype,
/// y_dtype)`.
///
/// **One router, three entry points**, exactly as the archive had it: the
/// `gemm.hpp` inlines `act_x_wt_channel_scaled`, `act_x_wt_grouped_scaled`
/// and `act_x_wt_mxfp4_marlin` each built a `WeightView` and called this,
/// and the three at the end of this file do the same. `Control::Switch { on:
/// "w_dtype" }` was the classification's word for it and it is literally true
/// of the program; each entry point merely pins the discriminant it can
/// produce.
///
/// The arms are in the archive's order and the order matters for reading it,
/// not for behaviour — the conditions are disjoint.
///
/// # The two arms no live row reaches
///
/// The `y_dtype == FP32` arm and the PerTensor FP8 path are both ported and
/// both currently unreachable, and they are ported for the same reason. Every
/// `gemm.hpp` inline defaulted `y_dtype` to BF16, so no caller ever asked for
/// fp32 output; and `gemm::act_x_wt_tensor_scaled` was struck from the table
/// when `ScaleLayout::PerTensor` was found to have no constructor outside
/// `dsl.rs`. Dropping either would be a silent narrowing of a router that
/// currently answers for them, which is the §45.2 failure exactly: *"porting
/// them unfaithfully is how you get 99.83% of the right answer."*
#[allow(clippy::too_many_arguments)]
pub unsafe fn act_x_w(
    handle: *mut c_void,
    act: *const c_void,
    w: WeightView,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    act_dtype: i32,
    y_dtype: i32,
) {
    /// `gemm.cpp:1055` — `gemm_bf16_to_fp32_impl`.
    ///
    /// One `cublasGemmEx`, bf16 in and fp32 out. **Not the same function as the
    /// already-ported `gemm_bf16_out_fp32_impl`** in [`super::dense`], despite
    /// the names: that one serves `gemm::act_x_wt_bf16_out_fp32`, a derived row;
    /// this one is only ever reached through [`act_x_w`]'s `y_dtype == FP32`
    /// arm, which nothing constructs. Ported because the arm is ported — see
    /// [`act_x_w`]'s note on the two unreachable arms.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_bf16_to_fp32(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
    ) {
    let alpha: f32 = 1.0;
    let status = unsafe {
    cublasGemmEx(
    handle.cast::<cublasContext>(),
    cublasOperation_t::CUBLAS_OP_T,
    cublasOperation_t::CUBLAS_OP_N,
    n,
    m,
    k,
    (&alpha as *const f32).cast(),
    w,
    cudaDataType::CUDA_R_16BF,
    k,
    act,
    cudaDataType::CUDA_R_16BF,
    k,
    (&beta as *const f32).cast(),
    y,
    cudaDataType::CUDA_R_32F,
    n,
    cublasComputeType_t::CUBLAS_COMPUTE_32F,
    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    };
    check(status, "cublasGemmEx[bf16 -> fp32]");
    }

    if act_dtype == dtype::BF16 && w.dtype == dtype::BF16 && y_dtype == dtype::BF16 {
        unsafe { gemm_bf16(handle, act, w.data, y, m, n, k, beta) };
        return;
    }
    if act_dtype == dtype::BF16 && w.dtype == dtype::BF16 && y_dtype == dtype::FP32 {
        unsafe { gemm_bf16_to_fp32(handle, act, w.data, y, m, n, k, beta) };
        return;
    }
    if act_dtype == dtype::BF16 && w.dtype == dtype::FP8_E4M3 && y_dtype == dtype::BF16 {
        // Pull the CUDA stream out of the classic cuBLAS handle so the FP8
        // path runs on the same stream as everything else this layer does.
        let stream = stream_of(handle);
        assert!(
            w.scale_dtype == dtype::FP32,
            "act_x_w[FP8_E4M3]: scale must be FP32 (got {})",
            w.scale_dtype
        );
        validate_quant_weight_view("act_x_w[FP8_E4M3]", &w, n, k);
        unsafe {
            fp8_e4m3_w_bf16_act(
                handle,
                act,
                w.data,
                w.scale_data,
                w.quant_kind,
                y,
                m,
                n,
                k,
                beta,
                stream,
                w.group_size,
            );
        }
        return;
    }
    if act_dtype == dtype::BF16 && w.dtype == dtype::INT8 && y_dtype == dtype::BF16 {
        let stream = stream_of(handle);
        assert!(
            w.scale_dtype == dtype::FP32,
            "act_x_w[INT8 W8A8]: scale must be FP32 (got {})",
            w.scale_dtype
        );
        assert!(
            w.quant_kind == quant_kind::PER_CHANNEL,
            "act_x_w[INT8 W8A8]: only PerChannel weight scale supported \
             (per-tensor / per-group not yet wired)"
        );
        validate_quant_weight_view("act_x_w[INT8 W8A8]", &w, n, k);
        unsafe {
            int8_w_bf16_act(handle, act, w.data, w.scale_data, y, m, n, k, beta, stream);
        }
        return;
    }
    if act_dtype == dtype::BF16 && w.dtype == dtype::INT4_PACKED && y_dtype == dtype::BF16 {
        // GPTQ/AWQ W4A16 was the vendored marlin tree's only caller in this
        // repository, and it was never reachable: `WeightRepr`
        // (model-compiler `dsl.rs:92`) has three variants — `Bf16`,
        // `Scaled`, `Mxfp4Marlin` — and none of them is INT4, so nothing can
        // construct a weight whose dtype arrives here as INT4_PACKED.
        // `QuantScheme::{GptqInt4, AwqInt4}` appear only inside `#[cfg(test)]`
        // bodies asserting that such a checkpoint is REFUSED, and
        // `loader/transcode_engine.hpp` — named by `kernels_manifest.hpp` as
        // the home of the marlin repack — does not exist. `third_party/marlin`
        // went in §46 with its two repack entry points, which had no callers
        // either.
        //
        // **This is a refusal, not a gap.** It panics rather than falling
        // back to any other arm, and substituting one would be answering a
        // question about INT4 with bf16 arithmetic.
        panic!(
            "act_x_w[INT4_PACKED]: GPTQ/AWQ W4A16 has no kernel here. The vendored marlin \
             tree that served it was removed once measurement showed no representation in \
             this driver can express an INT4 weight; see .wiki/driver/new-horizon.md §46."
        );
    }
    if act_dtype == dtype::BF16 && w.dtype == dtype::MXFP4_PACKED && y_dtype == dtype::BF16 {
        let stream = stream_of(handle);
        assert!(
            w.scale_dtype == dtype::UINT8,
            "act_x_w[MXFP4]: scale must be raw E8M0 bytes (got {})",
            w.scale_dtype
        );
        assert!(
            w.quant_kind == quant_kind::PER_GROUP && w.group_size == 32,
            "act_x_w[MXFP4]: expected per-group scales with group_size=32"
        );
        validate_quant_weight_view("act_x_w[MXFP4]", &w, n, k);

        // Dequant MXFP4 -> bf16 in a scratch buffer, then bf16 GEMM. Reuses
        // the shared `dequant` scratch, which auto-grows monotonically. Cost
        // is one extra weight read + write per call, acceptable for prefill
        // and small-batch decode — and the reason `act_x_wt_mxfp4_marlin`
        // still carries "marlin" in its name while having no marlin kernel
        // behind it (§46 removed the tree; the row name is the checkpoint
        // format's, not the kernel's).
        let weight_bf16_bytes = (n as usize) * (k as usize) * 2;
        let bf16_w = with_lt_ctx(|ctx| ctx.dequant.ensure(weight_bf16_bytes));
        // `quant::dequant_mxfp4_to_bf16`, `LaunchRule::RouteRows`:
        // `(packed, block_scale, out, in_dim)` — `out_dim` is the grid.
        empty_or_panic("quant::dequant_mxfp4_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::quant::dequant_mxfp4_to::<bf16>(
                &ctx,
                kernels::routine::In { ptr: w.data.cast(), rows: 0, width: 0 },
                kernels::routine::In { ptr: w.scale_data.cast(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: bf16_w.cast(), rows: 0, width: 0 },
                kernels::routine::Param(n),
                kernels::routine::Param(k),
            )
        });
        unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
        return;
    }
    unsupported("act_x_w", act_dtype, w.dtype, y_dtype);
}

/// An empty extent is a no-op; every other decline is a bug in this file.
///
/// Nine fires in this module reached `quant` through `super::jit::fire`,
/// which returns `()` and swallows its own refusals: a mistyped operand list
/// or a `Dims` that did not match the row was a silent no-launch. §5 step 5
/// took `quant` into fn-world, so all nine are now direct calls to
/// `crate::quant`, and the outcome is a `#[must_use] Fired`
/// with two arms that must not be collapsed:
///
///   * `Declined(Empty)` is the `.cu`'s own early return, moved into the host
///     program unchanged — `dequant_fp8.cu`'s `if (rows == 0 || cols == 0)`,
///     `quant_bf16_to_fp8.cu:71`, `:108` and `:128`. This router builds
///     rectangles from a `WeightView` and a live `m`/`n`/`k`, so an empty one
///     means the caller asked for nothing and getting nothing is right.
///   * Anything else means a host program refused an argument this file
///     built. `bind::jit::fire` had no way to say that and this file has no
///     way to handle it, so it aborts with the symbol.
///
/// Three of the nine changed shape and not just spelling. The two `Dims`
/// fields these calls filled — `rows` and `width` — were the `LaunchRule`'s
/// inputs, and the rules are gone: `route_rows`, `elementwise`, `rms` and the
/// two literal geometries live in `x::quant` beside the `<<<>>>` each came
/// from. So `n` and `k` are ordinary `i32` parameters now, passed in the
/// kernel's own order, and a swap is a type error at the two places where the
/// types differ rather than a transposed grid at run time.
// `rows: 0, width: 0` ON EVERY WRAPPER IN THIS FILE, AND IT IS A STATEMENT.
//
// These are HOST PROGRAMS: the arena fires them at load time, off a plan,
// with no trace and no statement placing anything. Every one of them used to
// spell `InSlot`/`OutSlot`, which carried an address and no extent at all --
// so the launchers they call have never read one, and could not have.
//
// F1 says a layout is 1:1 with the address and never absent, and that what
// looks like absence is *"a transport that dropped it"*. This is that
// transport, named. A zero here is not a guess at the rectangle; it is the
// record that this caller states none, and the moment one of these callees
// asks -- `In::all` refuses a zero width -- the refusal will name the caller
// that owes it, which is this file.

fn empty_or_panic(symbol: &str, fired: Result<(), crate::Refusal>) {
    if let Err(why) = fired
        && !matches!(why, crate::Refusal::Empty { .. })
    {
        panic!("{symbol} declined: {why:?}");
    }
}

/// `cublasGetStream(handle, &stream)`.
///
/// The router does this in four arms and the archive did it in four arms.
/// A quantized path that ran on a different stream from the layer around it
/// would be a race with no error, which is why the handle is the authority
/// rather than a passed-in stream.
fn stream_of(handle: *mut c_void) -> *mut c_void {
    let mut stream: cudarc::cublas::sys::cudaStream_t = std::ptr::null_mut();
    let status = unsafe { cublasGetStream_v2(handle.cast::<cublasContext>(), &raw mut stream) };
    check(status, "cublasGetStream");
    stream.cast()
}

// ─────────────────────────────────────────────────────────────────────────
// `gemm.hpp`'s three inlines — the entry points a trace symbol names
// ─────────────────────────────────────────────────────────────────────────
//
// These were `driver-cuda`'s `bind/service.rs`, and the reason they were
// there rather than here is written down at the module declaration that used
// to introduce this file: *"The three entry points stay in `service` because
// `execution::RUST_SERVED`'s spelling test reads that file."*
//
// **That test does not exist.** `every_rust_served_symbol_is_spelled_here`
// went with the archive crate and the classification table it read; three
// comments in `bind/service.rs` still cited it as a live requirement. So the
// only stated reason for the split was a test that had been deleted, and the
// split was costing a `WeightView` built one crate above the router that
// consumes it — eight fields assembled on one side of a boundary and read on
// the other, which is the shape §6.3 exists to close.
//
// Each is the `gemm.hpp` inline it replaces, unchanged: build a `WeightView`
// from the row's operands, then call the one router. `Control::Switch { on:
// "w_dtype" }` was the classification's word for it and it is literally true
// of the program; each entry point merely pins the discriminant it can
// produce.

/// `gemm::act_x_wt_channel_scaled` — `gemm.hpp:160`.
///
/// `y[M, N] = act[M, K] x W[N, K]^T`, with `W` quantized per output channel:
/// one scale per row of `W`. Serves both FP8 E4M3 and INT8 weights, and the
/// two take completely different routes inside — FP8 per-channel always
/// dequants to bf16 (cuBLASLt has no per-channel FP8 scale mode this tree
/// targets), INT8 per-channel runs the native `CUBLAS_COMPUTE_32I` path.
///
/// `channel_axis` is accepted and NOT read, exactly as the archive's inline
/// accepted and did not read it: the row states it because a per-channel
/// scale has an axis, and every weight this driver materialises is `[N, K]`
/// row-major with the channel on axis 0. A non-zero value is not refused
/// here because the C++ did not refuse it either — recording that is worth
/// more than inventing a check the archive never made.
///
/// # Safety
///
/// Every pointer must be a device address on the current device, `w` must
/// hold at least `N * K` elements of `w_dtype` and `scale` at least `N`
/// values; `y` must be writable for `M * N` bf16. Checked as far as
/// [`validate_quant_weight_view`] can check it, which is the byte counts.
#[allow(clippy::too_many_arguments)]
pub unsafe fn act_x_wt_channel_scaled(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    _channel_axis: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: quant_kind::PER_CHANNEL,
        group_size: 0,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        act_x_w(handle, act, view, y, m, n, k, beta, dtype::BF16, dtype::BF16);
    }
}

/// `gemm::act_x_wt_grouped_scaled` — `gemm.hpp:182`.
///
/// The same GEMM with `W` quantized per group along `K`. `group_size` is the
/// group extent, and for FP8 it is also the extent along `N`: DeepSeek's
/// `weight_block_size = [128, 128]` is a 2-D block scale, which is why
/// [`validate_quant_weight_view`] counts `ceil(N/gs) * ceil(K/gs)` scales for
/// FP8 and `N * ceil(K/gs)` for everything else.
///
/// **This is the entry point that reaches the block-scaled W8A8 path** — the
/// one arm here that does not dequant the weight, and the reason it exists is
/// a measurement: re-expanding a block-quantized FP8 weight to bf16 costs 5x
/// the weight bandwidth of the matmul and dominates decode.
///
/// # Safety
///
/// As [`act_x_wt_channel_scaled`], with the scale count above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn act_x_wt_grouped_scaled(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    group_size: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: quant_kind::PER_GROUP,
        group_size,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        act_x_w(handle, act, view, y, m, n, k, beta, dtype::BF16, dtype::BF16);
    }
}

/// `gemm::act_x_wt_mxfp4_marlin` — `gemm.hpp:206`.
///
/// MXFP4: four-bit elements packed two per byte with one raw E8M0 exponent
/// byte per 32-element block. The scale dtype is UINT8 and the group size is
/// 32, and both are asserted rather than defaulted.
///
/// **"marlin" in the name is the checkpoint format's, not a kernel's.** The
/// vendored marlin tree went in §54; this dequants to bf16 and runs the
/// classic GEMM, which is what the archive's arm did after the removal too.
///
/// # Safety
///
/// `w` must hold at least `ceil(N * K / 2)` bytes and `scale` at least
/// `N * ceil(K / 32)` bytes; `y` writable for `M * N` bf16.
#[allow(clippy::too_many_arguments)]
pub unsafe fn act_x_wt_mxfp4_marlin(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = WeightView {
        data: w,
        dtype: dtype::MXFP4_PACKED,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype: dtype::UINT8,
        scale_numel,
        quant_kind: quant_kind::PER_GROUP,
        group_size: 32,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        act_x_w(handle, act, view, y, m, n, k, beta, dtype::BF16, dtype::BF16);
    }
}
