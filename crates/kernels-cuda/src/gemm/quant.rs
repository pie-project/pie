use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use crate::jit::abi::bf16;

use kernels::routine::{In, InOut};

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmEx, cublasGetStream_v2,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};
use cudarc::cublaslt::sys as lt;

pub mod dtype {

    pub const BF16: i32 = 0;

    pub const FP16: i32 = 1;

    pub const FP32: i32 = 2;

    pub const INT8: i32 = 3;

    pub const INT32: i32 = 4;

    pub const INT64: i32 = 5;

    pub const UINT8: i32 = 6;

    pub const FP8_E4M3: i32 = 7;

    pub const FP8_E5M2: i32 = 8;

    pub const INT4_PACKED: i32 = 9;

    pub const MXFP4_PACKED: i32 = 10;

    pub const E8M0: i32 = 11;
}

pub mod quant_kind {

    pub const PER_TENSOR: i32 = 0;

    pub const PER_CHANNEL: i32 = 1;

    pub const PER_GROUP: i32 = 2;
}

#[derive(Clone, Copy, Debug)]
pub struct WeightView {
    pub data: *const c_void,
    pub dtype: i32,
    pub nbytes: usize,
    pub scale_data: *const c_void,
    pub scale_dtype: i32,
    pub scale_numel: usize,
    pub quant_kind: i32,
    pub group_size: i32,
}

fn unsupported(api: &str, act_dtype: i32, w_dtype: i32, y_dtype: i32) -> ! {
    panic!("ops::{api}: unsupported dtype combo (act={act_dtype}, w={w_dtype}, y={y_dtype})");
}

fn check_lt(status: lt::cublasStatus_t, expr: &str) {
    assert!(
        status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLASLt error ({}): at {expr}",
        status as i32
    );
}

fn check(status: cublasStatus_t, api: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "{api} failed with cuBLAS status {}",
        status as i32
    );
}

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

fn free_device_memory(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }
}

fn stream_synchronize(stream: *mut c_void) {
    use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
    let code = unsafe { cudaStreamSynchronize(stream.cast()) };
    assert!(
        code == cudaError::cudaSuccess,
        "cudaStreamSynchronize failed with {code:?}"
    );
}

#[derive(Debug)]
struct GrowScratch {
    ptr: *mut c_void,
    bytes: usize,
    sealed: bool,
    name: &'static str,
}

impl GrowScratch {
    const fn new(name: &'static str) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            sealed: false,
            name,
        }
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

const DEFAULT_LT_WORKSPACE_BYTES: usize = 32 << 20;

struct LtCtx {
    handle: lt::cublasLtHandle_t,
    workspace: *mut c_void,
    workspace_bytes: usize,
    compute_capability_major: i32,
    fp8_native_supported: bool,
    fp8_block_supported: bool,
    dequant: GrowScratch,
    int8_act: GrowScratch,
    int8_act_scale: GrowScratch,
    int32_acc: GrowScratch,
    fp8_act: GrowScratch,
    fp8_act_scale: GrowScratch,
}

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

                self.fp8_native_supported = prop.major > 8 || (prop.major == 8 && prop.minor >= 9);
            }
        }
    }
}

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DequantKey {
    weight: usize,
    scale: usize,
    n: i32,
    k: i32,
    group: i32,
    kind: i32,
}

struct DequantWeightCache {
    entries: HashMap<DequantKey, (*mut c_void, usize)>,
    used: usize,
    budget: usize,
    fill_stream: Option<*mut c_void>,
}

unsafe impl Send for DequantWeightCache {}

impl DequantWeightCache {
    fn new() -> Self {
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
        Self {
            entries: HashMap::new(),
            used: 0,
            budget,
            fill_stream: None,
        }
    }

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

const fn dtype_bytes(dtype: i32) -> usize {
    match dtype {
        dtype::INT64 => 8,
        dtype::FP32 | dtype::INT32 => 4,
        dtype::BF16 | dtype::FP16 => 2,
        _ => 1,
    }
}

pub fn validate_quant_weight_view(api: &str, w: &WeightView, n: i32, k: i32) {
    assert!(!w.data.is_null(), "{api}: quant weight data is null");
    assert!(!w.scale_data.is_null(), "{api}: quant scale data is null");

    let is_nibble_packed = w.dtype == dtype::INT4_PACKED || w.dtype == dtype::MXFP4_PACKED;
    let nk = (n as usize) * (k as usize);
    let expected_weight_bytes = if is_nibble_packed {
        (nk + 1) / 2
    } else {
        nk * dtype_bytes(w.dtype)
    };
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
    unsafe {
        super::dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
}

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

        None => (
            with_lt_ctx(|ctx| ctx.dequant.ensure(weight_elems * 2)),
            stream,
        ),
    };

    if scale_kind == quant_kind::PER_GROUP && group_size > 0 {
        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16_per_group", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to_bf16_per_group(
                &ctx,
                kernels::routine::In {
                    ptr: w_fp8.cast(),
                    rows: 0,
                    width: 0,
                },
                // The routine reads its column extent off this width, and the
                // caller's weight is `n` rows of `k` -- a zero here launched
                // nothing and left the dequant cache holding uninitialised
                // bf16.
                kernels::routine::Out {
                    ptr: bf16_w.cast(),
                    rows: n,
                    width: k,
                },
                kernels::routine::In {
                    ptr: w_scale_fp32_dev.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Const { v: group_size },
                kernels::routine::Const { v: n },
            )
        });
    } else if scale_kind == quant_kind::PER_CHANNEL {
        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16_per_channel", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to_bf16_per_channel(
                &ctx,
                kernels::routine::In {
                    ptr: w_fp8.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Out {
                    ptr: bf16_w.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::In {
                    ptr: w_scale_fp32_dev.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Const { v: n },
                kernels::routine::Const { v: k },
            )
        });
    } else {
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

        empty_or_panic("quant::dequant_fp8_e4m3_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(fill_stream);
            crate::quant::dequant_fp8_e4m3_to::<bf16>(
                &ctx,
                kernels::routine::In {
                    ptr: w_fp8.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Out {
                    ptr: bf16_w.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Const { v: scale },
                kernels::routine::Const { v: n },
                kernels::routine::Const { v: k },
            )
        });
    }

    if fill_stream != stream {
        stream_synchronize(fill_stream);
    }
    unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
}

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

    empty_or_panic("quant::dequant_int8_to_bf16_per_channel", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::dequant_int8_to_bf16_per_channel(
            &ctx,
            w_int8.cast(),
            bf16_w.cast(),
            w_scale_inv.cast(),
            n,
            k,
        )
    });
    unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
}

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
    const SCALE_MODE_VEC128_32F: i32 = 4;

    const SCALE_MODE_BLK128X128_32F: i32 = 5;

    if !with_lt_ctx(|ctx| ctx.fp8_block_supported) {
        return false;
    }
    if group_size != 128 {
        return false;
    }
    if k % 128 != 0 || n % 16 != 0 {
        return false;
    }

    let k_blocks = k / 128;
    let (handle_lt, workspace, workspace_bytes, act_fp8, act_scale) = with_lt_ctx(|ctx| {
        let a = ctx.fp8_act.ensure((m as usize) * (k as usize));
        let s = ctx
            .fp8_act_scale
            .ensure((m as usize) * (k_blocks as usize) * std::mem::size_of::<f32>());
        (ctx.handle, ctx.workspace, ctx.workspace_bytes, a, s)
    });

    empty_or_panic("quant::quantize_bf16_to_fp8_e4m3_per_token_group", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::quantize_bf16_to_fp8_e4m3_per_token_group(
            &ctx,
            act.cast(),
            act_fp8.cast(),
            act_scale.cast(),
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

    let a_layout = LtMatrixLayout::new(
        lt::cudaDataType::CUDA_R_8F_E4M3,
        k as u64,
        n as u64,
        k as i64,
    );
    let b_layout = LtMatrixLayout::new(
        lt::cudaDataType::CUDA_R_8F_E4M3,
        k as u64,
        m as u64,
        k as i64,
    );
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
        return false;
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
            blockwise_w8a8(
                act,
                w_fp8,
                w_scale_fp32_dev,
                y,
                m,
                n,
                k,
                beta,
                stream,
                group_size,
            )
        }
    {
        return;
    }

    let (native, handle_lt, workspace, workspace_bytes) = with_lt_ctx(|ctx| {
        (
            ctx.fp8_native_supported,
            ctx.handle,
            ctx.workspace,
            ctx.workspace_bytes,
        )
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

    let fast_accum: i8 = 1;
    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &fast_accum,
    );

    desc.set(
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &w_scale_fp32_dev,
    );

    let a_layout = LtMatrixLayout::new(
        lt::cudaDataType::CUDA_R_8F_E4M3,
        k as u64,
        n as u64,
        k as i64,
    );
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

    empty_or_panic("quant::quantize_bf16_to_int8_per_channel", unsafe {
        let ctx = crate::jit::Ctx::on(stream);
        crate::quant::quantize_bf16_to_int8_per_channel(
            &ctx,
            act_bf16.cast(),
            act_int8.cast(),
            act_scale.cast(),
            m,
            k,
        )
    });

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

    if beta == 0.0 {
        empty_or_panic("quant::dequant_int32_w8a8_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::quant::dequant_int32_w8a8_to_bf16(
                &ctx,
                acc_int32.cast::<i32>().cast_const(),
                act_scale.cast::<f32>().cast_const(),
                w_scale_inv.cast(),
                y_bf16.cast(),
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
                acc_int32.cast::<i32>().cast_const(),
                act_scale.cast::<f32>().cast_const(),
                w_scale_inv.cast(),
                dq_dst.cast(),
                m,
                n,
            )
        });

        empty_or_panic("norm::residual_add_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::norm::residual_add::<bf16>(
                &ctx,
                InOut {
                    ptr: y_bf16.cast::<bf16>(),
                    rows: m,
                    width: n,
                },
                In {
                    ptr: dq_dst.cast::<bf16>().cast_const(),
                    rows: m,
                    width: n,
                },
            )
        });
    }
}

/// The quantised dense GEMM, over any weight a [`WeightView`] describes.
///
/// NO CALLER, AS OF THE ROUTINE FOLD, AND THAT IS A MEASUREMENT RATHER THAN
/// AN OVERSIGHT. Three `#[routine]`s in `gemm.rs` were its entry points —
/// `act_x_wt_channel_scaled`, `act_x_wt_grouped_scaled`,
/// `act_x_wt_mxfp4_marlin` — and each was reached by the legacy driver by
/// SYMBOL and by nothing else, so all three went with the registry.
///
/// What would give it one is a POINT. `Gemm::matmul` declares
/// `w: Const<Self::Tensor<T>>` — one plane at the statement's element — and
/// every arm below takes a bank: codes, a scale plane, a quantisation kind
/// and a group size. That is the `Bank<R: Repr>` payload `.wiki/baker.md`
/// names and the floor does not carry yet; `moe.matmul_select` is the one
/// point that reaches a bank today, and it does it through a slot the MoE
/// family declares for itself. Until a dense point can say the same thing,
/// this file is the arithmetic with nothing to state it — kept whole,
/// because deleting a capability no declaration replaces is not a fold.
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

        let weight_bf16_bytes = (n as usize) * (k as usize) * 2;
        let bf16_w = with_lt_ctx(|ctx| ctx.dequant.ensure(weight_bf16_bytes));

        empty_or_panic("quant::dequant_mxfp4_to_bf16", unsafe {
            let ctx = crate::jit::Ctx::on(stream);
            crate::quant::dequant_mxfp4_to::<bf16>(
                &ctx,
                kernels::routine::In {
                    ptr: w.data.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::In {
                    ptr: w.scale_data.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Out {
                    ptr: bf16_w.cast(),
                    rows: 0,
                    width: 0,
                },
                kernels::routine::Const { v: n },
                kernels::routine::Const { v: k },
            )
        });
        unsafe { gemm_bf16(handle, act, bf16_w, y, m, n, k, beta) };
        return;
    }
    unsupported("act_x_w", act_dtype, w.dtype, y_dtype);
}

fn empty_or_panic(symbol: &str, fired: Result<(), crate::Refusal>) {
    if let Err(why) = fired
        && !matches!(why, crate::Refusal::Empty { .. })
    {
        panic!("{symbol} declined: {why:?}");
    }
}

fn stream_of(handle: *mut c_void) -> *mut c_void {
    let mut stream: cudarc::cublas::sys::cudaStream_t = std::ptr::null_mut();
    let status = unsafe { cublasGetStream_v2(handle.cast::<cublasContext>(), &raw mut stream) };
    check(status, "cublasGetStream");
    stream.cast()
}
