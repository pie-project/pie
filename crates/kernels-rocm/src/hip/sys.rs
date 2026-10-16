//! The HIP runtime and hiprtc entry points this crate calls, resolved at run
//! time from `libamdhip64` and `libhiprtc`. Nothing links against ROCm at
//! build time, so a build without a ROCm install still compiles; the first
//! call into [`api`] is where a missing install shows up.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::rt::{
    Event, Graph, GraphExec, GraphNode, HostFn, MemAccessDesc, MemAllocationProp, MemcpyAttributes,
    PointerAttributes,
};

pub type HipError = c_int;
pub type HiprtcResult = c_int;

pub type Stream = *mut c_void;
pub type Module = *mut c_void;
pub type Function = *mut c_void;
pub type Program = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIPRTC_SUCCESS: HiprtcResult = 0;

pub const ERROR_OUT_OF_MEMORY: HipError = 2;

pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub const STREAM_CAPTURE_STATUS_NONE: c_int = 0;

/// `hipDeviceAttributeMultiprocessorCount`.
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
/// `hipDeviceAttributeWarpSize`.
pub const ATTR_WARP_SIZE: c_int = 87;
/// `hipDeviceAttributeComputeCapabilityMajor`: the gfx major (9 on CDNA).
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 23;
/// `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`.
pub const ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 10002;
/// `hipDeviceAttributeSharedMemPerBlockOptin`.
pub const ATTR_SHARED_MEM_PER_BLOCK_OPTIN: c_int = 75;

/// `sizeof(hipDeviceProp_tR0600)` and the offset of its `gcnArchName`. The
/// R0600 layout is the versioned ABI HIP keeps stable across releases.
pub const DEVICE_PROP_BYTES: usize = 1472;
pub const DEVICE_PROP_GCN_ARCH_NAME: usize = 1160;

macro_rules! table {
    ($name:ident from $lib:literal { $( fn $f:ident ( $($a:ident : $t:ty),* $(,)? ) -> $r:ty; )+ }) => {
        #[allow(non_snake_case)]
        pub struct $name {
            _lib: libloading::Library,
            $( pub $f: unsafe extern "C" fn($($t),*) -> $r, )+
        }

        impl $name {
            fn open() -> Result<Self, String> {
                let lib = open_library($lib)?;
                // SAFETY: each symbol is resolved with the C signature ROCm
                // declares for it, and the `Library` that owns the code is kept
                // alive in `_lib` for as long as the pointers are.
                unsafe {
                    Ok(Self {
                        $( $f: *lib
                            .get::<unsafe extern "C" fn($($t),*) -> $r>(
                                concat!(stringify!($f), "\0").as_bytes(),
                            )
                            .map_err(|e| format!("`{}` has no `{}`: {e}", $lib, stringify!($f)))?, )+
                        _lib: lib,
                    })
                }
            }
        }
    };
}

table!(Hip from "libamdhip64.so" {
    fn hipInit(flags: c_uint) -> HipError;
    fn hipGetErrorString(code: HipError) -> *const c_char;
    fn hipGetDeviceCount(count: *mut c_int) -> HipError;
    fn hipGetLastError() -> HipError;
    fn hipSetDevice(device: c_int) -> HipError;
    fn hipGetDevice(device: *mut c_int) -> HipError;
    fn hipGetDevicePropertiesR0600(props: *mut c_void, device: c_int) -> HipError;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: Stream) -> HipError;
    fn hipStreamIsCapturing(stream: Stream, status: *mut c_int) -> HipError;
    fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> HipError;
    fn hipMalloc(ptr: *mut *mut c_void, bytes: usize) -> HipError;
    fn hipFree(ptr: *mut c_void) -> HipError;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> HipError;
    fn hipStreamCreate(stream: *mut Stream) -> HipError;
    fn hipStreamDestroy(stream: Stream) -> HipError;
    fn hipStreamSynchronize(stream: Stream) -> HipError;
    fn hipModuleLoadData(module: *mut Module, image: *const c_void) -> HipError;
    fn hipModuleUnload(module: Module) -> HipError;
    fn hipModuleGetFunction(function: *mut Function, module: Module, name: *const c_char) -> HipError;
    fn hipModuleLaunchKernel(
        function: Function,
        grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
        block_x: c_uint, block_y: c_uint, block_z: c_uint,
        shared_bytes: c_uint,
        stream: Stream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut c_int,
        function: Function,
        block_size: c_int,
        dynamic_shared_bytes: usize,
    ) -> HipError;
    fn hipModuleLaunchCooperativeKernel(
        function: Function,
        grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
        block_x: c_uint, block_y: c_uint, block_z: c_uint,
        shared_bytes: c_uint,
        stream: Stream,
        params: *mut *mut c_void,
    ) -> HipError;
    fn hipDeviceSynchronize() -> HipError;
    fn hipDeviceCanAccessPeer(can: *mut c_int, device: c_int, peer: c_int) -> HipError;
    fn hipDeviceGetP2PAttribute(value: *mut c_int, attr: c_int, src: c_int, dst: c_int) -> HipError;
    fn hipDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> HipError;
    fn hipMemset(dst: *mut c_void, value: c_int, bytes: usize) -> HipError;
    fn hipMemsetAsync(dst: *mut c_void, value: c_int, bytes: usize, stream: Stream) -> HipError;
    fn hipMemset2DAsync(dst: *mut c_void, pitch: usize, value: c_int, width: usize, height: usize, stream: Stream) -> HipError;
    fn hipMemcpyBatchAsync(dsts: *mut *mut c_void, srcs: *mut *mut c_void, sizes: *mut usize, count: usize, attrs: *mut MemcpyAttributes, attr_idxs: *mut usize, num_attrs: usize, fail_idx: *mut usize, stream: Stream) -> HipError;
    fn hipHostMalloc(ptr: *mut *mut c_void, bytes: usize, flags: c_uint) -> HipError;
    fn hipHostFree(ptr: *mut c_void) -> HipError;
    fn hipHostRegister(ptr: *mut c_void, bytes: usize, flags: c_uint) -> HipError;
    fn hipHostUnregister(ptr: *mut c_void) -> HipError;
    fn hipHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: c_uint) -> HipError;
    fn hipPointerGetAttributes(attrs: *mut PointerAttributes, ptr: *const c_void) -> HipError;
    fn hipStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> HipError;
    fn hipStreamWaitEvent(stream: Stream, event: Event, flags: c_uint) -> HipError;
    fn hipLaunchHostFunc(stream: Stream, func: HostFn, data: *mut c_void) -> HipError;
    fn hipStreamBeginCapture(stream: Stream, mode: c_int) -> HipError;
    fn hipStreamBeginCaptureToGraph(stream: Stream, graph: Graph, deps: *const GraphNode, edges: *const c_void, count: usize, mode: c_int) -> HipError;
    fn hipStreamEndCapture(stream: Stream, graph: *mut Graph) -> HipError;
    fn hipStreamGetCaptureInfo_v2(stream: Stream, status: *mut c_int, id: *mut u64, graph: *mut Graph, deps: *mut *const GraphNode, count: *mut usize) -> HipError;
    fn hipStreamUpdateCaptureDependencies(stream: Stream, deps: *mut GraphNode, count: usize, flags: c_uint) -> HipError;
    fn hipEventCreate(event: *mut Event) -> HipError;
    fn hipEventCreateWithFlags(event: *mut Event, flags: c_uint) -> HipError;
    fn hipEventDestroy(event: Event) -> HipError;
    fn hipEventRecord(event: Event, stream: Stream) -> HipError;
    fn hipEventSynchronize(event: Event) -> HipError;
    fn hipEventQuery(event: Event) -> HipError;
    fn hipEventElapsedTime(ms: *mut f32, start: Event, stop: Event) -> HipError;
    fn hipGraphInstantiateWithFlags(exec: *mut GraphExec, graph: Graph, flags: u64) -> HipError;
    fn hipGraphUpload(exec: GraphExec, stream: Stream) -> HipError;
    fn hipGraphLaunch(exec: GraphExec, stream: Stream) -> HipError;
    fn hipGraphExecDestroy(exec: GraphExec) -> HipError;
    fn hipGraphDestroy(graph: Graph) -> HipError;
    fn hipGraphDebugDotPrint(graph: Graph, path: *const c_char, flags: c_uint) -> HipError;
    fn hipGraphGetNodes(graph: Graph, nodes: *mut GraphNode, count: *mut usize) -> HipError;
    fn hipGraphGetEdges(graph: Graph, from: *mut GraphNode, to: *mut GraphNode, count: *mut usize) -> HipError;
    fn hipMemCreate(handle: *mut u64, bytes: usize, prop: *const MemAllocationProp, flags: u64) -> HipError;
    fn hipMemRelease(handle: u64) -> HipError;
    fn hipMemAddressReserve(ptr: *mut *mut c_void, bytes: usize, align: usize, addr: *mut c_void, flags: u64) -> HipError;
    fn hipMemAddressFree(ptr: *mut c_void, bytes: usize) -> HipError;
    fn hipMemMap(ptr: *mut c_void, bytes: usize, offset: usize, handle: u64, flags: u64) -> HipError;
    fn hipMemUnmap(ptr: *mut c_void, bytes: usize) -> HipError;
    fn hipMemSetAccess(ptr: *mut c_void, bytes: usize, desc: *const MemAccessDesc, count: usize) -> HipError;
    fn hipMemGetAllocationGranularity(granularity: *mut usize, prop: *const MemAllocationProp, option: c_int) -> HipError;
    fn hipFuncGetAttribute(value: *mut c_int, attr: c_int, function: Function) -> HipError;
});

table!(Hiprtc from "libhiprtc.so" {
    fn hiprtcGetErrorString(code: HiprtcResult) -> *const c_char;
    fn hiprtcVersion(major: *mut c_int, minor: *mut c_int) -> HiprtcResult;
    fn hiprtcCreateProgram(
        program: *mut Program,
        source: *const c_char,
        name: *const c_char,
        header_count: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcDestroyProgram(program: *mut Program) -> HiprtcResult;
    fn hiprtcAddNameExpression(program: Program, expression: *const c_char) -> HiprtcResult;
    fn hiprtcCompileProgram(program: Program, option_count: c_int, options: *const *const c_char) -> HiprtcResult;
    fn hiprtcGetLoweredName(program: Program, expression: *const c_char, lowered: *mut *const c_char) -> HiprtcResult;
    fn hiprtcGetProgramLogSize(program: Program, bytes: *mut usize) -> HiprtcResult;
    fn hiprtcGetProgramLog(program: Program, log: *mut c_char) -> HiprtcResult;
    fn hiprtcGetCodeSize(program: Program, bytes: *mut usize) -> HiprtcResult;
    fn hiprtcGetCode(program: Program, code: *mut c_char) -> HiprtcResult;
});

pub type Comm = *mut c_void;
pub type NcclResult = c_int;
pub type NcclDataType = c_int;
pub type NcclRedOp = c_int;

pub const NCCL_SUCCESS: NcclResult = 0;
pub const NCCL_SUM: NcclRedOp = 0;
pub const NCCL_FLOAT16: NcclDataType = 6;
pub const NCCL_FLOAT32: NcclDataType = 7;
pub const NCCL_BFLOAT16: NcclDataType = 9;

/// `ncclUniqueId`: 128 opaque bytes, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UniqueId {
    pub internal: [c_char; 128],
}

table!(Rccl from "librccl.so" {
    fn ncclGetErrorString(code: NcclResult) -> *const c_char;
    fn ncclGetUniqueId(id: *mut UniqueId) -> NcclResult;
    fn ncclCommInitRank(comm: *mut Comm, ranks: c_int, id: UniqueId, rank: c_int) -> NcclResult;
    fn ncclCommAbort(comm: Comm) -> NcclResult;
    fn ncclCommUserRank(comm: Comm, rank: *mut c_int) -> NcclResult;
    fn ncclGroupStart() -> NcclResult;
    fn ncclGroupEnd() -> NcclResult;
    fn ncclAllReduce(
        send: *const c_void, recv: *mut c_void, count: usize,
        dtype: NcclDataType, op: NcclRedOp, comm: Comm, stream: Stream,
    ) -> NcclResult;
    fn ncclAllGather(
        send: *const c_void, recv: *mut c_void, count: usize,
        dtype: NcclDataType, comm: Comm, stream: Stream,
    ) -> NcclResult;
    fn ncclReduceScatter(
        send: *const c_void, recv: *mut c_void, count: usize,
        dtype: NcclDataType, op: NcclRedOp, comm: Comm, stream: Stream,
    ) -> NcclResult;
});

/// RCCL, opened on first use and apart from [`api`]: a single-device
/// process never needs it, so a box without it still runs.
pub fn rccl() -> Result<&'static Rccl, String> {
    static RCCL: OnceLock<Result<Rccl, String>> = OnceLock::new();
    RCCL.get_or_init(Rccl::open).as_ref().map_err(Clone::clone)
}

pub type Blas = *mut c_void;
pub type BlasStatus = c_int;
pub type BlasOp = c_int;
pub type BlasType = c_int;

pub const BLAS_SUCCESS: BlasStatus = 0;
pub const BLAS_OP_N: BlasOp = 111;
pub const BLAS_OP_T: BlasOp = 112;
pub const BLAS_F16: BlasType = 150;
pub const BLAS_F32: BlasType = 151;
pub const BLAS_BF16: BlasType = 168;
/// `rocblas_gemm_algo_standard`.
pub const BLAS_ALGO_STANDARD: c_int = 0;

table!(Rocblas from "librocblas.so" {
    fn rocblas_create_handle(handle: *mut Blas) -> BlasStatus;
    fn rocblas_destroy_handle(handle: Blas) -> BlasStatus;
    fn rocblas_set_stream(handle: Blas, stream: Stream) -> BlasStatus;
    fn rocblas_status_to_string(status: BlasStatus) -> *const c_char;
    fn rocblas_gemm_ex(
        handle: Blas, trans_a: BlasOp, trans_b: BlasOp, m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, a_type: BlasType, lda: c_int,
        b: *const c_void, b_type: BlasType, ldb: c_int,
        beta: *const c_void,
        c: *const c_void, c_type: BlasType, ldc: c_int,
        d: *mut c_void, d_type: BlasType, ldd: c_int,
        compute_type: BlasType, algo: c_int, solution_index: i32, flags: u32,
    ) -> BlasStatus;
    fn rocblas_gemm_strided_batched_ex(
        handle: Blas, trans_a: BlasOp, trans_b: BlasOp, m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, a_type: BlasType, lda: c_int, stride_a: i64,
        b: *const c_void, b_type: BlasType, ldb: c_int, stride_b: i64,
        beta: *const c_void,
        c: *const c_void, c_type: BlasType, ldc: c_int, stride_c: i64,
        d: *mut c_void, d_type: BlasType, ldd: c_int, stride_d: i64,
        batch_count: c_int, compute_type: BlasType, algo: c_int, solution_index: i32, flags: u32,
    ) -> BlasStatus;
    fn rocblas_gemm_batched_ex(
        handle: Blas, trans_a: BlasOp, trans_b: BlasOp, m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, a_type: BlasType, lda: c_int,
        b: *const c_void, b_type: BlasType, ldb: c_int,
        beta: *const c_void,
        c: *const c_void, c_type: BlasType, ldc: c_int,
        d: *mut c_void, d_type: BlasType, ldd: c_int,
        batch_count: c_int, compute_type: BlasType, algo: c_int, solution_index: i32, flags: u32,
    ) -> BlasStatus;
});

/// rocBLAS, opened on first use: the dense projections and the batched
/// absorbs go through it.
pub fn rocblas() -> Result<&'static Rocblas, String> {
    static ROCBLAS: OnceLock<Result<Rocblas, String>> = OnceLock::new();
    ROCBLAS
        .get_or_init(Rocblas::open)
        .as_ref()
        .map_err(Clone::clone)
}

pub type Lt = *mut c_void;
pub type LtDesc = *mut c_void;
pub type LtLayout = *mut c_void;
pub type LtPreference = *mut c_void;
pub type LtStatus = c_int;

pub const LT_SUCCESS: LtStatus = 0;
/// `HIPBLAS_COMPUTE_32F`.
pub const LT_COMPUTE_F32: c_int = 2;
/// `hipDataType`.
pub const LT_R_32F: c_int = 0;
pub const LT_R_16F: c_int = 2;
pub const LT_R_16BF: c_int = 14;
/// `hipblasOperation_t`.
pub const LT_OP_N: c_int = 111;
pub const LT_OP_T: c_int = 112;
/// `hipblasLtMatmulDescAttributes_t`.
pub const LT_DESC_TRANSA: c_int = 0;
pub const LT_DESC_TRANSB: c_int = 1;
pub const LT_DESC_EPILOGUE: c_int = 2;
pub const LT_DESC_BIAS_POINTER: c_int = 3;
pub const LT_DESC_BIAS_DATA_TYPE: c_int = 4;
/// `hipblasLtEpilogue_t`.
pub const LT_EPILOGUE_DEFAULT: c_int = 1;
pub const LT_EPILOGUE_BIAS: c_int = 4;
/// `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`.
pub const LT_PREF_MAX_WORKSPACE_BYTES: c_int = 1;
/// `hipblasLtMatrixLayoutAttribute_t`.
pub const LT_LAYOUT_BATCH_COUNT: c_int = 0;
pub const LT_LAYOUT_STRIDED_BATCH_OFFSET: c_int = 1;

/// `hipblasLtMatmulHeuristicResult_t`: a 24-byte algo, then its workspace,
/// state and waves (56 bytes, as `/opt/rocm/include/hipblaslt/hipblaslt.h`
/// lays it out).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LtHeuristic {
    pub algo: [u64; 3],
    pub workspace: usize,
    pub state: LtStatus,
    pub waves: f32,
    pub reserved: [c_int; 4],
}

const _: () = assert!(core::mem::size_of::<LtHeuristic>() == 56);

table!(Hipblaslt from "libhipblaslt.so" {
    fn hipblasLtCreate(handle: *mut Lt) -> LtStatus;
    fn hipblasLtDestroy(handle: Lt) -> LtStatus;
    fn hipblasLtGetVersion(handle: Lt, version: *mut c_int) -> LtStatus;
    fn hipblasLtMatmulDescCreate(desc: *mut LtDesc, compute: c_int, scale: c_int) -> LtStatus;
    fn hipblasLtMatmulDescDestroy(desc: LtDesc) -> LtStatus;
    fn hipblasLtMatmulDescSetAttribute(desc: LtDesc, attr: c_int, value: *const c_void, size: usize) -> LtStatus;
    fn hipblasLtMatrixLayoutCreate(layout: *mut LtLayout, dtype: c_int, rows: u64, cols: u64, ld: i64) -> LtStatus;
    fn hipblasLtMatrixLayoutDestroy(layout: LtLayout) -> LtStatus;
    fn hipblasLtMatrixLayoutSetAttribute(layout: LtLayout, attr: c_int, value: *const c_void, size: usize) -> LtStatus;
    fn hipblasLtMatmulPreferenceCreate(pref: *mut LtPreference) -> LtStatus;
    fn hipblasLtMatmulPreferenceDestroy(pref: LtPreference) -> LtStatus;
    fn hipblasLtMatmulPreferenceSetAttribute(pref: LtPreference, attr: c_int, value: *const c_void, size: usize) -> LtStatus;
    fn hipblasLtMatmulAlgoGetHeuristic(
        handle: Lt, desc: LtDesc, a: LtLayout, b: LtLayout, c: LtLayout, d: LtLayout,
        pref: LtPreference, requested: c_int, results: *mut LtHeuristic, returned: *mut c_int,
    ) -> LtStatus;
    fn hipblasLtMatmul(
        handle: Lt, desc: LtDesc, alpha: *const c_void,
        a: *const c_void, a_layout: LtLayout, b: *const c_void, b_layout: LtLayout,
        beta: *const c_void, c: *const c_void, c_layout: LtLayout,
        d: *mut c_void, d_layout: LtLayout,
        algo: *const c_void, workspace: *mut c_void, workspace_bytes: usize, stream: Stream,
    ) -> LtStatus;
});

/// hipBLASLt, opened on first use; a box without it falls back to rocBLAS.
pub fn hipblaslt() -> Result<&'static Hipblaslt, String> {
    static LT: OnceLock<Result<Hipblaslt, String>> = OnceLock::new();
    LT.get_or_init(Hipblaslt::open)
        .as_ref()
        .map_err(Clone::clone)
}

/// Where to look for a ROCm library: the loader's own search first, then
/// `$ROCM_PATH/lib`, then the default install root.
fn open_library(name: &str) -> Result<libloading::Library, String> {
    let mut tried = Vec::new();
    let mut candidates = vec![PathBuf::from(name)];
    if let Some(root) = std::env::var_os("ROCM_PATH") {
        candidates.push(PathBuf::from(root).join("lib").join(name));
    }
    candidates.push(PathBuf::from("/opt/rocm/lib").join(name));
    for path in candidates {
        // SAFETY: loading a ROCm library runs its initialisers, which is what
        // every HIP program does on start-up.
        match unsafe { libloading::Library::new(&path) } {
            Ok(lib) => return Ok(lib),
            Err(e) => tried.push(format!("{}: {e}", path.display())),
        }
    }
    Err(format!(
        "no `{name}` could be loaded ({})",
        tried.join("; ")
    ))
}

pub struct Api {
    pub hip: Hip,
    pub rtc: Hiprtc,
}

/// The loaded runtime, opened once per process. `Err` carries why the ROCm
/// libraries could not be opened, and keeps answering it.
pub fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let api = Api {
            hip: Hip::open()?,
            rtc: Hiprtc::open()?,
        };
        // SAFETY: `hipInit` takes no pointers; flags must be zero.
        let code = unsafe { (api.hip.hipInit)(0) };
        if code != HIP_SUCCESS {
            return Err(format!("`hipInit` answered {code}"));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}
