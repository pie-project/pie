//! HIP under the CUDA runtime, driver, NCCL and cuBLAS names the engine
//! calls, as `cudarc`'s `sys` modules spell them: each entry forwards to
//! the HIP, RCCL or rocBLAS call it names, so the device layer reads the
//! same on either backend. Constants carry HIP's values, which are CUDA's
//! wherever both define them. Enumerators HIP writes stand whether or not
//! the engine names them.
#![allow(unsafe_code)]
#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

pub mod held;

/// The runtime API (`crate::device::rt`).
pub mod rt {
    use core::ffi::{c_char, c_int, c_uint, c_void};

    use kernels_cuda::hip::rt as hip;

    /// `hipError_t`, every code HIP defines under CUDA's name for it.
    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum cudaError {
        cudaSuccess = 0,
        cudaErrorInvalidValue = 1,
        cudaErrorMemoryAllocation = 2,
        cudaErrorInitializationError = 3,
        cudaErrorDeinitialized = 4,
        cudaErrorProfilerDisabled = 5,
        cudaErrorProfilerNotInitialized = 6,
        cudaErrorProfilerAlreadyStarted = 7,
        cudaErrorProfilerAlreadyStopped = 8,
        cudaErrorInvalidConfiguration = 9,
        cudaErrorInvalidPitchValue = 12,
        cudaErrorInvalidSymbol = 13,
        cudaErrorInvalidDevicePointer = 17,
        cudaErrorInvalidMemcpyDirection = 21,
        cudaErrorInsufficientDriver = 35,
        cudaErrorMissingConfiguration = 52,
        cudaErrorPriorLaunchFailure = 53,
        cudaErrorInvalidDeviceFunction = 98,
        cudaErrorNoDevice = 100,
        cudaErrorInvalidDevice = 101,
        cudaErrorInvalidImage = 200,
        cudaErrorInvalidContext = 201,
        cudaErrorContextAlreadyCurrent = 202,
        cudaErrorMapBufferObjectFailed = 205,
        cudaErrorUnmapFailed = 206,
        cudaErrorArrayIsMapped = 207,
        cudaErrorAlreadyMapped = 208,
        cudaErrorNoBinaryForGpu = 209,
        cudaErrorAlreadyAcquired = 210,
        cudaErrorNotMapped = 211,
        cudaErrorNotMappedAsArray = 212,
        cudaErrorNotMappedAsPointer = 213,
        cudaErrorECCNotCorrectable = 214,
        cudaErrorUnsupportedLimit = 215,
        cudaErrorContextAlreadyInUse = 216,
        cudaErrorPeerAccessUnsupported = 217,
        cudaErrorInvalidKernelFile = 218,
        cudaErrorInvalidGraphicsContext = 219,
        cudaErrorInvalidSource = 300,
        cudaErrorFileNotFound = 301,
        cudaErrorSharedObjectSymbolNotFound = 302,
        cudaErrorSharedObjectInitFailed = 303,
        cudaErrorOperatingSystem = 304,
        cudaErrorInvalidResourceHandle = 400,
        cudaErrorIllegalState = 401,
        cudaErrorNotFound = 500,
        cudaErrorNotReady = 600,
        cudaErrorIllegalAddress = 700,
        cudaErrorLaunchOutOfResources = 701,
        cudaErrorLaunchTimeOut = 702,
        cudaErrorPeerAccessAlreadyEnabled = 704,
        cudaErrorPeerAccessNotEnabled = 705,
        cudaErrorSetOnActiveProcess = 708,
        cudaErrorContextIsDestroyed = 709,
        cudaErrorAssert = 710,
        cudaErrorHostMemoryAlreadyRegistered = 712,
        cudaErrorHostMemoryNotRegistered = 713,
        cudaErrorLaunchFailure = 719,
        cudaErrorCooperativeLaunchTooLarge = 720,
        cudaErrorNotSupported = 801,
        cudaErrorStreamCaptureUnsupported = 900,
        cudaErrorStreamCaptureInvalidated = 901,
        cudaErrorStreamCaptureMerge = 902,
        cudaErrorStreamCaptureUnmatched = 903,
        cudaErrorStreamCaptureUnjoined = 904,
        cudaErrorStreamCaptureIsolation = 905,
        cudaErrorStreamCaptureImplicit = 906,
        cudaErrorCapturedEvent = 907,
        cudaErrorStreamCaptureWrongThread = 908,
        cudaErrorGraphExecUpdateFailure = 910,
        cudaErrorInvalidChannelDescriptor = 911,
        cudaErrorInvalidTexture = 912,
        cudaErrorInvalidResourceType = 914,
        cudaErrorInvalidResourceConfiguration = 915,
        cudaErrorStreamDetached = 916,
        cudaErrorUnknown = 999,
        cudaErrorRuntimeMemory = 1052,
        cudaErrorRuntimeOther = 1053,
        cudaErrorInvalidClusterSize = 1054,
    }

    pub type cudaError_t = cudaError;

    impl cudaError {
        /// The driver API's name for success (`CUresult` is this type).
        pub const CUDA_SUCCESS: cudaError = cudaError::cudaSuccess;

        /// HIP's answer as its enumerator; a code this table does not name
        /// (a newer runtime's) reads as `cudaErrorUnknown`.
        #[must_use]
        pub fn of(code: c_int) -> cudaError {
            if matches!(code, 0..=9 | 12..=13 | 17 | 21 | 35 | 52..=53 | 98 | 100..=101 | 200..=202 | 205..=219 | 300..=304 | 400..=401 | 500 | 600 | 700..=702 | 704..=705 | 708..=710 | 712..=713 | 719..=720 | 801 | 900..=908 | 910..=912 | 914..=916 | 999 | 1052..=1054)
            {
                // SAFETY: `code` is one of the discriminants above.
                unsafe { core::mem::transmute::<c_int, cudaError>(code) }
            } else {
                cudaError::cudaErrorUnknown
            }
        }
    }

    pub type cudaStream_t = *mut c_void;
    pub type cudaEvent_t = *mut c_void;
    pub type cudaGraph_t = *mut c_void;
    pub type cudaGraphExec_t = *mut c_void;
    pub type cudaHostFn_t = Option<unsafe extern "C" fn(*mut c_void)>;

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaMemcpyKind {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaStreamCaptureMode {
        cudaStreamCaptureModeGlobal = 0,
        cudaStreamCaptureModeThreadLocal = 1,
        cudaStreamCaptureModeRelaxed = 2,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaStreamCaptureStatus {
        cudaStreamCaptureStatusNone = 0,
        cudaStreamCaptureStatusActive = 1,
        cudaStreamCaptureStatusInvalidated = 2,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaMemLocationType {
        cudaMemLocationTypeInvalid = 0,
        cudaMemLocationTypeDevice = 1,
        cudaMemLocationTypeHost = 2,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaMemcpySrcAccessOrder {
        cudaMemcpySrcAccessOrderInvalid = 0,
        cudaMemcpySrcAccessOrderStream = 1,
        cudaMemcpySrcAccessOrderDuringApiCall = 2,
        cudaMemcpySrcAccessOrderAny = 3,
    }

    /// `hipMemoryType`, every value HIP writes.
    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaMemoryType {
        cudaMemoryTypeUnregistered = 0,
        cudaMemoryTypeHost = 1,
        cudaMemoryTypeDevice = 2,
        cudaMemoryTypeManaged = 3,
        cudaMemoryTypeArray = 10,
        cudaMemoryTypeUnified = 11,
    }

    /// The device attributes the engine asks, at HIP's ids.
    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaDeviceAttr {
        cudaDevAttrComputeCapabilityMajor = 23,
        cudaDevAttrComputeCapabilityMinor = 61,
        cudaDevAttrMultiProcessorCount = 63,
        cudaDevAttrPageableMemoryAccess = 65,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cudaDeviceP2PAttr {
        cudaDevP2PAttrPerformanceRank = 0,
        cudaDevP2PAttrAccessSupported = 1,
        cudaDevP2PAttrNativeAtomicSupported = 2,
    }

    pub const cudaHostAllocDefault: c_uint = 0;
    pub const cudaHostAllocPortable: c_uint = 1;
    pub const cudaHostAllocMapped: c_uint = 2;
    pub const cudaHostRegisterMapped: c_uint = 2;

    /// `hipMemLocation`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct cudaMemLocation {
        pub type_: cudaMemLocationType,
        pub id: c_int,
    }

    /// `hipMemcpyAttributes`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct cudaMemcpyAttributes {
        pub srcAccessOrder: cudaMemcpySrcAccessOrder,
        pub srcLocHint: cudaMemLocation,
        pub dstLocHint: cudaMemLocation,
        pub flags: c_uint,
    }

    /// `hipPointerAttribute_t`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct cudaPointerAttributes {
        pub type_: cudaMemoryType,
        pub device: c_int,
        pub devicePointer: *mut c_void,
        pub hostPointer: *mut c_void,
        pub isManaged: c_int,
        pub allocationFlags: c_uint,
    }

    const _: () = {
        assert!(size_of::<cudaMemcpyAttributes>() == size_of::<hip::MemcpyAttributes>());
        assert!(size_of::<cudaPointerAttributes>() == size_of::<hip::PointerAttributes>());
    };

    pub unsafe fn cudaGetLastError() -> cudaError {
        // SAFETY: takes no arguments; it reads and clears this thread's last
        // error.
        cudaError::of(unsafe { hip::hipGetLastError() })
    }
    pub unsafe fn cudaGetDeviceCount(count: *mut c_int) -> cudaError {
        // SAFETY: `hipGetDeviceCount` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGetDeviceCount(count) })
    }
    pub unsafe fn cudaSetDevice(device: c_int) -> cudaError {
        // SAFETY: takes a device index by value; HIP checks it.
        cudaError::of(unsafe { hip::hipSetDevice(device) })
    }
    pub unsafe fn cudaGetDevice(device: *mut c_int) -> cudaError {
        // SAFETY: `hipGetDevice` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGetDevice(device) })
    }
    pub unsafe fn cudaDeviceSynchronize() -> cudaError {
        // SAFETY: takes no arguments; it waits on the current device.
        cudaError::of(unsafe { hip::hipDeviceSynchronize() })
    }
    pub unsafe fn cudaDeviceGetAttribute(
        value: *mut c_int,
        attr: cudaDeviceAttr,
        device: c_int,
    ) -> cudaError {
        // SAFETY: the caller's `value` contract, forwarded; `attr` is one of
        // the enum's ids, which are HIP's.
        cudaError::of(unsafe { hip::hipDeviceGetAttribute(value, attr as c_int, device) })
    }
    pub unsafe fn cudaDeviceCanAccessPeer(
        can: *mut c_int,
        device: c_int,
        peer: c_int,
    ) -> cudaError {
        // SAFETY: `hipDeviceCanAccessPeer` takes the same arguments with the
        // same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipDeviceCanAccessPeer(can, device, peer) })
    }
    pub unsafe fn cudaDeviceGetP2PAttribute(
        value: *mut c_int,
        attr: cudaDeviceP2PAttr,
        src: c_int,
        dst: c_int,
    ) -> cudaError {
        // SAFETY: the caller's `value` contract, forwarded; `attr` is one of
        // the enum's ids, which are HIP's.
        cudaError::of(unsafe { hip::hipDeviceGetP2PAttribute(value, attr as c_int, src, dst) })
    }
    pub unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError {
        // SAFETY: `hipMemGetInfo` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipMemGetInfo(free, total) })
    }
    pub unsafe fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> cudaError {
        // SAFETY: `hipMalloc` takes the same arguments with the same contract,
        // which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipMalloc(ptr, bytes) })
    }
    pub unsafe fn cudaFree(ptr: *mut c_void) -> cudaError {
        // SAFETY: `hipFree` takes the same arguments with the same contract,
        // which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipFree(ptr) })
    }
    pub unsafe fn cudaMemset(dst: *mut c_void, value: c_int, bytes: usize) -> cudaError {
        // SAFETY: `hipMemset` takes the same arguments with the same contract,
        // which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipMemset(dst, value, bytes) })
    }
    pub unsafe fn cudaMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        bytes: usize,
        stream: cudaStream_t,
    ) -> cudaError {
        // SAFETY: `hipMemsetAsync` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipMemsetAsync(dst, value, bytes, stream) })
    }
    pub unsafe fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError {
        // SAFETY: the caller's span contract, forwarded; `kind` carries HIP's
        // value for the direction.
        cudaError::of(unsafe { hip::hipMemcpy(dst, src, bytes, kind as c_int) })
    }
    pub unsafe fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError {
        // SAFETY: the caller's span and stream contract, forwarded; `kind`
        // carries HIP's value for the direction.
        cudaError::of(unsafe { hip::hipMemcpyAsync(dst, src, bytes, kind as c_int, stream) })
    }
    /// HIP takes the arrays mutably and reports the failing copy; neither
    /// is written through here.
    pub unsafe fn cudaMemcpyBatchAsync(
        dsts: *const *mut c_void,
        srcs: *const *const c_void,
        sizes: *const usize,
        count: usize,
        attrs: *mut cudaMemcpyAttributes,
        attr_idxs: *mut usize,
        num_attrs: usize,
        stream: cudaStream_t,
    ) -> cudaError {
        let mut failed = 0usize;
        // SAFETY: the caller's span and stream contract, forwarded; HIP reads
        // the arrays though it types them `mut`, the layout assert above makes
        // `attrs` HIP's struct, and `failed` is a live local it writes.
        cudaError::of(unsafe {
            hip::hipMemcpyBatchAsync(
                dsts.cast_mut(),
                srcs.cast_mut().cast(),
                sizes.cast_mut(),
                count,
                attrs.cast(),
                attr_idxs,
                num_attrs,
                &raw mut failed,
                stream,
            )
        })
    }
    pub unsafe fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: c_uint) -> cudaError {
        // SAFETY: `hipHostMalloc` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipHostMalloc(ptr, bytes, flags) })
    }
    pub unsafe fn cudaFreeHost(ptr: *mut c_void) -> cudaError {
        // SAFETY: `hipHostFree` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipHostFree(ptr) })
    }
    pub unsafe fn cudaHostRegister(ptr: *mut c_void, bytes: usize, flags: c_uint) -> cudaError {
        // SAFETY: `hipHostRegister` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipHostRegister(ptr, bytes, flags) })
    }
    pub unsafe fn cudaHostUnregister(ptr: *mut c_void) -> cudaError {
        // SAFETY: `hipHostUnregister` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipHostUnregister(ptr) })
    }
    pub unsafe fn cudaHostGetDevicePointer(
        device: *mut *mut c_void,
        host: *mut c_void,
        flags: c_uint,
    ) -> cudaError {
        // SAFETY: `hipHostGetDevicePointer` takes the same arguments with the
        // same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipHostGetDevicePointer(device, host, flags) })
    }
    pub unsafe fn cudaPointerGetAttributes(
        attrs: *mut cudaPointerAttributes,
        ptr: *const c_void,
    ) -> cudaError {
        // SAFETY: the caller's `attrs` contract, forwarded; the layout assert
        // above makes the struct HIP's and `cudaMemoryType` names every value
        // HIP writes into it.
        cudaError::of(unsafe { hip::hipPointerGetAttributes(attrs.cast(), ptr) })
    }
    pub unsafe fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError {
        // SAFETY: `hipStreamCreate` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamCreate(stream) })
    }
    pub unsafe fn cudaStreamCreateWithFlags(stream: *mut cudaStream_t, flags: c_uint) -> cudaError {
        // SAFETY: `hipStreamCreateWithFlags` takes the same arguments with the
        // same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamCreateWithFlags(stream, flags) })
    }
    pub unsafe fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError {
        // SAFETY: `hipStreamDestroy` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamDestroy(stream) })
    }
    pub unsafe fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError {
        // SAFETY: `hipStreamSynchronize` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamSynchronize(stream) })
    }
    pub unsafe fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: c_uint,
    ) -> cudaError {
        // SAFETY: `hipStreamWaitEvent` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamWaitEvent(stream, event, flags) })
    }
    pub unsafe fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        func: cudaHostFn_t,
        data: *mut c_void,
    ) -> cudaError {
        match func {
            // SAFETY: the caller's stream, callback and `data` contract,
            // forwarded; `func` is non-null here.
            Some(func) => cudaError::of(unsafe { hip::hipLaunchHostFunc(stream, func, data) }),
            None => cudaError::cudaErrorInvalidValue,
        }
    }
    pub unsafe fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> cudaError {
        // SAFETY: the caller's stream contract, forwarded; `mode` carries HIP's
        // value.
        cudaError::of(unsafe { hip::hipStreamBeginCapture(stream, mode as c_int) })
    }
    pub unsafe fn cudaStreamEndCapture(stream: cudaStream_t, graph: *mut cudaGraph_t) -> cudaError {
        // SAFETY: `hipStreamEndCapture` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipStreamEndCapture(stream, graph) })
    }
    pub unsafe fn cudaStreamIsCapturing(
        stream: cudaStream_t,
        status: *mut cudaStreamCaptureStatus,
    ) -> cudaError {
        // SAFETY: the caller's `status` contract, forwarded; HIP writes one of
        // the enum's three values.
        cudaError::of(unsafe { hip::hipStreamIsCapturing(stream, status.cast()) })
    }
    pub unsafe fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError {
        // SAFETY: `hipEventCreate` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventCreate(event) })
    }
    pub unsafe fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError {
        // SAFETY: `hipEventCreateWithFlags` takes the same arguments with the
        // same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventCreateWithFlags(event, flags) })
    }
    pub unsafe fn cudaEventDestroy(event: cudaEvent_t) -> cudaError {
        // SAFETY: `hipEventDestroy` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventDestroy(event) })
    }
    pub unsafe fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError {
        // SAFETY: `hipEventRecord` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventRecord(event, stream) })
    }
    pub unsafe fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError {
        // SAFETY: `hipEventSynchronize` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventSynchronize(event) })
    }
    pub unsafe fn cudaEventQuery(event: cudaEvent_t) -> cudaError {
        // SAFETY: `hipEventQuery` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipEventQuery(event) })
    }
    pub unsafe fn cudaGraphInstantiateWithFlags(
        exec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> cudaError {
        // SAFETY: `hipGraphInstantiateWithFlags` takes the same arguments with
        // the same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphInstantiateWithFlags(exec, graph, flags) })
    }
    pub unsafe fn cudaGraphUpload(exec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError {
        // SAFETY: `hipGraphUpload` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphUpload(exec, stream) })
    }
    pub unsafe fn cudaGraphLaunch(exec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError {
        // SAFETY: `hipGraphLaunch` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphLaunch(exec, stream) })
    }
    pub unsafe fn cudaGraphExecDestroy(exec: cudaGraphExec_t) -> cudaError {
        // SAFETY: `hipGraphExecDestroy` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphExecDestroy(exec) })
    }
    pub unsafe fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError {
        // SAFETY: `hipGraphDestroy` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphDestroy(graph) })
    }
    pub unsafe fn cudaGraphDebugDotPrint(
        graph: cudaGraph_t,
        path: *const c_char,
        flags: c_uint,
    ) -> cudaError {
        // SAFETY: `hipGraphDebugDotPrint` takes the same arguments with the
        // same contract, which the caller of this fn upholds.
        cudaError::of(unsafe { hip::hipGraphDebugDotPrint(graph, path, flags) })
    }
}

/// The driver API (`crate::device::dr`).
pub mod dr {
    use core::ffi::{c_int, c_uint, c_void};

    use kernels_cuda::hip::rt as hip;

    pub use super::rt::cudaError as CUresult;

    pub type CUdevice = c_int;
    pub type CUdeviceptr = u64;
    pub type CUmodule = hip::Module;
    pub type CUfunction = hip::Function;
    pub type CUstream = *mut c_void;
    pub type CUgraph = *mut c_void;
    pub type CUgraphNode = *mut c_void;
    pub type CUmemGenericAllocationHandle = u64;

    /// The device attributes the engine asks, at HIP's ids.
    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUdevice_attribute {
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 89,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUfunction_attribute {
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUmemAllocationType {
        CU_MEM_ALLOCATION_TYPE_INVALID = 0,
        CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUmemLocationType {
        CU_MEM_LOCATION_TYPE_INVALID = 0,
        CU_MEM_LOCATION_TYPE_DEVICE = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUmemAccess_flags {
        CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
        CU_MEM_ACCESS_FLAGS_PROT_READ = 1,
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
    }

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CUmemAllocationGranularity_flags {
        CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
    }

    /// `hipMemLocation`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct CUmemLocation {
        pub type_: CUmemLocationType,
        pub id: c_int,
    }

    /// `hipMemAllocationProp`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct CUmemAllocationProp {
        pub type_: CUmemAllocationType,
        pub requestedHandleTypes: c_int,
        pub location: CUmemLocation,
        pub win32HandleMetaData: *mut c_void,
        pub compressionType: u8,
        pub gpuDirectRDMACapable: u8,
        pub usage: u16,
        pub reserved: u32,
    }

    /// `hipMemAccessDesc`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct CUmemAccessDesc {
        pub location: CUmemLocation,
        pub flags: CUmemAccess_flags,
    }

    const _: () = {
        assert!(size_of::<CUmemAllocationProp>() == size_of::<hip::MemAllocationProp>());
        assert!(size_of::<CUmemAccessDesc>() == size_of::<hip::MemAccessDesc>());
    };

    pub unsafe fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
        // SAFETY: `hipGetDevice` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipGetDevice(device) })
    }
    pub unsafe fn cuDeviceGetAttribute(
        value: *mut c_int,
        attr: CUdevice_attribute,
        device: CUdevice,
    ) -> CUresult {
        // SAFETY: the caller's `value` contract, forwarded; `attr` is one of
        // the enum's ids, which are HIP's.
        CUresult::of(unsafe { hip::hipDeviceGetAttribute(value, attr as c_int, device) })
    }
    pub unsafe fn cuFuncGetAttribute(
        value: *mut c_int,
        attr: CUfunction_attribute,
        function: CUfunction,
    ) -> CUresult {
        // SAFETY: the caller's `value` and `function` contract, forwarded;
        // `attr` is one of the enum's ids, which are HIP's.
        CUresult::of(unsafe { hip::hipFuncGetAttribute(value, attr as c_int, function) })
    }
    pub unsafe fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
        // SAFETY: `hipModuleLoadData` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipModuleLoadData(module, image) })
    }
    pub unsafe fn cuModuleGetFunction(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const core::ffi::c_char,
    ) -> CUresult {
        // SAFETY: `hipModuleGetFunction` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipModuleGetFunction(function, module, name) })
    }
    pub unsafe fn cuModuleUnload(module: CUmodule) -> CUresult {
        // SAFETY: `hipModuleUnload` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipModuleUnload(module) })
    }
    pub unsafe fn cuLaunchKernel(
        function: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared: c_uint,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult {
        // SAFETY: the caller's launch contract (a loaded function, a live
        // stream, `params` matching its signature), forwarded unchanged.
        CUresult::of(unsafe {
            hip::hipModuleLaunchKernel(
                function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared, stream,
                params, extra,
            )
        })
    }
    /// Only a value whose four bytes are one byte repeated (zero, as the
    /// engine sets): HIP has no 32-bit 2D memset, so the rows go as bytes.
    pub unsafe fn cuMemsetD2D32Async(
        dst: CUdeviceptr,
        pitch: usize,
        value: c_uint,
        width: usize,
        height: usize,
        stream: CUstream,
    ) -> CUresult {
        let byte = value & 0xff;
        if value != byte * 0x0101_0101 {
            return CUresult::cudaErrorInvalidValue;
        }
        // SAFETY: the caller's span contract for `height` rows of `width`
        // 32-bit words at `pitch`, forwarded as the same bytes; `byte` repeated
        // is the checked value.
        CUresult::of(unsafe {
            hip::hipMemset2DAsync(
                dst as *mut c_void,
                pitch,
                byte as c_int,
                width * 4,
                height,
                stream,
            )
        })
    }
    pub unsafe fn cuGraphGetNodes(
        graph: CUgraph,
        nodes: *mut CUgraphNode,
        count: *mut usize,
    ) -> CUresult {
        // SAFETY: `hipGraphGetNodes` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipGraphGetNodes(graph, nodes, count) })
    }
    /// HIP keeps no edge data; `data` must be null.
    pub unsafe fn cuGraphGetEdges_v2(
        graph: CUgraph,
        from: *mut CUgraphNode,
        to: *mut CUgraphNode,
        data: *mut c_void,
        count: *mut usize,
    ) -> CUresult {
        if !data.is_null() {
            return CUresult::cudaErrorNotSupported;
        }
        // SAFETY: the caller's `from`/`to`/`count` contract, forwarded; `data`
        // is null, so nothing of it is dropped.
        CUresult::of(unsafe { hip::hipGraphGetEdges(graph, from, to, count) })
    }
    pub unsafe fn cuMemCreate(
        handle: *mut CUmemGenericAllocationHandle,
        bytes: usize,
        prop: *const CUmemAllocationProp,
        flags: u64,
    ) -> CUresult {
        // SAFETY: the caller's `handle` and `prop` contract, forwarded; the
        // layout assert above makes `prop` HIP's struct.
        CUresult::of(unsafe { hip::hipMemCreate(handle, bytes, prop.cast(), flags) })
    }
    pub unsafe fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
        // SAFETY: `hipMemRelease` takes the same arguments with the same
        // contract, which the caller of this fn upholds.
        CUresult::of(unsafe { hip::hipMemRelease(handle) })
    }
    pub unsafe fn cuMemAddressReserve(
        ptr: *mut CUdeviceptr,
        bytes: usize,
        alignment: usize,
        addr: CUdeviceptr,
        flags: u64,
    ) -> CUresult {
        // SAFETY: the caller's `ptr` contract, forwarded; a device address is
        // the same 64 bits as HIP's pointer.
        CUresult::of(unsafe {
            hip::hipMemAddressReserve(ptr.cast(), bytes, alignment, addr as *mut c_void, flags)
        })
    }
    pub unsafe fn cuMemAddressFree(ptr: CUdeviceptr, bytes: usize) -> CUresult {
        // SAFETY: the caller's range contract, forwarded; a device address is
        // the same 64 bits as HIP's pointer.
        CUresult::of(unsafe { hip::hipMemAddressFree(ptr as *mut c_void, bytes) })
    }
    pub unsafe fn cuMemMap(
        ptr: CUdeviceptr,
        bytes: usize,
        offset: usize,
        handle: CUmemGenericAllocationHandle,
        flags: u64,
    ) -> CUresult {
        // SAFETY: the caller's range and handle contract, forwarded; a device
        // address is the same 64 bits as HIP's pointer.
        CUresult::of(unsafe { hip::hipMemMap(ptr as *mut c_void, bytes, offset, handle, flags) })
    }
    pub unsafe fn cuMemUnmap(ptr: CUdeviceptr, bytes: usize) -> CUresult {
        // SAFETY: the caller's range contract, forwarded; a device address is
        // the same 64 bits as HIP's pointer.
        CUresult::of(unsafe { hip::hipMemUnmap(ptr as *mut c_void, bytes) })
    }
    pub unsafe fn cuMemSetAccess(
        ptr: CUdeviceptr,
        bytes: usize,
        desc: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult {
        // SAFETY: the caller's range and `desc` contract, forwarded; the layout
        // assert above makes `desc` HIP's struct.
        CUresult::of(unsafe { hip::hipMemSetAccess(ptr as *mut c_void, bytes, desc.cast(), count) })
    }
    pub unsafe fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CUmemAllocationProp,
        option: CUmemAllocationGranularity_flags,
    ) -> CUresult {
        // SAFETY: the caller's `granularity` and `prop` contract, forwarded;
        // the layout assert above makes `prop` HIP's struct and `option`
        // carries HIP's value.
        CUresult::of(unsafe {
            hip::hipMemGetAllocationGranularity(granularity, prop.cast(), option as c_int)
        })
    }
}

/// RCCL (`crate::device::nccl`), opened on first use.
pub mod nccl {
    use core::ffi::c_int;

    use kernels_cuda::hip::sys;

    pub type ncclComm_t = sys::Comm;
    pub type ncclUniqueId = sys::UniqueId;

    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ncclResult_t {
        ncclSuccess = 0,
        ncclUnhandledCudaError = 1,
        ncclSystemError = 2,
        ncclInternalError = 3,
        ncclInvalidArgument = 4,
        ncclInvalidUsage = 5,
        ncclRemoteError = 6,
        ncclInProgress = 7,
    }

    impl ncclResult_t {
        /// RCCL's answer; one it does not define, or no RCCL to load, reads
        /// as `ncclSystemError`.
        fn of(code: c_int) -> ncclResult_t {
            match code {
                // SAFETY: 0..=7 are the discriminants above.
                0..=7 => unsafe { core::mem::transmute::<c_int, ncclResult_t>(code) },
                _ => ncclResult_t::ncclSystemError,
            }
        }
    }

    pub unsafe fn ncclGetUniqueId(id: *mut ncclUniqueId) -> ncclResult_t {
        match sys::rccl() {
            // SAFETY: the caller's `id` contract, forwarded to RCCL's entry of
            // the same signature.
            Ok(rccl) => ncclResult_t::of(unsafe { (rccl.ncclGetUniqueId)(id) }),
            Err(_) => ncclResult_t::ncclSystemError,
        }
    }
    pub unsafe fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        ranks: c_int,
        id: ncclUniqueId,
        rank: c_int,
    ) -> ncclResult_t {
        match sys::rccl() {
            // SAFETY: the caller's `comm` contract, forwarded to RCCL's entry
            // of the same signature.
            Ok(rccl) => ncclResult_t::of(unsafe { (rccl.ncclCommInitRank)(comm, ranks, id, rank) }),
            Err(_) => ncclResult_t::ncclSystemError,
        }
    }
    pub unsafe fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t {
        match sys::rccl() {
            // SAFETY: the caller's live-communicator contract, forwarded to
            // RCCL's entry of the same signature.
            Ok(rccl) => ncclResult_t::of(unsafe { (rccl.ncclCommAbort)(comm) }),
            Err(_) => ncclResult_t::ncclSystemError,
        }
    }
}

/// rocBLAS (`crate::device::blas`), opened on first use.
pub mod blas {
    use core::ffi::{c_int, c_void};

    use kernels_cuda::hip::sys;

    pub type cublasHandle_t = *mut c_void;

    /// `rocblas_status`.
    #[repr(i32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum cublasStatus_t {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_INVALID_HANDLE = 1,
        CUBLAS_STATUS_NOT_IMPLEMENTED = 2,
        CUBLAS_STATUS_INVALID_POINTER = 3,
        CUBLAS_STATUS_INVALID_SIZE = 4,
        CUBLAS_STATUS_MEMORY_ERROR = 5,
        CUBLAS_STATUS_INTERNAL_ERROR = 6,
    }

    impl cublasStatus_t {
        /// rocBLAS's answer; one past these, or no rocBLAS to load, reads
        /// as `CUBLAS_STATUS_INTERNAL_ERROR`.
        fn of(code: c_int) -> cublasStatus_t {
            match code {
                // SAFETY: 0..=6 are the discriminants above.
                0..=6 => unsafe { core::mem::transmute::<c_int, cublasStatus_t>(code) },
                _ => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
            }
        }
    }

    pub unsafe fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
        match sys::rocblas() {
            // SAFETY: the caller's `handle` contract, forwarded to rocBLAS's
            // entry of the same signature.
            Ok(blas) => cublasStatus_t::of(unsafe { (blas.rocblas_create_handle)(handle) }),
            Err(_) => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
        }
    }
    pub unsafe fn cublasSetStream_v2(
        handle: cublasHandle_t,
        stream: *mut c_void,
    ) -> cublasStatus_t {
        match sys::rocblas() {
            // SAFETY: the caller's handle and stream contract, forwarded to
            // rocBLAS's entry of the same signature.
            Ok(blas) => cublasStatus_t::of(unsafe { (blas.rocblas_set_stream)(handle, stream) }),
            Err(_) => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
        }
    }
    pub unsafe fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
        match sys::rocblas() {
            // SAFETY: the caller's live-handle contract, forwarded to rocBLAS's
            // entry of the same signature.
            Ok(blas) => cublasStatus_t::of(unsafe { (blas.rocblas_destroy_handle)(handle) }),
            Err(_) => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
        }
    }
}
