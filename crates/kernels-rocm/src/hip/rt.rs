//! The HIP runtime as the engine calls it, under HIP's own names: each
//! entry forwards to the table [`sys::api`] loaded, and answers
//! `hipErrorNotInitialized` when no ROCm runtime could be loaded.
//!
//! Generated from the prototypes in `hip/hip_runtime_api.h`; the constants
//! and struct layouts below were read off the installed headers.
//!
//! # Safety
//!
//! Every `pub unsafe fn` here carries the safety contract of the HIP call
//! it names, unchanged: the pointers, handles and sizes it takes must be
//! what `hip_runtime_api.h` documents for that entry point. A shim adds no
//! requirement of its own, and only answers `hipErrorNotInitialized`
//! without touching its arguments when the runtime is missing.
#![allow(unsafe_code)]
#![allow(non_snake_case, non_upper_case_globals, clippy::too_many_arguments)]
#![expect(
    clippy::missing_safety_doc,
    reason = "each shim's contract is the HIP call it names, stated once in the module doc"
)]

// Struct fields carry the C names (`type` as `type_`), as the headers spell them.

use core::ffi::{c_char, c_int, c_uint, c_void};

use super::sys;
pub use super::sys::{Function, HipError, Module, Stream};

pub type Event = *mut c_void;
pub type Graph = *mut c_void;
pub type GraphExec = *mut c_void;
pub type GraphNode = *mut c_void;
pub type HostFn = unsafe extern "C" fn(*mut c_void);

pub const hipErrorNotInitialized: HipError = 3;

pub const hipMemcpyHostToDevice: c_int = 1;

pub const hipEventDisableTiming: c_uint = 2;

pub const hipStreamCaptureModeThreadLocal: c_int = 1;

pub const hipHostMallocPortable: c_uint = 1;
pub const hipHostMallocMapped: c_uint = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemLocation {
    pub type_: c_int,
    pub id: c_int,
}

/// `hipMemAllocationProp`: 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemAllocationProp {
    pub type_: c_int,
    pub requestedHandleType: c_int,
    pub location: MemLocation,
    pub win32HandleMetaData: usize,
    pub compressionType: u8,
    pub gpuDirectRDMACapable: u8,
    pub usage: u16,
    pub _pad: u32,
}

/// `hipMemAccessDesc`: 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemAccessDesc {
    pub location: MemLocation,
    pub flags: c_int,
}

/// `hipPointerAttribute_t`: 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PointerAttributes {
    pub type_: c_int,
    pub device: c_int,
    pub devicePointer: *mut c_void,
    pub hostPointer: *mut c_void,
    pub isManaged: c_int,
    pub allocationFlags: c_uint,
}

impl Default for PointerAttributes {
    fn default() -> Self {
        Self {
            type_: 0,
            device: 0,
            devicePointer: core::ptr::null_mut(),
            hostPointer: core::ptr::null_mut(),
            isManaged: 0,
            allocationFlags: 0,
        }
    }
}

/// `hipMemcpyAttributes`: 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemcpyAttributes {
    pub srcAccessOrder: c_int,
    pub srcLocHint: MemLocation,
    pub dstLocHint: MemLocation,
    pub flags: c_uint,
}

const _: () = {
    assert!(core::mem::size_of::<MemAllocationProp>() == 32);
    assert!(core::mem::size_of::<MemAccessDesc>() == 12);
    assert!(core::mem::size_of::<PointerAttributes>() == 32);
    assert!(core::mem::size_of::<MemcpyAttributes>() == 24);
};

pub unsafe fn hipGetLastError() -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGetLastError)() },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGetDeviceCount(count: *mut c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGetDeviceCount)(count) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipSetDevice(device: c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipSetDevice)(device) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGetDevice(device: *mut c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGetDevice)(device) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipDeviceSynchronize() -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipDeviceSynchronize)() },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipDeviceGetAttribute)(value, attr, device) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipDeviceCanAccessPeer(can: *mut c_int, device: c_int, peer: c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipDeviceCanAccessPeer)(can, device, peer) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipDeviceGetP2PAttribute(
    value: *mut c_int,
    attr: c_int,
    src: c_int,
    dst: c_int,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipDeviceGetP2PAttribute)(value, attr, src, dst) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemGetInfo)(free, total) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMalloc(ptr: *mut *mut c_void, bytes: usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMalloc)(ptr, bytes) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipFree(ptr: *mut c_void) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipFree)(ptr) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemset(dst: *mut c_void, value: c_int, bytes: usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemset)(dst, value, bytes) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemsetAsync(
    dst: *mut c_void,
    value: c_int,
    bytes: usize,
    stream: Stream,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemsetAsync)(dst, value, bytes, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

/// `width` bytes zeroed (`value`) at each of `height` rows `pitch` apart.
pub unsafe fn hipMemset2DAsync(
    dst: *mut c_void,
    pitch: usize,
    value: c_int,
    width: usize,
    height: usize,
    stream: Stream,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemset2DAsync)(dst, pitch, value, width, height, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: c_int,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemcpy)(dst, src, bytes, kind) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: c_int,
    stream: Stream,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemcpyAsync)(dst, src, bytes, kind, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemcpyBatchAsync(
    dsts: *mut *mut c_void,
    srcs: *mut *mut c_void,
    sizes: *mut usize,
    count: usize,
    attrs: *mut MemcpyAttributes,
    attr_idxs: *mut usize,
    num_attrs: usize,
    fail_idx: *mut usize,
    stream: Stream,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe {
            (api.hip.hipMemcpyBatchAsync)(
                dsts, srcs, sizes, count, attrs, attr_idxs, num_attrs, fail_idx, stream,
            )
        },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipHostMalloc(ptr: *mut *mut c_void, bytes: usize, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipHostMalloc)(ptr, bytes, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipHostFree(ptr: *mut c_void) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipHostFree)(ptr) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipHostRegister(ptr: *mut c_void, bytes: usize, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipHostRegister)(ptr, bytes, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipHostUnregister(ptr: *mut c_void) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipHostUnregister)(ptr) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipHostGetDevicePointer(
    dev: *mut *mut c_void,
    host: *mut c_void,
    flags: c_uint,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipHostGetDevicePointer)(dev, host, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipPointerGetAttributes(
    attrs: *mut PointerAttributes,
    ptr: *const c_void,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipPointerGetAttributes)(attrs, ptr) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamCreate(stream: *mut Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamCreate)(stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamCreateWithFlags)(stream, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamDestroy(stream: Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamDestroy)(stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamSynchronize(stream: Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamSynchronize)(stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamWaitEvent(stream: Stream, event: Event, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamWaitEvent)(stream, event, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipLaunchHostFunc(stream: Stream, func: HostFn, data: *mut c_void) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipLaunchHostFunc)(stream, func, data) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamBeginCapture(stream: Stream, mode: c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamBeginCapture)(stream, mode) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamEndCapture(stream: Stream, graph: *mut Graph) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamEndCapture)(stream, graph) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipStreamIsCapturing(stream: Stream, status: *mut c_int) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipStreamIsCapturing)(stream, status) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventCreate(event: *mut Event) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventCreate)(event) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventCreateWithFlags(event: *mut Event, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventCreateWithFlags)(event, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventDestroy(event: Event) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventDestroy)(event) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventRecord(event: Event, stream: Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventRecord)(event, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventSynchronize(event: Event) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventSynchronize)(event) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipEventQuery(event: Event) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipEventQuery)(event) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphInstantiateWithFlags(
    exec: *mut GraphExec,
    graph: Graph,
    flags: u64,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphInstantiateWithFlags)(exec, graph, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphUpload(exec: GraphExec, stream: Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphUpload)(exec, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphLaunch(exec: GraphExec, stream: Stream) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphLaunch)(exec, stream) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphExecDestroy(exec: GraphExec) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphExecDestroy)(exec) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphDestroy(graph: Graph) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphDestroy)(graph) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphDebugDotPrint(graph: Graph, path: *const c_char, flags: c_uint) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphDebugDotPrint)(graph, path, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphGetNodes(graph: Graph, nodes: *mut GraphNode, count: *mut usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphGetNodes)(graph, nodes, count) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipGraphGetEdges(
    graph: Graph,
    from: *mut GraphNode,
    to: *mut GraphNode,
    count: *mut usize,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipGraphGetEdges)(graph, from, to, count) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemCreate(
    handle: *mut u64,
    bytes: usize,
    prop: *const MemAllocationProp,
    flags: u64,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemCreate)(handle, bytes, prop, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemRelease(handle: u64) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemRelease)(handle) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemAddressReserve(
    ptr: *mut *mut c_void,
    bytes: usize,
    align: usize,
    addr: *mut c_void,
    flags: u64,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemAddressReserve)(ptr, bytes, align, addr, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemAddressFree(ptr: *mut c_void, bytes: usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemAddressFree)(ptr, bytes) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemMap(
    ptr: *mut c_void,
    bytes: usize,
    offset: usize,
    handle: u64,
    flags: u64,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemMap)(ptr, bytes, offset, handle, flags) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemUnmap(ptr: *mut c_void, bytes: usize) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemUnmap)(ptr, bytes) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemSetAccess(
    ptr: *mut c_void,
    bytes: usize,
    desc: *const MemAccessDesc,
    count: usize,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemSetAccess)(ptr, bytes, desc, count) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const MemAllocationProp,
    option: c_int,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipMemGetAllocationGranularity)(granularity, prop, option) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipFuncGetAttribute(value: *mut c_int, attr: c_int, function: Function) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipFuncGetAttribute)(value, attr, function) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipModuleLoadData(module: *mut Module, image: *const c_void) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipModuleLoadData)(module, image) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipModuleUnload(module: Module) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipModuleUnload)(module) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipModuleGetFunction(
    function: *mut Function,
    module: Module,
    name: *const c_char,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe { (api.hip.hipModuleGetFunction)(function, module, name) },
        Err(_) => hipErrorNotInitialized,
    }
}

pub unsafe fn hipModuleLaunchKernel(
    function: Function,
    gx: c_uint,
    gy: c_uint,
    gz: c_uint,
    bx: c_uint,
    by: c_uint,
    bz: c_uint,
    shared: c_uint,
    stream: Stream,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> HipError {
    match sys::api() {
        // SAFETY: forwarded unchanged; the caller holds HIP's contract.
        Ok(api) => unsafe {
            (api.hip.hipModuleLaunchKernel)(
                function, gx, gy, gz, bx, by, bz, shared, stream, params, extra,
            )
        },
        Err(_) => hipErrorNotInitialized,
    }
}
