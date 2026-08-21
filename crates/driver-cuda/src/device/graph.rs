//! CUDA graphs, including the conditional nodes `batch/supergraph.cu` builds.
//! cudarc has no wrapper, so [`Graph::add_conditional_if`] builds the
//! `cudaGraphNodeParams.conditional` union arm directly. `cudaGraphSetConditional`
//! is deliberately absent from the bindings since it only runs on the GPU.

use std::ffi::c_void;
use std::marker::PhantomData;

use super::alloc::{Allocator, DeviceBuffer};

use cudarc::runtime::sys::{
    cudaConditionalNodeParams, cudaGraph_t, cudaGraphConditionalHandle,
    cudaGraphConditionalHandleCreate, cudaGraphConditionalHandleFlags,
    cudaGraphConditionalNodeType, cudaGraphDestroy, cudaGraphExec_t, cudaGraphExecDestroy,
    cudaGraphExecKernelNodeSetParams, cudaGraphInstantiate, cudaGraphKernelNodeGetParams,
    cudaGraphLaunch, cudaGraphNode_t, cudaGraphNodeParams, cudaGraphNodeType, cudaGraphUpload,
    cudaKernelNodeParams,
};

use crate::device::stream::StreamRef;
use crate::error::{Error, Result, check_rt, ignore_in_drop};

/// `cudaGraphAddNode`, spelled so one binary works on one runtime:
/// `libcudart.so.12` takes 5 args (6-arg `_v2`), `.so.13` takes 6 — the wrong
/// one segfaults with no error code.
#[cfg(feature = "cuda-12")]
pub(super) unsafe fn add_node(
    node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    deps: *const cudaGraphNode_t,
    num_deps: usize,
    params: *mut cudaGraphNodeParams,
) -> cudarc::runtime::sys::cudaError {
    // CUDA 12's headers `#define cudaGraphAddNode` to this `_v2` symbol.
    unsafe {
        cudarc::runtime::sys::cudaGraphAddNode_v2(
            node,
            graph,
            deps,
            std::ptr::null(),
            num_deps,
            params,
        )
    }
}

#[cfg(feature = "cuda-13")]
pub(super) unsafe fn add_node(
    node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    deps: *const cudaGraphNode_t,
    num_deps: usize,
    params: *mut cudaGraphNodeParams,
) -> cudarc::runtime::sys::cudaError {
    unsafe {
        cudarc::runtime::sys::cudaGraphAddNode(
            node,
            graph,
            deps,
            std::ptr::null(),
            num_deps,
            params,
        )
    }
}

/// A captured, not-yet-instantiated graph.
#[derive(Debug)]
pub struct Graph {
    raw: cudaGraph_t,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    /// Adopt a graph handle.
    /// # Safety
    /// `raw` must be a live `cudaGraph_t` nothing else will destroy.
    pub(crate) const unsafe fn from_raw(raw: cudaGraph_t) -> Self {
        Self { raw }
    }

    /// The raw handle.
    pub const fn as_raw(&self) -> cudaGraph_t {
        self.raw
    }

    /// Add an `if` node whose body is a child graph, returning the handle
    /// that selects it. `default_run: None` requires the predicate be set
    /// device-side; `Some(v)` sets it via `cudaGraphCondAssignDefault` —
    /// omit the flag and CUDA silently ignores the value. `deps` is empty for a root.
    pub fn add_conditional_if(
        &mut self,
        deps: &[cudaGraphNode_t],
        default_run: Option<bool>,
    ) -> Result<ConditionalIf<'_>> {
        let (default_value, flags) = match default_run {
            Some(v) => (
                u32::from(v),
                cudaGraphConditionalHandleFlags::cudaGraphCondAssignDefault as u32,
            ),
            None => (0, 0),
        };
        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe {
                cudaGraphConditionalHandleCreate(&mut handle, self.raw, default_value, flags)
            },
            "cudaGraphConditionalHandleCreate",
        )?;

        // `phGraph_out` is an out-pointer CUDA overwrites with the array's
        // address; left zeroed and read back. Owned by the parent graph.
        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudaConditionalNodeParams {
            handle,
            type_: cudaGraphConditionalNodeType::cudaGraphCondTypeIf,
            // `size = 1`: if/else (`size = 2`) gets its own constructor below.
            size: 1,
            phGraph_out: std::ptr::null_mut(),
        };

        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        check_rt(
            unsafe {
                add_node(
                    &raw mut node,
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &raw mut params,
                )
            },
            "cudaGraphAddNode",
        )?;

        // SAFETY: on success `phGraph_out` points at a `size` array; 1 here.
        let out = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if out.is_null() {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "the driver returned no body graph for the conditional node",
            ));
        }
        // SAFETY: `out` is non-null and has at least one element.
        let body = unsafe { *out };

        Ok(ConditionalIf {
            node,
            body,
            handle,
            _parent: PhantomData,
        })
    }

    /// A switch node: branches on an integer index rather than a boolean,
    /// reaching every arm from one node via the index the device writes into
    /// the handle — the mechanism a merged Decode/Prefill graph uses to share
    /// one exec across two topologies. `bodies` is the arm count; out-of-range runs none.
    pub fn add_conditional_switch(
        &mut self,
        deps: &[cudaGraphNode_t],
        bodies: u32,
    ) -> Result<ConditionalSwitch<'_>> {
        if bodies == 0 {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "a switch with no bodies selects nothing",
            ));
        }
        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe { cudaGraphConditionalHandleCreate(&mut handle, self.raw, 0, 0) },
            "cudaGraphConditionalHandleCreate",
        )?;
        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudaConditionalNodeParams {
            handle,
            type_: cudaGraphConditionalNodeType::cudaGraphCondTypeSwitch,
            size: bodies,
            phGraph_out: std::ptr::null_mut(),
        };
        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        check_rt(
            unsafe {
                add_node(
                    &raw mut node,
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &raw mut params,
                )
            },
            "cudaGraphAddNode",
        )?;
        let out = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if out.is_null() {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "the driver returned no body graphs for the switch node",
            ));
        }
        // SAFETY: `phGraph_out` points at exactly `bodies` graphs, owned by the parent.
        let graphs = unsafe { std::slice::from_raw_parts(out, bodies as usize) }.to_vec();
        Ok(ConditionalSwitch {
            node,
            bodies: graphs,
            handle,
            _parent: PhantomData,
        })
    }

    /// Instantiate into a launchable graph. The trailing `0` is the flags
    /// word; a bad graph reports itself through the return code alone.
    pub fn instantiate(&self) -> Result<GraphExec> {
        let mut raw: cudaGraphExec_t = std::ptr::null_mut();
        check_rt(
            unsafe { cudaGraphInstantiate(&mut raw, self.raw, 0) },
            "cudaGraphInstantiate",
        )?;
        Ok(GraphExec { raw })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            ignore_in_drop(unsafe { cudaGraphDestroy(self.raw) });
        }
    }
}

/// An `if` node added to a [`Graph`], and the handle that selects its body.
/// Borrows the parent graph, so this type has no `Drop` of its own.
#[derive(Debug, Clone, Copy)]
pub struct ConditionalIf<'g> {
    node: cudaGraphNode_t,
    body: cudaGraph_t,
    handle: cudaGraphConditionalHandle,
    _parent: PhantomData<&'g Graph>,
}

impl ConditionalIf<'_> {
    /// The node itself, to name as a dependency of a later node.
    pub const fn node(&self) -> cudaGraphNode_t {
        self.node
    }

    /// The body graph, to populate with the work the branch guards. Do not
    /// destroy it (owned by the parent).
    pub const fn body(&self) -> cudaGraph_t {
        self.body
    }

    /// The conditional handle, for the device-side kernel that sets the
    /// predicate via `cudaGraphSetConditional`.
    pub const fn handle(&self) -> cudaGraphConditionalHandle {
        self.handle
    }
}

/// A switch node added to a [`Graph`] and the handle that selects which of
/// its bodies runs. Borrows the parent graph, which owns the body graphs.
#[derive(Debug, Clone)]
pub struct ConditionalSwitch<'g> {
    node: cudaGraphNode_t,
    bodies: Vec<cudaGraph_t>,
    handle: cudaGraphConditionalHandle,
    _parent: PhantomData<&'g Graph>,
}

impl ConditionalSwitch<'_> {
    /// The node itself, to name as a dependency of a later node.
    pub const fn node(&self) -> cudaGraphNode_t {
        self.node
    }

    /// One body graph, by index. Owned by the parent graph; do not destroy it.
    #[must_use]
    pub fn body(&self, index: usize) -> Option<cudaGraph_t> {
        self.bodies.get(index).copied()
    }

    /// How many bodies this switch selects among.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Whether the switch has no bodies. Never true (the constructor refuses a
    /// zero-body switch) but clippy asks for the pair.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The conditional handle. Its consumer writes an index, not a boolean; see
    /// [`crate::fire::supergraph::set_switch`].
    pub const fn handle(&self) -> cudaGraphConditionalHandle {
        self.handle
    }
}

/// An instantiated graph, ready to launch.
#[derive(Debug)]
pub struct GraphExec {
    raw: cudaGraphExec_t,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// The raw handle.
    pub const fn as_raw(&self) -> cudaGraphExec_t {
        self.raw
    }

    /// Upload to the device without launching, so the first launch does not pay for it.
    pub fn upload(&self, stream: StreamRef<'_>) -> Result<()> {
        check_rt(
            unsafe { cudaGraphUpload(self.raw, stream.as_raw()) },
            "cudaGraphUpload",
        )
    }

    /// Launch onto `stream`.
    pub fn launch(&self, stream: StreamRef<'_>) -> Result<()> {
        check_rt(
            unsafe { cudaGraphLaunch(self.raw, stream.as_raw()) },
            "cudaGraphLaunch",
        )
    }

    /// Retune one node's launch rectangle on this instantiated graph, without
    /// recapturing. Only grid/block dims are updatable; a rejected update
    /// means the caller should recapture instead.
    pub fn set_kernel_grid(&self, node: cudaGraphNode_t, grid_x: u32) -> Result<()> {
        let mut params: cudaKernelNodeParams = unsafe { std::mem::zeroed() };
        check_rt(
            unsafe { cudaGraphKernelNodeGetParams(node, &raw mut params) },
            "cudaGraphKernelNodeGetParams",
        )?;
        params.gridDim.x = grid_x;
        check_rt(
            unsafe { cudaGraphExecKernelNodeSetParams(self.raw, node, &raw const params) },
            "cudaGraphExecKernelNodeSetParams",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            ignore_in_drop(unsafe { cudaGraphExecDestroy(self.raw) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These do not call CUDA; they pin that the sys vocabulary matches what
    // `supergraph.cu` assumes.

    #[test]
    fn the_conditional_vocabulary_exists_and_matches_the_cuda_headers() {
        assert_eq!(cudaGraphNodeType::cudaGraphNodeTypeConditional as u32, 13);
        assert_eq!(cudaGraphConditionalNodeType::cudaGraphCondTypeIf as u32, 0);
        // The handle is CUDA's `unsigned long long`, a plain scalar to the device.
        assert_eq!(
            std::mem::size_of::<cudaGraphConditionalHandle>(),
            std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn the_conditional_union_arm_is_populated_the_way_supergraph_cu_populates_it() {
        // Mirrors `add_conditional_if`; a union arm written/read via different
        // fields is silent corruption.
        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudaConditionalNodeParams {
            handle: 0xfeed_face,
            type_: cudaGraphConditionalNodeType::cudaGraphCondTypeIf,
            size: 1,
            phGraph_out: std::ptr::null_mut(),
        };

        let read_back = unsafe { params.__bindgen_anon_1.conditional };
        assert_eq!(read_back.handle, 0xfeed_face);
        assert_eq!(read_back.size, 1);
        assert_eq!(
            read_back.type_ as u32,
            cudaGraphConditionalNodeType::cudaGraphCondTypeIf as u32
        );
        assert_eq!(params.type_ as u32, 13);
    }

    #[test]
    fn a_conditional_if_has_no_drop_glue() {
        // A destructor here would double-free the parent's child graph.
        assert!(!std::mem::needs_drop::<ConditionalIf<'_>>());
    }
}

// ── The unionized supergraph, built on the conditional node above ──

// Rust port of `batch/supergraph.{cu,hpp}`: one captured graph per (R, N)
// bucket, branching on `if` nodes keyed by a device-resident predicate word
// ([`PredicateWord`]) a kernel arms per fire — no host round-trip needed.
// Hazard: nodes are added to whichever graph is capturing, but bodies are
// captured on a *different* stream; [`SupergraphBuilder`] tracks the nesting.

#[cfg(feature = "_cuda")]
use super::stream::OwnedStream;
#[cfg(feature = "_cuda")]
use cudarc::runtime::sys::{
    cudaStream_t, cudaStreamCaptureMode, cudaStreamCaptureStatus,
    cudaStreamUpdateCaptureDependenciesFlags,
};

/// How many predicate slots the device word holds — sized by the slot
/// vocabulary below: the highest wire in use, plus one.
pub const PRED_SLOTS: usize = 11;

/// `GuardPred::HasWriteDesc` — wire 0.
pub const SLOT_HAS_WRITE_DESC: u32 = 0;
/// `GuardPred::TokensLE(k)` — wire 1.
pub const SLOT_TOKENS_LE: u32 = 1;
/// `GuardPred::TokensGT(k)` — wire 2.
pub const SLOT_TOKENS_GT: u32 = 2;
/// `GuardPred::WantsAttnScore` — wire 3.
pub const SLOT_WANTS_ATTN_SCORE: u32 = 3;
/// `GuardPred::HasCustomMask` — wire 4.
pub const SLOT_HAS_CUSTOM_MASK: u32 = 4;
/// `GuardPred::HasStageHooks` — wire 5.
pub const SLOT_HAS_STAGE_HOOKS: u32 = 5;
/// `GuardPred::HasLora` — wire 6.
pub const SLOT_HAS_LORA: u32 = 6;
/// `GuardPred::WindowOne` — wire 7.
pub const SLOT_WINDOW_ONE: u32 = 7;
/// A Peel whose whole fire took the fast endpoint (`fast_rows == N`).
/// Above the `GuardPred` wire range: a lowering-produced region, not a guard.
pub const SLOT_PEEL_ALL_FAST: u32 = 8;
/// A Peel whose whole fire took the hooked endpoint (`fast_rows == 0`).
pub const SLOT_PEEL_ALL_HOOKED: u32 = 9;
/// `GuardPred::TokensMultipleOf(k)` — wire 10, placed above the Peel slots:
/// at wire 8 it would share a byte with `SLOT_PEEL_ALL_FAST`.
pub const SLOT_TOKENS_MULTIPLE: u32 = 10;

/// The device-resident predicate word: one byte per slot, not a bitfield —
/// an indexed byte read is a load, a bit read is a shift the kernel would need.
#[derive(Debug)]
pub struct PredicateWord {
    device: DeviceBuffer,
    host: [u8; PRED_SLOTS],
}

impl PredicateWord {
    /// Allocate the word. Every slot starts false.
    pub fn new(alloc: &Allocator) -> Result<Self> {
        let device = alloc.alloc(PRED_SLOTS)?;
        Ok(Self {
            device,
            host: [0u8; PRED_SLOTS],
        })
    }

    /// Set one slot in the host mirror; [`Self::upload`] is what the device sees.
    pub fn set(&mut self, slot: u32, on: bool) -> Result<()> {
        let i = usize::try_from(slot).unwrap_or(usize::MAX);
        let cell = self
            .host
            .get_mut(i)
            .ok_or_else(|| Error::invalid("supergraph", "predicate slot out of range"))?;
        *cell = u8::from(on);
        Ok(())
    }

    /// Clear every slot in the host mirror.
    pub const fn clear(&mut self) {
        self.host = [0u8; PRED_SLOTS];
    }

    /// Read one slot back out of the host mirror.
    pub fn get(&self, slot: u32) -> bool {
        usize::try_from(slot)
            .ok()
            .and_then(|i| self.host.get(i))
            .is_some_and(|&v| v != 0)
    }

    /// Push the host mirror to the device, ordered on `stream` — the only host
    /// participation a replay needs.
    pub fn upload(&mut self, stream: StreamRef<'_>) -> Result<()> {
        let host = self.host;
        self.device.copy_from_host(&host, stream)
    }

    /// The device address the set-cond kernel indexes.
    pub const fn device_ptr(&self) -> *const u8 {
        self.device.as_ptr().cast_const().cast::<u8>()
    }
}

/// A peel's row split, device-resident: `[start, count]`. `_devwin` kernels
/// read `win[0]`/`win[1]` and skip out-of-window rows, letting a captured
/// fire replay a different split without recapturing.
#[derive(Debug)]
pub struct PeelWindowWord {
    device: DeviceBuffer,
    host: [u32; 2],
}

impl PeelWindowWord {
    /// Allocate the word. Starts empty — "no rows", not "all rows" — so a
    /// kernel racing ahead of the host does nothing.
    pub fn new(alloc: &Allocator) -> Result<Self> {
        let device = alloc.alloc(std::mem::size_of::<u32>() * 2)?;
        Ok(Self {
            device,
            host: [0, 0],
        })
    }

    /// Set the window in the host mirror. [`Self::upload`] is what the device sees.
    pub const fn set(&mut self, start: u32, count: u32) {
        self.host = [start, count];
    }

    /// The window as the host currently believes it.
    pub const fn get(&self) -> (u32, u32) {
        (self.host[0], self.host[1])
    }

    /// Push the host mirror to the device, ordered on `stream`.
    pub fn upload(&mut self, stream: StreamRef<'_>) -> Result<()> {
        let bytes: [u8; 8] = {
            let mut b = [0u8; 8];
            b[..4].copy_from_slice(&self.host[0].to_ne_bytes());
            b[4..].copy_from_slice(&self.host[1].to_ne_bytes());
            b
        };
        self.device.copy_from_host(&bytes, stream)
    }

    /// The device address a `_devwin` launcher takes as `win_d`.
    pub const fn device_ptr(&self) -> *const u32 {
        self.device.as_ptr().cast_const().cast::<u32>()
    }
}

/// What CUDA's capture state says about a stream, at one instant.
#[cfg(feature = "_cuda")]
struct CaptureInfo {
    status: cudaStreamCaptureStatus,
    graph: cudaGraph_t,
    deps: *const cudaGraphNode_t,
    ndeps: usize,
}

/// `cudaStreamGetCaptureInfo`, spelled so one binary works on one runtime —
/// CUDA 13 uses the base symbol, CUDA 12 needs `_v3`. Calling the wrong one
/// segfaults, not an error code.
#[cfg(all(feature = "_cuda", feature = "cuda-12"))]
unsafe fn capture_info(stream: cudaStream_t) -> Result<CaptureInfo> {
    let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    let mut graph: cudaGraph_t = std::ptr::null_mut();
    let mut deps: *const cudaGraphNode_t = std::ptr::null();
    let mut edge_data: *const cudarc::runtime::sys::cudaGraphEdgeData = std::ptr::null();
    let mut ndeps: usize = 0;
    check_rt(
        unsafe {
            cudarc::runtime::sys::cudaStreamGetCaptureInfo_v3(
                stream,
                &raw mut status,
                std::ptr::null_mut(),
                &raw mut graph,
                &raw mut deps,
                &raw mut edge_data,
                &raw mut ndeps,
            )
        },
        "cudaStreamGetCaptureInfo",
    )?;
    Ok(CaptureInfo {
        status,
        graph,
        deps,
        ndeps,
    })
}

#[cfg(all(feature = "_cuda", feature = "cuda-13"))]
unsafe fn capture_info(stream: cudaStream_t) -> Result<CaptureInfo> {
    let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    let mut graph: cudaGraph_t = std::ptr::null_mut();
    let mut deps: *const cudaGraphNode_t = std::ptr::null();
    let mut edge_data: *const cudarc::runtime::sys::cudaGraphEdgeData = std::ptr::null();
    let mut ndeps: usize = 0;
    check_rt(
        unsafe {
            cudarc::runtime::sys::cudaStreamGetCaptureInfo(
                stream,
                &raw mut status,
                std::ptr::null_mut(),
                &raw mut graph,
                &raw mut deps,
                &raw mut edge_data,
                &raw mut ndeps,
            )
        },
        "cudaStreamGetCaptureInfo",
    )?;
    Ok(CaptureInfo {
        status,
        graph,
        deps,
        ndeps,
    })
}

/// `cudaStreamUpdateCaptureDependencies`, version-routed for the same reason [`capture_info`] is.
#[cfg(all(feature = "_cuda", feature = "cuda-12"))]
unsafe fn update_capture_deps(stream: cudaStream_t, node: *mut cudaGraphNode_t) -> Result<()> {
    check_rt(
        unsafe {
            cudarc::runtime::sys::cudaStreamUpdateCaptureDependencies_v2(
                stream,
                node,
                std::ptr::null(),
                1,
                cudaStreamUpdateCaptureDependenciesFlags::cudaStreamSetCaptureDependencies as u32,
            )
        },
        "cudaStreamUpdateCaptureDependencies",
    )
}

#[cfg(all(feature = "_cuda", feature = "cuda-13"))]
unsafe fn update_capture_deps(stream: cudaStream_t, node: *mut cudaGraphNode_t) -> Result<()> {
    check_rt(
        unsafe {
            cudarc::runtime::sys::cudaStreamUpdateCaptureDependencies(
                stream,
                node,
                std::ptr::null(),
                1,
                cudaStreamUpdateCaptureDependenciesFlags::cudaStreamSetCaptureDependencies as u32,
            )
        },
        "cudaStreamUpdateCaptureDependencies",
    )
}

/// A conditional node and its arm bodies. Raw handles, not a borrow: the
/// bodies belong to whichever graph was capturing, not a value the builder owns.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Cond {
    node: cudaGraphNode_t,
    if_body: cudaGraph_t,
    else_body: Option<cudaGraph_t>,
}

#[cfg(feature = "_cuda")]
impl Cond {
    /// The node, to name as a dependency.
    pub const fn node(&self) -> cudaGraphNode_t {
        self.node
    }

    /// The `if` arm's body graph, to capture into.
    pub const fn if_body(&self) -> cudaGraph_t {
        self.if_body
    }

    /// The `else` arm's body graph, when the node was opened with one.
    pub const fn else_body(&self) -> Option<cudaGraph_t> {
        self.else_body
    }
}

/// A switch node opened during capture and the bodies it selects among. See
/// [`SupergraphBuilder::open_switch`].
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone)]
pub struct Switch {
    node: cudaGraphNode_t,
    bodies: Vec<cudaGraph_t>,
}

#[cfg(feature = "_cuda")]
impl Switch {
    /// The node, to name as a dependency.
    pub const fn node(&self) -> cudaGraphNode_t {
        self.node
    }

    /// One arm's body graph, to capture into.
    #[must_use]
    pub fn body(&self, index: usize) -> Option<cudaGraph_t> {
        self.bodies.get(index).copied()
    }

    /// How many arms this switch selects among.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Whether the switch has no arms. Never true (`open_switch` refuses a
    /// zero-body switch) but clippy asks for the pair.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
}

/// The capture-time builder: a stack of body captures over a depth-indexed
/// stream pool. The root stream must already be capturing
/// ([`crate::device::CaptureScope`] opens one), which keeps the allocator shut.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct SupergraphBuilder<'a> {
    root: StreamRef<'a>,
    preds: *const u8,
    /// Depth-indexed pool for body captures, owned so streams die with the builder.
    pool: Vec<OwnedStream>,
    /// The capture stack, innermost last. `active[0]` is always the root.
    active: Vec<cudaStream_t>,
    /// The graph node each retained launch became, by launch index — needed
    /// so `cudaGraphExecKernelNodeSetParams` can retune grids without recapturing.
    nodes: Vec<Option<cudaGraphNode_t>>,
}

/// Ends any capture still running before the pooled streams are destroyed:
/// destroying one mid-capture is UB. Errors are swallowed — a destructor can't return one.
#[cfg(feature = "_cuda")]
impl Drop for SupergraphBuilder<'_> {
    fn drop(&mut self) {
        // Innermost first: an outer capture can't end while a nested one is
        // open; the root isn't ours.
        while self.active.len() > 1 {
            let stream = self.active.pop().expect("len > 1");
            // SAFETY: `stream` came from `active`, so it is one this builder
            // began a capture on and has not yet ended.
            let capturing = unsafe { capture_info(stream) }
                .is_ok_and(|i| i.status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
            if capturing {
                let mut graph: cudaGraph_t = std::ptr::null_mut();
                // SAFETY: the stream is capturing, which is exactly the
                // precondition `cudaStreamEndCapture` states.
                let ended = unsafe {
                    cudarc::runtime::sys::cudaStreamEndCapture(
                        stream,
                        std::ptr::from_mut(&mut graph),
                    )
                };
                if ended == cudarc::runtime::sys::cudaError::cudaSuccess && !graph.is_null() {
                    // SAFETY: `cudaStreamEndCapture` handed the graph over, and
                    // nothing will instantiate it.
                    unsafe {
                        let _ = cudarc::runtime::sys::cudaGraphDestroy(graph);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "_cuda")]
impl<'a> SupergraphBuilder<'a> {
    /// Start building on an already-capturing stream, reading predicates from `preds`.
    pub fn new(capture_stream: StreamRef<'a>, preds: &PredicateWord) -> Self {
        Self {
            root: capture_stream,
            preds: preds.device_ptr(),
            pool: Vec::new(),
            active: vec![capture_stream.as_raw()],
            nodes: Vec::new(),
        }
    }

    /// Records the node the launch just issued became, under `index`. A
    /// launch that created no node leaves the slot empty, not aliased to its predecessor.
    pub fn retain_node(&mut self, index: usize) {
        if self.nodes.len() <= index {
            self.nodes.resize(index + 1, None);
        }
        let Ok(info) = (unsafe { capture_info(self.raw_stream()) }) else {
            return;
        };
        if info.status != cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
            || info.ndeps != 1
            || info.deps.is_null()
        {
            return;
        }
        // SAFETY: `ndeps == 1` and `deps` is non-null.
        self.nodes[index] = Some(unsafe { *info.deps });
    }

    /// The nodes retained so far, by launch index.
    #[must_use]
    pub fn nodes(&self) -> &[Option<cudaGraphNode_t>] {
        &self.nodes
    }

    /// The stream launches should target: root at depth 0, else the
    /// innermost body stream. Returns a [`StreamRef`] so callers avoid `unsafe`.
    pub fn stream(&self) -> StreamRef<'_> {
        // SAFETY: `raw_stream` is either the root (borrowed for `'a`) or a
        // pool stream, both outliving `&self`.
        unsafe { StreamRef::from_raw(self.raw_stream()) }
    }

    /// The same handle, raw — for the FFI seams that take one.
    fn raw_stream(&self) -> cudaStream_t {
        self.active
            .last()
            .copied()
            .unwrap_or_else(|| self.root.as_raw())
    }

    /// How deep the body stack is; zero at the root.
    pub fn depth(&self) -> usize {
        self.active.len() - 1
    }

    /// Insert a conditional keyed on `pred_slot` at the current capture position.
    pub fn open_cond(&mut self, pred_slot: u32, with_else: bool) -> Result<Cond> {
        if pred_slot as usize >= PRED_SLOTS {
            return Err(Error::invalid("supergraph", "pred slot out of range"));
        }
        let s = self.raw_stream();

        // The handle belongs to whichever graph this stream is capturing —
        // root at depth 0, an arm's body graph when nested.
        let info = unsafe { capture_info(s) }?;
        if info.status != cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Err(Error::invalid("supergraph", "open_cond outside a capture"));
        }

        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe { cudaGraphConditionalHandleCreate(&raw mut handle, info.graph, 0, 0) },
            "cudaGraphConditionalHandleCreate",
        )?;

        // Launch the set-cond kernel first so the node picks it up as a
        // capture dependency, writing the predicate before the branch reads.
        crate::fire::supergraph::set_cond(handle, self.preds, pred_slot, s.cast::<c_void>())?;

        // Re-read: the deps now include the kernel just launched.
        let info = unsafe { capture_info(s) }?;

        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudarc::runtime::sys::cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudarc::runtime::sys::cudaConditionalNodeParams {
            handle,
            type_: cudarc::runtime::sys::cudaGraphConditionalNodeType::cudaGraphCondTypeIf,
            size: if with_else { 2 } else { 1 },
            phGraph_out: std::ptr::null_mut(),
        };

        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        check_rt(
            unsafe {
                add_node(
                    &raw mut node,
                    info.graph,
                    info.deps,
                    info.ndeps,
                    &raw mut params,
                )
            },
            "cudaGraphAddNode",
        )?;

        // SAFETY: on success CUDA has pointed `phGraph_out` at an array of
        // `size` graphs.
        let out = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if out.is_null() {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "the driver returned no body graph for the conditional node",
            ));
        }
        let if_body = unsafe { *out };
        let else_body = with_else.then(|| unsafe { *out.add(1) });

        Ok(Cond {
            node,
            if_body,
            else_body,
        })
    }

    /// Insert a switch keyed on `pred_slot`, with `bodies` arms — reaches
    /// every arm via the index the arming kernel writes; out-of-range selects no body.
    pub fn open_switch(&mut self, pred_slot: u32, bodies: u32) -> Result<Switch> {
        if pred_slot as usize >= PRED_SLOTS {
            return Err(Error::invalid("supergraph", "pred slot out of range"));
        }
        if bodies == 0 {
            return Err(Error::invalid(
                "supergraph",
                "a switch with no bodies selects nothing",
            ));
        }
        let s = self.raw_stream();
        let info = unsafe { capture_info(s) }?;
        if info.status != cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Err(Error::invalid(
                "supergraph",
                "open_switch outside a capture",
            ));
        }
        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe { cudaGraphConditionalHandleCreate(&raw mut handle, info.graph, 0, 0) },
            "cudaGraphConditionalHandleCreate",
        )?;
        // Arm first so the switch node picks up the write as a capture dependency.
        crate::fire::supergraph::set_switch(handle, self.preds, pred_slot, s.cast::<c_void>())?;
        let info = unsafe { capture_info(s) }?;
        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudarc::runtime::sys::cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudarc::runtime::sys::cudaConditionalNodeParams {
            handle,
            type_: cudarc::runtime::sys::cudaGraphConditionalNodeType::cudaGraphCondTypeSwitch,
            size: bodies,
            phGraph_out: std::ptr::null_mut(),
        };
        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        check_rt(
            unsafe {
                add_node(
                    &raw mut node,
                    info.graph,
                    info.deps,
                    info.ndeps,
                    &raw mut params,
                )
            },
            "cudaGraphAddNode",
        )?;
        let out = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if out.is_null() {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "the driver returned no body graphs for the switch node",
            ));
        }
        // SAFETY: `phGraph_out` points at exactly `bodies` graphs, owned by the capturing graph.
        let bodies = unsafe { std::slice::from_raw_parts(out, bodies as usize) }.to_vec();
        Ok(Switch { node, bodies })
    }

    /// Capture into an arm's body graph (push).
    pub fn begin_body(&mut self, body: cudaGraph_t) -> Result<()> {
        let depth = self.depth();
        if depth >= self.pool.len() {
            self.pool.push(OwnedStream::new(0)?);
        }
        let s = self.pool[depth].as_ref().as_raw();
        check_rt(
            unsafe {
                cudarc::runtime::sys::cudaStreamBeginCaptureToGraph(
                    s,
                    body,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    cudaStreamCaptureMode::cudaStreamCaptureModeGlobal,
                )
            },
            "cudaStreamBeginCaptureToGraph",
        )?;
        self.active.push(s);
        Ok(())
    }

    /// Finish the innermost body capture (pop).
    pub fn end_body(&mut self) -> Result<()> {
        if self.active.len() <= 1 {
            return Err(Error::invalid("supergraph", "end_body underflow"));
        }
        let s = self.raw_stream();
        let mut out: cudaGraph_t = std::ptr::null_mut();
        // The graph handed back is already the parent's; drop it, don't wrap
        // it, or `Drop` would double-destroy it.
        check_rt(
            unsafe { cudarc::runtime::sys::cudaStreamEndCapture(s, &raw mut out) },
            "cudaStreamEndCapture",
        )?;
        self.active.pop();
        Ok(())
    }

    /// Collapse the current stream's capture dependencies onto `cond`'s
    /// node, so later work follows the whole branch.
    pub fn close_cond(&mut self, cond: &Cond) -> Result<()> {
        let s = self.raw_stream();
        let mut node = cond.node;
        unsafe { update_capture_deps(s, &raw mut node) }
    }
}

#[cfg(test)]
mod tests_2 {
    use super::*;

    // No CUDA call: pins the slot vocabulary against `GuardPred::wire`, since
    // a wrong slot silently reads a neighbour.
    #[test]
    fn slots_match_the_guard_wire_vocabulary() {
        use model_ir::trace::GuardPred;
        assert_eq!(GuardPred::HasWriteDesc.wire().0, SLOT_HAS_WRITE_DESC);
        assert_eq!(GuardPred::TokensLE(0).wire().0, SLOT_TOKENS_LE);
        assert_eq!(GuardPred::TokensGT(0).wire().0, SLOT_TOKENS_GT);
        assert_eq!(
            GuardPred::TokensMultipleOf(0).wire().0,
            SLOT_TOKENS_MULTIPLE
        );
        // The new slot sits above the Peel pair, not beside the guards.
        assert!((SLOT_TOKENS_MULTIPLE as usize) < PRED_SLOTS);
        assert_ne!(SLOT_TOKENS_MULTIPLE, SLOT_PEEL_ALL_FAST);
        assert_ne!(SLOT_TOKENS_MULTIPLE, SLOT_PEEL_ALL_HOOKED);
        assert_eq!(GuardPred::WantsAttnScore.wire().0, SLOT_WANTS_ATTN_SCORE);
        assert_eq!(GuardPred::HasCustomMask.wire().0, SLOT_HAS_CUSTOM_MASK);
        assert_eq!(GuardPred::HasStageHooks.wire().0, SLOT_HAS_STAGE_HOOKS);
        assert_eq!(GuardPred::HasLora.wire().0, SLOT_HAS_LORA);
    }

    #[test]
    fn the_peel_bits_sit_above_the_guard_range() {
        assert!(SLOT_PEEL_ALL_FAST > SLOT_HAS_LORA);
        assert!((SLOT_PEEL_ALL_HOOKED as usize) < PRED_SLOTS);
    }
}
