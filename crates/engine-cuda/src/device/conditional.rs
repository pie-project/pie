//! **CONDITIONAL GRAPH NODES, PLACED INSIDE A RUNNING CAPTURE** — the
//! mechanism `Fault::Unlowered` used to name as missing (palo design §4, build
//! log 27, `.wiki/driver/new-horizon.md` §62.3).
//!
//! A region `model_compiler::lowering` stamps `Lowering::If` must reach a
//! RECORDING walk as a NODE and not as a branch the recorder took. The graph
//! outlives the fire that wrote it; a body recorded outside its node is a body
//! that runs under every composition the exec is ever replayed for. This
//! module is the four driver calls that make the node, in the one order the
//! driver accepts them:
//!
//! ```text
//! cuStreamGetCaptureInfo_v3    the graph being captured, and its frontier
//! cuGraphConditionalHandleCreate   a handle minted ON that graph
//!   <the predicate kernel, launched on the capturing stream>
//! cuStreamGetCaptureInfo_v3    the frontier again -- now the setter node
//! cuGraphAddNode_v2            CU_GRAPH_NODE_TYPE_CONDITIONAL, IF, size 1
//! cuStreamUpdateCaptureDependencies_v2   the capture continues BEHIND it
//! cuStreamBeginCaptureToGraph  the body, on a stream of its own
//!   <the region's launches>
//! cuStreamEndCapture           the body is closed, the parent runs on
//! ```
//!
//! # Why the driver API and not the runtime one
//!
//! Three of these calls have no single runtime spelling across the two ABIs
//! this crate builds for. `cudaGraphAddNode` keeps its name and changes ARITY
//! between `libcudart.so.12` and `.so.13` (five parameters against six) — the
//! exact hazard `kernels-cuda`'s manifest names as the reason the version is a
//! feature — and `cudaStreamGetCaptureInfo_v3` and
//! `cudaStreamUpdateCaptureDependencies_v2` exist under 12 and are gone under
//! 13. The `cu*` spellings `cuGraphAddNode_v2`,
//! `cuStreamUpdateCaptureDependencies_v2`, `cuStreamGetCaptureInfo_v3` and
//! `cuStreamBeginCaptureToGraph` are all present and identical under both. So
//! this module is driver-API from end to end, which is the same choice
//! [`capture_frontier`](super::graph::capture_frontier) already made next door
//! and for the same reason.
//!
//! # The body stream, and why there has to be one
//!
//! `cuStreamBeginCaptureToGraph` cannot be called on a stream that is already
//! capturing, and the parent capture is on the shell's main stream. So a
//! conditional body is recorded on a stream opened for it at load
//! ([`Context::open_conditional`](super::ctx::Context::open_conditional)) —
//! carrying its own cuBLAS handle and attached to the same scratch arena, so
//! that a body holding a projection is the same launch it would have been on
//! the main stream. Nothing runs on that stream: it exists to be captured on.
//!
//! # What is NOT here
//!
//! No `cudaGraphSetConditional`. That call is device-side and lives in
//! `kernels_cuda::graph`, which is the whole reason a conditional needs a
//! kernel at all; this module mints the handle it stores into and places the
//! node it steers.

use core::ffi::c_void;

use crate::error::{Fault, Result};

/// One conditional node, mid-capture: the handle the predicate kernel stores
/// into, the node itself, and the child graph its body is captured into.
///
/// **HELD BETWEEN `cond_begin` AND `cond_end` AND NOWHERE ELSE.** Every field
/// is owned by the driver and valid for the lifetime of the parent capture, so
/// this is a receipt rather than a resource: there is nothing to destroy, and
/// dropping one mid-capture leaves a conditional node with an empty body,
/// which is a graph that runs nothing rather than a graph that leaks.
#[derive(Debug, Clone, Copy)]
pub struct Conditional {
    /// `CUgraphConditionalHandle` — a `u64`, and the one argument the setter
    /// kernel takes that is not a pointer.
    pub handle: u64,
    /// The `CUgraphNode` this placed in the parent graph.
    pub node: *mut c_void,
    /// The `CUgraph` the driver minted for the body, which
    /// [`begin_body`] captures into.
    pub body: *mut c_void,
}

/// The graph a stream is capturing into, and the frontier the next node would
/// depend on.
///
/// [`capture_frontier`](super::graph::capture_frontier) answers the second
/// half of this; a conditional needs the first as well — a handle is minted ON
/// a graph — and asking twice would be two queries that could disagree.
#[cfg(feature = "_cuda")]
fn capture_info(
    stream: *mut c_void,
) -> Result<(
    cudarc::driver::sys::CUgraph,
    Vec<cudarc::driver::sys::CUgraphNode>,
)> {
    use cudarc::driver::sys as dr;

    let mut status = dr::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id: u64 = 0;
    let mut graph: dr::CUgraph = core::ptr::null_mut();
    let mut deps: *const dr::CUgraphNode = core::ptr::null();
    let mut dep_count: usize = 0;
    // SAFETY: every out-parameter is a live local, and the caller's contract
    // is that this thread began a capture on `stream`.
    let code = unsafe {
        let mut edges: *const dr::CUgraphEdgeData = core::ptr::null();
        dr::cuStreamGetCaptureInfo_v3(
            stream.cast(),
            &raw mut status,
            &raw mut id,
            &raw mut graph,
            &raw mut deps,
            &raw mut edges,
            &raw mut dep_count,
        )
    };
    said("cuStreamGetCaptureInfo_v3", code)?;
    if status != dr::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        return Err(Fault::Device {
            call: "cuStreamGetCaptureInfo_v3 (the stream is not capturing)",
            code: status as i32,
        });
    }
    if graph.is_null() {
        return Err(Fault::Device {
            call: "cuStreamGetCaptureInfo_v3 (an active capture with no graph)",
            code: 0,
        });
    }
    let frontier = if deps.is_null() || dep_count == 0 {
        Vec::new()
    } else {
        // SAFETY: the driver's array is `dep_count` handles long and lives
        // until the next capture-mutating call, which is after this copy.
        unsafe { core::slice::from_raw_parts(deps, dep_count) }.to_vec()
    };
    Ok((graph, frontier))
}

#[cfg(feature = "_cuda")]
fn said(call: &'static str, code: cudarc::driver::sys::CUresult) -> Result<()> {
    if code == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

/// Mint a conditional handle on the graph `stream` is capturing into.
///
/// `default` is the value the handle holds when no kernel sets it. **It is
/// `false` here and the caller may not ask for anything else**: a default of
/// `true` is a body that runs when the predicate kernel did not run, which is
/// a graph whose control flow is decided by whether a launch happened — and a
/// launch that did not happen is precisely the case a capture cannot
/// distinguish from one that decided `false`.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`] for a
/// stream that is not capturing or a mint the driver refused.
pub fn handle(stream: *mut c_void) -> Result<u64> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::driver::sys as dr;

        let (graph, _) = capture_info(stream)?;
        let mut ctx: dr::CUcontext = core::ptr::null_mut();
        // SAFETY: a live out-parameter; this thread bound the device.
        said("cuCtxGetCurrent", unsafe {
            dr::cuCtxGetCurrent(&raw mut ctx)
        })?;
        let mut handle: dr::CUgraphConditionalHandle = 0;
        // SAFETY: `graph` is the capture's own, `ctx` this thread's current
        // one, and the out-parameter is a live local.
        said("cuGraphConditionalHandleCreate", unsafe {
            dr::cuGraphConditionalHandleCreate(&raw mut handle, graph, ctx, 0, 0)
        })?;
        Ok(handle)
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

/// Place an `IF` node at the capture's current frontier and hand back its body
/// graph, leaving the capture depending on the node.
///
/// **CALLED AFTER THE PREDICATE KERNEL AND NOT BEFORE.** The setter's launch
/// is a node of the parent graph, and the frontier read here is what makes the
/// conditional depend on it; read the frontier first and the two are siblings,
/// which is a graph free to evaluate the handle before it was written.
///
/// # Errors
///
/// [`Fault::Runtimeless`], or [`Fault::Device`] for a frontier query, a node
/// the driver refused to add, or a dependency update it refused.
pub fn open(stream: *mut c_void, handle: u64) -> Result<Conditional> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::driver::sys as dr;

        let (graph, frontier) = capture_info(stream)?;
        let mut ctx: dr::CUcontext = core::ptr::null_mut();
        // SAFETY: a live out-parameter; this thread bound the device.
        said("cuCtxGetCurrent", unsafe {
            dr::cuCtxGetCurrent(&raw mut ctx)
        })?;

        // The driver POPULATES `phGraph_out` — it points into memory the
        // conditional node owns for its own lifetime — so what goes in is a
        // null the call overwrites, and reading it back is how the body graph
        // is learned. `size: 1` is the one body an `IF` has; the ELSE half
        // CUDA 12.8 grew would be `size: 2`, and P3 does not bake one.
        let mut params: dr::CUgraphNodeParams = unsafe { core::mem::zeroed() };
        params.type_ = dr::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        params.__bindgen_anon_1.conditional = dr::CUDA_CONDITIONAL_NODE_PARAMS {
            handle,
            type_: dr::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
            size: 1,
            phGraph_out: core::ptr::null_mut(),
            ctx,
        };

        let mut node: dr::CUgraphNode = core::ptr::null_mut();
        // SAFETY: `graph` is the capture's, the frontier is the array this
        // call is documented to take, and `params` is a live local the driver
        // writes `phGraph_out` back into.
        said("cuGraphAddNode_v2", unsafe {
            dr::cuGraphAddNode_v2(
                &raw mut node,
                graph,
                if frontier.is_empty() {
                    core::ptr::null()
                } else {
                    frontier.as_ptr()
                },
                core::ptr::null(),
                frontier.len(),
                &raw mut params,
            )
        })?;

        // SAFETY: the call above populated `phGraph_out` with an array of
        // `size` graphs, owned by the node and valid for its lifetime.
        let bodies = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if bodies.is_null() {
            return Err(Fault::Device {
                call: "cuGraphAddNode_v2 (a conditional node with no body graph)",
                code: 0,
            });
        }
        // SAFETY: `size` is 1, so element 0 is in bounds.
        let body = unsafe { *bodies };

        // **AND THE CAPTURE NOW HANGS OFF THE NODE.** `SET` and not `ADD`:
        // the launches after the bracket must depend on the conditional and
        // not on whatever the frontier held before it, or they are siblings of
        // the body and free to run beside it.
        let mut depend = [node];
        // SAFETY: `depend` is a live local of length 1 and `stream` is
        // capturing.
        said("cuStreamUpdateCaptureDependencies_v2", unsafe {
            dr::cuStreamUpdateCaptureDependencies_v2(
                stream.cast(),
                depend.as_mut_ptr(),
                core::ptr::null(),
                1,
                dr::CUstreamUpdateCaptureDependencies_flags::CU_STREAM_SET_CAPTURE_DEPENDENCIES
                    as u32,
            )
        })?;

        Ok(Conditional {
            handle,
            node: node.cast(),
            body: body.cast(),
        })
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, handle);
        Err(Fault::Runtimeless)
    }
}

/// Begin capturing `body` on a stream of its own — the child graph a
/// conditional node runs when its handle is set.
///
/// # Errors
///
/// [`Fault::Runtimeless`], or [`Fault::Device`] for a stream already capturing
/// or a graph the driver would not accept.
pub fn begin_body(stream: *mut c_void, body: *mut c_void) -> Result<()> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::driver::sys as dr;

        // RELAXED, and it is the one place this crate does not use
        // thread-local. The parent capture holds a thread-local restriction
        // already; a second thread-local begin on the same thread would be
        // asking the driver to nest a restriction it tracks per thread rather
        // than per stream. Nothing host-side happens between here and
        // `end_body` — the walk's dispatch is enqueue-only — so what the
        // stricter mode would be protecting is not on this path.
        //
        // SAFETY: `body` is the node's child graph, `stream` is the shell's
        // conditional-body stream, and no capture is active on it.
        said("cuStreamBeginCaptureToGraph", unsafe {
            dr::cuStreamBeginCaptureToGraph(
                stream.cast(),
                body.cast(),
                core::ptr::null(),
                core::ptr::null(),
                0,
                dr::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        })
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, body);
        Err(Fault::Runtimeless)
    }
}

/// Close the body capture. The graph it returns is the one
/// [`begin_body`] was handed, already owned by the node, so it is dropped
/// here rather than handed back.
///
/// # Errors
///
/// [`Fault::Runtimeless`], or [`Fault::Device`] when the body's capture was
/// invalidated — which is what a launch the body could not enqueue leaves
/// behind.
pub fn end_body(stream: *mut c_void) -> Result<()> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::driver::sys as dr;

        let mut out: dr::CUgraph = core::ptr::null_mut();
        // SAFETY: a live out-parameter, on the stream `begin_body` opened.
        said("cuStreamEndCapture (conditional body)", unsafe {
            dr::cuStreamEndCapture(stream.cast(), &raw mut out)
        })
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}
