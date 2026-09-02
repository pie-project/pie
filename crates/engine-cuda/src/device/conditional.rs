//! Conditional graph nodes placed inside a running capture. Uses the
//! driver-API (`cu*`) spellings throughout rather than the runtime ones:
//! `cudaGraphAddNode`'s arity differs between libcudart 12 and 13, and some
//! of the calls needed here don't exist under both — the `cu*` equivalents
//! are stable across both. A conditional's body is captured on a stream
//! opened for it at load (not the main stream), since
//! `cuStreamBeginCaptureToGraph` cannot run on a stream already capturing.
//! Device-side predicate stores (`cudaGraphSetConditional`) live in
//! `kernels_cuda::graph`; this module only mints the handle and places the node.

use core::ffi::c_void;

use crate::error::{Fault, Result};

/// One conditional node, mid-capture. Owned entirely by the driver for the
/// parent capture's lifetime, so this is a receipt, not a resource: valid
/// only between `cond_begin` and `cond_end`, nothing to destroy.
#[derive(Debug, Clone, Copy)]
pub struct Conditional {
    /// `CUgraphConditionalHandle`: the one setter-kernel argument that isn't a pointer.
    pub handle: u64,
    /// The `CUgraphNode` placed in the parent graph.
    pub node: *mut c_void,
    // Driver's array of body graphs, `arms` long (one per SWITCH arm, or one
    // for IF). Only legal read is `body()` (bounds-checked, cast to CUgraph).
    bodies: *mut *mut c_void,
    /// How many bodies the node has: `1` for an `IF`, the arm count for a `SWITCH`.
    pub arms: u32,
}

impl Conditional {
    /// The child graph of one arm, or `None` for an index this node has no
    /// body for. An `IF` has exactly one (`body(0)`); a `SWITCH` has one per
    /// arm in `Def::Merge`'s arm order.
    #[must_use]
    pub fn body(&self, arm: u32) -> Option<*mut c_void> {
        if self.bodies.is_null() || arm >= self.arms {
            return None;
        }
        // SAFETY: `bodies` is the driver's array of `arms` graphs from
        // `cuGraphAddNode_v2`, valid for the node's lifetime; `arm` is bounds-checked.
        Some(unsafe { *self.bodies.add(arm as usize) })
    }
}

/// Which flavour of conditional node to place, and how many bodies it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `CU_GRAPH_COND_TYPE_IF`: one body, taken when the handle is non-zero.
    If,
    /// `CU_GRAPH_COND_TYPE_SWITCH`: `arms` bodies; handle holds the index of
    /// the one that runs. An index at or past `arms` runs none (driver rule).
    Switch { arms: u32 },
}

impl Kind {
    /// How many bodies the node is asked for.
    #[must_use]
    pub const fn size(self) -> u32 {
        match self {
            Kind::If => 1,
            Kind::Switch { arms } => arms,
        }
    }

    /// The handle value that means "nothing stores, nothing runs": `0` for
    /// `IF`, `arms` (past the last body) for `SWITCH` — otherwise an empty
    /// fire would silently take arm 0.
    #[must_use]
    pub const fn quiescent(self) -> u32 {
        match self {
            Kind::If => 0,
            Kind::Switch { arms } => arms,
        }
    }
}

/// The graph a stream is capturing into, and the frontier the next node would
/// depend on. A conditional needs the graph itself (a handle is minted on
/// it), not just the frontier `capture_frontier` reads.
#[cfg(feature = "cuda")]
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
    // SAFETY: every out-parameter is a live local; caller's contract is that
    // this thread began a capture on `stream`.
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
        // SAFETY: driver's array is `dep_count` handles long, valid until
        // the next capture-mutating call (after this copy).
        unsafe { core::slice::from_raw_parts(deps, dep_count) }.to_vec()
    };
    Ok((graph, frontier))
}

#[cfg(feature = "cuda")]
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
/// The default launch value is fixed to the kind's quiescent one (see
/// [`Kind::quiescent`]), via `CU_GRAPH_COND_ASSIGN_DEFAULT`. That flag makes
/// the driver re-apply the default at every launch, not just the first —
/// without it a handle keeps whatever the last store put in it, which is
/// wrong for a predicate (like `set_switch`) that stores only when live.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`] for a
/// stream that is not capturing or a mint the driver refused.
pub fn handle(stream: *mut c_void, kind: Kind) -> Result<u64> {
    #[cfg(feature = "cuda")]
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
            dr::cuGraphConditionalHandleCreate(
                &raw mut handle,
                graph,
                ctx,
                kind.quiescent(),
                dr::CU_GRAPH_COND_ASSIGN_DEFAULT,
            )
        })?;
        Ok(handle)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, kind);
        Err(Fault::Runtimeless)
    }
}

/// Place a conditional node at the capture's current frontier and hand back
/// its body graphs, leaving the capture depending on the node.
///
/// Must be called after the predicate kernels, not before: the frontier read
/// here is what makes the conditional depend on their launches. Reading it
/// first would make them siblings, free to evaluate the handle unwritten.
///
/// # Errors
///
/// [`Fault::Runtimeless`], or [`Fault::Device`] for a frontier query, a node
/// the driver refused to add, or a dependency update it refused.
pub fn open(stream: *mut c_void, handle: u64, kind: Kind) -> Result<Conditional> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

        let (graph, frontier) = capture_info(stream)?;
        let mut ctx: dr::CUcontext = core::ptr::null_mut();
        // SAFETY: a live out-parameter; this thread bound the device.
        said("cuCtxGetCurrent", unsafe {
            dr::cuCtxGetCurrent(&raw mut ctx)
        })?;

        // Driver populates `phGraph_out` (memory the conditional node owns);
        // pass null and read it back afterward to learn the body graphs.
        let mut params: dr::CUgraphNodeParams = unsafe { core::mem::zeroed() };
        params.type_ = dr::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        params.__bindgen_anon_1.conditional = dr::CUDA_CONDITIONAL_NODE_PARAMS {
            handle,
            type_: match kind {
                Kind::If => dr::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
                Kind::Switch { .. } => dr::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_SWITCH,
            },
            size: kind.size(),
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

        // SAFETY: call above populated `phGraph_out` with an array of `size`
        // graphs, owned by the node, valid for its lifetime.
        let bodies = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if bodies.is_null() {
            return Err(Fault::Device {
                call: "cuGraphAddNode_v2 (a conditional node with no body graph)",
                code: 0,
            });
        }

        // SET, not ADD: launches after this must depend on the conditional
        // node, not on whatever the frontier held before it.
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
            bodies: bodies.cast(),
            arms: kind.size(),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, handle, kind);
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
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

        // RELAXED mode: the parent capture already holds a thread-local
        // restriction, and this call is on the same thread, so the stricter
        // mode would conflict rather than protect anything here.
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
    #[cfg(not(feature = "cuda"))]
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
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

        let mut out: dr::CUgraph = core::ptr::null_mut();
        // SAFETY: a live out-parameter, on the stream `begin_body` opened.
        said("cuStreamEndCapture (conditional body)", unsafe {
            dr::cuStreamEndCapture(stream.cast(), &raw mut out)
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}
