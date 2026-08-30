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
    /// **THE DRIVER'S ARRAY OF BODY GRAPHS**, [`arms`](Conditional::arms) long
    /// — one for an `IF`, one per arm for a `SWITCH`.
    ///
    /// Private because it is a raw array and the only legal reading of it is
    /// [`body`](Conditional::body)'s: in bounds, and as a `CUgraph`. The
    /// driver owns the storage and keeps it alive for the node's lifetime,
    /// which is the parent capture's, which is longer than this receipt's.
    bodies: *mut *mut c_void,
    /// How many bodies the node has: `1` for an `IF`, the arm count for a
    /// `SWITCH`.
    pub arms: u32,
}

impl Conditional {
    /// The child graph of one arm, or `None` for an index this node has no
    /// body for.
    ///
    /// An `IF` has exactly one and it is `body(0)`. A `SWITCH` has one per
    /// arm, in the arm order `Def::Merge` states and `model_exec::fire::walk`
    /// announces — which is why the recorder can hand `cond_arm`'s own number
    /// straight to this.
    #[must_use]
    pub fn body(&self, arm: u32) -> Option<*mut c_void> {
        if self.bodies.is_null() || arm >= self.arms {
            return None;
        }
        // SAFETY: `bodies` is the driver's array of `arms` graphs, populated
        // by `cuGraphAddNode_v2` and valid for the node's lifetime; `arm` was
        // just bounds-checked against the `size` that call was given.
        Some(unsafe { *self.bodies.add(arm as usize) })
    }
}

/// Which flavour of conditional node to place, and how many bodies it has.
///
/// **ONE ENUM RATHER THAN TWO FUNCTIONS**, because the two differ in exactly
/// two fields of one struct — `type_` and `size` — and everything around them
/// (the frontier read, the context, the dependency update) is the same
/// sequence for the same reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `CU_GRAPH_COND_TYPE_IF`: one body, taken when the handle is non-zero.
    If,
    /// `CU_GRAPH_COND_TYPE_SWITCH`: `arms` bodies, and the handle holds the
    /// INDEX of the one that runs. An index at or past `arms` runs none of
    /// them, which is the driver's own rule and is what makes a group with no
    /// live arm expressible without a store.
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

    /// **THE VALUE THE HANDLE HOLDS WHEN NOTHING STORES INTO IT**, and each
    /// kind's is the one that runs nothing.
    ///
    /// For an `IF` that is `0` — false, and the argument for it is that a
    /// launch which did not happen is indistinguishable from one that decided
    /// no, so the two must mean the same thing. For a `SWITCH` it is `arms`,
    /// which is past the last body: a group whose every arm stood down stores
    /// nothing at all, and "nothing stored" has to be "nothing runs" or the
    /// empty fire would silently take arm 0.
    #[must_use]
    pub const fn quiescent(self) -> u32 {
        match self {
            Kind::If => 0,
            Kind::Switch { arms } => arms,
        }
    }
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
/// **THE DEFAULT LAUNCH VALUE IS THE KIND'S QUIESCENT ONE AND THE CALLER MAY
/// NOT ASK FOR ANOTHER** — see [`Kind::quiescent`]. The whole argument is that
/// a predicate kernel which did not run must be indistinguishable from one
/// that decided to run nothing, because a capture cannot tell those apart: an
/// `IF` defaults to false and a `SWITCH` to an index past its last body.
///
/// **AND `CU_GRAPH_COND_ASSIGN_DEFAULT` IS WHAT MAKES THE DEFAULT MEAN THAT ON
/// EVERY LAUNCH AND NOT JUST THE FIRST.** Without the flag a handle is written
/// once at creation and then KEEPS whatever the last store put in it, across
/// launches of the same exec — which is invisible for a predicate that always
/// stores (an `IF`'s setter writes 0 or 1 either way) and silently wrong for
/// one that stores only when it has something to say. `set_switch` is exactly
/// that: it stores its arm's index only if its arm is live, so a fire where no
/// arm is live would replay the PREVIOUS fire's arm. Measured, on the gate's
/// fourth launch, before the flag went on. With it, the driver re-applies the
/// default at every launch before any node runs, and "nobody stored" is
/// "nothing runs" every time.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`] for a
/// stream that is not capturing or a mint the driver refused.
pub fn handle(stream: *mut c_void, kind: Kind) -> Result<u64> {
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
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, kind);
        Err(Fault::Runtimeless)
    }
}

/// Place a conditional node at the capture's current frontier and hand back
/// its body graphs, leaving the capture depending on the node.
///
/// **CALLED AFTER THE PREDICATE KERNELS AND NOT BEFORE.** Each setter's launch
/// is a node of the parent graph, and the frontier read here is what makes the
/// conditional depend on all of them; read the frontier first and they are
/// siblings, which is a graph free to evaluate the handle before it was
/// written. A `SWITCH` launches one setter per arm and the argument is the
/// same one `arms` times.
///
/// # Errors
///
/// [`Fault::Runtimeless`], or [`Fault::Device`] for a frontier query, a node
/// the driver refused to add, or a dependency update it refused.
pub fn open(stream: *mut c_void, handle: u64, kind: Kind) -> Result<Conditional> {
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
        // null the call overwrites, and reading it back is how the body graphs
        // are learned. `size` is the body count: `1` for the one body an `IF`
        // has (the ELSE half CUDA 12.8 grew would be `2`, and P3 does not bake
        // one), and the arm count for a `SWITCH`.
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

        // SAFETY: the call above populated `phGraph_out` with an array of
        // `size` graphs, owned by the node and valid for its lifetime.
        let bodies = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if bodies.is_null() {
            return Err(Fault::Device {
                call: "cuGraphAddNode_v2 (a conditional node with no body graph)",
                code: 0,
            });
        }

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
            bodies: bodies.cast(),
            arms: kind.size(),
        })
    }
    #[cfg(not(feature = "_cuda"))]
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
