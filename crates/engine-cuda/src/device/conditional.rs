use core::ffi::c_void;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy)]
pub struct Conditional {
    pub handle: u64,
    pub node: *mut c_void,
    bodies: *mut *mut c_void,
    pub arms: u32,
}

impl Conditional {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    If,
    Switch { arms: u32 },
}

impl Kind {
    #[must_use]
    pub const fn size(self) -> u32 {
        match self {
            Kind::If => 1,
            Kind::Switch { arms } => arms,
        }
    }

    #[must_use]
    pub const fn quiescent(self) -> u32 {
        match self {
            Kind::If => 0,
            Kind::Switch { arms } => arms,
        }
    }
}

#[cfg(feature = "cuda")]
fn capture_info(
    stream: *mut c_void,
) -> Result<(
    crate::device::dr::CUgraph,
    Vec<crate::device::dr::CUgraphNode>,
)> {
    use crate::device::dr;

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
fn said(call: &'static str, code: crate::device::dr::CUresult) -> Result<()> {
    if code == crate::device::dr::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

/// HIP graphs have no conditional node. The ROCm engine registers
/// conditionals as priced past any launch they could skip
/// (`chip::registry`), so the compiler lowers every region to
/// `AlwaysLaunch`; a profile that prices them back in reaches these, which
/// refuse.
#[cfg(feature = "rocm")]
fn absent(what: &str) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what}: HIP graphs have no conditional node, so a region lowered to `If` or \
             `Switch` has nowhere to go on ROCm; leave `compiler.cond_fixed_us` at this \
             engine's default so every region lowers to `AlwaysLaunch`"
        ),
    }
}

pub fn handle(stream: *mut c_void, kind: Kind) -> Result<u64> {
    #[cfg(feature = "rocm")]
    {
        let _ = (stream, kind);
        Err(absent("a conditional handle"))
    }
    #[cfg(feature = "cuda")]
    {
        use crate::device::dr;

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
    #[cfg(not(feature = "device"))]
    {
        let _ = (stream, kind);
        Err(Fault::Runtimeless)
    }
}

pub fn open(stream: *mut c_void, handle: u64, kind: Kind) -> Result<Conditional> {
    #[cfg(feature = "rocm")]
    {
        let _ = (stream, handle, kind);
        Err(absent("a conditional node"))
    }
    #[cfg(feature = "cuda")]
    {
        use crate::device::dr;

        let (graph, frontier) = capture_info(stream)?;
        let mut ctx: dr::CUcontext = core::ptr::null_mut();
        // SAFETY: a live out-parameter; this thread bound the device.
        said("cuCtxGetCurrent", unsafe {
            dr::cuCtxGetCurrent(&raw mut ctx)
        })?;

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
    #[cfg(not(feature = "device"))]
    {
        let _ = (stream, handle, kind);
        Err(Fault::Runtimeless)
    }
}

pub fn begin_body(stream: *mut c_void, body: *mut c_void) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        let _ = (stream, body);
        Err(absent("a conditional body"))
    }
    #[cfg(feature = "cuda")]
    {
        use crate::device::dr;

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
    #[cfg(not(feature = "device"))]
    {
        let _ = (stream, body);
        Err(Fault::Runtimeless)
    }
}

pub fn end_body(stream: *mut c_void) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        let _ = stream;
        Err(absent("a conditional body"))
    }
    #[cfg(feature = "cuda")]
    {
        use crate::device::dr;

        let mut out: dr::CUgraph = core::ptr::null_mut();
        // SAFETY: a live out-parameter, on the stream `begin_body` opened.
        said("cuStreamEndCapture (conditional body)", unsafe {
            dr::cuStreamEndCapture(stream.cast(), &raw mut out)
        })
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}
