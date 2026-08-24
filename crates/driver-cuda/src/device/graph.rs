//! CUDA graphs, including conditional nodes. cudarc has no wrapper, so
//! [`Graph::add_conditional_if`] builds the `cudaGraphNodeParams.conditional`
//! union arm directly. `cudaGraphSetConditional` is deliberately absent from
//! the bindings since it only runs on the GPU.

use std::marker::PhantomData;

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

// ── THE UNIONIZED SUPERGRAPH STOOD HERE ────────────────────────────
//
// 600 lines: `PredicateWord` and `PeelWindowWord` (the device words a fire
// armed its guards and its peel split from), the eleven `SLOT_*` wires that
// mirrored `model_ir::trace::GuardPred`, `Cond`/`Switch`, and
// `SupergraphBuilder` — one captured graph per (R, N) bucket, branching on
// conditional nodes a kernel armed per fire so a replay needed no host
// round-trip.
//
// It was the LEGACY WALK's guard story and nothing else's. A lowering under
// `GuardMode::Union` kept both arms of every guard and let the device pick;
// `model_compiler::program::bound` answers the same question by building ONE
// PROGRAM PER LANE, chosen on the host by the fact word before a byte is
// issued. There is no predicate for a conditional node to read.
//
// WHAT STAYS ABOVE: `Graph`, `GraphExec`, `ConditionalIf`,
// `ConditionalSwitch` — the CUDA graph API port itself, which is a device
// capability and not a lowering's opinion, and which `Allocator::
// begin_capture` still builds against. A capture of the eager baker walk
// starts from those and needs none of what was deleted here.
//
// `fire::supergraph`'s two arming launchers and `tests/gpu_supergraph.rs`
// went with it.
