//! CUDA graphs, including the conditional nodes `batch/supergraph.cu` builds.
//!
//! # The conditional-node question, settled
//!
//! Conditional graph nodes (CUDA 12.3+) are how the shell puts an `if` inside
//! a captured graph, and they are the most exotic thing it asks of CUDA. They
//! have no wrapper in cudarc at any level -- no safe type, no `result` helper,
//! no example, no test. What they do have is complete `sys` bindings, which is
//! all that was ever needed:
//!
//! | symbol | `cudarc::runtime::sys` |
//! |---|---|
//! | `cudaGraphConditionalHandle` | type alias, `c_ulonglong` |
//! | `cudaGraphConditionalHandleCreate` | gated `cuda-12030`+ |
//! | `cudaGraphNodeTypeConditional` | `cudaGraphNodeType` discriminant 13 |
//! | `cudaGraphCondTypeIf` | `cudaGraphConditionalNodeType` discriminant 0 |
//! | `cudaConditionalNodeParams` | `{ handle, type_, size, phGraph_out }` |
//! | `cudaGraphNodeParams.conditional` | union arm |
//!
//! [`Graph::add_conditional_if`] is that sequence, once, behind a signature
//! that cannot get the union arm wrong.
//!
//! # What cannot come to Rust, and why that is not a gap
//!
//! `cudaGraphSetConditional` -- the call that sets the predicate -- is absent
//! from the bindings, and correctly so: it is a `__device__` function. In the
//! C++ shell it is called from inside `supergraph_set_cond_kernel`, a
//! `__global__` that runs on the GPU and flips the branch for the next
//! iteration. Device code is nvcc's by definition, so that kernel stays a
//! `.cu` no matter how much of the host side moves. It belongs beside the
//! graph that uses it, not in `kernels-cuda`, because its argument is a
//! conditional handle -- a shell object -- rather than a tensor.

use std::marker::PhantomData;

use cudarc::runtime::sys::{
    cudaConditionalNodeParams, cudaGraphConditionalHandle,
    cudaGraphConditionalHandleCreate, cudaGraphConditionalNodeType, cudaGraphDestroy,
    cudaGraphExecDestroy, cudaGraphExecKernelNodeSetParams, cudaGraphExec_t,
    cudaGraphInstantiate, cudaGraphKernelNodeGetParams, cudaGraphLaunch,
    cudaGraphConditionalHandleFlags, cudaGraphNodeParams, cudaGraphNodeType, cudaGraphNode_t,
    cudaGraphUpload, cudaGraph_t, cudaKernelNodeParams,
};

use crate::cuda::stream::StreamRef;
use crate::error::{Error, Result, check_rt, ignore_in_drop};

/// `cudaGraphAddNode`, spelled so that one binary works on one runtime.
///
/// The symbol named `cudaGraphAddNode` **changed signature between CUDA major
/// versions**, and the two are not interchangeable:
///
/// | runtime | `cudaGraphAddNode` | `cudaGraphAddNode_v2` |
/// |---|---|---|
/// | `libcudart.so.12` | 5 args | 6 args (adds edge data) |
/// | `libcudart.so.13` | **6 args** | absent |
///
/// Because this crate resolves CUDA symbols by name at runtime, a build
/// configured for one major version that loads the other calls a six-parameter
/// function with five arguments. The sixth register holds whatever was left in
/// it, the driver dereferences that as `nodeParams`, and the process dies with
/// a segfault far from the cause. It is not a compile error and it is not a
/// CUDA error code; only hardware shows it.
///
/// So the call is routed through the binding that matches the runtime this
/// build targets, and [`crate::cuda::Device::bind`] refuses to start when the
/// runtime it finds disagrees.
#[cfg(feature = "cuda-12")]
pub(super) unsafe fn add_node(
    node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    deps: *const cudaGraphNode_t,
    num_deps: usize,
    params: *mut cudaGraphNodeParams,
) -> cudarc::runtime::sys::cudaError {
    // The 6-arg form, which on CUDA 12 is the `_v2` symbol. `supergraph.cu`
    // reaches the same entry point, because the CUDA 12 headers `#define`
    // `cudaGraphAddNode` to it.
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
    ///
    /// # Safety
    ///
    /// `raw` must be a live `cudaGraph_t` that nothing else will destroy.
    /// Produced by [`crate::cuda::CaptureScope::end`], which is the only
    /// caller that should exist.
    pub(crate) const unsafe fn from_raw(raw: cudaGraph_t) -> Self {
        Self { raw }
    }

    /// The raw handle.
    pub const fn as_raw(&self) -> cudaGraph_t {
        self.raw
    }

    /// Add an `if` node whose body is a child graph, returning the handle that
    /// selects it.
    ///
    /// This is the host half of `batch/supergraph.cu`'s
    /// `build_supergraph_conditional`. The `handle` in the returned
    /// [`ConditionalIf`] is what a device-side `cudaGraphSetConditional` call
    /// writes to decide whether the body runs on the next launch.
    ///
    /// `default_run` is an `Option` rather than a `bool` because CUDA has
    /// three states here, not two, and the difference is invisible until it
    /// misbehaves:
    ///
    /// * `None` -- the handle carries no default. Every launch **must** have a
    ///   device-side `cudaGraphSetConditional` before the node is reached, or
    ///   the predicate holds whatever the previous launch left in it. This is
    ///   what `supergraph.cu` does, and it passes `0` for both the default and
    ///   the flags precisely because its kernel always writes the value first.
    /// * `Some(v)` -- `v` is written to the handle at the start of every graph
    ///   execution, and a device-side set later in the graph still overrides
    ///   it.
    ///
    /// The distinction is the `cudaGraphCondAssignDefault` flag. Passing a
    /// default value *without* that flag -- which is what an earlier version
    /// of this function did -- means CUDA ignores the value entirely, so a
    /// caller asking for "default off" silently gets "whatever was there".
    ///
    /// `deps` are the nodes this one waits on -- empty for a root.
    pub fn add_conditional_if(
        &mut self,
        deps: &[cudaGraphNode_t],
        default_run: Option<bool>,
    ) -> Result<ConditionalIf<'_>> {
        let (default_value, flags) = match default_run {
            Some(v) => (u32::from(v), cudaGraphConditionalHandleFlags::cudaGraphCondAssignDefault as u32),
            None => (0, 0),
        };
        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe {
                cudaGraphConditionalHandleCreate(&mut handle, self.raw, default_value, flags)
            },
            "cudaGraphConditionalHandleCreate",
        )?;

        // `phGraph_out` is not an out-*parameter* but an out-*pointer*: CUDA
        // allocates the child-graph array itself and overwrites this field
        // with the address of it. Pointing it at caller storage, as an earlier
        // version of this function did, is silently useless -- the pointer is
        // replaced, the caller's variable is never written, and the body comes
        // back null.
        //
        // It is left zeroed here and read back after the call, which is what
        // `supergraph.cu` does. The array is owned by the parent graph and
        // lives as long as the conditional node; that ownership is what the
        // borrow on `ConditionalIf` records.
        let mut params: cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = cudaGraphNodeType::cudaGraphNodeTypeConditional;
        params.__bindgen_anon_1.conditional = cudaConditionalNodeParams {
            handle,
            type_: cudaGraphConditionalNodeType::cudaGraphCondTypeIf,
            // One body graph. `cudaGraphCondTypeIf` with `size = 1` is the
            // if-without-else form; the if/else form is `size = 2` and would
            // need a second body out-pointer, so it gets its own constructor
            // when something needs it rather than an `Option` here.
            size: 1,
            phGraph_out: std::ptr::null_mut(),
        };

        let mut node: cudaGraphNode_t = std::ptr::null_mut();
        check_rt(
            unsafe {
                add_node(&raw mut node, self.raw, deps.as_ptr(), deps.len(), &raw mut params)
            },
            "cudaGraphAddNode",
        )?;

        // SAFETY: on success CUDA has pointed `phGraph_out` at an array of
        // `size` graphs; `size` is 1 here.
        let out = unsafe { params.__bindgen_anon_1.conditional.phGraph_out };
        if out.is_null() {
            return Err(Error::invalid(
                "cudaGraphAddNode",
                "the driver returned no body graph for the conditional node",
            ));
        }
        // SAFETY: `out` is non-null and has at least one element.
        let body = unsafe { *out };

        Ok(ConditionalIf { node, body, handle, _parent: PhantomData })
    }

    /// A SWITCH node: one conditional that branches on an INTEGER index.
    ///
    /// `.wiki/driver/graph.md` §6.1. Every `GuardPred` is a boolean IF
    /// today and they NEST, which costs depth and arm pairs on any axis
    /// with more than two options — attention is
    /// `plain / capture / custom / xqa`, four bodies reached through three
    /// nested pairs. `cudaGraphCondTypeSwitch` reaches all four in one
    /// node, selected by the index the device writes into the handle.
    ///
    /// It is also what makes the merged Decode/Prefill graph achievable
    /// rather than aspirational: `cudaGraphExecUpdate` cannot ADD nodes,
    /// so two topologies cannot be one updatable exec — but a SWITCH
    /// selects among CHILD GRAPHS, and a child graph is a topology.
    ///
    /// `bodies` is how many arms; the device writes `0..bodies` and an
    /// out-of-range value runs none. There is no default-value flag on a
    /// switch — the handle is always written by the arming kernel, which
    /// is the same discipline `add_conditional_if` uses with `None`.
    ///
    /// # Errors
    ///
    /// If the handle or the node cannot be created, or if the driver
    /// returns no body array.
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
                add_node(&raw mut node, self.raw, deps.as_ptr(), deps.len(), &raw mut params)
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
        // SAFETY: on success CUDA has pointed `phGraph_out` at an array of
        // exactly `bodies` graphs, owned by the parent.
        let graphs = unsafe { std::slice::from_raw_parts(out, bodies as usize) }.to_vec();
        Ok(ConditionalSwitch { node, bodies: graphs, handle, _parent: PhantomData })
    }

    /// Instantiate into a launchable graph.
    ///
    /// The trailing `0` is the flags word. CUDA 12 replaced the older
    /// error-node/log-buffer out-parameters with it, so there is nowhere left
    /// for a per-node diagnostic to be written -- a bad graph reports itself
    /// through the return code alone.
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
///
/// Borrows the parent graph, because the body graph is the parent's to
/// destroy. Nothing here has a `Drop`: destroying the body separately is
/// exactly the mistake the borrow is here to prevent.
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

    /// The body graph, to populate with the work the branch guards.
    ///
    /// Owned by the parent graph; do not destroy it.
    pub const fn body(&self) -> cudaGraph_t {
        self.body
    }

    /// The conditional handle, to hand to the device-side kernel that sets the
    /// predicate.
    ///
    /// This value is the argument of `cudaGraphSetConditional`, which is a
    /// `__device__` function -- so its consumer is a `.cu`, reached through
    /// FFI, and never a Rust call.
    pub const fn handle(&self) -> cudaGraphConditionalHandle {
        self.handle
    }
}

/// A SWITCH node added to a [`Graph`] and the handle that selects which of
/// its bodies runs. See [`Graph::add_conditional_switch`].
///
/// Borrows the parent graph for the same reason [`ConditionalIf`] does:
/// the body graphs are the parent's to destroy.
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

    /// One body graph, by index. Owned by the parent graph; do not
    /// destroy it.
    #[must_use]
    pub fn body(&self, index: usize) -> Option<cudaGraph_t> {
        self.bodies.get(index).copied()
    }

    /// How many bodies this switch selects among.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Whether the switch has no bodies. Never true — the constructor
    /// refuses a zero-body switch — but clippy asks for the pair.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The conditional handle. Its consumer writes an INDEX rather than a
    /// boolean; see `pie_supergraph_set_switch` in `csrc/supergraph.cu`.
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

    /// Upload to the device without launching, so the first launch does not
    /// pay for it.
    pub fn upload(&self, stream: StreamRef<'_>) -> Result<()> {
        check_rt(unsafe { cudaGraphUpload(self.raw, stream.as_raw()) }, "cudaGraphUpload")
    }

    /// Launch onto `stream`.
    pub fn launch(&self, stream: StreamRef<'_>) -> Result<()> {
        check_rt(unsafe { cudaGraphLaunch(self.raw, stream.as_raw()) }, "cudaGraphLaunch")
    }

    /// Retune ONE node's launch rectangle on this instantiated graph,
    /// without recapturing.
    ///
    /// `.wiki/driver/graph.md` §6.2, and the axis it removes from
    /// [`crate::model::supergraph::BucketKey`]. Grid and block dims are
    /// updatable on an instantiated graph; the kernel function pointer and
    /// the topology are not. For two fires of the same `(model)` differing
    /// only in row count the launch LIST is identical and only the
    /// rectangles move — topology preserved, so the update is legal. It
    /// costs tens of microseconds per graph against a recapture's
    /// milliseconds, and it avoids the alternative of making every kernel
    /// `_devwin`-shaped and always launching the maximum grid, which pays
    /// empty warps on every small fire.
    ///
    /// The existing parameters are read back off the NODE (the recorded
    /// graph's, not the exec's) so that the function pointer, the shared
    /// memory and the argument array are carried over untouched. Only the
    /// grid moves.
    ///
    /// # Errors
    ///
    /// If the node is not a kernel node, or if the update is rejected —
    /// which is how CUDA reports a change the instantiated graph cannot
    /// absorb, and is the caller's cue to recapture rather than to
    /// continue.
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

    // These do not call CUDA. What they pin is that the sys vocabulary the
    // conditional path is written in exists and means what `supergraph.cu`
    // assumes -- which is the fact the whole design decision rested on, and
    // therefore the fact worth failing the build over if a cudarc bump or a
    // CUDA-version feature change takes it away.

    #[test]
    fn the_conditional_vocabulary_exists_and_matches_the_cuda_headers() {
        assert_eq!(cudaGraphNodeType::cudaGraphNodeTypeConditional as u32, 13);
        assert_eq!(cudaGraphConditionalNodeType::cudaGraphCondTypeIf as u32, 0);
        // The handle is CUDA's `unsigned long long`, which is what makes it
        // passable to a device kernel as a plain scalar.
        assert_eq!(
            std::mem::size_of::<cudaGraphConditionalHandle>(),
            std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn the_conditional_union_arm_is_populated_the_way_supergraph_cu_populates_it() {
        // Builds exactly the `cudaGraphNodeParams` that `add_conditional_if`
        // hands to `cudaGraphAddNode`, and reads it back, without a device.
        // A union arm written and read through different fields is a silent
        // corruption, so it is worth one test.
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
        // The body graph belongs to the parent. If this type ever grows a
        // destructor it will be double-freeing the parent's child graph.
        assert!(!std::mem::needs_drop::<ConditionalIf<'_>>());
    }
}
