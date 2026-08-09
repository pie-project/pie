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
//! # What is device code here, and why that is not nvcc's
//!
//! `cudaGraphSetConditional` -- the call that sets the predicate -- is absent
//! from the bindings, and correctly so: it is a `__device__` function. It is
//! called from inside `supergraph_set_cond`, a `__global__` that runs on the
//! GPU and flips the branch for the next iteration.
//!
//! That kernel used to be `csrc/supergraph.cu`, compiled by nvcc into its own
//! archive, under a header that called it *"the one device function the
//! supergraph cannot express in Rust"* and a `build.rs` comment that said
//! *"this needs nvcc"*. **Both were measured and are false.** `__device__` is
//! a fact about where the call RUNS, not about which frontend may emit it:
//! NVRTC compiles this kernel, emits `.extern .func cudaGraphSetConditional`
//! plus a `call.uni`, and the driver resolves the symbol at
//! `cuModuleLoadData` -- which it must, because the symbol is declared
//! `extern __device__ __cudart_builtin__` with no definition in any toolkit
//! header and no definition in `libcudadevrt.a`. nvcc's PTX for the same call
//! was the same `.extern .func`; the two frontends share `cicc`.
//!
//! So the device text is a JIT unit like every other:
//! `kernels-cuda-new/csrc/src/graph/supergraph.cuh`, fired by
//! [`crate::fire::supergraph`], whose header carries the whole measurement
//! and the one thing it does not show. It lives outside `kernels-cuda` for
//! the reason it always did -- its argument is a conditional handle, a SHELL
//! object, rather than a tensor -- which is now spelled as a family of its
//! own, `graph`, rather than as a `.cu` beside this file.

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
/// build targets, and [`crate::device::Device::bind`] refuses to start when the
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
    /// Produced by [`crate::device::CaptureScope::end`], which is the only
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

        Ok(ConditionalIf {
            node,
            body,
            handle,
            _parent: PhantomData,
        })
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
        // SAFETY: on success CUDA has pointed `phGraph_out` at an array of
        // exactly `bodies` graphs, owned by the parent.
        let graphs = unsafe { std::slice::from_raw_parts(out, bodies as usize) }.to_vec();
        Ok(ConditionalSwitch {
            node,
            bodies: graphs,
            handle,
            _parent: PhantomData,
        })
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
    /// boolean; see [`crate::fire::supergraph::set_switch`].
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

    /// Retune ONE node's launch rectangle on this instantiated graph,
    /// without recapturing.
    ///
    /// `.wiki/driver/graph.md` §6.2, and the axis it removes from
    /// [`crate::fire::recordings::BucketKey`]. Grid and block dims are
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

// ── The unionized supergraph, built on the conditional node above ──

// The unionized supergraph's capture scaffolding.
//
// This is the Rust port of `driver-cuda/csrc/src/batch/supergraph.{cu,hpp}`
// — S2 of the supergraph ladder, and the piece [`crate::device::graph`]'s
// conditional-node primitive was ported to serve.
//
// # What a supergraph is, and what it is not
//
// ONE captured CUDA graph per (R, N) bucket whose attachment branches — the
// declared trace's `GuardPred` vocabulary — are conditional `if` nodes. The
// predicates live in a DEVICE-resident word ([`PredicateWord`]) that the
// replay path updates per fire; a graph-embedded kernel reads a slot and
// arms the conditional handle, so a replay takes the fire's arms with no
// host round-trip and no recapture.
//
// That is not batching. Batching amortises one program over many rows; this
// amortises many *programs* over one capture. Concurrent requests that are
// structurally distinct — differing in hook attachment, mask kind,
// correction arm, depth, LoRA rank — fold into one conditional graph, so the
// operators they share execute exactly once.
//
// # The capture-time dance
//
// Conditional nodes are added to whichever graph is currently capturing, and
// their bodies are filled by capturing a DIFFERENT stream into the body
// graph. So the builder keeps a stack:
//
// * [`SupergraphBuilder::open_cond`] — create the handle on the capturing
//   graph (`cudaStreamGetCaptureInfo` answers for whichever stream is
//   capturing, at any nesting depth), launch the set-cond kernel so it
//   becomes the node's upstream dependency, then insert the conditional node
//   with the running deps;
// * [`SupergraphBuilder::begin_body`] — `cudaStreamBeginCaptureToGraph` on
//   the next pooled depth stream;
// * [`SupergraphBuilder::end_body`] — end that stream's capture;
// * [`SupergraphBuilder::close_cond`] — collapse the outer stream's capture
//   dependencies onto the conditional node, so post-branch work follows the
//   whole branch rather than racing it.
//
// Guard nesting maps to a stack of body captures over the depth-indexed
// stream pool. Nothing here knows about models.

#[cfg(feature = "bridge")]
use super::stream::OwnedStream;
#[cfg(feature = "bridge")]
use cudarc::runtime::sys::{
    cudaStream_t, cudaStreamCaptureMode, cudaStreamCaptureStatus,
    cudaStreamUpdateCaptureDependenciesFlags,
};

/// How many predicate slots the device word holds.
///
/// Sized by the wire vocabulary below, not by a guess: eight `GuardPred`
/// wire numbers plus the two Peel endpoint bits.
pub const PRED_SLOTS: usize = 10;

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
/// `GuardPred::WindowOne` — wire 7. The predicate `FireClass::Decode`
/// used to be (`.wiki/driver/graph.md` §4.1).
pub const SLOT_WINDOW_ONE: u32 = 7;
/// A Peel whose whole fire took the fast endpoint (`fast_rows == N`).
///
/// Above the `GuardPred` wire range because a Peel is not a guard: it is a
/// REGION the lowering produced, and its endpoint is a property of the fire
/// rather than of a statement.
pub const SLOT_PEEL_ALL_FAST: u32 = 8;
/// A Peel whose whole fire took the hooked endpoint (`fast_rows == 0`).
pub const SLOT_PEEL_ALL_HOOKED: u32 = 9;

/// The device-resident predicate word: one byte per slot.
///
/// A byte rather than a bitfield because the set-cond kernel indexes it, and
/// an indexed read of a byte is a load while an indexed read of a bit is a
/// shift the kernel would have to be told about.
#[derive(Debug)]
pub struct PredicateWord {
    device: DeviceBuffer,
    host: [u8; PRED_SLOTS],
}

impl PredicateWord {
    /// Allocate the word. Every slot starts false.
    ///
    /// # Errors
    ///
    /// If the allocation fails.
    pub fn new(alloc: &Allocator) -> Result<Self> {
        let device = alloc.alloc(PRED_SLOTS)?;
        Ok(Self {
            device,
            host: [0u8; PRED_SLOTS],
        })
    }

    /// Set one slot in the HOST mirror. [`Self::upload`] is what the device
    /// sees.
    ///
    /// # Errors
    ///
    /// If `slot` is outside the word.
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

    /// Push the host mirror to the device, ordered on `stream`.
    ///
    /// This is the per-fire update, and it is the ONLY host participation a
    /// replay needs: the graph reads the word from inside itself.
    ///
    /// # Errors
    ///
    /// If the copy fails.
    pub fn upload(&mut self, stream: StreamRef<'_>) -> Result<()> {
        let host = self.host;
        self.device.copy_from_host(&host, stream)
    }

    /// The device address the set-cond kernel indexes.
    pub const fn device_ptr(&self) -> *const u8 {
        self.device.as_ptr().cast_const().cast::<u8>()
    }
}

/// A peel's row split, device-resident: `[start, count]`.
///
/// The predicate word's sibling, and for the same reason. A peel's TAIL
/// region addresses rows at ABSOLUTE offsets in a full-N buffer, so the
/// statements there take `_devwin` kernels whose grid spans every lane and
/// whose out-of-window rows early-out on this word — which is what lets a
/// captured fire replay across a different split without recapturing.
/// `split_qkv_devwin_kernel` reads exactly `win[0]` and `win[1]`.
///
/// [`PredicateWord`] already reserves the two Peel ENDPOINT slots
/// ([`SLOT_PEEL_ALL_FAST`], [`SLOT_PEEL_ALL_HOOKED`]); those say whether a
/// fire is all-fast or all-hooked. This says where the split IS, which is
/// the fact the endpoints do not carry.
#[derive(Debug)]
pub struct PeelWindowWord {
    device: DeviceBuffer,
    host: [u32; 2],
}

impl PeelWindowWord {
    /// Allocate the word. The window starts empty, which reads as "no
    /// rows" rather than "all rows" — a kernel that runs before the host
    /// has said anything must do nothing.
    ///
    /// # Errors
    ///
    /// If the allocation fails.
    pub fn new(alloc: &Allocator) -> Result<Self> {
        let device = alloc.alloc(std::mem::size_of::<u32>() * 2)?;
        Ok(Self {
            device,
            host: [0, 0],
        })
    }

    /// Set the window in the host mirror. [`Self::upload`] is what the
    /// device sees.
    pub const fn set(&mut self, start: u32, count: u32) {
        self.host = [start, count];
    }

    /// The window as the host currently believes it.
    pub const fn get(&self) -> (u32, u32) {
        (self.host[0], self.host[1])
    }

    /// Push the host mirror to the device, ordered on `stream`.
    ///
    /// # Errors
    ///
    /// If the copy fails.
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
#[cfg(feature = "bridge")]
struct CaptureInfo {
    status: cudaStreamCaptureStatus,
    graph: cudaGraph_t,
    deps: *const cudaGraphNode_t,
    ndeps: usize,
}

/// `cudaStreamGetCaptureInfo`, spelled so that one binary works on one
/// runtime.
///
/// The same major-version split [`crate::device::graph`]'s `add_node`
/// documents: CUDA 13 renamed the 7-argument form back onto the base symbol,
/// while CUDA 12 keeps it as `_v3`. Calling the wrong one is a segfault far
/// from the cause rather than an error code.
#[cfg(all(feature = "bridge", feature = "cuda-12"))]
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

#[cfg(all(feature = "bridge", feature = "cuda-13"))]
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

/// `cudaStreamUpdateCaptureDependencies`, version-routed for the same reason
/// [`capture_info`] is.
#[cfg(all(feature = "bridge", feature = "cuda-12"))]
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

#[cfg(all(feature = "bridge", feature = "cuda-13"))]
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

/// A conditional node and its arm bodies.
///
/// Raw handles rather than a borrow: the bodies belong to whichever graph was
/// capturing when the node was added, which is not a value the builder owns.
/// Destroying one separately is the mistake, and the builder never hands out
/// anything that could.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone, Copy)]
pub struct Cond {
    node: cudaGraphNode_t,
    if_body: cudaGraph_t,
    else_body: Option<cudaGraph_t>,
}

#[cfg(feature = "bridge")]
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

/// A SWITCH node opened during capture and the bodies it selects among.
/// See [`SupergraphBuilder::open_switch`].
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct Switch {
    node: cudaGraphNode_t,
    bodies: Vec<cudaGraph_t>,
}

#[cfg(feature = "bridge")]
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

    /// Whether the switch has no arms. Never true — `open_switch` refuses
    /// a zero-body switch — but clippy asks for the pair.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
}

/// The capture-time builder: a stack of body captures over a depth-indexed
/// stream pool.
///
/// The root stream must already be inside a capture — [`crate::device::CaptureScope`]
/// is what opens one, and holding that scope is what keeps the allocator shut
/// for the capture's lifetime.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct SupergraphBuilder<'a> {
    root: StreamRef<'a>,
    preds: *const u8,
    /// Depth-indexed pool for body captures; guard nesting is shallow, and
    /// the pool grows on demand. Owned, so the streams die with the builder.
    pool: Vec<OwnedStream>,
    /// The capture stack, innermost last. `active[0]` is always the root.
    active: Vec<cudaStream_t>,
    /// The graph node each retained launch became, by LAUNCH INDEX.
    ///
    /// `.wiki/driver/graph.md` §6.2: `cudaGraphExecKernelNodeSetParams`
    /// can retune an instantiated graph's grids without recapturing, but
    /// only if something remembers which node came from which launch. A
    /// capture used to retain nothing, which is why `tokens` and
    /// `requests` are still bucket-key axes.
    nodes: Vec<Option<cudaGraphNode_t>>,
}

#[cfg(feature = "bridge")]
impl<'a> SupergraphBuilder<'a> {
    /// Start building on an already-capturing stream, reading predicates from
    /// `preds`.
    pub fn new(capture_stream: StreamRef<'a>, preds: &PredicateWord) -> Self {
        Self {
            root: capture_stream,
            preds: preds.device_ptr(),
            pool: Vec::new(),
            active: vec![capture_stream.as_raw()],
            nodes: Vec::new(),
        }
    }

    /// Record the node the launch just issued became, under `index`.
    ///
    /// Read out of the capture's own dependency set: after a kernel
    /// launch on a capturing stream, the stream's dependency list is
    /// exactly the node that launch created. A launch that created none
    /// (a dispatch arm that declined) leaves the slot empty rather than
    /// aliasing its predecessor -- an update applied to the wrong node is
    /// a wrong grid on a kernel nobody asked about.
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

    /// The stream launches should currently target: the root at depth 0, the
    /// innermost body stream inside a body.
    ///
    /// Safe, and that is the point of returning a [`StreamRef`] rather than
    /// the raw handle: the builder owns the pooled body streams and borrows
    /// the root, so the lifetime is one it can prove. A caller issuing work
    /// into a capture should not have to write `unsafe` to name the stream it
    /// is issuing onto.
    pub fn stream(&self) -> StreamRef<'_> {
        // SAFETY: `raw_stream` is either the root (borrowed for `'a`, which
        // outlives `&self`) or a stream in `self.pool`, which lives as long as
        // the builder.
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

    /// Insert a conditional keyed on `pred_slot` at the current capture
    /// position.
    ///
    /// # Errors
    ///
    /// If the slot is out of range, if the current stream is not capturing,
    /// or if any CUDA call refuses.
    pub fn open_cond(&mut self, pred_slot: u32, with_else: bool) -> Result<Cond> {
        if pred_slot as usize >= PRED_SLOTS {
            return Err(Error::invalid("supergraph", "pred slot out of range"));
        }
        let s = self.raw_stream();

        // The handle belongs to whichever graph this stream is capturing --
        // the root graph at depth 0, an arm's body graph when nested.
        let info = unsafe { capture_info(s) }?;
        if info.status != cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Err(Error::invalid("supergraph", "open_cond outside a capture"));
        }

        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe { cudaGraphConditionalHandleCreate(&raw mut handle, info.graph, 0, 0) },
            "cudaGraphConditionalHandleCreate",
        )?;

        // The set-cond kernel FIRST, so that the conditional node picks it up
        // as a capture dependency below and the predicate is written before
        // the branch reads it.
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

    /// Insert a SWITCH keyed on `pred_slot`, with `bodies` arms.
    ///
    /// `.wiki/driver/graph.md` §6.1, and the cheapest win the document
    /// names. [`Self::open_cond`] is a boolean IF and they NEST, so an
    /// axis with more than two options — attention is
    /// `plain / capture / custom / xqa` — costs nesting depth and arm
    /// pairs it should not. A switch reaches every arm from one node,
    /// selected by the INDEX the arming kernel writes.
    ///
    /// The predicate word needs no new storage: it is already a byte per
    /// slot, and a slot holding a kernel index rather than 0/1 is the same
    /// byte read differently. [`crate::fire::supergraph::set_switch`] is the
    /// whole device-side difference.
    ///
    /// An index past `bodies` selects NO body, which is CUDA's rule and is
    /// left as-is: a fire whose predicate names an arm the switch does not
    /// have is a lowering/driver disagreement, and running arm 0 instead
    /// would answer with the wrong program rather than with nothing.
    ///
    /// # Errors
    ///
    /// If the slot is out of range, if `bodies` is zero, if the current
    /// stream is not capturing, or if any CUDA call refuses.
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
        // The arming kernel FIRST, so the switch node picks it up as a
        // capture dependency and the index is written before it is read.
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
        // SAFETY: on success CUDA has pointed `phGraph_out` at an array of
        // exactly `bodies` graphs, owned by the capturing graph.
        let bodies = unsafe { std::slice::from_raw_parts(out, bodies as usize) }.to_vec();
        Ok(Switch { node, bodies })
    }

    /// Capture into an arm's body graph (push).
    ///
    /// # Errors
    ///
    /// If the pooled stream cannot be created or the capture cannot start.
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
    ///
    /// # Errors
    ///
    /// If nothing is open, or if ending the capture fails.
    pub fn end_body(&mut self) -> Result<()> {
        if self.active.len() <= 1 {
            return Err(Error::invalid("supergraph", "end_body underflow"));
        }
        let s = self.raw_stream();
        let mut out: cudaGraph_t = std::ptr::null_mut();
        // The graph handed back is the body graph the parent already owns, so
        // it is read and dropped rather than wrapped: wrapping it would give
        // it a `Drop` that destroys what the parent will destroy again.
        check_rt(
            unsafe { cudarc::runtime::sys::cudaStreamEndCapture(s, &raw mut out) },
            "cudaStreamEndCapture",
        )?;
        self.active.pop();
        Ok(())
    }

    /// Collapse the current stream's capture dependencies onto `cond`'s node,
    /// so subsequent work follows the whole branch.
    ///
    /// # Errors
    ///
    /// If the update fails.
    pub fn close_cond(&mut self, cond: &Cond) -> Result<()> {
        let s = self.raw_stream();
        let mut node = cond.node;
        unsafe { update_capture_deps(s, &raw mut node) }
    }
}

#[cfg(test)]
mod tests_2 {
    use super::*;

    // No CUDA call here. What this pins is that the slot vocabulary still
    // matches `model_compiler::trace::GuardPred::wire`, which is the one fact
    // the device word and the trace have to agree on -- and the one that
    // would otherwise drift silently, because a wrong slot reads a
    // NEIGHBOURING predicate rather than failing.
    #[test]
    fn slots_match_the_guard_wire_vocabulary() {
        use model_compiler::trace::GuardPred;
        assert_eq!(GuardPred::HasWriteDesc.wire().0, SLOT_HAS_WRITE_DESC);
        assert_eq!(GuardPred::TokensLE(0).wire().0, SLOT_TOKENS_LE);
        assert_eq!(GuardPred::TokensGT(0).wire().0, SLOT_TOKENS_GT);
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
