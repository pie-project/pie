//! The unionized supergraph's capture scaffolding.
//!
//! This is the Rust port of `driver-cuda/csrc/src/batch/supergraph.{cu,hpp}`
//! — S2 of the supergraph ladder, and the piece [`crate::cuda::graph`]'s
//! conditional-node primitive was ported to serve.
//!
//! # What a supergraph is, and what it is not
//!
//! ONE captured CUDA graph per (R, N) bucket whose attachment branches — the
//! declared trace's `GuardPred` vocabulary — are conditional `if` nodes. The
//! predicates live in a DEVICE-resident word ([`PredicateWord`]) that the
//! replay path updates per fire; a graph-embedded kernel reads a slot and
//! arms the conditional handle, so a replay takes the fire's arms with no
//! host round-trip and no recapture.
//!
//! That is not batching. Batching amortises one program over many rows; this
//! amortises many *programs* over one capture. Concurrent requests that are
//! structurally distinct — differing in hook attachment, mask kind,
//! correction arm, depth, LoRA rank — fold into one conditional graph, so the
//! operators they share execute exactly once.
//!
//! # The capture-time dance
//!
//! Conditional nodes are added to whichever graph is currently capturing, and
//! their bodies are filled by capturing a DIFFERENT stream into the body
//! graph. So the builder keeps a stack:
//!
//! * [`SupergraphBuilder::open_cond`] — create the handle on the capturing
//!   graph (`cudaStreamGetCaptureInfo` answers for whichever stream is
//!   capturing, at any nesting depth), launch the set-cond kernel so it
//!   becomes the node's upstream dependency, then insert the conditional node
//!   with the running deps;
//! * [`SupergraphBuilder::begin_body`] — `cudaStreamBeginCaptureToGraph` on
//!   the next pooled depth stream;
//! * [`SupergraphBuilder::end_body`] — end that stream's capture;
//! * [`SupergraphBuilder::close_cond`] — collapse the outer stream's capture
//!   dependencies onto the conditional node, so post-branch work follows the
//!   whole branch rather than racing it.
//!
//! Guard nesting maps to a stack of body captures over the depth-indexed
//! stream pool. Nothing here knows about models.

#[cfg(feature = "bridge")]
use std::ffi::c_void;

#[cfg(feature = "bridge")]
use cudarc::runtime::sys::{
    cudaGraphConditionalHandle, cudaGraphConditionalHandleCreate, cudaGraphNodeParams,
    cudaGraphNode_t, cudaGraph_t, cudaStreamCaptureMode, cudaStreamCaptureStatus,
    cudaStreamUpdateCaptureDependenciesFlags, cudaStream_t,
};

use super::alloc::{Allocator, DeviceBuffer};
use super::stream::StreamRef;
#[cfg(feature = "bridge")]
use super::stream::OwnedStream;
#[cfg(feature = "bridge")]
use crate::cuda::graph::add_node;
#[cfg(feature = "bridge")]
use crate::error::check_rt;
use crate::error::{Error, Result};

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
        Ok(Self { device, host: [0u8; PRED_SLOTS] })
    }

    /// Set one slot in the HOST mirror. [`Self::upload`] is what the device
    /// sees.
    ///
    /// # Errors
    ///
    /// If `slot` is outside the word.
    pub fn set(&mut self, slot: u32, on: bool) -> Result<()> {
        let i = usize::try_from(slot).unwrap_or(usize::MAX);
        let cell = self.host.get_mut(i).ok_or_else(|| {
            Error::invalid("supergraph", "predicate slot out of range")
        })?;
        *cell = u8::from(on);
        Ok(())
    }

    /// Clear every slot in the host mirror.
    pub const fn clear(&mut self) {
        self.host = [0u8; PRED_SLOTS];
    }

    /// Read one slot back out of the host mirror.
    pub fn get(&self, slot: u32) -> bool {
        usize::try_from(slot).ok().and_then(|i| self.host.get(i)).is_some_and(|&v| v != 0)
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
        Ok(Self { device, host: [0, 0] })
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

#[cfg(feature = "bridge")]
unsafe extern "C" {
    /// See `csrc/supergraph.cu`. Returns a `cudaError_t` as an int.
    fn pie_supergraph_set_cond(
        handle: u64,
        preds: *const u8,
        slot: i32,
        stream: *mut c_void,
    ) -> i32;

    /// See `csrc/supergraph.cu`. Arms a SWITCH handle from a slot read as
    /// a body INDEX. Returns a `cudaError_t` as an int.
    fn pie_supergraph_set_switch(
        handle: u64,
        preds: *const u8,
        slot: i32,
        stream: *mut c_void,
    ) -> i32;
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
/// The same major-version split [`crate::cuda::graph`]'s `add_node`
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
    Ok(CaptureInfo { status, graph, deps, ndeps })
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
    Ok(CaptureInfo { status, graph, deps, ndeps })
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
/// The root stream must already be inside a capture — [`crate::cuda::CaptureScope`]
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
        let Ok(info) = (unsafe { capture_info(self.raw_stream()) }) else { return };
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
        self.active.last().copied().unwrap_or_else(|| self.root.as_raw())
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
        let rc = unsafe {
            pie_supergraph_set_cond(
                handle,
                self.preds,
                i32::try_from(pred_slot).unwrap_or(0),
                s.cast::<c_void>(),
            )
        };
        if rc != 0 {
            return Err(Error::invalid("pie_supergraph_set_cond", "the set-cond launch failed"));
        }

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
                add_node(&raw mut node, info.graph, info.deps, info.ndeps, &raw mut params)
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

        Ok(Cond { node, if_body, else_body })
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
    /// byte read differently. `pie_supergraph_set_switch` is the whole
    /// device-side difference.
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
            return Err(Error::invalid("supergraph", "a switch with no bodies selects nothing"));
        }
        let s = self.raw_stream();
        let info = unsafe { capture_info(s) }?;
        if info.status != cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Err(Error::invalid("supergraph", "open_switch outside a capture"));
        }
        let mut handle: cudaGraphConditionalHandle = 0;
        check_rt(
            unsafe { cudaGraphConditionalHandleCreate(&raw mut handle, info.graph, 0, 0) },
            "cudaGraphConditionalHandleCreate",
        )?;
        // The arming kernel FIRST, so the switch node picks it up as a
        // capture dependency and the index is written before it is read.
        let rc = unsafe {
            pie_supergraph_set_switch(
                handle,
                self.preds,
                i32::try_from(pred_slot).unwrap_or(0),
                s.cast::<c_void>(),
            )
        };
        if rc != 0 {
            return Err(Error::invalid(
                "pie_supergraph_set_switch",
                "the set-switch launch failed",
            ));
        }
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
                add_node(&raw mut node, info.graph, info.deps, info.ndeps, &raw mut params)
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
mod tests {
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
