//! Stream capture: the recorded fire, instantiated and replayed.
//!
//! **CAPTURE DOES NOT EXECUTE, AND EVERY DECISION IN `record.rs` FOLLOWS FROM
//! THAT.** Between [`cudaStreamBeginCapture`] and `cudaStreamEndCapture` a
//! launch is *written down* rather than run: the stream produces a
//! `cudaGraph_t` and no numbers. So a fire that captures has to get its
//! numbers some other way — `record.rs` runs the walk eagerly first and
//! captures a second walk of the same regions over the same buffers — and a
//! fire that replays gets them from [`GraphExec::launch`].
//!
//! # What this module is, and what it refuses to be
//!
//! Four calls and two handles. It does not know what a region is, what a
//! bucket is, or when a capture is worth doing; those are `record.rs`'s and
//! `serve.rs`'s, exactly as the eager plane's policy is. What it owns is the
//! part that is unsafe: a capture that is begun and not ended leaves the
//! stream unusable for the rest of the process, so [`Graph::capture`] ends the
//! capture on EVERY path out of the body, including the one where the body
//! refused.
//!
//! # The capture mode, and why it is thread-local
//!
//! `cudaStreamCaptureModeThreadLocal` is the middle of the three, and it is
//! the one that matches this shell's shape:
//!
//! ```text
//! Global       an unsafe call on ANY thread invalidates this capture —
//!              another shell's `cudaMalloc`, a loader still landing bytes
//! ThreadLocal  an unsafe call on THIS thread does. Which is the constraint
//!              we actually want enforced: the captured body must contain no
//!              host work, and this is the runtime saying so.
//! Relaxed      nothing is refused, and a `cudaMalloc` inside the body
//!              silently becomes a graph that reads freed memory
//! ```
//!
//! One shell fires at a time per process (`serve.rs`), so Global would buy
//! nothing this does not already have, and it would make an unrelated
//! thread's allocation a failed capture.
//!
//! # Several streams, one graph (P6)
//!
//! [`Event`] is the other half of this module and it is what makes a capture
//! span more than one stream. `.wiki/tart/evidence/green_contexts.md` Finding
//! 3 measured the pattern on real hardware and called it the make-or-break
//! result: record an event on the capturing stream, have a second stream
//! `cudaStreamWaitEvent` on it, launch work there, then record on the second
//! and wait from the first — and `cudaStreamEndCapture` returns **one** graph
//! containing both. The waits are what carry the capture across: a stream
//! that waits on an event recorded by a capturing stream is thereafter part of
//! that capture, and one that never rejoins leaves it
//! `cudaErrorStreamCaptureUnjoined`.
//!
//! So nothing here knows what a fork group is either. The compiler's P6 says
//! which stream and which event; `record.rs` holds the handles; this module
//! is four more calls.
//!
//! [`cudaStreamBeginCapture`]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html

use core::ffi::c_void;

use crate::error::{Fault, Result};

/// A recorded fire, before it is instantiated: the topology and every kernel
/// argument, as the capture wrote them down.
///
/// Owning one is cheap — it is a handle — but it is not the thing that runs.
/// [`Graph::instantiate`] turns it into the executable, and this can then be
/// dropped: the exec does not borrow it.
#[derive(Debug)]
pub struct Graph {
    raw: *mut c_void,
}

/// An instantiated graph: the thing [`launch`](GraphExec::launch) submits.
///
/// **ITS KERNEL ARGUMENTS ARE THE ONES CAPTURE SAW.** Every pointer, every
/// extent, every grid dimension is fixed at instantiation, which is why the
/// shell keys its cache by everything a fire could change about them and why
/// `inputs.rs` reserves at the ceiling and never reallocates. Content flows
/// through those fixed addresses; shape does not flow at all.
#[derive(Debug)]
pub struct GraphExec {
    raw: *mut c_void,
    nodes: usize,
}

impl Graph {
    /// Record `body` on `stream` instead of running it.
    ///
    /// The body enqueues exactly as it would eagerly — that is the whole
    /// point of the two-mode walk — and nothing it enqueues executes. It must
    /// therefore contain no host work whose EFFECT the replay needs: a
    /// pageable `cudaMemcpyAsync`, a `cudaMalloc`, a plan builder's work
    /// estimation. The first two the runtime refuses (thread-local mode); the
    /// third would simply be missing from every replay, which is why the
    /// prepare phase runs outside this call rather than being trusted to
    /// behave inside it.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a capture the runtime refused, and whatever the body refused —
    /// after the capture has been ended either way, because a stream left
    /// mid-capture answers every later call with
    /// `cudaErrorStreamCaptureUnjoined` for the rest of the process.
    pub fn capture(stream: *mut c_void, body: impl FnOnce() -> Result<()>) -> Result<Graph> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: `stream` is the shell's, live for the whole call, and
            // this thread is the one that bound the device.
            unsafe {
                crate::device::ctx::check(
                    "cudaStreamBeginCapture",
                    rt::cudaStreamBeginCapture(
                        stream.cast(),
                        rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
                    ),
                )?;
            }

            let walked = body();

            let mut raw: rt::cudaGraph_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live local; the capture was begun above and
            // is ended here on every path.
            let ended = unsafe {
                crate::device::ctx::check(
                    "cudaStreamEndCapture",
                    rt::cudaStreamEndCapture(stream.cast(), &raw mut raw),
                )
            };

            let graph = (!raw.is_null()).then(|| Graph { raw: raw.cast() });
            walked?;
            ended?;
            graph.ok_or(Fault::Device {
                call: "cudaStreamEndCapture",
                code: 0,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (stream, body);
            Err(Fault::Runtimeless)
        }
    }

    /// **PROBE SEAM (`palo cuda-abi` wave).** The raw `cudaGraph_t`, so a
    /// probe can walk its kernel nodes with `cuGraphGetNodes` /
    /// `cuGraphKernelNodeGetParams`. Nothing in the fire path reads it;
    /// `tests/descriptor_abi.rs` is the only caller.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// How many nodes it recorded.
    ///
    /// Reported, not used: it is the number the rebind arithmetic of decision
    /// #15 would be multiplied by (~0.11 µs per node, measured in
    /// `tart/evidence/layout_planning.md`), and the honest way to compare a
    /// per-fire rebind against a per-key capture is to say it out loud.
    #[must_use]
    pub fn nodes(&self) -> usize {
        #[cfg(feature = "_cuda")]
        {
            let mut count: usize = 0;
            // SAFETY: a null node array with a live count is the documented
            // way to ask for the count alone.
            let status = unsafe {
                cudarc::runtime::sys::cudaGraphGetNodes(
                    self.raw.cast(),
                    core::ptr::null_mut(),
                    &raw mut count,
                )
            };
            if status == cudarc::runtime::sys::cudaError::cudaSuccess {
                count
            } else {
                0
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            0
        }
    }

    /// How many EDGES it recorded — the observable a fork actually has.
    ///
    /// **NODE COUNT CANNOT SEE A FORK AND EDGE COUNT CAN.** Stream capture
    /// turns a `cudaEventRecord` and the `cudaStreamWaitEvent` behind it into
    /// a DEPENDENCY between the launches on either side rather than into nodes
    /// of their own, which is exactly the lowering one wants and exactly what
    /// makes `cudaGraphGetNodes` answer the same number for a sequential
    /// capture and a forked one. The topology is where the difference lives: a
    /// capture on one stream is a chain, `N` nodes and `N-1` edges, and every
    /// fork/join pair adds one edge that the chain does not have while
    /// removing none.
    ///
    /// So this is what a measurement asks to say its two arms are two
    /// different graphs, and it is the only thing on either handle that a
    /// mis-wired side stream could not fake.
    #[must_use]
    pub fn edges(&self) -> usize {
        #[cfg(feature = "_cuda")]
        {
            let mut count: usize = 0;
            // SAFETY: two null endpoint arrays with a live count is the
            // documented way to ask for the count alone.
            let status = unsafe {
                cudarc::runtime::sys::cudaGraphGetEdges(
                    self.raw.cast(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &raw mut count,
                )
            };
            if status == cudarc::runtime::sys::cudaError::cudaSuccess {
                count
            } else {
                0
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            0
        }
    }

    /// Instantiate it, and upload it to `stream`.
    ///
    /// The upload is not decoration: instantiation allocates the exec's
    /// device-side node parameters, and a first `launch` that had to do that
    /// too would put a millisecond of one-off cost inside the first replay,
    /// where a measurement would read it as the replay's price.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn instantiate(&self, stream: *mut c_void) -> Result<GraphExec> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut raw: rt::cudaGraphExec_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live local and `self.raw` is this graph's
            // handle. `cudaGraphInstantiateWithFlags` is spelled the same way
            // under both runtimes this crate builds against — plain
            // `cudaGraphInstantiate` is not (the manifest's note on arity).
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphInstantiateWithFlags",
                    rt::cudaGraphInstantiateWithFlags(&raw mut raw, self.raw.cast(), 0),
                )?;
            }
            let exec = GraphExec {
                raw: raw.cast(),
                nodes: self.nodes(),
            };
            // SAFETY: the exec was just created and the stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphUpload",
                    rt::cudaGraphUpload(raw, stream.cast()),
                )?;
            }
            Ok(exec)
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

/// The dependency FRONTIER of the capture in progress on `stream`: the node
/// handles the next thing enqueued would depend on — for a single-stream
/// capture, the last node recorded so far.
///
/// **THE FOLD'S REGION CENSUS RIDES ON THIS** (`.wiki/palo/cuda-abi.md` §6,
/// D5-lite), and on nothing stronger because nothing stronger is answered:
/// full node ENUMERATION of a capture-in-progress graph is refused by this
/// toolkit — `cudaGraphGetNodes` and `cuGraphGetNodes` both answer
/// InvalidValue (code 1) mid-capture, measured on CUDA 13.0 / 580.159 —
/// while the frontier is `cuStreamGetCaptureInfo`'s documented out-parameter
/// (it exists so `cuStreamUpdateCaptureDependencies` callers can read before
/// they write). So the census records the frontier at every region boundary
/// and places nodes AFTER the capture ends, when the finished graph
/// enumerates freely: on a serial capture the graph is a chain, the frontier
/// at a boundary is the chain position the region ended at, and every node
/// between two boundary positions belongs to the region between them. This
/// is why the fold captures SERIALLY — a forked capture's frontier is one
/// stream's, and its finished graph is not a chain positions can be read
/// off.
///
/// The handles stay valid after `cudaStreamEndCapture`: ending a capture
/// finishes the same `cudaGraph_t` these nodes already belong to.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime; [`Fault::Device`]
/// when the stream is not capturing or the query refuses.
pub fn capture_frontier(stream: *mut c_void) -> Result<Vec<*mut c_void>> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::driver::sys as dr;

        let mut status = dr::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id: u64 = 0;
        let mut graph: dr::CUgraph = core::ptr::null_mut();
        let mut deps: *const dr::CUgraphNode = core::ptr::null();
        let mut dep_count: usize = 0;
        // `_v3` is the one spelling CUDA 12 and 13 share — 13 retired `_v2`.
        //
        // SAFETY: every out-parameter is a live local; the stream is the
        // shell's and this thread began the capture.
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
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Fault::Device {
                call: "cuStreamGetCaptureInfo_v3",
                code: code as i32,
            });
        }
        if status != dr::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(Fault::Device {
                call: "cuStreamGetCaptureInfo_v3 (the stream is not capturing)",
                code: status as i32,
            });
        }
        if deps.is_null() || dep_count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the driver's array is `dep_count` handles long and lives
        // until the next capture-mutating call, which is after this copy.
        let frontier = unsafe { core::slice::from_raw_parts(deps, dep_count) };
        Ok(frontier.iter().map(|node| node.cast()).collect())
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

impl GraphExec {
    /// Turn one node of this exec on or off, host-side.
    ///
    /// **THE FOLD'S COMPOSITION AXIS** (`.wiki/palo/cuda-abi.md` §6, D5-lite):
    /// a folded exec holds every region of the full composition, and an empty
    /// window's nodes are turned OFF here rather than keyed away. For a
    /// LIBRARY node this is the correctness mechanism — there is no zero-row
    /// contract to fall back on — and for one of ours it is economy (the
    /// zero-row early exit would be correct at ~1 µs of dispatch). The PoC
    /// priced the call at ~0.22 µs with NO re-upload needed, which is what
    /// makes a per-fire enable diff affordable where `cuGraphExecUpdate`
    /// would not be.
    ///
    /// `node` is the handle in the `cudaGraph_t` this exec was instantiated
    /// from — the same coordinate `cudaGraphExecKernelNodeSetParams` takes —
    /// so the template graph must still be alive, which is the rule the fold
    /// keeps by owning it beside the exec.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`] (a node kind the runtime
    /// cannot toggle — anything but kernel, memcpy, memset).
    pub fn set_node_enabled(&self, node: *mut c_void, enabled: bool) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: the exec is this handle's and `node` belongs to the
            // graph it was instantiated from — the caller's stated contract.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphNodeSetEnabled",
                    cudarc::runtime::sys::cudaGraphNodeSetEnabled(
                        self.raw.cast(),
                        node.cast(),
                        u32::from(enabled),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (node, enabled);
            Err(Fault::Runtimeless)
        }
    }

    /// Re-land this exec's device-side state on `stream`.
    ///
    /// **THE OTHER HALF OF A HOST REBIND, PAID AT BIND TIME.**
    /// `.wiki/palo/cuda-abi.md` §2 states the rule — after a host-side
    /// update the exec must be re-uploaded before the next launch — for the
    /// device-updatable case, and the plain-exec documentation promises the
    /// opposite ("the next launch will use the updated parameters"). The
    /// fold measured the question rather than picking a doc: steady folded
    /// launches after a ~500-node restatement run at EXACT parity with the
    /// keyed replay once structure is equal (the steady-mixed gate,
    /// streams off: 4.511 against 4.511 ms/fire), with this upload in the
    /// rebind. It is kept there because it is the stated-safe order, it is
    /// enqueue-only, and it costs the binding — which a throwaway capture
    /// already dwarfs — rather than any launch.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn upload(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: the exec is this handle's, alive until `Drop`, and the
            // stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphUpload",
                    cudarc::runtime::sys::cudaGraphUpload(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Launch it on `stream`.
    ///
    /// **THIS IS THE WHOLE FIRE PATH, ONCE THE PREPARE PHASE IS DONE.** One
    /// submission in place of the hundreds of `ctx.fire` calls the eager walk
    /// makes, reading the same buffers they would have read.
    ///
    /// Enqueue-only, like every other call this crate makes on a fire: the
    /// caller synchronizes when it wants numbers.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn launch(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: the exec is this handle's, alive until `Drop`, and the
            // stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphLaunch",
                    cudarc::runtime::sys::cudaGraphLaunch(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// How many nodes it replays.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// **PROBE SEAM (`palo cuda-abi` wave).** The raw `cudaGraphExec_t`, so a
    /// probe can price `cudaGraphExecKernelNodeSetParams` against it.
    /// Nothing in the fire path reads it.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if !self.raw.is_null() {
            // SAFETY: the handle came from this graph's own capture and is
            // destroyed exactly once; an exec instantiated from it does not
            // borrow it.
            unsafe {
                let _ = cudarc::runtime::sys::cudaGraphDestroy(self.raw.cast());
            }
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if !self.raw.is_null() {
            // SAFETY: destroyed once, and the shell synchronizes its stream
            // before it drops a cache entry (`record.rs`'s eviction).
            unsafe {
                let _ = cudarc::runtime::sys::cudaGraphExecDestroy(self.raw.cast());
            }
        }
    }
}

/// One `cudaEvent_t`: a point on a stream that another stream can wait for.
///
/// **CREATED WITH `cudaEventDisableTiming`**, because nothing asks it when it
/// happened. A timing-enabled event is materially more expensive to record —
/// the runtime writes a timestamp on the device — and the whole point of P6's
/// gate is that an event pair has to be cheap enough to be worth a fork.
///
/// One event is created per `model_compiler::EventId` at load and recorded on
/// EVERY fire that captures. That is legal and is the ordinary use: recording
/// an event again simply overwrites what it names, and inside a capture the
/// record and the wait are not a runtime synchronization at all — they are a
/// DEPENDENCY between the launches on either side. Measured on this build
/// (build log 24): a qwen capture holds 621 nodes and 620 edges with the
/// streams off, which is exactly a chain, and 621 nodes and 631 edges with
/// them on. The event points cost no nodes and eleven edges.
#[derive(Debug)]
pub struct Event {
    raw: *mut c_void,
}

impl Event {
    /// Create one.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn new() -> Result<Event> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;
            let mut raw: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter and this thread bound the
            // device.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventCreateWithFlags",
                    rt::cudaEventCreateWithFlags(&raw mut raw, 2), // cudaEventDisableTiming
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// **A TIMING event** — the same handle, created without
    /// `cudaEventDisableTiming` so that two of them can be subtracted.
    ///
    /// Not for the fire path: timing events cost a little more to record and
    /// the fork/join edges P6 uses want none of it. What wants them is a GATE
    /// — F2b's saturation gate measures the device-side gap between
    /// consecutive steps, and that measurement is what article 1's enforcement
    /// clause asks for ("an e2e gate measures inter-wave stream gaps").
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn timing() -> Result<Event> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;
            let mut raw: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter and this thread bound the
            // device. Flag 0 is `cudaEventDefault` — timing enabled.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventCreateWithFlags",
                    rt::cudaEventCreateWithFlags(&raw mut raw, 0),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// **Has everything this event was recorded behind completed?** — asked,
    /// never waited on.
    ///
    /// `cudaEventQuery`, which is the one device call that answers a question
    /// about progress without blocking on it. The routed-expert tier (alto
    /// design §7) asks it before it reuses the pinned staging words a previous
    /// promotion's copies may still be reading: a `false` skips that gap's
    /// promotion, which is the honest answer for a mechanism whose whole
    /// doctrine is that residency is a promotion and never a wait.
    ///
    /// An event that was never recorded answers `true` — there is nothing
    /// outstanding behind it — which is what makes the first round free.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] for a status that is
    /// neither "done" nor "not yet".
    pub fn done(&self) -> Result<bool> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::cudaEventQuery(self.raw.cast()) };
            match status {
                rt::cudaError::cudaSuccess => Ok(true),
                rt::cudaError::cudaErrorNotReady => {
                    // The status is consumed here rather than left to be
                    // re-reported by the next unrelated call.
                    #[allow(unused_must_use)]
                    unsafe {
                        rt::cudaGetLastError();
                    }
                    Ok(false)
                }
                code => Err(Fault::Device {
                    call: "cudaEventQuery",
                    code: code as i32,
                }),
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// **Block this thread until everything recorded before this event has
    /// happened** — `cudaEventSynchronize`, and the narrow form of a wait.
    ///
    /// **NOT `cudaStreamSynchronize`.** A stream synchronize drains the WHOLE
    /// stream, including work enqueued after the point of interest, so a
    /// caller that only needs one boundary's kernels to have landed pays for
    /// every launch behind them and leaves the device with nothing queued.
    /// This waits for exactly the recorded point, so the work enqueued after
    /// it keeps running while the host blocks — which is the difference
    /// between a host that waits and a GPU that idles.
    ///
    /// An event that was never recorded returns at once, for the same reason
    /// [`Event::done`] answers `true` for one.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] for whatever the recorded
    /// work said — this is a blocking call, so an asynchronous fault from any
    /// launch before the record surfaces here.
    pub fn settle(&self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::cudaEventSynchronize(self.raw.cast()) };
            if status != rt::cudaError::cudaSuccess {
                return Err(Fault::Device {
                    call: "cudaEventSynchronize",
                    code: status as i32,
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// **Milliseconds of device time from `self` to `end`**, for two events
    /// created by [`Event::timing`] and both already completed.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] when either event has not
    /// completed or was created with timing disabled.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        #[cfg(feature = "_cuda")]
        {
            let mut ms: f32 = 0.0;
            // SAFETY: both handles are live and this crate created them.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventElapsedTime",
                    cudarc::runtime::sys::cudaEventElapsedTime(
                        &raw mut ms,
                        self.raw.cast(),
                        end.raw.cast(),
                    ),
                )?;
            }
            Ok(ms)
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = end;
            Err(Fault::Runtimeless)
        }
    }

    /// Record this event on `stream`: the FORK half.
    ///
    /// Everything already enqueued on `stream` is what a waiter will have
    /// waited for. Inside a capture this becomes an event-record node.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn record(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventRecord",
                    cudarc::runtime::sys::cudaEventRecord(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Make `stream` wait for this event: the JOIN half.
    ///
    /// Enqueue-only, like everything else this crate does on a fire: the host
    /// does not block, the stream does. Inside a capture this is the edge that
    /// carries the capture ONTO `stream` (or back off it), which is the whole
    /// of Finding 3's mechanism.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn wait(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaStreamWaitEvent",
                    cudarc::runtime::sys::cudaStreamWaitEvent(
                        stream.cast(),
                        self.raw.cast(),
                        0,
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if !self.raw.is_null() {
            // SAFETY: created by this handle's `new`, destroyed once, and the
            // shell synchronizes before it tears a load down.
            unsafe {
                let _ = cudarc::runtime::sys::cudaEventDestroy(self.raw.cast());
            }
        }
    }
}
