//! Stream capture: the recorded fire, instantiated and replayed. Capture
//! does not execute — a launch between `cudaStreamBeginCapture` and
//! `cudaStreamEndCapture` is written down, not run. [`Graph::capture`] uses
//! thread-local capture mode and always ends the capture on every path out,
//! since a stream left mid-capture answers every later call with
//! `cudaErrorStreamCaptureUnjoined`. [`Event`] lets a capture span more than
//! one stream: a `cudaStreamWaitEvent` on an event a capturing stream
//! recorded pulls the waiting stream into the same graph.

use core::ffi::c_void;

use crate::error::{Fault, Result};

/// A recorded fire, before it is instantiated: the topology and every kernel
/// argument, as the capture wrote them down. Owning one is cheap — it is a
/// handle, not the thing that runs. [`Graph::instantiate`] turns it into the
/// executable, and this can then be dropped: the exec does not borrow it.
#[derive(Debug)]
pub struct Graph {
    raw: *mut c_void,
}

/// An instantiated graph: the thing [`launch`](GraphExec::launch) submits.
/// Kernel arguments are fixed at instantiation to what capture saw (every
/// pointer, extent, grid dimension), which is why the shell keys its cache
/// by everything a fire could change about them, and why `inputs.rs`
/// reserves at the ceiling and never reallocates.
#[derive(Debug)]
pub struct GraphExec {
    raw: *mut c_void,
    nodes: usize,
}

impl Graph {
    /// Records `body` on `stream` instead of running it.
    ///
    /// `body` must enqueue no host work whose effect the replay needs: a
    /// pageable `cudaMemcpyAsync` or a `cudaMalloc` (refused by thread-local
    /// mode), or a plan builder's work estimation (would be missing from
    /// every replay, which is why the prepare phase runs outside this call).
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a capture the runtime refused, and whatever the body refused —
    /// after the capture has been ended either way, because a stream left
    /// mid-capture answers every later call with
    /// `cudaErrorStreamCaptureUnjoined` for the rest of the process.
    pub fn capture(stream: *mut c_void, body: impl FnOnce() -> Result<()>) -> Result<Graph> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (stream, body);
            Err(Fault::Runtimeless)
        }
    }

    /// Debug probe: the raw `cudaGraph_t`, for `cuGraphGetNodes` /
    /// `cuGraphKernelNodeGetParams`. Not read by the fire path;
    /// `tests/descriptor_abi.rs` is the only caller.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// How many nodes it recorded, or `None` when the driver would not say.
    ///
    /// `None` and `0` mean different things: a refused query (e.g. a node
    /// type the query can't represent, such as a conditional node) is not
    /// proof the graph is empty. Callers must handle `None` rather than
    /// treating it as zero.
    #[must_use]
    pub fn nodes(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            let mut count: usize = 0;
            // SAFETY: null node array + live count is the documented way to
            // ask for count alone; `raw` is this graph's handle
            // (`cudaGraph_t`/`CUgraph` are one pointer).
            let code = unsafe {
                dr::cuGraphGetNodes(self.raw.cast(), core::ptr::null_mut(), &raw mut count)
            };
            (code == dr::CUresult::CUDA_SUCCESS).then_some(count)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// How many edges it recorded — the observable a fork actually has. Node
    /// count can't see a fork: stream capture turns an event record/wait
    /// pair into a dependency edge between launches, not new nodes, so a
    /// sequential and a forked capture report the same node count. `None`
    /// for a query the driver refused, for [`nodes`](Graph::nodes)'s reasons.
    #[must_use]
    pub fn edges(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            let mut count: usize = 0;
            // SAFETY: null endpoint/edge-data arrays + live count asks for
            // count alone, on this graph's own handle.
            let code = unsafe {
                dr::cuGraphGetEdges_v2(
                    self.raw.cast(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &raw mut count,
                )
            };
            (code == dr::CUresult::CUDA_SUCCESS).then_some(count)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Instantiates it, and uploads it to `stream`. The upload isn't
    /// decoration: skipping it would push instantiation's one-off device-side
    /// allocation cost into the first `launch`, where it would read as
    /// replay cost.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn instantiate(&self, stream: *mut c_void) -> Result<GraphExec> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut raw: rt::cudaGraphExec_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live local; `self.raw` is this graph's
            // handle. Uses `cudaGraphInstantiateWithFlags`, not plain
            // `cudaGraphInstantiate`, since only the flagged form is spelled
            // the same way under both runtimes this crate builds against.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphInstantiateWithFlags",
                    rt::cudaGraphInstantiateWithFlags(&raw mut raw, self.raw.cast(), 0),
                )?;
            }
            let exec = GraphExec {
                raw: raw.cast(),
                // A count the driver refused is stored as 0 here and decides
                // nothing; the one caller that acts on it asks `Graph::nodes`
                // directly and handles `None`.
                nodes: self.nodes().unwrap_or(0),
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

/// The dependency frontier of the capture in progress on `stream`: the node
/// handles the next thing enqueued would depend on (for a single-stream
/// capture, the last node recorded so far). Full node enumeration is refused
/// mid-capture, so this reads `cuStreamGetCaptureInfo`'s frontier instead.
/// Not called on the fire path. Handles stay valid after
/// `cudaStreamEndCapture`.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime; [`Fault::Device`]
/// when the stream is not capturing or the query refuses.
pub fn capture_frontier(stream: *mut c_void) -> Result<Vec<*mut c_void>> {
    #[cfg(feature = "cuda")]
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

impl GraphExec {
    /// Launches it on `stream`: one submission in place of the eager walk's
    /// many `ctx.fire` calls, reading the same buffers they would have read.
    ///
    /// Enqueue-only, like every other call this crate makes on a fire: the
    /// caller synchronizes when it wants numbers.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn launch(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
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

    /// Debug probe: the raw `cudaGraphExec_t`, to price
    /// `cudaGraphExecKernelNodeSetParams` against it. Not read by the fire
    /// path.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.raw.is_null() {
            // SAFETY: handle came from this graph's own capture, destroyed
            // once; an exec instantiated from it does not borrow it.
            unsafe {
                let _ = cudarc::runtime::sys::cudaGraphDestroy(self.raw.cast());
            }
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
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
/// Created with `cudaEventDisableTiming`, since nothing asks when it
/// happened. One event is created per `model_compiler::EventId` at load and
/// re-recorded on every capturing fire — legal, since recording again just
/// overwrites what it names, and inside a capture a record/wait pair is a
/// dependency edge between launches, not a runtime synchronization.
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
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// A timing event: same handle, created without `cudaEventDisableTiming`
    /// so two can be subtracted. Not for the fire path; used by the
    /// saturation gate to measure inter-step device gaps.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn timing() -> Result<Event> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;
            let mut raw: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter; this thread bound the
            // device. Flag 0 is `cudaEventDefault` (timing enabled).
            unsafe {
                crate::device::ctx::check(
                    "cudaEventCreateWithFlags",
                    rt::cudaEventCreateWithFlags(&raw mut raw, 0),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Whether everything recorded behind this event has completed — asked,
    /// never waited on (`cudaEventQuery`). The routed-expert tier asks this
    /// before reusing pinned staging words a previous promotion's copies may
    /// still be reading, skipping that round's promotion rather than
    /// waiting. An event never recorded answers `true`.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] for a status that is
    /// neither "done" nor "not yet".
    pub fn done(&self) -> Result<bool> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Blocks this thread until everything recorded before this event has
    /// happened (`cudaEventSynchronize`) — unlike `cudaStreamSynchronize`,
    /// which also drains work enqueued after this point. An event that was
    /// never recorded returns at once.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] for whatever the recorded
    /// work said — an asynchronous fault from any launch before the record
    /// surfaces here.
    pub fn settle(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Milliseconds of device time from `self` to `end`, for two events
    /// created by [`Event::timing`] and both already completed.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] when either event has not
    /// completed or was created with timing disabled.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = end;
            Err(Fault::Runtimeless)
        }
    }

    /// Record this event on `stream`: the fork half. Everything already
    /// enqueued on `stream` is what a waiter will have waited for. Inside a
    /// capture this becomes an event-record node.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn record(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventRecord",
                    cudarc::runtime::sys::cudaEventRecord(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Make `stream` wait for this event: the join half. Enqueue-only: the
    /// host does not block, the stream does. Inside a capture this is the
    /// edge that carries the capture onto `stream` (or back off it).
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn wait(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.raw.is_null() {
            // SAFETY: created by this handle's `new`, destroyed once, and the
            // shell synchronizes before it tears a load down.
            unsafe {
                let _ = cudarc::runtime::sys::cudaEventDestroy(self.raw.cast());
            }
        }
    }
}
