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

impl GraphExec {
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
