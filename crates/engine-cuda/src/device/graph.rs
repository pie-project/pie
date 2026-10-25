#![allow(unsafe_code)]
use core::ffi::c_void;

use crate::error::{Fault, Result};

#[derive(Debug)]
pub struct Graph {
    raw: *mut c_void,
}

#[derive(Debug)]
pub struct GraphExec {
    raw: *mut c_void,
    nodes: usize,
}

impl Graph {
    pub fn capture(stream: *mut c_void, body: impl FnOnce() -> Result<()>) -> Result<Graph> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;

            // What the stream held before is marked for the fences kept
            // memory waits on.
            crate::device::held::begin_capture(stream, || {
                // SAFETY: `stream` is the shell's, live for the whole call,
                // and this thread is the one that bound the device.
                unsafe {
                    crate::device::ctx::check(
                        "cudaStreamBeginCapture",
                        rt::cudaStreamBeginCapture(
                            stream.cast(),
                            rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
                        ),
                    )
                }
            })?;

            let walked = body();

            let mut raw: rt::cudaGraph_t = core::ptr::null_mut();
            let ended = crate::device::held::end_capture(stream, || {
                // SAFETY: `raw` is a live local; the capture was begun above
                // and is ended here on every path.
                unsafe {
                    crate::device::ctx::check(
                        "cudaStreamEndCapture",
                        rt::cudaStreamEndCapture(stream.cast(), &raw mut raw),
                    )
                }
            });

            let graph = (!raw.is_null()).then(|| Graph { raw: raw.cast() });
            walked?;
            ended?;
            graph.ok_or(Fault::Device {
                call: "cudaStreamEndCapture",
                code: 0,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (stream, body);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn debug_dot(&self, path: &str) -> bool {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
            let Ok(c_path) = std::ffi::CString::new(path) else {
                return false;
            };
            let flags = 1 | 4 | 512;
            // SAFETY: `raw` is this graph's live handle; `c_path` outlives the call.
            let code =
                unsafe { rt::cudaGraphDebugDotPrint(self.raw.cast(), c_path.as_ptr(), flags) };
            code == rt::cudaError_t::cudaSuccess
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = path;
            false
        }
    }

    #[must_use]
    pub fn nodes(&self) -> Option<usize> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

            let mut count: usize = 0;
            // SAFETY: null node array + live count is the documented way to
            // ask for count alone; `raw` is this graph's handle
            // (`cudaGraph_t`/`CUgraph` are one pointer).
            let code = unsafe {
                dr::cuGraphGetNodes(self.raw.cast(), core::ptr::null_mut(), &raw mut count)
            };
            (code == dr::CUresult::CUDA_SUCCESS).then_some(count)
        }
        #[cfg(not(feature = "device"))]
        {
            None
        }
    }

    #[must_use]
    pub fn edges(&self) -> Option<usize> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

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
        #[cfg(not(feature = "device"))]
        {
            None
        }
    }

    /// Device bytes one instantiation of this graph holds, averaged over
    /// `copies` held at once. Each is uploaded on this thread's own stream
    /// (the legacy one would wait on, and invalidate, a capture another
    /// thread makes on a blocking stream), landed before it is weighed.
    pub fn instantiated_bytes(&self, copies: usize) -> Result<f64> {
        #[cfg(feature = "device")]
        let own = crate::device::alloc::OWN_STREAM;
        #[cfg(not(feature = "device"))]
        let own = core::ptr::null_mut();
        let before = crate::device::free_bytes().unwrap_or(0);
        let held = (0..copies)
            .map(|_| self.instantiate(own))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "device")]
        crate::device::ctx::sync(own)?;
        let after = crate::device::free_bytes().unwrap_or(0);
        drop(held);
        Ok(before.saturating_sub(after) as f64 / copies as f64)
    }

    pub fn instantiate(&self, stream: *mut c_void) -> Result<GraphExec> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;

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
        #[cfg(not(feature = "device"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl GraphExec {
    pub fn launch(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "device")]
        {
            // SAFETY: the exec is this handle's, alive until `Drop`, and the
            // stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "cudaGraphLaunch",
                    crate::device::rt::cudaGraphLaunch(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        if !self.raw.is_null() {
            // SAFETY: handle came from this graph's own capture, destroyed
            // once; an exec instantiated from it does not borrow it.
            unsafe {
                let _ = crate::device::rt::cudaGraphDestroy(self.raw.cast());
            }
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        if !self.raw.is_null() {
            // SAFETY: destroyed once, and the shell synchronizes its stream
            // before it drops a cache entry (`record.rs`'s eviction).
            unsafe {
                let _ = crate::device::rt::cudaGraphExecDestroy(self.raw.cast());
            }
        }
    }
}

#[derive(Debug)]
pub struct Event {
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    raw: *mut c_void,
}

impl Event {
    pub fn new() -> Result<Event> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
            let mut raw: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter and this thread bound the
            // device.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventCreateWithFlags",
                    rt::cudaEventCreateWithFlags(&raw mut raw, 2),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn timing() -> Result<Event> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
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
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn done(&self) -> Result<bool> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::cudaEventQuery(self.raw.cast()) };
            match status {
                rt::cudaError::cudaSuccess => Ok(true),
                rt::cudaError::cudaErrorNotReady => {
                    #[allow(unused_must_use)]
                    // SAFETY: takes no arguments; it only clears the
                    // not-ready status the query left on this thread.
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
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn settle(&self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
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
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn record(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "device")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventRecord",
                    crate::device::rt::cudaEventRecord(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    pub fn wait(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "device")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaStreamWaitEvent",
                    crate::device::rt::cudaStreamWaitEvent(stream.cast(), self.raw.cast(), 0),
                )
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        if !self.raw.is_null() {
            // SAFETY: created by this handle's `new`, destroyed once, and the
            // shell synchronizes before it tears a load down.
            unsafe {
                let _ = crate::device::rt::cudaEventDestroy(self.raw.cast());
            }
        }
    }
}
