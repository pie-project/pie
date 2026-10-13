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
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: `stream` is the shell's, live for the whole call, and
            // this thread is the one that bound the device.
            unsafe {
                crate::device::ctx::check(
                    "hipStreamBeginCapture",
                    rt::hipStreamBeginCapture(stream.cast(), rt::hipStreamCaptureModeThreadLocal),
                )?;
            }

            let walked = body();

            let mut raw: rt::Graph = core::ptr::null_mut();
            // SAFETY: `raw` is a live local; the capture was begun above and
            // is ended here on every path.
            let ended = unsafe {
                crate::device::ctx::check(
                    "hipStreamEndCapture",
                    rt::hipStreamEndCapture(stream.cast(), &raw mut raw),
                )
            };

            let graph = (!raw.is_null()).then(|| Graph { raw: raw.cast() });
            walked?;
            ended?;
            graph.ok_or(Fault::Device {
                call: "hipStreamEndCapture",
                code: 0,
            })
        }
        #[cfg(not(feature = "rocm"))]
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
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            let Ok(c_path) = std::ffi::CString::new(path) else {
                return false;
            };
            let flags = 1 | 4 | 512;
            // SAFETY: `raw` is this graph's live handle; `c_path` outlives the call.
            let code =
                unsafe { rt::hipGraphDebugDotPrint(self.raw.cast(), c_path.as_ptr(), flags) };
            code == rt::hipSuccess
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = path;
            false
        }
    }

    #[must_use]
    pub fn nodes(&self) -> Option<usize> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt as dr;

            let mut count: usize = 0;
            // SAFETY: null node array + live count is the documented way to
            // ask for count alone; `raw` is this graph's handle
            // (`cudaGraph_t`/`CUgraph` are one pointer).
            let code = unsafe {
                dr::hipGraphGetNodes(self.raw.cast(), core::ptr::null_mut(), &raw mut count)
            };
            (code == dr::hipSuccess).then_some(count)
        }
        #[cfg(not(feature = "rocm"))]
        {
            None
        }
    }

    #[must_use]
    pub fn edges(&self) -> Option<usize> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt as dr;

            let mut count: usize = 0;
            // SAFETY: null endpoint/edge-data arrays + live count asks for
            // count alone, on this graph's own handle.
            let code = unsafe {
                dr::hipGraphGetEdges(
                    self.raw.cast(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &raw mut count,
                )
            };
            (code == dr::hipSuccess).then_some(count)
        }
        #[cfg(not(feature = "rocm"))]
        {
            None
        }
    }

    /// Device bytes one instantiation of this graph holds, averaged over
    /// `copies` held at once.
    pub fn instantiated_bytes(&self, copies: usize) -> Result<f64> {
        let before = crate::device::free_bytes().unwrap_or(0);
        let held = (0..copies)
            .map(|_| self.instantiate(core::ptr::null_mut()))
            .collect::<Result<Vec<_>>>()?;
        let after = crate::device::free_bytes().unwrap_or(0);
        drop(held);
        Ok(before.saturating_sub(after) as f64 / copies as f64)
    }

    pub fn instantiate(&self, stream: *mut c_void) -> Result<GraphExec> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            let mut raw: rt::GraphExec = core::ptr::null_mut();
            // SAFETY: `raw` is a live local; `self.raw` is this graph's
            // handle. Uses `cudaGraphInstantiateWithFlags`, not plain
            // `cudaGraphInstantiate`, since only the flagged form is spelled
            // the same way under both runtimes this crate builds against.
            unsafe {
                crate::device::ctx::check(
                    "hipGraphInstantiateWithFlags",
                    rt::hipGraphInstantiateWithFlags(&raw mut raw, self.raw.cast(), 0),
                )?;
            }
            let exec = GraphExec {
                raw: raw.cast(),
                nodes: self.nodes().unwrap_or(0),
            };
            // SAFETY: the exec was just created and the stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "hipGraphUpload",
                    rt::hipGraphUpload(raw, stream.cast()),
                )?;
            }
            Ok(exec)
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl GraphExec {
    pub fn launch(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            // SAFETY: the exec is this handle's, alive until `Drop`, and the
            // stream is the shell's.
            unsafe {
                crate::device::ctx::check(
                    "hipGraphLaunch",
                    kernels_rocm::hip::rt::hipGraphLaunch(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
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
        #[cfg(feature = "rocm")]
        if !self.raw.is_null() {
            // SAFETY: handle came from this graph's own capture, destroyed
            // once; an exec instantiated from it does not borrow it.
            unsafe {
                let _ = kernels_rocm::hip::rt::hipGraphDestroy(self.raw.cast());
            }
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        if !self.raw.is_null() {
            // SAFETY: destroyed once, and the shell synchronizes its stream
            // before it drops a cache entry (`record.rs`'s eviction).
            unsafe {
                let _ = kernels_rocm::hip::rt::hipGraphExecDestroy(self.raw.cast());
            }
        }
    }
}

#[derive(Debug)]
pub struct Event {
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    raw: *mut c_void,
}

impl Event {
    pub fn new() -> Result<Event> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            let mut raw: rt::Event = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter and this thread bound the
            // device.
            unsafe {
                crate::device::ctx::check(
                    "hipEventCreateWithFlags",
                    rt::hipEventCreateWithFlags(&raw mut raw, 2),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn timing() -> Result<Event> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            let mut raw: rt::Event = core::ptr::null_mut();
            // SAFETY: `raw` is a live out-parameter; this thread bound the
            // device. Flag 0 is `cudaEventDefault` (timing enabled).
            unsafe {
                crate::device::ctx::check(
                    "hipEventCreateWithFlags",
                    rt::hipEventCreateWithFlags(&raw mut raw, 0),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn done(&self) -> Result<bool> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::hipEventQuery(self.raw.cast()) };
            match status {
                rt::hipSuccess => Ok(true),
                rt::hipErrorNotReady => {
                    #[allow(unused_must_use)]
                    unsafe {
                        rt::hipGetLastError();
                    }
                    Ok(false)
                }
                code => Err(Fault::Device {
                    call: "hipEventQuery",
                    code,
                }),
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn settle(&self) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::hipEventSynchronize(self.raw.cast()) };
            if status != rt::hipSuccess {
                return Err(Fault::Device {
                    call: "hipEventSynchronize",
                    code: status as i32,
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn record(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "hipEventRecord",
                    kernels_rocm::hip::rt::hipEventRecord(self.raw.cast(), stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    pub fn wait(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "hipStreamWaitEvent",
                    kernels_rocm::hip::rt::hipStreamWaitEvent(stream.cast(), self.raw.cast(), 0),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        if !self.raw.is_null() {
            // SAFETY: created by this handle's `new`, destroyed once, and the
            // shell synchronizes before it tears a load down.
            unsafe {
                let _ = kernels_rocm::hip::rt::hipEventDestroy(self.raw.cast());
            }
        }
    }
}
