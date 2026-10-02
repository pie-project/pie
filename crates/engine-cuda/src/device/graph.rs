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

    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn debug_dot(&self, path: &str) -> bool {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;
            let Ok(c_path) = std::ffi::CString::new(path) else {
                return false;
            };
            let flags = 1 | 4 | 512;
            // SAFETY: `raw` is this graph's live handle; `c_path` outlives the call.
            let code =
                unsafe { rt::cudaGraphDebugDotPrint(self.raw.cast(), c_path.as_ptr(), flags) };
            code == rt::cudaError_t::cudaSuccess
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = path;
            false
        }
    }

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

pub fn capture_frontier(stream: *mut c_void) -> Result<Vec<*mut c_void>> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

        let mut status = dr::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id: u64 = 0;
        let mut graph: dr::CUgraph = core::ptr::null_mut();
        let mut deps: *const dr::CUgraphNode = core::ptr::null();
        let mut dep_count: usize = 0;
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

#[derive(Debug)]
pub struct Event {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    raw: *mut c_void,
}

impl Event {
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
                    rt::cudaEventCreateWithFlags(&raw mut raw, 2),
                )?;
            }
            Ok(Event { raw: raw.cast() })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

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

    pub fn done(&self) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;
            // SAFETY: the handle is live and this crate created it.
            let status = unsafe { rt::cudaEventQuery(self.raw.cast()) };
            match status {
                rt::cudaError::cudaSuccess => Ok(true),
                rt::cudaError::cudaErrorNotReady => {
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

    pub fn wait(&self, stream: *mut c_void) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: both handles are the shell's and live for the call.
            unsafe {
                crate::device::ctx::check(
                    "cudaStreamWaitEvent",
                    cudarc::runtime::sys::cudaStreamWaitEvent(stream.cast(), self.raw.cast(), 0),
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
