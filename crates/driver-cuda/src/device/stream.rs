//! Streams and events.
//!
//! [`StreamRef`] is a borrowed, `Copy`, no-drop-glue `CUstream`: cudarc's
//! `CudaStream` destroys on drop with no borrowed form (#589). [`OwnedStream`] owns and destroys.

use std::marker::PhantomData;

use cudarc::runtime::sys::{
    cudaError, cudaEvent_t, cudaEventCreateWithFlags, cudaEventDestroy, cudaEventDisableTiming,
    cudaEventElapsedTime, cudaEventQuery, cudaEventRecord, cudaEventSynchronize,
    cudaLaunchHostFunc, cudaStream_t, cudaStreamCreateWithPriority, cudaStreamDestroy,
    cudaStreamNonBlocking, cudaStreamSynchronize, cudaStreamWaitEvent,
};

use crate::error::{Result, check_rt, ignore_in_drop};

/// A stream this crate does not own. The lifetime ties it to its owner — an
/// `OwnedStream` or C++ `CudaContext` — so a `StreamRef` cannot outlive what destroys it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRef<'a> {
    raw: cudaStream_t,
    _owner: PhantomData<&'a ()>,
}

// `CUstream` is usable from any thread — CUDA guarantees submission is thread-safe;
// only the raw pointer blocks the auto-impls.
unsafe impl Send for StreamRef<'_> {}
unsafe impl Sync for StreamRef<'_> {}

impl<'a> StreamRef<'a> {
    /// The raw handle, for a launcher's last argument.
    pub const fn as_raw(self) -> cudaStream_t {
        self.raw
    }

    /// Block the calling thread until every work item submitted so far has
    /// finished. Illegal during capture, so unreachable from [`crate::device::CaptureScope`].
    pub fn synchronize(self) -> Result<()> {
        check_rt(
            unsafe { cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }

    /// Make this stream wait on `event` without blocking the host.
    pub fn wait_event(self, event: &Event) -> Result<()> {
        check_rt(
            unsafe { cudaStreamWaitEvent(self.raw, event.raw, 0) },
            "cudaStreamWaitEvent",
        )
    }

    /// Enqueue a host callback in stream order — it runs once everything queued
    /// before it has retired.
    ///
    /// # Safety
    /// `f` runs on a CUDA-owned thread and must not call back into the CUDA
    /// runtime (no alloc/launch/sync); `data` must outlive the callback and be `Send`.
    pub unsafe fn host_fn(
        self,
        f: unsafe extern "C" fn(*mut std::ffi::c_void),
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        check_rt(
            unsafe { cudaLaunchHostFunc(self.raw, Some(f), data) },
            "cudaLaunchHostFunc",
        )
    }

    /// Record `event` at this point in this stream's order.
    pub fn record(self, event: &Event) -> Result<()> {
        check_rt(
            unsafe { cudaEventRecord(event.raw, self.raw) },
            "cudaEventRecord",
        )
    }
}

/// A stream this crate created and will destroy.
#[derive(Debug)]
pub struct OwnedStream {
    raw: cudaStream_t,
}

unsafe impl Send for OwnedStream {}
unsafe impl Sync for OwnedStream {}

impl OwnedStream {
    /// Create a non-blocking stream at the given priority (CUDA order: lower is
    /// higher, `0` default) — blocking would barrier on every unrelated default-stream copy.
    pub fn new(priority: i32) -> Result<Self> {
        let mut raw: cudaStream_t = std::ptr::null_mut();
        check_rt(
            unsafe { cudaStreamCreateWithPriority(&mut raw, cudaStreamNonBlocking, priority) },
            "cudaStreamCreateWithPriority",
        )?;
        Ok(Self { raw })
    }

    /// Borrow this stream. The returned reference cannot outlive `self`.
    pub fn as_ref(&self) -> StreamRef<'_> {
        StreamRef {
            raw: self.raw,
            _owner: PhantomData,
        }
    }
}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        ignore_in_drop(unsafe { cudaStreamDestroy(self.raw) });
    }
}

/// A CUDA event; timing is off by default since an event used only for
/// ordering (`cudaStreamWaitEvent`) is cheaper. [`Event::with_timing`] opts in.
#[derive(Debug)]
pub struct Event {
    raw: cudaEvent_t,
    timing: bool,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// An ordering-only event.
    pub fn new() -> Result<Self> {
        Self::create(cudaEventDisableTiming, false)
    }

    /// An event that can be measured with [`Event::elapsed_ms`].
    pub fn with_timing() -> Result<Self> {
        Self::create(0, true)
    }

    fn create(flags: u32, timing: bool) -> Result<Self> {
        let mut raw: cudaEvent_t = std::ptr::null_mut();
        check_rt(
            unsafe { cudaEventCreateWithFlags(&mut raw, flags) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Self { raw, timing })
    }

    /// Block until this event has been reached.
    pub fn synchronize(&self) -> Result<()> {
        check_rt(
            unsafe { cudaEventSynchronize(self.raw) },
            "cudaEventSynchronize",
        )
    }

    /// Has this event been reached? Does not block.
    pub fn is_complete(&self) -> Result<bool> {
        match unsafe { cudaEventQuery(self.raw) } {
            cudaError::cudaSuccess => Ok(true),
            cudaError::cudaErrorNotReady => Ok(false),
            code => Err(crate::Error::Runtime {
                call: "cudaEventQuery",
                code,
            }),
        }
    }

    /// Milliseconds between two recorded events. Refuses if either was created
    /// without timing, rather than surfacing an opaque CUDA error code.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        if !self.timing || !end.timing {
            return Err(crate::Error::invalid(
                "cudaEventElapsedTime",
                "both events must be created with `Event::with_timing`",
            ));
        }
        let mut ms = 0.0f32;
        check_rt(
            unsafe { cudaEventElapsedTime(&mut ms, self.raw, end.raw) },
            "cudaEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        ignore_in_drop(unsafe { cudaEventDestroy(self.raw) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_ref_is_a_word() {
        // No drop glue, one word — every launcher signature pays if this grows.
        assert_eq!(
            std::mem::size_of::<StreamRef<'_>>(),
            std::mem::size_of::<cudaStream_t>()
        );
        assert!(!std::mem::needs_drop::<StreamRef<'_>>());
    }

    #[test]
    fn elapsed_refuses_untimed_events_without_calling_cuda() {
        // Hand-built fake events; `ManuallyDrop` keeps them out of `Drop`/destroy.
        let a = std::mem::ManuallyDrop::new(Event {
            raw: std::ptr::null_mut(),
            timing: false,
        });
        let b = std::mem::ManuallyDrop::new(Event {
            raw: std::ptr::null_mut(),
            timing: true,
        });
        let err = a.elapsed_ms(&b).unwrap_err();
        assert_eq!(err.call(), "cudaEventElapsedTime");
    }
}

/// Pinned host memory: `cudaMemcpyAsync` into pageable memory blocks (the
/// runtime must stage it), so pinning keeps the run-ahead's async copies actually async.
pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

// Driver-owned host memory; the pointer is not thread-affine and crosses to a callback thread.
unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    /// Pin `len` bytes, or fail.
    pub fn new(len: usize) -> Result<Self> {
        use cudarc::runtime::sys::cudaMallocHost;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        check_rt(
            unsafe { cudaMallocHost(&mut p, len.max(1)) },
            "cudaMallocHost",
        )?;
        Ok(Self {
            ptr: p.cast::<u8>(),
            len,
        })
    }

    /// How many bytes are pinned.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Was this a zero-length request?
    ///
    /// No caller, and it stays for the reason `ConditionalIf::is_empty`
    /// states in its own doc: clippy asks for the pair, and a public `len`
    /// without one is an API shape the lint exists to prevent. Deleting it
    /// with the rest of this file's zero-reader surface turned
    /// `len_without_is_empty` on, which is the lint doing its job.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes, as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes, mutably — what a D2H writes into.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        use cudarc::runtime::sys::cudaFreeHost;
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFreeHost(self.ptr.cast()) };
        }
    }
}
