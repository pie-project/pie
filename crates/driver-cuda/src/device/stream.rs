//! Streams and events.
//!
//! # Why there is a borrowed stream type here
//!
//! cudarc's `CudaStream` owns its handle and destroys it on drop, and there is
//! no borrowed counterpart (cudarc issue #589). For a from-scratch program
//! that is fine. For this one it is disqualifying, and on day one rather than
//! eventually: while `driver-cuda` and this crate both exist, a subsystem
//! ported to Rust runs on streams the C++ `CudaContext` created, and a
//! subsystem still in C++ runs on streams this crate created. A type that
//! destroys what it is handed cannot be pointed at either.
//!
//! So the borrowed form is the primary one. [`StreamRef`] is a `CUstream` plus
//! a lifetime and nothing else -- `Copy`, no drop glue, no ownership claim --
//! and every API in this crate that needs a stream takes one. [`OwnedStream`]
//! is a thin thing on top that adds the destructor, and it hands out
//! `StreamRef`s. Ownership is a property of one end of the migration, not of
//! the stream.

use std::marker::PhantomData;

use cudarc::runtime::sys::{
    cudaError, cudaEvent_t, cudaEventCreateWithFlags, cudaEventDestroy, cudaEventDisableTiming,
    cudaEventElapsedTime, cudaEventQuery, cudaEventRecord, cudaEventSynchronize,
    cudaLaunchHostFunc, cudaStream_t, cudaStreamCreateWithPriority, cudaStreamDestroy,
    cudaStreamNonBlocking, cudaStreamQuery, cudaStreamSynchronize, cudaStreamWaitEvent,
};

use crate::error::{Result, check_rt, ignore_in_drop};

/// A stream this crate does not own.
///
/// The lifetime is the whole safety story: it ties the handle to whatever does
/// own it, so a `StreamRef` cannot outlive the `OwnedStream` -- or the C++
/// `CudaContext` -- that will destroy it.
///
/// `Copy`, because a stream handle is an identifier and passing one to a
/// launch should read like passing an integer, which is what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRef<'a> {
    raw: cudaStream_t,
    _owner: PhantomData<&'a ()>,
}

// A `CUstream` is usable from any thread; the CUDA docs make thread-safety of
// stream submission an explicit guarantee. The raw pointer is what makes the
// auto-impls bail, not anything about the semantics.
unsafe impl Send for StreamRef<'_> {}
unsafe impl Sync for StreamRef<'_> {}

impl<'a> StreamRef<'a> {
    /// Borrow a stream created elsewhere -- by [`OwnedStream`], or by the C++
    /// shell across the FFI.
    ///
    /// # Safety
    ///
    /// `raw` must be a live `cudaStream_t` that stays live for `'a`, and the
    /// caller is asserting that lifetime rather than proving it. This is the
    /// one place the C++/Rust seam is unchecked, which is why it is the only
    /// `unsafe fn` in the module: everything downstream takes a `StreamRef`
    /// and is safe because this call was made correctly once.
    pub const unsafe fn from_raw(raw: cudaStream_t) -> Self {
        Self {
            raw,
            _owner: PhantomData,
        }
    }

    /// The default (`NULL`) stream.
    ///
    /// `'static` because the default stream is not destroyed and outlives
    /// anything that could hold it. The C++ `DeviceBuffer` uses exactly this
    /// stream for every copy it makes.
    pub const fn null() -> StreamRef<'static> {
        StreamRef {
            raw: std::ptr::null_mut(),
            _owner: PhantomData,
        }
    }

    /// The raw handle, for a launcher's last argument.
    pub const fn as_raw(self) -> cudaStream_t {
        self.raw
    }

    /// Block the calling thread until every work item submitted so far has
    /// finished.
    ///
    /// Illegal while a capture is open on this stream, which is why it is not
    /// reachable from inside a [`crate::device::CaptureScope`].
    pub fn synchronize(self) -> Result<()> {
        check_rt(
            unsafe { cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }

    /// Has everything submitted so far finished? Does not block.
    pub fn is_idle(self) -> Result<bool> {
        match unsafe { cudaStreamQuery(self.raw) } {
            cudaError::cudaSuccess => Ok(true),
            cudaError::cudaErrorNotReady => Ok(false),
            code => Err(crate::Error::Runtime {
                call: "cudaStreamQuery",
                code,
            }),
        }
    }

    /// Make this stream wait on `event` without blocking the host.
    pub fn wait_event(self, event: &Event) -> Result<()> {
        check_rt(
            unsafe { cudaStreamWaitEvent(self.raw, event.raw, 0) },
            "cudaStreamWaitEvent",
        )
    }

    /// Enqueue a HOST callback in stream order — it runs when everything
    /// queued before it has retired.
    ///
    /// This is what lets a fire complete without the next call coming to
    /// collect it. An event plus a poll would need someone to poll, and
    /// "someone" is either a thread or the next launch; the second hangs a
    /// stream that goes quiet, and the first is a thread this driver does
    /// not otherwise need.
    ///
    /// # Safety
    ///
    /// `f` runs on a CUDA-owned thread, and CUDA forbids calling back into
    /// the runtime from it — no allocation on a device, no launch, no
    /// synchronize. `data` must outlive the callback and be `Send`; the
    /// caller owns keeping it alive, which in practice means leaking a
    /// `Box` into the call and reclaiming it inside.
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
    /// Create a non-blocking stream at the given priority.
    ///
    /// `cudaStreamNonBlocking` rather than the default, matching the C++
    /// shell: a stream that implicitly synchronizes with the NULL stream turns
    /// every unrelated default-stream copy into a barrier, which is precisely
    /// what a multi-stream shell is built to avoid.
    ///
    /// Priority is CUDA's convention, where LOWER is higher-priority. Pass
    /// `0` for the default; the valid range is device-specific.
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

    /// Give up ownership, returning the raw handle.
    ///
    /// The caller becomes responsible for `cudaStreamDestroy`. This is the
    /// handoff in the other direction: a stream created here and passed to the
    /// C++ shell to own for the rest of a migration step.
    ///
    /// `ManuallyDrop` rather than `mem::forget` -- same effect, but it says
    /// "this value's destructor is deliberately suppressed" at the point of
    /// construction instead of leaving a bare `forget` for a reader to
    /// classify as a leak.
    pub fn into_raw(self) -> cudaStream_t {
        std::mem::ManuallyDrop::new(self).raw
    }
}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        ignore_in_drop(unsafe { cudaStreamDestroy(self.raw) });
    }
}

/// A CUDA event.
///
/// Timing is off by default (`cudaEventDisableTiming`), which is what the
/// shell wants for the ordinary case: an event used only for
/// `cudaStreamWaitEvent` is cheaper without it, and the C++ shell creates 26
/// of its events with exactly that flag. [`Event::with_timing`] opts back in.
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

    /// The raw handle.
    pub const fn as_raw(&self) -> cudaEvent_t {
        self.raw
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

    /// Milliseconds between two recorded events.
    ///
    /// Refuses rather than returning nonsense if either event was created
    /// without timing -- CUDA's own answer there is
    /// `cudaErrorInvalidResourceHandle`, which says nothing about which of the
    /// two was wrong.
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
    fn the_null_stream_is_static_and_null() {
        let s = StreamRef::null();
        assert!(s.as_raw().is_null());
    }

    #[test]
    fn a_stream_ref_is_a_word() {
        // The whole claim of the type: no drop glue, no ownership, cheap to
        // pass. If this ever grows, every launcher signature pays for it.
        assert_eq!(
            std::mem::size_of::<StreamRef<'_>>(),
            std::mem::size_of::<cudaStream_t>()
        );
        assert!(!std::mem::needs_drop::<StreamRef<'_>>());
    }

    #[test]
    fn borrowing_a_raw_handle_round_trips() {
        // The C++ seam, exercised without CUDA: `from_raw` must hand back
        // exactly what it was given, because a launcher's last argument is
        // that value and nothing else.
        let fake = 0xdead_beefusize as cudaStream_t;
        let borrowed = unsafe { StreamRef::from_raw(fake) };
        assert_eq!(borrowed.as_raw(), fake);
    }

    #[test]
    fn elapsed_refuses_untimed_events_without_calling_cuda() {
        // Constructed by hand rather than through `Event::new`, so this runs
        // on a box with no CUDA: the point is that the refusal happens BEFORE
        // `cudaEventElapsedTime` is reached. `ManuallyDrop` for the same
        // reason -- these are not real events, so their destructors must not
        // reach `cudaEventDestroy` either.
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

/// PINNED HOST MEMORY, and the reason the run-ahead needs it.
///
/// `cudaMemcpyAsync` into PAGEABLE host memory is asynchronous in name
/// only: the runtime must stage it, so the call blocks until the copy
/// completes. A fire that D2H's its logits into a `Vec` therefore drains
/// its own stream inside `pie_cuda_launch`, which is precisely the
/// synchronization run-ahead exists to remove —
/// `a_launch_returns_before_its_fire_retires` measured it doing so.
///
/// Pinned memory is what makes the copy actually asynchronous, and it is
/// why the retired C++ tree sized a pinned staging pool from
/// `runahead.hpp` rather than allocating per fire. This is the one-slot
/// version of that pool: enough for the property, and the shape the pool
/// grows from when the depth does.
pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

// The buffer is host memory the driver owns; nothing about the pointer is
// thread-affine, and the debt it belongs to crosses to a callback thread.
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
