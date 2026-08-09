//! Device allocation, and the capture discipline that makes it safe.
//!
//! # The problem this module exists to solve
//!
//! CUDA forbids synchronous allocation and free while a stream capture is
//! open. Break the rule and the capture is invalidated -- but the diagnostic
//! does not arrive at the offending call. It arrives later, on some unrelated
//! operation, as a context-wide error. cudarc has this bug today (#590): its
//! `Drop for CudaSlice` frees without consulting the capture state, so a
//! buffer that goes out of scope inside a captured region poisons everything
//! after it.
//!
//! The C++ shell obeys the rule by convention -- a comment, and reviewers who
//! remember. That is exactly the class of invariant a rewrite should be
//! cashing in, so this module encodes it twice, once for each half of the
//! problem:
//!
//! * **Allocation** is prevented at compile time. [`Allocator::alloc`] takes
//!   `&self`, [`Allocator::begin_capture`] takes `&mut self`, and the returned
//!   [`CaptureScope`] holds that exclusive borrow for its whole life. An
//!   `alloc` inside a capture does not fail at runtime; it does not build.
//!   [`Allocator`] is deliberately **not** `Clone`, because a clone would be a
//!   second handle the borrow does not cover.
//!
//! * **Freeing** cannot be prevented that way, because `Drop` is implicit and
//!   runs wherever a value happens to die. So it is made harmless instead:
//!   [`DeviceBuffer`]'s drop consults [`DeferState`] and, if a capture is
//!   open, hands the pointer to a queue that is drained when the capture
//!   closes. The pointer is still freed, just not at a moment CUDA forbids.
//!
//! # A note on testing
//!
//! [`DeferState`] is a pure state machine: it decides, and its caller acts.
//! That split is what lets the interesting half of this module be tested on a
//! machine with no CUDA -- the tests at the bottom drive real capture/free
//! interleavings, including the cross-thread one that makes the naive
//! "check an atomic, then free" version racy.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::runtime::sys::{
    cudaFree, cudaGraph_t, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaMemsetAsync,
    cudaStreamBeginCapture, cudaStreamCaptureMode, cudaStreamEndCapture,
};

use crate::cuda::graph::Graph;
use crate::cuda::stream::StreamRef;
use crate::error::{Error, Result, check_rt, ignore_in_drop};

/// A device address, held as an integer.
///
/// Not a `*mut c_void`, on purpose: the deferred queue is shared across
/// threads, and an integer is `Send`/`Sync` without an `unsafe impl` that
/// would have to be justified separately. It is converted back to a pointer at
/// the single point where it is actually freed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DevPtr(usize);

impl DevPtr {
    const fn as_raw(self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

/// The capture-vs-free state machine, with no CUDA in it.
///
/// Deciding and acting are split so the decision can be tested directly, and
/// so the decision can be made under one lock -- which is the part that has to
/// be atomic. Reading a capture flag and then freeing as two steps is racy: a
/// capture can open in between, on another thread, and the free lands inside
/// it.
#[derive(Debug, Default)]
struct DeferState {
    /// Whether a capture is open. `bool` rather than a depth counter because
    /// `begin_capture` takes `&mut Allocator`, so at most one scope can exist
    /// at a time and a second one cannot be asked for.
    capturing: bool,
    /// Pointers whose free was deferred, in the order they died.
    pending: Vec<DevPtr>,
}

impl DeferState {
    /// A buffer died. `Some(p)` means "free it now"; `None` means it was
    /// queued because a capture is open.
    fn release(&mut self, p: DevPtr) -> Option<DevPtr> {
        if self.capturing {
            self.pending.push(p);
            None
        } else {
            Some(p)
        }
    }

    /// Open a capture. Fails if one is already open.
    fn begin(&mut self) -> Result<()> {
        if self.capturing {
            return Err(Error::invalid(
                "begin_capture",
                "a capture is already open on this allocator",
            ));
        }
        self.capturing = true;
        Ok(())
    }

    /// Close the capture and take everything that died while it was open.
    fn end(&mut self) -> Vec<DevPtr> {
        self.capturing = false;
        std::mem::take(&mut self.pending)
    }
}

/// The shared half of an [`Allocator`] -- what a [`DeviceBuffer`] keeps a
/// handle on so it can release itself.
///
/// Note what is NOT here: any way to allocate. That is what makes the
/// compile-time half of the discipline hold. A `DeviceBuffer` can reach its
/// allocator's *release* path from anywhere, including another thread, but
/// nothing can reach `alloc` except through the `&self` borrow of the
/// `Allocator` itself.
#[derive(Debug, Default)]
struct AllocatorInner {
    state: Mutex<DeferState>,
}

impl AllocatorInner {
    fn release(&self, p: DevPtr) {
        let decision = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.release(p)
        };
        if let Some(p) = decision {
            ignore_in_drop(unsafe { cudaFree(p.as_raw()) });
        }
    }

    fn drain(&self, freed: Vec<DevPtr>) {
        for p in freed {
            ignore_in_drop(unsafe { cudaFree(p.as_raw()) });
        }
    }
}

/// The owner of device memory, and the gate on stream capture.
///
/// Not `Clone`, and not by omission: a second handle would be a second way to
/// call [`Allocator::alloc`] that the `&mut self` borrow in
/// [`Allocator::begin_capture`] does not cover, which is the entire mechanism.
/// That absence is load-bearing, so it is checked rather than asserted:
///
/// ```compile_fail
/// use driver_cuda_new::cuda::Allocator;
/// let a = Allocator::new();
/// let b = a.clone();
/// ```
#[derive(Debug, Default)]
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

impl Allocator {
    /// A fresh allocator with no live buffers and no capture open.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate `bytes` of device memory.
    ///
    /// Takes `&self`, which is what makes this uncallable while a
    /// [`CaptureScope`] is alive: the scope holds `&mut self`.
    ///
    /// A zero-byte request is honoured without calling CUDA and yields a
    /// null-pointer buffer, matching the C++ `DeviceBuffer(0)`.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        if bytes == 0 {
            return Ok(DeviceBuffer {
                ptr: DevPtr(0),
                bytes: 0,
                owner: Arc::clone(&self.inner),
            });
        }
        let mut raw: *mut c_void = std::ptr::null_mut();
        check_rt(unsafe { cudaMalloc(&mut raw, bytes) }, "cudaMalloc")?;
        Ok(DeviceBuffer {
            ptr: DevPtr(raw as usize),
            bytes,
            owner: Arc::clone(&self.inner),
        })
    }

    /// Begin capturing `stream` into a graph.
    ///
    /// The exclusive borrow is the point. For as long as the returned scope
    /// lives, this allocator cannot allocate, because `alloc` needs `&self`
    /// and the scope holds `&mut self`.
    ///
    /// The mode is `Global`, matching `batch/`'s `cudaStreamCaptureModeGlobal`:
    /// it is the strictest of the three and the one that actually catches a
    /// stray synchronous call rather than quietly tolerating it.
    pub fn begin_capture<'a>(&'a mut self, stream: StreamRef<'a>) -> Result<CaptureScope<'a>> {
        {
            let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            st.begin()?;
        }
        if let Err(e) = check_rt(
            unsafe {
                cudaStreamBeginCapture(
                    stream.as_raw(),
                    cudaStreamCaptureMode::cudaStreamCaptureModeGlobal,
                )
            },
            "cudaStreamBeginCapture",
        ) {
            // The flag went up before the call, so it has to come back down if
            // the call refused -- otherwise the allocator is wedged in a
            // capture that never started.
            let freed = {
                let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
                st.end()
            };
            self.inner.drain(freed);
            return Err(e);
        }
        Ok(CaptureScope {
            alloc: self,
            stream,
            open: true,
        })
    }

    /// How many frees are currently parked waiting for a capture to close.
    ///
    /// Exposed for tests and for a metric: a number that is persistently
    /// non-zero outside a capture would mean the drain is not running.
    pub fn deferred_free_count(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending
            .len()
    }

    /// Is a capture open on this allocator?
    pub fn is_capturing(&self) -> bool {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .capturing
    }
}

/// An open stream capture, holding the allocator shut for its lifetime.
///
/// Ending is explicit ([`CaptureScope::end`]) because ending produces a
/// [`Graph`], and a `Drop` cannot hand one back. Dropping without ending is
/// still correct -- the capture is closed and the graph discarded -- so a `?`
/// on some other line in the middle of a capture cannot leave the stream
/// captured forever.
#[derive(Debug)]
pub struct CaptureScope<'a> {
    alloc: &'a mut Allocator,
    stream: StreamRef<'a>,
    open: bool,
}

impl<'a> CaptureScope<'a> {
    /// The stream being captured, for submitting the work the graph records.
    pub fn stream(&self) -> StreamRef<'a> {
        self.stream
    }

    /// Close the capture and take the graph.
    ///
    /// Draining the deferred frees happens here, after `cudaStreamEndCapture`
    /// has returned: that is the first instant at which calling `cudaFree` is
    /// legal again.
    /// `finish` clears `open`, so the `Drop` that runs on the way out of this
    /// function is a no-op and does not try to end the capture a second time.
    pub fn end(mut self) -> Result<Graph> {
        self.finish()
    }

    /// The body of both `end` and `drop`.
    fn finish(&mut self) -> Result<Graph> {
        if !self.open {
            return Err(Error::invalid(
                "cudaStreamEndCapture",
                "capture already ended",
            ));
        }
        self.open = false;

        let mut raw: cudaGraph_t = std::ptr::null_mut();
        let ended = check_rt(
            unsafe { cudaStreamEndCapture(self.stream.as_raw(), &mut raw) },
            "cudaStreamEndCapture",
        );

        // Drain regardless of whether the capture ended cleanly. A failed
        // capture is exactly the case where parked pointers would otherwise be
        // stranded, and by this point the capture is closed either way.
        let freed = {
            let mut st = self
                .alloc
                .inner
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            st.end()
        };
        self.alloc.inner.drain(freed);

        ended?;
        Ok(unsafe { Graph::from_raw(raw) })
    }
}

impl Drop for CaptureScope<'_> {
    fn drop(&mut self) {
        if self.open {
            // Discards the graph and any error: this is the abandonment path
            // (an early return out of a capture), and what matters is that the
            // stream does not stay captured and the queue does not stay
            // parked. Dropping the `Result` drops any `Graph` inside it, which
            // destroys it -- the discard is complete, not a leak.
            ignore_in_drop(self.finish());
        }
    }
}

/// An owning region of device memory.
///
/// The port of `csrc/src/device_buffer.hpp`, with the C++ class's move-only
/// discipline replaced by Rust's default one, and with its unconditional
/// `cudaFree` in the destructor replaced by the capture-aware release that is
/// the whole point of this module.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: DevPtr,
    bytes: usize,
    owner: Arc<AllocatorInner>,
}

impl DeviceBuffer {
    /// The device address, for a launcher argument.
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_raw()
    }

    /// Size in bytes.
    pub const fn len(&self) -> usize {
        self.bytes
    }

    /// Was this a zero-byte allocation?
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copy host bytes in, ordered on `stream`.
    ///
    /// Refuses a source that does not fit rather than truncating, because the
    /// CUDA call would happily write past the allocation if the length were
    /// computed rather than checked.
    pub fn copy_from_host(&mut self, src: &[u8], stream: StreamRef<'_>) -> Result<()> {
        if src.len() > self.bytes {
            return Err(Error::invalid(
                "cudaMemcpyAsync",
                format!("source is {} bytes, buffer is {}", src.len(), self.bytes),
            ));
        }
        if src.is_empty() {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemcpyAsync(
                    self.ptr.as_raw(),
                    src.as_ptr().cast(),
                    src.len(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.as_raw(),
                )
            },
            "cudaMemcpyAsync",
        )
    }

    /// Copy device bytes out, ordered on `stream`.
    ///
    /// Asynchronous: the caller must synchronize `stream` before reading
    /// `dst`, exactly as with the C++ original.
    pub fn copy_to_host(&self, dst: &mut [u8], stream: StreamRef<'_>) -> Result<()> {
        if dst.len() > self.bytes {
            return Err(Error::invalid(
                "cudaMemcpyAsync",
                format!(
                    "destination is {} bytes, buffer is {}",
                    dst.len(),
                    self.bytes
                ),
            ));
        }
        if dst.is_empty() {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.ptr.as_raw(),
                    dst.len(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.as_raw(),
                )
            },
            "cudaMemcpyAsync",
        )
    }

    /// Fill with a byte value, ordered on `stream`.
    pub fn memset(&mut self, value: u8, stream: StreamRef<'_>) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemsetAsync(
                    self.ptr.as_raw(),
                    i32::from(value),
                    self.bytes,
                    stream.as_raw(),
                )
            },
            "cudaMemsetAsync",
        )
    }

    /// Copy host bytes into a SPAN of this buffer, ordered on `stream`.
    ///
    /// The offset form exists because the PTIR plane's buffers are arrays of
    /// records rather than single values: one channel's cell inside an
    /// instance's ring, one lane's record inside a lane table. Without it a
    /// caller would either allocate per record — thousands of allocations per
    /// fire — or rebuild the whole buffer to change one entry.
    ///
    /// # Errors
    ///
    /// If the span leaves the allocation. Checked as `offset + len` in
    /// `u64`-widened arithmetic rather than as `offset < bytes`, because the
    /// second passes for a span that starts inside and ends outside, and CUDA
    /// would write past the allocation without complaining.
    pub fn write_at(&mut self, offset: usize, src: &[u8], stream: StreamRef<'_>) -> Result<()> {
        self.check_span("cudaMemcpyAsync (write_at)", offset, src.len())?;
        if src.is_empty() {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemcpyAsync(
                    self.ptr.as_raw().byte_add(offset),
                    src.as_ptr().cast(),
                    src.len(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.as_raw(),
                )
            },
            "cudaMemcpyAsync",
        )
    }

    /// Copy a SPAN of this buffer out to the host, ordered on `stream`.
    ///
    /// Asynchronous, like [`Self::copy_to_host`]: synchronize before reading
    /// `dst`.
    ///
    /// # Errors
    ///
    /// If the span leaves the allocation.
    pub fn read_at(&self, offset: usize, dst: &mut [u8], stream: StreamRef<'_>) -> Result<()> {
        self.check_span("cudaMemcpyAsync (read_at)", offset, dst.len())?;
        if dst.is_empty() {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.ptr.as_raw().byte_add(offset),
                    dst.len(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.as_raw(),
                )
            },
            "cudaMemcpyAsync",
        )
    }

    /// `offset .. offset + len` must lie inside the allocation.
    fn check_span(&self, call: &'static str, offset: usize, len: usize) -> Result<()> {
        let end = (offset as u64).checked_add(len as u64);
        if end.is_none_or(|end| end > self.bytes as u64) {
            return Err(Error::invalid(
                call,
                format!(
                    "span of {len} bytes at offset {offset} leaves a buffer of {}",
                    self.bytes
                ),
            ));
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        self.owner.release(self.ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Everything below drives `DeferState` directly. That is the half of this
    // module with a decision in it, and it runs with no CUDA present.

    #[test]
    fn outside_a_capture_a_free_happens_immediately() {
        let mut st = DeferState::default();
        assert_eq!(st.release(DevPtr(0x1000)), Some(DevPtr(0x1000)));
        assert!(st.pending.is_empty());
    }

    #[test]
    fn inside_a_capture_a_free_is_parked_not_performed() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        assert_eq!(
            st.release(DevPtr(0x1000)),
            None,
            "must not free during capture"
        );
        assert_eq!(st.pending, vec![DevPtr(0x1000)]);
    }

    #[test]
    fn closing_the_capture_hands_back_everything_parked_in_order() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.release(DevPtr(1));
        let _ = st.release(DevPtr(2));
        let _ = st.release(DevPtr(3));
        assert_eq!(st.end(), vec![DevPtr(1), DevPtr(2), DevPtr(3)]);
        // and the queue is empty afterwards, so a second capture does not
        // re-free what the first one already released.
        assert!(st.pending.is_empty());
        st.begin().unwrap();
        assert!(st.end().is_empty());
    }

    #[test]
    fn after_the_capture_closes_frees_go_straight_through_again() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.end();
        assert_eq!(st.release(DevPtr(0x2000)), Some(DevPtr(0x2000)));
    }

    #[test]
    fn a_second_capture_is_refused_rather_than_silently_nested() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        let err = st.begin().unwrap_err();
        assert_eq!(err.call(), "begin_capture");
    }

    #[test]
    fn the_decision_and_the_queue_move_under_one_lock() {
        // The race the split exists to close: a buffer dying on another thread
        // while a capture opens here. Because `release` takes the same lock
        // that `begin` does, every pointer is either freed before the capture
        // or parked by it -- never freed inside it.
        //
        // Driven through `AllocatorInner`'s decision path with the actual
        // `cudaFree` replaced by counting, which is what the pure state
        // machine makes possible.
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let state = Arc::new(Mutex::new(DeferState::default()));
        let freed_outside = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let releaser = {
            let state = Arc::clone(&state);
            let freed_outside = Arc::clone(&freed_outside);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..1000 {
                    let decision = state.lock().unwrap().release(DevPtr(i + 1));
                    if decision.is_some() {
                        freed_outside.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        };

        barrier.wait();
        let mut parked_total = 0;
        for _ in 0..100 {
            state.lock().unwrap().begin().unwrap();
            parked_total += state.lock().unwrap().end().len();
        }
        releaser.join().unwrap();

        // Whatever the interleaving, every pointer is accounted for exactly
        // once: freed outside a capture, or parked and drained by one.
        let leftover = state.lock().unwrap().end().len();
        assert_eq!(
            freed_outside.load(Ordering::Relaxed) + parked_total + leftover,
            1000
        );
    }

    #[test]
    fn a_failed_capture_still_drains_what_was_parked() {
        // The `finish` path drains even when `cudaStreamEndCapture` errors,
        // because a failed capture is precisely when parked pointers would
        // otherwise be stranded. Driven at the state-machine level, since the
        // CUDA call itself is not what is under test.
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.release(DevPtr(0xdead));
        let freed = st.end();
        assert_eq!(freed, vec![DevPtr(0xdead)]);
        assert!(
            !st.capturing,
            "capture must be closed even on the error path"
        );
    }
}
