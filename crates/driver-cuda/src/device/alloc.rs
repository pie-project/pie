//! Device allocation, and the capture discipline that makes it safe. CUDA
//! forbids synchronous alloc/free during a stream capture (cudarc #590).
//! Alloc is compile-time blocked (`&self` vs `begin_capture`'s `&mut self`,
//! `Allocator: !Clone`); free can't be, so a dropped [`DeviceBuffer`] defers
//! its pointer via [`DeferState`] until capture ends.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::runtime::sys::{
    cudaFree, cudaGraph_t, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaMemsetAsync,
    cudaStreamBeginCapture, cudaStreamCaptureMode, cudaStreamEndCapture,
};

use crate::device::graph::Graph;
use crate::device::stream::StreamRef;
use crate::error::{Error, Result, check_rt, ignore_in_drop};

/// A device address, held as an integer so the shared deferred queue is
/// `Send`/`Sync` without an `unsafe impl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DevPtr(usize);

impl DevPtr {
    const fn as_raw(self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

/// The capture-vs-free state machine, with no CUDA in it.
///
/// Deciding and acting are split under one lock: reading a capture flag then
/// freeing is racy, since a capture can open on another thread in between.
#[derive(Debug, Default)]
struct DeferState {
    /// Whether a capture is open — a `bool`, not a counter, since only one
    /// scope can exist at a time.
    capturing: bool,
    /// Pointers whose free was deferred, in the order they died.
    pending: Vec<DevPtr>,
    /// Bytes handed out and not yet freed — what the shell owns, not device
    /// free memory (`cudaMemGetInfo` can't tell a leak from another consumer).
    live: usize,
    /// Bytes freed during a capture, not yet reclaimed; never draining this is a leak.
    deferred: usize,
}

impl DeferState {
    /// A buffer died. `Some(p)` means "free it now"; `None` means it was
    /// queued because a capture is open.
    fn release(&mut self, p: DevPtr, bytes: usize) -> Option<DevPtr> {
        self.live = self.live.saturating_sub(bytes);
        if self.capturing {
            self.pending.push(p);
            self.deferred = self.deferred.saturating_add(bytes);
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

/// The shared half of an [`Allocator`], the handle a [`DeviceBuffer`] keeps so
/// it can release itself from any thread. No `alloc` here: that stays behind `&self`.
#[derive(Debug, Default)]
struct AllocatorInner {
    state: Mutex<DeferState>,
}

impl AllocatorInner {
    fn release(&self, p: DevPtr, bytes: usize) {
        let decision = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.release(p, bytes)
        };
        if let Some(p) = decision {
            ignore_in_drop(unsafe { cudaFree(p.as_raw()) });
        }
    }

    fn drain(&self, freed: Vec<DevPtr>) {
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.deferred = 0;
        }
        for p in freed {
            ignore_in_drop(unsafe { cudaFree(p.as_raw()) });
        }
    }
}

/// The owner of device memory, and the gate on stream capture.
///
/// Not `Clone`: a second handle would be an uncovered path to
/// [`Allocator::alloc`] around `begin_capture`'s `&mut self` borrow.
#[derive(Debug, Default)]
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

impl Allocator {
    /// A fresh allocator with no live buffers and no capture open.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate `bytes` of device memory. Takes `&self`, so it is uncallable
    /// while a [`CaptureScope`] holds `&mut self`; zero bytes yields a null buffer.
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
        {
            let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            st.live = st.live.saturating_add(bytes);
        }
        Ok(DeviceBuffer {
            ptr: DevPtr(raw as usize),
            bytes,
            owner: Arc::clone(&self.inner),
        })
    }

    /// Bytes handed out and not yet freed — what a leak test reads (excludes deferred frees).
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .live
    }

    /// Bytes freed during an open capture and not yet reclaimed; nonzero with
    /// no capture open means a drain never ran.
    #[must_use]
    pub fn deferred_bytes(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .deferred
    }

    /// Begin capturing `stream` into a graph.
    ///
    /// The returned scope holds `&mut self`, so no allocation while it lives.
    /// Mode is `Global`, the strictest, to catch a stray sync call.
    ///
    /// A `fire::supergraph::warm()` call STOOD HERE: the conditional-node
    /// arming kernels had to be JIT-compiled BEFORE a capture opened,
    /// because their first resolve does an illegal `cudaFree(null)` inside
    /// one. Those kernels are deleted with the union supergraph. Any kernel a
    /// future capture arms will need the same warming, and this is where it
    /// goes.
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
            // Undo the flag on refusal, or the allocator wedges in a capture that never started.
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

    /// Frees parked waiting for a capture to close; persistently nonzero
    /// outside a capture means the drain isn't running.
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
/// Ending is explicit ([`CaptureScope::end`]) since `Drop` can't return the
/// [`Graph`] it makes; dropping without ending still closes the capture, so a
/// `?` mid-capture can't strand the stream.
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
    /// Deferred frees drain here, once `cudaStreamEndCapture` returns — the
    /// first instant `cudaFree` is legal again.
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

        // Drain even on failure: a failed capture is exactly when parked pointers strand.
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
            // Abandonment path: close the stream and drain; the graph and any error are discarded.
            ignore_in_drop(self.finish());
        }
    }
}

/// An owning region of device memory; `Drop` performs the capture-aware
/// release this module exists for.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: DevPtr,
    bytes: usize,
    owner: Arc<AllocatorInner>,
}

/// A device-to-host read from a raw base; prefer [`DeviceBuffer::read_at`]
/// when the caller owns the buffer (this is for a published weight's base).
///
/// # Safety
///
/// `src` must name at least `dst.len()` readable device bytes for the copy's
/// duration, and `stream` must outlive it.
pub unsafe fn read_raw_span(
    src: *const c_void,
    dst: &mut [u8],
    stream: StreamRef<'_>,
) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    check_rt(
        unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src,
                dst.len(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream.as_raw(),
            )
        },
        "cudaMemcpyAsync (read_raw_span)",
    )
}

/// A host-to-device write into a raw base; prefer [`DeviceBuffer::write_at`]
/// when the caller owns the buffer (this is for a fire destination by byte count).
///
/// # Safety
///
/// `dst` must name at least `src.len()` writable device bytes for the copy's
/// duration; pageable `src` is staged asynchronously, so keep it alive until
/// the stream is synchronised.
pub unsafe fn write_raw_span(dst: *mut c_void, src: &[u8], stream: StreamRef<'_>) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    check_rt(
        unsafe {
            cudaMemcpyAsync(
                dst,
                src.as_ptr().cast(),
                src.len(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream.as_raw(),
            )
        },
        "cudaMemcpyAsync (write_raw_span)",
    )
}

/// A device-to-device copy between two raw bases, which
/// [`DeviceBuffer::write_at`] cannot express.
///
/// # Safety
///
/// `src`/`dst` must each name `bytes` valid device memory (read/write
/// respectively) for the copy's duration, and `stream` must outlive it. The
/// spans must not overlap — `cudaMemcpyAsync` is not a `memmove`.
pub unsafe fn copy_raw_span(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_rt(
        unsafe {
            cudaMemcpyAsync(
                dst,
                src,
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream.as_raw(),
            )
        },
        "cudaMemcpyAsync (copy_raw_span)",
    )
}

/// A byte fill over a raw base that is the fire's, not one of ours; prefer
/// [`DeviceBuffer::memset`]/[`DeviceBuffer::memset_at`] when owned.
///
/// # Safety
///
/// `dst` must name at least `bytes` writable device bytes for the fill's
/// duration, and `stream` must outlive it.
pub unsafe fn fill_raw_span(
    dst: *mut c_void,
    value: u8,
    bytes: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_rt(
        unsafe { cudaMemsetAsync(dst, i32::from(value), bytes, stream.as_raw()) },
        "cudaMemsetAsync (fill_raw_span)",
    )
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

    /// The address `offset` bytes in, or `None` if `offset + len` leaves the
    /// allocation — checks `len` too, since in-bounds for an empty read can be
    /// out-of-bounds for the read that follows.
    pub fn ptr_at(&self, offset: usize, len: usize) -> Option<*mut c_void> {
        let end = offset.checked_add(len)?;
        (end <= self.bytes).then(|| self.ptr.as_raw().wrapping_byte_add(offset))
    }

    /// Copy host bytes in, ordered on `stream`; refuses an oversized source rather than truncating.
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

    /// Copy device bytes out, ordered on `stream` (async — synchronize before reading `dst`).
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

    /// Fill a span of this buffer with a byte value, ordered on `stream`.
    pub fn memset_at(
        &mut self,
        offset: usize,
        len: usize,
        value: u8,
        stream: StreamRef<'_>,
    ) -> Result<()> {
        self.check_span("cudaMemsetAsync (memset_at)", offset, len)?;
        if len == 0 {
            return Ok(());
        }
        check_rt(
            unsafe {
                cudaMemsetAsync(
                    self.ptr.as_raw().byte_add(offset),
                    i32::from(value),
                    len,
                    stream.as_raw(),
                )
            },
            "cudaMemsetAsync",
        )
    }

    /// Copy host bytes into a span of this buffer, ordered on `stream`. Bounds
    /// are checked as `offset + len` (`u64`-widened), not `offset < bytes` —
    /// which would let a span starting inside and ending outside write past.
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

    /// Copy a span of this buffer to the host, ordered on `stream` (async —
    /// synchronize before reading `dst`).
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
        self.owner.release(self.ptr, self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Everything below drives `DeferState` directly, with no CUDA present.

    /// A pure function of what the shell asked for and gave back, checkable with no GPU.
    #[test]
    fn the_live_count_follows_the_buffers_and_not_the_device() {
        let mut st = DeferState::default();
        st.live = 4096;
        // A plain free returns the bytes.
        assert_eq!(st.release(DevPtr(1), 1024), Some(DevPtr(1)));
        assert_eq!(st.live, 3072);
        assert_eq!(st.deferred, 0, "no capture, nothing deferred");

        // During a capture, `live` drops but nothing is reclaimed yet.
        st.begin().unwrap();
        assert_eq!(st.release(DevPtr(2), 2048), None, "parked, not freed");
        assert_eq!(st.live, 1024, "the caller gave it back");
        assert_eq!(st.deferred, 2048, "and the driver still holds it");

        assert_eq!(st.end(), vec![DevPtr(2)]);
    }

    #[test]
    fn outside_a_capture_a_free_happens_immediately() {
        let mut st = DeferState::default();
        assert_eq!(st.release(DevPtr(0x1000), 0), Some(DevPtr(0x1000)));
        assert!(st.pending.is_empty());
    }

    #[test]
    fn inside_a_capture_a_free_is_parked_not_performed() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        assert_eq!(
            st.release(DevPtr(0x1000), 0),
            None,
            "must not free during capture"
        );
        assert_eq!(st.pending, vec![DevPtr(0x1000)]);
    }

    #[test]
    fn closing_the_capture_hands_back_everything_parked_in_order() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.release(DevPtr(1), 0);
        let _ = st.release(DevPtr(2), 0);
        let _ = st.release(DevPtr(3), 0);
        assert_eq!(st.end(), vec![DevPtr(1), DevPtr(2), DevPtr(3)]);
        // Empty afterward: a second capture can't re-free what the first released.
        assert!(st.pending.is_empty());
        st.begin().unwrap();
        assert!(st.end().is_empty());
    }

    #[test]
    fn after_the_capture_closes_frees_go_straight_through_again() {
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.end();
        assert_eq!(st.release(DevPtr(0x2000), 0), Some(DevPtr(0x2000)));
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
        // The race this closes: a buffer dying on one thread while a capture
        // opens on another. `cudaFree` is replaced here by counting.
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
                    let decision = state.lock().unwrap().release(DevPtr(i + 1), 0);
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

        // Every pointer is accounted for exactly once, whatever the interleaving.
        let leftover = state.lock().unwrap().end().len();
        assert_eq!(
            freed_outside.load(Ordering::Relaxed) + parked_total + leftover,
            1000
        );
    }

    #[test]
    fn a_failed_capture_still_drains_what_was_parked() {
        // `finish` must drain even when `cudaStreamEndCapture` errors, or parked pointers strand.
        let mut st = DeferState::default();
        st.begin().unwrap();
        let _ = st.release(DevPtr(0xdead), 0);
        let freed = st.end();
        assert_eq!(freed, vec![DevPtr(0xdead)]);
        assert!(
            !st.capturing,
            "capture must be closed even on the error path"
        );
    }
}
