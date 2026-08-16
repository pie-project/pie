//! The one host allocation this crate makes, and the reason it is not a `Vec`.
//!
//! A captured CUDA graph bakes a memcpy node's SOURCE ADDRESS and performs no
//! copy at capture time, so the host buffer an H2D reads from is part of the
//! graph's ABI. `jit::device::upload`'s contract is written against this
//! type; `attn::fa2::plan`'s two caches hold one each.
//!
//! Not inside `device`, which is `_cuda`-gated: the FA2 planners are pure
//! host code and their tests run with no CUDA runtime at all, so this type
//! has to exist in both builds. Without the feature it is an ordinary heap
//! allocation with the same fixed-capacity behaviour and no pin — which is
//! exactly right for a build that cannot capture a graph either.

#[cfg(feature = "_cuda")]
use core::ffi::c_void;

#[cfg(feature = "_cuda")]
use cudarc::runtime::sys as rt;
use kernels::routine::Refusal;

/// A fixed-capacity, page-locked host buffer: the only legal source for an
/// H2D copy that a graph will capture.
///
/// # Why this exists and a `Vec<u8>` will not do
///
/// A captured graph bakes the memcpy node's SOURCE ADDRESS, not its bytes
/// (`driver-cuda/src/fire/recordings.rs`). A `Vec` that is cleared and
/// re-`resize`d each fire keeps its address only while the new length fits
/// the old capacity; the first fire that needs one byte more reallocates, and
/// every replay after it copies from freed memory. Nothing observes that: the
/// replay does not run the arm that would have grown the buffer, so the graph
/// goes on copying from the address the capturing fire happened to have.
///
/// So the capacity is taken from a bound that does not move — for the FA2
/// decode plan, the padded batch size, which under `enable_cuda_graph` is
/// `max_grid_size / gdy`, head geometry alone — and every later fill reuses
/// it at the same address. A fill that fits is guaranteed not to move.
///
/// # Growth, and why it is not a hole
///
/// A fill that does NOT fit reallocates, and the address moves. That is legal
/// here for one reason: this buffer's length is the plan allocator's cursor,
/// so a length that changed means an OFFSET changed, and the fire's recording
/// digest hashes those offsets. A moved buffer therefore cannot be replayed
/// against the graph that baked the old address — the digest already
/// mismatched and the fire recaptured. Under `enable_cuda_graph` the decode
/// plan's cursor is constant after the first plan, so this branch is taken
/// once and never again; prefill, whose padding is still a function of the
/// fire's rows, keeps the eager behaviour it had.
///
/// Page-locked rather than merely stable because the copy is then genuinely
/// asynchronous: `cudaMemcpyAsync` from pageable memory stages through a
/// driver buffer before returning, which is the property the old `Vec` was
/// leaning on, and which is exactly what a captured node does not re-do.
pub struct PinnedBytes {
    /// `cudaHostAlloc`'s pointer. Null only for a zero-capacity buffer.
    ptr: *mut u8,
    /// Bytes actually written by the last [`PinnedBytes::fill`].
    len: usize,
    /// Bytes allocated. Never changes after construction.
    cap: usize,
}

// SAFETY: the buffer owns its allocation exclusively; the pointer is not
// shared and nothing in it is thread-affine. Page-locked memory is process
// memory like any other -- the pin is a property of the mapping, not of the
// allocating thread.
unsafe impl Send for PinnedBytes {}
// SAFETY: as above. `&PinnedBytes` hands out only `&[u8]`.
unsafe impl Sync for PinnedBytes {}

impl PinnedBytes {
    /// An empty buffer that owns nothing. Every write to it refuses.
    #[must_use]
    pub const fn empty() -> Self {
        Self { ptr: core::ptr::null_mut(), len: 0, cap: 0 }
    }

    /// Take `cap` page-locked bytes, once.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if the pin fails, which on a busy host it can.
    pub fn with_capacity(cap: usize) -> Result<Self, Refusal> {
        if cap == 0 {
            return Ok(Self::empty());
        }
        #[cfg(feature = "_cuda")]
        {
            let mut p: *mut c_void = core::ptr::null_mut();
            // SAFETY: `p` is a live, writable out-parameter and `cap` is
            // non-zero.
            let code = unsafe { rt::cudaMallocHost(&raw mut p, cap) };
            if code != rt::cudaError::cudaSuccess || p.is_null() {
                return Err(Refusal::Device { why: "the pinned plan buffer could not be taken" });
            }
            Ok(Self { ptr: p.cast::<u8>(), len: 0, cap })
        }
        // No runtime to pin with, and none to capture a graph either. A
        // leaked fixed-capacity heap block has every property the rest of
        // this type promises except the pin, which is what the host-side
        // planner tests need and all they need.
        #[cfg(not(feature = "_cuda"))]
        {
            let mut v = vec![0u8; cap];
            let p = v.as_mut_ptr();
            core::mem::forget(v);
            Ok(Self { ptr: p, len: 0, cap })
        }
    }

    /// How many bytes this buffer can ever hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// The bytes the last fill wrote. **The address is the same every time.**
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `ptr` addresses `cap >= len` bytes this buffer owns, and
        // every one below `len` was written by `fill`.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Where the bytes are. Hashing this is how a caller notices a move.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.cast_const()
    }

    /// Overwrite the contents, in place when they fit.
    ///
    /// Returns `true` when the buffer moved — which only a `src` longer than
    /// the current capacity can cause, and which the caller must treat as
    /// invalidating any graph that captured a copy from here.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if a growth's pin fails.
    pub fn fill(&mut self, src: &[u8]) -> Result<bool, Refusal> {
        let mut moved = false;
        if src.len() > self.cap {
            *self = Self::with_capacity(src.len())?;
            moved = true;
        }
        if !src.is_empty() {
            // SAFETY: `ptr` owns `cap >= src.len()` bytes and the two regions
            // are distinct -- `src` is a caller's slice, never this buffer.
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len()) };
        }
        self.len = src.len();
        Ok(moved)
    }
}

impl Default for PinnedBytes {
    fn default() -> Self {
        Self::empty()
    }
}

impl core::fmt::Debug for PinnedBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedBytes")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("cap", &self.cap)
            .finish()
    }
}

impl Drop for PinnedBytes {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        #[cfg(feature = "_cuda")]
        // SAFETY: `ptr` came from `cudaMallocHost` in this type's only
        // constructor and is freed exactly once.
        unsafe {
            let _ = rt::cudaFreeHost(self.ptr.cast::<c_void>());
        }
        #[cfg(not(feature = "_cuda"))]
        // SAFETY: the fallback constructor `forget`s a `Vec<u8>` of exactly
        // `cap` bytes, and this rebuilds the same one to drop it once.
        unsafe {
            drop(Vec::from_raw_parts(self.ptr, self.len, self.cap));
        }
    }
}
