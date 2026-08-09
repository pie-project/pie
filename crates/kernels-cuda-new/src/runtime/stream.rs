//! The stream a launch is ordered on.
//!
//! # Why this crate has its own
//!
//! This crate is now the one that LAUNCHES, and a launch takes a stream. The
//! obvious move — take `driver_cuda::device::StreamRef`, which already exists
//! and is already borrowed — points the dependency backwards: the shell
//! depends on the kernels, the kernels do not depend on the shell, and a
//! `kernels-*` crate that needs a driver crate to spell its own launch
//! argument cannot be built, tested or read without one. `model-loader` calls
//! four rows directly and holds no `driver-cuda` at all.
//!
//! # Why it is not a raw pointer
//!
//! The other obvious move is to put a `CUstream` in the signature and let the
//! caller keep it alive. That is a `void*` with a comment: a launch that
//! outlives the stream it was queued on is undefined behaviour, and the
//! program that does it is one `cudaStreamDestroy` in a `Drop` away — a
//! lifetime is how that gets SAID, and it is the only form of it a compiler
//! reads. So the type is a borrow: [`Stream`] ties the handle to whatever
//! owns it, and every entry point in layer 3 takes one.
//!
//! # What it deliberately does not do
//!
//! No creation, no destruction, no synchronisation. Streams belong to the
//! shell, which creates them with priorities and flags this crate has no
//! opinion about — `cudaStreamNonBlocking`, a priority band per subsystem —
//! and destroys them on its own schedule. A kernel crate that could create
//! one would be a second place streams come from, and the first symptom of
//! two is a launch on a stream nothing else waits for. This crate only ever
//! ORDERS work on a stream it was handed.
//!
//! `driver-cuda`'s `StreamRef` is the same shape for a related reason and it
//! records it: ownership is a property of one end of a migration, not of the
//! stream.

use std::marker::PhantomData;

use cudarc::driver::sys::CUstream;

/// A CUDA stream this crate borrows for the length of a launch.
///
/// A handle and a lifetime, and nothing else. `Copy`, because a stream is an
/// identifier and passing one to a fire should read like passing an integer,
/// which is what it is.
///
/// # Threads
///
/// There is no `unsafe impl Send`/`Sync` here, so the raw pointer inside
/// makes this neither, and that is the intended answer rather than an
/// oversight. CUDA does guarantee that stream submission is thread-safe, but
/// the guarantee is about the STREAM and the assertion would be about this
/// crate's borrow of someone else's — which the owner is in a position to
/// make and a kernel table is not. `driver-cuda` asserts it on `StreamRef`
/// because the shell holds streams in structures that cross threads and owns
/// what that means. A caller here that must move one has [`Stream::as_raw`]
/// and [`Stream::from_raw`], which is the same assertion made where the
/// lifetime is actually known.
#[derive(Clone, Copy, Debug)]
pub struct Stream<'a> {
    raw: CUstream,
    life: PhantomData<&'a ()>,
}

impl Stream<'static> {
    /// The default stream. `cuLaunchKernel` takes a null `CUstream` to mean
    /// it.
    ///
    /// `'static` because the default stream is never created and never
    /// destroyed, so there is nothing for a borrow to outlive. This is what a
    /// test fires on and what a caller with no stream of its own should pass
    /// — not because it is fast, but because the alternative is inventing a
    /// stream in the crate that must not create one.
    pub const NULL: Self = Self { raw: std::ptr::null_mut(), life: PhantomData };
}

impl<'a> Stream<'a> {
    /// Borrow a stream the driver API created.
    ///
    /// # Safety
    ///
    /// `raw` must be a live `CUstream` for `'a`. The caller is ASSERTING
    /// that lifetime, not proving it; this is the one unchecked step, and
    /// every safe thing downstream is safe because this call was made
    /// correctly once. Getting it wrong is a launch queued on a destroyed
    /// stream, which CUDA reports as whatever the freed handle now points at.
    pub const unsafe fn from_raw(raw: CUstream) -> Self {
        Self { raw, life: PhantomData }
    }

    /// Borrow a stream the RUNTIME API created — a `cudaStream_t`.
    ///
    /// The two APIs share stream objects: a `cudaStream_t` and a `CUstream`
    /// are the same pointer to the same object under two typedefs, one per
    /// header, which is exactly what lets a process that creates streams with
    /// `cudaStreamCreateWithPriority` order work on them with
    /// `cuLaunchKernel`. The cast is a change of spelling and nothing more.
    ///
    /// It is a named constructor rather than a cast at the call site because
    /// the interoperability rule is the thing worth writing down once. Every
    /// runtime-API caller — `driver-cuda`'s shell, which creates every stream
    /// it owns that way, and the C++ that still hands streams across — would
    /// otherwise carry an unexplained `as` at each seam, and an unexplained
    /// pointer cast is indistinguishable from a wrong one.
    ///
    /// The parameter is `*mut c_void` rather than `cudaStream_t` so that this
    /// signature costs no `cudarc::runtime` in the caller's dependency graph:
    /// a crate holding a runtime handle can pass it without taking on the
    /// runtime bindings to name its type.
    ///
    /// # Safety
    ///
    /// `raw` must be a live `cudaStream_t` for `'a` — [`Stream::from_raw`]'s
    /// obligation, in the other API's words.
    pub const unsafe fn from_runtime(raw: *mut std::ffi::c_void) -> Self {
        Self { raw: raw.cast(), life: PhantomData }
    }

    /// The handle, for `cuLaunchKernel`'s `hStream` argument.
    pub const fn as_raw(self) -> CUstream {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::Stream;

    /// The default stream is the null handle, which is CUDA's spelling of it
    /// rather than a sentinel this crate invented.
    #[test]
    fn the_null_stream_is_null() {
        assert!(Stream::NULL.as_raw().is_null());
    }

    /// A borrow hands back exactly what it was given.
    ///
    /// The whole content of the type: a launch's `hStream` is this value and
    /// nothing else, so a constructor that transformed it would queue work on
    /// a stream nobody is waiting on. Exercised with a fake handle, because
    /// the claim is about the plumbing and holds on a machine with no CUDA.
    #[test]
    fn borrowing_a_raw_handle_round_trips() {
        let fake = std::ptr::without_provenance_mut(0xdead_beef);
        // SAFETY: never launched on — `from_raw`'s obligation is on the
        // caller that fires, and this one only reads the handle back.
        let borrowed = unsafe { Stream::from_raw(fake) };
        assert_eq!(borrowed.as_raw(), fake);
    }

    /// And so does the runtime API's spelling of the same pointer.
    #[test]
    fn a_runtime_handle_is_the_same_pointer() {
        let fake: *mut std::ffi::c_void = std::ptr::without_provenance_mut(0xfeed_face);
        // SAFETY: as above — the handle is never submitted to.
        let borrowed = unsafe { Stream::from_runtime(fake) };
        assert_eq!(borrowed.as_raw().cast::<std::ffi::c_void>(), fake);
    }

    /// A stream costs a word and drops nothing.
    ///
    /// Both halves are load-bearing. Every fire in the crate takes one by
    /// value, so a type that grew would be paid for on the launch path; and
    /// drop glue on a BORROWED handle would mean this crate destroys a stream
    /// the shell created, which is the one thing the module says it does not
    /// do.
    #[test]
    fn a_stream_is_a_word_with_no_destructor() {
        assert_eq!(
            std::mem::size_of::<Stream<'_>>(),
            std::mem::size_of::<cudarc::driver::sys::CUstream>()
        );
        assert!(!std::mem::needs_drop::<Stream<'_>>());
    }
}
