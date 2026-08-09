use std::marker::PhantomData;

use cudarc::driver::sys::CUstream;

/// A CUDA stream this crate borrows for the length of a launch.
#[derive(Clone, Copy, Debug)]
pub struct Stream<'a> {
    raw: CUstream,
    life: PhantomData<&'a ()>,
}

impl Stream<'static> {
    /// The default stream. `cuLaunchKernel` takes a null `CUstream` to mean
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
    #[test]
    fn the_null_stream_is_null() {
        assert!(Stream::NULL.as_raw().is_null());
    }

    /// A borrow hands back exactly what it was given.
    #[test]
    fn borrowing_a_raw_handle_round_trips() {
        let fake = std::ptr::without_provenance_mut(0xdead_beef);
        // SAFETY: never launched on — `from_raw`'s obligation is on the
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
    #[test]
    fn a_stream_is_a_word_with_no_destructor() {
        assert_eq!(
            std::mem::size_of::<Stream<'_>>(),
            std::mem::size_of::<cudarc::driver::sys::CUstream>()
        );
        assert!(!std::mem::needs_drop::<Stream<'_>>());
    }
}
