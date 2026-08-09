//! Zeroing and copying spans of a device buffer, from the host.
//!
//! The C++ names these `zero_buffer_range` and `copy_buffer_range` and files
//! them among the command-encoding methods, which is misleading: neither one
//! encodes anything. On unified memory a device buffer in shared storage is
//! already host-addressable, so both are a `memset` and a `memmove` on a
//! pointer Metal handed over at allocation time. No blit encoder, no command
//! buffer, no fence -- and no synchronisation either, which is the part worth
//! being explicit about. See [`Region::zero`] for what that costs.
//!
//! # Why this is in the portable half
//!
//! Nothing below names a Metal type. The inputs are a pointer, a length and
//! two offsets, and the only thing that can go wrong is the arithmetic. That
//! arithmetic is also the one part with a history of being wrong, so it is
//! here where a Linux `cargo test` reaches it, with a `Vec`-backed [`Region`]
//! standing in for a buffer. `crate::gpu` adds the two `impl`s and no
//! logic.
//!
//! # The bound, and why it is written the way it is
//!
//! `offset + bytes > len` is the obvious check and it is wrong: the sum can
//! pass the range and wrap, and a wrapped sum compares small. Every bound
//! here is written as `offset > len || bytes > len - offset`, where the
//! subtraction cannot go negative because the first test already refused
//! that. The C++ does this correctly; it is preserved rather than
//! rediscovered.
//!
//! # What a length means
//!
//! The length a [`Region`] reports is what the caller asked for, not what the
//! device allocated. Both a heap slot and a pooled buffer round up, and both
//! keep the padding out of their `len`. Writing into that padding would not
//! fault -- the bytes exist and are mapped -- so the bound here is the only
//! thing that separates a slot from its neighbour.

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::{Error, Result};

/// A host-addressable span of device memory.
///
/// Implemented by `Slot` and `Transient` (both under the `metal-4` gate),
/// which differ in who owns the buffer and not at all in what a byte range
/// within one means.
///
/// # Safety
///
/// An implementor promises that [`contents`](Region::contents) is valid for
/// reads and writes of [`len`](Region::len) bytes for as long as `&self` is
/// held, and that no other `Region` reachable from safe code overlaps it
/// except through an explicit alias. The trait's provided methods dereference
/// that pointer on the strength of this.
pub unsafe trait Region {
    /// The first host-visible byte.
    fn contents(&self) -> NonNull<c_void>;

    /// The number of bytes the caller may touch, starting at `contents`.
    fn len(&self) -> u64;

    /// Whether the span is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check that `bytes` starting at `offset` lies within the region.
    ///
    /// `what` names the side for the message, which is the whole reason this
    /// is separate: a failed [`copy`](Region::copy) has two candidate ranges
    /// and saying which one was short is the difference between a fixable
    /// report and "invalid range".
    fn check(&self, what: &'static str, offset: u64, bytes: u64) -> Result<()> {
        let len = self.len();
        if offset > len || bytes > len - offset {
            return Err(Error::OutOfRange {
                what,
                offset,
                bytes,
                len,
            });
        }
        Ok(())
    }

    /// Write `bytes` zero bytes at `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if the span leaves the region. Nothing is written
    /// in that case -- the bound is checked before the first byte, not walked
    /// into, which matters because a partial zero is a buffer in a state no
    /// caller asked for and no caller can detect.
    ///
    /// # Safety
    ///
    /// The GPU must not be reading or writing these bytes. Shared storage
    /// means concurrent access is a data race that neither Metal nor Rust
    /// will diagnose: there is no barrier here and no `didModifyRange:` to
    /// forget, so the wrong call site produces a wrong number rather than a
    /// fault. The only thing that establishes it is a step boundary -- the
    /// step that last read the buffer has signalled its event, and the step
    /// that next reads it has not been committed.
    unsafe fn zero(&self, offset: u64, bytes: u64) -> Result<()> {
        self.check("region", offset, bytes)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `check` established that `offset .. offset + bytes` is
        // within the `len` bytes the implementor promises are valid, so the
        // offset does not leave the allocation. The caller's obligation
        // covers the GPU.
        unsafe {
            self.contents()
                .cast::<u8>()
                .add(usize_of(offset))
                .write_bytes(0, usize_of(bytes));
        }
        Ok(())
    }

    /// Copy `bytes` from `src` at `src_offset` into `self` at `dst_offset`.
    ///
    /// Overlapping ranges are defined, as `memmove` is and `memcpy` is not.
    /// The C++ takes the same care, and it is not hypothetical: a KV cache
    /// compaction slides a region over itself.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] naming `"destination"` or `"source"`, whichever
    /// range was short. Both are checked before anything is written.
    ///
    /// # Safety
    ///
    /// As [`zero`](Region::zero), for both regions.
    unsafe fn copy<S: Region + ?Sized>(
        &self,
        dst_offset: u64,
        src: &S,
        src_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        self.check("destination", dst_offset, bytes)?;
        src.check("source", src_offset, bytes)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: both ranges were just checked against their own region's
        // length. `copy` is `memmove`, so the two being the same allocation
        // is permitted.
        unsafe {
            let from = src.contents().cast::<u8>().add(usize_of(src_offset));
            let to = self.contents().cast::<u8>().add(usize_of(dst_offset));
            core::ptr::copy(from.as_ptr(), to.as_ptr(), usize_of(bytes));
        }
        Ok(())
    }

    /// Copy `src` into the region at `offset`.
    ///
    /// The length comes from the slice, so the common upload has no separate
    /// count to get wrong.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if the slice does not fit at `offset`.
    ///
    /// # Safety
    ///
    /// As [`zero`](Region::zero).
    unsafe fn write(&self, offset: u64, src: &[u8]) -> Result<()> {
        let bytes = src.len() as u64;
        self.check("region", offset, bytes)?;
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: the range was just checked, and `src` is a live slice that
        // cannot overlap device memory the implementor owns.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.contents().cast::<u8>().add(usize_of(offset)).as_ptr(),
                src.len(),
            );
        }
        Ok(())
    }
}

/// Narrow a checked offset or length to a host index.
///
/// Every caller has already compared the value against a `len` that came from
/// an allocation this process made, so it fits. The cast is written here once
/// rather than at four call sites so that the reason is written once too.
#[allow(clippy::cast_possible_truncation)]
const fn usize_of(v: u64) -> usize {
    v as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region that is a heap allocation, so the arithmetic can be tested
    /// off-device.
    ///
    /// The backing store is deliberately longer than the region claims. An
    /// off-by-one then writes a byte the test can look at, rather than
    /// running off the end of the allocation where the failure would be a
    /// crash instead of an assertion -- which is the same reason a real slot
    /// keeps its rounding-up out of its `len`.
    struct Fake {
        ptr: NonNull<u8>,
        total: usize,
        claimed: u64,
    }

    impl Fake {
        fn new(len: usize) -> Self {
            let total = len + 16;
            let owned = vec![0xAA_u8; total].into_boxed_slice();
            let ptr = NonNull::new(Box::into_raw(owned).cast::<u8>()).expect("a box is never null");
            Self {
                ptr,
                total,
                claimed: len as u64,
            }
        }

        /// Everything, including the slack past `claimed`.
        fn all(&self) -> &[u8] {
            // SAFETY: `ptr` owns `total` initialised bytes until `drop`.
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.total) }
        }
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let slice = core::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.total);
            // SAFETY: exactly what `Box::into_raw` handed over in `new`.
            drop(unsafe { Box::from_raw(slice) });
        }
    }

    // SAFETY: `ptr` is valid for reads and writes of `total` bytes, and
    // `claimed` is smaller than `total`. The pointer comes from `Box::into_raw`
    // rather than from a `&self` borrow, so writing through it is permitted.
    unsafe impl Region for Fake {
        fn contents(&self) -> NonNull<c_void> {
            self.ptr.cast()
        }
        fn len(&self) -> u64 {
            self.claimed
        }
    }

    #[test]
    fn a_zero_clears_its_span_and_nothing_beside_it() {
        let r = Fake::new(64);
        unsafe { r.zero(8, 16) }.unwrap();
        assert_eq!(&r.all()[..8], &[0xAA; 8]);
        assert_eq!(&r.all()[8..24], &[0; 16]);
        assert_eq!(&r.all()[24..32], &[0xAA; 8]);
    }

    #[test]
    fn a_span_that_ends_exactly_at_the_end_is_in_bounds() {
        let r = Fake::new(64);
        unsafe { r.zero(48, 16) }.unwrap();
        assert_eq!(&r.all()[64..], &[0xAA; 16], "the slack is not ours");

        assert!(unsafe { r.zero(48, 17) }.is_err());
        assert!(unsafe { r.zero(64, 0) }.is_ok(), "empty at the end");
        assert!(unsafe { r.zero(65, 0) }.is_err(), "empty past the end");
    }

    /// The check the obvious spelling gets wrong.
    #[test]
    fn an_offset_and_a_length_that_wrap_are_still_refused() {
        let r = Fake::new(64);
        // `offset + bytes` is 8, which is comfortably inside 64.
        let err = unsafe { r.zero(u64::MAX, 9) }.unwrap_err();
        assert!(matches!(err, Error::OutOfRange { .. }));
        assert_eq!(r.all(), &[0xAA; 80], "nothing was written");
    }

    #[test]
    fn a_refused_copy_says_which_side_was_short() {
        let dst = Fake::new(16);
        let src = Fake::new(64);

        let err = unsafe { dst.copy(0, &src, 0, 32) }.unwrap_err();
        let Error::OutOfRange { what, len, .. } = err else {
            panic!("wrong variant")
        };
        assert_eq!((what, len), ("destination", 16));

        let err = unsafe { src.copy(0, &dst, 0, 32) }.unwrap_err();
        let Error::OutOfRange { what, len, .. } = err else {
            panic!("wrong variant")
        };
        assert_eq!((what, len), ("source", 16));
    }

    #[test]
    fn a_copy_moves_the_bytes_between_regions() {
        let dst = Fake::new(32);
        let src = Fake::new(32);
        unsafe { src.write(4, b"pie!") }.unwrap();

        unsafe { dst.copy(20, &src, 4, 4) }.unwrap();
        assert_eq!(&dst.all()[20..24], b"pie!");
        assert_eq!(dst.all()[19], 0xAA);
    }

    /// `memmove`, not `memcpy`: a compaction slides a region over itself, and
    /// which direction it slides decides whether `memcpy` would smear.
    #[test]
    fn an_overlapping_copy_slides_rather_than_smears() {
        let down = Fake::new(16);
        unsafe { down.write(0, b"0123456789abcdef") }.unwrap();
        unsafe { down.copy(0, &down, 4, 12) }.unwrap();
        assert_eq!(&down.all()[..12], b"456789abcdef");

        let up = Fake::new(16);
        unsafe { up.write(0, b"0123456789abcdef") }.unwrap();
        unsafe { up.copy(4, &up, 0, 12) }.unwrap();
        assert_eq!(&up.all()[..16], b"01230123456789ab");
    }

    #[test]
    fn a_write_takes_its_length_from_the_slice() {
        let r = Fake::new(8);
        unsafe { r.write(4, b"abcd") }.unwrap();
        assert_eq!(&r.all()[..8], b"\xAA\xAA\xAA\xAAabcd");

        assert!(unsafe { r.write(5, b"abcd") }.is_err());
        assert!(unsafe { r.write(8, b"") }.is_ok());
    }
}
