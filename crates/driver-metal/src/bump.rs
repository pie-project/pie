//! Where the next slot goes: the placement heap's arithmetic, without the heap.
//!
//! A placement heap does not allocate. It is a range of device address space
//! and a promise that a buffer created at an offset inside it is backed; every
//! decision about WHICH offset is the caller's. In the C++ shell that decision
//! is three lines in the middle of a method that also creates the buffer,
//! registers it for residency and fills in a handle -- so the arithmetic, which
//! is the only part that can be wrong in a way a test could catch, is reachable
//! only from a machine with a GPU.
//!
//! It is here instead, where it is nine lines of integers.
//!
//! # Bump, not free-list
//!
//! Nothing this allocator hands out is ever returned. The driver's heap holds
//! weights, KV pages and scratch for the lifetime of a model, and the one
//! lifetime that is shorter than that -- transient per-step buffers -- is
//! served by a separate pool. So `free` would be an unreachable branch, and a
//! bump pointer is the honest shape rather than a simplification.

/// The next free offset in a range of device address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bump {
    capacity: u64,
    next: u64,
}

/// A granted range: where it starts and how much of the heap it consumed.
///
/// `size` is not the requested length. Metal answers
/// `heapBufferSizeAndAlignWithLength:` with a size of its own, which is the
/// requested length rounded up to whatever the device needs -- and it is that
/// number, not the caller's, that the next allocation has to start after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// Byte offset into the heap.
    pub offset: u64,
    /// Bytes consumed from the heap, as the device sized them.
    pub size: u64,
}

impl Bump {
    /// An empty allocator over `capacity` bytes.
    #[must_use]
    pub const fn new(capacity: u64) -> Self {
        Self { capacity, next: 0 }
    }

    /// Total bytes.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes handed out, including alignment padding.
    #[must_use]
    pub const fn used(&self) -> u64 {
        self.next
    }

    /// Bytes left, which is what a request larger than this will be refused by.
    #[must_use]
    pub const fn available(&self) -> u64 {
        self.capacity - self.next
    }

    /// Place `size` bytes at `align`, or say why not.
    ///
    /// `align` must be a power of two, which every alignment Metal reports is;
    /// a caller-supplied one that is not is treated as a refusal rather than
    /// rounded, because rounding it would place the buffer somewhere the
    /// device did not agree to.
    ///
    /// The error carries the numbers rather than being a bare `None`: a heap
    /// that is too small and an allocator that is leaking produce the same
    /// failure at the same call site, and only the numbers separate them.
    pub fn alloc(&mut self, size: u64, align: u64) -> Result<Placement, Exhausted> {
        let refuse = |requested| {
            Err(Exhausted {
                requested,
                available: self.available(),
                capacity: self.capacity,
            })
        };

        if align == 0 || !align.is_power_of_two() {
            return refuse(size);
        }

        // Aligning can carry past u64. Checked rather than wrapping: a wrapped
        // offset is SMALLER than `next`, so it would pass the capacity test
        // below and hand out a range overlapping one already given away.
        let Some(offset) = align_up(self.next, align) else {
            return refuse(size);
        };
        let Some(end) = offset.checked_add(size) else {
            return refuse(size);
        };
        if end > self.capacity {
            // Reported as the padded request, because that is what did not
            // fit: a caller told "16 bytes did not fit in 1 MB free" cannot
            // tell that the 1 MB was unusable padding.
            return refuse(end - self.next);
        }

        self.next = end;
        Ok(Placement { offset, size })
    }
}

/// A request the heap had no room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exhausted {
    /// Bytes needed, including the padding alignment forced.
    pub requested: u64,
    /// Bytes that were free.
    pub available: u64,
    /// Total heap size.
    pub capacity: u64,
}

/// `value` rounded up to a multiple of `align`, or `None` on overflow.
///
/// `align` must be a power of two; the caller checks.
const fn align_up(value: u64, align: u64) -> Option<u64> {
    let Some(sum) = value.checked_add(align - 1) else {
        return None;
    };
    Some(sum & !(align - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_placed_end_to_end() {
        let mut bump = Bump::new(1024);
        let a = bump.alloc(100, 1).expect("fits");
        let b = bump.alloc(100, 1).expect("fits");
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 100);
        assert_eq!(bump.used(), 200);
        assert_eq!(bump.available(), 824);
    }

    #[test]
    fn alignment_pads_the_offset() {
        let mut bump = Bump::new(1024);
        bump.alloc(1, 1).expect("fits");
        let b = bump.alloc(16, 256).expect("fits");
        assert_eq!(
            b.offset, 256,
            "the second allocation is aligned, not packed"
        );
        assert_eq!(bump.used(), 272, "padding is consumed, not free");
    }

    #[test]
    fn an_exact_fit_is_a_fit() {
        let mut bump = Bump::new(256);
        assert_eq!(bump.alloc(256, 1).expect("exact").offset, 0);
        assert_eq!(bump.available(), 0);
    }

    #[test]
    fn one_byte_past_the_capacity_is_refused() {
        let mut bump = Bump::new(256);
        let err = bump.alloc(257, 1).expect_err("does not fit");
        assert_eq!(err.available, 256);
        assert_eq!(err.capacity, 256);
    }

    /// A refusal must not consume anything. The C++ shell returns an invalid
    /// handle on OOM and leaves its bump alone; a version that advanced it
    /// would turn one refused allocation into a heap that stays broken.
    #[test]
    fn a_refusal_leaves_the_allocator_where_it_was() {
        let mut bump = Bump::new(256);
        bump.alloc(128, 1).expect("fits");
        let before = bump;
        bump.alloc(1024, 1).expect_err("does not fit");
        assert_eq!(bump, before);
        bump.alloc(128, 1).expect("the rest is still available");
    }

    /// The padding is what did not fit, so the padding is what is reported.
    #[test]
    fn the_refusal_counts_the_padding_it_needed() {
        let mut bump = Bump::new(260);
        bump.alloc(1, 1).expect("fits");
        // 16 bytes at 256 alignment starts at 256 and ends at 272, so it needs
        // 271 more bytes -- 255 of which are padding -- and only 259 are free.
        let err = bump.alloc(16, 256).expect_err("padding overruns");
        assert_eq!(err.requested, 271, "reported as the padded cost");
        assert_eq!(err.available, 259);
        assert!(
            err.requested > err.available,
            "the 16-byte request did not fit in 259 free bytes only because of padding"
        );
    }

    #[test]
    fn a_non_power_of_two_alignment_is_refused() {
        let mut bump = Bump::new(1024);
        bump.alloc(16, 3).expect_err("3 is not an alignment");
        bump.alloc(16, 0).expect_err("0 is not an alignment");
        assert_eq!(bump.used(), 0, "a refusal consumed nothing");
    }

    /// Aligning near the top of the address space carries past u64. It cannot
    /// be allowed to wrap: a wrapped offset is BELOW `next`, so the capacity
    /// test would pass and the allocator would hand out a range it had
    /// already given away.
    #[test]
    fn an_alignment_that_would_overflow_is_refused() {
        let mut bump = Bump::new(u64::MAX);
        bump.alloc(u64::MAX - 8, 1).expect("fits");
        bump.alloc(16, 1 << 40).expect_err("aligning overflows");
        // And the size itself, at the very top.
        let mut bump = Bump::new(u64::MAX);
        bump.alloc(u64::MAX - 1, 1).expect("fits");
        bump.alloc(u64::MAX, 1)
            .expect_err("offset + size overflows");
    }

    #[test]
    fn a_zero_capacity_allocator_refuses_everything_but_nothing() {
        let mut bump = Bump::new(0);
        bump.alloc(0, 1).expect("zero bytes always fit");
        bump.alloc(1, 1).expect_err("and one does not");
    }
}
