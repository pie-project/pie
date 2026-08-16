use super::Error;

/// A bump allocator over an arena of known size, handing out offsets.
#[derive(Clone, Copy, Debug)]
pub struct AlignedAllocator {
    allocated: usize,
    remaining: usize,
}

impl AlignedAllocator {
    /// An allocator over an arena of `space` bytes.
    #[must_use]
    pub const fn new(space: usize) -> Self {
        Self { allocated: 0, remaining: space }
    }

    /// The unbounded allocator, which is upstream's default-constructed one.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Carve `size` bytes at `alignment`, and answer where they start.
    pub fn alloc(
        &mut self,
        size: usize,
        alignment: usize,
        what: &'static str,
    ) -> Result<usize, Error> {
        let padding =
            if alignment > 1 { (alignment - (self.allocated % alignment)) % alignment } else { 0 };
        if padding > self.remaining || size > self.remaining - padding {
            return Err(Error::WorkspaceOverflow {
                what,
                size,
                alignment,
                remaining: self.remaining,
            });
        }
        let result = self.allocated + padding;
        self.allocated = result + size;
        self.remaining -= padding + size;
        Ok(result)
    }

    /// `num_allocated_bytes()`: the end of the last allocation, and the length
    #[must_use]
    pub const fn used(&self) -> usize {
        self.allocated
    }
}

/// The page-locked staging buffer, as bytes plus the writes a planner makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staging {
    bytes: Vec<u8>,
}

impl Staging {
    /// A zeroed staging buffer of `len` bytes.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { bytes: vec![0u8; len] }
    }

    /// The staging buffer of a sizing pass: empty, and never written.
    #[must_use]
    pub const fn sizing() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Whether this buffer records writes.
    #[must_use]
    pub const fn materialises(&self) -> bool {
        !self.bytes.is_empty()
    }

    /// Write `values` as `int32_t` starting at `offset`.
    pub fn put_i32s(
        &mut self,
        offset: usize,
        values: &[i32],
        what: &'static str,
    ) -> Result<(), Error> {
        let end = offset + values.len() * 4;
        self.check(end, values.len() * 4, what)?;
        for (i, v) in values.iter().enumerate() {
            self.bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    /// Write one `int32_t` at `offset`.
    pub fn put_i32(&mut self, offset: usize, value: i32, what: &'static str) -> Result<(), Error> {
        self.put_i32s(offset, &[value], what)
    }

    /// Write one `uint32_t` at `offset`.
    pub fn put_u32(&mut self, offset: usize, value: u32, what: &'static str) -> Result<(), Error> {
        self.check(offset + 4, 4, what)?;
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Write `values` as C++ `bool`s — one byte each, `0` or `1`.
    pub fn put_bools(
        &mut self,
        offset: usize,
        values: impl Iterator<Item = bool>,
        what: &'static str,
    ) -> Result<(), Error> {
        for (i, v) in values.enumerate() {
            self.check(offset + i + 1, 1, what)?;
            self.bytes[offset + i] = u8::from(v);
        }
        Ok(())
    }

    /// The first `len` bytes: exactly what upstream copies H2D.
    #[must_use]
    pub fn into_upload(mut self, len: usize) -> Vec<u8> {
        self.bytes.truncate(len);
        self.bytes
    }

    fn check(&self, end: usize, size: usize, what: &'static str) -> Result<(), Error> {
        if end > self.bytes.len() {
            return Err(Error::WorkspaceOverflow {
                what,
                size,
                alignment: 1,
                remaining: self.bytes.len().saturating_sub(end - size),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 4-byte, alignment-1 allocation in the middle of both FA2 planners
    #[test]
    fn a_misaligned_carve_pads_the_next_one() {
        let mut a = AlignedAllocator::new(1024);
        assert_eq!(a.alloc(64, 16, "first").unwrap(), 0);
        assert_eq!(a.alloc(4, 1, "kv_chunk_size_ptr").unwrap(), 64);
        assert_eq!(a.alloc(16, 16, "next").unwrap(), 80);
        assert_eq!(a.used(), 96);
    }

    /// A refusal names the allocation, because the answer is a bigger grant.
    #[test]
    fn an_overflow_names_what_did_not_fit() {
        let mut a = AlignedAllocator::new(32);
        assert!(a.alloc(16, 16, "fits").is_ok());
        let err = a.alloc(64, 16, "batch_prefill_merge_indptr").unwrap_err();
        assert!(matches!(
            err,
            Error::WorkspaceOverflow { what: "batch_prefill_merge_indptr", size: 64, .. }
        ));
    }

    /// The unbounded allocator is why a sizing pass never refuses.
    #[test]
    fn the_unbounded_allocator_never_refuses() {
        let mut a = AlignedAllocator::unbounded();
        assert_eq!(a.alloc(1 << 40, 16, "huge").unwrap(), 0);
        assert_eq!(a.used(), 1 << 40);
    }
}
