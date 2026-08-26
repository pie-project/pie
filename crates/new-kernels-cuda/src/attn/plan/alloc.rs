//! The plan builders' two working surfaces: an aligned bump allocator that
//! carves offsets out of a granted workspace, and the staging buffer whose
//! bytes become the plan's `int_upload`. Both carry the plan op's name so an
//! overflow refuses with attribution instead of a bare capacity number.

use new_kernels::KernelError;

use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct AlignedAllocator {
    op: &'static str,
    allocated: usize,
    remaining: usize,
}

impl AlignedAllocator {
    #[must_use]
    pub const fn new(op: &'static str, space: usize) -> Self {
        Self {
            op,
            allocated: 0,
            remaining: space,
        }
    }

    /// The sizing pass's allocator: never refuses, and `used()` afterwards
    /// is the workspace this plan would need.
    #[must_use]
    pub const fn unbounded(op: &'static str) -> Self {
        Self::new(op, usize::MAX)
    }

    pub fn alloc(
        &mut self,
        size: usize,
        alignment: usize,
        what: &'static str,
    ) -> Result<usize, KernelError> {
        let padding = if alignment > 1 {
            (alignment - (self.allocated % alignment)) % alignment
        } else {
            0
        };
        if padding > self.remaining || size > self.remaining - padding {
            return Err(refuse(
                self.op,
                format!(
                    "`{what}` does not fit the granted workspace: {size} bytes asked, \
                     {} left",
                    self.remaining
                ),
            ));
        }
        let result = self.allocated + padding;
        self.allocated = result + size;
        self.remaining -= padding + size;
        Ok(result)
    }

    #[must_use]
    pub const fn used(&self) -> usize {
        self.allocated
    }
}

/// The host image of the int workspace. A sizing pass carries an empty one
/// (`materialises()` is false) so the same builder body computes sizes
/// without writing a byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staging {
    op: &'static str,
    bytes: Vec<u8>,
}

impl Staging {
    #[must_use]
    pub fn new(op: &'static str, len: usize) -> Self {
        Self {
            op,
            bytes: vec![0u8; len],
        }
    }

    #[must_use]
    pub const fn sizing(op: &'static str) -> Self {
        Self {
            op,
            bytes: Vec::new(),
        }
    }

    #[must_use]
    pub const fn materialises(&self) -> bool {
        !self.bytes.is_empty()
    }

    pub fn put_i32s(
        &mut self,
        offset: usize,
        values: &[i32],
        what: &'static str,
    ) -> Result<(), KernelError> {
        let end = offset + values.len() * 4;
        self.check(end, values.len() * 4, what)?;
        for (i, v) in values.iter().enumerate() {
            self.bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    pub fn put_i32(
        &mut self,
        offset: usize,
        value: i32,
        what: &'static str,
    ) -> Result<(), KernelError> {
        self.put_i32s(offset, &[value], what)
    }

    pub fn put_u32(
        &mut self,
        offset: usize,
        value: u32,
        what: &'static str,
    ) -> Result<(), KernelError> {
        self.check(offset + 4, 4, what)?;
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn put_bools(
        &mut self,
        offset: usize,
        values: impl Iterator<Item = bool>,
        what: &'static str,
    ) -> Result<(), KernelError> {
        for (i, v) in values.enumerate() {
            self.check(offset + i + 1, 1, what)?;
            self.bytes[offset + i] = u8::from(v);
        }
        Ok(())
    }

    #[must_use]
    pub fn into_upload(mut self, len: usize) -> Vec<u8> {
        self.bytes.truncate(len);
        self.bytes
    }

    fn check(&self, end: usize, size: usize, what: &'static str) -> Result<(), KernelError> {
        if end > self.bytes.len() {
            return Err(refuse(
                self.op,
                format!(
                    "`{what}` writes past the staged workspace: {size} bytes at the tail, \
                     {} staged",
                    self.bytes.len()
                ),
            ));
        }
        Ok(())
    }
}
