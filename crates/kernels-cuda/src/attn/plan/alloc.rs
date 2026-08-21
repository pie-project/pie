use super::Error;

#[derive(Clone, Copy, Debug)]
pub struct AlignedAllocator {
    allocated: usize,
    remaining: usize,
}

impl AlignedAllocator {

    #[must_use]
    pub const fn new(space: usize) -> Self {
        Self { allocated: 0, remaining: space }
    }

    #[must_use]
    pub const fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

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

    #[must_use]
    pub const fn used(&self) -> usize {
        self.allocated
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staging {
    bytes: Vec<u8>,
}

impl Staging {

    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { bytes: vec![0u8; len] }
    }

    #[must_use]
    pub const fn sizing() -> Self {
        Self { bytes: Vec::new() }
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
    ) -> Result<(), Error> {
        let end = offset + values.len() * 4;
        self.check(end, values.len() * 4, what)?;
        for (i, v) in values.iter().enumerate() {
            self.bytes[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    pub fn put_i32(&mut self, offset: usize, value: i32, what: &'static str) -> Result<(), Error> {
        self.put_i32s(offset, &[value], what)
    }

    pub fn put_u32(&mut self, offset: usize, value: u32, what: &'static str) -> Result<(), Error> {
        self.check(offset + 4, 4, what)?;
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

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
