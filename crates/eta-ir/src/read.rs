//! The one little-endian cursor the ETA decoders share, so the structural
//! ceiling on table counts can't drift between per-format copies. Returns its
//! own [`ReadError`]; each decoder converts with `From`.

use core::convert::TryInto;
use core::fmt;

/// What a cursor can fail at. Deliberately small: everything above this layer
/// is a format question, not a bytes question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadError {
    /// Fewer bytes remain than the read asked for.
    UnexpectedEof,
    /// A count or length in the input is not backed by the bytes present, or
    /// exceeds the structural ceiling for its table. Carries the table name so
    /// the caller can say which one.
    CountTooLarge(&'static str),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::UnexpectedEof => f.write_str("unexpected end of input"),
            ReadError::CountTooLarge(table) => {
                write!(f, "{table} count exceeds what the input can back")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReadError {}

/// A bounds-checked little-endian cursor over untrusted bytes.
///
/// `Copy` because plan decoding takes a snapshot to re-read a section.
#[derive(Clone, Copy, Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    /// A cursor positioned at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// How far in the cursor has advanced. Decoders compare this against the
    /// input length to reject trailing bytes.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// How many bytes are left ahead of the cursor.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    /// The next `count` bytes, advancing past them.
    ///
    /// # Errors
    ///
    /// [`ReadError::UnexpectedEof`] if fewer than `count` bytes remain. The
    /// cursor does not move when that happens, so a failed read cannot leave
    /// a decoder reading from a half-consumed field.
    pub fn take(&mut self, count: usize) -> Result<&'a [u8], ReadError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(ReadError::UnexpectedEof)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ReadError::UnexpectedEof)?;
        self.offset = end;
        Ok(value)
    }

    /// The next u8 (1 little-endian byte).
    ///
    /// # Errors
    ///
    /// [`ReadError::UnexpectedEof`] if fewer than 1 byte remain.
    pub fn u8(&mut self) -> Result<u8, ReadError> {
        Ok(self.take(1)?[0])
    }

    /// The next u16 (2 little-endian bytes).
    ///
    /// # Errors
    ///
    /// [`ReadError::UnexpectedEof`] if fewer than 2 bytes remain.
    pub fn u16(&mut self) -> Result<u16, ReadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    /// The next u32 (4 little-endian bytes).
    ///
    /// # Errors
    ///
    /// [`ReadError::UnexpectedEof`] if fewer than 4 bytes remain.
    pub fn u32(&mut self) -> Result<u32, ReadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// The next u64 (8 little-endian bytes).
    ///
    /// # Errors
    ///
    /// [`ReadError::UnexpectedEof`] if fewer than 8 bytes remain.
    pub fn u64(&mut self) -> Result<u64, ReadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Reads a table count that the following bytes must be able to back.
    /// Two independent bounds: `minimum_record_bytes` ties the count to bytes
    /// actually present; `structural_maximum` is an absolute ceiling, needed
    /// because a one-byte record makes the first bound alone too weak.
    /// `minimum_record_bytes == 0` is rejected rather than treated as unbounded.
    pub fn bounded_count(
        &self,
        raw_count: u32,
        minimum_record_bytes: usize,
        structural_maximum: usize,
        table: &'static str,
    ) -> Result<usize, ReadError> {
        let count = raw_count as usize;
        let minimum_bytes = count
            .checked_mul(minimum_record_bytes)
            .ok_or(ReadError::CountTooLarge(table))?;
        if minimum_record_bytes == 0
            || count > structural_maximum
            || minimum_bytes > self.remaining()
        {
            return Err(ReadError::CountTooLarge(table));
        }
        Ok(count)
    }

    /// Reads a byte length that the remaining input must be able to back.
    /// Unlike [`Self::bounded_count`] there is no record size to multiply by,
    /// so the bytes present are the only bound there is.
    pub fn length(&self, raw_length: u32, table: &'static str) -> Result<usize, ReadError> {
        let length = raw_length as usize;
        if length > self.remaining() {
            return Err(ReadError::CountTooLarge(table));
        }
        Ok(length)
    }
}

