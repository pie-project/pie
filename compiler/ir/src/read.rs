//! The one little-endian cursor the PTIR decoders share.
//!
//! There used to be three of these — `container::Reader`, `sidecar::Reader`
//! and `compile::PlanReader` — identical apart from the error enum they
//! returned. That is the shape of duplication that looks harmless until the
//! copies stop agreeing: the container's cursor was for a while the only one
//! without a structural ceiling on table counts, which is what let a hostile
//! trace claim one op per byte and get all of them decoded, bound and compiled
//! under a global lock.
//!
//! Rather than make the cursor generic over the error type, it returns its own
//! [`ReadError`] and each decoder converts with `From`. `?` does the rest, so
//! call sites read the same as before and each format keeps its own public
//! error enum.

use core::convert::TryInto;

/// What a cursor can fail at. Deliberately small: everything above this layer
/// is a format question, not a bytes question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// Fewer bytes remain than the read asked for.
    UnexpectedEof,
    /// A count or length in the input is not backed by the bytes present, or
    /// exceeds the structural ceiling for its table. Carries the table name so
    /// the caller can say which one.
    CountTooLarge(&'static str),
}

/// A bounds-checked little-endian cursor over untrusted bytes.
///
/// `Copy` because plan decoding takes a snapshot to re-read a section.
#[derive(Clone, Copy, Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// How far in the cursor has advanced. Decoders compare this against the
    /// input length to reject trailing bytes.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

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

    pub fn u8(&mut self) -> Result<u8, ReadError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ReadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, ReadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, ReadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Reads a table count that the following bytes must be able to back.
    ///
    /// Two independent bounds, because either alone is insufficient:
    ///
    /// * `minimum_record_bytes` ties the count to the bytes actually present,
    ///   so a header claiming four billion records cannot make us allocate for
    ///   four billion.
    /// * `structural_maximum` is an absolute ceiling, because the first bound
    ///   is only as strong as the record size. The op table has a one-byte
    ///   minimum record, so without a ceiling a caller who controls the input
    ///   gets one op per byte — and the ops are then decoded, bound and
    ///   compiled while a global lock is held. That is a denial of service
    ///   with a text file.
    ///
    /// A `minimum_record_bytes` of zero is rejected rather than accepted as
    /// "unbounded": a zero-size record means the first bound is vacuous, and
    /// silently falling back to the ceiling alone would hide the mistake.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_rejects_overrun_and_overflow() {
        let mut r = Reader::new(&[1, 2, 3]);
        assert_eq!(r.take(4), Err(ReadError::UnexpectedEof));
        assert_eq!(r.take(2), Ok(&[1u8, 2][..]));
        assert_eq!(r.offset(), 2);
        assert_eq!(r.remaining(), 1);
        // `pos + count` must not wrap into a valid-looking range.
        assert_eq!(r.take(usize::MAX), Err(ReadError::UnexpectedEof));
    }

    #[test]
    fn bounded_count_needs_both_bounds() {
        let r = Reader::new(&[0; 64]);
        // Backed by bytes and under the ceiling.
        assert_eq!(r.bounded_count(8, 4, 16, "t"), Ok(8));
        // Backed by bytes, over the ceiling.
        assert_eq!(
            r.bounded_count(17, 1, 16, "t"),
            Err(ReadError::CountTooLarge("t"))
        );
        // Under the ceiling, not backed by bytes.
        assert_eq!(
            r.bounded_count(17, 4, 1024, "t"),
            Err(ReadError::CountTooLarge("t"))
        );
        // A zero-size record makes the byte bound vacuous, so it is refused.
        assert_eq!(
            r.bounded_count(1, 0, 16, "t"),
            Err(ReadError::CountTooLarge("t"))
        );
        // The multiply must not wrap.
        assert_eq!(
            r.bounded_count(u32::MAX, usize::MAX, usize::MAX, "t"),
            Err(ReadError::CountTooLarge("t"))
        );
    }

    #[test]
    fn length_is_bounded_by_the_bytes_present() {
        let mut r = Reader::new(&[0; 8]);
        assert_eq!(r.length(8, "t"), Ok(8));
        assert_eq!(r.length(9, "t"), Err(ReadError::CountTooLarge("t")));
        r.take(4).unwrap();
        assert_eq!(r.length(5, "t"), Err(ReadError::CountTooLarge("t")));
    }
}
