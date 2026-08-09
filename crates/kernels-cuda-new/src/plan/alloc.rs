//! `AlignedAllocator`, which is a bump pointer that returns offsets.
//!
//! # What it actually is
//!
//! Upstream's allocator has two modes and one of them is a decoy. In
//! materialising mode it holds a `void*` and calls `std::align`; in counting
//! mode it holds an integer and does the same padding by hand. But every call
//! site in `scheduler.cuh` uses `aligned_alloc_offset`, which in materialising
//! mode immediately subtracts the base pointer back off — so **both modes
//! produce offsets, and they produce the same offsets whenever the base is
//! aligned to the largest alignment requested**. The largest alignment any call
//! site asks for is 16; a `cudaMalloc` returns 256-byte-aligned memory and
//! `cudaHostAlloc` a page. So one implementation covers both, and it is the
//! counting one.
//!
//! That is the port's one structural simplification, and it is stated here
//! rather than hidden: **if a caller ever hands a workspace whose base is not
//! 16-byte aligned, this allocator's offsets and upstream's will diverge** —
//! upstream's by the base's misalignment, ours by nothing. No caller in this
//! tree can: the buffers come from `cudaMalloc`/`cudaHostAlloc`, and
//! `AttentionWorkspaceView` carves them at 256-byte boundaries.
//!
//! # Why the refusal is worth having
//!
//! The overflow check is the only thing standing between a too-small workspace
//! and a plan whose arrays overlap. Upstream throws; a `Result` is the same
//! decision with the numbers attached. The counting mode's check is
//! deliberately the one ported, because it is the one that is exact in integer
//! arithmetic: `std::align`'s `size + diff > space` can overflow for absurd
//! sizes, and `padding > remaining || size > remaining - padding` cannot.

use super::Error;

/// A bump allocator over an arena of known size, handing out offsets.
///
/// Alignment is per allocation, measured from the arena's base — which is why
/// this is not simply a running sum: a 4-byte allocation at alignment 1
/// (`kv_chunk_size_ptr`, in both FA2 planners) leaves the cursor unaligned, and
/// the next 16-aligned allocation pads by 12. Those 12 bytes are in the
/// uploaded region and in every offset after it.
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
    ///
    /// `remaining_space = std::numeric_limits<size_t>::max()`. Used by the
    /// sizing passes and — this is the part worth knowing — by the *float*
    /// allocator of every non-splitting plan, which is why a plan that does not
    /// split KV reports zero float bytes rather than refusing to size them.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Carve `size` bytes at `alignment`, and answer where they start.
    ///
    /// `what` is upstream's allocation label, carried only so a refusal names
    /// the same string the C++ would have thrown.
    ///
    /// # Errors
    ///
    /// [`Error::WorkspaceOverflow`] when the padded allocation does not fit,
    /// which is `FLASHINFER_ERROR("Buffer overflow ...")` upstream.
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
    /// of the H2D copy upstream issues.
    #[must_use]
    pub const fn used(&self) -> usize {
        self.allocated
    }
}

/// The page-locked staging buffer, as bytes plus the writes a planner makes.
///
/// Upstream writes through `IdType*` and `bool*` obtained from
/// `GetPtrFromBaseOffset`, into pinned memory it then copies wholesale. We
/// write into a `Vec<u8>` and hand it back. Two consequences, both wanted:
/// the module needs no pinned allocation to be tested, and a write that would
/// leave the arena is a [`Error::WorkspaceOverflow`] instead of a stomp on
/// whatever the previous plan left there.
///
/// The buffer is **zero-filled**, including the alignment padding between
/// arrays. Upstream's padding holds the previous plan's bytes; no kernel reads
/// it, and zeroing makes the byte-for-byte comparison in `tests/plan.rs` mean
/// something.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staging {
    bytes: Vec<u8>,
}

impl Staging {
    /// A zeroed staging buffer of `len` bytes.
    ///
    /// `len` is the int workspace size. `usize::MAX` — [`super::Workspace::unbounded`]
    /// — is not a size anyone can allocate, so a sizing pass must use
    /// [`Staging::sizing`] instead.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { bytes: vec![0u8; len] }
    }

    /// The staging buffer of a sizing pass: empty, and never written.
    ///
    /// `DecodePlanImpl<false>` and `PrefillPlanImpl<false>` run the whole
    /// planner with the writes compiled out (`if constexpr (MATERIALIZE)`).
    /// Here the writes are skipped by asking [`Staging::materialises`].
    #[must_use]
    pub const fn sizing() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Whether this buffer records writes.
    ///
    /// The runtime spelling of upstream's `if constexpr (MATERIALIZE)`. A
    /// sizing pass and a materialising pass must walk the same branches in the
    /// same order or their offsets diverge, so this is a predicate rather than
    /// two functions.
    #[must_use]
    pub const fn materialises(&self) -> bool {
        !self.bytes.is_empty()
    }

    /// Write `values` as `int32_t` starting at `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::WorkspaceOverflow`] if the write would leave the arena.
    /// Upstream cannot notice this: `std::copy` into a raw pointer past the
    /// allocation is exactly how a too-small `padded_batch_size` corrupts the
    /// next plan's descriptor, and `attention_flashinfer_common.cuh` carries a
    /// 20-line comment about having hit it.
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
    ///
    /// # Errors
    ///
    /// [`Error::WorkspaceOverflow`] if the write would leave the arena.
    pub fn put_i32(&mut self, offset: usize, value: i32, what: &'static str) -> Result<(), Error> {
        self.put_i32s(offset, &[value], what)
    }

    /// Write one `uint32_t` at `offset`.
    ///
    /// Only `batch_prefill_total_num_rows` is written this way, and it is
    /// written through a `uint32_t*` while every neighbour is an `IdType*`.
    /// Same four bytes; the port keeps the type because the kernel reads it
    /// back as `uint32_t`.
    ///
    /// # Errors
    ///
    /// [`Error::WorkspaceOverflow`] if the write would leave the arena.
    pub fn put_u32(&mut self, offset: usize, value: u32, what: &'static str) -> Result<(), Error> {
        self.check(offset + 4, 4, what)?;
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Write `values` as C++ `bool`s — one byte each, `0` or `1`.
    ///
    /// # Errors
    ///
    /// [`Error::WorkspaceOverflow`] if the write would leave the arena.
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
    /// is what makes this an aligning allocator rather than a running sum.
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
