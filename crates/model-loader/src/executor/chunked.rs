//! One address space over several allocations.
//!
//! A plan's arena is a single flat range: every offset in it is a number
//! between zero and `plan.memory.persistent_bytes`. Some devices cannot
//! provide that as one allocation — Metal caps a buffer at 4 GiB, so a 20 GiB
//! model is at least five buffers — and the walker must not learn about it,
//! because a walker that knew how many buffers a device used would be
//! executing a different plan per device.
//!
//! [`Chunked`] is the join: it takes the allocations and the offsets they
//! start at, and turns each arena offset into a chunk and an offset inside
//! it, splitting any access that crosses a cut.
//!
//! # Why this is not `executor::metal`
//!
//! It lived in `driver-metal`, and `.wiki/fix/weight-loader.md` §8.6 asks for
//! it here behind a `metal` feature, mirroring [`super::cuda`]. The feature is
//! not needed, and its own justification is why: the code "needs no Metal API
//! — it is chunk arithmetic over borrowed allocations", so the gate would
//! exist only to name the concrete type the driver passes. Naming that type
//! with a trait removes the reason for the gate, and with it the possibility
//! that this compiles on one platform and not another.
//!
//! What is left is arithmetic that any driver with a segmented arena wants,
//! under a name that says what it does rather than who happened to write it.
//!
//! # Why it is in the loader at all
//!
//! Because it is an [`ArenaBacking`], and a backing is the loader's word for
//! how bytes reach an arena. A driver holding its own [`ArenaBacking`] impl is
//! a second place the seam is described; the CUDA one moved here for that
//! reason and this is the same argument.
//!
//! It is also not only `driver-metal`'s any more, whatever the history says:
//! `tests/arena_transforms.rs` builds a segmented arena out of [`Chunked`] and
//! runs the same transforms through it that the flat host arena runs, which is
//! how a chunk boundary falling inside a tensor gets checked at all. Moving
//! this file to the driver would take that test's subject with it.

use std::borrow::Cow;
use std::ptr::NonNull;

use crate::error::Error;
use crate::executor::arena::ArenaBacking;

/// A host-addressable span of memory the loader may read and write.
///
/// # Safety
///
/// An implementor promises that [`base`](Chunk::base) is valid for reads and
/// writes of [`len`](Chunk::len) bytes for as long as `&self` is held, and
/// that no other `Chunk` in the same slice overlaps it. Every method of
/// [`Chunked`] dereferences that pointer on the strength of this.
pub unsafe trait Chunk {
    /// The first host-visible byte.
    fn base(&self) -> NonNull<u8>;

    /// How many bytes the loader may touch, starting at [`base`](Chunk::base).
    fn len(&self) -> u64;

    /// Whether the span is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Which chunk holds `offset`.
///
/// `cuts` is ascending and starts at zero, so this is the last cut at or
/// below `offset`. Public because a driver placing tensors wants the same
/// answer — which chunk a span landed in — and computing it a second way is
/// how the two would come to disagree.
#[must_use]
pub fn chunk_of(cuts: &[u64], offset: u64) -> usize {
    cuts.partition_point(|&c| c <= offset).saturating_sub(1)
}

/// Several allocations, addressed as one arena.
///
/// Reads that stay inside one chunk are lent rather than copied — the
/// [`ArenaBacking::read`] `Cow` exists for exactly this — so the common case
/// of a tensor that did not straddle a cut costs nothing.
pub struct Chunked<'a, C: Chunk> {
    chunks: &'a [C],
    cuts: &'a [u64],
}

impl<'a, C: Chunk> Chunked<'a, C> {
    /// Join `chunks` into one address space, cut at `cuts`.
    ///
    /// `cuts` holds one more entry than `chunks`: the offset each chunk starts
    /// at, and then the total. Chunk `i` therefore spans
    /// `cuts[i]..cuts[i + 1]`, and must be at least that many bytes.
    ///
    /// # Errors
    ///
    /// The cuts do not describe these chunks: wrong count, not starting at
    /// zero, not ascending, or naming more bytes of a chunk than it has. Each
    /// of those would make an in-bounds arena offset resolve to a pointer past
    /// the end of an allocation, which is why it is checked once here rather
    /// than assumed at every access.
    pub fn new(chunks: &'a [C], cuts: &'a [u64]) -> Result<Self, Error> {
        if cuts.len() != chunks.len() + 1 {
            return Err(Error::Internal(format!(
                "{} chunks need {} cuts, not {}",
                chunks.len(),
                chunks.len() + 1,
                cuts.len()
            )));
        }
        if cuts.first() != Some(&0) && !chunks.is_empty() {
            return Err(Error::Internal("the first cut is not zero".to_string()));
        }
        for (i, chunk) in chunks.iter().enumerate() {
            let want = cuts[i + 1].checked_sub(cuts[i]).ok_or_else(|| {
                Error::Internal(format!("cut {} runs backwards from cut {i}", i + 1))
            })?;
            if want > chunk.len() {
                return Err(Error::Internal(format!(
                    "chunk {i} is {} bytes and the cuts give it {want}",
                    chunk.len()
                )));
            }
        }
        Ok(Self { chunks, cuts })
    }

    /// Call `f(chunk, offset_in_chunk, len)` for each piece of
    /// `offset..offset + len`.
    ///
    /// The loop advances to the end of whichever chunk [`chunk_of`] names, so
    /// it terminates only because `chunk_of` names a chunk that ENDS after
    /// `at` — an invariant that lives in a different function from the loop
    /// that depends on it. When it does not hold the loop does not spin
    /// slowly, it spins forever calling `f` with zero bytes, which is the
    /// worst way for a bug to present: no panic, no wrong answer, no output.
    /// A mutation of `chunk_of`'s comparison hung a test rather than failing
    /// it, which is how this assert got here.
    fn pieces(&self, offset: u64, len: u64, mut f: impl FnMut(usize, u64, u64)) {
        let (mut at, end) = (offset, offset + len);
        while at < end {
            let i = chunk_of(self.cuts, at);
            let stop = end.min(self.cuts[i + 1]);
            assert!(
                stop > at,
                "chunk {i} spans {}..{} and cannot carry the access at {at}",
                self.cuts[i],
                self.cuts[i + 1]
            );
            f(i, at - self.cuts[i], stop - at);
            at = stop;
        }
    }

    /// The range is inside the arena.
    fn bounds(&self, offset: u64, len: u64) -> Result<(), Error> {
        let total = *self.cuts.last().unwrap_or(&0);
        if offset.checked_add(len).is_none_or(|end| end > total) {
            return Err(Error::Overflow(format!(
                "{len} bytes at {offset} leaves a {total}-byte arena"
            )));
        }
        Ok(())
    }

    /// The bytes of one piece, as the pointer arithmetic every method shares.
    ///
    /// # Safety
    ///
    /// `i` is a chunk index and `at..at + n` is inside that chunk, which is
    /// what [`Chunked::pieces`] and [`Chunked::bounds`] together guarantee.
    unsafe fn piece(&self, i: usize, at: u64, n: u64) -> *mut u8 {
        debug_assert!(
            at + n <= self.chunks[i].len(),
            "piece {at}..{} escapes chunk {i}",
            at + n
        );
        // SAFETY: the caller's obligation, checked above in debug builds, and
        // `new` checked that each chunk is at least as long as its cuts claim.
        unsafe { self.chunks[i].base().as_ptr().add(at as usize) }
    }
}

impl<C: Chunk> ArenaBacking for Chunked<'_, C> {
    fn len(&self) -> usize {
        usize::try_from(*self.cuts.last().unwrap_or(&0)).unwrap_or(usize::MAX)
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let (offset, len) = (offset as u64, len as u64);
        self.bounds(offset, len)?;
        let i = chunk_of(self.cuts, offset);
        if offset + len <= self.cuts[i + 1] {
            // SAFETY: inside one chunk, and the executor is the only writer.
            return Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(self.piece(i, offset - self.cuts[i], len), len as usize)
            }));
        }
        let mut out = Vec::with_capacity(len as usize);
        self.pieces(offset, len, |i, at, n| {
            // SAFETY: as above, per piece.
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(self.piece(i, at, n), n as usize)
            });
        });
        Ok(Cow::Owned(out))
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let offset = offset as u64;
        self.bounds(offset, bytes.len() as u64)?;
        let mut taken = 0usize;
        self.pieces(offset, bytes.len() as u64, |i, at, n| {
            let n = n as usize;
            // SAFETY: `pieces` keeps every span inside its chunk, `bytes` is a
            // live slice that cannot overlap device memory, and no GPU work
            // references these buffers until staging returns.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes[taken..taken + n].as_ptr(),
                    self.piece(i, at, n as u64),
                    n,
                );
            }
            taken += n;
        });
        Ok(())
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let (offset, len) = (offset as u64, len as u64);
        self.bounds(offset, len)?;
        self.pieces(offset, len, |i, at, n| {
            // SAFETY: as `write`.
            unsafe { std::ptr::write_bytes(self.piece(i, at, n), byte, n as usize) };
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk backed by a leaked host allocation, so the tests can read what
    /// they wrote through the same pointers a device would hand over.
    struct HostChunk(Box<[u8]>);

    // SAFETY: each `HostChunk` owns its box, so the spans cannot overlap, and
    // the box outlives every borrow of the `HostChunk`.
    unsafe impl Chunk for HostChunk {
        fn base(&self) -> NonNull<u8> {
            NonNull::new(self.0.as_ptr().cast_mut()).expect("a box is never null")
        }
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
    }

    fn chunks(sizes: &[usize]) -> Vec<HostChunk> {
        sizes
            .iter()
            .map(|&n| HostChunk(vec![0u8; n].into_boxed_slice()))
            .collect()
    }

    #[test]
    fn a_write_that_crosses_a_cut_lands_in_both_chunks() {
        let mut c = chunks(&[4, 4]);
        let cuts = [0, 4, 8];
        {
            let mut arena = Chunked::new(&c, &cuts).unwrap();
            arena.write(2, &[1, 2, 3, 4]).unwrap();
        }
        assert_eq!(&*c[0].0, &[0, 0, 1, 2], "the first chunk takes the head");
        assert_eq!(&*c[1].0, &[3, 4, 0, 0], "and the second takes the tail");
        c.clear();
    }

    #[test]
    fn a_read_inside_one_chunk_is_lent_and_one_that_crosses_is_copied() {
        let c = chunks(&[4, 4]);
        let cuts = [0, 4, 8];
        let mut arena = Chunked::new(&c, &cuts).unwrap();
        arena.write(0, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert!(matches!(arena.read(1, 2).unwrap(), Cow::Borrowed(b) if b == [2, 3]));
        match arena.read(3, 2).unwrap() {
            Cow::Owned(b) => assert_eq!(b, [4, 5]),
            Cow::Borrowed(_) => panic!("a span across a cut is not contiguous"),
        }
    }

    #[test]
    fn a_fill_crosses_cuts_too() {
        let c = chunks(&[2, 2, 2]);
        let cuts = [0, 2, 4, 6];
        let mut arena = Chunked::new(&c, &cuts).unwrap();
        arena.fill(1, 4, 0xAB).unwrap();
        assert_eq!(&*c[0].0, &[0, 0xAB]);
        assert_eq!(&*c[1].0, &[0xAB, 0xAB]);
        assert_eq!(&*c[2].0, &[0xAB, 0]);
    }

    #[test]
    fn past_the_end_is_refused_rather_than_wrapped() {
        let c = chunks(&[4]);
        let cuts = [0, 4];
        let mut arena = Chunked::new(&c, &cuts).unwrap();
        assert!(arena.write(2, &[0; 4]).is_err());
        assert!(arena.read(0, 5).is_err());
        assert!(arena.fill(4, 1, 0).is_err());
        // The overflow that a `checked_add` catches and a `+` would not.
        assert!(arena.read(usize::MAX, 8).is_err());
    }

    /// Cuts that do not describe the chunks are refused at construction.
    ///
    /// This is the check the driver's version did not have: it took the two
    /// slices on trust, so a cut naming more bytes than a buffer held would
    /// have produced a pointer past the end of an allocation at the first
    /// access rather than an error at the first opportunity.
    #[test]
    fn cuts_that_do_not_describe_the_chunks_are_refused() {
        let c = chunks(&[4, 4]);
        assert!(Chunked::new(&c, &[0, 4]).is_err(), "too few cuts");
        assert!(Chunked::new(&c, &[1, 5, 9]).is_err(), "does not start at 0");
        assert!(
            Chunked::new(&c, &[0, 8, 9]).is_err(),
            "chunk 0 is not 8 long"
        );
        assert!(Chunked::new(&c, &[0, 4, 8]).is_ok());
    }

    #[test]
    fn chunk_of_names_the_chunk_a_cut_opens() {
        let cuts = [0, 4, 9];
        assert_eq!(chunk_of(&cuts, 0), 0);
        assert_eq!(chunk_of(&cuts, 3), 0);
        assert_eq!(chunk_of(&cuts, 4), 1, "a cut belongs to the chunk it opens");
        assert_eq!(chunk_of(&cuts, 8), 1);
    }
}
