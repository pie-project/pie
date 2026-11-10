//! The state pool: one buffer of `total_chunks` chunks of `chunk_bytes`,
//! physical chunk `p` at `p * chunk_bytes` from its base, every kind of held
//! state (kv pages, recurrent records, rs buffer pages) in chunks of it.
//!
//! On a Metal 4 device the buffer is placement sparse: its address range is
//! the whole pool from the load on, and memory comes and goes beneath it a
//! segment (`segment_chunks` chunks) at a time, always the first `n`
//! segments ([`Pool::commit_segments`], [`Pool::release_segments`]). A
//! segment edge inside a 256 KiB tile keeps that tile mapped while either
//! side is committed. A device without sparse buffers holds the pool flat,
//! whole from the load.
//!
//! The runtime names chunks logically; the pool places each on a physical
//! chunk through its [`ChunkTable`], so emptying the top segment
//! ([`Pool::evacuate`]) moves bytes and never renames a held slot.
#![allow(unsafe_code)]

use engine::chunk::{ChunkTable, Evacuation, Growth};

use crate::device::ctx::Frame;
#[cfg(target_vendor = "apple")]
use crate::device::elastic;
use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

/// Bytes a segment aims at: four 16 MiB placement heaps.
pub const SEGMENT_BYTES: u64 = 64 << 20;

#[derive(Debug)]
enum Backing {
    Flat(Buffer),
    #[cfg(target_vendor = "apple")]
    Sparse {
        /// The binding view over the sparse buffer.
        view: Buffer,
        store: elastic::Elastic,
        arena: elastic::Arena,
    },
}

#[derive(Debug)]
pub struct Pool {
    backing: Backing,
    chunk_bytes: u64,
    total_chunks: u32,
    segment_chunks: u32,
    /// Segments backed, the first ones.
    committed: u32,
    table: ChunkTable,
}

/// The chunks a segment of `chunk_bytes` chunks holds: as many as fill
/// [`SEGMENT_BYTES`], one at least.
#[must_use]
pub fn segment_chunks(chunk_bytes: u64) -> u32 {
    u32::try_from((SEGMENT_BYTES / chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
}

impl Pool {
    /// A pool of `total_chunks` chunks of `chunk_bytes`, segments of
    /// `segment_chunks`: placement sparse with nothing committed where
    /// `device` maps sparse buffers, else flat and whole.
    pub fn reserve(
        device: &Context,
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
    ) -> Result<Pool> {
        if chunk_bytes == 0 || total_chunks == 0 || segment_chunks == 0 {
            return Err(Fault::Ceiling {
                what: "chunks of the state pool",
                need: 1,
                have: 0,
            });
        }
        let bytes = chunk_bytes.saturating_mul(u64::from(total_chunks));
        #[cfg(target_vendor = "apple")]
        if let Some(sparse) = device.sparse()? {
            let arena = elastic::Arena::new(elastic::pages_up(bytes));
            let store = elastic::create(&sparse, &arena, bytes)?;
            return Ok(Pool {
                backing: Backing::Sparse {
                    view: store.view(),
                    store,
                    arena,
                },
                chunk_bytes,
                total_chunks,
                segment_chunks,
                committed: 0,
                table: ChunkTable::identity(total_chunks),
            });
        }
        let mut pool = Pool {
            backing: Backing::Flat(Buffer::zeroed(device, bytes)?),
            chunk_bytes,
            total_chunks,
            segment_chunks,
            committed: 0,
            table: ChunkTable::identity(total_chunks),
        };
        pool.committed = pool.segments();
        Ok(pool)
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    #[must_use]
    pub fn total_chunks(&self) -> u32 {
        self.total_chunks
    }

    #[must_use]
    pub fn segment_chunks(&self) -> u32 {
        self.segment_chunks
    }

    /// The segments the pool's range holds.
    #[must_use]
    pub fn segments(&self) -> u32 {
        self.total_chunks.div_ceil(self.segment_chunks)
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.committed
    }

    /// The chunks with memory behind them: the first ones, a whole number
    /// of segments (the last one short where the pool ends).
    #[must_use]
    pub fn committed_chunks(&self) -> u32 {
        self.committed
            .saturating_mul(self.segment_chunks)
            .min(self.total_chunks)
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        if self.is_sparse() {
            Growth::Bottom
        } else {
            Growth::Fixed
        }
    }

    #[must_use]
    pub fn is_sparse(&self) -> bool {
        match &self.backing {
            Backing::Flat(_) => false,
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { .. } => true,
        }
    }

    /// The buffer kernels and blits bind: the whole pool.
    #[must_use]
    pub fn view(&self) -> &Buffer {
        match &self.backing {
            Backing::Flat(buffer) => buffer,
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { view, .. } => view,
        }
    }

    /// The pool's whole range.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunk_bytes * u64::from(self.total_chunks)
    }

    /// The bytes with memory behind them now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        match &self.backing {
            Backing::Flat(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.committed(),
        }
    }

    /// The most that has had memory behind it at once.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        match &self.backing {
            Backing::Flat(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { arena, .. } => arena.budget().high_water,
        }
    }

    /// The physical chunk logical chunk `logical` sits on.
    pub fn physical(&self, logical: u32) -> Result<u32> {
        self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total_chunks),
        })
    }

    /// Where logical chunk `logical` starts, in bytes from the pool's base.
    pub fn offset(&self, logical: u32) -> Result<u64> {
        Ok(u64::from(self.physical(logical)?) * self.chunk_bytes)
    }

    /// The logical-to-physical table, by logical chunk.
    #[must_use]
    pub fn table(&self) -> &ChunkTable {
        &self.table
    }

    /// Refuses a physical chunk with no memory behind it: a kernel that
    /// reads an unmapped tile faults the whole queue.
    pub fn admit(&self, physical: u32) -> Result<()> {
        let have = self.committed_chunks();
        if physical >= have {
            return Err(Fault::Ceiling {
                what: "committed chunks of the state pool",
                need: u64::from(physical) + 1,
                have: u64::from(have),
            });
        }
        Ok(())
    }

    /// Backs the first `segments` segments; what is backed already stands,
    /// and fewer than are backed is no change. Newly backed memory is zeroed
    /// before this returns, since a placement heap holds what it last held.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        let max = self.segments();
        if segments > max {
            return Err(Fault::Ceiling {
                what: "segments of the state pool",
                need: u64::from(segments),
                have: u64::from(max),
            });
        }
        if segments <= self.committed {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        if let Backing::Sparse { store, .. } = &mut self.backing {
            let chunks = segments
                .saturating_mul(self.segment_chunks)
                .min(self.total_chunks);
            let bytes = u64::from(chunks) * self.chunk_bytes;
            elastic::grow_all(&mut [elastic::Target {
                buffer: store,
                bytes,
            }])?;
        }
        self.committed = segments;
        Ok(())
    }

    /// Gives back every segment at and past `segments`.
    ///
    /// # Safety
    ///
    /// No frame reads the chunks given back: the caller has drained every
    /// frame, and the runtime evacuated them first.
    pub unsafe fn release_segments(&mut self, segments: u32) -> Result<()> {
        if segments >= self.committed {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        if let Backing::Sparse { store, .. } = &mut self.backing {
            let bytes = u64::from(segments.saturating_mul(self.segment_chunks)) * self.chunk_bytes;
            // SAFETY: the caller's contract above.
            unsafe {
                elastic::shrink_all(&mut [elastic::Target {
                    buffer: store,
                    bytes,
                }])?;
            }
        }
        #[cfg(target_vendor = "apple")]
        if matches!(self.backing, Backing::Sparse { .. }) {
            self.committed = segments;
        }
        Ok(())
    }

    /// Copies each moved chunk onto its new physical chunk in `frame`, then
    /// takes the moves into the table: the caller commits `frame` before any
    /// later fire is encoded, so the copies land first.
    pub fn evacuate(&mut self, frame: &mut Frame, evacuation: &Evacuation) -> Result<()> {
        let mut table = self.table.clone();
        table.apply(evacuation).map_err(|why| Fault::Program {
            at: "store::pool::evacuate",
            why: why.to_string(),
        })?;
        for &(logical, physical) in &evacuation.moves {
            let from = self.physical(logical)?;
            self.admit(from)?;
            self.admit(physical)?;
            if from == physical {
                continue;
            }
            let view = self.view();
            frame.copy_span(
                view,
                u64::from(from) * self.chunk_bytes,
                view,
                u64::from(physical) * self.chunk_bytes,
                self.chunk_bytes,
            )?;
        }
        self.table = table;
        Ok(())
    }

    /// The host address of physical chunk `physical`'s bytes, valid until
    /// the segment holding it is given back; the caller writes through it
    /// only where no frame in flight reads them.
    pub fn chunk_host(&self, physical: u32) -> Result<std::ptr::NonNull<u8>> {
        self.admit(physical)?;
        let offset = u64::from(physical) * self.chunk_bytes;
        match &self.backing {
            Backing::Flat(buffer) => buffer.host_span(offset, self.chunk_bytes),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.host_span(offset, self.chunk_bytes),
        }
    }

    /// `into.len()` bytes from `offset`, read on the host.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        match &self.backing {
            Backing::Flat(buffer) => buffer.read(offset, into),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.read_into(offset, into),
        }
    }

    /// Writes `from` at `offset` on the host. The caller has drained every
    /// frame that touches these bytes.
    pub fn write(&mut self, offset: u64, from: &[u8]) -> Result<()> {
        match &mut self.backing {
            Backing::Flat(buffer) => buffer.write(offset, from),
            #[cfg(target_vendor = "apple")]
            // SAFETY: the caller's contract, restated on every host write.
            Backing::Sparse { store, .. } => unsafe { store.write_from(offset, from) },
        }
    }

    /// Zeroes `len` bytes at `offset` on the host; as [`Pool::write`].
    pub fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        match &mut self.backing {
            Backing::Flat(buffer) => buffer.zero_span(offset, len),
            #[cfg(target_vendor = "apple")]
            // SAFETY: as `write`.
            Backing::Sparse { store, .. } => unsafe { store.zero(offset, len) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<Context> {
        if !crate::device::present() {
            eprintln!("no Metal device on this machine; skipping");
            return None;
        }
        Some(Context::bind().expect("the system device"))
    }

    const C: u64 = 192 << 10;

    fn pattern(chunk: u32) -> Vec<u8> {
        (0..C)
            .map(|at| (at as u32).wrapping_mul(31).wrapping_add(chunk * 7) as u8)
            .collect()
    }

    #[test]
    fn a_segment_count_fills_four_heaps() {
        assert_eq!(segment_chunks(2 << 20), 32);
        assert_eq!(segment_chunks(3 << 20), 21);
        assert_eq!(segment_chunks(100 << 20), 1);
    }

    /// Ten chunks of 192 KiB in segments of three (segment edges inside
    /// tiles): the first segments back, the top one empties by evacuation
    /// with every held chunk's bytes kept, and goes back.
    #[test]
    fn a_sparse_pool_commits_evacuates_and_releases_segments() {
        let Some(device) = device() else { return };
        let mut pool = Pool::reserve(&device, C, 10, 3).expect("a pool");
        assert_eq!(pool.segments(), 4);
        if !pool.is_sparse() {
            eprintln!("not a Metal 4 device: the pool is flat and whole");
            assert_eq!(pool.committed_chunks(), 10);
            assert_eq!(pool.growth(), Growth::Fixed);
            return;
        }
        assert_eq!(pool.growth(), Growth::Bottom);
        assert_eq!(pool.committed_chunks(), 0);
        assert_eq!(
            pool.committed_bytes(),
            0,
            "a sparse pool maps nothing at load"
        );
        assert!(pool.admit(0).is_err());

        pool.commit_segments(2).expect("two segments");
        assert_eq!(pool.committed_chunks(), 6);
        assert!(pool.committed_bytes() >= 6 * C);
        pool.admit(5).expect("chunk 5 is backed");
        assert!(pool.admit(6).is_err(), "chunk 6 is not");
        assert!(pool.commit_segments(5).is_err(), "past the range");

        // Held logical chunks 1, 4 and 5 hold patterns; 5 sits in the top
        // segment and moves down onto physical 2.
        for chunk in [1u32, 4, 5] {
            let at = pool.offset(chunk).unwrap();
            pool.write(at, &pattern(chunk)).expect("a host write");
        }
        let mut frame = device.frame().expect("a frame");
        pool.evacuate(
            &mut frame,
            &Evacuation {
                moves: vec![(5, 2), (4, 0)],
            },
        )
        .expect("the moves stand");
        frame.commit().expect("the blits land");
        assert_eq!(pool.physical(5).unwrap(), 2);
        assert_eq!(pool.physical(4).unwrap(), 0);
        assert_eq!(
            pool.physical(2).unwrap(),
            5,
            "the free chunk took its place"
        );
        for chunk in [1u32, 4, 5] {
            let mut back = vec![0u8; C as usize];
            pool.read(pool.offset(chunk).unwrap(), &mut back).unwrap();
            assert!(
                back == pattern(chunk),
                "logical chunk {chunk} kept its bytes"
            );
        }

        // The top segment holds nothing held now: it goes back, and what is
        // below reads on.
        // SAFETY: every frame above was waited for.
        unsafe { pool.release_segments(1) }.expect("released");
        assert_eq!(pool.committed_chunks(), 3);
        assert!(pool.committed_bytes() < 6 * C);
        for chunk in [1u32, 4, 5] {
            let mut back = vec![0u8; C as usize];
            pool.read(pool.offset(chunk).unwrap(), &mut back).unwrap();
            assert!(
                back == pattern(chunk),
                "logical chunk {chunk} survives the release"
            );
        }
        // A move onto an unbacked chunk is refused, the table untouched.
        let mut frame = device.frame().expect("a frame");
        assert!(
            pool.evacuate(
                &mut frame,
                &Evacuation {
                    moves: vec![(1, 7)]
                }
            )
            .is_err()
        );
        frame.commit().expect("nothing encoded");
        assert_eq!(pool.physical(1).unwrap(), 1);

        // Committed again, the regrown segment reads zero.
        pool.commit_segments(4).expect("all of it");
        assert_eq!(pool.committed_chunks(), 10);
        let mut back = vec![0xffu8; C as usize];
        pool.read(9 * C, &mut back).unwrap();
        assert!(back.iter().all(|&b| b == 0), "fresh memory reads zero");
        assert!(pool.high_water_bytes() >= 10 * C);
    }
}
